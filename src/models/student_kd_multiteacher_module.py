# src/models/student_kd_multiteacher_module.py
import math
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification
from seqeval.metrics import f1_score

from src.models.legalbert_ner_module import LegalBertForNER


class MultiTeacherKDNerModule(pl.LightningModule):
    """
    Multi-teacher KD for legal NER.

    - student: 轻量 Transformer（如 distilbert）
    - teachers: 一个或多个已微调的 LegalBERT / BERT NER 模型
    - loss:
        L = α_ce * L_ce  (gold labels, 可选)
          + α_kd * L_kd  (logits KD, 多教师加权平均)
          + α_inter * L_inter (中间层 MSE, 多教师加权平均)
    """

    def __init__(
        self,
        student_model_name: str,
        teacher_model_names: List[str],
        teacher_ckpt_paths: List[Optional[str]],
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int = 1000,
        temperature: float = 2.0,
        alpha_ce: float = 1.0,
        alpha_kd: float = 1.0,
        alpha_inter: float = 1.0,
        teacher_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps

        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.alpha_inter = alpha_inter

        # ---------- student ----------
        self.student = AutoModelForTokenClassification.from_pretrained(
            student_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # ---------- teachers ----------
        assert len(teacher_model_names) == len(
            teacher_ckpt_paths
        ), "teacher_model_names and teacher_ckpt_paths must have same length."

        self.teachers = nn.ModuleList()
        for name, ckpt in zip(teacher_model_names, teacher_ckpt_paths):
            if ckpt:
                # 用 Lightning ckpt 的方式加载（不会触发 HF 的 weights_only 限制）
                teacher_module = LegalBertForNER.load_from_checkpoint(
                    ckpt,
                    pretrained_model_name=name,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    warmup_ratio=warmup_ratio,
                    total_steps=total_steps,
                )
                teacher = teacher_module.model
                print(f"[MultiTeacherKD] Loaded teacher from ckpt: {ckpt}")
            else:
                # 没有 ckpt 时，仅尝试加载 safetensors 版本的 HF 模型
                try:
                    teacher = AutoModelForTokenClassification.from_pretrained(
                        name,
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id=label2id,
                        use_safetensors=True,
                    )
                    print(
                        f"[MultiTeacherKD] Loaded HF teacher with safetensors: {name}"
                    )
                except Exception as e:
                    msg = (
                        f"[MultiTeacherKD] Failed to load teacher '{name}' from "
                        "HuggingFace with safetensors.\n"
                        "This usually happens when:\n"
                        "  1) The model repo only provides .bin weights (no safetensors), and\n"
                        "  2) Your torch version is < 2.6, so transformers cannot use "
                        "torch.load(weights_only=...).\n\n"
                        "Recommended fixes:\n"
                        "  - Either upgrade torch to >= 2.6, OR\n"
                        "  - Fine-tune this teacher separately and pass its Lightning ckpt "
                        "via --teacher_ckpts, so we can load it with "
                        "LegalBertForNER.load_from_checkpoint.\n"
                    )
                    raise RuntimeError(msg) from e

            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
            self.teachers.append(teacher)

        # teacher 权重（归一化）
        if teacher_weights is None:
            teacher_weights = [1.0] * len(self.teachers)
        w_sum = sum(teacher_weights)
        self.teacher_weights = [w / w_sum for w in teacher_weights]

        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # same layer map as v2/v3
        # hidden_states: 0 = embeddings, 1..L = layers
        self.layer_map = [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]

    # ---------- optim & scheduler ----------
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [
                    p
                    for n, p in self.student.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.student.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(grouped_params, lr=self.learning_rate)

        def lr_lambda(step: int):
            if self.total_steps <= 0:
                return 1.0
            warmup_steps = int(self.warmup_ratio * self.total_steps)
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(self.total_steps - step)
                / float(max(1, self.total_steps - warmup_steps)),
            )

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ---------- core KD computation ----------
    def _multi_teacher_kd(self, input_ids, attention_mask):
        # 1) student
        s_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = s_outputs.logits          # (B, L, C)
        student_hs = s_outputs.hidden_states       # list[1+L_s]

        # 2) teachers → 加权平均 logits + hidden states
        logits_acc = None
        hs_acc = None  # 用第一个 teacher 的 hidden_states 初始化

        with torch.no_grad():
            for weight, teacher in zip(self.teacher_weights, self.teachers):
                t_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                t_logits = t_outputs.logits
                t_hs = t_outputs.hidden_states      # list[1+L_t]

                if logits_acc is None:
                    logits_acc = weight * t_logits
                    hs_acc = [weight * h for h in t_hs]
                else:
                    logits_acc = logits_acc + weight * t_logits
                    for i in range(len(hs_acc)):
                        hs_acc[i] = hs_acc[i] + weight * t_hs[i]

        teacher_logits = logits_acc
        teacher_hs = hs_acc

        # 3) KD losses -------------------------------------------------
        T = self.temperature
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        loss_kd = self.kl_loss(student_log_probs, teacher_probs) * (T * T)

        # 根据实际层数裁剪一遍 layer_map，保证不会越界
        max_t_idx = len(teacher_hs) - 1
        max_s_idx = len(student_hs) - 1
        effective_layer_map = [
            (t_idx, s_idx)
            for (t_idx, s_idx) in self.layer_map
            if t_idx <= max_t_idx and s_idx <= max_s_idx
        ]
        if not effective_layer_map:
            raise ValueError(
                f"[MultiTeacherKD] layer_map {self.layer_map} 与实际层数不兼容："
                f"teacher_hs={len(teacher_hs)}, student_hs={len(student_hs)}"
            )

        inter_losses = []
        for t_idx, s_idx in effective_layer_map:
            t_h = teacher_hs[t_idx]
            s_h = student_hs[s_idx]
            inter_losses.append(F.mse_loss(s_h, t_h))
        loss_inter = torch.stack(inter_losses).mean()

        return student_logits, loss_kd, loss_inter

    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        student_logits, loss_kd, loss_inter = self._multi_teacher_kd(
            input_ids, attention_mask
        )

        # CE loss（可关掉：设 alpha_ce=0）
        loss_ce = self.ce_loss_fct(
            student_logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        loss = (
            self.alpha_ce * loss_ce
            + self.alpha_kd * loss_kd
            + self.alpha_inter * loss_inter
        )

        # F1
        preds = torch.argmax(student_logits, dim=-1)
        preds_list, labels_list = [], []
        for p_seq, l_seq in zip(preds, labels):
            p_seq = p_seq.tolist()
            l_seq = l_seq.tolist()
            p_tags, l_tags = [], []
            for p, l in zip(p_seq, l_seq):
                if l == -100:
                    continue
                p_tags.append(self.id2label[p])
                l_tags.append(self.id2label[l])
            if l_tags:
                preds_list.append(p_tags)
                labels_list.append(l_tags)

        f1 = f1_score(labels_list, preds_list) if labels_list else 0.0

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
        )
        self.log(
            f"{stage}_f1",
            f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        if stage == "train":
            self.log(
                "train_ce_loss",
                loss_ce,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "train_kd_loss",
                loss_kd,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "train_inter_loss",
                loss_inter,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")
