import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification

from seqeval.metrics import f1_score

from src.models.legalbert_ner_module import LegalBertForNER


class StudentKDNerV2Module(pl.LightningModule):
    """
    Extended KD student for legal NER.

    - student: smaller encoder (e.g., distilbert-base-uncased)
    - teacher: finetuned LegalBERT loaded from checkpoint (frozen)
    - losses:
        * supervised CE
        * logit KD (KL with temperature)
        * intermediate-layer KD (MSE over hidden states)
    """

    def __init__(
        self,
        student_model_name: str,
        teacher_model_name: str,
        teacher_ckpt_path: str | None,
        num_labels: int,
        id2label,
        label2id,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int = 1000,
        temperature: float = 2.0,
        alpha_ce: float = 1.0,
        alpha_kd: float = 1.0,
        alpha_inter: float = 1.0,
    ) -> None:
        super().__init__()

        # 保存超参数，方便从 ckpt 恢复
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

        # --------------- Student 模型（支持 hidden_states） ---------------
        self.student = AutoModelForTokenClassification.from_pretrained(
            student_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # --------------- Teacher 模型（从 ckpt 恢复，冻结） ---------------
        if teacher_ckpt_path is not None:
            teacher_module = LegalBertForNER.load_from_checkpoint(
                teacher_ckpt_path,
                pretrained_model_name=teacher_model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                total_steps=total_steps,
            )
            self.teacher = teacher_module.model
            print(f"[KD-V2] Loaded teacher from ckpt: {teacher_ckpt_path}")
        else:
            self.teacher = AutoModelForTokenClassification.from_pretrained(
                teacher_model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            print(
                "[KD-V2] WARNING: teacher_ckpt_path is None, "
                "using raw HF teacher instead of finetuned LegalBERT."
            )

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # teacher 12 层 → student 6 层 的 layer mapping
        # hidden_states: index 0 = embeddings, 1..L = transformer layers
        # teacher: 0..12, student: 0..6
        self.layer_map = [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]

    # ------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.student.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.student.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        def lr_lambda(current_step: int):
            if self.total_steps <= 0:
                return 1.0
            warmup_steps = int(self.warmup_ratio * self.total_steps)
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(self.total_steps - current_step)
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

    def forward(self, **batch):
        return self.student(**batch)

    # ------------------------------------------------
    # Shared step: train / val / test
    # ------------------------------------------------
    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # -------- Student forward（需要 hidden_states） --------
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = student_outputs.logits
        student_hs = student_outputs.hidden_states  # tuple

        # -------- Teacher forward（no grad, hidden_states） --------
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            teacher_logits = teacher_outputs.logits
            teacher_hs = teacher_outputs.hidden_states

        # -------- 1) supervised CE loss --------
        loss_ce = self.ce_loss(
            student_logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        # -------- 2) logit KD loss --------
        T = self.temperature
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        loss_kd = self.kl_loss(student_log_probs, teacher_probs) * (T * T)

        # -------- 3) intermediate-layer KD loss --------
        inter_losses = []
        for t_idx, s_idx in self.layer_map:
            t_h = teacher_hs[t_idx]  # (B, L, H)
            s_h = student_hs[s_idx]  # (B, L, H)
            inter_losses.append(F.mse_loss(s_h, t_h))
        loss_inter = torch.stack(inter_losses).mean()

        # 总 loss
        loss = (
            self.alpha_ce * loss_ce
            + self.alpha_kd * loss_kd
            + self.alpha_inter * loss_inter
        )

        # -------- Metrics: F1 --------
        preds = torch.argmax(student_logits, dim=-1)

        preds_list = []
        labels_list = []
        for p_seq, l_seq in zip(preds, labels):
            p_seq = p_seq.tolist()
            l_seq = l_seq.tolist()

            p_tags = []
            l_tags = []
            for p, l in zip(p_seq, l_seq):
                if l == -100:
                    continue
                p_tags.append(self.id2label[p])
                l_tags.append(self.id2label[l])
            if len(l_tags) > 0:
                preds_list.append(p_tags)
                labels_list.append(l_tags)

        if len(labels_list) > 0:
            f1 = f1_score(labels_list, preds_list)
        else:
            f1 = 0.0

        # logging
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)
        if stage == "train":
            # 方便后面 ablation 分析
            self.log("train_ce_loss", loss_ce, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train_kd_loss", loss_kd, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train_inter_loss", loss_inter, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    # Lightning hooks
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")
