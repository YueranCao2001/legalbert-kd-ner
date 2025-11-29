import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification
from seqeval.metrics import f1_score

from src.models.legalbert_ner_module import LegalBertForNER


class StudentKDNerV3Module(pl.LightningModule):
    """
    Stage-2 student for legal NER.

    - student: 从 HF 或 Stage1 v3 ckpt 初始化
    - teacher: 冻结的 LegalBERT NER（从 teacher ckpt 恢复）
    - loss: CE (gold labels) + KD (logits) + intermediate KD (hidden states)
    """

    def __init__(
        self,
        student_model_name: str,
        teacher_model_name: str,
        teacher_ckpt_path: str | None,
        num_labels: int,
        id2label,
        label2id,
        total_steps: int,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        temperature: float = 2.0,
        alpha_ce: float = 1.0,
        alpha_kd: float = 1.0,
        alpha_inter: float = 1.0,
        alpha_soft: float = 1.0,         # 目前占位，不单独使用
        init_student_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.alpha_inter = alpha_inter
        self.alpha_soft = alpha_soft  # 先存下来，方便之后做 alpha sweep

        # ----------------- student -----------------
        self.student = AutoModelForTokenClassification.from_pretrained(
            student_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # 如果提供了 Stage1 v3 ckpt，就用其中的 student 权重初始化
        if init_student_ckpt is not None:
            print(f"[KD-Stage2] Init student from Stage1 v3 ckpt: {init_student_ckpt}")
            ckpt = torch.load(init_student_ckpt, map_location="cpu")
            state_dict = ckpt["state_dict"]
            student_sd = {
                k.replace("student.", ""): v
                for k, v in state_dict.items()
                if k.startswith("student.")
            }
            missing, unexpected = self.student.load_state_dict(student_sd, strict=False)
            print(f"[KD-Stage2] Loaded student weights from Stage1. "
                  f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            print("[KD-Stage2] WARNING: init_student_ckpt is None, "
                  "student is initialized from HF weights only.")

        # ----------------- teacher（冻结） -----------------
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
            print(f"[KD-Stage2] Loaded teacher from ckpt: {teacher_ckpt_path}")
        else:
            self.teacher = AutoModelForTokenClassification.from_pretrained(
                teacher_model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            print("[KD-Stage2] WARNING: teacher_ckpt_path is None, "
                  "using raw HF teacher instead of finetuned LegalBERT.")

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")

        # teacher 12 层 → student 6 层
        self.layer_map = [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]

    # ----------------- optim & scheduler -----------------
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
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

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

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ----------------- KD + CE for one batch -----------------
    def _compute_losses(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # student
        s_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = s_outputs.logits
        student_hs = s_outputs.hidden_states

        # teacher
        with torch.no_grad():
            t_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        teacher_logits = t_outputs.logits
        teacher_hs = t_outputs.hidden_states

        # 1) supervised CE
        loss_ce = self.ce_loss_fct(
            student_logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        # 2) logit KD
        T = self.temperature
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        loss_kd = self.kl_loss_fct(student_log_probs, teacher_probs) * (T * T)

        # 3) intermediate KD
        inter_losses = []
        for t_idx, s_idx in self.layer_map:
            t_h = teacher_hs[t_idx]
            s_h = student_hs[s_idx]
            inter_losses.append(F.mse_loss(s_h, t_h))
        loss_inter = torch.stack(inter_losses).mean()

        loss = (
            self.alpha_ce * loss_ce
            + self.alpha_kd * loss_kd
            + self.alpha_inter * loss_inter
        )

        return loss, loss_ce, loss_kd, loss_inter, student_logits, labels

    def _shared_step(self, batch, stage: str):
        loss, loss_ce, loss_kd, loss_inter, student_logits, labels = self._compute_losses(batch)

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

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}_f1", f1, prog_bar=True, on_step=False, on_epoch=True)

        if stage == "train":
            self.log("train_ce_loss", loss_ce, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train_kd_loss", loss_kd, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train_inter_loss", loss_inter, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    # Lightning hooks
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")
