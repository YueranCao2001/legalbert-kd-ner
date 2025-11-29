import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification

from seqeval.metrics import f1_score, classification_report

# 复用已经写好的 LegalBERT NER 模块，用它来从 ckpt 里恢复 teacher
from src.models.legalbert_ner_module import LegalBertForNER


class StudentKDNerModule(pl.LightningModule):
    """
    KD student for legal NER.

    - student: 可训练的轻量模型 (e.g., distilbert-base-uncased)
    - teacher: 冻结的 LegalBERT NER (从你训练好的 ckpt 恢复)
    - losses: supervised CE + KD (logit distillation)
    """

    def __init__(
        self,
        student_model_name: str,
        teacher_model_name: str,
        num_labels: int,
        id2label,
        label2id,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int = 1000,
        alpha_ce: float = 1.0,
        alpha_kd: float = 1.0,
        temperature: float = 2.0,
        teacher_ckpt_path: str | None = None,
    ) -> None:
        super().__init__()

        # 保存超参数，方便之后从 ckpt 恢复
        self.save_hyperparameters()

        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps

        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd
        self.temperature = temperature

        # ---------------- Student 模型 ----------------
        self.student = AutoModelForTokenClassification.from_pretrained(
            student_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # ---------------- Teacher 模型（冻结） ----------------
        if teacher_ckpt_path is not None:
            # 从已经微调好的 LegalBERT NER checkpoint 恢复 teacher
            # 注意：load_from_checkpoint 会调用 __init__，
            # 所以这里把所有必须的 init 参数都传进去
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
            # LegalBertForNER 里面通常有 self.model 是 HF 的 token classification 模型
            self.teacher = teacher_module.model
            print(f"[StudentKD] Loaded teacher from ckpt: {teacher_ckpt_path}")
        else:
            # 退回到直接用 HF 上的原始 LegalBERT（不推荐，只是备选）
            self.teacher = AutoModelForTokenClassification.from_pretrained(
                teacher_model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            print(
                "[StudentKD] WARNING: teacher_ckpt_path is None, "
                "using raw HF teacher weights instead of finetuned checkpoint."
            )

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")

    # ---------- optimizer & scheduler ----------
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
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

    # ---------- forward & step ----------
    def forward(self, **batch):
        return self.student(**batch)

    def _shared_step(self, batch, stage: str):
        # batch: input_ids, attention_mask, labels
        labels = batch["labels"]

        # --------- student forward ---------
        student_outputs = self.student(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        student_logits = student_outputs.logits  # (B, L, C)

        # --------- teacher forward (no_grad) ---------
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_logits = teacher_outputs.logits

        # --------- supervised CE loss ---------
        loss_ce = self.ce_loss_fct(
            student_logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        # --------- KD loss (logits) ---------
        T = self.temperature
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        loss_kd = self.kl_loss_fct(student_log_probs, teacher_probs) * (T * T)

        loss = self.alpha_ce * loss_ce + self.alpha_kd * loss_kd

        # --------- metrics: F1 ---------
        preds = torch.argmax(student_logits, dim=-1)  # (B, L)

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

        return loss

    # ---------- Lightning hooks ----------
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, stage="test")
