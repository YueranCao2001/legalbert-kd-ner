from typing import Dict, List, Optional, Any

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from seqeval.metrics import f1_score


class LegalBertForNER(pl.LightningModule):
    """
    Baseline LegalBERT model for token-level NER.

    - Uses AutoModelForTokenClassification
    - Computes token-level F1 on validation & test sets (ignoring label == -100)
    """

    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        id2label: Dict[int, str],
        label2id: Dict[str, int],
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["id2label", "label2id"])

        self.id2label = id2label
        self.label2id = label2id

        config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name,
            config=config,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.total_steps = total_steps  # 可以在外部 set

        # buffers for epoch-level metrics
        self.val_preds: List[List[int]] = []
        self.val_labels: List[List[int]] = []

        self.test_preds: List[List[int]] = []
        self.test_labels: List[List[int]] = []

    def forward(self, **inputs) -> Any:
        return self.model(**inputs)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # collect predictions & labels for F1
        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        for p_seq, l_seq in zip(preds, labels):
            p_seq_list: List[int] = []
            l_seq_list: List[int] = []
            for p, l in zip(p_seq.tolist(), l_seq.tolist()):
                if l == -100:
                    continue
                p_seq_list.append(p)
                l_seq_list.append(l)
            if l_seq_list:  # 非空序列
                self.val_preds.append(p_seq_list)
                self.val_labels.append(l_seq_list)

    def on_validation_epoch_end(self) -> None:
        if not self.val_labels:
            return

        # convert id -> tag string for seqeval
        true_tags: List[List[str]] = []
        pred_tags: List[List[str]] = []

        for y_true, y_pred in zip(self.val_labels, self.val_preds):
            true_tags.append([self.id2label[i] for i in y_true])
            pred_tags.append([self.id2label[i] for i in y_pred])

        val_f1 = f1_score(true_tags, pred_tags)
        self.log("val_f1", val_f1, prog_bar=True, logger=True)

        # clear buffers
        self.val_preds = []
        self.val_labels = []

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        logits = outputs.logits

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        for p_seq, l_seq in zip(preds, labels):
            p_seq_list: List[int] = []
            l_seq_list: List[int] = []
            for p, l in zip(p_seq.tolist(), l_seq.tolist()):
                if l == -100:
                    continue
                p_seq_list.append(p)
                l_seq_list.append(l)
            if l_seq_list:
                self.test_preds.append(p_seq_list)
                self.test_labels.append(l_seq_list)

    def on_test_epoch_end(self) -> None:
        if not self.test_labels:
            return

        true_tags: List[List[str]] = []
        pred_tags: List[List[str]] = []

        for y_true, y_pred in zip(self.test_labels, self.test_preds):
            true_tags.append([self.id2label[i] for i in y_true])
            pred_tags.append([self.id2label[i] for i in y_pred])

        test_f1 = f1_score(true_tags, pred_tags)
        self.log("test_f1", test_f1, prog_bar=True, logger=True)

        self.test_preds = []
        self.test_labels = []

    def configure_optimizers(self):
        # standard AdamW + linear warmup / decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
        )

        if self.total_steps is None:
            # Lightning will set this later via `trainer.estimated_stepping_batches`,
            # but in simple use we can fallback to no scheduler.
            return optimizer

        warmup_steps = int(self.warmup_ratio * self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]
