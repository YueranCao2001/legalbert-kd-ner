# scripts/train_student_kd_stage1_v3.py
import os
import sys
import math
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AutoModelForTokenClassification, AutoTokenizer
from seqeval.metrics import f1_score

# 确保可以 import src.*
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule_legalner import LegalNERDataModule
from src.models.legalbert_ner_module import LegalBertForNER


class StudentKDNerStage1V3Module(pl.LightningModule):
    """
    Stage-1 KD student for legal NER (无监督 KD 预训练).

    - 只用 teacher 的 logits + hidden states 做 KD
    - 不用 gold label 做 CE loss（labels 仅用于评估 F1）
    - 在 input_ids 上做 [MASK] 式数据增强
    """

    def __init__(
        self,
        student_model_name: str,
        teacher_model_name: str,
        teacher_ckpt_path: str | None,
        num_labels: int,
        id2label,
        label2id,
        mask_token_id: int,
        augment_times: int = 3,
        mask_prob: float = 0.15,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int = 1000,
        temperature: float = 2.0,
        alpha_kd: float = 1.0,
        alpha_inter: float = 1.0,
        alpha_soft: float = 1.0,   # 目前只是为了参数兼容，Stage1 中不参与 loss
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
        self.alpha_kd = alpha_kd
        self.alpha_inter = alpha_inter
        self.alpha_soft = alpha_soft  # 占位，方便以后扩展

        self.mask_token_id = mask_token_id
        self.augment_times = augment_times
        self.mask_prob = mask_prob

        # student 模型
        self.student = AutoModelForTokenClassification.from_pretrained(
            student_model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # teacher 模型（从 ckpt 恢复，冻结）
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
            print(f"[KD-Stage1] Loaded teacher from ckpt: {teacher_ckpt_path}")
        else:
            self.teacher = AutoModelForTokenClassification.from_pretrained(
                teacher_model_name,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
            )
            print(
                "[KD-Stage1] WARNING: teacher_ckpt_path is None, "
                "using raw HF teacher instead of finetuned LegalBERT."
            )

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

        # teacher 12 层 → student 6 层 的 layer mapping
        # hidden_states: 0 = embeddings, 1..L = layers
        self.layer_map = [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5), (12, 6)]

    # ----------------- optimizer & scheduler -----------------
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

    # ----------------- 简单 [MASK] 式增强 -----------------
    def _mask_augment(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        对非 -100 的 label 位置，以 mask_prob 概率把 token 换成 [MASK]。
        保持长度不变，因此不会破坏 label 对齐。
        """
        mask = (labels != -100)  # 只在有效 token 上考虑 mask
        rand = torch.rand_like(input_ids, dtype=torch.float)
        do_mask = (rand < self.mask_prob) & mask

        aug_input_ids = input_ids.clone()
        aug_input_ids[do_mask] = self.mask_token_id
        return aug_input_ids

    # ----------------- KD 损失（针对一个 batch） -----------------
    def _kd_loss_for_batch(self, input_ids, attention_mask):
        # teacher 输出
        with torch.no_grad():
            t_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        teacher_logits = t_outputs.logits
        teacher_hs = t_outputs.hidden_states

        # student 输出
        s_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        student_logits = s_outputs.logits
        student_hs = s_outputs.hidden_states

        # 1) logit KD
        T = self.temperature
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
        loss_kd = self.kl_loss(student_log_probs, teacher_probs) * (T * T)

        # 2) intermediate KD
        inter_losses = []
        for t_idx, s_idx in self.layer_map:
            t_h = teacher_hs[t_idx]
            s_h = student_hs[s_idx]
            inter_losses.append(F.mse_loss(s_h, t_h))
        loss_inter = torch.stack(inter_losses).mean()

        return loss_kd, loss_inter, student_logits

    # ----------------- 共享 step：train/val/test -----------------
    def _shared_step(self, batch, stage: str):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # train 阶段：做多次增强；val/test 阶段：不增强
        if stage == "train" and self.augment_times > 1:
            total_kd = 0.0
            total_inter = 0.0
            last_logits = None

            for k in range(self.augment_times):
                if k == 0:
                    ids = input_ids
                else:
                    ids = self._mask_augment(input_ids, labels)
                loss_kd, loss_inter, logits = self._kd_loss_for_batch(ids, attention_mask)
                total_kd += loss_kd
                total_inter += loss_inter
                last_logits = logits

            loss_kd = total_kd / self.augment_times
            loss_inter = total_inter / self.augment_times
            loss = self.alpha_kd * loss_kd + self.alpha_inter * loss_inter
            student_logits = last_logits
        else:
            loss_kd, loss_inter, student_logits = self._kd_loss_for_batch(
                input_ids, attention_mask
            )
            loss = self.alpha_kd * loss_kd + self.alpha_inter * loss_inter

        # 评估用 F1（只用 student logits）
        preds = torch.argmax(student_logits, dim=-1)

        preds_list = []
        labels_list = []
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
            self.log("train_kd_loss", loss_kd, prog_bar=False, on_step=True, on_epoch=True)
            self.log("train_inter_loss", loss_inter, prog_bar=False, on_step=True, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 KD (logits + intermediate) with data augmentation (masking)"
    )

    parser.add_argument("--data_dir", type=str, default="data/raw/inlegalner")
    parser.add_argument(
        "--teacher_model_name",
        type=str,
        default="nlpaueb/legal-bert-base-uncased",
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help="Path to finetuned teacher checkpoint.",
    )
    parser.add_argument(
        "--student_model_name",
        type=str,
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints/student_kd_stage1_v3",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_kd", type=float, default=1.0)
    parser.add_argument("--alpha_inter", type=float, default=1.0)
    parser.add_argument(
        "--alpha_soft",
        type=float,
        default=1.0,
        help="占位参数，Stage1 当前不会用到（为兼容命令行预留）。",
    )

    parser.add_argument(
        "--augment_times",
        type=int,
        default=10,
        help="How many augmented views per batch (B=10 for stronger KD).",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.15,
        help="Probability of masking a token (for KD data augmentation).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning precision (e.g., 16-mixed, 32-true).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # DataModule（这里仍然用 teacher 的 tokenizer）
    dm = LegalNERDataModule(
        data_dir=args.data_dir,
        pretrained_model_name=args.teacher_model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup("fit")

    label2id = dm.label2id
    id2label = dm.id2label
    num_labels = len(label2id)

    # 总步数估计
    train_loader = dm.train_dataloader()
    steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs

    # mask_token_id
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_name)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer has no mask_token_id; choose a BERT-like model for teacher.")

    # 模型
    model = StudentKDNerStage1V3Module(
        student_model_name=args.student_model_name,
        teacher_model_name=args.teacher_model_name,
        teacher_ckpt_path=args.teacher_ckpt,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        mask_token_id=mask_token_id,
        augment_times=args.augment_times,
        mask_prob=args.mask_prob,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
        temperature=args.temperature,
        alpha_kd=args.alpha_kd,
        alpha_inter=args.alpha_inter,
        alpha_soft=args.alpha_soft,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="student-kd-stage1-v3-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    test_results = trainer.test(ckpt_path="best", datamodule=dm)

    os.makedirs("results", exist_ok=True)
    exp_name = os.path.basename(args.output_dir.rstrip("/\\")) or "student_kd_stage1_v3"
    out_path = os.path.join("results", f"{exp_name}_inlegalner_test.json")

    metrics = test_results[0] if isinstance(test_results, list) and len(test_results) > 0 else {}
    payload = {
        "experiment": exp_name,
        "stage": "stage1_kd_only",
        "data_dir": args.data_dir,
        "teacher_model_name": args.teacher_model_name,
        "teacher_ckpt": args.teacher_ckpt,
        "student_model_name": args.student_model_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "temperature": args.temperature,
        "alpha_kd": args.alpha_kd,
        "alpha_inter": args.alpha_inter,
        "alpha_soft": args.alpha_soft,
        "augment_times": args.augment_times,
        "mask_prob": args.mask_prob,
        "test_metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Student KD Stage1 v3] Saved final test metrics to: {out_path}\n")


if __name__ == "__main__":
    main()
