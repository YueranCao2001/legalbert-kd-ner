# scripts/train_student_kd_multiteacher.py
import os
import sys
import math
import argparse
import json
from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule_legalner import LegalNERDataModule
from src.models.student_kd_multiteacher_module import MultiTeacherKDNerModule


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-teacher KD for Legal NER (logits + intermediate + CE)."
    )

    parser.add_argument("--data_dir", type=str, default="data/raw/inlegalner")

    parser.add_argument(
        "--teacher_model_names",
        type=str,
        required=True,
        help="Comma-separated HF model ids, e.g. 'ggomarr/legal-bert-base-uncased-safetensors,nlpaueb/legal-bert-base-uncased'",
    )
    parser.add_argument(
        "--teacher_ckpts",
        type=str,
        default="",
        help="Comma-separated ckpt paths (same length as teacher_model_names). "
             "Use empty string for any teacher without ckpt.",
    )

    parser.add_argument(
        "--student_model_name",
        type=str,
        default="distilbert-base-uncased",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/checkpoints/student_kd_multi_teacher",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha_ce", type=float, default=1.0)
    parser.add_argument("--alpha_kd", type=float, default=1.0)
    parser.add_argument("--alpha_inter", type=float, default=1.0)

    parser.add_argument(
        "--teacher_weights",
        type=str,
        default="",
        help="Optional comma-separated weights for each teacher, e.g. '0.7,0.3'. "
             "If empty, use uniform weights.",
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

    teacher_model_names = _split_csv(args.teacher_model_names)
    teacher_ckpts_raw = _split_csv(args.teacher_ckpts) if args.teacher_ckpts else []
    # pad ckpts list with None if shorter
    teacher_ckpts: List[Optional[str]] = []
    for i, name in enumerate(teacher_model_names):
        if i < len(teacher_ckpts_raw) and teacher_ckpts_raw[i] != "":
            teacher_ckpts.append(teacher_ckpts_raw[i])
        else:
            teacher_ckpts.append(None)

    teacher_weights = None
    if args.teacher_weights:
        teacher_weights = [float(x) for x in _split_csv(args.teacher_weights)]
        assert len(teacher_weights) == len(teacher_model_names), \
            "teacher_weights length must match teacher_model_names."

    # ----- DataModule -----
    dm = LegalNERDataModule(
        data_dir=args.data_dir,
        pretrained_model_name=teacher_model_names[0],
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup("fit")

    label2id = dm.label2id
    id2label = dm.id2label
    num_labels = len(label2id)

    # ----- total steps -----
    train_loader = dm.train_dataloader()
    steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs

    # ----- Model -----
    model = MultiTeacherKDNerModule(
        student_model_name=args.student_model_name,
        teacher_model_names=teacher_model_names,
        teacher_ckpt_paths=teacher_ckpts,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
        temperature=args.temperature,
        alpha_ce=args.alpha_ce,
        alpha_kd=args.alpha_kd,
        alpha_inter=args.alpha_inter,
        teacher_weights=teacher_weights,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="student-kd-multi-{epoch:02d}-{val_f1:.4f}",
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

    # ----- save JSON -----
    os.makedirs("results", exist_ok=True)
    exp_name = os.path.basename(args.output_dir.rstrip("/\\")) or "student_kd_multi_teacher"
    out_path = os.path.join("results", f"{exp_name}_inlegalner_test.json")

    metrics = test_results[0] if isinstance(test_results, list) and len(test_results) > 0 else {}
    payload = {
        "experiment": exp_name,
        "data_dir": args.data_dir,
        "teacher_model_names": teacher_model_names,
        "teacher_ckpts": teacher_ckpts,
        "student_model_name": args.student_model_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "temperature": args.temperature,
        "alpha_ce": args.alpha_ce,
        "alpha_kd": args.alpha_kd,
        "alpha_inter": args.alpha_inter,
        "teacher_weights": teacher_weights,
        "test_metrics": metrics,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Multi-Teacher KD] Saved final test metrics to: {out_path}\n")


if __name__ == "__main__":
    main()
