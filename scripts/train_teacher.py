import os
import sys
import math
import argparse
import json   # <-- 新增

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# 确保可以 import src.*
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.datamodule_legalner import LegalNERDataModule
from src.models.legalbert_ner_module import LegalBertForNER


def parse_args():
    parser = argparse.ArgumentParser(description="Train LegalBERT teacher model for Legal NER")

    parser.add_argument("--data_dir", type=str, default="data/raw/inlegalner",
                        help="Directory containing train.conll / dev.conll / test.conll")
    
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="ggomarr/legal-bert-base-uncased-safetensors",
        help="HF model id for the teacher (safetensors version of nlpaueb/legal-bert-base-uncased)",
    )
    
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/teacher_legalbert",
                        help="Where to save checkpoints and logs")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--precision", type=str, default="16-mixed",
                        help="Precision setting for Lightning Trainer (e.g., 16-mixed, 32-true)")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------ 1) DataModule ------------------
    dm = LegalNERDataModule(
        data_dir=args.data_dir,
        pretrained_model_name=args.pretrained_model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.prepare_data()
    dm.setup("fit")

    label2id = dm.label2id
    id2label = dm.id2label
    num_labels = len(label2id)

    # ------------------ 2) Total Training Steps ------------------
    train_loader = dm.train_dataloader()
    steps_per_epoch = math.ceil(len(train_loader.dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.max_epochs

    # ------------------ 3) Model ------------------
    model = LegalBertForNER(
        pretrained_model_name=args.pretrained_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        total_steps=total_steps,
    )

    # ------------------ 4) Callbacks ------------------
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="teacher-legalbert-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ------------------ 5) Trainer ------------------
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=10,
    )

    # ------------------ 6) Fit ------------------
    trainer.fit(model, datamodule=dm)

    # ------------------ 7) Test: use best checkpoint ------------------
    test_results = trainer.test(ckpt_path="best", datamodule=dm)

    # ------------------ 8) 保存结果为 JSON ------------------
    os.makedirs("results", exist_ok=True)

    exp_name = os.path.basename(args.output_dir.rstrip("/\\")) or "teacher_legalbert"
    out_path = os.path.join("results", f"{exp_name}_inlegalner_test.json")

    # 最终 test 结果（Lightning 返回 List[Dict]）
    metrics = test_results[0] if isinstance(test_results, list) and len(test_results) > 0 else {}

    payload = {
        "experiment": exp_name,
        "data_dir": args.data_dir,
        "pretrained_model_name": args.pretrained_model_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "test_metrics": metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Teacher] Saved final test metrics to: {out_path}\n")


if __name__ == "__main__":
    main()
