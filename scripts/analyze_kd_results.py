# scripts/analyze_kd_results.py
"""
Collect and summarize KD experiments on InLegalNER.

It scans results/*_inlegalner_test.json, parses the metrics, and prints a
compact comparison table (params, F1, retention vs. teacher, etc.).

Usage:
  python scripts/analyze_kd_results.py

Optionally:
  python scripts/analyze_kd_results.py --results \
      results/teacher_legalbert_inlegalner_test.json \
      results/student_kd_distilbert_inlegalner_test.json \
      results/student_kd_v2_distilbert_inter_inlegalner_test.json \
      results/student_kd_v3_stage1_inlegalner_test.json \
      results/student_kd_v3_stage2_inlegalner_test.json \
      results/student_kd_multiteachers_inlegalner_test.json
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Any


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize KD experiments on InLegalNER."
    )
    parser.add_argument(
        "--results",
        type=str,
        nargs="*",
        default=None,
        help=(
            "List of result JSON files. "
            "If omitted, will use glob('results/*_inlegalner_test.json')."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.results is None or len(args.results) == 0:
        pattern = os.path.join("results", "*_inlegalner_test.json")
        result_files = sorted(glob.glob(pattern))
        if not result_files:
            raise FileNotFoundError(
                f"No result files found with pattern: {pattern}"
            )
    else:
        result_files = args.results

    print("[analyze_kd_results] Using result files:")
    for p in result_files:
        print(f"  - {p}")
    print()

    # 手动写一个参数量 lookup（方便以后统一更新）
    # 单位：M 参数
    param_lookup_m = {
        "teacher_legalbert": 108.0,
        "student_kd_distilbert": 66.4,
        "student_kd_v2_distilbert_inter": 66.4,
        "student_kd_v3_stage1": 66.4,
        "student_kd_v3_stage2": 66.4,
        "student_kd_multiteachers": 66.4,
    }

    rows: List[Dict[str, Any]] = []
    teacher_f1 = None

    for path in result_files:
        data = load_json(path)
        exp_name = data.get("experiment", os.path.basename(path))

        metrics = data.get("test_metrics", data)
        # 兼容不同 key 命名
        f1 = (
            metrics.get("test_f1")
            or metrics.get("test_f1_epoch")
            or metrics.get("test_f1_epoch_0")
        )
        loss = metrics.get("test_loss")

        params_m = param_lookup_m.get(exp_name, None)

        rows.append(
            {
                "experiment": exp_name,
                "file": os.path.basename(path),
                "params_M": params_m,
                "test_f1": f1,
                "test_loss": loss,
            }
        )

        # teacher F1 用来算 retention
        if exp_name == "teacher_legalbert" and f1 is not None:
            teacher_f1 = float(f1)

    # 计算 F1 retention
    if teacher_f1 is not None:
        for r in rows:
            if r["test_f1"] is not None:
                r["f1_retention"] = float(r["test_f1"]) / teacher_f1
            else:
                r["f1_retention"] = None
    else:
        for r in rows:
            r["f1_retention"] = None

    # 排序：先 teacher，再按 F1 从高到低
    def sort_key(r: Dict[str, Any]):
        if r["experiment"] == "teacher_legalbert":
            return (0, 0.0)
        f1 = r["test_f1"]
        return (1, -(f1 if f1 is not None else -1e9))

    rows = sorted(rows, key=sort_key)

    # pretty print
    print("=" * 80)
    print(f"{'Experiment':30s} {'Params(M)':>10s} {'Test F1':>10s} "
          f"{'F1%Teacher':>12s} {'Result JSON':>32s}")
    print("-" * 80)

    for r in rows:
        exp = r["experiment"]
        params_m = r["params_M"]
        f1 = r["test_f1"]
        retention = r["f1_retention"]
        fname = r["file"]

        params_str = f"{params_m:,.1f}" if params_m is not None else "?"
        f1_str = f"{f1:.4f}" if f1 is not None else "?"
        if retention is not None:
            retention_str = f"{retention * 100:5.1f}%"
        else:
            retention_str = "  ?  "

        print(
            f"{exp:30s} {params_str:>10s} {f1_str:>10s} "
            f"{retention_str:>12s} {fname:>32s}"
        )

    print("=" * 80)
    print("\nDone.\n")


if __name__ == "__main__":
    main()
