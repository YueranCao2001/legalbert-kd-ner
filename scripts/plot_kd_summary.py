# scripts/plot_kd_summary.py
"""
Plot summary figures for LegalBERT KD experiments.

This script:
  1) Loads final test metrics from results/*.json
  2) Plots:
     - Bar chart of test F1 for all models
     - Param(FLOPs proxy) vs F1 trade-off scatter plot
     - F1-per-parameter efficiency bar chart
     - Relative-to-teacher F1 bar chart

Usage (from repo root):

  python scripts/plot_kd_summary.py \
    --results_dir results \
    --output_dir outputs/figures

It will create:
  - outputs/figures/kd_test_f1_bar.png
  - outputs/figures/kd_param_f1_tradeoff.png
  - outputs/figures/kd_f1_per_param_bar.png
  - outputs/figures/kd_relative_f1_bar.png
"""

import os
import json
import argparse

import matplotlib.pyplot as plt


def load_json(path: str):
    if not os.path.exists(path):
        print(f"[WARN] File not found, skip: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot summary figures for KD experiments on InLegalNER."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing *_inlegalner_test.json files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures",
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Define experiments here ----
    # num_params in millions (M)
    experiments = [
        {
            "name": "Teacher (LegalBERT)",
            "key": "teacher_legalbert",
            "json": "teacher_legalbert_inlegalner_test.json",
            "params_m": 108.0,
        },
        {
            "name": "KD-v1 (logits)",
            "key": "student_kd_v1",
            "json": "student_kd_distilbert_inlegalner_test.json",
            "params_m": 66.4,
        },
        {
            "name": "KD-v2 (logits+inter)",
            "key": "student_kd_v2",
            "json": "student_kd_v2_distilbert_inter_inlegalner_test.json",
            "params_m": 66.4,
        },
        {
            "name": "KD-v3 Stage1",
            "key": "student_kd_v3_stage1",
            "json": "student_kd_v3_stage1_inlegalner_test.json",
            "params_m": 66.4,
        },
        {
            "name": "KD-v3 Stage2",
            "key": "student_kd_v3_stage2",
            "json": "student_kd_v3_stage2_inlegalner_test.json",
            "params_m": 66.4,
        },
        {
            "name": "Multi-Teacher",
            "key": "student_kd_multiteacher",
            "json": "student_kd_multi2teachers_inlegalner_test.json",
            "params_m": 66.4,
        },
    ]

    # 简单配色：teacher 深蓝、KD 渐变蓝、Stage2 绿色、multi-teacher 紫色
    color_map = {
        "Teacher (LegalBERT)": "#1b4f72",
        "KD-v1 (logits)": "#2874a6",
        "KD-v2 (logits+inter)": "#3498db",
        "KD-v3 Stage1": "#5dade2",
        "KD-v3 Stage2": "#1abc9c",
        "Multi-Teacher": "#a569bd",
    }

    names = []
    f1s = []
    params_m = []

    teacher_f1 = None

    # -------- Load metrics --------
    for exp in experiments:
        json_path = os.path.join(args.results_dir, exp["json"])
        data = load_json(json_path)
        if data is None:
            continue

        metrics = data.get("test_metrics", {})
        # Try common keys
        f1 = (
            metrics.get("test_f1")
            or metrics.get("test_f1_epoch")
            or metrics.get("test_f1_macro")
        )
        if f1 is None:
            print(
                f"[WARN] No test_f1 found in {json_path}, "
                f"available keys: {list(metrics.keys())}"
            )
            continue

        f1 = float(f1)

        names.append(exp["name"])
        f1s.append(f1)
        params_m.append(exp["params_m"])

        if exp["key"] == "teacher_legalbert":
            teacher_f1 = f1

        rel = ""
        if teacher_f1 is not None:
            rel = f" ({f1 / teacher_f1 * 100:.1f}% of teacher)"
        print(
            f"[INFO] Loaded {exp['name']}: "
            f"test_f1={f1:.4f}, params={exp['params_m']}M{rel}"
        )

    if not names:
        print("[ERROR] No valid experiment metrics loaded. Check your results/*.json files.")
        return

    plt.style.use("seaborn-v0_8-whitegrid")

    # ===============================
    # Figure 1: Test F1 bar chart
    # ===============================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(names))
    bar_colors = [color_map.get(n, "#3498db") for n in names]

    bars = ax.bar(
        x,
        f1s,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.1,
    )

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Test F1 (macro)", fontsize=13)
    ax.set_title("InLegalNER — KD Experimental Results (Test F1)", fontsize=16)

    ymin = max(0.49, min(f1s) - 0.02)
    ymax = min(0.82, max(f1s) + 0.03)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig1_path = os.path.join(args.output_dir, "kd_test_f1_bar.png")
    fig.savefig(fig1_path, dpi=300)
    print(f"[SAVE] Bar chart saved to {fig1_path}")
    plt.close(fig)

    # ===============================
    # Figure 2: Param vs F1 trade-off
    # ===============================
    fig, ax = plt.subplots(figsize=(7.5, 7))

    for p, f, name in zip(params_m, f1s, names):
        size = 180 if "Teacher" in name else 120
        ax.scatter(
            p,
            f,
            s=size,
            color=color_map.get(name, "#3498db"),
            edgecolor="black",
            linewidth=1.3,
            zorder=3,
        )

        dx = 1.0 if "Teacher" not in name else 2.0
        dy = 0.003
        ax.text(
            p + dx,
            f + dy,
            name if "Teacher" in name else name,
            fontsize=11,
            fontweight="bold",
            color=color_map.get(name, "#2c3e50"),
        )

    ax.set_xlabel("Parameters (M)", fontsize=13)
    ax.set_ylabel("Test F1 (macro)", fontsize=13)
    ax.set_title("Accuracy–Size Trade-off (InLegalNER)", fontsize=16)

    x_min = min(params_m) - 3
    x_max = max(params_m) + 3
    ax.set_xlim(x_min, x_max)

    y_min = max(0.49, min(f1s) - 0.02)
    y_max = min(0.80, max(f1s) + 0.03)
    ax.set_ylim(y_min, y_max)

    ax.grid(linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig2_path = os.path.join(args.output_dir, "kd_param_f1_tradeoff.png")
    fig.savefig(fig2_path, dpi=300)
    print(f"[SAVE] Trade-off plot saved to {fig2_path}")
    plt.close(fig)

    # ===============================
    # Figure 3: F1-per-parameter efficiency
    # ===============================
    # F1 / Params(M)，越高说明“每单位参数”的收益越大
    efficiency = [f / p for f, p in zip(f1s, params_m)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(names))
    bars = ax.bar(
        x,
        efficiency,
        color=bar_colors,
        edgecolor="black",
        linewidth=1.1,
    )

    for bar, eff in zip(bars, efficiency):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            eff + max(efficiency) * 0.01,
            f"{eff:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("F1 / Params (per M)", fontsize=13)
    ax.set_title("Efficiency: Test F1 per Parameter (InLegalNER)", fontsize=16)

    fig.tight_layout()
    fig3_path = os.path.join(args.output_dir, "kd_f1_per_param_bar.png")
    fig.savefig(fig3_path, dpi=300)
    print(f"[SAVE] Efficiency bar chart saved to {fig3_path}")
    plt.close(fig)

    # ===============================
    # Figure 4: Relative F1 vs Teacher
    # ===============================
    if teacher_f1 is not None:
        rel_f1 = [f / teacher_f1 * 100.0 for f in f1s]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(names))
        bars = ax.bar(
            x,
            rel_f1,
            color=bar_colors,
            edgecolor="black",
            linewidth=1.1,
        )

        for bar, r in zip(bars, rel_f1):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                r + 0.5,
                f"{r:.1f}%",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.2)
        ax.text(
            -0.4,
            100.5,
            "Teacher = 100%",
            fontsize=11,
            color="#555555",
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel("Relative F1 to Teacher (%)", fontsize=13)
        ax.set_title("Relative Performance vs Teacher (InLegalNER)", fontsize=16)

        ax.set_ylim(min(rel_f1) - 2.0, 105.0)

        fig.tight_layout()
        fig4_path = os.path.join(args.output_dir, "kd_relative_f1_bar.png")
        fig.savefig(fig4_path, dpi=300)
        print(f"[SAVE] Relative F1 bar chart saved to {fig4_path}")
        plt.close(fig)
    else:
        print("[WARN] teacher_f1 is None, skip relative-to-teacher plot.")
    # ===============================
    # Figure 5: Progressive KD improvements (v1 -> v2 -> v3)
    # ===============================
    # 只使用 student 系列：KD-v1, KD-v2, KD-v3 Stage1, KD-v3 Stage2
    kd_order = ["KD-v1 (logits)",
                "KD-v2 (logits+inter)",
                "KD-v3 Stage1",
                "KD-v3 Stage2"]

    kd_names = []
    kd_f1s = []

    for target in kd_order:
        if target in names:
            idx = names.index(target)
            kd_names.append(names[idx])
            kd_f1s.append(f1s[idx])

    if kd_names:
        fig, ax = plt.subplots(figsize=(8, 5))

        x = range(len(kd_names))

        ax.plot(
            x,
            kd_f1s,
            marker="o",
            linestyle="-",
            linewidth=2.0,
            markersize=8,
            color="#1abc9c",
            label="KD student progression",
        )

        # 标出每个点的数值
        for xi, fi in zip(x, kd_f1s):
            ax.text(
                xi,
                fi + 0.003,
                f"{fi:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Teacher 作为参照线
        if teacher_f1 is not None:
            ax.axhline(
                teacher_f1,
                color="#555555",
                linestyle="--",
                linewidth=1.3,
                label="Teacher (LegalBERT)",
            )

        ax.set_xticks(list(x))
        ax.set_xticklabels(kd_names, rotation=15, ha="right", fontsize=11)
        ax.set_ylabel("Test F1 (macro)", fontsize=13)
        ax.set_title("Progressive KD Improvements (InLegalNER)", fontsize=16)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=11)

        y_min = max(0.49, min(kd_f1s + ([teacher_f1] if teacher_f1 is not None else [])) - 0.02)
        y_max = min(0.80, max(kd_f1s + ([teacher_f1] if teacher_f1 is not None else [])) + 0.03)
        ax.set_ylim(y_min, y_max)

        fig.tight_layout()
        fig5_path = os.path.join(args.output_dir, "kd_progression_line.png")
        fig.savefig(fig5_path, dpi=300)
        print(f"[SAVE] KD progression line plot saved to {fig5_path}")
        plt.close(fig)
    else:
        print("[WARN] No KD progression points found, skip Figure 5.")



if __name__ == "__main__":
    main()
