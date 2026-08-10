"""Aggregate repeated LSTM seeds and compare them with existing baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qd.config import OUTPUT_DIR


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="summarize LSTM seed robustness")
    parser.add_argument(
        "--runs", nargs="+", type=Path,
        default=[
            OUTPUT_DIR / "lstm_ensemble",
            OUTPUT_DIR / "lstm_ensemble_seed43",
            OUTPUT_DIR / "lstm_ensemble_seed44",
        ],
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_ensemble")
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        metrics = _load(run / "test_metrics.json")
        model = __import__("torch").load(
            run / "model.pt", map_location="cpu", weights_only=False
        )
        seed = int(model["config"]["component_seeds"]["direction"])
        rows.append({
            "run": run.name,
            "seed": seed,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "balanced_tier_accuracy": metrics["selective_accuracy"]["balanced"]["test_accuracy"],
            "balanced_tier_coverage": metrics["selective_accuracy"]["balanced"]["test_coverage"],
            "strict_tier_accuracy": metrics["selective_accuracy"]["strict"]["test_accuracy"],
            "strict_tier_coverage": metrics["selective_accuracy"]["strict"]["test_coverage"],
            "move_bias": metrics["fusion"]["move_bias"],
            "joint_weight": metrics["fusion"]["joint_weight"],
        })
    runs = pd.DataFrame(rows).sort_values("seed")
    runs.to_csv(args.out_dir / "seed_robustness.csv", index=False)

    aggregate = {
        metric: {
            "mean": float(runs[metric].mean()),
            "sample_std": float(runs[metric].std(ddof=1)),
            "min": float(runs[metric].min()),
            "max": float(runs[metric].max()),
        }
        for metric in (
            *METRICS,
            "balanced_tier_accuracy", "balanced_tier_coverage",
            "strict_tier_accuracy", "strict_tier_coverage",
        )
    }

    baseline_paths = {
        "direct_three_class": OUTPUT_DIR / "lstm_next_minute" / "test_metrics.json",
        "shared_hierarchical": OUTPUT_DIR / "lstm_next_minute_hierarchical" / "test_metrics.json",
    }
    comparison_rows = []
    for name, path in baseline_paths.items():
        metrics = _load(path)
        comparison_rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "accuracy_std": 0.0,
            "macro_f1_std": 0.0,
        })
    comparison_rows.append({
        "model": "three_component_mean",
        "accuracy": aggregate["accuracy"]["mean"],
        "balanced_accuracy": aggregate["balanced_accuracy"]["mean"],
        "macro_f1": aggregate["macro_f1"]["mean"],
        "accuracy_std": aggregate["accuracy"]["sample_std"],
        "macro_f1_std": aggregate["macro_f1"]["sample_std"],
    })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(args.out_dir / "model_comparison.csv", index=False)

    report = {
        "runs": rows,
        "aggregate": aggregate,
        "formal_model_rule": "seed 42 is retained because it was fixed before test comparison",
        "comparison": comparison_rows,
    }
    (args.out_dir / "seed_robustness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Direct\n3-class", "Shared\n2-head", "3-component\nmean"]
    x = np.arange(len(labels))
    width = 0.24
    figure, axis = plt.subplots(figsize=(9, 5))
    for offset, metric, title in (
        (-width, "accuracy", "Accuracy"),
        (0.0, "balanced_accuracy", "Balanced accuracy"),
        (width, "macro_f1", "Macro F1"),
    ):
        values = comparison[metric].to_numpy(float)
        bars = axis.bar(x + offset, values, width, label=title)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    for _, row in runs.iterrows():
        axis.scatter(
            [x[-1] - width, x[-1], x[-1] + width],
            [row["accuracy"], row["balanced_accuracy"], row["macro_f1"]],
            color="black", s=18, alpha=0.65, zorder=3,
        )
    axis.axhline(0.43677966101694915, color="gray", linestyle="--", label="Majority baseline")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.40, 0.49)
    axis.set_ylabel("Score")
    axis.set_title("Task 5 model comparison (three-component dots are individual seeds)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(args.out_dir / "model_comparison.png", dpi=180)
    plt.close(figure)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
