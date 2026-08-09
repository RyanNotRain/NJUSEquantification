"""Run leakage-aware research diagnostics on the saved LSTM artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from qd.config import OUTPUT_DIR
from qd.lstm_research import (
    FULL_CLASS_NAMES,
    aggregate_walk_forward_predictions,
    baseline_report,
    compare_conditional_binary_model,
    date_stability_report,
    evaluate_saved_components,
    make_walk_forward_splits,
    probability_metrics,
    temperature_calibration_report,
    validate_prediction_frame,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty research output directory {path}; "
            "choose another --out-dir or pass --overwrite"
        )
    path.mkdir(parents=True, exist_ok=True)


def _load_full_config(model_path: Path) -> dict[str, Any]:
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    if set(bundle) != {"components", "config"}:
        raise ValueError("full model bundle must contain exactly components and config")
    config = bundle["config"]
    if config.get("class_names") != list(FULL_CLASS_NAMES):
        raise ValueError("full model class order is not down/flat/up")
    return config


def _flatten_reliability(report: dict[str, Any], scope: str) -> list[dict[str, Any]]:
    calibration = report["calibration"]
    rows = [
        {"scope": scope, "kind": "top_label", "class": "predicted", **row}
        for row in calibration["top_label_bins"]
    ]
    for class_name, class_report in calibration["classes"].items():
        rows.extend(
            {"scope": scope, "kind": "one_vs_rest", "class": class_name, **row}
            for row in class_report["bins"]
        )
    return rows


def _walk_forward_sources(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--fold-prediction values must use FOLD=PATH format")
        fold, raw_path = value.split("=", 1)
        fold = fold.strip()
        if not fold or fold in result:
            raise ValueError("walk-forward fold names must be non-empty and unique")
        result[fold] = Path(raw_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "evaluate saved full-window LSTM probabilities, causal baselines, calibration, "
            "component ablations, and true per-refit walk-forward prediction files"
        )
    )
    parser.add_argument(
        "--full-predictions", type=Path,
        default=OUTPUT_DIR / "lstm_full" / "test_predictions.csv",
    )
    parser.add_argument(
        "--conditional-predictions", type=Path,
        default=OUTPUT_DIR / "lstm" / "test_predictions.csv",
    )
    parser.add_argument(
        "--model", type=Path, default=OUTPUT_DIR / "lstm_full" / "model.pt"
    )
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_research")
    parser.add_argument("--n-bins", type=_positive_int, default=10)
    parser.add_argument("--batch-size", type=_positive_int, default=512)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument(
        "--evaluate-components", action="store_true",
        help="reload raw val/test features for joint/two-stage/fused structural ablation",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="fit temperature on validation predictions only and apply it to test",
    )
    parser.add_argument("--min-train-dates", type=_positive_int, default=180)
    parser.add_argument("--validation-dates", type=_positive_int, default=10)
    parser.add_argument("--test-dates", type=_positive_int, default=10)
    parser.add_argument("--step-dates", type=_positive_int, default=10)
    parser.add_argument("--train-window-dates", type=_positive_int, default=None)
    parser.add_argument("--max-folds", type=_positive_int, default=None)
    parser.add_argument(
        "--fold-prediction", action="append", default=[], metavar="FOLD=PATH",
        help=(
            "test predictions from one independently refitted fold; repeat for multiple folds"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_bins < 2:
        parser.error("--n-bins must be at least 2")
    if args.calibrate and not args.evaluate_components:
        parser.error("--calibrate requires --evaluate-components to recreate validation probabilities")

    _prepare_output(args.out_dir, args.overwrite)
    full = validate_prediction_frame(args.full_predictions, FULL_CLASS_NAMES)
    probability_columns = ["prob_down", "prob_flat", "prob_up"]
    labels = full["true_label"].to_numpy(np.int64)
    probability = full[probability_columns].to_numpy(float)
    config = _load_full_config(args.model)
    training_rates = config.get("class_rates", {}).get("train")

    full_metrics = probability_metrics(
        labels, probability, FULL_CLASS_NAMES, args.n_bins, include_reliability=True
    )
    daily = date_stability_report(full, FULL_CLASS_NAMES, args.n_bins)
    report: dict[str, Any] = {
        "methodology": {
            "saved_test_scope": config.get("splits", {}).get("test"),
            "test_history_warning": (
                "The saved test dates have already been inspected. These diagnostics improve "
                "measurement but do not turn that interval into a fresh blind test."
            ),
            "date_stability_is_not_walk_forward": True,
            "walk_forward_requirement": (
                "Each manifest fold must refit preprocessing, all three components, fusion, "
                "confidence thresholds, and temperature using only that fold's train/validation "
                "dates before its test file may be passed via --fold-prediction."
            ),
        },
        "full_window_model": full_metrics,
        "baselines": baseline_report(full, training_rates, args.n_bins),
        "date_stability": daily,
    }
    if args.conditional_predictions.exists():
        report["conditional_direction_comparison"] = compare_conditional_binary_model(
            full, args.conditional_predictions, args.n_bins
        )

    component_test: dict[str, Any] | None = None
    reliability_rows = _flatten_reliability(full_metrics, "saved_test_uncalibrated")
    if args.evaluate_components:
        component_test = evaluate_saved_components(
            args.model, args.data_dir, "test", args.batch_size, args.device, args.n_bins
        )
        replay_frame = component_test["prediction_frame"]
        replay_key = ["stock", "target_time"]
        aligned_replay = full[replay_key].merge(
            replay_frame[[
                *replay_key,
                "fused_prob_down", "fused_prob_flat", "fused_prob_up",
            ]],
            on=replay_key,
            how="left",
            validate="one_to_one",
        )
        replayed = aligned_replay[[
            "fused_prob_down", "fused_prob_flat", "fused_prob_up"
        ]].to_numpy(float)
        if replayed.shape != probability.shape or not np.isfinite(replayed).all() or not np.allclose(
            replayed, probability, atol=2e-6, rtol=2e-6
        ):
            raise RuntimeError("strict component replay disagrees with saved test probabilities")
        report["component_structural_ablation"] = {
            "interpretation": (
                "joint-only and two-stage-only reuse frozen branches; this is a structural "
                "ablation, not a feature-removal retraining experiment"
            ),
            "date_range": component_test["date_range"],
            "metrics": component_test["metrics"],
        }
        component_test["prediction_frame"].to_csv(
            args.out_dir / "component_test_predictions.csv", index=False
        )

    if args.calibrate:
        validation = evaluate_saved_components(
            args.model, args.data_dir, "val", args.batch_size, args.device, args.n_bins
        )
        calibration, calibrated_test = temperature_calibration_report(
            validation["labels"], validation["probabilities"]["fused"],
            labels, probability, FULL_CLASS_NAMES, args.n_bins,
        )
        report["temperature_calibration"] = calibration
        calibrated_metrics = probability_metrics(
            labels, calibrated_test, FULL_CLASS_NAMES, args.n_bins,
            include_reliability=True,
        )
        report["calibrated_test_model"] = calibrated_metrics
        reliability_rows.extend(
            _flatten_reliability(calibrated_metrics, "saved_test_validation_temperature_scaled")
        )
        # Saved selective flags were selected on the uncalibrated validation
        # confidence distribution and must not be carried onto new scores.
        calibrated_frame = full.drop(
            columns=[column for column in full.columns if column.startswith("selected_")],
            errors="ignore",
        ).copy()
        for class_id, column in enumerate(probability_columns):
            calibrated_frame[column] = calibrated_test[:, class_id]
        calibrated_frame["predicted_label"] = calibrated_test.argmax(axis=1)
        calibrated_frame["confidence"] = calibrated_test.max(axis=1)
        calibrated_frame.to_csv(args.out_dir / "calibrated_test_predictions.csv", index=False)

    available_dates = sorted(path.stem for path in (args.data_dir / "close").glob("*.csv"))
    folds = make_walk_forward_splits(
        available_dates,
        min_train_dates=args.min_train_dates,
        validation_dates=args.validation_dates,
        test_dates=args.test_dates,
        step_dates=args.step_dates,
        train_window_dates=args.train_window_dates,
        max_folds=args.max_folds,
    )
    report["walk_forward_manifest"] = {
        "status": "protocol_only_until_each_fold_is_independently_refitted",
        "fold_count": len(folds),
        "folds": folds,
    }
    pd.DataFrame(folds).to_csv(args.out_dir / "walk_forward_manifest.csv", index=False)

    if args.fold_prediction:
        sources = _walk_forward_sources(args.fold_prediction)
        report["walk_forward_oos"] = aggregate_walk_forward_predictions(
            sources, FULL_CLASS_NAMES, args.n_bins
        )

    per_date = pd.DataFrame(daily["per_date"])
    # Nested confusion matrices are useful in JSON and noisy in the tabular view.
    per_date.drop(columns=["confusion_matrix"], errors="ignore").to_csv(
        args.out_dir / "date_stability.csv", index=False
    )
    pd.DataFrame(reliability_rows).to_csv(
        args.out_dir / "reliability_bins.csv", index=False
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(
        pd.to_datetime(per_date["date"]), per_date["accuracy"] * 100.0,
        marker="o",
    )
    axes[0].axhline(full_metrics["accuracy"] * 100.0, color="black", linestyle="--")
    axes[0].set_title("Frozen-model accuracy by date")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(alpha=0.3)

    reliability_frame = pd.DataFrame(reliability_rows)
    top_label = reliability_frame[
        (reliability_frame["kind"] == "top_label")
        & reliability_frame["mean_forecast"].notna()
    ]
    for scope, group in top_label.groupby("scope", sort=False):
        axes[1].plot(
            group["mean_forecast"], group["observed_rate"], marker="o", label=scope
        )
    axes[1].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Top-label reliability")
    axes[1].set_xlabel("Mean forecast confidence")
    axes[1].set_ylabel("Observed accuracy")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)

    if component_test is not None:
        ablation = report["component_structural_ablation"]["metrics"]
        names = ["joint_only", "two_stage_only", "fused_ensemble"]
        x = np.arange(len(names))
        axes[2].bar(x - 0.18, [ablation[name]["accuracy"] for name in names], 0.36, label="Accuracy")
        axes[2].bar(x + 0.18, [ablation[name]["macro_f1"] for name in names], 0.36, label="Macro F1")
        axes[2].set_xticks(x, [name.replace("_", "\n") for name in names])
        axes[2].set_ylim(0.40, 0.50)
        axes[2].set_title("Structural ablation")
        axes[2].legend()
        axes[2].grid(axis="y", alpha=0.3)
    else:
        axes[2].axis("off")
    figure.tight_layout()
    figure.savefig(args.out_dir / "lstm_research_diagnostics.png", dpi=170)
    plt.close(figure)
    _dump_json(args.out_dir / "research_metrics.json", report)
    readme = [
        "# LSTM 研究增强与严格评估",
        "",
        "本目录补充强基线、概率校准、组件结构消融、逐日稳定性和 walk-forward 协议。",
        "保存测试段已经在开发中被查看；本报告不会把它重新称为盲测。",
        "",
        "## 当前冻结模型",
        "",
        f"- Accuracy：{full_metrics['accuracy']:.2%}",
        f"- Macro F1：{full_metrics['macro_f1']:.2%}",
        f"- Brier：{full_metrics['brier_score']:.6f}",
        f"- NLL：{full_metrics['calibration']['negative_log_likelihood']:.6f}",
        f"- Top-label ECE：{full_metrics['calibration']['top_label_ece']:.2%}",
        "",
    ]
    if component_test is not None:
        ablation = report["component_structural_ablation"]["metrics"]
        readme.extend([
            "## 结构消融",
            "",
            "| 结构 | Accuracy | Macro F1 |",
            "|---|---:|---:|",
            *[
                f"| {name} | {ablation[name]['accuracy']:.2%} | "
                f"{ablation[name]['macro_f1']:.2%} |"
                for name in ("joint_only", "two_stage_only", "fused_ensemble")
            ],
            "",
        ])
    if args.calibrate:
        calibration = report["temperature_calibration"]
        readme.extend([
            "## 验证集温度校准",
            "",
            f"温度 T={calibration['temperature']:.6f} 只使用验证集标签拟合。",
            f"测试 ECE 从 {calibration['test_before']['top_label_ece']:.2%} "
            f"降至 {calibration['test_after']['top_label_ece']:.2%}；"
            "测试 Brier 略有恶化，因此校准概率作为可选产物保留，没有覆盖原模型。",
            "",
        ])
    readme.extend([
        "## Walk-forward",
        "",
        f"已生成 {len(folds)} 个严格按时间排序的折。只有每折独立重训、冻结验证参数后生成的",
        "测试预测，才可通过 `--fold-prediction FOLD=PATH` 纳入 OOS 聚合；静态模型的逐日切片",
        "不算 walk-forward。",
        "",
    ])
    (args.out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(json.dumps({
        "out_dir": str(args.out_dir),
        "test_accuracy": full_metrics["accuracy"],
        "test_macro_f1": full_metrics["macro_f1"],
        "test_brier": full_metrics["brier_score"],
        "test_top_label_ece": full_metrics["calibration"]["top_label_ece"],
        "component_ablation": bool(args.evaluate_components),
        "temperature_calibration": bool(args.calibrate),
        "walk_forward_folds_planned": len(folds),
        "walk_forward_folds_evaluated": len(args.fold_prediction),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
