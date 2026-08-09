"""Validation-frozen hybrid of the sequence LSTM and a strong tree baseline.

The hybrid is deliberately simple: it linearly blends the saved full-window
LSTM probabilities with a saved HistGradientBoosting baseline.  The blend
weight and selective-confidence thresholds are chosen on the validation
split, persisted, and only then applied to the test split.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from .config import OUTPUT_DIR
from .lstm_baselines import _aligned_probability, _load_summary_split
from .lstm_research import evaluate_saved_components, probability_metrics


CLASS_NAMES = ("down", "flat", "up")
DEFAULT_WEIGHT_GRID = np.linspace(0.0, 1.0, 101)
METADATA_COLUMNS = ("stock", "stock_id", "date", "window_end", "target_time")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _normalise_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES) or not len(values):
        raise ValueError("probability must have non-empty (samples, 3) shape")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probability must be finite and non-negative")
    row_sum = values.sum(axis=1, keepdims=True)
    if (row_sum <= 0.0).any():
        raise ValueError("each probability row must have positive mass")
    return values / row_sum


def blend_model_probabilities(
    lstm_probability: np.ndarray,
    tree_probability: np.ndarray,
    lstm_weight: float,
) -> np.ndarray:
    """Blend two aligned three-class probability matrices."""

    if not 0.0 <= float(lstm_weight) <= 1.0:
        raise ValueError("lstm_weight must be within [0, 1]")
    lstm = _normalise_probability(lstm_probability)
    tree = _normalise_probability(tree_probability)
    if lstm.shape != tree.shape:
        raise ValueError("LSTM and tree probabilities are not aligned")
    mixed = float(lstm_weight) * lstm + (1.0 - float(lstm_weight)) * tree
    return mixed / mixed.sum(axis=1, keepdims=True)


def select_validation_blend(
    labels: np.ndarray,
    lstm_probability: np.ndarray,
    tree_probability: np.ndarray,
    *,
    weights: Sequence[float] = DEFAULT_WEIGHT_GRID,
    objective: str = "macro_f1_then_accuracy_then_nll",
) -> tuple[dict[str, float | str], np.ndarray]:
    """Select the hybrid weight using validation labels only."""

    if objective not in {
        "macro_f1_then_accuracy_then_nll",
        "accuracy_then_macro_f1_then_nll",
    }:
        raise ValueError("unknown blend objective")
    y = np.asarray(labels, dtype=np.int64)
    lstm = _normalise_probability(lstm_probability)
    tree = _normalise_probability(tree_probability)
    if lstm.shape != tree.shape or len(y) != len(lstm):
        raise ValueError("labels and component probabilities are not aligned")
    if len(y) == 0 or y.min() < 0 or y.max() >= len(CLASS_NAMES):
        raise ValueError("labels are empty or outside the three-class range")
    grid = np.asarray(list(weights), dtype=np.float64)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or (grid < 0.0).any()
        or (grid > 1.0).any()
    ):
        raise ValueError("weights must be a finite non-empty grid within [0, 1]")

    best_score: tuple[float, ...] | None = None
    best_selection: dict[str, float | str] | None = None
    best_probability: np.ndarray | None = None
    for weight in grid:
        probability = blend_model_probabilities(lstm, tree, float(weight))
        predicted = probability.argmax(axis=1)
        accuracy = float(accuracy_score(y, predicted))
        macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
        nll = float(-np.log(np.clip(probability[np.arange(len(y)), y], 1e-15, 1.0)).mean())
        primary = (
            (macro_f1, accuracy)
            if objective == "macro_f1_then_accuracy_then_nll"
            else (accuracy, macro_f1)
        )
        # Prefer lower NLL, then the simpler balanced blend, for deterministic
        # ties that have identical hard predictions.
        score = (*primary, -nll, -abs(float(weight) - 0.5))
        if best_score is None or score > best_score:
            best_score = score
            best_selection = {
                "objective": objective,
                "lstm_weight": float(weight),
                "tree_weight": float(1.0 - weight),
                "validation_accuracy": accuracy,
                "validation_macro_f1": macro_f1,
                "validation_negative_log_likelihood": nll,
            }
            best_probability = probability
    if best_selection is None or best_probability is None:
        raise RuntimeError("validation blend selection produced no candidate")
    return best_selection, best_probability


def _assert_aligned(
    lstm_labels: np.ndarray,
    lstm_metadata: pd.DataFrame,
    tree_labels: np.ndarray,
    tree_metadata: pd.DataFrame,
) -> None:
    if not np.array_equal(np.asarray(lstm_labels), np.asarray(tree_labels)):
        raise ValueError("LSTM and tree labels are not aligned")
    missing = [
        name
        for name in METADATA_COLUMNS
        if name not in lstm_metadata or name not in tree_metadata
    ]
    if missing:
        raise ValueError(f"metadata columns are missing: {missing}")
    left = lstm_metadata[list(METADATA_COLUMNS)].reset_index(drop=True).astype(str)
    right = tree_metadata[list(METADATA_COLUMNS)].reset_index(drop=True).astype(str)
    if not left.equals(right):
        raise ValueError("LSTM and tree sample timestamps are not aligned")


def _mcnemar(first_correct: np.ndarray, second_correct: np.ndarray) -> dict[str, Any]:
    first = np.asarray(first_correct, dtype=bool)
    second = np.asarray(second_correct, dtype=bool)
    if first.shape != second.shape:
        raise ValueError("correctness vectors are not aligned")
    first_only = int((first & ~second).sum())
    second_only = int((~first & second).sum())
    discordant = first_only + second_only
    p_value = 1.0 if discordant == 0 else float(
        binomtest(first_only, discordant, p=0.5, alternative="two-sided").pvalue
    )
    return {
        "hybrid_only_correct": first_only,
        "comparison_only_correct": second_only,
        "discordant": discordant,
        "exact_two_sided_p_value": p_value,
    }


def _prepare_output(target: Path, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty hybrid directory {target}"
        )
    target.mkdir(parents=True, exist_ok=True)


def _write_plot(target: Path, metrics: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Original LSTM", "HistGB", "Hybrid"]
    records = [
        metrics["comparators"]["original_lstm"],
        metrics["comparators"]["hist_gradient_boosting"],
        metrics["hybrid"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    x = np.arange(len(names))
    accuracy_bars = axes[0].bar(
        x - 0.18, [r["accuracy"] for r in records], 0.36, label="Accuracy"
    )
    f1_bars = axes[0].bar(
        x + 0.18, [r["macro_f1"] for r in records], 0.36, label="Macro-F1"
    )
    axes[0].set_ylim(0.40, 0.49)
    axes[0].set_xticks(x, names, rotation=12)
    axes[0].set_ylabel("Score")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].set_title("Three-class classification")
    axes[0].bar_label(accuracy_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[0].bar_label(f1_bars, fmt="%.3f", padding=2, fontsize=8)
    brier_bars = axes[1].bar(
        x - 0.18, [r["brier_score"] for r in records], 0.36, label="Brier"
    )
    nll_bars = axes[1].bar(
        x + 0.18,
        [r["calibration"]["negative_log_likelihood"] for r in records],
        0.36,
        label="NLL",
    )
    axes[1].set_xticks(x, names, rotation=12)
    axes[1].set_ylabel("Loss (lower is better)")
    axes[1].legend(frameon=False, loc="lower left")
    axes[1].set_title("Probability quality")
    axes[1].bar_label(brier_bars, fmt="%.3f", padding=2, fontsize=8)
    axes[1].bar_label(nll_bars, fmt="%.3f", padding=2, fontsize=8)
    fig.suptitle("Validation-frozen LSTM + HistGB hybrid")
    p_lstm = metrics["paired_comparisons"]["hybrid_vs_original_lstm"][
        "exact_two_sided_p_value"
    ]
    p_tree = metrics["paired_comparisons"]["hybrid_vs_hist_gradient_boosting"][
        "exact_two_sided_p_value"
    ]
    fig.text(
        0.5,
        0.005,
        f"Paired McNemar: hybrid vs LSTM p={p_lstm:.3f}; vs HistGB p={p_tree:.3f} (not significant)",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(target / "lstm_hybrid_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_lstm_hybrid(
    lstm_model_path: str | Path = OUTPUT_DIR / "lstm_full" / "model.pt",
    baseline_model_path: str | Path = OUTPUT_DIR / "lstm_baselines" / "models.joblib",
    data_dir: str | Path = OUTPUT_DIR / "minute",
    out_dir: str | Path = OUTPUT_DIR / "lstm_hybrid",
    *,
    tree_model_name: str = "hist_gradient_boosting",
    weight_step: float = 0.01,
    objective: str = "macro_f1_then_accuracy_then_nll",
    balanced_quantile: float = 0.70,
    strict_quantile: float = 0.90,
    batch_size: int = 512,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Freeze a validation-selected hybrid and evaluate its test split once."""

    if not 0.0 < weight_step <= 1.0:
        raise ValueError("weight_step must be within (0, 1]")
    if not 0.0 < balanced_quantile < strict_quantile < 1.0:
        raise ValueError("require 0 < balanced_quantile < strict_quantile < 1")
    lstm_path = Path(lstm_model_path)
    baseline_path = Path(baseline_model_path)
    minute_dir = Path(data_dir)
    target = Path(out_dir)
    _prepare_output(target, overwrite)

    baseline_bundle = joblib.load(baseline_path)
    if tree_model_name not in baseline_bundle.get("models", {}):
        raise ValueError(f"baseline bundle does not contain {tree_model_name!r}")
    baseline_config = baseline_bundle["config"]
    lstm_bundle = torch.load(lstm_path, map_location="cpu", weights_only=False)
    lstm_config = lstm_bundle["config"]
    stocks = list(lstm_config["stock_codes"])
    if list(baseline_config["stock_codes"]) != stocks:
        raise ValueError("baseline and LSTM stock universes differ")
    for split in ("val", "test"):
        if tuple(baseline_config["splits"][split]) != tuple(lstm_config["splits"][split]):
            raise ValueError(f"baseline and LSTM {split} date ranges differ")
    if baseline_config.get("target_mode") != "three_class":
        raise ValueError("baseline bundle is not a full-window three-class model")

    validation_lstm = evaluate_saved_components(
        lstm_path, minute_dir, "val", batch_size, device, 10
    )
    validation_tree_x, validation_tree_y, _, _, validation_tree_metadata, _ = (
        _load_summary_split(
            stocks,
            tuple(baseline_config["splits"]["val"]),
            int(baseline_config["seq_len"]),
            minute_dir,
            str(baseline_config["feature_set"]),
            True,
        )
    )
    assert validation_tree_metadata is not None
    _assert_aligned(
        validation_lstm["labels"],
        validation_lstm["prediction_frame"],
        validation_tree_y,
        validation_tree_metadata,
    )
    validation_tree_probability = _aligned_probability(
        baseline_bundle["models"][tree_model_name], validation_tree_x
    )
    weights = np.arange(0.0, 1.0 + 1e-12, weight_step, dtype=np.float64)
    if weights[-1] < 1.0 - 1e-12:
        weights = np.append(weights, 1.0)
    else:
        weights[-1] = 1.0
    selection, validation_probability = select_validation_blend(
        validation_lstm["labels"],
        validation_lstm["probabilities"]["fused"],
        validation_tree_probability,
        weights=weights,
        objective=objective,
    )
    validation_confidence = validation_probability.max(axis=1)
    thresholds = {
        "balanced": float(np.quantile(validation_confidence, balanced_quantile)),
        "strict": float(np.quantile(validation_confidence, strict_quantile)),
    }
    frozen_at = _utc_now()
    freeze_record: dict[str, Any] = {
        "frozen_before_hybrid_test_load": True,
        "frozen_at_utc": frozen_at,
        "selection": selection,
        "selective_thresholds": thresholds,
        "weight_grid": weights.tolist(),
        "test_evaluation_count": 0,
    }
    freeze_path = target / "selection_frozen_before_test.json"
    _json_dump(freeze_path, freeze_record)

    # No hybrid test labels or probabilities are loaded above this line.
    test_loaded_at = _utc_now()
    test_lstm = evaluate_saved_components(
        lstm_path, minute_dir, "test", batch_size, device, 10
    )
    test_tree_x, test_tree_y, _, test_coverage, test_tree_metadata, _ = (
        _load_summary_split(
            stocks,
            tuple(baseline_config["splits"]["test"]),
            int(baseline_config["seq_len"]),
            minute_dir,
            str(baseline_config["feature_set"]),
            True,
        )
    )
    assert test_tree_metadata is not None
    _assert_aligned(
        test_lstm["labels"],
        test_lstm["prediction_frame"],
        test_tree_y,
        test_tree_metadata,
    )
    test_tree_probability = _aligned_probability(
        baseline_bundle["models"][tree_model_name], test_tree_x
    )
    test_lstm_probability = test_lstm["probabilities"]["fused"]
    test_probability = blend_model_probabilities(
        test_lstm_probability,
        test_tree_probability,
        float(selection["lstm_weight"]),
    )
    labels = np.asarray(test_lstm["labels"], dtype=np.int64)
    predicted = test_probability.argmax(axis=1)
    hybrid_metrics = probability_metrics(
        labels, test_probability, CLASS_NAMES, 10, include_reliability=True
    )
    lstm_metrics = probability_metrics(
        labels, test_lstm_probability, CLASS_NAMES, 10, include_reliability=True
    )
    tree_metrics = probability_metrics(
        labels, test_tree_probability, CLASS_NAMES, 10, include_reliability=True
    )
    confidence = test_probability.max(axis=1)
    selected_masks: dict[str, np.ndarray] = {}
    selective: dict[str, dict[str, float | int]] = {}
    for name, threshold in thresholds.items():
        selected = confidence >= threshold
        selected_masks[name] = selected
        selective[name] = {
            "validation_threshold": float(threshold),
            "test_coverage": float(selected.mean()),
            "test_accuracy": float((predicted[selected] == labels[selected]).mean()),
            "test_n": int(selected.sum()),
        }
    hybrid_metrics["selective_accuracy"] = selective

    hybrid_correct = predicted == labels
    report: dict[str, Any] = {
        "methodology": {
            "selection_scope": "saved_validation_split_only",
            "test_history_warning": (
                "The fixed test dates were previously inspected elsewhere in the "
                "project. This run preserves validation-only selection but is not a "
                "fresh blind test."
            ),
        },
        "selection": selection,
        "hybrid": hybrid_metrics,
        "comparators": {
            "original_lstm": lstm_metrics,
            "hist_gradient_boosting": tree_metrics,
        },
        "paired_comparisons": {
            "hybrid_vs_original_lstm": _mcnemar(
                hybrid_correct, test_lstm_probability.argmax(axis=1) == labels
            ),
            "hybrid_vs_hist_gradient_boosting": _mcnemar(
                hybrid_correct, test_tree_probability.argmax(axis=1) == labels
            ),
        },
        "coverage": test_coverage,
        "audit": {
            "selection_frozen_at_utc": frozen_at,
            "test_loaded_at_utc": test_loaded_at,
            "test_loaded_after_selection_frozen": test_loaded_at >= frozen_at,
            "hybrid_test_evaluation_count": 1,
        },
    }

    predictions = test_lstm["prediction_frame"][list(METADATA_COLUMNS)].copy()
    predictions["true_label"] = labels
    predictions["predicted_label"] = predicted
    predictions["prob_down"] = test_probability[:, 0]
    predictions["prob_flat"] = test_probability[:, 1]
    predictions["prob_up"] = test_probability[:, 2]
    predictions["confidence"] = confidence
    predictions["selected_balanced"] = selected_masks["balanced"]
    predictions["selected_strict"] = selected_masks["strict"]
    predictions.to_csv(target / "test_predictions.csv", index=False, float_format="%.10f")
    _json_dump(target / "test_metrics.json", report)
    manifest = {
        "pipeline_version": 1,
        "kind": "validation_frozen_probability_hybrid",
        "class_names": list(CLASS_NAMES),
        "stock_codes": stocks,
        "splits": baseline_config["splits"],
        "selection": selection,
        "selective_thresholds": thresholds,
        "sources": {
            "lstm_model": {"path": str(lstm_path), "sha256": _sha256(lstm_path)},
            "baseline_model": {
                "path": str(baseline_path),
                "sha256": _sha256(baseline_path),
                "model_name": tree_model_name,
            },
        },
    }
    _json_dump(target / "model_manifest.json", manifest)
    freeze_record["test_evaluation_count"] = 1
    freeze_record["test_evaluated_at_utc"] = _utc_now()
    _json_dump(freeze_path, freeze_record)

    _write_plot(target, report)
    (target / "README.md").write_text(
        "\n".join([
            "# 验证集冻结的 LSTM + HistGB 混合模型",
            "",
            "融合权重只在验证集选择，选择记录先落盘，随后才加载测试区间。",
            f"LSTM / HistGB 权重为 {selection['lstm_weight']:.2f} / {selection['tree_weight']:.2f}。",
            "",
            "| 模型 | Accuracy | Macro-F1 | Brier | NLL |",
            "|---|---:|---:|---:|---:|",
            f"| 原 LSTM | {lstm_metrics['accuracy']:.2%} | {lstm_metrics['macro_f1']:.2%} | {lstm_metrics['brier_score']:.6f} | {lstm_metrics['calibration']['negative_log_likelihood']:.6f} |",
            f"| HistGB | {tree_metrics['accuracy']:.2%} | {tree_metrics['macro_f1']:.2%} | {tree_metrics['brier_score']:.6f} | {tree_metrics['calibration']['negative_log_likelihood']:.6f} |",
            f"| Hybrid | {hybrid_metrics['accuracy']:.2%} | {hybrid_metrics['macro_f1']:.2%} | {hybrid_metrics['brier_score']:.6f} | {hybrid_metrics['calibration']['negative_log_likelihood']:.6f} |",
            "",
            "测试段已在既往开发中被查看，因此上述结果是统一口径诊断，不是新的盲测。",
            "配对 McNemar 检验仍不显著，不能宣称混合模型已经建立稳定结构优势。",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    return {"out_dir": str(target), "report": report, "manifest": manifest}
