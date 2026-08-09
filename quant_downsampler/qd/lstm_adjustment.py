"""Validation-only fusion adjustment for an existing full LSTM bundle."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from .config import OUTPUT_DIR
from .lstm_full import blend_probabilities, evaluate_full_model, two_stage_probabilities
from .lstm_research import evaluate_saved_components, probability_metrics


def select_validation_fusion(
    labels: np.ndarray,
    direction_probability: np.ndarray,
    movement_probability: np.ndarray,
    joint_probability: np.ndarray,
    *,
    move_biases: np.ndarray,
    joint_weights: np.ndarray,
    objective: str = "macro_f1_then_accuracy",
    reference_bias: float = 0.0,
    reference_weight: float = 0.5,
) -> tuple[dict[str, float | str], np.ndarray]:
    """Select fusion parameters without consulting test labels."""
    y = np.asarray(labels, dtype=np.int64)
    direction = np.asarray(direction_probability, dtype=np.float64)
    movement = np.asarray(movement_probability, dtype=np.float64)
    joint = np.asarray(joint_probability, dtype=np.float64)
    if objective not in {"macro_f1_then_accuracy", "accuracy_then_macro_f1"}:
        raise ValueError("unknown fusion objective")
    if (
        direction.shape != (len(y), 2)
        or movement.shape != (len(y), 2)
        or joint.shape != (len(y), 3)
    ):
        raise ValueError("component probabilities do not align with labels")
    if not len(y) or (y < 0).any() or (y > 2).any():
        raise ValueError("labels must be non-empty down/flat/up class indices")
    for name, probability in (
        ("direction", direction), ("movement", movement), ("joint", joint)
    ):
        if (
            not np.isfinite(probability).all()
            or (probability < 0.0).any()
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
        ):
            raise ValueError(f"{name} probabilities are invalid")
    biases = np.asarray(move_biases, dtype=np.float64)
    weights = np.asarray(joint_weights, dtype=np.float64)
    if biases.ndim != 1 or weights.ndim != 1 or not len(biases) or not len(weights):
        raise ValueError("fusion grids cannot be empty")
    if (
        not np.isfinite(biases).all()
        or not np.isfinite(weights).all()
        or (weights < 0.0).any()
        or (weights > 1.0).any()
    ):
        raise ValueError("fusion grids contain invalid values")
    if not np.isfinite([reference_bias, reference_weight]).all():
        raise ValueError("reference fusion parameters must be finite")

    best_score: tuple[float, ...] | None = None
    best: dict[str, float | str] | None = None
    best_probability: np.ndarray | None = None
    for move_bias in biases:
        staged = two_stage_probabilities(
            movement[:, 1], direction[:, 1], float(move_bias)
        )
        for joint_weight in weights:
            probability = blend_probabilities(joint, staged, float(joint_weight))
            predicted = probability.argmax(axis=1)
            accuracy = float(accuracy_score(y, predicted))
            macro_f1 = float(f1_score(y, predicted, average="macro", zero_division=0))
            primary = (
                (macro_f1, accuracy)
                if objective == "macro_f1_then_accuracy"
                else (accuracy, macro_f1)
            )
            score = (
                *primary,
                -abs(float(move_bias) - reference_bias),
                -abs(float(joint_weight) - reference_weight),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "objective": objective,
                    "move_bias": float(move_bias),
                    "joint_weight": float(joint_weight),
                    "two_stage_weight": float(1.0 - joint_weight),
                    "validation_accuracy": accuracy,
                    "validation_macro_f1": macro_f1,
                }
                best_probability = probability
    if best is None or best_probability is None:
        raise RuntimeError("fusion selection produced no candidate")
    return best, best_probability


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if not np.isfinite([start, stop, step]).all() or step <= 0.0 or start > stop:
        raise ValueError("invalid grid bounds")
    count = int(round((stop - start) / step))
    values = start + np.arange(count + 1, dtype=np.float64) * step
    if values[-1] > stop + 1e-9:
        values = values[:-1]
    return np.round(values, 10)


def _prepare_output(target: Path, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty adjusted-model directory {target}"
        )
    target.mkdir(parents=True, exist_ok=True)


def adjust_saved_fusion(
    model_path: str | Path = OUTPUT_DIR / "lstm_full" / "model.pt",
    data_dir: str | Path = OUTPUT_DIR / "minute",
    out_dir: str | Path = OUTPUT_DIR / "lstm_adjusted",
    *,
    objective: str = "macro_f1_then_accuracy",
    move_bias_min: float = -0.30,
    move_bias_max: float = 0.30,
    move_bias_step: float = 0.01,
    joint_weight_step: float = 0.01,
    balanced_quantile: float = 0.70,
    strict_quantile: float = 0.90,
    batch_size: int = 512,
    device: str = "cpu",
    overwrite: bool = False,
) -> dict:
    """Freeze a validation-selected bundle, then evaluate its test split once."""
    if not 0.0 < balanced_quantile < strict_quantile < 1.0:
        raise ValueError("require 0 < balanced_quantile < strict_quantile < 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    source = Path(model_path)
    target = Path(out_dir)
    _prepare_output(target, overwrite)
    move_biases = _grid(move_bias_min, move_bias_max, move_bias_step)
    joint_weights = _grid(0.0, 1.0, joint_weight_step)

    bundle = torch.load(source, map_location="cpu", weights_only=False)
    original_config = dict(bundle["config"])
    validation = evaluate_saved_components(
        source, data_dir, "val", batch_size, device, 10
    )
    selection, validation_probability = select_validation_fusion(
        validation["labels"],
        validation["probabilities"]["direction"],
        validation["probabilities"]["movement"],
        validation["probabilities"]["joint"],
        move_biases=move_biases,
        joint_weights=joint_weights,
        objective=objective,
        reference_bias=float(original_config["move_bias"]),
        reference_weight=float(original_config["joint_weight"]),
    )
    validation_confidence = validation_probability.max(axis=1)
    thresholds = {
        "balanced": float(np.quantile(validation_confidence, balanced_quantile)),
        "strict": float(np.quantile(validation_confidence, strict_quantile)),
    }
    frozen_at = datetime.now(timezone.utc).isoformat()
    config = bundle["config"]
    config["move_bias"] = selection["move_bias"]
    config["joint_weight"] = selection["joint_weight"]
    config["two_stage_weight"] = selection["two_stage_weight"]
    config["selective_thresholds"] = thresholds
    config["fusion_selection"] = {
        **selection,
        "move_bias_grid": move_biases.tolist(),
        "joint_weight_grid": joint_weights.tolist(),
        "fit_scope": "saved_validation_split_only",
        "frozen_at_utc": frozen_at,
        "source_model": str(source),
    }
    config["adjustment"] = {
        "kind": "validation_only_fusion_retune",
        "source_move_bias": float(original_config["move_bias"]),
        "source_joint_weight": float(original_config["joint_weight"]),
        "source_model": str(source),
        "test_unopened_until_after_bundle_write": True,
    }
    adjusted_model = target / "model.pt"
    torch.save(bundle, adjusted_model)
    freeze_record = {
        "selection": selection,
        "selective_thresholds": thresholds,
        "frozen_at_utc": frozen_at,
        "test_evaluation_count": 0,
    }
    freeze_path = target / "selection_frozen_before_test.json"
    freeze_path.write_text(
        json.dumps(freeze_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # The adjusted bundle and validation thresholds are now immutable inputs
    # to this single test evaluation.
    test = evaluate_full_model(
        adjusted_model, data_dir, split="test", batch_size=batch_size, device=device
    )
    test_probability = test["probability"]
    test_labels = test["labels"]
    metrics = probability_metrics(
        test_labels, test_probability, ("down", "flat", "up"), 10,
        include_reliability=True,
    )
    test_confidence = test_probability.max(axis=1)
    predicted = test_probability.argmax(axis=1)
    selective: dict[str, dict[str, float | int | None]] = {}
    selected_masks: dict[str, np.ndarray] = {}
    for name, threshold in thresholds.items():
        selected = test_confidence >= threshold
        selected_masks[name] = selected
        selected_n = int(selected.sum())
        selective[name] = {
            "validation_threshold": threshold,
            "test_coverage": float(selected.mean()),
            "test_accuracy": (
                float((predicted[selected] == test_labels[selected]).mean())
                if selected_n else None
            ),
            "test_n": selected_n,
        }
    metrics["selective_accuracy"] = selective
    metrics["validation_selection"] = selection
    metrics["coverage"] = test["metrics"]["coverage"]

    predictions = test["metadata"].copy()
    predictions["true_label"] = test_labels
    predictions["predicted_label"] = predicted
    predictions["prob_down"] = test_probability[:, 0]
    predictions["prob_flat"] = test_probability[:, 1]
    predictions["prob_up"] = test_probability[:, 2]
    predictions["confidence"] = test_confidence
    predictions["selected_balanced"] = selected_masks["balanced"]
    predictions["selected_strict"] = selected_masks["strict"]
    predictions.to_csv(target / "test_predictions.csv", index=False)
    (target / "test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    freeze_record["test_evaluation_count"] = 1
    freeze_record["test_evaluated_at_utc"] = datetime.now(timezone.utc).isoformat()
    freeze_path.write_text(
        json.dumps(freeze_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (target / "README.md").write_text(
        "\n".join([
            "# 验证集驱动的 LSTM 融合调整",
            "",
            "本模型没有重训或查看测试标签选择参数。它复用原有三个组件，",
            "仅在保存的验证集上以 Macro F1 优先重新选择 move bias 与融合权重，",
            "冻结新模型包和置信阈值后才评价测试集。",
            "",
            f"- move bias：{float(original_config['move_bias']):.2f} → {selection['move_bias']:.2f}",
            f"- joint weight：{float(original_config['joint_weight']):.2f} → {selection['joint_weight']:.2f}",
            f"- 测试 Accuracy：{metrics['accuracy']:.2%}",
            f"- 测试 Macro F1：{metrics['macro_f1']:.2%}",
            f"- 测试 Brier：{metrics['brier_score']:.6f}",
            "",
            "现有测试段已在开发中被查看，因此这仍是统一口径诊断；最终确认必须依靠新增日期或多折独立重训。",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    return {
        "out_dir": str(target),
        "model": str(adjusted_model),
        "selection": selection,
        "test_metrics": metrics,
    }
