"""Validation-only audit of the saved LSTM ensemble fusion grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qd.config import OUTPUT_DIR
from qd.lstm_components import classification_metrics
from qd.lstm_ensemble import blend_probabilities, two_stage_probabilities


def _probability(table: pd.DataFrame, move_bias: float, joint_weight: float) -> np.ndarray:
    staged = two_stage_probabilities(
        table["movement_prob_move"].to_numpy(float),
        table["direction_prob_up"].to_numpy(float),
        move_bias,
    )
    joint = table[["joint_prob_down", "joint_prob_flat", "joint_prob_up"]].to_numpy(float)
    return blend_probabilities(joint, staged, joint_weight)


def _metrics(table: pd.DataFrame, probability: np.ndarray) -> dict:
    labels = table["true_label"].to_numpy(int)
    result = classification_metrics(labels, probability, threshold=None)
    return {
        "accuracy": float(result["accuracy"]),
        "macro_f1": float(result["f1"]),
        "confusion_matrix": result["confusion_matrix"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="expand the validation fusion grid without retraining components"
    )
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_DIR / "lstm_ensemble")
    parser.add_argument("--move-bias-min", type=float, default=-0.8)
    parser.add_argument("--move-bias-max", type=float, default=0.3)
    parser.add_argument("--move-bias-step", type=float, default=0.025)
    parser.add_argument("--joint-weight-step", type=float, default=0.025)
    args = parser.parse_args()

    validation = pd.read_csv(args.run_dir / "validation_predictions.csv")
    test = pd.read_csv(args.run_dir / "test_predictions.csv")
    saved_metrics = json.loads((args.run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    current_bias = float(saved_metrics["fusion"]["move_bias"])
    current_weight = float(saved_metrics["fusion"]["joint_weight"])

    biases = np.arange(
        args.move_bias_min, args.move_bias_max + args.move_bias_step / 2,
        args.move_bias_step,
    )
    weights = np.arange(0.0, 1.0 + args.joint_weight_step / 2, args.joint_weight_step)
    labels = validation["true_label"].to_numpy(int)
    rows = []
    best_score = None
    best = None
    for bias in biases:
        staged = two_stage_probabilities(
            validation["movement_prob_move"].to_numpy(float),
            validation["direction_prob_up"].to_numpy(float), float(bias),
        )
        joint = validation[
            ["joint_prob_down", "joint_prob_flat", "joint_prob_up"]
        ].to_numpy(float)
        for weight in weights:
            probability = blend_probabilities(joint, staged, float(weight))
            metrics = classification_metrics(labels, probability, threshold=None)
            row = {
                "move_bias": float(bias),
                "joint_weight": float(weight),
                "validation_accuracy": float(metrics["accuracy"]),
                "validation_macro_f1": float(metrics["f1"]),
            }
            rows.append(row)
            score = (
                row["validation_accuracy"], row["validation_macro_f1"],
                -abs(float(weight) - 0.5), -abs(float(bias)),
            )
            if best_score is None or score > best_score:
                best_score, best = score, row
    assert best is not None
    grid = pd.DataFrame(rows)
    grid.to_csv(args.run_dir / "fusion_validation_grid.csv", index=False)

    current_validation = _metrics(
        validation, _probability(validation, current_bias, current_weight)
    )
    candidate_validation_probability = _probability(
        validation, best["move_bias"], best["joint_weight"]
    )
    candidate_test_probability = _probability(test, best["move_bias"], best["joint_weight"])
    candidate_validation = _metrics(validation, candidate_validation_probability)
    candidate_test = _metrics(test, candidate_test_probability)

    val_confidence = candidate_validation_probability.max(axis=1)
    test_confidence = candidate_test_probability.max(axis=1)
    selective = {}
    for name, quantile in (("balanced", 0.70), ("strict", 0.90)):
        threshold = float(np.quantile(val_confidence, quantile))
        val_mask = val_confidence >= threshold
        test_mask = test_confidence >= threshold
        selective[name] = {
            "validation_threshold": threshold,
            "validation_coverage": float(val_mask.mean()),
            "validation_accuracy": float(
                (candidate_validation_probability[val_mask].argmax(axis=1)
                 == validation.loc[val_mask, "true_label"].to_numpy(int)).mean()
            ),
            "test_coverage": float(test_mask.mean()),
            "test_accuracy": float(
                (candidate_test_probability[test_mask].argmax(axis=1)
                 == test.loc[test_mask, "true_label"].to_numpy(int)).mean()
            ),
            "test_n": int(test_mask.sum()),
        }

    report = {
        "selection_rule": "validation accuracy, then validation macro F1",
        "grid": {
            "move_bias_min": args.move_bias_min,
            "move_bias_max": args.move_bias_max,
            "move_bias_step": args.move_bias_step,
            "joint_weight_step": args.joint_weight_step,
            "candidates": int(len(grid)),
        },
        "current": {
            "move_bias": current_bias,
            "joint_weight": current_weight,
            "validation": current_validation,
            "test": {
                "accuracy": saved_metrics["accuracy"],
                "macro_f1": saved_metrics["macro_f1"],
                "confusion_matrix": saved_metrics["confusion_matrix"],
            },
        },
        "expanded_grid_candidate": {
            "move_bias": best["move_bias"],
            "joint_weight": best["joint_weight"],
            "at_bias_boundary": bool(
                np.isclose(best["move_bias"], biases[0])
                or np.isclose(best["move_bias"], biases[-1])
            ),
            "validation": candidate_validation,
            "test": candidate_test,
            "selective_accuracy": selective,
        },
        "promotion_rule": (
            "Only promote if validation improves; test metrics are reported for diagnosis "
            "and are not part of selection."
        ),
    }
    (args.run_dir / "fusion_grid_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
