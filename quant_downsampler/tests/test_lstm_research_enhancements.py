from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.lstm_research import (
    aggregate_walk_forward_predictions,
    baseline_report,
    calibration_report,
    compare_conditional_binary_model,
    fit_temperature,
    make_walk_forward_splits,
    multiclass_brier_score,
    temperature_scale,
    validate_prediction_frame,
)


def _full_frame(start: str = "2026-01-01", dates: int = 2) -> pd.DataFrame:
    rows = []
    for day_id, day in enumerate(pd.date_range(start, periods=dates, freq="D")):
        labels = [0, 1, 2, 0]
        probabilities = [
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.3, 0.2, 0.5],
        ]
        for minute, (label, probability) in enumerate(zip(labels, probabilities), start=31):
            target = day + pd.Timedelta(hours=10, minutes=minute)
            rows.append({
                "stock": "S",
                "date": day.strftime("%Y-%m-%d"),
                "window_end": (target - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
                "target_time": target.strftime("%Y-%m-%d %H:%M:%S"),
                "true_label": label,
                "predicted_label": int(np.argmax(probability)),
                "prob_down": probability[0],
                "prob_flat": probability[1],
                "prob_up": probability[2],
                "confidence": max(probability),
            })
    return pd.DataFrame(rows)


class LSTMResearchEnhancementTests(unittest.TestCase):
    def test_prediction_validation_rejects_stale_labels_and_bad_normalisation(self) -> None:
        frame = _full_frame(dates=1)
        frame.loc[0, "predicted_label"] = 2
        with self.assertRaisesRegex(ValueError, "predicted_label"):
            validate_prediction_frame(frame)
        frame = _full_frame(dates=1)
        frame.loc[0, "prob_down"] = 0.6
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_prediction_frame(frame)

    def test_multiclass_brier_has_documented_unscaled_value(self) -> None:
        labels = np.array([0, 1])
        perfect = np.array([[1.0, 0.0], [0.0, 1.0]])
        opposite = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(multiclass_brier_score(labels, perfect), 0.0)
        self.assertEqual(multiclass_brier_score(labels, opposite), 2.0)

    def test_reliability_bins_include_probability_one_and_ece_is_exact(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probability = np.array([
            [1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]
        ])
        report = calibration_report(labels, probability, ("no", "yes"), n_bins=5)
        self.assertEqual(sum(row["count"] for row in report["top_label_bins"]), 4)
        self.assertAlmostEqual(report["top_label_ece"], 0.1)
        self.assertAlmostEqual(report["brier_score"], 0.04)

    def test_temperature_is_fit_on_supplied_validation_labels(self) -> None:
        # Correct predictions are intentionally overconfident; T > 1 lowers NLL.
        labels = np.array([0, 1, 0, 1])
        probability = np.array([
            [0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.9, 0.1]
        ])
        temperature = fit_temperature(labels, probability)
        scaled = temperature_scale(probability, temperature)
        before = calibration_report(labels, probability, ("a", "b"), 5)
        after = calibration_report(labels, scaled, ("a", "b"), 5)
        self.assertGreater(temperature, 1.0)
        self.assertLess(after["negative_log_likelihood"], before["negative_log_likelihood"])
        self.assertTrue(np.allclose(scaled.sum(axis=1), 1.0))

    def test_last_move_baselines_require_contiguous_previous_target(self) -> None:
        frame = _full_frame(dates=2)
        report = baseline_report(frame, training_class_rates=[0.3, 0.4, 0.3])
        persistence = report["last_move_persistence"]
        self.assertEqual(persistence["n"], 6)
        self.assertEqual(persistence["excluded_without_contiguous_previous_target"], 2)
        self.assertAlmostEqual(persistence["coverage"], 0.75)

    def test_conditional_binary_comparison_aligns_nonflat_timestamps(self) -> None:
        full = _full_frame(dates=1)
        nonflat = full[full["true_label"] != 1].copy()
        binary = nonflat[["stock", "date", "window_end", "target_time"]].copy()
        binary["true_label"] = (nonflat["true_label"].to_numpy() == 2).astype(int)
        binary["prob_down"] = [0.8, 0.2, 0.6]
        binary["prob_up"] = 1.0 - binary["prob_down"]
        binary["predicted_label"] = binary[["prob_down", "prob_up"]].to_numpy().argmax(axis=1)
        report = compare_conditional_binary_model(full, binary)
        self.assertEqual(report["n"], 3)
        self.assertAlmostEqual(report["share_of_all_full_windows"], 0.75)
        self.assertIn("future-conditioned", report["warning"])

    def test_walk_forward_manifest_is_strictly_chronological(self) -> None:
        dates = pd.date_range("2025-01-01", periods=20, freq="D")
        folds = make_walk_forward_splits(
            dates, min_train_dates=8, validation_dates=3, test_dates=2, step_dates=2
        )
        self.assertEqual(len(folds), 4)
        for previous, current in zip(folds, folds[1:]):
            self.assertLess(previous["train"][1], previous["validation"][0])
            self.assertLess(previous["validation"][1], previous["test"][0])
            self.assertLess(previous["test"][1], current["test"][0])
        trailing = make_walk_forward_splits(
            dates, 8, 3, 2, 2, train_window_dates=8, max_folds=2
        )
        self.assertEqual(trailing[1]["n_train_dates"], 8)
        self.assertEqual(trailing[1]["mode"], "trailing")

    def test_walk_forward_aggregation_rejects_overlapping_test_samples(self) -> None:
        first = _full_frame("2026-01-01", dates=1)
        second = _full_frame("2026-01-02", dates=1)
        report = aggregate_walk_forward_predictions({"f1": first, "f2": second})
        self.assertEqual(report["n_folds"], 2)
        self.assertEqual(report["pooled"]["n"], 8)
        with self.assertRaisesRegex(ValueError, "overlap"):
            aggregate_walk_forward_predictions({"f1": first, "f2": first.copy()})

        # Disjoint stocks do not make overlapping test dates independent folds.
        same_date_other_stock = first.copy()
        same_date_other_stock["stock"] = "OTHER"
        with self.assertRaisesRegex(ValueError, "calendar date"):
            aggregate_walk_forward_predictions({
                "f1": first, "f2": same_date_other_stock,
            })

    def test_prediction_validation_checks_window_target_order(self) -> None:
        frame = _full_frame(dates=1)
        frame.loc[0, "window_end"] = frame.loc[0, "target_time"]
        with self.assertRaisesRegex(ValueError, "earlier"):
            validate_prediction_frame(frame)


if __name__ == "__main__":
    unittest.main()
