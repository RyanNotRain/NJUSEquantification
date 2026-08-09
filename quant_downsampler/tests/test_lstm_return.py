from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from qd.lstm_model import CHANNELS
from qd.lstm_return import (
    ReturnMultiTaskLSTM,
    _prepare_output_dir,
    attach_return_targets,
    fit_magnitude_transform,
    grouped_return_report,
    inverse_magnitude_predictions,
    load_return_model,
    prediction_metrics,
    run_return_lstm,
    select_opening_thresholds,
    transform_magnitude_targets,
)


def _minute_index(date: str) -> pd.DatetimeIndex:
    day = pd.Timestamp(date)
    morning = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=121, freq="min")
    afternoon = pd.date_range(day + pd.Timedelta(hours=13), periods=121, freq="min")
    return morning.append(afternoon)


def _write_synthetic_minute_data(root: Path, dates: list[str]) -> None:
    for channel in CHANNELS:
        (root / channel).mkdir(parents=True, exist_ok=True)
    for day_id, date in enumerate(dates):
        index = _minute_index(date)
        minute = np.arange(len(index), dtype=np.float64)
        close = 10.0 + 0.01 * ((minute.astype(int) + day_id) % 5)
        # Preserve the lunch boundary while producing up/down/flat outcomes.
        close[121:] += 0.03
        values: dict[str, np.ndarray] = {
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
            "volume": 1000.0 + minute,
            "trade_count": 20.0 + minute % 7,
            "amount": (1000.0 + minute) * close,
            "buy_volume": 520.0 + minute / 2.0,
            "sell_volume": 480.0 + minute / 2.0,
            "buy_amount": (520.0 + minute / 2.0) * close,
            "sell_amount": (480.0 + minute / 2.0) * close,
        }
        for channel in CHANNELS:
            table = pd.DataFrame({"A": values[channel]}, index=index)
            table.index.name = "datetime"
            table.to_csv(root / channel / f"{date}.csv")


class LSTMReturnTests(unittest.TestCase):
    def test_return_target_uses_exact_next_close_ratio(self) -> None:
        metadata = pd.DataFrame({
            "stock": ["A", "A"],
            "window_end": ["2025-01-02 10:00:00", "2025-01-02 10:01:00"],
            "target_time": ["2025-01-02 10:01:00", "2025-01-02 10:02:00"],
        })
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = pd.DataFrame(
                {"A": [100.0, 101.0, 99.0]},
                index=pd.to_datetime([
                    "2025-01-02 10:00:00",
                    "2025-01-02 10:01:00",
                    "2025-01-02 10:02:00",
                ]),
            )
            table.index.name = "datetime"
            table.to_csv(root / "20250102.csv")
            signed, absolute = attach_return_targets(metadata, root)
        np.testing.assert_allclose(
            signed,
            [100.0, (99.0 / 101.0 - 1.0) * 10_000.0],
        )
        np.testing.assert_allclose(absolute, np.abs(signed))

    def test_magnitude_transform_is_fitted_and_clipped_from_supplied_training(self) -> None:
        training = np.array([0.0, 5.0, 10.0, 0.0, 20.0, 40.0])
        stock_ids = np.array([0, 0, 0, 1, 1, 1])
        transform = fit_magnitude_transform(
            training, stock_ids, n_stocks=2, clip_quantile=0.9
        )
        np.testing.assert_allclose(transform["stock_scales_bps"], [7.5, 30.0])
        evaluation = np.array([75.0, 300.0])
        normalized = transform_magnitude_targets(
            evaluation, np.array([0, 1]), transform
        )
        self.assertTrue((normalized <= transform["normalized_clip"]).all())
        restored = inverse_magnitude_predictions(
            normalized, np.array([0, 1]), transform
        )
        self.assertTrue((restored <= evaluation).all())

    def test_multitask_model_has_shared_encoder_and_nonnegative_magnitude(self) -> None:
        model = ReturnMultiTaskLSTM(input_size=7, hidden_size=8, num_layers=1, dropout=0.0)
        logits, magnitude = model(torch.randn(4, 6, 7))
        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(magnitude.shape), (4,))
        self.assertTrue(torch.all(magnitude >= 0.0))
        self.assertIs(model.lstm, model.lstm)

    def test_grouped_return_report_detects_monotonic_signal(self) -> None:
        prediction = np.linspace(-2.0, 2.0, 100)
        report = grouped_return_report(prediction, prediction * 3.0, n_groups=10)
        self.assertAlmostEqual(report["monotonic_spearman"], 1.0)
        self.assertGreater(report["top_minus_bottom_bps"], 0.0)

    def test_magnitude_metrics_include_zero_prediction_baseline(self) -> None:
        labels = np.array([0, 1, 2])
        probability = np.eye(3, dtype=float)
        realised = np.array([-2.0, 0.0, 4.0])
        predicted_absolute = np.array([1.0, 1.0, 3.0])
        expected = np.array([-1.0, 0.0, 3.0])
        metrics = prediction_metrics(
            labels, probability, predicted_absolute, expected, realised
        )
        magnitude = metrics["magnitude"]
        self.assertAlmostEqual(magnitude["zero_prediction_baseline_mae_bps"], 2.0)
        self.assertAlmostEqual(magnitude["mae_bps"], 1.0)
        self.assertAlmostEqual(magnitude["mae_improvement_vs_zero_bps"], 1.0)
        signed = metrics["signed_expected_return"]
        self.assertAlmostEqual(signed["zero_prediction_baseline_mae_bps"], 2.0)
        self.assertAlmostEqual(signed["mae_bps"], 2.0 / 3.0)
        self.assertAlmostEqual(signed["mae_improvement_vs_zero_bps"], 4.0 / 3.0)

    def test_opening_threshold_candidates_respect_one_way_cost_floor(self) -> None:
        predictions = pd.DataFrame({
            "stock": ["A", "B", "A", "B"],
            "window_end": ["2025-01-02 10:00:00"] * 2
            + ["2025-01-02 10:01:00"] * 2,
            "target_time": ["2025-01-02 10:01:00"] * 2
            + ["2025-01-02 10:02:00"] * 2,
            "prob_down": [0.2, 0.6, 0.2, 0.6],
            "prob_flat": [0.2, 0.2, 0.2, 0.2],
            "prob_up": [0.6, 0.2, 0.6, 0.2],
            "expected_return_bps": [1.0, -2.0, 3.0, -4.0],
            "realised_return": [0.001, -0.001, 0.001, -0.001],
        })
        selected, table = select_opening_thresholds(
            predictions, base_cost_bps=5.0, quantiles=(0.0, 0.5, 0.9)
        )
        self.assertTrue((table["threshold_bps"] >= 5.0).all())
        self.assertGreaterEqual(selected["long_short"], 5.0)
        self.assertGreaterEqual(selected["long_only"], 5.0)

        gap_selected, gap_table = select_opening_thresholds(
            predictions,
            base_cost_bps=5.0,
            quantiles=(0.0, 0.5, 0.9),
            signal_mode="probability_gap",
        )
        self.assertTrue(gap_table["threshold_bps"].isna().all())
        self.assertTrue((gap_table["score_threshold"] <= 1.0).all())
        self.assertTrue(all(0.0 <= value <= 1.0 for value in gap_selected.values()))

    def test_overwrite_cleans_only_known_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "result"
            strategy = target / "strategies" / "expected_return_long_short"
            positions = strategy / "positions"
            positions.mkdir(parents=True)
            (target / "test_metrics.json").write_text("old", encoding="utf-8")
            (positions / "old.csv").write_text("old", encoding="utf-8")
            unknown = target / "notes.txt"
            unknown.write_text("keep", encoding="utf-8")
            _prepare_output_dir(target, overwrite=True)
            self.assertFalse((target / "test_metrics.json").exists())
            self.assertFalse((positions / "old.csv").exists())
            self.assertEqual(unknown.read_text(encoding="utf-8"), "keep")

    def test_small_end_to_end_run_freezes_before_test_and_strictly_replays(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            minute = root / "minute"
            output = root / "result"
            dates = ["20250102", "20250103", "20250106"]
            _write_synthetic_minute_data(minute, dates)
            result = run_return_lstm(
                stock_codes=["A"],
                seq_len=3,
                hidden_size=4,
                num_layers=1,
                dropout=0.0,
                epochs=1,
                patience=1,
                batch_size=64,
                learning_rate=1e-3,
                magnitude_lambdas=(0.1,),
                clip_quantile=0.95,
                threshold_quantiles=(0.0, 0.5),
                base_cost_bps=1.0,
                cost_grid_bps=(0.0, 1.0),
                splits={
                    "train": (dates[0], dates[0]),
                    "val": (dates[1], dates[1]),
                    "test": (dates[2], dates[2]),
                },
                data_dir=minute,
                out_dir=output,
                device="cpu",
                seed=7,
            )
            freeze = json.loads(
                (output / "selection_frozen_before_test.json").read_text(encoding="utf-8")
            )
            self.assertFalse(freeze["test_loaded_before_freeze"])
            self.assertFalse(freeze["test_metrics_used_for_selection"])
            self.assertGreaterEqual(
                freeze["selected_opening_threshold_bps"]["long_short"], 1.0
            )
            self.assertIn("selected_probability_gap_threshold", freeze)
            self.assertTrue(result["replay_audit"]["passed"])
            predictions = pd.read_csv(output / "test_predictions.csv")
            self.assertIn("predicted_abs_return_bps", predictions)
            self.assertIn("expected_return_bps", predictions)
            self.assertTrue((predictions["predicted_abs_return_bps"] >= 0.0).all())
            strategy = pd.read_csv(output / "strategy_comparison.csv")
            expected_rows = strategy["comparison_signal"].eq("expected_return")
            gap_rows = strategy["comparison_signal"].eq("probability_gap")
            self.assertTrue(
                strategy.loc[expected_rows, "validation_frozen_threshold_bps"]
                .ge(1.0)
                .all()
            )
            self.assertTrue(
                strategy.loc[gap_rows, "validation_frozen_threshold_bps"].isna().all()
            )
            self.assertEqual(
                set(strategy["validation_frozen_threshold_unit"]),
                {"bps", "probability_gap"},
            )
            model, config = load_return_model(output / "model.pt")
            self.assertIsInstance(model, ReturnMultiTaskLSTM)
            self.assertEqual(config["state_sha256"], freeze["state_sha256"])


if __name__ == "__main__":
    unittest.main()
