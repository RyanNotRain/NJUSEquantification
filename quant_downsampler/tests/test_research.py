from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from qd.backtest import corrected_weights
from qd.evaluation import evaluate_all_factors
from qd.factors import FACTOR_NAMES
from qd.lstm_model import (
    CHANNELS,
    MinuteLSTM,
    create_sequences,
    create_sequences_with_metadata,
    engineer_features,
    feature_names,
    load_minute_data_for_stocks,
    prepare_dataloaders,
    target_coverage,
)
from qd.lstm_full import (
    blend_probabilities,
    load_full_model,
    two_stage_probabilities,
)


class ResearchPipelineTests(unittest.TestCase):
    def test_corrected_ic_weights_have_two_day_information_lag(self) -> None:
        index = pd.date_range("2025-01-01", periods=15, freq="D")
        daily_ic = pd.DataFrame({"example_factor": np.arange(15.0)}, index=index)
        weights = corrected_weights(daily_ic, lookback=3)
        # t uses the rolling mean ending at t-2, never IC[t-1] or IC[t].
        self.assertEqual(weights.iloc[12, 0], np.mean([8.0, 9.0, 10.0]))

    def test_factor_summary_distinguishes_ir_and_icir(self) -> None:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2025-01-01", periods=8)
        stocks = [f"S{i:03d}" for i in range(40)]
        forward = pd.DataFrame(rng.normal(size=(8, 40)), index=dates, columns=stocks)
        factors = {
            name: forward + rng.normal(scale=0.1, size=(8, 40))
            for name in FACTOR_NAMES
        }
        summary, _, _ = evaluate_all_factors(factors, forward)
        self.assertIn("IR", summary.columns)
        self.assertIn("ICIR", summary.columns)
        self.assertTrue(np.allclose(
            summary["IR"], summary["ICIR"] * np.sqrt(252), equal_nan=True
        ))

    def test_lstm_label_is_immediate_next_minute(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        # First valid window with seq_len=2 ends at index 1.  The immediate
        # next move (index 1 -> 2) is down; index 2 -> 3 is up.
        day[0, :4] = 10.0
        day[1, :4] = 11.0
        day[2, :4] = 10.0
        day[3, :4] = 12.0
        X, y = create_sequences([day], seq_len=2)
        self.assertGreater(len(X), 0)
        self.assertEqual(int(y[0]), 0)

    def test_lstm_drops_flat_targets(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        X, y = create_sequences([day], seq_len=5)
        self.assertEqual(len(X), 0)
        self.assertEqual(len(y), 0)

    def test_lstm_three_class_keeps_flat_targets(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        X, y = create_sequences(
            [day], seq_len=5, feature_set="legacy", target_mode="three_class"
        )
        self.assertGreater(len(X), 0)
        self.assertTrue(np.all(y == 1))

    def test_lstm_metadata_identifies_window_and_target_minutes(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        _, labels, metadata = create_sequences_with_metadata(
            [day], ["20250102"], seq_len=60,
            feature_set="legacy", target_mode="three_class",
        )
        self.assertEqual(len(labels), 118)
        self.assertEqual(metadata.iloc[0]["window_end"], "2025-01-02 10:29:00")
        self.assertEqual(metadata.iloc[0]["target_time"], "2025-01-02 10:30:00")
        self.assertEqual(metadata.iloc[61]["window_end"], "2025-01-02 13:59:00")

    def test_lstm_reports_conditional_target_coverage(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        coverage = target_coverage([day], seq_len=5, feature_set="legacy")
        self.assertGreater(coverage["all_valid_windows"], 0)
        self.assertEqual(coverage["nonflat_windows"], 0)
        self.assertEqual(coverage["flat_rate"], 1.0)

    def test_enhanced_features_match_model_input_and_reload(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        features = engineer_features(day, feature_set="enhanced")
        self.assertEqual(features.shape[1], len(feature_names("enhanced")))
        model = MinuteLSTM(
            input_size=features.shape[1], hidden_size=16, model_version="residual"
        )
        batch = torch.from_numpy(features[None, :60].astype(np.float32))
        logits = model(batch)
        self.assertEqual(tuple(logits.shape), (1, 2))
        reloaded = MinuteLSTM(
            input_size=features.shape[1], hidden_size=16, model_version="residual"
        )
        reloaded.load_state_dict(model.state_dict(), strict=True)
        model.eval()
        reloaded.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(batch), reloaded(batch)))

    def test_enhanced_return_does_not_cross_lunch(self) -> None:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        day[121:, :4] = 11.0
        legacy = engineer_features(day, feature_set="legacy")
        enhanced = engineer_features(day, feature_set="enhanced")
        self.assertNotEqual(float(legacy[121, 3]), 0.0)
        self.assertEqual(float(enhanced[121, 3]), 0.0)

    def test_full_lstm_probabilities_are_normalized_and_reloadable(self) -> None:
        direction = MinuteLSTM(
            input_size=2, hidden_size=4, num_layers=1, dropout=0.0,
            model_version="legacy", num_classes=2,
        )
        movement = MinuteLSTM(
            input_size=3, hidden_size=4, num_layers=1, dropout=0.0,
            model_version="residual", num_classes=2,
        )
        joint = MinuteLSTM(
            input_size=3, hidden_size=4, num_layers=1, dropout=0.0,
            model_version="residual", num_classes=3,
        )
        direction_config = {
            "feature_names": ["a", "b"], "feature_set": "legacy",
            "include_stock_id": False,
            "scaler_mode": "global", "scaler_mean": [0.0, 0.0],
            "scaler_std": [1.0, 1.0], "hidden_size": 4, "num_layers": 1,
            "dropout": 0.0, "model_version": "legacy", "num_classes": 2,
            "class_names": ["down", "up"], "seq_len": 3,
            "target_mode": "nonflat_binary",
        }
        enhanced_config = {
            "feature_names": ["a", "b", "stock_id::S"], "feature_set": "enhanced",
            "include_stock_id": True, "input_size": 3,
            "scaler_mode": "per_stock", "scaler_mean": [[0.0, 0.0]],
            "scaler_std": [[1.0, 1.0]], "hidden_size": 4, "num_layers": 1,
            "dropout": 0.0, "model_version": "residual", "num_classes": 2,
            "class_names": ["flat", "move"], "seq_len": 3,
            "target_mode": "move_vs_flat",
        }
        bundle = {
            "components": {
                "direction": {
                    "state_dict": direction.state_dict(), "config": direction_config,
                },
                "movement": {
                    "state_dict": movement.state_dict(), "config": enhanced_config,
                },
                "joint": {
                    "state_dict": joint.state_dict(),
                    "config": {
                        **enhanced_config, "num_classes": 3,
                        "class_names": ["down", "flat", "up"],
                        "target_mode": "three_class",
                    },
                },
            },
            "config": {
                "stock_codes": ["S"], "class_names": ["down", "flat", "up"],
                "move_bias": -0.05, "joint_weight": 0.5,
            },
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            torch.save(bundle, path)
            ensemble = load_full_model(path)
            probability = ensemble.predict_from_features(
                np.zeros((4, 3, 2), dtype=np.float32),
                np.zeros((4, 3, 2), dtype=np.float32),
                np.zeros(4, dtype=np.int64),
                batch_size=2,
            )
            with self.assertRaisesRegex(ValueError, "sequence length"):
                ensemble.predict_from_features(
                    np.zeros((1, 2, 2), dtype=np.float32),
                    np.zeros((1, 2, 2), dtype=np.float32),
                    np.zeros(1, dtype=np.int64),
                )
        self.assertEqual(probability.shape, (4, 3))
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))

        staged = two_stage_probabilities(
            np.array([0.25, 0.75]), np.array([0.4, 0.6]), move_bias=0.0
        )
        blended = blend_probabilities(staged, staged, joint_weight=0.5)
        self.assertTrue(np.allclose(blended, staged))

    def test_lstm_rejects_overlapping_splits_before_loading(self) -> None:
        bad_splits = {
            "train": ("20250101", "20250110"),
            "val": ("20250110", "20250120"),
            "test": ("20250121", "20250131"),
        }
        with TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare_dataloaders(
                    ["000001.SZ"], data_dir=Path(tmp), splits=bad_splits
                )

    def test_minute_loader_rejects_channel_index_misalignment(self) -> None:
        stock = "000001.SZ"
        index = pd.date_range("2025-01-02 09:30", periods=242, freq="min")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for channel in CHANNELS:
                (root / channel).mkdir()
                frame = pd.DataFrame({stock: np.ones(242)}, index=index)
                frame.index.name = "datetime"
                if channel == "high":
                    frame = frame.iloc[::-1]
                frame.to_csv(root / channel / "20250102.csv")
            with self.assertRaisesRegex(ValueError, "misaligned"):
                load_minute_data_for_stocks(
                    [stock], ("20250102", "20250102"), root
                )


if __name__ == "__main__":
    unittest.main()
