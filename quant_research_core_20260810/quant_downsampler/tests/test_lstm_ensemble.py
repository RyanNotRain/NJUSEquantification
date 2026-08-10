from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from qd.evaluation import compute_ic_stats
from qd.lstm_components import (
    CHANNELS,
    MinuteLSTM,
    create_sequences,
    create_sequences_with_metadata,
    engineer_features,
    feature_names,
)
from qd.lstm_ensemble import (
    blend_probabilities,
    load_full_model,
    two_stage_probabilities,
)
from qd.task4_strict import historical_ic_weights


class LSTMEnsembleTests(unittest.TestCase):
    def _flat_day(self) -> np.ndarray:
        day = np.ones((242, len(CHANNELS)), dtype=np.float64)
        day[:, :4] = 10.0
        return day

    def test_enhanced_features_are_causal_and_reset_at_lunch(self) -> None:
        day = self._flat_day()
        day[121:, :4] = 11.0
        features = engineer_features(day, feature_set="enhanced")
        self.assertEqual(features.shape, (242, len(feature_names("enhanced"))))
        self.assertEqual(float(features[121, 3]), 0.0)

    def test_three_class_keeps_flat_targets_and_metadata(self) -> None:
        day = self._flat_day()
        x, labels, metadata = create_sequences_with_metadata(
            [day], ["20250102"], seq_len=60,
            feature_set="legacy", target_mode="three_class",
        )
        self.assertEqual(x.shape[0], 118)
        self.assertTrue(np.all(labels == 1))
        self.assertEqual(metadata.iloc[0]["window_end"], "2025-01-02 10:29:00")
        self.assertEqual(metadata.iloc[0]["target_time"], "2025-01-02 10:30:00")
        self.assertEqual(metadata.iloc[61]["window_end"], "2025-01-02 13:59:00")

    def test_nonflat_direction_excludes_flat_targets(self) -> None:
        day = self._flat_day()
        x, labels = create_sequences(
            [day], seq_len=5, feature_set="legacy",
            target_mode="nonflat_binary",
        )
        self.assertEqual(len(x), 0)
        self.assertEqual(len(labels), 0)

    def test_probability_fusion_is_normalized(self) -> None:
        staged = two_stage_probabilities(
            np.array([0.25, 0.75]), np.array([0.4, 0.6]), move_bias=0.0,
        )
        blended = blend_probabilities(staged, staged, joint_weight=0.5)
        self.assertTrue(np.allclose(blended, staged))
        self.assertTrue(np.allclose(blended.sum(axis=1), 1.0))

    def test_ir_is_annualized_icir(self) -> None:
        stats = compute_ic_stats(pd.Series([0.01, 0.03, -0.01, 0.02]))
        self.assertAlmostEqual(stats["IR"], stats["ICIR"] * np.sqrt(252.0))

    def test_task4_rolling_ic_has_two_day_information_lag(self) -> None:
        index = pd.date_range("2025-01-01", periods=100, freq="D")
        daily_ic = pd.DataFrame({"factor": np.arange(100.0)}, index=index)
        weights = historical_ic_weights(daily_ic, lookback=60, method="rolling")
        # At row 70, the latest observable raw IC is row 68.  A 60-row
        # lookback therefore averages raw rows 9 through 68.
        self.assertAlmostEqual(weights.iloc[70, 0], np.mean(np.arange(9.0, 69.0)))

    def test_three_component_checkpoint_reloads_strictly(self) -> None:
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
            "include_stock_id": False, "input_size": 2,
            "scaler_mode": "global", "scaler_mean": [0.0, 0.0],
            "scaler_std": [1.0, 1.0], "hidden_size": 4,
            "num_layers": 1, "dropout": 0.0, "model_version": "legacy",
            "num_classes": 2, "class_names": ["down", "up"],
            "seq_len": 3, "target_mode": "nonflat_binary",
        }
        enhanced_config = {
            "feature_names": ["a", "b", "stock_id::S"],
            "feature_set": "enhanced", "include_stock_id": True,
            "input_size": 3, "scaler_mode": "per_stock",
            "scaler_mean": [[0.0, 0.0]], "scaler_std": [[1.0, 1.0]],
            "hidden_size": 4, "num_layers": 1, "dropout": 0.0,
            "model_version": "residual", "num_classes": 2,
            "class_names": ["flat", "move"], "seq_len": 3,
            "target_mode": "move_vs_flat",
        }
        bundle = {
            "components": {
                "direction": {"state_dict": direction.state_dict(), "config": direction_config},
                "movement": {"state_dict": movement.state_dict(), "config": enhanced_config},
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
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save(bundle, path)
            loaded = load_full_model(path)
            probability = loaded.predict_from_features(
                np.zeros((4, 3, 2), dtype=np.float32),
                np.zeros((4, 3, 2), dtype=np.float32),
                np.zeros(4, dtype=np.int64), batch_size=2,
            )
        self.assertEqual(probability.shape, (4, 3))
        self.assertTrue(np.allclose(probability.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
