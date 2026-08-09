from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from qd.lstm_return_baselines import (
    MODEL_NAMES,
    evaluate_saved_return_baselines,
    make_return_estimator,
    return_prediction_metrics,
    run_return_baselines,
    select_regressor_on_validation,
    select_validation_opening_thresholds,
)


def _metadata(n: int, start: str) -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq="min")
    return pd.DataFrame({
        "stock": ["A" if index % 2 == 0 else "B" for index in range(n)],
        "stock_id": [index % 2 for index in range(n)],
        "date": times.strftime("%Y-%m-%d"),
        "window_end": times,
        "target_time": times + pd.Timedelta(minutes=1),
    })


def _split(seed: int, n: int, start: str) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, 4)).astype(np.float32)
    realised = (2.0 * matrix[:, 0] - matrix[:, 1] + rng.normal(scale=0.1, size=n))
    return {
        "matrix": matrix,
        "labels": np.where(realised < 0, 0, 2).astype(np.int64),
        "stock_ids": np.arange(n, dtype=np.int64) % 2,
        "coverage": {"observed_samples": n},
        "metadata": _metadata(n, start),
        "signed_return_bps": realised,
        "absolute_return_bps": np.abs(realised),
        "feature_names": ["f0", "f1", "f2", "f3"],
    }


class LSTMReturnBaselineTests(unittest.TestCase):
    def test_estimators_fit_and_predict_finite_values(self) -> None:
        split = _split(1, 40, "2025-01-02 10:00")
        for name, params in (
            ("ridge", {"alpha": 1.0}),
            (
                "hist_gradient_boosting_regressor",
                {"max_iter": 5, "min_samples_leaf": 2},
            ),
        ):
            model = make_return_estimator(name, params)
            model.fit(split["matrix"], split["signed_return_bps"])
            prediction = model.predict(split["matrix"])
            self.assertTrue(np.isfinite(prediction).all())

    def test_return_metrics_include_groups_and_direction(self) -> None:
        realised = np.array([-2.0, -1.0, 1.0, 2.0])
        predicted = np.array([-1.5, -0.5, 0.5, 1.5])
        metrics = return_prediction_metrics(realised, predicted, n_groups=2)
        self.assertAlmostEqual(metrics["mae_bps"], 0.5)
        self.assertAlmostEqual(metrics["direction_hit_rate_both_nonzero"], 1.0)
        self.assertGreater(metrics["grouped_returns"]["top_minus_bottom_bps"], 0.0)
        self.assertTrue(metrics["outperforms_zero_prediction_mae"])
        self.assertTrue(metrics["outperforms_zero_prediction_rmse"])

    def test_validation_selection_uses_expected_objective(self) -> None:
        train = _split(2, 80, "2025-01-02 10:00")
        validation = _split(3, 40, "2025-01-03 10:00")
        model, selection, prediction = select_regressor_on_validation(
            "ridge",
            ({"alpha": 0.01}, {"alpha": 1000.0}),
            train["matrix"],
            train["signed_return_bps"],
            validation["matrix"],
            validation["signed_return_bps"],
        )
        self.assertEqual(selection["selected_candidate_id"], 0)
        self.assertEqual(len(prediction), 40)
        self.assertTrue(hasattr(model, "predict"))

    def test_validation_thresholds_never_fall_below_base_cost(self) -> None:
        frame = _metadata(6, "2025-01-02 10:00")
        frame["realised_return"] = np.array([0.001, -0.001] * 3)
        frame["model_expected_return_bps"] = [1.0, -2.0, 3.0, -4.0, 8.0, -9.0]
        chosen, table = select_validation_opening_thresholds(
            frame,
            {"model": "model_expected_return_bps"},
            base_cost_bps=5.0,
            quantiles=(0.0, 0.5, 0.9),
        )
        self.assertTrue((table["threshold_bps"] >= 5.0).all())
        self.assertGreaterEqual(chosen["model"]["long_short"], 5.0)
        self.assertGreaterEqual(chosen["model"]["long_only"], 5.0)

    def test_freeze_precedes_test_and_saved_bundle_replays(self) -> None:
        train = _split(4, 80, "2025-01-02 10:00")
        validation = _split(5, 40, "2025-01-03 10:00")
        test = _split(6, 40, "2025-01-04 10:00")
        calls: list[tuple[str, str]] = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            out.mkdir()
            user_file = out / "user_notes.txt"
            user_file.write_text("preserve me\n", encoding="utf-8")
            (out / "test_metrics.json").write_text("stale\n", encoding="utf-8")

            def loader(
                stocks: list[str],
                date_range: tuple[str, str],
                seq_len: int,
                minute_dir: Path,
            ) -> dict[str, object]:
                del stocks, seq_len, minute_dir
                if date_range == ("20250101", "20250102"):
                    calls.append(("train", "loaded"))
                    return train
                if date_range == ("20250103", "20250103"):
                    calls.append(("val", "loaded"))
                    return validation
                self.assertTrue((out / "selection_frozen_before_test.json").is_file())
                calls.append(("test", "loaded_after_freeze"))
                return test

            with patch(
                "qd.lstm_return_baselines._load_regression_split", side_effect=loader
            ):
                result = run_return_baselines(
                    stock_codes=["A", "B"],
                    splits={
                        "train": ("20250101", "20250102"),
                        "val": ("20250103", "20250103"),
                        "test": ("20250104", "20250104"),
                    },
                    data_dir=root,
                    out_dir=out,
                    ridge_grid=({"alpha": 1.0},),
                    tree_grid=({"max_iter": 5, "min_samples_leaf": 2},),
                    threshold_quantiles=(0.0, 0.5),
                    overwrite=True,
                )
            self.assertEqual(calls.count(("test", "loaded_after_freeze")), 1)
            self.assertEqual(user_file.read_text(encoding="utf-8"), "preserve me\n")
            frozen = json.loads(
                (out / "selection_frozen_before_test.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frozen["test_evaluation_count"], 1)
            self.assertEqual(set(result["test_metrics"]), set(MODEL_NAMES))
            bundle = joblib.load(out / "models.joblib")
            self.assertEqual(set(bundle["models"]), set(MODEL_NAMES))
            replay_audit = json.loads(
                (out / "replay_audit.json").read_text(encoding="utf-8")
            )
            self.assertTrue(replay_audit["passed"])
            self.assertEqual(replay_audit["raw_test_data_load_count"], 1)
            for name in MODEL_NAMES:
                canonical = pd.read_csv(out / f"{name}_test_predictions.csv")
                self.assertIn("expected_return_bps", canonical)

            # Replay is an explicit later action, separate from the one primary
            # test evaluation asserted above.
            with patch(
                "qd.lstm_return_baselines._load_regression_split", return_value=test
            ):
                replay = evaluate_saved_return_baselines(
                    out / "models.joblib", root, split="test"
                )
            self.assertEqual(len(replay["predictions"]), len(test["matrix"]))
            for name in MODEL_NAMES:
                original = pd.read_csv(out / "test_predictions.csv")[
                    f"{name}_expected_return_bps"
                ].to_numpy()
                repeated = replay["predictions"][
                    f"{name}_expected_return_bps"
                ].to_numpy()
                np.testing.assert_allclose(original, repeated, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
