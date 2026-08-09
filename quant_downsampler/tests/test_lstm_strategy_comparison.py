from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from qd.lstm_strategy_comparison import (
    _prepare_output_directory,
    align_prediction_sources,
    build_equal_weight_buy_and_hold,
    build_equal_weight_market_proxy,
    generate_signal_weights,
    run_strategy_comparison,
)


def _predictions(prefix: str = "") -> pd.DataFrame:
    return pd.DataFrame({
        "stock": ["A", "B", "A", "B"],
        "window_end": [
            "2025-01-02 10:00:00",
            "2025-01-02 10:00:00",
            "2025-01-02 10:01:00",
            "2025-01-02 10:01:00",
        ],
        "target_time": [
            "2025-01-02 10:01:00",
            "2025-01-02 10:01:00",
            "2025-01-02 10:02:00",
            "2025-01-02 10:02:00",
        ],
        f"{prefix}prob_down": [0.10, 0.60, 0.20, 0.30],
        f"{prefix}prob_flat": [0.20, 0.25, 0.60, 0.40],
        f"{prefix}prob_up": [0.70, 0.15, 0.20, 0.30],
        "true_label": [2, 0, 1, 1],
    })


class LSTMStrategyComparisonTests(unittest.TestCase):
    def test_overwrite_removes_only_owned_comparison_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "comparison"
            paths = target / "portfolio_paths"
            paths.mkdir(parents=True)
            (target / "metrics.json").write_text(
                json.dumps({"sources": {"old_model": {}}}), encoding="utf-8"
            )
            (target / "strategy_comparison.csv").write_text("old", encoding="utf-8")
            (paths / "old_model.csv").write_text("old", encoding="utf-8")
            unknown = paths / "notes.csv"
            unknown.write_text("keep", encoding="utf-8")
            _prepare_output_directory(target, True, ("new_model",))
            self.assertFalse((target / "metrics.json").exists())
            self.assertFalse((paths / "old_model.csv").exists())
            self.assertEqual(unknown.read_text(encoding="utf-8"), "keep")

    def test_alignment_is_exact_and_accepts_histgb_column_mapping(self) -> None:
        first = _predictions()
        second = _predictions("hist_tree_")
        aligned, audit = align_prediction_sources(
            {"lstm": first, "tree": second},
            probability_columns={
                "tree": {
                    "prob_down": "hist_tree_prob_down",
                    "prob_flat": "hist_tree_prob_flat",
                    "prob_up": "hist_tree_prob_up",
                }
            },
            expected_stock_count=2,
            expected_day_count=1,
        )
        self.assertEqual(audit["n_rows"], 4)
        self.assertTrue(audit["all_sample_keys_exactly_equal"])
        np.testing.assert_allclose(aligned["lstm"]["prob_up"], aligned["tree"]["prob_up"])

    def test_alignment_refuses_to_intersect_missing_keys(self) -> None:
        first = _predictions()
        second = _predictions().iloc[:-1]
        with self.assertRaisesRegex(ValueError, "sample keys do not exactly match"):
            align_prediction_sources(
                {"first": first, "second": second},
                expected_stock_count=None,
                expected_day_count=None,
            )

    def test_probability_weights_do_not_use_realised_returns(self) -> None:
        first = _predictions()
        first["realised_return"] = [0.01, -0.01, 0.0, 0.0]
        second = first.copy()
        second["realised_return"] = [-0.80, 0.70, 0.20, -0.30]
        left = generate_signal_weights(first)
        right = generate_signal_weights(second)
        np.testing.assert_allclose(left["target_weight"], right["target_weight"])
        np.testing.assert_array_equal(left["active_signal"], right["active_signal"])

    def test_expected_return_interface_uses_basis_point_signal(self) -> None:
        frame = _predictions().drop(columns=["prob_down", "prob_flat", "prob_up"])
        frame["expected_return_bps"] = [8.0, -4.0, 1.0, -12.0]
        weighted = generate_signal_weights(
            frame,
            signal_mode="expected_return_bps",
            score_threshold=5.0,
        )
        first = weighted[weighted["window_end"].eq("2025-01-02 10:00:00")]
        self.assertEqual(int(first["active_signal"].sum()), 1)
        self.assertAlmostEqual(float(first.loc[first["stock"].eq("A"), "target_weight"].iloc[0]), 1.0)
        second = weighted[weighted["window_end"].eq("2025-01-02 10:01:00")]
        self.assertEqual(int(second["active_signal"].sum()), 1)
        self.assertAlmostEqual(float(second.loc[second["stock"].eq("B"), "target_weight"].iloc[0]), -1.0)

    def test_benchmarks_use_same_cross_sectional_returns(self) -> None:
        realised = _predictions()
        realised["realised_return"] = [0.10, -0.10, 0.20, 0.00]
        market = build_equal_weight_market_proxy(realised)
        buy_hold = build_equal_weight_buy_and_hold(realised)
        self.assertAlmostEqual(float(market.iloc[0]["gross_return"]), 0.0)
        self.assertAlmostEqual(float(market.iloc[1]["gross_return"]), 0.10)
        self.assertAlmostEqual(float(buy_hold.iloc[0]["gross_return"]), 0.0)
        # First-period +10%/-10% drifts weights to 55%/45%; no rebalance is used.
        self.assertAlmostEqual(float(buy_hold.iloc[1]["gross_return"]), 0.11)
        self.assertAlmostEqual(float(buy_hold["turnover"].sum()), 2.0)

    def test_end_to_end_outputs_and_relative_return_identity(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            close_dir = root / "close"
            close_dir.mkdir()
            prices = pd.DataFrame(
                {
                    "A": [100.0, 101.0, 102.0],
                    "B": [100.0, 99.0, 100.0],
                },
                index=pd.to_datetime([
                    "2025-01-02 10:00:00",
                    "2025-01-02 10:01:00",
                    "2025-01-02 10:02:00",
                ]),
            )
            prices.index.name = "datetime"
            prices.to_csv(close_dir / "20250102.csv")
            lstm_path = root / "lstm.csv"
            tree_path = root / "tree.csv"
            _predictions().to_csv(lstm_path, index=False)
            _predictions("tree_").to_csv(tree_path, index=False)
            out_dir = root / "out"
            result = run_strategy_comparison(
                {"lstm": lstm_path, "tree": tree_path},
                close_dir,
                out_dir,
                probability_columns={
                    "lstm": {name: name for name in ("prob_down", "prob_flat", "prob_up")},
                    "tree": {
                        "prob_down": "tree_prob_down",
                        "prob_flat": "tree_prob_flat",
                        "prob_up": "tree_prob_up",
                    },
                },
                expected_stock_count=2,
                expected_day_count=1,
            )
            summary = result["summary"].set_index("name")
            self.assertEqual(set(summary["kind"]), {"model", "benchmark"})
            self.assertAlmostEqual(
                float(summary.at["equal_weight_market_proxy", "gross_relative_to_market_proxy"]),
                0.0,
            )
            self.assertTrue((out_dir / "strategy_comparison.png").is_file())
            self.assertTrue((out_dir / "aligned_sample_returns.csv").is_file())
            report = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(
                report["methodology"]["benchmark_used_for_model_or_threshold_selection"]
            )


if __name__ == "__main__":
    unittest.main()
