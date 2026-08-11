from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.strategy_analysis import (
    _build_lstm_t1_comparison,
    aggregate_signal_strategy,
    aggregate_t1_daily_strategy,
    performance_metrics,
    relative_metrics,
)


class StrategyAnalysisTests(unittest.TestCase):
    def test_performance_metrics_compound_returns(self) -> None:
        metrics = performance_metrics(pd.Series([0.10, -0.05]), periods_per_year=2)
        self.assertAlmostEqual(metrics["total_return"], 0.045)
        self.assertEqual(metrics["observations"], 2)

    def test_relative_metrics_uses_geometric_excess(self) -> None:
        strategy = pd.Series([0.10, 0.00])
        benchmark = pd.Series([0.05, 0.00])
        metrics = relative_metrics(strategy, benchmark)
        self.assertAlmostEqual(metrics["geometric_excess_return"], 1.10 / 1.05 - 1.0)
        self.assertAlmostEqual(metrics["cumulative_return_spread"], 1.10 - 1.05)

    def test_t1_comparison_contains_market_subtraction_path(self) -> None:
        returns = pd.DataFrame({
            "five_stock_market_return": [0.01, -0.02],
        })
        for strategy in ("all_up", "balanced_up", "strict_up"):
            prefix = f"{strategy}__fee_5bp"
            returns[f"{prefix}__net_return"] = [0.02, 0.00]
            returns[f"{prefix}__exposure_matched_market_return"] = [0.01, -0.01]
        comparison = _build_lstm_t1_comparison(returns, 5.0)
        prefix = "balanced_up__fee_5bp"
        expected_strategy_nav = 1.02
        expected_matched_nav = 1.01 * 0.99
        self.assertAlmostEqual(
            comparison.iloc[-1][f"{prefix}__return_gap_vs_matched_market"],
            expected_strategy_nav - expected_matched_nav,
        )

    def test_signal_strategy_charges_fee_only_when_active(self) -> None:
        samples = pd.DataFrame({
            "target_time": pd.to_datetime([
                "2026-01-01 09:31", "2026-01-01 09:31", "2026-01-01 09:32"
            ]),
            "predicted_label": [2, 2, 1],
            "return": [0.01, -0.01, 0.02],
        })
        selected = samples["predicted_label"].eq(2)
        path = aggregate_signal_strategy(
            samples, selected, "return", direction="long", sell_fee_bps=5.0
        )
        self.assertEqual(path["active"].tolist(), [1, 0])
        self.assertAlmostEqual(path.iloc[0]["gross_return"], 0.0)
        self.assertAlmostEqual(path.iloc[0]["net_return"], -0.0005)
        self.assertAlmostEqual(path.iloc[1]["net_return"], 0.0)

    def test_long_short_uses_prediction_direction(self) -> None:
        samples = pd.DataFrame({
            "target_time": pd.to_datetime(["2026-01-01 09:31", "2026-01-01 09:31"]),
            "predicted_label": [0, 2],
            "return": [-0.01, 0.02],
        })
        path = aggregate_signal_strategy(
            samples, pd.Series([True, True]), "return", direction="long_short"
        )
        self.assertAlmostEqual(path.iloc[0]["gross_return"], 0.015)
        self.assertTrue(np.isfinite(path.to_numpy()).all())

    def test_t1_strategy_uses_cash_sleeves_and_matched_market(self) -> None:
        samples = pd.DataFrame({
            "date": ["2026-01-01"] * 4,
            "target_time": pd.to_datetime([
                "2026-01-01 09:31", "2026-01-01 09:31",
                "2026-01-01 09:32", "2026-01-01 09:32",
            ]),
            "next_day_same_minute_open_return": [0.02, 0.00, -0.01, 0.01],
        })
        selected = pd.Series([True, False, False, False])
        path = aggregate_t1_daily_strategy(samples, selected, sell_fee_bps=5.0)
        self.assertAlmostEqual(path.iloc[0]["gross_return"], 0.01)
        self.assertAlmostEqual(path.iloc[0]["sell_fee"], 0.00025)
        self.assertAlmostEqual(path.iloc[0]["active"], 0.5)
        self.assertAlmostEqual(path.iloc[0]["five_stock_market_return"], 0.005)
        self.assertAlmostEqual(path.iloc[0]["exposure_matched_market_return"], 0.005)


if __name__ == "__main__":
    unittest.main()
