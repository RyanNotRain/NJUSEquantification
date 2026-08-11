from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.tradable_return_research import (
    scheduled_signal_mask,
    strategy_metrics,
)


class TradableReturnResearchTests(unittest.TestCase):
    def test_schedule_resets_at_lunch_and_respects_interval(self) -> None:
        times = pd.Series(pd.to_datetime([
            "2026-01-02 10:30", "2026-01-02 10:35", "2026-01-02 10:36",
            "2026-01-02 14:00", "2026-01-02 14:05",
        ]))
        selected = scheduled_signal_mask(times, 5)
        np.testing.assert_array_equal(selected, [True, True, False, True, True])

    def test_daily_sleeves_average_inactive_cash_before_compounding(self) -> None:
        path = pd.DataFrame({
            "date": ["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-05"],
            "gross_return": [0.02, 0.0, -0.01, 0.0],
            "net_return": [0.0195, 0.0, -0.0105, 0.0],
            "matched_market_return": [0.01, 0.0, -0.02, 0.0],
            "active": [1, 0, 1, 0],
            "selected_names": [1, 0, 2, 0],
        })
        metrics = strategy_metrics(path, daily_sleeves=True)
        expected = (1 + 0.0195 / 2) * (1 - 0.0105 / 2) - 1
        self.assertAlmostEqual(metrics["net_total_return"], expected)
        self.assertEqual(metrics["periods"], 2)
        self.assertAlmostEqual(metrics["coverage"], 0.5)

    def test_sequential_strategy_uses_geometric_matched_excess(self) -> None:
        path = pd.DataFrame({
            "date": ["2026-01-02", "2026-01-02"],
            "gross_return": [0.02, -0.01],
            "net_return": [0.0195, -0.0105],
            "matched_market_return": [0.01, -0.02],
            "active": [1, 1],
            "selected_names": [1, 1],
        })
        metrics = strategy_metrics(path, daily_sleeves=False)
        strategy_growth = (1.0195 * 0.9895)
        market_growth = (1.01 * 0.98)
        self.assertAlmostEqual(
            metrics["excess_vs_matched_market"], strategy_growth / market_growth - 1
        )
        self.assertAlmostEqual(
            metrics["cumulative_return_gap_vs_matched_market"],
            strategy_growth - market_growth,
        )


if __name__ == "__main__":
    unittest.main()
