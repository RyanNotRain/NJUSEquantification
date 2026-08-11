from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.task4_excess_significance import (
    bootstrap_relative_performance,
    moving_block_indices,
    rolling_relative_metrics,
)


class Task4ExcessSignificanceTests(unittest.TestCase):
    def test_moving_blocks_are_deterministic_and_preserve_length(self) -> None:
        first = moving_block_indices(11, 4, 20, seed=7)
        second = moving_block_indices(11, 4, 20, seed=7)
        self.assertEqual(first.shape, (20, 11))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(((first >= 0) & (first < 11)).all())

    def test_clear_outperformance_has_positive_bootstrap_interval(self) -> None:
        rng = np.random.default_rng(3)
        benchmark = pd.Series(rng.normal(0.0, 0.001, 180))
        strategy = benchmark + 0.002
        result = bootstrap_relative_performance(
            strategy, benchmark, block_length=5, iterations=500, seed=4
        )
        self.assertGreater(result["geometric_excess_ci_low"], 0.0)
        self.assertLess(result["one_sided_pvalue_mean_daily_excess_gt_zero"], 0.01)

    def test_rolling_metrics_use_compounded_relative_return(self) -> None:
        strategy = pd.Series([0.02, 0.01, -0.01], index=pd.date_range("2026-01-01", periods=3))
        benchmark = pd.Series([0.01, 0.00, -0.01], index=strategy.index)
        rolling = rolling_relative_metrics(strategy, benchmark, window=3)
        expected = np.prod(1.0 + strategy) / np.prod(1.0 + benchmark) - 1.0
        self.assertAlmostEqual(rolling.iloc[0]["geometric_excess_return"], expected)


if __name__ == "__main__":
    unittest.main()

