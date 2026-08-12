from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.factor_robustness import (
    block_bootstrap_mean_ci,
    build_market_regimes,
    compute_ic_table,
    ic_by_regime,
    mask_nontradable_forward_returns,
)


class FactorRobustnessTests(unittest.TestCase):
    def test_cross_sectional_ic_and_regime_split(self) -> None:
        dates = pd.date_range("2026-01-01", periods=4)
        columns = [f"s{i}" for i in range(30)]
        values = np.tile(np.arange(30, dtype=float), (4, 1))
        factor = pd.DataFrame(values, index=dates, columns=columns)
        forward = factor.copy()
        ic = compute_ic_table({"factor": factor}, forward, min_stocks=30)
        self.assertTrue(np.allclose(ic["factor"], 1.0))
        result = ic_by_regime(ic, pd.Series(["up", "down", "up", "down"], index=dates), "direction")
        self.assertEqual(set(result["regime"]), {"up", "down"})

    def test_block_bootstrap_detects_strong_nonzero_mean(self) -> None:
        dates = pd.date_range("2026-01-01", periods=80)
        ic = pd.DataFrame({"factor": np.linspace(0.05, 0.10, len(dates))}, index=dates)
        result = block_bootstrap_mean_ci(ic, iterations=300, seed=7)
        self.assertGreater(result.loc["factor", "ci_lower"], 0.0)
        self.assertTrue(bool(result.loc["factor", "significant_at_5pct"]))

    def test_market_regime_trend_uses_current_and_past_market_returns(self) -> None:
        dates = pd.date_range("2026-01-01", periods=5)
        close = pd.DataFrame({"a": [100, 101, 102, 103, 104], "b": [100, 102, 104, 106, 108]}, index=dates)
        regimes = build_market_regimes(close, trend_window=2)
        self.assertEqual(regimes.loc[dates[-1], "known_market_trend"], "bull")
        self.assertTrue(pd.isna(regimes.loc[dates[-1], "target_market_return"]))

    def test_robustness_masks_signal_or_realization_day_halts(self) -> None:
        dates = pd.date_range("2026-01-01", periods=3)
        forward = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.4, 0.5, 0.6]}, index=dates)
        volume = pd.DataFrame({"a": [1.0, 0.0, 1.0], "b": [1.0, 1.0, 1.0]}, index=dates)
        masked = mask_nontradable_forward_returns(forward, volume)
        self.assertTrue(pd.isna(masked.loc[dates[0], "a"]))
        self.assertTrue(pd.isna(masked.loc[dates[1], "a"]))
        self.assertEqual(masked.loc[dates[0], "b"], 0.4)


if __name__ == "__main__":
    unittest.main()
