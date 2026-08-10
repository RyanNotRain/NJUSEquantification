from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.evaluation import factor_layering, factor_layering_daily_returns
from qd.factors import (
    REQUIRED_FACTOR_NAMES,
    compute_example_factor,
    select_required_factors,
)


class FactorEvaluationTests(unittest.TestCase):
    def test_example_factor_requires_complete_twenty_day_window(self) -> None:
        amount = pd.DataFrame(
            {"A": np.arange(1.0, 22.0)},
            index=pd.date_range("2025-01-01", periods=21),
        )
        factor = compute_example_factor(amount)
        self.assertTrue(factor.iloc[:19, 0].isna().all())
        self.assertTrue(np.isfinite(factor.iloc[19, 0]))

    def test_prompt_required_set_is_example_plus_three_original_factors(self) -> None:
        frame = pd.DataFrame([[1.0]])
        bundle = {name: frame for name in REQUIRED_FACTOR_NAMES}
        bundle["extension"] = frame
        selected = select_required_factors(bundle)
        self.assertEqual(tuple(selected), REQUIRED_FACTOR_NAMES)
        self.assertEqual(len(selected), 4)

    def test_layering_preserves_daily_returns_for_five_nav_curves(self) -> None:
        dates = pd.date_range("2025-01-01", periods=3)
        stocks = [f"S{i:02d}" for i in range(20)]
        factor = pd.DataFrame(
            np.tile(np.arange(20, dtype=float), (3, 1)),
            index=dates, columns=stocks,
        )
        # Each higher factor rank earns a lower return, so all five groups
        # must be strictly decreasing on every date.
        returns = pd.DataFrame(
            np.tile(np.linspace(0.02, -0.02, 20), (3, 1)),
            index=dates, columns=stocks,
        )
        daily = factor_layering_daily_returns(factor, returns, n_groups=5)
        self.assertEqual(daily.shape, (3, 5))
        self.assertEqual(list(daily.columns), ["Q1", "Q2", "Q3", "Q4", "Q5"])
        self.assertTrue((daily.diff(axis=1).iloc[:, 1:] < 0).all().all())

        summary = factor_layering(factor, returns, n_groups=5)
        np.testing.assert_allclose(
            summary["mean_return"].to_numpy(), daily.mean().to_numpy(),
        )


if __name__ == "__main__":
    unittest.main()
