from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.backtest_robustness import (
    _buffered_target,
    _equal_weight_turnover,
    _performance_from_returns,
    apply_transaction_costs,
    estimate_break_even_cost_bps,
    symmetric_cost_stress,
)


class BacktestRobustnessTests(unittest.TestCase):
    def test_turnover_accounts_for_resizing_retained_equal_weights(self) -> None:
        sell, buy = _equal_weight_turnover({"A", "B"}, {"A", "B", "C", "D"})
        self.assertAlmostEqual(sell, 0.5)
        self.assertAlmostEqual(buy, 0.5)
        initial_sell, initial_buy = _equal_weight_turnover(set(), {"A", "B"})
        self.assertEqual(initial_sell, 0.0)
        self.assertEqual(initial_buy, 1.0)

    def test_performance_uses_standard_sharpe_and_initial_capital_drawdown(self) -> None:
        returns = pd.Series([-0.10, 0.05, 0.02])
        _, metrics = _performance_from_returns(returns, initial_capital=100.0)
        expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252.0)
        self.assertAlmostEqual(metrics["sharpe_ratio"], expected_sharpe)
        self.assertAlmostEqual(metrics["max_drawdown"], -0.10)

    def test_buffer_and_replacement_cap_retain_incumbents(self) -> None:
        signal = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0})
        target, _ = _buffered_target(
            previous={"C", "D", "E"},
            locked=set(),
            full_signal=signal,
            eligible_signal=signal,
            top_n=3,
            buffer_n=4,
            max_replacements=1,
        )
        self.assertEqual(len({"C", "D", "E"} - target), 1)
        self.assertEqual(len(target), 3)

    def test_cost_repricing_is_monotonic_and_break_even_is_finite(self) -> None:
        index = pd.date_range("2025-01-01", periods=4)
        periods = pd.DataFrame({
            "gross_return": [0.01, 0.01, 0.01, 0.01],
            "sell_turnover": [0.0, 0.5, 0.5, 0.5],
            "buy_turnover": [1.0, 0.5, 0.5, 0.5],
            "total_turnover": [1.0, 1.0, 1.0, 1.0],
        }, index=index)
        result = {"periods": periods}
        stress = symmetric_cost_stress(result, (0.0, 10.0, 20.0))
        self.assertTrue(np.all(np.diff(stress["total_return"]) < 0))
        break_even = estimate_break_even_cost_bps(result)
        self.assertGreater(break_even, 0.0)
        priced = apply_transaction_costs(
            result, sell_cost=break_even / 10_000.0, buy_cost=break_even / 10_000.0
        )
        self.assertAlmostEqual(priced["metrics"]["total_return"], 0.0, places=10)

    def test_negative_cost_is_rejected(self) -> None:
        periods = pd.DataFrame({
            "gross_return": [0.0], "sell_turnover": [0.0],
            "buy_turnover": [0.0], "total_turnover": [0.0],
        }, index=[pd.Timestamp("2025-01-01")])
        with self.assertRaises(ValueError):
            apply_transaction_costs({"periods": periods}, sell_cost=-0.1, buy_cost=0.0)


if __name__ == "__main__":
    unittest.main()
