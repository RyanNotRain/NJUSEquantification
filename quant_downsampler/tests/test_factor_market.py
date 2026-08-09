from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.backtest_robustness import (
    build_equal_weight_market_proxy,
    evaluate_strategy_against_market,
    factor_diagnostics_for_market_report,
    history_only_factor_weights,
    run_turnover_aware_backtest,
)
from qd.factors import (
    EXPERIMENTAL_FACTOR_NAMES,
    FACTOR_NAMES,
    compute_illiquidity_20d,
)


class ExperimentalFactorTests(unittest.TestCase):
    def test_official_factor_universe_remains_exactly_four(self) -> None:
        self.assertEqual(
            FACTOR_NAMES,
            (
                "example_factor",
                "momentum_5d",
                "buy_sell_imbalance",
                "intraday_range",
            ),
        )
        self.assertEqual(EXPERIMENTAL_FACTOR_NAMES, ("illiquidity_20d",))
        self.assertTrue(set(FACTOR_NAMES).isdisjoint(EXPERIMENTAL_FACTOR_NAMES))

    def test_illiquidity_uses_trailing_returns_and_amount_through_table_date(self) -> None:
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        close = pd.DataFrame(
            {"A": [100.0, 110.0, 99.0, 99.0, 108.9]}, index=dates
        )
        amount = pd.DataFrame({"A": [1000.0] * 5}, index=dates)
        factor = compute_illiquidity_20d(close, amount, window=2)
        self.assertTrue(factor.iloc[:2].isna().all().all())
        self.assertAlmostEqual(factor.loc[dates[2], "A"], 0.0001)
        self.assertAlmostEqual(factor.loc[dates[3], "A"], 0.00005)

        # A later observation cannot rewrite an already available factor row.
        changed_close = close.copy()
        changed_amount = amount.copy()
        changed_close.loc[dates[3], "A"] = 150.0
        changed_amount.loc[dates[3], "A"] = 1.0
        changed = compute_illiquidity_20d(changed_close, changed_amount, window=2)
        self.assertAlmostEqual(changed.loc[dates[2], "A"], factor.loc[dates[2], "A"])

    def test_illiquidity_rejects_misaligned_axes(self) -> None:
        close = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2026-01-01", periods=2))
        amount = pd.DataFrame({"B": [1.0, 2.0]}, index=close.index)
        with self.assertRaises(ValueError):
            compute_illiquidity_20d(close, amount)

    def test_custom_factor_direction_is_history_only(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=9)
        stocks = [f"S{i:02d}" for i in range(32)]
        cross_section = np.linspace(-1.0, 1.0, len(stocks))
        factor = pd.DataFrame(
            np.tile(cross_section, (len(dates), 1)), index=dates, columns=stocks
        )
        forward = pd.DataFrame(
            np.tile(cross_section * 0.01, (len(dates), 1)),
            index=dates,
            columns=stocks,
        )
        original = history_only_factor_weights(
            {"illiquidity_20d": factor},
            forward,
            ("illiquidity_20d",),
            lookback=3,
        )
        changed_forward = forward.copy()
        changed_forward.loc[dates[4]:] *= -1.0
        changed = history_only_factor_weights(
            {"illiquidity_20d": factor},
            changed_forward,
            ("illiquidity_20d",),
            lookback=3,
        )
        pd.testing.assert_series_equal(
            original.loc[: dates[5], "illiquidity_20d"],
            changed.loc[: dates[5], "illiquidity_20d"],
        )
        self.assertNotEqual(
            original.loc[dates[6], "illiquidity_20d"],
            changed.loc[dates[6], "illiquidity_20d"],
        )

    def test_experimental_factor_can_run_the_economic_backtest(self) -> None:
        dates = pd.bdate_range("2026-01-01", periods=12)
        stocks = [f"S{i:02d}" for i in range(32)]
        scores = np.linspace(-1.0, 1.0, len(stocks))
        factor = pd.DataFrame(
            np.tile(scores, (len(dates), 1)), index=dates, columns=stocks
        )
        close = pd.DataFrame(index=dates, columns=stocks, dtype=float)
        close.iloc[0] = 100.0
        for row in range(1, len(dates)):
            close.iloc[row] = close.iloc[row - 1].to_numpy(float) * (
                1.0 + scores * 0.001
            )
        forward = close.shift(-1).divide(close) - 1.0
        volume = pd.DataFrame(1.0, index=dates, columns=stocks)
        amount = pd.DataFrame(1000.0, index=dates, columns=stocks)
        result = run_turnover_aware_backtest(
            {"illiquidity_20d": factor},
            forward,
            close,
            close,
            volume,
            amount,
            lookback=3,
            top_n=5,
            buffer_n=5,
            max_replacements=5,
            min_liquidity_quantile=None,
            max_volatility_quantile=None,
            factor_names=("illiquidity_20d",),
        )
        self.assertEqual(result["factor_names"], ("illiquidity_20d",))
        self.assertFalse(result["periods"].empty)
        self.assertFalse(result["weight_history"].empty)
        self.assertTrue(
            (result["weight_history"]["history_only_direction"] == 1).all()
        )


class MarketBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2026-01-01", periods=3, freq="D")
        self.close = pd.DataFrame(
            {
                "A": [100.0, 110.0, 99.0],
                "B": [100.0, 90.0, 99.0],
            },
            index=self.dates,
        )

    def test_market_proxy_is_exact_cross_sectional_equal_weight_return(self) -> None:
        proxy = build_equal_weight_market_proxy(self.close, self.dates[1:])
        np.testing.assert_allclose(proxy["market_return"], [0.0, 0.0], atol=1e-15)
        self.assertEqual(proxy["n_market_stocks"].tolist(), [2, 2])
        self.assertTrue(proxy.index.equals(self.dates[1:]))

    def test_market_comparison_reports_all_requested_relative_metrics(self) -> None:
        strategy = pd.Series([0.10, -0.10], index=self.dates[1:])
        periods, metrics = evaluate_strategy_against_market(
            strategy, self.close, "test_strategy"
        )
        self.assertTrue(periods.index.equals(strategy.index))
        self.assertAlmostEqual(metrics["strategy_total_return"], -0.01)
        self.assertAlmostEqual(metrics["benchmark_total_return"], 0.0)
        self.assertAlmostEqual(metrics["percentage_point_lead"], -1.0)
        self.assertAlmostEqual(metrics["relative_wealth"], 0.99)
        self.assertAlmostEqual(metrics["relative_max_drawdown"], -0.10)
        self.assertAlmostEqual(metrics["outperformance_day_ratio"], 0.5)
        expected_te = strategy.std(ddof=1) * np.sqrt(252.0)
        self.assertAlmostEqual(metrics["tracking_error"], expected_te)
        expected_ir = strategy.mean() / strategy.std(ddof=1) * np.sqrt(252.0)
        self.assertAlmostEqual(metrics["information_ratio"], expected_ir)

    def test_market_proxy_rejects_missing_or_unsorted_dates(self) -> None:
        with self.assertRaises(ValueError):
            build_equal_weight_market_proxy(
                self.close, pd.DatetimeIndex([pd.Timestamp("2026-01-10")])
            )
        with self.assertRaises(ValueError):
            build_equal_weight_market_proxy(self.close, self.dates[[2, 1]])

    def test_experimental_diagnostics_include_required_fields(self) -> None:
        dates = pd.bdate_range("2026-04-01", periods=40)
        stocks = [f"S{i:02d}" for i in range(35)]
        x = np.linspace(-1.0, 1.0, len(stocks))
        factors = {
            name: pd.DataFrame(
                np.tile(x * (offset + 1), (len(dates), 1)),
                index=dates,
                columns=stocks,
            )
            for offset, name in enumerate((*FACTOR_NAMES, *EXPERIMENTAL_FACTOR_NAMES))
        }
        forward = pd.DataFrame(
            np.tile(x * 0.01, (len(dates), 1)), index=dates, columns=stocks
        )
        table, diagnostics = factor_diagnostics_for_market_report(
            factors, forward, bootstrap_iterations=100, bootstrap_seed=7
        )
        experimental = diagnostics["illiquidity_20d"]
        required = {
            "full_sample_mean_ic",
            "full_sample_mean_rank_ic",
            "block_bootstrap_5d_ci_lower",
            "block_bootstrap_5d_ci_upper",
            "mean_ic_2026q2",
            "most_correlated_official_factor",
            "max_average_cross_sectional_spearman_with_official",
            "max_abs_average_cross_sectional_spearman_with_official",
        }
        self.assertTrue(required.issubset(experimental))
        self.assertEqual(experimental["factor_scope"], "experimental")
        self.assertEqual(len(table), 5)
        self.assertEqual(len(FACTOR_NAMES), 4)


if __name__ == "__main__":
    unittest.main()
