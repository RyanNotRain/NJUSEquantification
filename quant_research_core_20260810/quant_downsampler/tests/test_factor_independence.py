from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from qd.factor_independence import (
    correlation_clusters,
    historical_ic_quality,
    independence_metrics,
    orthogonalize_cross_sectionally,
    select_cluster_representatives,
)


class FactorIndependenceTests(unittest.TestCase):
    def test_clustering_uses_connected_absolute_correlation(self) -> None:
        matrix = pd.DataFrame(
            [[1.0, .7, .1, 0], [.7, 1.0, -.65, 0], [.1, -.65, 1.0, .2], [0, 0, .2, 1.0]],
            index=list("abcd"), columns=list("abcd"),
        )
        self.assertEqual(correlation_clusters(matrix, .6), [["a", "b", "c"], ["d"]])

    def test_historical_quality_delays_unrealized_ic(self) -> None:
        dates = pd.date_range("2025-01-01", periods=6)
        ic = pd.DataFrame({"a": [1, 1, 1, 100, 100, 100], "b": [.5] * 6}, index=dates)
        quality = historical_ic_quality(ic, dates[4], realization_lag=2)
        self.assertAlmostEqual(float(quality.loc["a", "mean_ic"]), 1.0)
        self.assertEqual(int(quality.loc["a", "available_ic_days"]), 3)

    def test_representative_is_selected_by_frozen_icir(self) -> None:
        quality = pd.DataFrame(
            {"absolute_icir": [.2, .8, .1], "available_ic_days": [50, 50, 50],
             "mean_ic": [.01, -.03, .01], "ic_std": [.05, .04, .1]},
            index=["a", "b", "c"],
        )
        selected, table = select_cluster_representatives([["a", "b"], ["c"]], quality)
        self.assertEqual(selected, ["b", "c"])
        self.assertEqual(int(table["selected"].sum()), 2)

    def test_orthogonal_components_remove_linear_rank_dependence(self) -> None:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2025-01-01", periods=3)
        stocks = [f"s{i}" for i in range(80)]
        base = rng.normal(size=(3, 80))
        factors = {
            "a": pd.DataFrame(base, index=dates, columns=stocks),
            "b": pd.DataFrame(2 * base + rng.normal(scale=.1, size=(3, 80)), index=dates, columns=stocks),
            "c": pd.DataFrame(rng.normal(size=(3, 80)), index=dates, columns=stocks),
        }
        transformed = orthogonalize_cross_sectionally(factors, ["a", "b", "c"], min_stocks=30)
        first = pd.DataFrame({name: frame.iloc[0] for name, frame in transformed.items()}).dropna()
        correlation = first.corr()
        offdiag = np.abs(correlation.to_numpy()[np.triu_indices(3, 1)])
        self.assertLess(float(offdiag.max()), 1e-10)
        metrics = independence_metrics(correlation)
        self.assertAlmostEqual(float(metrics["effective_rank"]), 3.0, places=8)


if __name__ == "__main__":
    unittest.main()
