from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from qd.factor_robustness import (
    RobustnessConfig,
    aggregate_ic_by_period,
    average_cross_sectional_factor_correlation,
    block_bootstrap_mean_ci,
    directional_stability,
    neutralize_factor,
    realised_volatility_regimes,
    run_robustness_analysis,
)
from qd.factors import FACTOR_NAMES


class FactorRobustnessTests(unittest.TestCase):
    def test_period_regime_and_bootstrap_diagnostics(self) -> None:
        dates = pd.bdate_range("2025-01-01", periods=80)
        ic = pd.DataFrame({
            "positive": np.linspace(0.01, 0.08, len(dates)),
            "negative": np.linspace(-0.08, -0.01, len(dates)),
        }, index=dates)
        monthly = aggregate_ic_by_period(ic, "M")
        self.assertEqual(set(monthly["factor"]), {"positive", "negative"})
        self.assertTrue((monthly["n_days"] > 0).all())

        bootstrap = block_bootstrap_mean_ci(
            ic, iterations=500, block_length=5, random_seed=7
        )
        self.assertGreater(bootstrap.loc["positive", "ci_lower"], 0)
        self.assertLess(bootstrap.loc["negative", "ci_upper"], 0)
        self.assertTrue(bootstrap["significant_at_5pct"].all())

        rng = np.random.default_rng(3)
        returns = pd.DataFrame(
            rng.normal(size=(80, 40)) * np.linspace(0.5, 2.0, 80)[:, None],
            index=dates,
        )
        regimes = realised_volatility_regimes(returns)
        self.assertEqual(set(regimes["volatility_regime"]), {"high", "low"})
        self.assertTrue(regimes["median_threshold"].nunique() == 1)

    def test_correlation_and_history_only_direction_hit_rate(self) -> None:
        rng = np.random.default_rng(4)
        dates = pd.bdate_range("2025-01-01", periods=60)
        stocks = [f"S{i:03d}" for i in range(50)]
        base = pd.DataFrame(rng.normal(size=(60, 50)), index=dates, columns=stocks)
        factors = {
            "base": base,
            "same": 2.0 * base + rng.normal(scale=0.01, size=base.shape),
            "opposite": -base,
        }
        correlation, counts = average_cross_sectional_factor_correlation(
            factors, min_stocks=30
        )
        self.assertGreater(correlation.loc["base", "same"], 0.99)
        self.assertLess(correlation.loc["base", "opposite"], -0.99)
        self.assertEqual(counts.loc["base", "same"], len(dates))

        stable_ic = pd.DataFrame({"stable": np.full(60, -0.05)}, index=dates)
        stability = directional_stability(stable_ic, warmup=10)
        self.assertEqual(stability.loc["stable", "full_sample_direction"], "negative")
        self.assertEqual(stability.loc["stable", "prequential_hit_rate"], 1.0)
        self.assertTrue(stability.loc["stable", "split_direction_agrees"])

    def test_explicit_neutralization_removes_continuous_exposure(self) -> None:
        rng = np.random.default_rng(9)
        dates = pd.bdate_range("2025-01-01", periods=8)
        stocks = [f"S{i:03d}" for i in range(50)]
        size = pd.DataFrame(
            np.tile(np.linspace(-2, 2, 50), (8, 1)), index=dates, columns=stocks
        )
        factor = 3.0 * size + pd.DataFrame(
            rng.normal(scale=0.05, size=size.shape), index=dates, columns=stocks
        )
        residual = neutralize_factor(
            factor, continuous_exposures={"log_size": size}, min_stocks=30
        )
        correlations = [residual.loc[d].corr(size.loc[d]) for d in dates]
        self.assertLess(max(abs(x) for x in correlations), 1e-10)
        with self.assertRaisesRegex(ValueError, "real exposure"):
            neutralize_factor(factor)

    def test_end_to_end_run_records_skipped_neutralization(self) -> None:
        rng = np.random.default_rng(12)
        dates = pd.bdate_range("2025-01-01", periods=80)
        stocks = [f"S{i:03d}" for i in range(40)]
        forward = pd.DataFrame(
            rng.normal(scale=0.02, size=(80, 40)), index=dates, columns=stocks
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            factor_dir = root / "factors"
            out_dir = root / "robustness"
            factor_dir.mkdir()
            forward.to_csv(factor_dir / "forward_return_1d.csv")
            for offset, name in enumerate(FACTOR_NAMES):
                factor = forward + rng.normal(
                    scale=0.1 + offset * 0.01, size=forward.shape
                )
                factor.to_csv(factor_dir / f"{name}.csv")

            config = RobustnessConfig(
                rolling_window=20,
                rolling_min_periods=10,
                bootstrap_iterations=200,
                bootstrap_block_length=3,
                min_stocks=30,
                direction_warmup=10,
            )
            manifest = run_robustness_analysis(
                factor_dir=factor_dir, out_dir=out_dir, config=config
            )
            self.assertEqual(manifest["neutralization"]["status"], "skipped")
            self.assertTrue((out_dir / "ic_monthly.csv").is_file())
            self.assertTrue((out_dir / "factor_spearman_correlation.csv").is_file())
            saved = json.loads((out_dir / "analysis_manifest.json").read_text())
            self.assertIn("never use as a live feature", saved["analysis"]["volatility_regime_note"])


if __name__ == "__main__":
    unittest.main()
