"""Out-of-sample-minded diagnostics for daily cross-sectional factors.

The functions in this module deliberately separate descriptive diagnostics
from signals that could have been known in real time.  In particular, the
volatility regimes are based on realised *target-period* dispersion and are
therefore intended for ex-post stress analysis only.  Directional stability
also includes a prequential hit rate whose expected sign is estimated only
from earlier IC observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .evaluation import compute_ic_series
from .factors import FACTOR_NAMES


@dataclass(frozen=True)
class RobustnessConfig:
    """Configuration recorded alongside every robustness run."""

    rolling_window: int = 60
    rolling_min_periods: int = 20
    bootstrap_iterations: int = 2_000
    bootstrap_block_length: int = 5
    bootstrap_confidence: float = 0.95
    random_seed: int = 20250809
    min_stocks: int = 30
    direction_warmup: int = 20

    def validate(self) -> None:
        if self.rolling_window < 2:
            raise ValueError("rolling_window must be at least 2")
        if not 2 <= self.rolling_min_periods <= self.rolling_window:
            raise ValueError(
                "rolling_min_periods must be between 2 and rolling_window"
            )
        if self.bootstrap_iterations < 100:
            raise ValueError("bootstrap_iterations must be at least 100")
        if self.bootstrap_block_length < 1:
            raise ValueError("bootstrap_block_length must be positive")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must lie between 0 and 1")
        if self.min_stocks < 3:
            raise ValueError("min_stocks must be at least 3")
        if self.direction_warmup < 2:
            raise ValueError("direction_warmup must be at least 2")


def _read_wide_csv(path: Path, numeric: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame.columns = frame.columns.astype(str)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"duplicate dates in {path}")
    if frame.columns.has_duplicates:
        raise ValueError(f"duplicate stock columns in {path}")
    if numeric:
        frame = frame.apply(pd.to_numeric, errors="coerce").astype(np.float64)
    return frame


def load_factor_products(
    factor_dir: Path | None = None,
    factor_names: tuple[str, ...] = FACTOR_NAMES,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load the saved factor matrices and their one-day forward return."""

    root = Path(factor_dir or (OUTPUT_DIR / "factors"))
    factors: dict[str, pd.DataFrame] = {}
    for name in factor_names:
        path = root / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing factor product: {path}")
        factors[name] = _read_wide_csv(path)
    forward_path = root / "forward_return_1d.csv"
    if not forward_path.exists():
        raise FileNotFoundError(f"missing forward-return product: {forward_path}")
    forward = _read_wide_csv(forward_path)

    reference = forward
    for name, factor in factors.items():
        if not factor.index.equals(reference.index):
            raise ValueError(f"date axis does not match forward returns: {name}")
        if not factor.columns.equals(reference.columns):
            raise ValueError(f"stock axis does not match forward returns: {name}")
    return factors, forward


def compute_ic_table(
    factors: Mapping[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    method: str = "pearson",
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Compute one cross-sectional IC observation per factor and date."""

    return pd.DataFrame(
        {
            name: compute_ic_series(
                factor, forward_return, method=method, min_stocks=min_stocks
            )
            for name, factor in factors.items()
        }
    ).sort_index()


def _summarise_group(values: pd.Series) -> dict[str, float | int]:
    x = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if x.empty:
        return {
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "annualized_ir": np.nan,
            "positive_ratio": np.nan,
            "n_days": 0,
        }
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if len(x) > 1 else np.nan
    icir = mean / std if np.isfinite(std) and std > 0 else np.nan
    return {
        "mean_ic": mean,
        "std_ic": std,
        "icir": icir,
        "annualized_ir": icir * np.sqrt(252) if np.isfinite(icir) else np.nan,
        "positive_ratio": float((x > 0).mean()),
        "n_days": int(len(x)),
    }


def aggregate_ic_by_period(ic_daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate daily IC by a pandas period frequency (for example M or Q)."""

    if not isinstance(ic_daily.index, pd.DatetimeIndex):
        raise TypeError("ic_daily must use a DatetimeIndex")
    rows: list[dict[str, object]] = []
    periods = ic_daily.index.to_period(frequency)
    for factor in ic_daily.columns:
        grouped = ic_daily[factor].groupby(periods)
        for period, values in grouped:
            rows.append({
                "factor": str(factor),
                "period": str(period),
                **_summarise_group(values),
            })
    return pd.DataFrame(rows).sort_values(["factor", "period"]).reset_index(drop=True)


def aggregate_ic_by_half_year(ic_daily: pd.DataFrame) -> pd.DataFrame:
    """Summarise IC in calendar half-year segments."""

    labels = pd.Index(
        [f"{date.year}-H{1 if date.month <= 6 else 2}" for date in ic_daily.index],
        name="half_year",
    )
    rows: list[dict[str, object]] = []
    for factor in ic_daily.columns:
        for label, values in ic_daily[factor].groupby(labels):
            rows.append({
                "factor": str(factor),
                "half_year": str(label),
                **_summarise_group(values),
            })
    return (
        pd.DataFrame(rows)
        .sort_values(["factor", "half_year"])
        .reset_index(drop=True)
    )


def rolling_ic_statistics(
    ic_daily: pd.DataFrame,
    window: int = 60,
    min_periods: int = 20,
) -> pd.DataFrame:
    """Return long-form rolling mean, volatility, ICIR and sign ratio."""

    if window < 2 or not 2 <= min_periods <= window:
        raise ValueError("require 2 <= min_periods <= window")
    rows: list[pd.DataFrame] = []
    for factor in ic_daily.columns:
        series = ic_daily[factor].replace([np.inf, -np.inf], np.nan)
        rolling = series.rolling(window=window, min_periods=min_periods)
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        count = rolling.count()
        frame = pd.DataFrame({
            "date": ic_daily.index,
            "factor": str(factor),
            "rolling_mean_ic": mean.to_numpy(),
            "rolling_std_ic": std.to_numpy(),
            "rolling_icir": (mean / std.replace(0, np.nan)).to_numpy(),
            "rolling_positive_ratio": rolling.apply(
                lambda x: float(np.mean(x > 0)), raw=True
            ).to_numpy(),
            "n_days": count.fillna(0).astype(int).to_numpy(),
        })
        rows.append(frame.loc[frame["n_days"] >= min_periods])
    if not rows:
        return pd.DataFrame(columns=[
            "date", "factor", "rolling_mean_ic", "rolling_std_ic",
            "rolling_icir", "rolling_positive_ratio", "n_days",
        ])
    return pd.concat(rows, ignore_index=True).sort_values(["factor", "date"])


def realised_volatility_regimes(
    forward_return: pd.DataFrame,
) -> pd.DataFrame:
    """Label ex-post target-period cross-sectional dispersion as high or low.

    Because this uses the realised forward return, the label must never be
    used as a feature or trading-time filter.  It is a conditional stress-test
    label only.
    """

    clean = forward_return.replace([np.inf, -np.inf], np.nan)
    dispersion = clean.std(axis=1, ddof=1).dropna()
    threshold = float(dispersion.median())
    regime = pd.Series(
        np.where(dispersion > threshold, "high", "low"),
        index=dispersion.index,
        name="volatility_regime",
    )
    return pd.DataFrame({
        "realised_target_cross_sectional_volatility": dispersion,
        "volatility_regime": regime,
        "median_threshold": threshold,
    })


def ic_by_regime(ic_daily: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Summarise daily IC conditionally on a diagnostic regime label."""

    common = ic_daily.index.intersection(regimes.dropna().index)
    rows: list[dict[str, object]] = []
    for factor in ic_daily.columns:
        for regime, dates in regimes.loc[common].groupby(regimes.loc[common]).groups.items():
            rows.append({
                "factor": str(factor),
                "regime": str(regime),
                **_summarise_group(ic_daily.loc[list(dates), factor]),
            })
    return pd.DataFrame(rows).sort_values(["factor", "regime"]).reset_index(drop=True)


def block_bootstrap_mean_ci(
    ic_daily: pd.DataFrame,
    iterations: int = 2_000,
    block_length: int = 5,
    confidence: float = 0.95,
    random_seed: int = 20250809,
) -> pd.DataFrame:
    """Circular moving-block bootstrap confidence interval for mean daily IC."""

    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between 0 and 1")
    rng = np.random.default_rng(random_seed)
    alpha = 1.0 - confidence
    rows: list[dict[str, object]] = []
    for factor in ic_daily.columns:
        values = ic_daily[factor].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        n_days = len(values)
        if n_days == 0:
            continue
        effective_block = min(block_length, n_days)
        n_blocks = int(np.ceil(n_days / effective_block))
        starts = rng.integers(0, n_days, size=(iterations, n_blocks))
        offsets = np.arange(effective_block)[None, None, :]
        sample_index = (starts[:, :, None] + offsets) % n_days
        flat_index = sample_index.reshape(iterations, -1)[:, :n_days]
        samples = values[flat_index]
        means = samples.mean(axis=1)
        lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        # Re-centre observations under H0: mean(IC) == 0.  Measuring the
        # uncentred bootstrap's mass across zero would be a sign probability,
        # not a valid null-test p-value.
        null_means = (values - values.mean())[flat_index].mean(axis=1)
        p_value = (
            np.count_nonzero(np.abs(null_means) >= abs(values.mean())) + 1
        ) / (iterations + 1)
        rows.append({
            "factor": str(factor),
            "mean_ic": float(values.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "bootstrap_std": float(means.std(ddof=1)),
            "two_sided_p_value": float(p_value),
            "significant_at_5pct": bool(lower > 0 or upper < 0),
            "n_days": int(n_days),
            "block_length": int(effective_block),
            "iterations": int(iterations),
        })
    return pd.DataFrame(rows).set_index("factor")


def average_cross_sectional_factor_correlation(
    factors: Mapping[str, pd.DataFrame],
    method: str = "spearman",
    min_stocks: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average each day's cross-sectional factor-correlation matrix."""

    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    names = list(factors)
    if not names:
        raise ValueError("at least one factor is required")
    dates = factors[names[0]].index
    stocks = factors[names[0]].columns
    for name in names[1:]:
        dates = dates.intersection(factors[name].index)
        stocks = stocks.intersection(factors[name].columns)
    sums = pd.DataFrame(0.0, index=names, columns=names)
    counts = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for date in dates:
        cross_section = pd.DataFrame({
            name: factors[name].loc[date, stocks] for name in names
        }).replace([np.inf, -np.inf], np.nan)
        correlation = cross_section.corr(method=method, min_periods=min_stocks)
        valid = correlation.notna()
        sums = sums.add(correlation.fillna(0.0), fill_value=0.0)
        counts = counts.add(valid.astype(int), fill_value=0).astype(int)
    mean = sums.divide(counts.replace(0, np.nan))
    return mean, counts


def _direction_hit(values: pd.Series, expected_sign: float) -> float:
    signs = np.sign(values.replace([np.inf, -np.inf], np.nan).dropna())
    signs = signs[signs != 0]
    if signs.empty or expected_sign == 0 or not np.isfinite(expected_sign):
        return np.nan
    return float((signs == expected_sign).mean())


def _maximum_opposite_streak(values: pd.Series, expected_sign: float) -> int:
    longest = current = 0
    for sign in np.sign(values.dropna().to_numpy(float)):
        if sign != 0 and sign != expected_sign:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def directional_stability(
    ic_daily: pd.DataFrame,
    warmup: int = 20,
) -> pd.DataFrame:
    """Measure descriptive, split-sample and history-only sign stability."""

    if warmup < 2:
        raise ValueError("warmup must be at least 2")
    rows: list[dict[str, object]] = []
    for factor in ic_daily.columns:
        x = ic_daily[factor].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if x.empty:
            continue
        full_sign = float(np.sign(x.mean()))
        split = max(1, len(x) // 2)
        first = x.iloc[:split]
        second = x.iloc[split:]
        first_sign = float(np.sign(first.mean()))
        second_sign = float(np.sign(second.mean())) if not second.empty else np.nan

        prior_mean = x.expanding(min_periods=warmup).mean().shift(1)
        prequential = pd.DataFrame({"ic": x, "expected": np.sign(prior_mean)}).dropna()
        prequential = prequential.loc[
            (prequential["ic"] != 0) & (prequential["expected"] != 0)
        ]
        prequential_hit = (
            float((np.sign(prequential["ic"]) == prequential["expected"]).mean())
            if not prequential.empty else np.nan
        )

        monthly = x.groupby(x.index.to_period("M")).mean()
        quarterly = x.groupby(x.index.to_period("Q")).mean()
        rows.append({
            "factor": str(factor),
            "full_sample_mean_ic": float(x.mean()),
            "full_sample_direction": (
                "positive" if full_sign > 0 else "negative" if full_sign < 0 else "flat"
            ),
            "descriptive_daily_direction_hit_rate": _direction_hit(x, full_sign),
            "monthly_direction_hit_rate": _direction_hit(monthly, full_sign),
            "quarterly_direction_hit_rate": _direction_hit(quarterly, full_sign),
            "first_half_mean_ic": float(first.mean()),
            "second_half_mean_ic": float(second.mean()) if not second.empty else np.nan,
            "split_direction_agrees": bool(first_sign == second_sign)
            if np.isfinite(second_sign) else np.nan,
            "second_half_hit_rate_using_first_half_direction": _direction_hit(
                second, first_sign
            ),
            "prequential_hit_rate": prequential_hit,
            "prequential_n_days": int(len(prequential)),
            "maximum_opposite_daily_streak": _maximum_opposite_streak(x, full_sign),
            "n_days": int(len(x)),
        })
    return pd.DataFrame(rows).set_index("factor")


def neutralize_factor(
    factor: pd.DataFrame,
    continuous_exposures: Mapping[str, pd.DataFrame] | None = None,
    categorical_exposures: Mapping[str, pd.DataFrame] | None = None,
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Return daily cross-sectional OLS residuals for supplied exposures.

    No exposure is inferred.  Callers must provide real, date-by-stock
    continuous exposures (for example log market capitalisation) and/or
    categorical exposures (for example an industry code).  Missing exposure
    values are excluded independently on every date.
    """

    continuous = dict(continuous_exposures or {})
    categorical = dict(categorical_exposures or {})
    if not continuous and not categorical:
        raise ValueError("neutralization requires at least one real exposure table")
    if min_stocks < 3:
        raise ValueError("min_stocks must be at least 3")

    output = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns, dtype=float)
    for date in factor.index:
        data = pd.DataFrame({"factor": factor.loc[date]})
        for name, exposure in continuous.items():
            if date in exposure.index:
                data[f"continuous::{name}"] = pd.to_numeric(
                    exposure.loc[date].reindex(factor.columns), errors="coerce"
                )
            else:
                data[f"continuous::{name}"] = np.nan
        for name, exposure in categorical.items():
            if date in exposure.index:
                data[f"categorical::{name}"] = exposure.loc[date].reindex(factor.columns)
            else:
                data[f"categorical::{name}"] = np.nan
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < min_stocks:
            continue

        design_parts: list[pd.DataFrame] = []
        continuous_columns = [x for x in data if x.startswith("continuous::")]
        if continuous_columns:
            numeric = data[continuous_columns].astype(float)
            scale = numeric.std(ddof=0).replace(0, np.nan)
            numeric = (numeric - numeric.mean()).divide(scale)
            numeric = numeric.dropna(axis=1, how="all")
            if not numeric.empty:
                design_parts.append(numeric)
        categorical_columns = [x for x in data if x.startswith("categorical::")]
        for column in categorical_columns:
            dummies = pd.get_dummies(
                data[column].astype(str), prefix=column, drop_first=True, dtype=float
            )
            if not dummies.empty:
                design_parts.append(dummies)
        design = pd.concat(design_parts, axis=1) if design_parts else pd.DataFrame(index=data.index)
        design.insert(0, "intercept", 1.0)
        if len(data) <= design.shape[1] or len(data) < min_stocks:
            continue
        x_values = design.to_numpy(dtype=float)
        y_values = data["factor"].to_numpy(dtype=float)
        coefficients, _, _, _ = np.linalg.lstsq(x_values, y_values, rcond=None)
        output.loc[date, data.index] = y_values - x_values @ coefficients
    return output


def neutralize_factors(
    factors: Mapping[str, pd.DataFrame],
    continuous_exposures: Mapping[str, pd.DataFrame] | None = None,
    categorical_exposures: Mapping[str, pd.DataFrame] | None = None,
    min_stocks: int = 30,
) -> dict[str, pd.DataFrame]:
    """Neutralise each supplied factor with the same explicit exposures."""

    return {
        name: neutralize_factor(
            factor,
            continuous_exposures=continuous_exposures,
            categorical_exposures=categorical_exposures,
            min_stocks=min_stocks,
        )
        for name, factor in factors.items()
    }


def _write_analysis(
    factors: Mapping[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    target: Path,
    config: RobustnessConfig,
) -> dict[str, object]:
    target.mkdir(parents=True, exist_ok=True)
    ic_daily = compute_ic_table(
        factors, forward_return, method="pearson", min_stocks=config.min_stocks
    )
    rank_ic_daily = compute_ic_table(
        factors, forward_return, method="spearman", min_stocks=config.min_stocks
    )
    regimes = realised_volatility_regimes(forward_return)
    correlation, correlation_counts = average_cross_sectional_factor_correlation(
        factors, method="spearman", min_stocks=config.min_stocks
    )

    outputs: dict[str, pd.DataFrame] = {
        "ic_daily.csv": ic_daily,
        "rank_ic_daily.csv": rank_ic_daily,
        "ic_monthly.csv": aggregate_ic_by_period(ic_daily, "M"),
        "rank_ic_monthly.csv": aggregate_ic_by_period(rank_ic_daily, "M"),
        "ic_quarterly.csv": aggregate_ic_by_period(ic_daily, "Q"),
        "rank_ic_quarterly.csv": aggregate_ic_by_period(rank_ic_daily, "Q"),
        "ic_half_year.csv": aggregate_ic_by_half_year(ic_daily),
        "rank_ic_half_year.csv": aggregate_ic_by_half_year(rank_ic_daily),
        "ic_rolling.csv": rolling_ic_statistics(
            ic_daily, config.rolling_window, config.rolling_min_periods
        ),
        "rank_ic_rolling.csv": rolling_ic_statistics(
            rank_ic_daily, config.rolling_window, config.rolling_min_periods
        ),
        "volatility_regime_by_date.csv": regimes,
        "ic_by_volatility_regime.csv": ic_by_regime(
            ic_daily, regimes["volatility_regime"]
        ),
        "rank_ic_by_volatility_regime.csv": ic_by_regime(
            rank_ic_daily, regimes["volatility_regime"]
        ),
        "ic_bootstrap_ci.csv": block_bootstrap_mean_ci(
            ic_daily,
            config.bootstrap_iterations,
            config.bootstrap_block_length,
            config.bootstrap_confidence,
            config.random_seed,
        ),
        "rank_ic_bootstrap_ci.csv": block_bootstrap_mean_ci(
            rank_ic_daily,
            config.bootstrap_iterations,
            config.bootstrap_block_length,
            config.bootstrap_confidence,
            config.random_seed + 1,
        ),
        "factor_spearman_correlation.csv": correlation,
        "factor_spearman_correlation_n_days.csv": correlation_counts,
        "directional_stability.csv": directional_stability(
            ic_daily, config.direction_warmup
        ),
        "rank_directional_stability.csv": directional_stability(
            rank_ic_daily, config.direction_warmup
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            target / filename,
            float_format="%.8f",
            index=not isinstance(frame.index, pd.RangeIndex),
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    rolling = outputs["ic_rolling.csv"]
    for factor, group in rolling.groupby("factor", sort=False):
        axes[0].plot(group["date"], group["rolling_mean_ic"], label=factor)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"Rolling {config.rolling_window}-day mean IC")
    axes[0].set_ylabel("Mean IC")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    bootstrap = outputs["ic_bootstrap_ci.csv"].reset_index()
    factor_column = bootstrap.columns[0]
    positions = np.arange(len(bootstrap))
    lower = bootstrap["mean_ic"] - bootstrap["ci_lower"]
    upper = bootstrap["ci_upper"] - bootstrap["mean_ic"]
    axes[1].errorbar(
        positions,
        bootstrap["mean_ic"],
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, bootstrap[factor_column], rotation=25, ha="right")
    axes[1].set_title("Block-bootstrap 95% CI of mean IC")
    axes[1].grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target / "factor_robustness.png", dpi=170)
    plt.close(figure)

    return {
        "factor_names": list(factors),
        "date_start": str(forward_return.index.min().date()),
        "date_end": str(forward_return.index.max().date()),
        "n_calendar_rows": int(len(forward_return)),
        "n_stocks": int(len(forward_return.columns)),
        "volatility_regime_note": (
            "Ex-post stress diagnostic based on realised target-period "
            "cross-sectional return dispersion; never use as a live feature."
        ),
        "files": sorted(outputs),
    }


def run_robustness_analysis(
    factor_dir: Path | None = None,
    out_dir: Path | None = None,
    config: RobustnessConfig | None = None,
    continuous_exposures: Mapping[str, pd.DataFrame] | None = None,
    categorical_exposures: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, object]:
    """Run and persist the complete robustness suite.

    When actual exposure tables are supplied an additional ``neutralized``
    directory is produced.  Otherwise neutralisation is explicitly marked as
    skipped, rather than silently manufacturing industry or size data.
    """

    settings = config or RobustnessConfig()
    settings.validate()
    factors, forward_return = load_factor_products(factor_dir)
    target = Path(out_dir or (OUTPUT_DIR / "factor_robustness"))
    base = _write_analysis(factors, forward_return, target, settings)

    continuous = dict(continuous_exposures or {})
    categorical = dict(categorical_exposures or {})
    if continuous or categorical:
        residual_factors = neutralize_factors(
            factors,
            continuous_exposures=continuous,
            categorical_exposures=categorical,
            min_stocks=settings.min_stocks,
        )
        neutralized = _write_analysis(
            residual_factors, forward_return, target / "neutralized", settings
        )
        neutralization_status: dict[str, object] = {
            "status": "completed",
            "continuous_exposures": sorted(continuous),
            "categorical_exposures": sorted(categorical),
            "output_directory": "neutralized",
            "analysis": neutralized,
        }
    else:
        neutralization_status = {
            "status": "skipped",
            "reason": (
                "No real industry or size exposure table was supplied. "
                "Use the explicit continuous_exposures/categorical_exposures "
                "interface when those data become available."
            ),
            "continuous_exposures": [],
            "categorical_exposures": [],
        }

    manifest: dict[str, object] = {
        "config": asdict(settings),
        "analysis": base,
        "neutralization": neutralization_status,
    }
    (target / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ic_daily = compute_ic_table(
        factors, forward_return, method="pearson", min_stocks=settings.min_stocks
    )
    quarterly = aggregate_ic_by_period(ic_daily, "Q")
    latest_period = quarterly["period"].max()
    latest = quarterly[quarterly["period"] == latest_period].set_index("factor")
    bootstrap = block_bootstrap_mean_ci(
        ic_daily,
        settings.bootstrap_iterations,
        settings.bootstrap_block_length,
        settings.bootstrap_confidence,
        settings.random_seed,
    )
    lines = [
        "# 因子稳健性分析",
        "",
        "本目录在原有全样本 IC/分层评价之上，增加时间分段、滚动 IC、区块 bootstrap、",
        "波动状态、因子相关性和仅使用历史 IC 的方向稳定性检验。",
        "",
        "## 全样本与最新季度",
        "",
        "| 因子 | 全样本平均 IC | bootstrap 95% CI | 最新季度平均 IC |",
        "|---|---:|---:|---:|",
    ]
    for factor in factors:
        row = bootstrap.loc[factor]
        latest_ic = latest.loc[factor, "mean_ic"] if factor in latest.index else np.nan
        lines.append(
            f"| {factor} | {row['mean_ic']:.4f} | "
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}] | {latest_ic:.4f} |"
        )
    lines.extend([
        "",
        f"最新季度为 `{latest_period}`。近期方向漂移说明不能把全样本负号固定成永久方向；",
        "组合应继续使用严格滞后的滚动 IC，并在滚动 IC 接近 0 或翻转时自动降权。",
        "",
        "## 中性化边界",
        "",
        (
            "当前没有真实行业/市值暴露表，因此中性化被明确标记为 `skipped`。"
            if neutralization_status["status"] == "skipped"
            else "本次已使用用户提供的真实暴露完成中性化分析。"
        ),
        "命令行支持传入真实连续暴露和分类暴露；不会用股票代码或随机分组伪造行业数据。",
        "",
    ])
    (target / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
