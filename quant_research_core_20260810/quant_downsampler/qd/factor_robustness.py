"""Factor stability, market-regime, and benchmark-relative diagnostics.

All factor statistics are descriptive.  Trading results delegate to the strict
cash-and-share ledger in :mod:`qd.task4_strict` so diagnostics cannot silently
replace executable holdings with a set of equal-weight names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .factors import compute_all_factors, load_daily_data
from .strategy_analysis import relative_metrics
from .task4_strict import run_strict_backtest


def compute_ic_table(
    factors: Mapping[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    method: str = "pearson",
    min_stocks: int = 30,
) -> pd.DataFrame:
    """Return one cross-sectional IC observation per factor and date."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be pearson or spearman")
    rows: dict[str, pd.Series] = {}
    for name, factor in factors.items():
        values: list[float] = []
        for date in forward_return.index:
            sample = pd.concat(
                [factor.loc[date].rename("factor"), forward_return.loc[date].rename("return")],
                axis=1,
            ).replace([np.inf, -np.inf], np.nan).dropna()
            if (
                len(sample) < min_stocks
                or sample["factor"].nunique() < 2
                or sample["return"].nunique() < 2
            ):
                values.append(np.nan)
                continue
            values.append(float(sample["factor"].corr(sample["return"], method=method)))
        rows[str(name)] = pd.Series(values, index=forward_return.index)
    return pd.DataFrame(rows)


def mask_nontradable_forward_returns(
    forward_return: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Keep labels only when the stock traded on signal and realization dates."""
    tradable = (volume > 0) & (volume.shift(-1) > 0)
    return forward_return.where(tradable)


def aggregate_ic(ic_daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Summarise daily IC by a pandas calendar period."""
    periods = ic_daily.index.to_period(frequency)
    rows: list[dict[str, object]] = []
    for factor in ic_daily.columns:
        for period, group in ic_daily[factor].groupby(periods):
            clean = group.dropna()
            std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
            rows.append({
                "factor": factor,
                "period": str(period),
                "mean_ic": float(clean.mean()) if len(clean) else np.nan,
                "std_ic": std,
                "icir": float(clean.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
                "positive_ratio": float((clean > 0).mean()) if len(clean) else np.nan,
                "n_days": int(len(clean)),
            })
    return pd.DataFrame(rows)


def aggregate_ic_half_year(ic_daily: pd.DataFrame) -> pd.DataFrame:
    """Summarise daily IC in unambiguous calendar H1/H2 buckets."""
    labels = pd.Index(
        [f"{date.year}-H{1 if date.month <= 6 else 2}" for date in ic_daily.index],
        name="half_year",
    )
    rows: list[dict[str, object]] = []
    for factor in ic_daily:
        for period, group in ic_daily[factor].groupby(labels):
            clean = group.dropna()
            std = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
            rows.append({
                "factor": factor,
                "period": str(period),
                "mean_ic": float(clean.mean()) if len(clean) else np.nan,
                "std_ic": std,
                "icir": float(clean.mean() / std) if np.isfinite(std) and std > 0 else np.nan,
                "positive_ratio": float((clean > 0).mean()) if len(clean) else np.nan,
                "n_days": int(len(clean)),
            })
    return pd.DataFrame(rows)


def rolling_ic(ic_daily: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
    """Return long-form rolling IC means, risks, and sign ratios."""
    if not 2 <= min_periods <= window:
        raise ValueError("require 2 <= min_periods <= window")
    frames: list[pd.DataFrame] = []
    for factor in ic_daily:
        rolling = ic_daily[factor].rolling(window, min_periods=min_periods)
        frame = pd.DataFrame({
            "date": ic_daily.index,
            "factor": factor,
            "rolling_mean_ic": rolling.mean().to_numpy(),
            "rolling_std_ic": rolling.std(ddof=1).to_numpy(),
            "rolling_positive_ratio": rolling.apply(lambda x: float(np.mean(x > 0)), raw=True).to_numpy(),
            "n_days": rolling.count().to_numpy(dtype=int),
        })
        frames.append(frame.loc[frame["n_days"] >= min_periods])
    return pd.concat(frames, ignore_index=True)


def block_bootstrap_mean_ci(
    ic_daily: pd.DataFrame,
    iterations: int = 2_000,
    block_length: int = 5,
    confidence: float = 0.95,
    seed: int = 20250809,
) -> pd.DataFrame:
    """Circular moving-block bootstrap intervals and centred-null p-values."""
    if iterations < 100 or block_length < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    rows: list[dict[str, object]] = []
    for factor in ic_daily:
        values = ic_daily[factor].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if not len(values):
            continue
        block = min(block_length, len(values))
        n_blocks = int(np.ceil(len(values) / block))
        starts = rng.integers(0, len(values), size=(iterations, n_blocks))
        index = (starts[:, :, None] + np.arange(block)[None, None, :]) % len(values)
        index = index.reshape(iterations, -1)[:, : len(values)]
        means = values[index].mean(axis=1)
        null_means = (values - values.mean())[index].mean(axis=1)
        lower, upper = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        p_value = (np.count_nonzero(np.abs(null_means) >= abs(values.mean())) + 1) / (iterations + 1)
        rows.append({
            "factor": factor,
            "mean_ic": float(values.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "two_sided_p_value": float(p_value),
            "significant_at_5pct": bool(lower > 0 or upper < 0),
            "n_days": int(len(values)),
            "block_length": int(block),
        })
    return pd.DataFrame(rows).set_index("factor")


def build_market_regimes(close: pd.DataFrame, trend_window: int = 20) -> pd.DataFrame:
    """Build ex-post target-day direction and signal-time trailing-trend regimes."""
    returns = close.pct_change(fill_method=None)
    market = returns.mean(axis=1, skipna=True).rename("market_return")
    trailing = (1.0 + market).rolling(trend_window, min_periods=trend_window).apply(np.prod, raw=True) - 1.0
    target_market = market.shift(-1)
    return pd.DataFrame({
        "market_return": market,
        "target_market_return": target_market,
        "target_market_direction": np.where(target_market >= 0, "up", "down"),
        "trailing_market_return": trailing,
        "known_market_trend": np.where(trailing >= 0, "bull", "bear"),
    }, index=close.index)


def ic_by_regime(ic_daily: pd.DataFrame, regimes: pd.Series, regime_name: str) -> pd.DataFrame:
    """Summarise each factor conditional on a diagnostic market label."""
    rows: list[dict[str, object]] = []
    for factor in ic_daily:
        frame = pd.concat([ic_daily[factor].rename("ic"), regimes.rename("regime")], axis=1).dropna()
        for regime, group in frame.groupby("regime"):
            rows.append({
                "factor": factor,
                "regime_type": regime_name,
                "regime": str(regime),
                "mean_ic": float(group["ic"].mean()),
                "positive_ratio": float((group["ic"] > 0).mean()),
                "n_days": int(len(group)),
            })
    return pd.DataFrame(rows)


def average_factor_correlation(factors: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Average daily cross-sectional Spearman factor correlations."""
    names = list(factors)
    daily: list[pd.DataFrame] = []
    for date in next(iter(factors.values())).index:
        panel = pd.DataFrame({name: factors[name].loc[date] for name in names})
        valid = [name for name in names if panel[name].nunique(dropna=True) >= 2]
        correlation = pd.DataFrame(np.nan, index=names, columns=names)
        if valid:
            correlation.loc[valid, valid] = panel[valid].corr(method="spearman", min_periods=30)
        daily.append(correlation)
    total = sum(frame.fillna(0.0) for frame in daily)
    count = sum(frame.notna().astype(int) for frame in daily)
    return total.divide(count.where(count > 0))


def _single_factor_market_results(
    factors: Mapping[str, pd.DataFrame],
    daily: dict[str, pd.DataFrame],
    benchmark: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, factor in factors.items():
        result = run_strict_backtest({name: factor}, daily, execution="close", method="adaptive")
        selections = result["selections"]
        first_execution = pd.to_datetime(selections["execution_date"]).min() if len(selections) else None
        strategy = result["daily_return"]
        if first_execution is not None:
            strategy = strategy.loc[strategy.index >= first_execution]
        aligned_benchmark = benchmark.reindex(strategy.index)
        comparison = relative_metrics(strategy, aligned_benchmark)
        rows.append({
            "factor": name,
            "first_execution_date": first_execution,
            "strategy_total_return": comparison["strategy"]["total_return"],
            "market_total_return": comparison["benchmark"]["total_return"],
            "geometric_excess_return": comparison["geometric_excess_return"],
            "information_ratio": comparison["information_ratio"],
            "beta": comparison["beta"],
            "annualized_arithmetic_alpha": comparison["annualized_arithmetic_alpha"],
        })
    return pd.DataFrame(rows).sort_values("geometric_excess_return", ascending=False)


def run_factor_robustness(
    output_dir: str | Path = OUTPUT_DIR,
    iterations: int = 2_000,
    block_length: int = 5,
) -> dict[str, object]:
    """Run and persist the complete factor/market diagnostic package."""
    root = Path(output_dir)
    target = root / "factor_robustness"
    target.mkdir(parents=True, exist_ok=True)
    daily = load_daily_data(root / "daily")
    bundle = compute_all_factors(daily)
    forward = bundle["forward_return_1d"]
    # Match the formal Task2–3 evaluation universe: a stock contributes to
    # IC only when it traded on both the signal and realization dates.
    forward = mask_nontradable_forward_returns(forward, daily["volume"])
    factors = {name: value for name, value in bundle.items() if name != "forward_return_1d"}

    ic = compute_ic_table(factors, forward, "pearson")
    rank_ic = compute_ic_table(factors, forward, "spearman")
    regimes = build_market_regimes(daily["close"])
    regime_ic = pd.concat([
        ic_by_regime(ic, regimes["target_market_direction"], "target_market_direction"),
        ic_by_regime(ic, regimes["known_market_trend"], "known_market_trend"),
    ], ignore_index=True)
    bootstrap = block_bootstrap_mean_ci(ic, iterations, block_length)
    rank_bootstrap = block_bootstrap_mean_ci(rank_ic, iterations, block_length)
    correlation = average_factor_correlation(factors)
    benchmark = daily["close"].pct_change(fill_method=None).mean(axis=1, skipna=True)
    single_factor = _single_factor_market_results(factors, daily, benchmark)

    outputs = {
        "ic_daily.csv": ic,
        "rank_ic_daily.csv": rank_ic,
        "ic_monthly.csv": aggregate_ic(ic, "M"),
        "ic_quarterly.csv": aggregate_ic(ic, "Q"),
        "ic_half_year.csv": aggregate_ic_half_year(ic),
        "rolling_ic_60d.csv": rolling_ic(ic),
        "bootstrap_ic.csv": bootstrap,
        "bootstrap_rank_ic.csv": rank_bootstrap,
        "market_regimes.csv": regimes,
        "ic_by_market_regime.csv": regime_ic,
        "factor_correlation.csv": correlation,
        "single_factor_market_metrics.csv": single_factor,
    }
    indexed_outputs = {
            "ic_daily.csv", "rank_ic_daily.csv", "bootstrap_ic.csv",
            "bootstrap_rank_ic.csv", "market_regimes.csv", "factor_correlation.csv",
    }
    for filename, frame in outputs.items():
        frame.to_csv(
            target / filename,
            float_format="%.10f",
            index=filename in indexed_outputs,
            index_label="factor" if filename.startswith("bootstrap_") else None,
        )

    latest_quarter = aggregate_ic(ic, "Q")
    latest_period = latest_quarter["period"].max()
    latest = latest_quarter.loc[latest_quarter["period"].eq(latest_period)]
    summary = {
        "status": "completed",
        "factor_count": len(factors),
        "date_start": str(ic.index.min().date()),
        "date_end": str(ic.index.max().date()),
        "latest_quarter": latest_period,
        "latest_quarter_positive_factor_count": int((latest["mean_ic"] > 0).sum()),
        "bootstrap_significant_factor_count": int(bootstrap["significant_at_5pct"].sum()),
        "neutralization": "skipped: no genuine market-cap or industry exposure tables are supplied",
        "market_regime_note": (
            "target_market_direction is ex-post diagnostic only; known_market_trend uses trailing data. "
            "A common market move does not mechanically change cross-sectional IC."
        ),
        "best_single_factor_by_excess": single_factor.iloc[0].to_dict(),
        "worst_single_factor_by_excess": single_factor.iloc[-1].to_dict(),
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
