"""Bootstrap uncertainty and rolling risk diagnostics for Task 4 excess returns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR


def moving_block_indices(
    observations: int,
    block_length: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    """Return circular moving-block bootstrap indices with deterministic length."""
    if observations < 2:
        raise ValueError("at least two observations are required")
    if block_length <= 0 or iterations <= 0:
        raise ValueError("block_length and iterations must be positive")
    length = min(int(block_length), int(observations))
    blocks = int(np.ceil(observations / length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, observations, size=(iterations, blocks))
    offsets = np.arange(length, dtype=np.int64)
    indices = (starts[..., None] + offsets) % observations
    return indices.reshape(iterations, -1)[:, :observations]


def bootstrap_relative_performance(
    strategy: pd.Series,
    benchmark: pd.Series,
    block_length: int = 5,
    iterations: int = 5000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Paired block-bootstrap confidence intervals for benchmark-relative returns."""
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(aligned) < 2:
        raise ValueError("strategy and benchmark need at least two aligned returns")
    strategy_values = aligned["strategy"].to_numpy(dtype=np.float64)
    benchmark_values = aligned["benchmark"].to_numpy(dtype=np.float64)
    excess = strategy_values - benchmark_values
    indices = moving_block_indices(len(aligned), block_length, iterations, seed)
    sampled_strategy = strategy_values[indices]
    sampled_benchmark = benchmark_values[indices]
    strategy_growth = np.prod(1.0 + sampled_strategy, axis=1)
    benchmark_growth = np.prod(1.0 + sampled_benchmark, axis=1)
    geometric = strategy_growth / benchmark_growth - 1.0
    spread = strategy_growth - benchmark_growth
    mean_excess = (sampled_strategy - sampled_benchmark).mean(axis=1)

    observed_strategy_growth = float(np.prod(1.0 + strategy_values))
    observed_benchmark_growth = float(np.prod(1.0 + benchmark_values))
    observed_mean = float(excess.mean())
    centered = excess - observed_mean
    null_means = centered[indices].mean(axis=1)
    one_sided_p = float((1.0 + np.sum(null_means >= observed_mean)) / (iterations + 1.0))

    def interval(values: np.ndarray) -> tuple[float, float]:
        low, high = np.quantile(values, [0.025, 0.975])
        return float(low), float(high)

    geometric_low, geometric_high = interval(geometric)
    spread_low, spread_high = interval(spread)
    mean_low, mean_high = interval(mean_excess * 252.0)
    return {
        "observations": int(len(aligned)),
        "block_length": int(min(block_length, len(aligned))),
        "iterations": int(iterations),
        "strategy_total_return": observed_strategy_growth - 1.0,
        "benchmark_total_return": observed_benchmark_growth - 1.0,
        "cumulative_return_spread": observed_strategy_growth - observed_benchmark_growth,
        "cumulative_return_spread_ci_low": spread_low,
        "cumulative_return_spread_ci_high": spread_high,
        "geometric_excess_return": observed_strategy_growth / observed_benchmark_growth - 1.0,
        "geometric_excess_ci_low": geometric_low,
        "geometric_excess_ci_high": geometric_high,
        "probability_geometric_excess_positive": float(np.mean(geometric > 0.0)),
        "annualized_mean_daily_excess": observed_mean * 252.0,
        "annualized_mean_daily_excess_ci_low": mean_low,
        "annualized_mean_daily_excess_ci_high": mean_high,
        "one_sided_pvalue_mean_daily_excess_gt_zero": one_sided_p,
        "daily_excess_win_rate": float(np.mean(excess > 0.0)),
    }


def rolling_relative_metrics(
    strategy: pd.Series,
    benchmark: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Calculate trailing compounded excess, beta and information ratio."""
    aligned = pd.concat(
        [strategy.rename("strategy_return"), benchmark.rename("benchmark_return")],
        axis=1,
    ).dropna()
    if window < 2:
        raise ValueError("window must be at least two")
    rows: list[dict[str, float | pd.Timestamp]] = []
    for end in range(window, len(aligned) + 1):
        sample = aligned.iloc[end - window:end]
        strategy_values = sample["strategy_return"]
        benchmark_values = sample["benchmark_return"]
        excess = strategy_values - benchmark_values
        strategy_growth = float((1.0 + strategy_values).prod())
        benchmark_growth = float((1.0 + benchmark_values).prod())
        benchmark_variance = float(benchmark_values.var(ddof=1))
        tracking_error = float(excess.std(ddof=1))
        rows.append({
            "date": sample.index[-1],
            "strategy_total_return": strategy_growth - 1.0,
            "benchmark_total_return": benchmark_growth - 1.0,
            "cumulative_return_spread": strategy_growth - benchmark_growth,
            "geometric_excess_return": strategy_growth / benchmark_growth - 1.0,
            "annualized_mean_daily_excess": float(excess.mean() * 252.0),
            "annualized_tracking_error": tracking_error * np.sqrt(252.0),
            "information_ratio": (
                float(excess.mean() / tracking_error * np.sqrt(252.0))
                if tracking_error > 0 else 0.0
            ),
            "beta": (
                float(strategy_values.cov(benchmark_values) / benchmark_variance)
                if benchmark_variance > 0 else np.nan
            ),
            "correlation": float(strategy_values.corr(benchmark_values)),
        })
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _plot_required_rolling(frame: pd.DataFrame, path: Path, window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = {"adaptive_close": "#2563EB", "adaptive_open": "#D97706"}
    for strategy, group in frame.groupby("strategy", sort=False):
        group = group.sort_values("date")
        color = colors.get(strategy, None)
        axes[0].plot(
            group["date"], group["geometric_excess_return"],
            label=strategy, color=color, lw=1.4,
        )
        axes[1].plot(group["date"], group["beta"], label=strategy, color=color, lw=1.4)
        axes[2].plot(
            group["date"], group["information_ratio"],
            label=strategy, color=color, lw=1.4,
        )
    axes[0].set_title(f"Trailing {window}-day geometric excess")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylabel("Excess")
    axes[1].set_title(f"Trailing {window}-day market beta")
    axes[1].set_ylabel("Beta")
    axes[2].set_title(f"Trailing {window}-day information ratio")
    axes[2].set_ylabel("IR")
    for axis in axes:
        axis.axhline(0.0, color="grey", lw=0.8, ls=":")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_bootstrap_ci(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    active = summary[summary["period"].eq("from_first_execution")].copy()
    labels = [f"{row.factor_set}\n{row.strategy}" for row in active.itertuples()]
    estimate = active["geometric_excess_return"].to_numpy(float)
    low = active["geometric_excess_ci_low"].to_numpy(float)
    high = active["geometric_excess_ci_high"].to_numpy(float)
    x = np.arange(len(active))
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.errorbar(
        x, estimate, yerr=np.vstack([estimate - low, high - estimate]),
        fmt="o", capsize=5, color="#2563EB", ecolor="#64748B",
    )
    axis.axhline(0.0, color="black", lw=0.9, ls=":")
    axis.set_xticks(x, labels)
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_ylabel("Geometric excess return")
    axis.set_title("Task 4 paired moving-block bootstrap: 95% intervals")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_task4_excess_significance(
    output_dir: str | Path = OUTPUT_DIR,
    block_length: int = 5,
    iterations: int = 5000,
    rolling_window: int = 60,
    recent_days: int = 45,
    seed: int = 42,
) -> dict:
    """Evaluate formal and extended Task 4 strategies without changing them."""
    root = Path(output_dir)
    target = root / "task4_excess_significance"
    target.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    rolling_rows: list[pd.DataFrame] = []
    for factor_set, directory in (("required_4", "backtest_required"), ("extended_10", "backtest_strict")):
        for strategy in ("adaptive_close", "adaptive_open"):
            comparison = pd.read_csv(
                root / directory / f"benchmark_comparison_{strategy}_daily.csv",
                index_col=0,
            )
            comparison.index = pd.to_datetime(comparison.index)
            metadata = json.loads(
                (root / directory / f"benchmark_metrics_{strategy}.json")
                .read_text(encoding="utf-8")
            )
            first_execution = pd.Timestamp(metadata["first_execution_date"])
            periods = {
                "full_period": comparison,
                "from_first_execution": comparison.loc[comparison.index >= first_execution],
                f"last_{min(recent_days, len(comparison))}_days": comparison.tail(recent_days),
            }
            for period_name, period in periods.items():
                result = bootstrap_relative_performance(
                    period["strategy_return"], period["benchmark_return"],
                    block_length=block_length, iterations=iterations,
                    seed=seed + len(summary_rows),
                )
                summary_rows.append({
                    "factor_set": factor_set,
                    "strategy": strategy,
                    "period": period_name,
                    "start": period.index.min().strftime("%Y-%m-%d"),
                    "end": period.index.max().strftime("%Y-%m-%d"),
                    **result,
                })
            rolling = rolling_relative_metrics(
                comparison["strategy_return"], comparison["benchmark_return"],
                window=rolling_window,
            ).reset_index()
            rolling.insert(0, "strategy", strategy)
            rolling.insert(0, "factor_set", factor_set)
            rolling_rows.append(rolling)

    summary = pd.DataFrame(summary_rows)
    rolling = pd.concat(rolling_rows, ignore_index=True)
    summary.to_csv(target / "bootstrap_summary.csv", index=False, float_format="%.10f")
    rolling.to_csv(target / "rolling_relative_metrics.csv", index=False, float_format="%.10f")
    required_rolling = rolling[rolling["factor_set"].eq("required_4")]
    _plot_required_rolling(
        required_rolling, target / "required_rolling_excess_and_beta.png", rolling_window,
    )
    _plot_bootstrap_ci(summary, target / "bootstrap_excess_ci.png")
    report = {
        "status": "completed",
        "method": "paired circular moving-block bootstrap of synchronized strategy/market daily returns",
        "selection_policy": "diagnostic only; no strategy or parameter is selected from bootstrap results",
        "block_length": int(block_length),
        "iterations": int(iterations),
        "rolling_window": int(rolling_window),
        "rows": summary.to_dict(orient="records"),
    }
    (target / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report

