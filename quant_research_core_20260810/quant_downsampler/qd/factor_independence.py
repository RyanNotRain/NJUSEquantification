"""Leakage-aware factor de-redundancy and strict Task 4 comparison.

The first calibration block freezes correlation clusters, representatives, and
the orthogonalization order. Trading starts only after that block. No return
realized after the freeze date participates in any factor-set decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .factor_robustness import average_factor_correlation
from .factors import compute_all_factors, load_daily_data
from .strategy_analysis import relative_metrics
from .task4_strict import build_signal_target, compute_daily_ic, run_strict_backtest


def correlation_clusters(
    correlation: pd.DataFrame,
    threshold: float = 0.60,
) -> list[list[str]]:
    """Connected components under an absolute-correlation threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    names = [str(name) for name in correlation.index]
    if set(names) != set(map(str, correlation.columns)):
        raise ValueError("correlation matrix must have matching rows and columns")
    matrix = correlation.copy()
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    unseen = set(names)
    clusters: list[list[str]] = []
    for root in names:
        if root not in unseen:
            continue
        stack = [root]
        unseen.remove(root)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbours = [
                candidate for candidate in names
                if candidate in unseen
                and np.isfinite(matrix.loc[current, candidate])
                and abs(float(matrix.loc[current, candidate])) >= threshold
            ]
            for candidate in neighbours:
                unseen.remove(candidate)
                stack.append(candidate)
        clusters.append(sorted(component, key=names.index))
    return clusters


def historical_ic_quality(
    raw_ic: pd.DataFrame,
    freeze_date: str | pd.Timestamp,
    realization_lag: int = 2,
) -> pd.DataFrame:
    """Score factors using only IC observations fully realized by freeze_date."""
    available = raw_ic.shift(realization_lag).loc[:pd.Timestamp(freeze_date)]
    rows: list[dict[str, object]] = []
    for factor in available:
        values = available[factor].replace([np.inf, -np.inf], np.nan).dropna()
        mean = float(values.mean()) if len(values) else np.nan
        std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        score = abs(mean) / std if np.isfinite(std) and std > 0 else 0.0
        rows.append({
            "factor": str(factor),
            "available_ic_days": int(len(values)),
            "mean_ic": mean,
            "ic_std": std,
            "absolute_icir": float(score),
        })
    return pd.DataFrame(rows).set_index("factor")


def select_cluster_representatives(
    clusters: Sequence[Sequence[str]],
    quality: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """Select one frozen representative per cluster by historical absolute ICIR."""
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for cluster_id, members_value in enumerate(clusters, 1):
        members = [str(member) for member in members_value]
        ranked = sorted(
            members,
            key=lambda name: (-float(quality.loc[name, "absolute_icir"]), name),
        )
        representative = ranked[0]
        selected.append(representative)
        for member in members:
            rows.append({
                "cluster_id": cluster_id,
                "factor": member,
                "representative": representative,
                "selected": member == representative,
                **quality.loc[member].to_dict(),
            })
    return selected, pd.DataFrame(rows)


def orthogonalize_cross_sectionally(
    factors: Mapping[str, pd.DataFrame],
    order: Sequence[str],
    min_stocks: int = 30,
) -> dict[str, pd.DataFrame]:
    """Create daily rank exposures and Gram-Schmidt residual components.

    The order is frozen before the trading interval. Each date uses only that
    date's factor cross-section; no return label enters this transformation.
    """
    names = [str(name) for name in order]
    if set(names) != set(map(str, factors)):
        raise ValueError("order must contain every factor exactly once")
    template = next(iter(factors.values()))
    outputs = {
        f"orth_{name}": pd.DataFrame(np.nan, index=template.index, columns=template.columns)
        for name in names
    }
    for date in template.index:
        panel = pd.DataFrame({name: factors[name].loc[date] for name in names})
        complete = panel.replace([np.inf, -np.inf], np.nan).dropna()
        if len(complete) < min_stocks:
            continue
        ranked = complete.rank(method="average", pct=True)
        standardized = (ranked - ranked.mean()).div(ranked.std(ddof=1).replace(0.0, np.nan))
        if standardized.isna().any().any():
            continue
        basis: list[np.ndarray] = []
        for name in names:
            residual = standardized[name].to_numpy(dtype=float).copy()
            for unit in basis:
                residual -= float(np.dot(residual, unit)) * unit
            norm = float(np.linalg.norm(residual))
            if norm <= 1e-10:
                continue
            unit = residual / norm
            basis.append(unit)
            component = residual / (float(np.std(residual, ddof=1)) + 1e-12)
            outputs[f"orth_{name}"].loc[date, complete.index] = component
    return outputs


def independence_metrics(correlation: pd.DataFrame) -> dict[str, float | int]:
    """Summarize off-diagonal dependence and spectral effective rank."""
    matrix = correlation.to_numpy(dtype=float)
    matrix = np.nan_to_num((matrix + matrix.T) / 2.0, nan=0.0)
    np.fill_diagonal(matrix, 1.0)
    offdiag = np.abs(matrix[np.triu_indices_from(matrix, k=1)])
    eigenvalues = np.clip(np.linalg.eigvalsh(matrix), 0.0, None)
    probabilities = eigenvalues / eigenvalues.sum() if eigenvalues.sum() > 0 else eigenvalues
    positive = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum())) if len(positive) else 0.0
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    condition = (
        float(positive_eigenvalues.max() / positive_eigenvalues.min())
        if len(positive_eigenvalues) else np.inf
    )
    return {
        "factor_count": int(len(correlation)),
        "mean_abs_off_diagonal_correlation": float(offdiag.mean()) if len(offdiag) else 0.0,
        "max_abs_off_diagonal_correlation": float(offdiag.max()) if len(offdiag) else 0.0,
        "pairs_abs_correlation_ge_0_60": int((offdiag >= 0.60).sum()),
        "effective_rank": effective_rank,
        "condition_number": condition,
    }


def _evaluation_metrics(
    result: dict,
    benchmark: pd.Series,
    evaluation_start: pd.Timestamp,
) -> dict[str, float | int]:
    strategy = result["daily_return"].loc[result["daily_return"].index >= evaluation_start]
    relative = relative_metrics(strategy, benchmark.reindex(strategy.index))
    return {
        "observations": relative["strategy"]["observations"],
        "strategy_total_return": relative["strategy"]["total_return"],
        "market_total_return": relative["benchmark"]["total_return"],
        "geometric_excess_return": relative["geometric_excess_return"],
        "sharpe_ratio": relative["strategy"]["sharpe"],
        "information_ratio": relative["information_ratio"],
        "max_drawdown": relative["strategy"]["max_drawdown"],
        "average_sell_turnover": float(
            result["turnover"].loc[result["turnover"].index >= evaluation_start].mean()
        ),
    }


def _plot_comparison(daily_returns: pd.DataFrame, target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    nav = (1.0 + daily_returns).cumprod()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for column in nav:
        axes[0].plot(nav.index, nav[column], label=column, lw=1.3)
    axes[0].set(title="Post-freeze NAV comparison", ylabel="NAV", xlabel="Date")
    axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=8)
    excess = nav.drop(columns="market").div(nav["market"], axis=0)
    for column in excess:
        axes[1].plot(excess.index, excess[column], label=column, lw=1.3)
    axes[1].axhline(1.0, color="grey", ls=":", lw=1)
    axes[1].set(title="Geometric excess NAV vs market", ylabel="Relative NAV", xlabel="Date")
    axes[1].grid(alpha=.3)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(target, dpi=160)
    plt.close(fig)


def run_factor_independence(
    output_dir: str | Path = OUTPUT_DIR,
    calibration_days: int = 120,
    correlation_threshold: float = 0.60,
    lookback: int = 60,
    top_n: int = 10,
    sell_fee_bps: float = 5.0,
) -> dict[str, object]:
    """Freeze factor sets, compare dependence, and run matched strict backtests."""
    root = Path(output_dir)
    target = root / "factor_independence"
    target.mkdir(parents=True, exist_ok=True)
    daily = load_daily_data(root / "daily")
    bundle = compute_all_factors(daily)
    raw = {name: value for name, value in bundle.items() if name != "forward_return_1d"}
    dates = pd.DatetimeIndex(daily["close"].index)
    if calibration_days < max(40, lookback) or calibration_days + 2 >= len(dates):
        raise ValueError("calibration_days must leave a non-trivial post-freeze interval")
    freeze_date = dates[calibration_days - 1]
    trade_start = dates[calibration_days]
    evaluation_start = dates[calibration_days + 1]

    calibration_factors = {name: frame.loc[:freeze_date] for name, frame in raw.items()}
    calibration_correlation = average_factor_correlation(calibration_factors)
    clusters = correlation_clusters(calibration_correlation, correlation_threshold)
    target_return = build_signal_target(daily["close"])
    raw_ic = compute_daily_ic(raw, target_return, daily["volume"])
    quality = historical_ic_quality(raw_ic, freeze_date)
    selected, cluster_table = select_cluster_representatives(clusters, quality)
    pruned = {name: raw[name] for name in selected}
    order = quality.sort_values(
        ["absolute_icir"], ascending=False, kind="stable"
    ).index.astype(str).tolist()
    orthogonal = orthogonalize_cross_sectionally(raw, order)

    sets: dict[str, dict[str, pd.DataFrame]] = {
        "raw_10": raw,
        "cluster_pruned": pruned,
        "orthogonalized_10": orthogonal,
    }
    evaluation_dates = dates[dates >= trade_start]
    correlation_outputs: dict[str, pd.DataFrame] = {}
    independence_rows: list[dict[str, object]] = []
    results: dict[str, dict] = {}
    metric_rows: list[dict[str, object]] = []
    benchmark = daily["close"].pct_change(fill_method=None).mean(axis=1, skipna=True)
    comparison_returns: dict[str, pd.Series] = {}
    for label, factor_set in sets.items():
        evaluation_factors = {name: frame.loc[evaluation_dates] for name, frame in factor_set.items()}
        correlation = average_factor_correlation(evaluation_factors)
        correlation_outputs[label] = correlation
        independence_rows.append({"factor_set": label, **independence_metrics(correlation)})
        result = run_strict_backtest(
            factor_set,
            daily,
            execution="close",
            method="adaptive",
            lookback=lookback,
            top_n=top_n,
            sell_fee=sell_fee_bps / 10_000.0,
            trade_start_date=trade_start,
        )
        results[label] = result
        metrics = _evaluation_metrics(result, benchmark, evaluation_start)
        metric_rows.append({"factor_set": label, **metrics})
        comparison_returns[label] = result["daily_return"].loc[
            result["daily_return"].index >= evaluation_start
        ]

    independence = pd.DataFrame(independence_rows)
    backtest_metrics = pd.DataFrame(metric_rows).sort_values(
        "geometric_excess_return", ascending=False
    )
    comparison = pd.DataFrame(comparison_returns)
    comparison["market"] = benchmark.reindex(comparison.index)

    calibration_correlation.to_csv(
        target / "calibration_correlation.csv", float_format="%.10f"
    )
    cluster_table.to_csv(target / "clusters.csv", index=False, float_format="%.10f")
    quality.reset_index().to_csv(target / "historical_ic_quality.csv", index=False, float_format="%.10f")
    pd.DataFrame({"position": range(1, len(order) + 1), "factor": order}).to_csv(
        target / "orthogonalization_order.csv", index=False
    )
    for label, correlation in correlation_outputs.items():
        correlation.to_csv(target / f"evaluation_correlation_{label}.csv", float_format="%.10f")
    independence.to_csv(target / "independence_metrics.csv", index=False, float_format="%.10f")
    backtest_metrics.to_csv(target / "backtest_metrics.csv", index=False, float_format="%.10f")
    comparison.to_csv(target / "daily_returns.csv", float_format="%.10f")
    for label, result in results.items():
        result["selections"].to_csv(target / f"{label}_selections.csv", index=False)
        result["ic_weights"].to_csv(target / f"{label}_ic_weights.csv", float_format="%.10f")
    _plot_comparison(comparison, target / "comparison.png")

    raw_metrics = independence.set_index("factor_set").loc["raw_10"]
    pruned_metrics = independence.set_index("factor_set").loc["cluster_pruned"]
    orth_metrics = independence.set_index("factor_set").loc["orthogonalized_10"]
    best = backtest_metrics.iloc[0]
    summary: dict[str, object] = {
        "status": "completed",
        "selection_status": "frozen_before_post_calibration_backtest",
        "calibration_days": calibration_days,
        "calibration_start": str(dates[0].date()),
        "freeze_date": str(freeze_date.date()),
        "trade_start": str(trade_start.date()),
        "evaluation_start": str(evaluation_start.date()),
        "correlation_threshold": correlation_threshold,
        "realization_lag_days_for_ic_selection": 2,
        "raw_factor_count": len(raw),
        "cluster_count": len(clusters),
        "selected_factors": selected,
        "orthogonalization_order": order,
        "post_freeze_independence": {
            "raw_max_abs_correlation": float(raw_metrics["max_abs_off_diagonal_correlation"]),
            "pruned_max_abs_correlation": float(pruned_metrics["max_abs_off_diagonal_correlation"]),
            "orthogonalized_max_abs_correlation": float(orth_metrics["max_abs_off_diagonal_correlation"]),
            "raw_effective_rank": float(raw_metrics["effective_rank"]),
            "pruned_effective_rank": float(pruned_metrics["effective_rank"]),
            "orthogonalized_effective_rank": float(orth_metrics["effective_rank"]),
        },
        "best_post_freeze_by_geometric_excess": {
            "factor_set": str(best["factor_set"]),
            "strategy_total_return": float(best["strategy_total_return"]),
            "market_total_return": float(best["market_total_return"]),
            "geometric_excess_return": float(best["geometric_excess_return"]),
        },
        "neutralization": (
            "industry/market-cap neutralization remains unavailable because genuine exposure "
            "tables are not supplied; no proxy is fabricated"
        ),
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
