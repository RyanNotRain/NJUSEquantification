"""Broad, diagnostic-only extensions across Tasks 1--5.

The module deliberately separates full-data checks from sampled minute checks,
and never uses the reported extension results to replace a formal strategy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import METRICS, OUTPUT_DIR
from .evaluation import compute_ic_series, factor_layering_daily_returns
from .factors import compute_all_factors, load_daily_data, select_required_factors
from .task4_excess_significance import bootstrap_relative_performance
from .task4_strict import run_strict_backtest


def _read_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, format="mixed")
    return frame.astype(float)


def ohlc_violation_count(
    open_: pd.DataFrame, high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
) -> int:
    valid = open_.notna() & high.notna() & low.notna() & close.notna()
    max_open_close = open_.where(open_.ge(close), close)
    min_open_close = open_.where(open_.le(close), close)
    violation = valid & (
        high.lt(max_open_close)
        | low.gt(min_open_close)
        | high.lt(low)
    )
    return int(violation.to_numpy().sum())


def _basic_quality_rows(frames: dict[str, pd.DataFrame], scope: str) -> list[dict]:
    rows: list[dict] = []
    open_, high, low, close = (frames[name] for name in ("open", "high", "low", "close"))
    rows.append({"scope": scope, "check": "ohlc_order", "violations": ohlc_violation_count(open_, high, low, close)})
    for name in ("volume", "trade_count", "amount", "buy_volume", "sell_volume", "buy_amount", "sell_amount"):
        values = frames[name]
        rows.append({
            "scope": scope, "check": f"{name}_nonnegative",
            "violations": int((values < -1e-9).to_numpy().sum()),
        })
    volume_gap = frames["buy_volume"] + frames["sell_volume"] - frames["volume"]
    amount_gap = frames["buy_amount"] + frames["sell_amount"] - frames["amount"]
    rows.extend([
        {"scope": scope, "check": "buy_sell_volume_not_above_total", "violations": int((volume_gap > 1e-8).to_numpy().sum())},
        {"scope": scope, "check": "buy_sell_amount_not_above_total", "violations": int((amount_gap > 1e-6).to_numpy().sum())},
    ])
    return rows


def run_data_quality_extension(
    output_dir: str | Path = OUTPUT_DIR, sampled_days: int = 24,
) -> dict:
    root = Path(output_dir)
    target = root / "data_quality_extension"
    target.mkdir(parents=True, exist_ok=True)
    daily = {name: _read_table(root / "daily" / f"{name}.csv") for name in METRICS}
    rows = _basic_quality_rows(daily, "daily_full")
    dates = sorted(path.stem for path in (root / "minute" / "close").glob("*.csv"))
    positions = np.linspace(0, len(dates) - 1, min(sampled_days, len(dates)), dtype=int)
    selected = [dates[position] for position in sorted(set(positions))]
    minute_missing_rows: list[dict] = []
    for date in selected:
        frames = {name: _read_table(root / "minute" / name / f"{date}.csv") for name in METRICS}
        rows.extend(_basic_quality_rows(frames, f"minute_sample:{date}"))
        waiting = frames["close"].between_time("14:57", "14:59")
        waiting_flow = sum(
            frames[name].between_time("14:57", "14:59").fillna(0).abs().to_numpy().sum()
            for name in ("volume", "trade_count", "amount", "buy_volume", "sell_volume", "buy_amount", "sell_amount")
        )
        minute_missing_rows.append({
            "date": date,
            "bars": int(len(frames["close"])),
            "stocks": int(frames["close"].shape[1]),
            "waiting_price_nonmissing": int(waiting.notna().to_numpy().sum()),
            "waiting_flow_absolute_sum": float(waiting_flow),
            "close_1500_vs_daily_max_abs_error": float(
                (frames["close"].iloc[-1] - daily["close"].loc[pd.Timestamp(date)]).abs().max()
            ),
        })
    checks = pd.DataFrame(rows)
    minute_summary = pd.DataFrame(minute_missing_rows)
    checks.to_csv(target / "quality_checks.csv", index=False)
    minute_summary.to_csv(target / "sampled_minute_summary.csv", index=False, float_format="%.10f")
    summary = {
        "status": "completed",
        "daily_scope": "all 302 dates x 300 stocks",
        "minute_scope": f"{len(selected)} evenly spaced dates x 300 stocks",
        "sampled_dates": selected,
        "total_rule_violations": int(checks["violations"].sum()),
        "all_sampled_minutes_have_242_bars": bool(minute_summary["bars"].eq(242).all()),
        "max_1500_close_error": float(minute_summary["close_1500_vs_daily_max_abs_error"].max()),
        "selection_policy": "diagnostic only; sampled minute checks are not described as a full minute-file audit",
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def factor_horizon_row(
    factor_name: str, factor: pd.DataFrame, forward: pd.DataFrame, horizon: int,
) -> dict:
    pearson = compute_ic_series(factor, forward, "pearson")
    rank = compute_ic_series(factor, forward, "spearman")
    layers = factor_layering_daily_returns(factor, forward, 5)
    spread = layers["Q1"] - layers["Q5"]
    return {
        "factor": factor_name, "horizon_days": horizon,
        "pearson_ic": float(pearson.mean()), "rank_ic": float(rank.mean()),
        "rank_ic_t_stat": float(rank.mean() / (rank.std(ddof=1) / np.sqrt(len(rank)))) if len(rank) > 1 and rank.std(ddof=1) > 0 else np.nan,
        "q1_minus_q5_mean": float(spread.mean()),
        "q1_minus_q5_t_stat": float(spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))) if len(spread) > 1 and spread.std(ddof=1) > 0 else np.nan,
        "ic_days": int(len(rank)), "layer_days": int(len(spread)),
    }


def run_factor_horizon_extension(output_dir: str | Path = OUTPUT_DIR) -> dict:
    root = Path(output_dir)
    target = root / "factor_horizon_extension"
    target.mkdir(parents=True, exist_ok=True)
    daily = load_daily_data(root / "daily")
    bundle = compute_all_factors(daily)
    factors = select_required_factors(bundle)
    rows = []
    for horizon in (1, 2, 5, 10):
        forward = daily["close"].shift(-horizon).div(daily["close"]).sub(1.0)
        # Match the formal Task 3 convention: both the signal date and the
        # realization date must be tradable. Forward-filled halt prices must
        # not enter IC or layering as artificial zero returns.
        tradable = (daily["volume"] > 0) & (daily["volume"].shift(-horizon) > 0)
        forward = forward.where(tradable)
        for name, factor in factors.items():
            rows.append(factor_horizon_row(name, factor, forward, horizon))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(target / "horizon_metrics.csv", index=False, float_format="%.10f")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, group in metrics.groupby("factor", sort=False):
        axes[0].plot(group["horizon_days"], group["rank_ic"], marker="o", label=name)
        axes[1].plot(group["horizon_days"], group["q1_minus_q5_mean"], marker="o", label=name)
    axes[0].axhline(0, color="grey", ls=":"); axes[1].axhline(0, color="grey", ls=":")
    axes[0].set(title="Required factors: Rank IC by horizon", xlabel="Forward horizon (days)", ylabel="Rank IC")
    axes[1].set(title="Required factors: Q1-Q5 by horizon", xlabel="Forward horizon (days)", ylabel="Mean return spread")
    for axis in axes: axis.grid(alpha=.25); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "factor_horizon_decay.png", dpi=180); plt.close(fig)
    summary = {
        "status": "completed", "horizons": [1, 2, 5, 10], "factor_count": len(factors),
        "rows": len(metrics),
        "rank_ic_sign_consistency": {
            name: bool(np.sign(group["rank_ic"]).nunique() == 1)
            for name, group in metrics.groupby("factor")
        },
        "selection_policy": "diagnostic only; no horizon is selected from test performance",
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def fold_relative_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    s_growth = float((1 + aligned["strategy"]).prod())
    b_growth = float((1 + aligned["benchmark"]).prod())
    excess = aligned["strategy"] - aligned["benchmark"]
    volatility = float(aligned["strategy"].std(ddof=1))
    nav = (1 + aligned["strategy"]).cumprod()
    return {
        "observations": int(len(aligned)), "strategy_total_return": s_growth - 1,
        "benchmark_total_return": b_growth - 1, "geometric_excess_return": s_growth / b_growth - 1,
        "information_ratio": float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else 0.0,
        "sharpe_ratio": float(aligned["strategy"].mean() / volatility * np.sqrt(252)) if volatility > 0 else 0.0,
        "max_drawdown": float(nav.div(nav.cummax()).sub(1).min()),
    }


def run_task4_chronological_extension(output_dir: str | Path = OUTPUT_DIR) -> dict:
    root = Path(output_dir)
    target = root / "task4_chronological_extension"
    target.mkdir(parents=True, exist_ok=True)
    daily = load_daily_data(root / "daily")
    factors = select_required_factors(compute_all_factors(daily))
    fold_rows, config_rows = [], []
    for execution in ("close", "open"):
        benchmark = daily[execution].pct_change(fill_method=None).mean(axis=1, skipna=True)
        for lookback in (40, 60, 90):
            for top_n in (5, 10, 20):
                result = run_strict_backtest(
                    factors, daily, execution=execution, method="adaptive",
                    lookback=lookback, top_n=top_n,
                )
                first = pd.Timestamp(result["selections"]["execution_date"].min())
                dates = result["daily_return"].loc[result["daily_return"].index >= first].index
                folds = np.array_split(np.arange(len(dates)), 4)
                current_rows = []
                for fold_number, positions in enumerate(folds, 1):
                    fold_dates = dates[positions]
                    metrics = fold_relative_metrics(result["daily_return"].reindex(fold_dates), benchmark.reindex(fold_dates))
                    row = {"execution": execution, "lookback": lookback, "top_n": top_n, "fold": fold_number,
                           "start": fold_dates.min(), "end": fold_dates.max(), **metrics}
                    fold_rows.append(row); current_rows.append(row)
                frame = pd.DataFrame(current_rows)
                config_rows.append({
                    "execution": execution, "lookback": lookback, "top_n": top_n,
                    "positive_excess_folds": int((frame["geometric_excess_return"] > 0).sum()),
                    "median_fold_geometric_excess": float(frame["geometric_excess_return"].median()),
                    "worst_fold_geometric_excess": float(frame["geometric_excess_return"].min()),
                    "median_fold_sharpe": float(frame["sharpe_ratio"].median()),
                })
    folds = pd.DataFrame(fold_rows); configs = pd.DataFrame(config_rows)
    folds.to_csv(target / "fold_metrics.csv", index=False, float_format="%.10f")
    configs.to_csv(target / "configuration_summary.csv", index=False, float_format="%.10f")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, execution in zip(axes, ("close", "open")):
        canonical = folds[(folds.execution == execution) & (folds.lookback == 60) & (folds.top_n == 10)]
        axis.bar(canonical["fold"].astype(str), canonical["geometric_excess_return"], color="#2563EB" if execution == "close" else "#D97706")
        axis.axhline(0, color="black", ls=":"); axis.grid(axis="y", alpha=.25)
        axis.set(title=f"Canonical adaptive_{execution}: chronological folds", xlabel="Non-overlapping fold", ylabel="Geometric excess")
    fig.tight_layout(); fig.savefig(target / "canonical_fold_excess.png", dpi=180); plt.close(fig)
    canonical_configs = configs[(configs.lookback == 60) & (configs.top_n == 10)].set_index("execution")
    summary = {
        "status": "completed", "configurations": int(len(configs)), "fold_rows": int(len(folds)),
        "fold_policy": "four non-overlapping chronological folds after each configuration's first execution",
        "canonical_positive_folds": canonical_configs["positive_excess_folds"].astype(int).to_dict(),
        "all_configuration_positive_fold_distribution": configs["positive_excess_folds"].value_counts().sort_index().astype(int).to_dict(),
        "selection_policy": "sensitivity diagnostic only; no configuration is promoted using these results",
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    recalls, f1s = [], []
    for label in (0, 1, 2):
        tp = np.sum((y_true == label) & (y_pred == label))
        actual = np.sum(y_true == label); predicted = np.sum(y_pred == label)
        recall = tp / actual if actual else 0.0; precision = tp / predicted if predicted else 0.0
        recalls.append(recall); f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float(np.mean(y_true == y_pred)), "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s))}


def cluster_bootstrap_classification(
    predictions: pd.DataFrame, iterations: int = 5000, seed: int = 42,
) -> pd.DataFrame:
    dates = sorted(predictions["date"].astype(str).unique())
    groups = {date: predictions[predictions["date"].astype(str).eq(date)] for date in dates}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(iterations):
        sample_dates = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat([groups[date] for date in sample_dates], ignore_index=True)
        rows.append(classification_metrics(sample["true_label"].to_numpy(), sample["predicted_label"].to_numpy()))
    return pd.DataFrame(rows)


def run_task5_uncertainty_extension(
    output_dir: str | Path = OUTPUT_DIR, iterations: int = 5000,
) -> dict:
    root = Path(output_dir); target = root / "task5_uncertainty_extension"; target.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(root / "lstm_ensemble" / "test_predictions.csv")
    point = classification_metrics(predictions["true_label"].to_numpy(), predictions["predicted_label"].to_numpy())
    boot = cluster_bootstrap_classification(predictions, iterations=iterations)
    rows = []
    majority = float(predictions["true_label"].value_counts(normalize=True).max())
    for metric, estimate in point.items():
        low, high = boot[metric].quantile([.025, .975])
        rows.append({"metric": metric, "estimate": estimate, "ci_low": low, "ci_high": high,
                     "probability_above_majority_baseline": float((boot[metric] > majority).mean()) if metric == "accuracy" else np.nan})
    classification = pd.DataFrame(rows)
    classification.to_csv(target / "classification_day_cluster_bootstrap.csv", index=False, float_format="%.10f")
    daily = pd.read_csv(root / "lstm_ensemble" / "t1_strategy_returns.csv", index_col=0)
    strategy_rows = []
    for tier in ("balanced_up", "strict_up"):
        result = bootstrap_relative_performance(
            daily[f"{tier}__fee_5bp__net_return"],
            daily[f"{tier}__fee_5bp__exposure_matched_market_return"],
            block_length=2, iterations=iterations, seed=43 if tier == "balanced_up" else 44,
        )
        strategy_rows.append({"strategy": tier, **result})
    strategy = pd.DataFrame(strategy_rows)
    strategy.to_csv(target / "t1_strategy_bootstrap.csv", index=False, float_format="%.10f")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].errorbar(np.arange(len(classification)), classification.estimate,
                     yerr=np.vstack([classification.estimate-classification.ci_low, classification.ci_high-classification.estimate]), fmt="o", capsize=5)
    axes[0].axhline(majority, color="grey", ls=":", label="majority accuracy")
    axes[0].set_xticks(np.arange(len(classification)), classification.metric, rotation=15); axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=.25)
    axes[0].set_title("Day-cluster bootstrap: classification")
    axes[1].errorbar(np.arange(len(strategy)), strategy.geometric_excess_return,
                     yerr=np.vstack([strategy.geometric_excess_return-strategy.geometric_excess_ci_low, strategy.geometric_excess_ci_high-strategy.geometric_excess_return]), fmt="o", capsize=5)
    axes[1].axhline(0, color="grey", ls=":"); axes[1].set_xticks(np.arange(len(strategy)), strategy.strategy); axes[1].grid(axis="y", alpha=.25)
    axes[1].set_title("T+1 matched-market excess: 95% intervals")
    fig.tight_layout(); fig.savefig(target / "task5_uncertainty.png", dpi=180); plt.close(fig)
    summary = {
        "status": "completed", "iterations": iterations, "classification_cluster": "test trading date",
        "classification_days": int(predictions["date"].nunique()), "t1_settled_days": int(len(daily)),
        "accuracy_probability_above_majority": float(classification.loc[classification.metric.eq("accuracy"), "probability_above_majority_baseline"].iloc[0]),
        "known_limitation": "only 10 classification dates and 9 settled T+1 dates; intervals are necessarily wide",
    }
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def run_all_comprehensive_extensions(output_dir: str | Path = OUTPUT_DIR) -> dict:
    return {
        "data_quality": run_data_quality_extension(output_dir),
        "factor_horizons": run_factor_horizon_extension(output_dir),
        "task4_chronological": run_task4_chronological_extension(output_dir),
        "task5_uncertainty": run_task5_uncertainty_extension(output_dir),
    }
