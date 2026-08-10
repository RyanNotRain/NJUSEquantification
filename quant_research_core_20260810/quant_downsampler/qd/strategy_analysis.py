"""Benchmark-relative Task 4 analysis and execution-aware LSTM diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR


def performance_metrics(returns: pd.Series, periods_per_year: float = 252.0) -> dict[str, float | int]:
    """Return compact compounded performance statistics for a return series."""
    values = pd.Series(returns, dtype=np.float64).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        raise ValueError("returns cannot be empty")
    nav = (1.0 + values).cumprod()
    years = len(values) / periods_per_year
    volatility = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "observations": int(len(values)),
        "total_return": float(nav.iloc[-1] - 1.0),
        "cagr": float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan,
        "annualized_volatility": float(volatility * np.sqrt(periods_per_year)),
        "sharpe": float(values.mean() / volatility * np.sqrt(periods_per_year))
        if volatility > 0 else 0.0,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
    }


def relative_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, float | int | dict]:
    """Compare a daily strategy with a synchronized daily benchmark."""
    aligned = pd.concat([strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 2:
        raise ValueError("strategy and benchmark need at least two aligned observations")
    excess = aligned["strategy"] - aligned["benchmark"]
    benchmark_variance = float(aligned["benchmark"].var(ddof=1))
    beta = (
        float(aligned["strategy"].cov(aligned["benchmark"]) / benchmark_variance)
        if benchmark_variance > 0 else np.nan
    )
    tracking_error = float(excess.std(ddof=1))
    alpha_daily = float(aligned["strategy"].mean() - beta * aligned["benchmark"].mean())
    strategy_growth = float((1.0 + aligned["strategy"]).prod())
    benchmark_growth = float((1.0 + aligned["benchmark"]).prod())
    return {
        "strategy": performance_metrics(aligned["strategy"]),
        "benchmark": performance_metrics(aligned["benchmark"]),
        "geometric_excess_return": float(strategy_growth / benchmark_growth - 1.0),
        "information_ratio": float(excess.mean() / tracking_error * np.sqrt(252.0))
        if tracking_error > 0 else 0.0,
        "beta": beta,
        "annualized_arithmetic_alpha": float(alpha_daily * 252.0),
    }


def run_task4_benchmark_analysis(
    output_dir: str | Path = OUTPUT_DIR,
    strategy_name: str = "adaptive_close",
    recent_days: int = 45,
) -> dict:
    """Build a daily-rebalanced equal-weight benchmark for the strict Task 4 result."""
    root = Path(output_dir)
    close = pd.read_csv(root / "daily" / "close.csv", index_col=0)
    close.index = pd.to_datetime(close.index)
    benchmark_return = close.pct_change(fill_method=None).mean(axis=1, skipna=True)
    benchmark_return.name = "benchmark_return"

    strategy_path = root / "backtest_strict" / f"{strategy_name}_daily.csv"
    strategy_frame = pd.read_csv(strategy_path, index_col=0)
    strategy_frame.index = pd.to_datetime(strategy_frame.index)
    comparison = pd.concat(
        [strategy_frame["daily_return"].rename("strategy_return"), benchmark_return], axis=1
    ).dropna()
    comparison["excess_return"] = comparison["strategy_return"] - comparison["benchmark_return"]
    comparison["strategy_nav"] = (1.0 + comparison["strategy_return"]).cumprod()
    comparison["benchmark_nav"] = (1.0 + comparison["benchmark_return"]).cumprod()
    comparison["excess_nav"] = comparison["strategy_nav"] / comparison["benchmark_nav"]

    recent_count = min(int(recent_days), len(comparison))
    selections_path = root / "backtest_strict" / f"{strategy_name}_selections.csv"
    selections = pd.read_csv(selections_path)
    first_execution = pd.to_datetime(selections["execution_date"]).min()
    active_period = comparison.loc[comparison.index >= first_execution]
    summary = {
        "benchmark_definition": (
            "Daily-rebalanced equal weight across all 300 adjusted close series; "
            "forward-filled halted prices contribute zero return. Benchmark trading costs are omitted."
        ),
        "strategy": strategy_name,
        "full_period": relative_metrics(
            comparison["strategy_return"], comparison["benchmark_return"]
        ),
        "from_first_execution": relative_metrics(
            active_period["strategy_return"], active_period["benchmark_return"]
        ),
        "first_execution_date": first_execution.strftime("%Y-%m-%d"),
        f"last_{recent_count}_days": relative_metrics(
            comparison["strategy_return"].tail(recent_count),
            comparison["benchmark_return"].tail(recent_count),
        ),
        "recent_start": comparison.index[-recent_count].strftime("%Y-%m-%d"),
        "recent_end": comparison.index[-1].strftime("%Y-%m-%d"),
    }

    target = root / "backtest_strict"
    comparison.to_csv(target / "benchmark_comparison_daily.csv", float_format="%.10f")
    (target / "benchmark_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _plot_task4_benchmark(comparison, target / "benchmark_comparison.png")
    return summary


def _load_lstm_execution_returns(predictions: pd.DataFrame, minute_dir: Path) -> pd.DataFrame:
    """Attach label, price-time proxy, and A-share T+1 return horizons."""
    frame = predictions.copy().reset_index(drop=True)
    for column in ("window_end", "target_time"):
        frame[column] = pd.to_datetime(frame[column])
    ideal = np.empty(len(frame), dtype=np.float64)
    feasible = np.empty(len(frame), dtype=np.float64)
    t1_return = np.full(len(frame), np.nan, dtype=np.float64)
    date_keys = sorted(path.stem for path in (minute_dir / "open").glob("*.csv"))
    next_date = {date_keys[i]: date_keys[i + 1] for i in range(len(date_keys) - 1)}

    for date, positions in frame.groupby("date", sort=True).groups.items():
        key = str(date).replace("-", "")
        open_table = pd.read_csv(minute_dir / "open" / f"{key}.csv", index_col=0)
        close_table = pd.read_csv(minute_dir / "close" / f"{key}.csv", index_col=0)
        open_table.index = pd.to_datetime(open_table.index)
        close_table.index = pd.to_datetime(close_table.index)
        next_open_table = None
        if key in next_date:
            next_open_table = pd.read_csv(
                minute_dir / "open" / f"{next_date[key]}.csv", index_col=0
            )
            next_open_table.index = pd.to_datetime(next_open_table.index)
        for position in positions:
            row = frame.loc[position]
            stock = str(row["stock"])
            current_close = float(close_table.at[row["window_end"], stock])
            target_open = float(open_table.at[row["target_time"], stock])
            target_close = float(close_table.at[row["target_time"], stock])
            ideal[position] = target_close / current_close - 1.0
            feasible[position] = target_close / target_open - 1.0
            if next_open_table is not None:
                exit_time = pd.Timestamp.combine(
                    next_open_table.index[0].date(), row["target_time"].time()
                )
                exit_open = float(next_open_table.at[exit_time, stock])
                t1_return[position] = exit_open / target_open - 1.0

    frame["same_close_label_return"] = ideal
    frame["next_minute_open_to_close_return"] = feasible
    frame["next_day_same_minute_open_return"] = t1_return
    return frame


def aggregate_signal_strategy(
    samples: pd.DataFrame,
    selected: pd.Series,
    return_column: str,
    direction: str = "long",
    sell_fee_bps: float = 0.0,
) -> pd.DataFrame:
    """Aggregate one-minute independent trades into a portfolio return path.

    Long strategies invest equally in selected predicted-up stocks and otherwise
    hold cash.  Long-short is a diagnostic only and takes equal signed exposure
    to every selected non-flat prediction.
    """
    if sell_fee_bps < 0:
        raise ValueError("sell_fee_bps cannot be negative")
    mask = pd.Series(selected, index=samples.index).fillna(False).astype(bool)
    chosen = samples.loc[mask].copy()
    if direction == "long":
        chosen["signed_return"] = chosen[return_column]
    elif direction == "long_short":
        sign = np.where(chosen["predicted_label"].eq(2), 1.0, -1.0)
        chosen["signed_return"] = sign * chosen[return_column]
    else:
        raise ValueError("direction must be 'long' or 'long_short'")

    times = pd.DatetimeIndex(sorted(pd.to_datetime(samples["target_time"]).unique()))
    gross = chosen.groupby("target_time")["signed_return"].mean().reindex(times, fill_value=0.0)
    active = chosen.groupby("target_time").size().reindex(times, fill_value=0).gt(0)
    fee = active.astype(np.float64) * (float(sell_fee_bps) / 10_000.0)
    result = pd.DataFrame({
        "gross_return": gross,
        "sell_fee": fee,
        "net_return": gross - fee,
        "active": active.astype(np.int8),
    }, index=times)
    result.index.name = "target_time"
    return result


def aggregate_t1_daily_strategy(
    samples: pd.DataFrame,
    selected: pd.Series,
    sell_fee_bps: float = 0.0,
) -> pd.DataFrame:
    """Aggregate T+1 trades as equal capital sleeves across intraday signal times.

    Each target minute is one capital sleeve. It buys selected predicted-up stocks
    at that minute's open and exits at the same minute's open on the next trading
    day. An inactive sleeve holds cash. Daily cohort returns are the equal-weight
    mean across sleeves, so overlapping signals do not reuse the same capital.
    """
    if sell_fee_bps < 0:
        raise ValueError("sell_fee_bps cannot be negative")
    valid = samples.dropna(subset=["next_day_same_minute_open_return"]).copy()
    mask = pd.Series(selected, index=samples.index).reindex(valid.index).fillna(False).astype(bool)
    bars = pd.MultiIndex.from_frame(
        valid[["date", "target_time"]].drop_duplicates(), names=["date", "target_time"]
    )
    chosen = valid.loc[mask]
    gross = (
        chosen.groupby(["date", "target_time"])["next_day_same_minute_open_return"]
        .mean()
        .reindex(bars, fill_value=0.0)
    )
    active = (
        chosen.groupby(["date", "target_time"]).size().reindex(bars, fill_value=0).gt(0)
    )
    market = (
        valid.groupby(["date", "target_time"])["next_day_same_minute_open_return"]
        .mean()
        .reindex(bars)
    )
    bar_path = pd.DataFrame({
        "gross_return": gross,
        "sell_fee": active.astype(np.float64) * (float(sell_fee_bps) / 10_000.0),
        "active": active.astype(np.int8),
        "five_stock_market_return": market,
    }, index=bars)
    bar_path["net_return"] = bar_path["gross_return"] - bar_path["sell_fee"]
    bar_path["exposure_matched_market_return"] = (
        bar_path["five_stock_market_return"] * bar_path["active"]
    )
    daily = bar_path.groupby(level="date").mean()
    daily.index.name = "signal_date"
    return daily


def _minute_strategy_metrics(path: pd.DataFrame) -> dict[str, float | int]:
    gross_nav = (1.0 + path["gross_return"]).cumprod()
    net_nav = (1.0 + path["net_return"]).cumprod()
    net_std = float(path["net_return"].std(ddof=1))
    return {
        "bars": int(len(path)),
        "active_bars": int(path["active"].sum()),
        "bar_coverage": float(path["active"].mean()),
        "gross_total_return": float(gross_nav.iloc[-1] - 1.0),
        "net_total_return": float(net_nav.iloc[-1] - 1.0),
        "mean_net_return_bp": float(path["net_return"].mean() * 10_000.0),
        "net_bar_sharpe_sqrt_n": float(
            path["net_return"].mean() / net_std * np.sqrt(len(path))
        ) if net_std > 0 else 0.0,
        "net_max_drawdown": float((net_nav / net_nav.cummax() - 1.0).min()),
    }


def run_lstm_strategy_analysis(
    output_dir: str | Path = OUTPUT_DIR,
    sell_fee_bps: float = 5.0,
) -> dict:
    """Evaluate frozen LSTM predictions under label, price-time, and T+1 timings."""
    root = Path(output_dir)
    run_dir = root / "lstm_ensemble"
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    samples = _load_lstm_execution_returns(predictions, root / "minute")

    definitions = {
        "all_up": samples["predicted_label"].eq(2),
        "balanced_up": samples["predicted_label"].eq(2) & samples["selected_balanced"].astype(bool),
        "strict_up": samples["predicted_label"].eq(2) & samples["selected_strict"].astype(bool),
        "all_direction": samples["predicted_label"].ne(1),
        "strict_direction": samples["predicted_label"].ne(1) & samples["selected_strict"].astype(bool),
    }
    return_columns = {
        "same_close_label": "same_close_label_return",
        "next_minute_open_to_close": "next_minute_open_to_close_return",
    }
    rows: list[dict] = []
    paths: dict[str, pd.DataFrame] = {}
    for execution, return_column in return_columns.items():
        for strategy, selected in definitions.items():
            direction = "long_short" if "direction" in strategy else "long"
            for fee_bps in (0.0, float(sell_fee_bps)):
                path = aggregate_signal_strategy(
                    samples, selected, return_column, direction=direction,
                    sell_fee_bps=fee_bps,
                )
                key = f"{execution}__{strategy}__fee_{fee_bps:g}bp"
                paths[key] = path
                rows.append({
                    "execution": execution,
                    "strategy": strategy,
                    "position_type": direction,
                    "sell_fee_bps": fee_bps,
                    **_minute_strategy_metrics(path),
                })

    market = samples.groupby("target_time")["next_minute_open_to_close_return"].mean().sort_index()
    market_total = float((1.0 + market).prod() - 1.0)
    metrics = pd.DataFrame(rows)
    metrics["five_stock_market_total_return"] = market_total
    metrics["excess_vs_five_stock_market"] = (
        (1.0 + metrics["net_total_return"]) / (1.0 + market_total) - 1.0
    )
    metrics.to_csv(run_dir / "strategy_metrics.csv", index=False, float_format="%.10f")

    returns = pd.DataFrame(index=market.index)
    returns.index.name = "target_time"
    returns["five_stock_market_return"] = market
    for key, path in paths.items():
        returns[f"{key}__net_return"] = path["net_return"]
    returns.to_csv(run_dir / "strategy_returns.csv", float_format="%.10f")
    _plot_lstm_strategies(returns, run_dir / "strategy_nav.png", sell_fee_bps)

    t1_rows: list[dict] = []
    t1_returns = pd.DataFrame()
    for strategy in ("all_up", "balanced_up", "strict_up"):
        for fee_bps in (0.0, float(sell_fee_bps)):
            path = aggregate_t1_daily_strategy(samples, definitions[strategy], fee_bps)
            if t1_returns.empty:
                t1_returns["five_stock_market_return"] = path["five_stock_market_return"]
            key = f"{strategy}__fee_{fee_bps:g}bp"
            t1_returns[f"{key}__net_return"] = path["net_return"]
            t1_returns[f"{key}__exposure_matched_market_return"] = path[
                "exposure_matched_market_return"
            ]
            strategy_growth = float((1.0 + path["net_return"]).prod())
            full_market_growth = float((1.0 + path["five_stock_market_return"]).prod())
            matched_market_growth = float(
                (1.0 + path["exposure_matched_market_return"]).prod()
            )
            t1_rows.append({
                "strategy": strategy,
                "sell_fee_bps": fee_bps,
                "settled_signal_days": int(len(path)),
                "bar_coverage": float(path["active"].mean()),
                "net_total_return": strategy_growth - 1.0,
                "five_stock_market_total_return": full_market_growth - 1.0,
                "excess_vs_full_market": strategy_growth / full_market_growth - 1.0,
                "exposure_matched_market_total_return": matched_market_growth - 1.0,
                "excess_vs_exposure_matched_market": (
                    strategy_growth / matched_market_growth - 1.0
                ),
            })
    t1_metrics = pd.DataFrame(t1_rows)
    t1_metrics.to_csv(run_dir / "t1_strategy_metrics.csv", index=False, float_format="%.10f")
    t1_returns.index.name = "signal_date"
    t1_returns.to_csv(run_dir / "t1_strategy_returns.csv", float_format="%.10f")

    summary = {
        "status": "completed",
        "test_rows": int(len(samples)),
        "test_dates": sorted(samples["date"].astype(str).unique().tolist()),
        "sell_fee_bps": float(sell_fee_bps),
        "five_stock_market_total_return": market_total,
        "timing": {
            "same_close_label": (
                "close[t] to close[t+1]; diagnostic upper bound only because the signal "
                "uses the completed close[t] bar"
            ),
            "next_minute_open_to_close": (
                "signal after minute t; enter at open[t+1] and exit at close[t+1]; "
                "price-time diagnostic that does not satisfy A-share T+1"
            ),
            "next_day_same_minute_open": (
                "enter at open[t+1] and exit at the same clock-minute open on the next "
                "trading day; satisfies T+1 but has only nine settled test signal days"
            ),
        },
        "cost_model": {
            "price_time_proxy": (
                "Independent one-minute diagnostics; buy cost omitted and sell fee charged "
                "once on every active bar."
            ),
            "a_share_t1": (
                "Equal capital sleeves across 118 target minutes; inactive sleeves hold cash; "
                "sell fee charged at next-trading-day exit."
            ),
            "omitted": "Buy cost, slippage, impact, and capacity are omitted.",
        },
        "price_time_proxy_long_cash_rows": metrics[
            (metrics["execution"] == "next_minute_open_to_close")
            & (metrics["position_type"] == "long")
        ].to_dict(orient="records"),
        "a_share_t1_long_cash_rows": t1_metrics.to_dict(orient="records"),
        "research_only_long_short": True,
    }
    (run_dir / "strategy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "STRATEGY_README.md").write_text(
        """# LSTM 策略收益诊断

本分析使用正式 `test_predictions.csv`，不重新训练模型，也不根据测试收益修改置信度阈值。

- `same_close_label`：`close[t]` 到 `close[t+1]`，仅作为不可交易的标签收益上限。
- `next_minute_open_to_close`：分钟 t 完成后产生信号，下一分钟开盘买入、收盘卖出；它满足价格时点约束，但不满足 A 股 T+1，只作信号衰减诊断。
- `next_day_same_minute_open`：下一分钟开盘买入，次交易日同分钟开盘退出，满足 T+1；最后一个测试日无法结算，因此只有 9 个信号日。
- long/cash：只交易预测为上涨的样本；低置信度或非上涨预测持有现金。
- balanced/strict：沿用验证集冻结的置信度阈值。
- 成本：买入费忽略，每个实际持仓分钟收取一次卖出费；未计滑点与冲击。
- long-short 仅用于研究诊断，不作为 A 股可执行主结果。

`strategy_metrics.csv` 保存价格时点代理结果，`t1_strategy_metrics.csv` 保存 T+1 小样本结果；
两者都必须与市场及相同持仓覆盖的市场基准一起解释，不能仅凭绝对收益下结论。
""",
        encoding="utf-8",
    )
    return summary


def run_lstm_model_strategy_comparison(
    output_dir: str | Path = OUTPUT_DIR,
    sell_fee_bps: float = 5.0,
) -> dict:
    """Compare frozen LSTM and classical models on identical price keys."""
    from .lstm_baselines import export_saved_validation_predictions

    root = Path(output_dir)
    baseline_root = root / "lstm_baselines"
    validation_paths = export_saved_validation_predictions(root)
    sources = {
        "lstm_ensemble": (
            root / "lstm_ensemble" / "validation_predictions.csv",
            root / "lstm_ensemble" / "test_predictions.csv",
        ),
        **{
            name: (validation_path, baseline_root / f"{name}_test_predictions.csv")
            for name, validation_path in validation_paths.items()
        },
    }
    rows: list[dict] = []
    reference_keys: pd.DataFrame | None = None
    for model_name, (validation_path, test_path) in sources.items():
        validation = pd.read_csv(validation_path)
        test = pd.read_csv(test_path)
        confidence_thresholds = {
            "all": 0.0,
            "balanced": float(validation["confidence"].quantile(0.70)),
            "strict": float(validation["confidence"].quantile(0.90)),
        }
        keys = test[["stock", "window_end", "target_time", "true_label"]].copy()
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise ValueError(f"{model_name} test keys do not match the LSTM reference")
        samples = _load_lstm_execution_returns(test, root / "minute")
        for tier, threshold in confidence_thresholds.items():
            tier_mask = samples["confidence"].ge(threshold)
            for position_type, selected in (
                ("long", tier_mask & samples["predicted_label"].eq(2)),
                ("long_short", tier_mask & samples["predicted_label"].ne(1)),
            ):
                proxy = aggregate_signal_strategy(
                    samples,
                    selected,
                    "next_minute_open_to_close_return",
                    direction=position_type,
                    sell_fee_bps=sell_fee_bps,
                )
                row = {
                    "model": model_name,
                    "tier": tier,
                    "position_type": position_type,
                    "validation_confidence_threshold": threshold,
                    **_minute_strategy_metrics(proxy),
                }
                if position_type == "long":
                    t1 = aggregate_t1_daily_strategy(samples, selected, sell_fee_bps)
                    strategy_growth = float((1.0 + t1["net_return"]).prod())
                    full_market_growth = float((1.0 + t1["five_stock_market_return"]).prod())
                    matched_growth = float((1.0 + t1["exposure_matched_market_return"]).prod())
                    row.update({
                        "t1_settled_days": int(len(t1)),
                        "t1_net_total_return": strategy_growth - 1.0,
                        "t1_full_market_total_return": full_market_growth - 1.0,
                        "t1_excess_vs_full_market": strategy_growth / full_market_growth - 1.0,
                        "t1_exposure_matched_market_return": matched_growth - 1.0,
                        "t1_excess_vs_exposure_matched_market": strategy_growth / matched_growth - 1.0,
                    })
                rows.append(row)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(baseline_root / "strategy_comparison.csv", index=False, float_format="%.10f")
    report = {
        "status": "completed",
        "models": list(sources),
        "test_keys_exactly_aligned": True,
        "sell_fee_bps": float(sell_fee_bps),
        "threshold_rule": "70th/90th validation-confidence quantiles, frozen before test",
        "price_time_proxy": "next-minute open-to-close; does not satisfy A-share T+1",
        "t1_primary": "long/cash only; next-minute open to next-day same-minute open",
        "rows": comparison.to_dict(orient="records"),
    }
    (baseline_root / "strategy_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def _plot_task4_benchmark(comparison: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(comparison.index, comparison["strategy_nav"], label="adaptive_close", lw=1.5)
    axes[0].plot(comparison.index, comparison["benchmark_nav"], label="300-stock equal weight", lw=1.3)
    axes[0].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[0].set_ylabel("Normalized NAV")
    axes[0].set_title("Task 4 Strategy vs Equal-Weight Market")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(comparison.index, comparison["excess_nav"], color="#B45309", lw=1.5)
    axes[1].axhline(1.0, color="grey", lw=0.8, ls=":")
    axes[1].set_ylabel("Relative NAV")
    axes[1].set_title("Strategy / Benchmark")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_lstm_strategies(returns: pd.DataFrame, path: Path, sell_fee_bps: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    market_nav = (1.0 + returns["five_stock_market_return"]).cumprod()
    for axis, fee in zip(axes, (0.0, float(sell_fee_bps))):
        axis.plot(returns.index, market_nav, label="5-stock equal-weight market", color="black", lw=1.2)
        for strategy, color in (
            ("all_up", "#2563EB"), ("balanced_up", "#059669"), ("strict_up", "#D97706")
        ):
            column = (
                f"next_minute_open_to_close__{strategy}__fee_{fee:g}bp__net_return"
            )
            nav = (1.0 + returns[column]).cumprod()
            axis.plot(returns.index, nav, label=strategy, color=color, lw=1.1)
        axis.axhline(1.0, color="grey", lw=0.8, ls=":")
        axis.set_ylabel("NAV")
        axis.set_title(
            f"Price-time proxy (ignores A-share T+1), sell fee = {fee:g} bp per active bar"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
