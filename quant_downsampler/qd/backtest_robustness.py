"""Turnover-aware, cost-stressed evaluation for the corrected factor strategy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import combined_signal, compute_daily_ic, corrected_weights
from .config import OUTPUT_DIR
from .factors import compute_all_factors, load_daily_data


def _performance_from_returns(
    returns: pd.Series,
    initial_capital: float = 10_000_000.0,
) -> tuple[pd.Series, dict[str, float | int]]:
    if returns.empty:
        raise ValueError("returns cannot be empty")
    values = returns.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("returns must be finite and greater than -100%")
    if not np.isfinite(initial_capital) or initial_capital <= 0.0:
        raise ValueError("initial_capital must be positive and finite")
    nav = initial_capital * (1.0 + returns).cumprod()
    total_return = float(nav.iloc[-1] / initial_capital - 1.0)
    periods = len(returns)
    annual_return = float((1.0 + total_return) ** (252.0 / periods) - 1.0)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(252.0))
    wealth_with_initial = np.concatenate(([initial_capital], nav.to_numpy(float)))
    drawdown = wealth_with_initial / np.maximum.accumulate(wealth_with_initial) - 1.0
    daily_std = float(returns.std(ddof=1))
    metrics: dict[str, float | int] = {
        "initial_capital": initial_capital,
        "final_nav": float(nav.iloc[-1]),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": (
            float(returns.mean() / daily_std * np.sqrt(252.0))
            if np.isfinite(daily_std) and daily_std > 0.0 else float("nan")
        ),
        "max_drawdown": float(np.min(drawdown)),
        "periods": periods,
    }
    nav.name = "nav"
    return nav, metrics


def _historical_filters(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    liquidity_window: int,
    volatility_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build decision-time filters using observations through t-1 only."""
    min_liquidity = max(2, liquidity_window // 2)
    min_volatility = max(2, volatility_window // 2)
    trailing_amount = amount.rolling(
        liquidity_window, min_periods=min_liquidity
    ).mean().shift(1)
    trailing_volatility = close.pct_change(fill_method=None).rolling(
        volatility_window, min_periods=min_volatility
    ).std(ddof=1).shift(1)
    return trailing_amount, trailing_volatility


def _eligible_new_positions(
    signal: pd.Series,
    execution_date: pd.Timestamp,
    volume: pd.DataFrame,
    trailing_amount: pd.DataFrame,
    trailing_volatility: pd.DataFrame,
    min_liquidity_quantile: float | None,
    max_volatility_quantile: float | None,
) -> pd.Series:
    actual_tradeable = volume.loc[execution_date] > 0
    if min_liquidity_quantile is None and max_volatility_quantile is None:
        return signal[signal.index.isin(actual_tradeable.index[actual_tradeable])].dropna().sort_values(
            ascending=False
        )
    liquidity = trailing_amount.loc[execution_date]
    volatility = trailing_volatility.loc[execution_date]
    valid = actual_tradeable & liquidity.notna() & volatility.notna()
    if not valid.any():
        return pd.Series(dtype=np.float64)
    liquidity_floor = (
        liquidity[valid].quantile(min_liquidity_quantile)
        if min_liquidity_quantile is not None else -np.inf
    )
    volatility_ceiling = (
        volatility[valid].quantile(max_volatility_quantile)
        if max_volatility_quantile is not None else np.inf
    )
    eligible = valid & (liquidity >= liquidity_floor) & (volatility <= volatility_ceiling)
    return signal[signal.index.isin(eligible.index[eligible])].dropna().sort_values(
        ascending=False
    )


def _buffered_target(
    previous: set[str],
    locked: set[str],
    full_signal: pd.Series,
    eligible_signal: pd.Series,
    top_n: int,
    buffer_n: int,
    max_replacements: int,
) -> tuple[set[str], dict[str, int]]:
    """Select a top-N portfolio while retaining incumbents inside a wider buffer."""
    if top_n <= 0 or buffer_n < top_n or max_replacements < 0:
        raise ValueError("require top_n > 0, buffer_n >= top_n, max_replacements >= 0")
    capacity = max(top_n, len(locked))
    sellable_previous = previous - locked
    # Locked names consume portfolio slots but cannot appear in the tradeable
    # signal ranking.  Reduce both the target and buffer capacities accordingly.
    effective_buffer = max(0, buffer_n - len(locked))
    buffer = set(eligible_signal.head(effective_buffer).index)
    retained = sellable_previous & buffer
    target = set(locked) | retained

    # Keep the best remaining incumbents when the replacement cap requires it.
    required_previous = max(0, len(sellable_previous) - max_replacements)
    remaining_previous = list(sellable_previous - retained)
    remaining_previous.sort(
        key=lambda stock: float(full_signal.get(stock, -np.inf)), reverse=True
    )
    for stock in remaining_previous:
        if len(target & sellable_previous) >= required_previous or len(target) >= capacity:
            break
        target.add(stock)

    for stock in eligible_signal.index:
        if len(target) >= capacity:
            break
        target.add(str(stock))

    rank = {str(stock): position for position, stock in enumerate(eligible_signal.index, 1)}
    return target, rank


def _equal_weight_turnover(
    previous: set[str],
    target: set[str],
) -> tuple[float, float]:
    """Return exact sell/buy weight changes between equal-weight portfolios."""
    previous_weight = 1.0 / len(previous) if previous else 0.0
    target_weight = 1.0 / len(target) if target else 0.0
    sell = buy = 0.0
    for stock in previous | target:
        old = previous_weight if stock in previous else 0.0
        new = target_weight if stock in target else 0.0
        change = new - old
        if change > 0.0:
            buy += change
        elif change < 0.0:
            sell -= change
    return float(sell), float(buy)


def run_turnover_aware_backtest(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    close: pd.DataFrame,
    open_price: pd.DataFrame,
    volume: pd.DataFrame,
    amount: pd.DataFrame,
    *,
    lookback: int = 60,
    top_n: int = 10,
    rebalance_at: str = "close",
    buffer_n: int = 20,
    max_replacements: int = 3,
    liquidity_window: int = 20,
    volatility_window: int = 20,
    min_liquidity_quantile: float | None = 0.20,
    max_volatility_quantile: float | None = 0.90,
    initial_capital: float = 10_000_000.0,
) -> dict:
    """Run a corrected strategy whose holdings do not depend on assumed costs.

    New-position filters use rolling values shifted by one day.  Same-day volume
    is used only as an execution feasibility check; halted incumbents remain
    locked rather than being treated as sold.
    """
    if rebalance_at not in {"close", "open"}:
        raise ValueError("rebalance_at must be close or open")
    if min_liquidity_quantile is not None and not 0.0 <= min_liquidity_quantile < 1.0:
        raise ValueError("min_liquidity_quantile must be within [0, 1)")
    if max_volatility_quantile is not None and not 0.0 < max_volatility_quantile <= 1.0:
        raise ValueError("max_volatility_quantile must be within (0, 1]")

    dates = close.index
    price = close if rebalance_at == "close" else open_price
    weights = corrected_weights(compute_daily_ic(factors, forward_return), lookback)
    trailing_amount, trailing_volatility = _historical_filters(
        close, amount, liquidity_window, volatility_window
    )
    previous: set[str] = set()
    rows: list[dict] = []
    selections: list[dict] = []

    for i in range(1, len(dates) - 1):
        execution_date = dates[i]
        next_date = dates[i + 1]
        signal_date = dates[i - 1]
        daily_weights = weights.loc[execution_date]
        if daily_weights.notna().sum() == 0:
            continue
        signal = combined_signal(factors, daily_weights, signal_date)
        if signal.empty:
            continue
        tradeable = set(volume.columns[volume.loc[execution_date] > 0])
        locked = previous - tradeable
        eligible = _eligible_new_positions(
            signal,
            execution_date,
            volume,
            trailing_amount,
            trailing_volatility,
            min_liquidity_quantile,
            max_volatility_quantile,
        )
        target, rank = _buffered_target(
            previous,
            locked,
            signal,
            eligible,
            top_n,
            buffer_n,
            max_replacements,
        )
        if not target:
            continue

        sell_turnover, buy_turnover = _equal_weight_turnover(previous, target)
        valid_returns = []
        for stock in target:
            start_price = price.loc[execution_date, stock]
            end_price = price.loc[next_date, stock]
            if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                valid_returns.append(float(end_price / start_price - 1.0))
        gross_return = float(np.mean(valid_returns)) if valid_returns else 0.0
        rows.append({
            "date": next_date,
            "gross_return": gross_return,
            "sell_turnover": sell_turnover,
            "buy_turnover": buy_turnover,
            "total_turnover": sell_turnover + buy_turnover,
            "n_holdings": len(target),
            "n_locked": len(locked),
        })
        for stock in sorted(target, key=lambda item: rank.get(item, buffer_n + 1)):
            selections.append({
                "signal_date": signal_date,
                "execution_date": execution_date,
                "stock": stock,
                "signal_rank": rank.get(stock, np.nan),
                "signal": float(signal.get(stock, np.nan)),
                "retained": stock in previous,
                "locked": stock in locked,
            })
        previous = target

    if not rows:
        raise ValueError("no valid turnover-aware backtest periods")
    # Charge a full exit after the last realised holding return.  Without this
    # explicit liquidation, every cost scenario omits one side of the final
    # round trip and slightly overstates terminal performance.
    rows[-1]["sell_turnover"] += 1.0
    rows[-1]["total_turnover"] += 1.0
    rows[-1]["final_liquidation_turnover"] = 1.0
    periods = pd.DataFrame(rows).set_index("date")
    periods["final_liquidation_turnover"] = periods[
        "final_liquidation_turnover"
    ].fillna(0.0)
    gross_nav, gross_metrics = _performance_from_returns(
        periods["gross_return"], initial_capital
    )
    gross_metrics.update({
        "mean_sell_turnover": float(periods["sell_turnover"].mean()),
        "mean_buy_turnover": float(periods["buy_turnover"].mean()),
        "mean_total_turnover": float(periods["total_turnover"].mean()),
    })
    return {
        "periods": periods,
        "gross_nav": gross_nav,
        "gross_metrics": gross_metrics,
        "selections": pd.DataFrame(selections),
    }


def apply_transaction_costs(
    result: dict,
    *,
    sell_cost: float,
    buy_cost: float,
    initial_capital: float = 10_000_000.0,
) -> dict:
    """Reprice fixed holdings under explicit one-way buy and sell costs."""
    if sell_cost < 0.0 or buy_cost < 0.0:
        raise ValueError("transaction costs cannot be negative")
    periods = result["periods"].copy()
    periods["transaction_cost"] = (
        periods["sell_turnover"] * sell_cost
        + periods["buy_turnover"] * buy_cost
    )
    periods["net_return"] = periods["gross_return"] - periods["transaction_cost"]
    nav, metrics = _performance_from_returns(periods["net_return"], initial_capital)
    metrics.update({
        "sell_cost": sell_cost,
        "buy_cost": buy_cost,
        "mean_sell_turnover": float(periods["sell_turnover"].mean()),
        "mean_buy_turnover": float(periods["buy_turnover"].mean()),
        "mean_total_turnover": float(periods["total_turnover"].mean()),
        "total_cost_fraction": float(periods["transaction_cost"].sum()),
    })
    return {"periods": periods, "nav": nav, "metrics": metrics}


def symmetric_cost_stress(
    result: dict,
    costs_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0),
) -> pd.DataFrame:
    rows = []
    for bps in costs_bps:
        if bps < 0:
            raise ValueError("cost grid cannot contain negative values")
        priced = apply_transaction_costs(
            result, sell_cost=bps / 10_000.0, buy_cost=bps / 10_000.0
        )
        rows.append({"one_way_cost_bps": bps, **priced["metrics"]})
    return pd.DataFrame(rows).set_index("one_way_cost_bps")


def estimate_break_even_cost_bps(result: dict, upper_bps: float = 500.0) -> float:
    """Find the symmetric one-way cost that reduces final total return to zero."""
    low, high = 0.0, upper_bps
    if apply_transaction_costs(
        result, sell_cost=high / 10_000.0, buy_cost=high / 10_000.0
    )["metrics"]["total_return"] > 0:
        return float("nan")
    for _ in range(60):
        mid = (low + high) / 2.0
        total_return = apply_transaction_costs(
            result, sell_cost=mid / 10_000.0, buy_cost=mid / 10_000.0
        )["metrics"]["total_return"]
        if total_return > 0:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def _plot_robustness(
    baseline_stress: pd.DataFrame,
    robust_stress: pd.DataFrame,
    baseline_nav: pd.Series,
    robust_nav: pd.Series,
    target: Path,
    initial_capital: float = 10_000_000.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        baseline_stress.index, baseline_stress["total_return"] * 100.0,
        marker="o", label="published-style baseline",
    )
    axes[0].plot(
        robust_stress.index, robust_stress["total_return"] * 100.0,
        marker="o", label="turnover-aware",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Symmetric transaction-cost stress")
    axes[0].set_xlabel("One-way cost (bps)")
    axes[0].set_ylabel("Total return (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        baseline_nav.index, baseline_nav / initial_capital,
        label="published-style baseline",
    )
    axes[1].plot(
        robust_nav.index, robust_nav / initial_capital,
        label="turnover-aware",
    )
    axes[1].set_title("Sell-only 5 bps NAV")
    axes[1].set_ylabel("Normalised NAV")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(target / "backtest_robustness.png", dpi=170)
    plt.close(figure)


def run_backtest_robustness(
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    *,
    lookback: int = 60,
    top_n: int = 10,
    buffer_n: int = 20,
    max_replacements: int = 3,
    costs_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0, 50.0),
) -> dict:
    target = Path(out_dir or (OUTPUT_DIR / "backtest_robustness"))
    target.mkdir(parents=True, exist_ok=True)
    data = load_daily_data(data_dir)
    factors = compute_all_factors(data)
    baseline = run_turnover_aware_backtest(
        factors,
        factors["forward_return_1d"],
        data["close"],
        data["open"],
        data["volume"],
        data["amount"],
        lookback=lookback,
        top_n=top_n,
        buffer_n=top_n,
        max_replacements=top_n,
        min_liquidity_quantile=None,
        max_volatility_quantile=None,
    )
    result = run_turnover_aware_backtest(
        factors,
        factors["forward_return_1d"],
        data["close"],
        data["open"],
        data["volume"],
        data["amount"],
        lookback=lookback,
        top_n=top_n,
        buffer_n=buffer_n,
        max_replacements=max_replacements,
    )
    baseline_stress = symmetric_cost_stress(baseline, costs_bps)
    stress = symmetric_cost_stress(result, costs_bps)
    baseline_break_even = estimate_break_even_cost_bps(baseline)
    break_even = estimate_break_even_cost_bps(result)
    baseline_mandated = apply_transaction_costs(
        baseline, sell_cost=0.0005, buy_cost=0.0
    )
    mandated = apply_transaction_costs(result, sell_cost=0.0005, buy_cost=0.0)
    baseline["periods"].to_csv(
        target / "baseline_gross_periods.csv", float_format="%.10f"
    )
    baseline["selections"].to_csv(target / "baseline_selections.csv", index=False)
    result["periods"].to_csv(target / "gross_periods.csv", float_format="%.10f")
    result["selections"].to_csv(target / "selections.csv", index=False)
    baseline_stress.to_csv(target / "baseline_cost_stress.csv", float_format="%.10f")
    stress.to_csv(target / "cost_stress.csv", float_format="%.10f")
    mandated["periods"].to_csv(target / "mandated_cost_periods.csv", float_format="%.10f")
    summary = {
        "configuration": {
            "lookback": lookback,
            "top_n": top_n,
            "buffer_n": buffer_n,
            "max_replacements": max_replacements,
            "liquidity_filter": "exclude bottom 20% trailing-20-day amount for new positions",
            "volatility_filter": "exclude top 10% trailing-20-day volatility for new positions",
        },
        "published_style_baseline": {
            "mandated_sell_only_5bps": baseline_mandated["metrics"],
            "symmetric_break_even_one_way_bps": baseline_break_even,
        },
        "gross_metrics": result["gross_metrics"],
        "mandated_sell_only_5bps": mandated["metrics"],
        "symmetric_break_even_one_way_bps": break_even,
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    baseline_metrics = baseline_mandated["metrics"]
    robust_metrics = mandated["metrics"]
    cost_rows = [
        "| 单边对称成本 | 原策略累计收益 | 换手约束策略累计收益 |",
        "|---:|---:|---:|",
    ]
    for bps in stress.index:
        cost_rows.append(
            f"| {float(bps):.0f} bps | "
            f"{baseline_stress.loc[bps, 'total_return']:.2%} | "
            f"{stress.loc[bps, 'total_return']:.2%} |"
        )
    (target / "README.md").write_text(
        "\n".join([
            "# 因子组合稳健性与交易成本压力测试",
            "",
            "本报告只使用修正版的信息时序。换手约束版本使用 Top 20 缓冲区、每日最多替换 3 只，",
            "并仅允许历史 20 日成交额非底部 20%、历史波动非顶部 10% 的股票新建仓。",
            "过滤变量均滞后一天；停牌持仓仍按 locked 处理。",
            "",
            "## 关键结果",
            "",
            f"- 原策略按题目卖出 5 bps 口径：累计收益 {baseline_metrics['total_return']:.2%}，"
            f"日均卖出换手 {baseline_metrics['mean_sell_turnover']:.2%}。",
            f"- 换手约束策略同口径：累计收益 {robust_metrics['total_return']:.2%}，"
            f"日均卖出换手 {robust_metrics['mean_sell_turnover']:.2%}，"
            f"最大回撤 {robust_metrics['max_drawdown']:.2%}。",
            f"- 对称双边成本下，原策略与换手约束策略的盈亏平衡单边成本分别约为 "
            f"{baseline_break_even:.2f} bps 和 {break_even:.2f} bps。",
            "",
            "## 成本压力",
            "",
            *cost_rows,
            "",
            "这里的压力测试使用固定持仓路径和线性成本，没有模拟冲击、涨跌停或容量；",
            "结果仍是研究诊断，不是实盘收益承诺。",
            "",
        ]) + "\n",
        encoding="utf-8",
    )
    _plot_robustness(
        baseline_stress,
        stress,
        baseline_mandated["nav"],
        mandated["nav"],
        target,
    )
    return {
        **result,
        "baseline": baseline,
        "baseline_stress": baseline_stress,
        "baseline_mandated": baseline_mandated,
        "stress": stress,
        "mandated": mandated,
        "summary": summary,
    }
