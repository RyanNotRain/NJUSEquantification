"""IC-weighted top-10 portfolio backtests with explicit information timing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .evaluation import compute_ic_series
from .factors import FACTOR_NAMES, compute_all_factors, load_daily_data


def _markdown_table(table: pd.DataFrame) -> str:
    headers = [str(table.index.name or "strategy"), *map(str, table.columns)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in table.iterrows():
        values = [str(idx)] + [f"{float(v):.6f}" if pd.notna(v) else "NaN" for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compute_daily_ic(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame({
        name: compute_ic_series(factors[name], forward_return, "pearson")
        for name in FACTOR_NAMES
    }).reindex(forward_return.index)


def corrected_weights(daily_ic: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Weights available before date t opens.

    IC[s] uses return s->s+1 and becomes known at the close of s+1.  Therefore
    a date-t decision may use IC through t-2, implemented by shift(2).
    """
    min_periods = min(lookback, max(2, lookback // 3))
    return daily_ic.rolling(lookback, min_periods=min_periods).mean().shift(2)


def combined_signal(
    factors: dict[str, pd.DataFrame],
    weights: pd.Series,
    signal_date: pd.Timestamp,
) -> pd.Series:
    active = weights.dropna()
    active = active[active.index.isin(FACTOR_NAMES)]
    if active.empty or active.abs().sum() == 0:
        return pd.Series(dtype=np.float64)
    active = active / active.abs().sum()
    pieces = []
    for name, weight in active.items():
        row = factors[name].loc[signal_date].replace([np.inf, -np.inf], np.nan)
        std = row.std(ddof=1)
        if not np.isfinite(std) or std == 0:
            continue
        pieces.append(((row - row.mean()) / std) * weight)
    if not pieces:
        return pd.Series(dtype=np.float64)
    signal = pieces[0]
    for piece in pieces[1:]:
        signal = signal.add(piece, fill_value=0.0)
    return signal.sort_values(ascending=False)


def _performance(
    nav: pd.Series,
    returns: pd.Series,
    turnover: pd.Series,
    initial_capital: float,
) -> dict[str, float | int]:
    periods = len(returns)
    total_return = float(nav.iloc[-1] / initial_capital - 1)
    annual_return = (1 + total_return) ** (252 / periods) - 1 if periods else np.nan
    annual_vol = float(returns.std(ddof=1) * np.sqrt(252)) if periods > 1 else np.nan
    sharpe = annual_return / annual_vol if annual_vol and annual_vol > 0 else np.nan
    drawdown = nav / nav.cummax() - 1
    return {
        "initial_capital": initial_capital,
        "final_nav": float(nav.iloc[-1]),
        "total_return": total_return,
        "annual_return": float(annual_return),
        "annual_volatility": annual_vol,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "mean_sell_turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "periods": periods,
    }


def run_backtest(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    close: pd.DataFrame,
    open_price: pd.DataFrame,
    volume: pd.DataFrame,
    lookback: int = 60,
    top_n: int = 10,
    sell_fee: float = 0.0005,
    initial_capital: float = 10_000_000.0,
    rebalance_at: str = "close",
    naive: bool = False,
) -> dict:
    if rebalance_at not in {"close", "open"}:
        raise ValueError("rebalance_at must be close or open")
    dates = close.index
    daily_ic = compute_daily_ic(factors, forward_return)
    if naive:
        # Original statement: same-day future-return IC.  For next-open
        # execution the signal day's IC is shifted to that execution date.
        weight_table = daily_ic if rebalance_at == "close" else daily_ic.shift(1)
    else:
        weight_table = corrected_weights(daily_ic, lookback)

    price = close if rebalance_at == "close" else open_price
    nav_values = [initial_capital]
    nav_dates: list[pd.Timestamp] = []
    period_returns: list[float] = []
    turnovers: list[float] = []
    selections: list[dict] = []
    previous: set[str] = set()
    initial_date: pd.Timestamp | None = None

    for i in range(1, len(dates) - 1):
        execution_date = dates[i]
        next_date = dates[i + 1]
        if naive and rebalance_at == "close":
            signal_date = execution_date
        else:
            signal_date = dates[i - 1]
        weights = weight_table.loc[execution_date]
        if weights.notna().sum() == 0:
            continue

        signal = combined_signal(factors, weights, signal_date)
        tradeable = set(volume.columns[volume.loc[execution_date] > 0])
        locked = previous - tradeable
        slots = max(0, top_n - len(locked))
        candidates = signal[signal.index.isin(tradeable)].dropna()
        chosen = list(candidates.nlargest(slots).index)
        target = locked | set(chosen)
        if not target:
            continue

        if initial_date is None:
            initial_date = execution_date

        sold = previous - target
        sell_turnover = len(sold) / max(len(previous), 1) if previous else 0.0
        valid_returns = []
        for stock in target:
            p0 = price.loc[execution_date, stock]
            p1 = price.loc[next_date, stock]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                valid_returns.append(float(p1 / p0 - 1.0))
        gross = float(np.mean(valid_returns)) if valid_returns else 0.0
        net = gross - sell_turnover * sell_fee
        nav_values.append(nav_values[-1] * (1 + net))
        nav_dates.append(next_date)
        period_returns.append(net)
        turnovers.append(sell_turnover)
        rank_lookup = {stock: rank for rank, stock in enumerate(chosen, 1)}
        for stock in sorted(target, key=lambda x: rank_lookup.get(x, top_n + 1)):
            selections.append({
                "signal_date": signal_date,
                "execution_date": execution_date,
                "stock": stock,
                "signal_rank": rank_lookup.get(stock, np.nan),
                "signal": float(signal.get(stock, np.nan)),
                "locked": stock in locked,
            })
        previous = target

    if not nav_dates or initial_date is None:
        raise ValueError("no valid backtest periods")
    nav = pd.Series(nav_values, index=[initial_date, *nav_dates], name="nav")
    returns = pd.Series(period_returns, index=nav_dates, name="return")
    turnover = pd.Series(turnovers, index=nav_dates, name="sell_turnover")
    return {
        "nav": nav,
        "returns": returns,
        "turnover": turnover,
        "selections": pd.DataFrame(selections),
        "metrics": _performance(nav, returns, turnover, initial_capital),
    }


def _plot_nav(results: dict[str, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for name, result in results.items():
        normalized = result["nav"] / result["metrics"]["initial_capital"]
        axes[0].plot(normalized.index, normalized, label=name)
        drawdown = result["nav"] / result["nav"].cummax() - 1
        axes[1].plot(drawdown.index, drawdown, label=name)
    axes[0].set_title("IC-weighted top-10 portfolio NAV")
    axes[0].set_ylabel("NAV / initial capital")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "nav_curve.png", dpi=160)
    plt.close(fig)


def _write_report(results: dict[str, dict], out_dir: Path, lookback: int) -> None:
    metrics = pd.DataFrame({name: x["metrics"] for name, x in results.items()}).T
    show = metrics[[
        "final_nav", "total_return", "annual_return", "annual_volatility",
        "sharpe_ratio", "max_drawdown", "mean_sell_turnover", "periods",
    ]]
    lines = [
        "# 策略回测报告",
        "",
        "初始资金 1,000 万元；每日选择 10 只；卖出手续费万 5；买入免费。",
        "",
        "## 结果",
        "",
        _markdown_table(show),
        "",
        "## 前视偏差与修正",
        "",
        "题目原始的当日 IC 使用了当日因子与次日收益，交易时尚不可知，因此 Naive 结果仅用于展示前视偏差。",
        f"修正版在交易日 t 使用 t-1 因子，并用截至 t-2 已实现收益形成的过去 {lookback} 日平均 IC 权重。",
        "同时只允许当日成交量大于 0 的股票新建仓；已持有但停牌的股票标记为 locked，不假设能够卖出。",
        "",
        "年化收益使用复合年化，夏普率无风险利率取 0。每日净值、卖出换手和选股明细均另存为 CSV。",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_backtest(
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    lookback: int = 60,
    top_n: int = 10,
) -> dict[str, dict]:
    target = Path(out_dir or (OUTPUT_DIR / "backtest"))
    target.mkdir(parents=True, exist_ok=True)
    data = load_daily_data(data_dir)
    factors = compute_all_factors(data)
    forward = factors["forward_return_1d"]
    results = {
        "corrected_close": run_backtest(
            factors, forward, data["close"], data["open"], data["volume"],
            lookback, top_n, rebalance_at="close", naive=False,
        ),
        "corrected_open": run_backtest(
            factors, forward, data["close"], data["open"], data["volume"],
            lookback, top_n, rebalance_at="open", naive=False,
        ),
        "naive_close": run_backtest(
            factors, forward, data["close"], data["open"], data["volume"],
            lookback, top_n, rebalance_at="close", naive=True,
        ),
        "naive_open": run_backtest(
            factors, forward, data["close"], data["open"], data["volume"],
            lookback, top_n, rebalance_at="open", naive=True,
        ),
    }
    pd.DataFrame({name: result["nav"] for name, result in results.items()}).to_csv(
        target / "nav.csv", float_format="%.6f"
    )
    pd.DataFrame({name: result["returns"] for name, result in results.items()}).to_csv(
        target / "daily_returns.csv", float_format="%.8f"
    )
    pd.DataFrame({name: result["turnover"] for name, result in results.items()}).to_csv(
        target / "sell_turnover.csv", float_format="%.8f"
    )
    metrics = {name: result["metrics"] for name, result in results.items()}
    (target / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, result in results.items():
        result["selections"].to_csv(target / f"selections_{name}.csv", index=False)
    _plot_nav(results, target)
    _write_report(results, target, lookback)
    return results
