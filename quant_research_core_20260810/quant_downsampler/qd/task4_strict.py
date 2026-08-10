"""Strict Task 4 backtest with target-aligned IC and realistic order timing."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from .config import APPLY_ADJFACTOR, OUTPUT_DIR
from .factors import compute_all_factors, load_daily_data, select_required_factors


def build_signal_target(execution_price: pd.DataFrame) -> pd.DataFrame:
    """Return earned by a signal at close t: enter t+1, rebalance t+2."""
    return execution_price.shift(-2).div(execution_price.shift(-1)).sub(1.0)


def compute_daily_ic(
    factors: dict[str, pd.DataFrame],
    target_return: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-sectional IC indexed by signal date; target may be known later."""
    tradable_target = (volume.shift(-1) > 0) & (volume.shift(-2) > 0)
    target = target_return.where(tradable_target)
    return pd.DataFrame({
        name: factor.corrwith(target, axis=1)
        for name, factor in factors.items()
    }, index=target.index)


def historical_ic_weights(
    daily_ic: pd.DataFrame,
    lookback: int = 60,
    method: str = "rolling",
) -> pd.DataFrame:
    """Signed IC weights available at each decision close without look-ahead.

    A signal dated t-2 earns its full t-1 to t execution-price return at t.
    Therefore raw IC must be delayed by two trading rows before it can be used.
    """
    if method not in {"rolling", "expanding", "hybrid", "adaptive", "naive"}:
        raise ValueError("unknown IC method")
    if method == "naive":
        return daily_ic.copy()  # deliberately biased comparison required by the prompt

    available = daily_ic.shift(2)
    min_periods = max(20, lookback // 3)
    recent = available.rolling(lookback, min_periods=min_periods).mean()
    long_term = available.expanding(min_periods=min_periods).mean()
    if method == "rolling":
        return recent
    if method == "expanding":
        return long_term
    blended = 0.7 * long_term + 0.3 * recent
    if method == "hybrid":
        return blended

    # Risk-adjusted, regime-aware variant: weak or sign-conflicted factors turn off.
    ic_risk = available.rolling(lookback, min_periods=min_periods).std(ddof=1)
    stable = (np.sign(recent) == np.sign(long_term)) & (recent.abs() >= 0.01)
    return (blended.div(ic_risk + 1e-6).clip(-3.0, 3.0)).where(stable, 0.0)


def _robust_zscore(values: pd.Series) -> pd.Series:
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 10:
        return pd.Series(dtype=float)
    median = values.median()
    mad = (values - median).abs().median()
    clipped = values.clip(median - 5 * (mad + 1e-8), median + 5 * (mad + 1e-8))
    std = clipped.std(ddof=1)
    return (clipped - clipped.mean()) / (std + 1e-12)


def combined_signal(
    factors: dict[str, pd.DataFrame], signed_weights: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Series:
    if date not in signed_weights.index:
        return pd.Series(dtype=float)
    weights = signed_weights.loc[date].replace([np.inf, -np.inf], np.nan).dropna()
    weights = weights[weights.abs() > 0]
    if weights.empty:
        return pd.Series(dtype=float)
    weights = weights / weights.abs().sum()
    signal = pd.Series(dtype=float)
    for name, weight in weights.items():
        if name not in factors or date not in factors[name].index:
            continue
        contribution = _robust_zscore(factors[name].loc[date]) * weight
        signal = contribution if signal.empty else signal.add(contribution, fill_value=0.0)
    return signal.replace([np.inf, -np.inf], np.nan).dropna()


def buffered_selection(
    signal: pd.Series,
    previous: set[str],
    locked: set[str],
    eligible_new: set[str],
    top_n: int,
    buffer_n: int = 20,
    max_replacements: int = 3,
) -> list[str]:
    """Choose a Top-N target with a rank buffer and replacement cap.

    Locked holdings consume capacity but are never presented as executable
    sells.  The replacement cap is applied only to sellable incumbents.
    """
    if top_n <= 0 or buffer_n < top_n or max_replacements < 0:
        raise ValueError("require top_n > 0, buffer_n >= top_n, replacements >= 0")
    ranking = signal.sort_values(ascending=False).index.astype(str).tolist()
    rank = {stock: position for position, stock in enumerate(ranking, 1)}
    locked_ordered = sorted(locked, key=lambda stock: rank.get(stock, len(rank) + 1))
    capacity = max(0, top_n - len(locked_ordered))
    sellable_previous = previous - locked
    ranked_previous = sorted(sellable_previous, key=lambda stock: rank.get(stock, len(rank) + 1))
    inside_buffer = [stock for stock in ranked_previous if rank.get(stock, len(rank) + 1) <= buffer_n]
    minimum_retained = max(0, min(len(ranked_previous), capacity - max_replacements))
    keep_count = max(minimum_retained, min(len(inside_buffer), capacity))
    retained = ranked_previous[:keep_count]
    target = locked_ordered + retained
    for stock in ranking:
        if len(target) >= top_n:
            break
        if stock in target or stock in previous or stock not in eligible_new:
            continue
        target.append(stock)
    if len(target) < top_n:
        for stock in ranked_previous:
            if len(target) >= top_n:
                break
            if stock not in target:
                target.append(stock)
    return target


def _metrics(nav: pd.Series, daily_return: pd.Series, turnover: pd.Series) -> dict[str, float]:
    if len(daily_return) < 2:
        return {}
    years = len(daily_return) / 252.0
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(daily_return.std(ddof=1) * np.sqrt(252))
    sharpe = float(daily_return.mean() / daily_return.std(ddof=1) * np.sqrt(252)) if daily_return.std(ddof=1) > 0 else 0.0
    drawdown = nav.div(nav.cummax()).sub(1.0)
    return {"annual_return_cagr": cagr, "annual_return_arithmetic": float(daily_return.mean() * 252),
            "annual_volatility": vol, "sharpe_ratio": sharpe, "max_drawdown": float(drawdown.min()),
            "final_nav": float(nav.iloc[-1]), "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
            "average_sell_turnover": float(turnover.mean()), "trading_days": int(len(daily_return))}


def run_strict_backtest(
    factors: dict[str, pd.DataFrame], daily: dict[str, pd.DataFrame],
    execution: str = "open", method: str = "rolling", lookback: int = 60,
    top_n: int = 10, initial_capital: float = 10_000_000.0,
    sell_fee: float = 0.0005,
    selection_policy: str = "top_n",
    buffer_n: int = 20,
    max_replacements: int = 3,
    min_liquidity_quantile: float = 0.20,
    max_volatility_quantile: float = 0.90,
    trade_start_date: str | pd.Timestamp | None = None,
) -> dict:
    """Run a long-only top-N portfolio with next-session execution and locked halts."""
    if selection_policy not in {"top_n", "buffered"}:
        raise ValueError("selection_policy must be top_n or buffered")
    price = daily[execution]
    volume, amount = daily["volume"], daily["amount"]
    dates = list(price.index)
    trade_start = pd.Timestamp(trade_start_date) if trade_start_date is not None else None
    target = build_signal_target(price)
    raw_ic = compute_daily_ic(factors, target, volume)
    weights = historical_ic_weights(raw_ic, lookback, method)
    avg_amount = amount.rolling(20, min_periods=10).mean()
    trailing_volatility = daily["close"].pct_change(fill_method=None).rolling(20, min_periods=10).std(ddof=1)

    cash = initial_capital
    shares: dict[str, float] = {}
    nav_values = [initial_capital]
    nav_dates = [dates[0]]
    returns, turnovers, selected_rows = [], [], []

    for i in range(1, len(dates)):
        execution_date, decision_date = dates[i], dates[i - 1]
        px = price.loc[execution_date]
        can_trade = (volume.loc[execution_date] > 0) & px.notna() & (px > 0)
        before = cash + sum(qty * px.get(stock, np.nan) for stock, qty in shares.items() if pd.notna(px.get(stock, np.nan)))

        signal = (
            combined_signal(factors, weights, decision_date)
            if trade_start is None or execution_date >= trade_start
            else pd.Series(dtype=float)
        )
        if not signal.empty:
            known_liquid = (volume.loc[decision_date].reindex(signal.index).fillna(0) > 0)
            liquid_amount = avg_amount.loc[decision_date].reindex(signal.index).fillna(0) > 0
            signal = signal[known_liquid & liquid_amount]
        if selection_policy == "buffered" and not signal.empty:
            amount_row = avg_amount.loc[decision_date].reindex(signal.index)
            volatility_row = trailing_volatility.loc[decision_date].reindex(signal.index)
            amount_cut = amount_row.dropna().quantile(min_liquidity_quantile)
            volatility_cut = volatility_row.dropna().quantile(max_volatility_quantile)
            eligible_new = set(signal.index[
                amount_row.ge(amount_cut).fillna(False)
                & volatility_row.le(volatility_cut).fillna(False)
            ].astype(str))
            previous = set(shares)
            locked = {stock for stock in previous if not bool(can_trade.get(stock, False))}
            desired = buffered_selection(
                signal, previous, locked, eligible_new, top_n,
                buffer_n=buffer_n, max_replacements=max_replacements,
            )
        else:
            desired = signal.nlargest(top_n).index.tolist() if len(signal) >= top_n else []
        for rank, stock in enumerate(desired, 1):
            selected_rows.append({"decision_date": decision_date, "execution_date": execution_date,
                                  "rank": rank, "stock": stock, "signal": float(signal.loc[stock])})

        sold_value = 0.0
        # Sell names no longer desired.  Halted names remain locked in the book.
        for stock in list(shares):
            if stock not in desired and bool(can_trade.get(stock, False)):
                value = shares.pop(stock) * float(px[stock])
                sold_value += value; cash += value * (1 - sell_fee)

        current_equity = cash + sum(qty * float(px[stock]) for stock, qty in shares.items() if pd.notna(px.get(stock)))
        target_value = current_equity / top_n if desired else 0.0
        # Rebalance tradable desired holdings; sells first, buys second.
        for stock in desired:
            if not bool(can_trade.get(stock, False)) or stock not in shares:
                continue
            current = shares[stock] * float(px[stock])
            if current > target_value:
                value = current - target_value
                shares[stock] -= value / float(px[stock])
                sold_value += value; cash += value * (1 - sell_fee)
        for stock in desired:
            if not bool(can_trade.get(stock, False)):
                continue
            current = shares.get(stock, 0.0) * float(px[stock])
            buy_value = min(max(target_value - current, 0.0), cash)
            if buy_value > 0:
                shares[stock] = shares.get(stock, 0.0) + buy_value / float(px[stock])
                cash -= buy_value

        after = cash + sum(qty * float(px[stock]) for stock, qty in shares.items() if pd.notna(px.get(stock)))
        daily_ret = after / nav_values[-1] - 1.0
        nav_dates.append(execution_date); nav_values.append(after)
        returns.append(daily_ret); turnovers.append(sold_value / before if before > 0 else 0.0)

    nav = pd.Series(nav_values, index=nav_dates, name="nav")
    daily_return = pd.Series(returns, index=nav_dates[1:], name="daily_return")
    turnover = pd.Series(turnovers, index=nav_dates[1:], name="sell_turnover")
    return {"nav": nav, "daily_return": daily_return, "turnover": turnover,
            "metrics": _metrics(nav, daily_return, turnover),
            "selections": pd.DataFrame(selected_rows), "ic_weights": weights,
            "metadata": {"execution": execution, "method": method, "lookback": lookback,
                         "top_n": top_n, "sell_fee": sell_fee,
                         "factor_count": len(factors),
                         "factor_names": ",".join(factors),
                         "selection_policy": selection_policy, "buffer_n": buffer_n,
                         "max_replacements": max_replacements,
                         "min_liquidity_quantile": min_liquidity_quantile,
                         "max_volatility_quantile": max_volatility_quantile,
                         "trade_start_date": str(trade_start.date()) if trade_start is not None else None,
                         "adjusted_prices": bool(APPLY_ADJFACTOR)}}


def save_strict_results(results: dict[str, dict], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or OUTPUT_DIR / "backtest_strict"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, result in results.items():
        frame = pd.concat([result["nav"], result["daily_return"], result["turnover"]], axis=1)
        frame.to_csv(out_dir / f"{label}_daily.csv", float_format="%.8f")
        result["selections"].to_csv(out_dir / f"{label}_selections.csv", index=False)
        result["ic_weights"].to_csv(out_dir / f"{label}_ic_weights.csv", float_format="%.6f")
        rows.append({"strategy": label, **result["metadata"], **result["metrics"]})
    metrics = pd.DataFrame(rows); metrics.to_csv(out_dir / "metrics.csv", index=False, float_format="%.8f")
    (out_dir / "metadata.json").write_text(json.dumps({
        "price_adjustment": "not applied: anonymized stock codes cannot be mapped safely" if not APPLY_ADJFACTOR else "applied",
        "signal_timing": "factor at close t, execution at t+1 open/close",
        "ic_timing": "only IC labels fully realized by decision date are used",
    }, indent=2), encoding="utf-8")
    _plot(results, out_dir)
    return out_dir


def _plot(results: dict[str, dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot executable strategies separately.  Putting the deliberately leaked
    # naive control (20x+ NAV) on the same linear axis made the real curves look
    # blank even though they contained valid data.
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, result in results.items():
        if label.startswith("naive_"):
            continue
        ax.plot(result["nav"].index, result["nav"] / result["nav"].iloc[0], label=label, linewidth=1.2)
    ax.axhline(1, color="grey", lw=1, ls=":")
    ax.set(title="Strict Task 4 NAV (leakage-free strategies)", ylabel="NAV", xlabel="Date")
    ax.grid(alpha=.3); ax.legend(ncol=2, fontsize=8); fig.tight_layout(); fig.savefig(out_dir / "nav_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in ("adaptive_close", "adaptive_open", "naive_close", "naive_open"):
        result = results[label]
        ax.plot(result["nav"].index, result["nav"] / result["nav"].iloc[0], label=label, linewidth=1.2)
    ax.set_yscale("log")
    ax.axhline(1, color="grey", lw=1, ls=":")
    ax.set(title="Look-ahead bias comparison (log scale)", ylabel="NAV, log scale", xlabel="Date")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out_dir / "lookahead_bias_comparison.png", dpi=150)
    plt.close(fig)


def run_all_strict(
    lookback: int = 60,
    top_n: int = 10,
    factor_set: str = "required",
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    if factor_set not in {"required", "extended"}:
        raise ValueError("factor_set must be required or extended")
    daily = load_daily_data()
    factor_bundle = compute_all_factors(daily)
    all_factors = {k: v for k, v in factor_bundle.items() if k != "forward_return_1d"}
    factors = select_required_factors(all_factors) if factor_set == "required" else all_factors
    results = {}
    for execution in ("close", "open"):
        for method in ("rolling", "expanding", "hybrid", "adaptive", "naive"):
            label = f"{method}_{execution}"
            print(f"Running {label}...")
            results[label] = run_strict_backtest(
                factors, daily, execution=execution, method=method,
                lookback=lookback, top_n=top_n,
            )
    out_dir = out_dir or OUTPUT_DIR / (
        "backtest_required" if factor_set == "required" else "backtest_strict"
    )
    out_dir = save_strict_results(results, out_dir)
    metadata_path = out_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "factor_set": factor_set,
        "factor_count": len(factors),
        "factor_names": list(factors),
    })
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metrics = pd.read_csv(out_dir / "metrics.csv")
    print(metrics[["strategy", "annual_return_cagr", "sharpe_ratio", "max_drawdown", "average_sell_turnover"]].to_string(index=False))
    return metrics, out_dir
