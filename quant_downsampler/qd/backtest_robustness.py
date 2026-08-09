"""Turnover-aware, cost-stressed evaluation for the corrected factor strategy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import combined_signal, compute_daily_ic, corrected_weights
from .config import OUTPUT_DIR
from .evaluation import compute_ic_series
from .factor_robustness import (
    average_cross_sectional_factor_correlation,
    block_bootstrap_mean_ci,
    compute_ic_table,
)
from .factors import (
    EXPERIMENTAL_FACTOR_NAMES,
    FACTOR_NAMES,
    compute_all_factors,
    load_daily_data,
)


MARKET_PROXY_NAME = "sample_universe_equal_weight"


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


def _validate_factor_names(
    factors: Mapping[str, pd.DataFrame],
    factor_names: Sequence[str] | None,
) -> tuple[str, ...]:
    selected = tuple(FACTOR_NAMES if factor_names is None else factor_names)
    if not selected:
        raise ValueError("at least one factor name is required")
    if len(set(selected)) != len(selected):
        raise ValueError("factor names cannot contain duplicates")
    missing = sorted(set(selected).difference(factors))
    if missing:
        raise ValueError(f"missing requested factors: {missing}")
    return selected


def history_only_factor_weights(
    factors: Mapping[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    factor_names: Sequence[str],
    lookback: int = 60,
) -> pd.DataFrame:
    """Return rolling-IC weights that are knowable before each decision date.

    The original four-factor path deliberately reuses the published IC helper.
    Custom/experimental paths compute exactly the requested columns and then
    apply ``corrected_weights``, whose two-date lag means date ``t`` can use IC
    only through ``t-2``.  No full-sample sign is ever consulted.
    """
    selected = _validate_factor_names(factors, factor_names)
    if selected == FACTOR_NAMES:
        daily_ic = compute_daily_ic(dict(factors), forward_return)
    else:
        daily_ic = pd.DataFrame({
            name: compute_ic_series(
                factors[name], forward_return, method="pearson"
            )
            for name in selected
        }).reindex(forward_return.index)
    return corrected_weights(daily_ic, lookback=lookback)


def _combined_signal_for_names(
    factors: Mapping[str, pd.DataFrame],
    weights: pd.Series,
    signal_date: pd.Timestamp,
    factor_names: Sequence[str],
) -> pd.Series:
    """Cross-sectionally standardise and combine an explicit factor set."""
    selected = set(factor_names)
    active = weights.dropna()
    active = active[active.index.isin(selected)]
    if active.empty or active.abs().sum() == 0.0:
        return pd.Series(dtype=np.float64)
    active = active / active.abs().sum()
    pieces: list[pd.Series] = []
    for name, weight in active.items():
        row = factors[str(name)].loc[signal_date].replace(
            [np.inf, -np.inf], np.nan
        )
        standard_deviation = float(row.std(ddof=1))
        if not np.isfinite(standard_deviation) or standard_deviation == 0.0:
            continue
        pieces.append(((row - row.mean()) / standard_deviation) * float(weight))
    if not pieces:
        return pd.Series(dtype=np.float64)
    signal = pieces[0]
    for piece in pieces[1:]:
        signal = signal.add(piece, fill_value=0.0)
    return signal.sort_values(ascending=False)


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
    factor_names: Sequence[str] | None = None,
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

    selected_factors = _validate_factor_names(factors, factor_names)
    dates = close.index
    price = close if rebalance_at == "close" else open_price
    weights = history_only_factor_weights(
        factors, forward_return, selected_factors, lookback=lookback
    )
    trailing_amount, trailing_volatility = _historical_filters(
        close, amount, liquidity_window, volatility_window
    )
    previous: set[str] = set()
    rows: list[dict] = []
    selections: list[dict] = []
    weight_history: list[dict] = []

    for i in range(1, len(dates) - 1):
        execution_date = dates[i]
        next_date = dates[i + 1]
        signal_date = dates[i - 1]
        daily_weights = weights.loc[execution_date]
        if daily_weights.notna().sum() == 0:
            continue
        if selected_factors == FACTOR_NAMES:
            signal = combined_signal(factors, daily_weights, signal_date)
        else:
            signal = _combined_signal_for_names(
                factors, daily_weights, signal_date, selected_factors
            )
        if signal.empty:
            continue
        for factor_name in selected_factors:
            historical_weight = daily_weights.get(factor_name, np.nan)
            if pd.notna(historical_weight):
                weight_history.append({
                    "execution_date": execution_date,
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "historical_ic_weight": float(historical_weight),
                    "history_only_direction": int(np.sign(historical_weight)),
                })
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
        "weight_history": pd.DataFrame(weight_history),
        "factor_names": selected_factors,
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


def build_equal_weight_market_proxy(
    close: pd.DataFrame,
    period_dates: pd.DatetimeIndex | Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Build the same-universe equal-weight close-to-close market proxy.

    A row dated ``t`` is the cross-sectional mean of each valid stock's
    ``close[t] / close[t-1] - 1``.  The requested dates must be unique,
    increasing, and present in the close table; this deliberately rejects
    silent date filling or nearest-date alignment.
    """
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError("close must use a DatetimeIndex")
    if close.empty or close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("close must have non-empty unique date and stock axes")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close dates must be increasing")
    dates = pd.DatetimeIndex(period_dates)
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("period dates must be non-empty, unique, and increasing")
    missing = dates.difference(close.index)
    if not missing.empty:
        shown = ", ".join(str(date.date()) for date in missing[:3])
        raise ValueError(f"benchmark dates are absent from close data: {shown}")

    stock_returns = close.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    ).loc[dates]
    valid_count = stock_returns.notna().sum(axis=1).astype(int)
    market_return = stock_returns.mean(axis=1, skipna=True)
    if (valid_count <= 0).any() or not np.isfinite(market_return.to_numpy(float)).all():
        raise ValueError("every benchmark period must have at least one valid stock return")
    return pd.DataFrame({
        "market_return": market_return.astype(float),
        "n_market_stocks": valid_count,
    }, index=dates)


def evaluate_strategy_against_market(
    strategy_returns: pd.Series,
    close: pd.DataFrame,
    strategy_name: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str | None]]:
    """Evaluate one strategy against the exactly aligned market proxy."""
    if not isinstance(strategy_returns.index, pd.DatetimeIndex):
        raise TypeError("strategy returns must use a DatetimeIndex")
    if (
        strategy_returns.empty
        or strategy_returns.index.has_duplicates
        or not strategy_returns.index.is_monotonic_increasing
    ):
        raise ValueError("strategy returns must be non-empty, unique, and increasing")
    clean_strategy = pd.to_numeric(strategy_returns, errors="coerce").astype(float)
    values = clean_strategy.to_numpy(float)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("strategy returns must be finite and greater than -100%")
    benchmark = build_equal_weight_market_proxy(close, clean_strategy.index)
    if not benchmark.index.equals(clean_strategy.index):
        raise RuntimeError("strategy and benchmark date alignment failed")

    market_return = benchmark["market_return"]
    strategy_nav = (1.0 + clean_strategy).cumprod()
    benchmark_nav = (1.0 + market_return).cumprod()
    relative_wealth = strategy_nav / benchmark_nav
    active_return = clean_strategy - market_return
    active_std = float(active_return.std(ddof=1))
    tracking_error = (
        active_std * np.sqrt(252.0)
        if len(active_return) > 1 and np.isfinite(active_std)
        else np.nan
    )
    information_ratio = (
        float(active_return.mean() / active_std * np.sqrt(252.0))
        if np.isfinite(active_std) and active_std > 0.0 else np.nan
    )
    relative_with_initial = np.concatenate(
        ([1.0], relative_wealth.to_numpy(dtype=float))
    )
    relative_drawdown = (
        relative_with_initial / np.maximum.accumulate(relative_with_initial) - 1.0
    )
    strategy_total = float(strategy_nav.iloc[-1] - 1.0)
    benchmark_total = float(benchmark_nav.iloc[-1] - 1.0)
    periods = pd.DataFrame({
        "strategy": strategy_name,
        "strategy_net_return": clean_strategy,
        "market_return": market_return,
        "active_return": active_return,
        "strategy_nav": strategy_nav,
        "benchmark_nav": benchmark_nav,
        "relative_wealth": relative_wealth,
        "n_market_stocks": benchmark["n_market_stocks"],
    }, index=clean_strategy.index)
    periods.index.name = "date"
    metrics: dict[str, float | int | str | None] = {
        "strategy": strategy_name,
        "date_start": str(clean_strategy.index.min().date()),
        "date_end": str(clean_strategy.index.max().date()),
        "periods": int(len(clean_strategy)),
        "strategy_total_return": strategy_total,
        "benchmark_total_return": benchmark_total,
        "total_return_difference": strategy_total - benchmark_total,
        "percentage_point_lead": (strategy_total - benchmark_total) * 100.0,
        "relative_wealth": float(relative_wealth.iloc[-1]),
        "relative_wealth_gain": float(relative_wealth.iloc[-1] - 1.0),
        "tracking_error": float(tracking_error) if np.isfinite(tracking_error) else None,
        "information_ratio": (
            float(information_ratio) if np.isfinite(information_ratio) else None
        ),
        "relative_max_drawdown": float(np.min(relative_drawdown)),
        "outperformance_day_ratio": float((active_return > 0.0).mean()),
        "mean_market_stock_count": float(benchmark["n_market_stocks"].mean()),
        "minimum_market_stock_count": int(benchmark["n_market_stocks"].min()),
    }
    return periods, metrics


def factor_diagnostics_for_market_report(
    factors: Mapping[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    *,
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20250809,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int | str | bool | None]]]:
    """Return IC diagnostics without changing the official evaluation table."""
    factor_names = (*FACTOR_NAMES, *EXPERIMENTAL_FACTOR_NAMES)
    selected = {name: factors[name] for name in factor_names}
    ic_daily = compute_ic_table(selected, forward_return, method="pearson")
    rank_ic_daily = compute_ic_table(selected, forward_return, method="spearman")
    bootstrap = block_bootstrap_mean_ci(
        ic_daily,
        iterations=bootstrap_iterations,
        block_length=5,
        confidence=0.95,
        random_seed=bootstrap_seed,
    )
    correlation, _ = average_cross_sectional_factor_correlation(
        selected, method="spearman"
    )
    q2_mask = ic_daily.index.to_period("Q") == pd.Period("2026Q2", freq="Q")
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, dict[str, float | int | str | bool | None]] = {}
    for name in factor_names:
        official_others = [factor for factor in FACTOR_NAMES if factor != name]
        official_correlations = correlation.loc[name, official_others].dropna()
        if official_correlations.empty:
            most_correlated = None
            max_correlation = np.nan
        else:
            most_correlated = str(official_correlations.abs().idxmax())
            max_correlation = float(official_correlations.loc[most_correlated])
        full_ic = ic_daily[name].dropna()
        full_rank_ic = rank_ic_daily[name].dropna()
        q2_ic = ic_daily.loc[q2_mask, name].dropna()
        boot = bootstrap.loc[name]
        scope = "experimental" if name in EXPERIMENTAL_FACTOR_NAMES else "official"
        row = {
            "factor": name,
            "factor_scope": scope,
            "full_sample_mean_ic": float(full_ic.mean()) if not full_ic.empty else np.nan,
            "full_sample_mean_rank_ic": (
                float(full_rank_ic.mean()) if not full_rank_ic.empty else np.nan
            ),
            "block_bootstrap_5d_ci_lower": float(boot["ci_lower"]),
            "block_bootstrap_5d_ci_upper": float(boot["ci_upper"]),
            "block_bootstrap_5d_p_value": float(boot["two_sided_p_value"]),
            "block_bootstrap_5d_significant": bool(boot["significant_at_5pct"]),
            "block_bootstrap_iterations": int(boot["iterations"]),
            "block_bootstrap_block_length": int(boot["block_length"]),
            "mean_ic_2026q2": float(q2_ic.mean()) if not q2_ic.empty else np.nan,
            "most_correlated_official_factor": most_correlated,
            "max_average_cross_sectional_spearman_with_official": max_correlation,
            "max_abs_average_cross_sectional_spearman_with_official": (
                abs(max_correlation) if np.isfinite(max_correlation) else np.nan
            ),
            "n_ic_days": int(len(full_ic)),
        }
        rows.append(row)
        diagnostics[name] = {
            key: (
                None
                if isinstance(value, (float, np.floating)) and not np.isfinite(value)
                else value.item() if isinstance(value, np.generic) else value
            )
            for key, value in row.items()
            if key not in {"factor", "factor_scope"}
        }
        diagnostics[name]["factor_scope"] = scope
    return pd.DataFrame(rows).set_index("factor"), diagnostics


def _plot_factor_vs_market(
    benchmark_periods: pd.DataFrame,
    target: Path,
) -> None:
    """Plot all strategies on their common exactly aligned evaluation dates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = benchmark_periods.pivot(
        index="date", columns="strategy", values="strategy_net_return"
    ).sort_index()
    common = returns.dropna(axis=0, how="any")
    if common.empty:
        raise ValueError("strategies have no common dates for the market plot")
    market_by_date = benchmark_periods.groupby("date")["market_return"]
    if (market_by_date.max() - market_by_date.min()).abs().max() > 1e-12:
        raise RuntimeError("market proxy differs across strategies on the same date")
    market = market_by_date.first().reindex(common.index)
    if market.isna().any():
        raise RuntimeError("market plot alignment failed")
    strategy_nav = (1.0 + common).cumprod()
    benchmark_nav = (1.0 + market).cumprod()

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(
        benchmark_nav.index, benchmark_nav,
        color="black", linewidth=2.5, label="300-stock equal-weight proxy",
    )
    for name in strategy_nav:
        linewidth = 2.0 if "combined" in str(name) else 1.2
        axes[0].plot(strategy_nav.index, strategy_nav[name], label=name, linewidth=linewidth)
        axes[1].plot(
            strategy_nav.index,
            strategy_nav[name] / benchmark_nav,
            label=name,
            linewidth=linewidth,
        )
    axes[0].set_title("Cost-adjusted factor strategies vs sample-universe market proxy")
    axes[0].set_ylabel("Normalised wealth")
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_title("Relative wealth (strategy / market proxy)")
    axes[1].set_ylabel("Relative wealth")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(target / "factor_vs_market.png", dpi=170)
    plt.close(figure)


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
    bootstrap_iterations: int = 2_000,
    bootstrap_seed: int = 20250809,
) -> dict:
    if bootstrap_iterations < 100:
        raise ValueError("bootstrap_iterations must be at least 100")
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

    # Single-factor portfolios use the same unbuffered Top-N construction as
    # the published-style combined strategy.  Their sign is recomputed from
    # lagged rolling IC at every date; the full-sample IC sign is diagnostic
    # only and never enters holdings.
    single_factor_results: dict[str, dict] = {}
    single_factor_priced: dict[str, dict] = {}
    for factor_name in (*FACTOR_NAMES, *EXPERIMENTAL_FACTOR_NAMES):
        factor_result = run_turnover_aware_backtest(
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
            factor_names=(factor_name,),
        )
        single_factor_results[factor_name] = factor_result
        single_factor_priced[factor_name] = apply_transaction_costs(
            factor_result, sell_cost=0.0005, buy_cost=0.0
        )

    comparison_inputs: dict[str, pd.Series] = {
        "official_combined": baseline_mandated["periods"]["net_return"],
        "turnover_aware_combined": mandated["periods"]["net_return"],
        **{
            f"single_{name}": priced["periods"]["net_return"]
            for name, priced in single_factor_priced.items()
        },
    }
    comparison_frames: list[pd.DataFrame] = []
    market_strategy_metrics: dict[str, dict[str, object]] = {}
    for strategy_name, returns in comparison_inputs.items():
        periods, metrics = evaluate_strategy_against_market(
            returns, data["close"], strategy_name
        )
        comparison_frames.append(periods.reset_index())
        market_strategy_metrics[strategy_name] = metrics
    benchmark_periods = pd.concat(comparison_frames, ignore_index=True)
    diagnostic_table, factor_diagnostics = factor_diagnostics_for_market_report(
        factors,
        factors["forward_return_1d"],
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    benchmark_report = {
        "benchmark_definition": {
            "name": MARKET_PROXY_NAME,
            "universe": "all 300 sample stocks with a valid close-to-close return on each date",
            "return_definition": (
                "cross-sectional arithmetic mean of close[t] / close[t-1] - 1; "
                "the row is labelled t"
            ),
            "date_alignment": (
                "each strategy is compared only on its exact realised-return dates; "
                "missing or nearest-date alignment is rejected"
            ),
            "benchmark_costs": "none; this is an analytical equal-weight market proxy",
            "not_an_index_claim": (
                "the proxy is not labelled CSI 300 or any external market index"
            ),
        },
        "strategy_cost_definition": (
            "fixed holdings repriced with the assignment convention: sell 5 bps, buy 0; "
            "terminal liquidation is included"
        ),
        "single_factor_definition": (
            "unbuffered Top-N portfolio; direction and scale use only rolling IC "
            "available through t-2; no full-sample sign is used"
        ),
        "comparability_note": (
            "factor availability can produce different strategy start dates; every "
            "metric uses that strategy's exactly matched market dates, while "
            "factor_vs_market.png restricts every line to the common date intersection"
        ),
        "official_factor_names": list(FACTOR_NAMES),
        "experimental_factor_names": list(EXPERIMENTAL_FACTOR_NAMES),
        "experimental_scope_note": (
            "illiquidity_20d is saved and evaluated as an experiment; it is not one "
            "of the assignment's three required additional factors and is excluded "
            "from the official four-row evaluation summary"
        ),
        "strategies": market_strategy_metrics,
        "factor_diagnostics": factor_diagnostics,
    }

    factors["illiquidity_20d"].to_csv(
        target / "illiquidity_20d.csv", float_format="%.12g"
    )
    benchmark_periods.to_csv(
        target / "benchmark_periods.csv", index=False, float_format="%.10f"
    )
    (target / "benchmark_metrics.json").write_text(
        json.dumps(benchmark_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    diagnostic_table.to_csv(
        target / "single_factor_diagnostics.csv", float_format="%.10f"
    )
    single_factor_metric_rows = []
    history_rows: list[pd.DataFrame] = []
    for factor_name, factor_result in single_factor_results.items():
        strategy_name = f"single_{factor_name}"
        single_factor_metric_rows.append({
            "factor": factor_name,
            "factor_scope": (
                "experimental"
                if factor_name in EXPERIMENTAL_FACTOR_NAMES else "official"
            ),
            **market_strategy_metrics[strategy_name],
            **diagnostic_table.loc[factor_name].to_dict(),
        })
        history = factor_result["weight_history"].copy()
        history.insert(0, "strategy", strategy_name)
        history_rows.append(history)
    pd.DataFrame(single_factor_metric_rows).set_index("factor").to_csv(
        target / "single_factor_market_metrics.csv", float_format="%.10f"
    )
    pd.concat(history_rows, ignore_index=True).to_csv(
        target / "single_factor_history_weights.csv", index=False,
        float_format="%.10f",
    )
    _plot_factor_vs_market(benchmark_periods, target)
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
            "official_factor_names": list(FACTOR_NAMES),
            "experimental_factor_names": list(EXPERIMENTAL_FACTOR_NAMES),
        },
        "published_style_baseline": {
            "mandated_sell_only_5bps": baseline_mandated["metrics"],
            "symmetric_break_even_one_way_bps": baseline_break_even,
        },
        "gross_metrics": result["gross_metrics"],
        "mandated_sell_only_5bps": mandated["metrics"],
        "symmetric_break_even_one_way_bps": break_even,
        "market_benchmark": {
            "definition": benchmark_report["benchmark_definition"],
            "strategy_cost_definition": benchmark_report["strategy_cost_definition"],
            "single_factor_definition": benchmark_report["single_factor_definition"],
            "comparability_note": benchmark_report["comparability_note"],
            "experimental_scope_note": benchmark_report["experimental_scope_note"],
            "metrics_file": "benchmark_metrics.json",
            "periods_file": "benchmark_periods.csv",
            "plot_file": "factor_vs_market.png",
            "strategies": market_strategy_metrics,
        },
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
    market_rows = [
        "| 策略 | 成本后收益 | 同期 300 股代理 | 领先/落后 |",
        "|---|---:|---:|---:|",
    ]
    market_labels = (
        ("正式四因子组合", "official_combined"),
        ("换手约束组合", "turnover_aware_combined"),
        ("示例因子", "single_example_factor"),
        ("5 日动量", "single_momentum_5d"),
        ("主买主卖失衡", "single_buy_sell_imbalance"),
        ("日内振幅", "single_intraday_range"),
        ("实验非流动性因子", "single_illiquidity_20d"),
    )
    for label, strategy_name in market_labels:
        metrics = market_strategy_metrics[strategy_name]
        market_rows.append(
            f"| {label} | {float(metrics['strategy_total_return']):.2%} | "
            f"{float(metrics['benchmark_total_return']):.2%} | "
            f"{float(metrics['percentage_point_lead']):+.2f} 个百分点 |"
        )
    experimental = factor_diagnostics["illiquidity_20d"]
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
            "## 相对样本市场",
            "",
            "市场基准是样本 300 股在策略实际收益日期上的等权 close-to-close 收益，",
            "不是沪深 300 或其他外部指数。策略扣卖出 5 bps 并计入期末退出，代理本身不扣费。",
            "",
            *market_rows,
            "",
            f"`illiquidity_20d` 的平均 IC 为 {float(experimental['full_sample_mean_ic']):.4f}，"
            f"5 日区块 bootstrap `p={float(experimental['block_bootstrap_5d_p_value']):.3f}`；"
            "它成本后未跑赢同期代理，因此仅保留为失败实验，不进入正式四因子。",
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
        "benchmark_periods": benchmark_periods,
        "benchmark_report": benchmark_report,
        "single_factor_results": single_factor_results,
        "single_factor_priced": single_factor_priced,
        "factor_diagnostics": diagnostic_table,
        "summary": summary,
    }
