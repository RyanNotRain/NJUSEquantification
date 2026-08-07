"""Daily aggregation of tick trades."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .adjfactor import full_stock_code, get_adjust_ratio
from .config import METRICS, PRICE_METRICS
from .data_loader import load_one_day


def calc_daily_metrics(
    df: pd.DataFrame,
    adjust_ratio: float = 1.0,
) -> dict[str, float]:
    if df is None or df.empty:
        return {m: (np.nan if m in PRICE_METRICS else 0.0) for m in METRICS}

    ordered = df.sort_values("time_sec", kind="mergesort")
    price_raw = ordered["Price"].to_numpy(dtype=np.float64) / 100.0
    price_adjusted = price_raw * adjust_ratio
    volume = ordered["Volume"].to_numpy(dtype=np.int64)
    bsflag = ordered["BSFlag"].to_numpy(dtype=np.int8)
    # Turnover is an actually transacted amount and is not price-adjusted.
    amount = price_raw * volume
    buy = bsflag == 0
    sell = bsflag == 1
    return {
        "open": float(price_adjusted[0]),
        "high": float(price_adjusted.max()),
        "low": float(price_adjusted.min()),
        "close": float(price_adjusted[-1]),
        "volume": float(volume.sum()),
        "trade_count": float(len(ordered)),
        "amount": float(amount.sum()),
        "buy_volume": float(volume[buy].sum()),
        "sell_volume": float(volume[sell].sum()),
        "buy_amount": float(amount[buy].sum()),
        "sell_amount": float(amount[sell].sum()),
    }


def aggregate_day(date: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for data_code, df in load_one_day(date).items():
        result[full_stock_code(data_code)] = calc_daily_metrics(
            df, get_adjust_ratio(data_code, date)
        )
    return result


def _fill_daily_prices(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fill every missing OHLC value with the previous trading day's close."""
    close_filled = raw["close"].ffill(axis=0)
    previous_close = close_filled.shift(1)
    filled: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        if metric in PRICE_METRICS:
            table = raw[metric].where(raw[metric].notna(), previous_close)
            # A multi-day suspension needs the last known close on every day.
            table = table.where(table.notna(), close_filled)
            filled[metric] = table
        else:
            filled[metric] = raw[metric].fillna(0.0)
    return filled


def build_daily_tables(dates: Iterable[str]) -> dict[str, pd.DataFrame]:
    dates = sorted(dict.fromkeys(dates))
    if not dates:
        raise ValueError("dates cannot be empty")

    per_date: dict[str, dict[str, dict[str, float]]] = {}
    all_stocks: set[str] = set()
    for date in dates:
        per_date[date] = aggregate_day(date)
        all_stocks.update(per_date[date])
    stocks = sorted(all_stocks)

    raw: dict[str, pd.DataFrame] = {}
    iso_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    for metric in METRICS:
        values = [
            [per_date[d].get(s, {}).get(metric, np.nan) for s in stocks]
            for d in dates
        ]
        table = pd.DataFrame(values, index=iso_dates, columns=stocks, dtype=np.float64)
        table.index.name = "date"
        table.columns.name = "stock"
        raw[metric] = table
    return _fill_daily_prices(raw)
