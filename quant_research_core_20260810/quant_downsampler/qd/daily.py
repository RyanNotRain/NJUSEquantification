"""Daily aggregation of tick trades with normalized price adjustment."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .adjfactor import full_stock_code, get_adjust_ratio
from .config import METRICS, PRICE_METRICS
from .data_loader import load_one_day


def calc_daily_metrics(df: pd.DataFrame, adjust_ratio: float = 1.0) -> dict[str, float]:
    if df is None or df.empty:
        return {metric: (np.nan if metric in PRICE_METRICS else 0.0) for metric in METRICS}
    ordered = df.sort_values("time_sec", kind="mergesort")
    raw_price = ordered["Price"].to_numpy(dtype=np.float64) / 100.0
    adjusted_price = raw_price * adjust_ratio
    volume = ordered["Volume"].to_numpy(dtype=np.int64)
    flag = ordered["BSFlag"].to_numpy(dtype=np.int8)
    # Amount is the actually transacted currency amount and is not adjusted.
    amount = raw_price * volume
    buy, sell = flag == 0, flag == 1
    return {
        "open": float(adjusted_price[0]), "high": float(adjusted_price.max()),
        "low": float(adjusted_price.min()), "close": float(adjusted_price[-1]),
        "volume": float(volume.sum()), "trade_count": float(len(ordered)),
        "amount": float(amount.sum()), "buy_volume": float(volume[buy].sum()),
        "sell_volume": float(volume[sell].sum()), "buy_amount": float(amount[buy].sum()),
        "sell_amount": float(amount[sell].sum()),
    }


def aggregate_day(date: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for data_code, frame in load_one_day(date).items():
        result[full_stock_code(data_code)] = calc_daily_metrics(
            frame, get_adjust_ratio(data_code, date),
        )
    return result


def _fill_daily_prices(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close_filled = raw["close"].ffill(axis=0)
    previous_close = close_filled.shift(1)
    filled: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        if metric in PRICE_METRICS:
            table = raw[metric].where(raw[metric].notna(), previous_close)
            filled[metric] = table.where(table.notna(), close_filled)
        else:
            filled[metric] = raw[metric].fillna(0.0)
    return filled


def build_daily_tables(dates: Iterable[str]) -> dict[str, pd.DataFrame]:
    dates = sorted(dict.fromkeys(dates))
    if not dates:
        raise ValueError("dates cannot be empty")
    per_date: dict[str, dict[str, dict[str, float]]] = {}
    stocks: set[str] = set()
    for trading_date in dates:
        per_date[trading_date] = aggregate_day(trading_date)
        stocks.update(per_date[trading_date])
    ordered_stocks = sorted(stocks)
    raw: dict[str, pd.DataFrame] = {}
    iso_dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dates]
    for metric in METRICS:
        values = [
            [per_date[trading_date].get(stock, {}).get(metric, np.nan) for stock in ordered_stocks]
            for trading_date in dates
        ]
        table = pd.DataFrame(values, index=iso_dates, columns=ordered_stocks, dtype=np.float64)
        table.index.name = "date"
        table.columns.name = "stock"
        raw[metric] = table
    return _fill_daily_prices(raw)
