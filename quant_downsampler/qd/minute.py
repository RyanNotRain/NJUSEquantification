"""Minute aggregation matching the README's displayed trading-time rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adjfactor import full_stock_code, get_adjust_ratio
from .config import (
    CONTINUOUS_AFTERNOON_END_SEC,
    METRICS,
    MORNING_BARS,
    PRICE_METRICS,
)
from .data_loader import load_one_day
from .time_utils import MINUTE_BAR_LABELS


def calc_minute_metrics(
    df: pd.DataFrame,
    adjust_ratio: float = 1.0,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {m: np.nan if m in PRICE_METRICS else 0.0 for m in METRICS},
        index=range(len(MINUTE_BAR_LABELS)),
        dtype=np.float64,
    )
    if df is None or df.empty:
        return out

    trades = df[df["minute_bar"] >= 0].copy()
    if trades.empty:
        return out
    price_raw = trades["Price"].to_numpy(dtype=np.float64) / 100.0
    price_adjusted = price_raw * adjust_ratio
    volume = trades["Volume"].to_numpy(dtype=np.float64)
    flags = trades["BSFlag"].to_numpy(dtype=np.int8)
    amount = price_raw * volume
    work = pd.DataFrame({
        "minute_bar": trades["minute_bar"].to_numpy(dtype=np.int32),
        "time_sec": trades["time_sec"].to_numpy(dtype=np.float64),
        "price": price_adjusted,
        "volume": volume,
        "trade_count": np.ones(len(trades), dtype=np.float64),
        "amount": amount,
        "buy_volume": np.where(flags == 0, volume, 0.0),
        "sell_volume": np.where(flags == 1, volume, 0.0),
        "buy_amount": np.where(flags == 0, amount, 0.0),
        "sell_amount": np.where(flags == 1, amount, 0.0),
    }).sort_values(["minute_bar", "time_sec"], kind="mergesort")
    agg = work.groupby("minute_bar", sort=True).agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
        amount=("amount", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        buy_amount=("buy_amount", "sum"),
        sell_amount=("sell_amount", "sum"),
    )
    out.loc[agg.index, agg.columns] = agg
    return out


def aggregate_day_minute(date: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for data_code, df in load_one_day(date).items():
        result[full_stock_code(data_code)] = calc_minute_metrics(
            df, get_adjust_ratio(data_code, date)
        )
    return result


def _fill_continuous_prices(
    raw_prices: dict[str, pd.DataFrame],
    prev_close: dict[str, float] | None,
) -> dict[str, pd.DataFrame]:
    """Fill missing continuous-session OHLC with the previous minute close.

    14:57-14:59 are call-auction waiting rows and deliberately remain NaN.
    The 15:00 row contains only actually matched closing-auction prices.
    """
    close = raw_prices["close"].copy()
    if prev_close:
        for stock, value in prev_close.items():
            if stock in close.columns and pd.notna(value) and pd.isna(close.iloc[0][stock]):
                close.iloc[0, close.columns.get_loc(stock)] = float(value)

    # Fill morning and continuous afternoon separately, carrying the morning
    # close into 13:00 when necessary.
    close.iloc[:MORNING_BARS] = close.iloc[:MORNING_BARS].ffill(axis=0)
    afternoon_continuous_count = int((CONTINUOUS_AFTERNOON_END_SEC - 13 * 3600) // 60)
    afternoon_start = MORNING_BARS
    afternoon_stop = afternoon_start + afternoon_continuous_count
    seed = close.iloc[MORNING_BARS - 1]
    close.iloc[afternoon_start] = close.iloc[afternoon_start].fillna(seed)
    close.iloc[afternoon_start:afternoon_stop] = close.iloc[
        afternoon_start:afternoon_stop
    ].ffill(axis=0)

    previous = close.shift(1)
    if prev_close:
        for stock, value in prev_close.items():
            if stock in previous.columns and pd.notna(value):
                previous.iloc[0, previous.columns.get_loc(stock)] = float(value)
    previous.iloc[afternoon_start] = seed
    filled: dict[str, pd.DataFrame] = {"close": close}
    for metric in ("open", "high", "low"):
        table = raw_prices[metric].copy()
        table.iloc[:MORNING_BARS] = table.iloc[:MORNING_BARS].where(
            table.iloc[:MORNING_BARS].notna(), previous.iloc[:MORNING_BARS]
        )
        table.iloc[afternoon_start:afternoon_stop] = table.iloc[
            afternoon_start:afternoon_stop
        ].where(
            table.iloc[afternoon_start:afternoon_stop].notna(),
            previous.iloc[afternoon_start:afternoon_stop],
        )
        filled[metric] = table
    return filled


def build_minute_table_for_date(
    date: str,
    prev_close: dict[str, float] | None = None,
    all_stocks: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    per_stock = aggregate_day_minute(date)
    stocks = sorted(set(all_stocks or []) | set(per_stock))
    if not stocks:
        return {}

    timestamp_index = pd.to_datetime(
        [f"{pd.Timestamp(date).strftime('%Y-%m-%d')} {t}" for t in MINUTE_BAR_LABELS]
    )
    raw: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        data = {
            stock: (
                per_stock[stock][metric].to_numpy()
                if stock in per_stock
                else np.full(len(timestamp_index), np.nan)
            )
            for stock in stocks
        }
        table = pd.DataFrame(data, index=timestamp_index, dtype=np.float64)
        table.index.name = "datetime"
        table.columns.name = "stock"
        raw[metric] = table

    filled_prices = _fill_continuous_prices(
        {m: raw[m] for m in PRICE_METRICS}, prev_close
    )
    out: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        out[metric] = (
            filled_prices[metric]
            if metric in PRICE_METRICS
            else raw[metric].fillna(0.0)
        ).astype(np.float64)
    return out
