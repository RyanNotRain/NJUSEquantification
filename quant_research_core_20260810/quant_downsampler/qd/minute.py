"""Minute aggregation matching the supplied 242 displayed time rows."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adjfactor import full_stock_code, get_adjust_ratio
from .config import CONTINUOUS_AFTERNOON_END_SEC, METRICS, MORNING_BARS, PRICE_METRICS
from .data_loader import load_one_day
from .time_utils import MINUTE_BAR_LABELS


def calc_minute_metrics(df: pd.DataFrame, adjust_ratio: float = 1.0) -> pd.DataFrame:
    output = pd.DataFrame(
        {metric: np.nan if metric in PRICE_METRICS else 0.0 for metric in METRICS},
        index=range(len(MINUTE_BAR_LABELS)), dtype=np.float64,
    )
    if df is None or df.empty:
        return output
    trades = df[df["minute_bar"] >= 0].copy()
    if trades.empty:
        return output
    raw_price = trades["Price"].to_numpy(dtype=np.float64) / 100.0
    adjusted_price = raw_price * adjust_ratio
    volume = trades["Volume"].to_numpy(dtype=np.float64)
    flags = trades["BSFlag"].to_numpy(dtype=np.int8)
    amount = raw_price * volume
    work = pd.DataFrame({
        "minute_bar": trades["minute_bar"].to_numpy(dtype=np.int32),
        "time_sec": trades["time_sec"].to_numpy(dtype=np.float64),
        "price": adjusted_price, "volume": volume,
        "trade_count": np.ones(len(trades), dtype=np.float64), "amount": amount,
        "buy_volume": np.where(flags == 0, volume, 0.0),
        "sell_volume": np.where(flags == 1, volume, 0.0),
        "buy_amount": np.where(flags == 0, amount, 0.0),
        "sell_amount": np.where(flags == 1, amount, 0.0),
    }).sort_values(["minute_bar", "time_sec"], kind="mergesort")
    aggregate = work.groupby("minute_bar", sort=True).agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("volume", "sum"),
        trade_count=("trade_count", "sum"), amount=("amount", "sum"),
        buy_volume=("buy_volume", "sum"), sell_volume=("sell_volume", "sum"),
        buy_amount=("buy_amount", "sum"), sell_amount=("sell_amount", "sum"),
    )
    output.loc[aggregate.index, aggregate.columns] = aggregate
    return output


def aggregate_day_minute(date: str) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for data_code, frame in load_one_day(date).items():
        result[full_stock_code(data_code)] = calc_minute_metrics(
            frame, get_adjust_ratio(data_code, date),
        )
    return result


def _fill_continuous_prices(raw: dict[str, pd.DataFrame],
                            prev_close: dict[str, float] | None) -> dict[str, pd.DataFrame]:
    """Fill continuous-session OHLC; auction waiting rows remain NaN."""
    close = raw["close"].copy()
    if prev_close:
        for stock, value in prev_close.items():
            if stock in close.columns and pd.notna(value) and pd.isna(close.iloc[0][stock]):
                close.iloc[0, close.columns.get_loc(stock)] = float(value)
    close.iloc[:MORNING_BARS] = close.iloc[:MORNING_BARS].ffill(axis=0)
    continuous_count = int((CONTINUOUS_AFTERNOON_END_SEC - 13 * 3600) // 60)
    afternoon_start = MORNING_BARS
    afternoon_stop = afternoon_start + continuous_count
    morning_close = close.iloc[MORNING_BARS - 1]
    close.iloc[afternoon_start] = close.iloc[afternoon_start].fillna(morning_close)
    close.iloc[afternoon_start:afternoon_stop] = close.iloc[
        afternoon_start:afternoon_stop
    ].ffill(axis=0)

    previous = close.shift(1)
    if prev_close:
        for stock, value in prev_close.items():
            if stock in previous.columns and pd.notna(value):
                previous.iloc[0, previous.columns.get_loc(stock)] = float(value)
    previous.iloc[afternoon_start] = morning_close
    filled: dict[str, pd.DataFrame] = {"close": close}
    for metric in ("open", "high", "low"):
        table = raw[metric].copy()
        table.iloc[:MORNING_BARS] = table.iloc[:MORNING_BARS].where(
            table.iloc[:MORNING_BARS].notna(), previous.iloc[:MORNING_BARS],
        )
        table.iloc[afternoon_start:afternoon_stop] = table.iloc[
            afternoon_start:afternoon_stop
        ].where(
            table.iloc[afternoon_start:afternoon_stop].notna(),
            previous.iloc[afternoon_start:afternoon_stop],
        )
        filled[metric] = table
    return filled


def build_minute_table_for_date(date: str, prev_close: dict[str, float] | None = None,
                                all_stocks: list[str] | None = None) -> dict[str, pd.DataFrame]:
    per_stock = aggregate_day_minute(date)
    stocks = sorted(set(all_stocks or []) | set(per_stock))
    if not stocks:
        return {}
    timestamps = pd.to_datetime([
        f"{pd.Timestamp(date).strftime('%Y-%m-%d')} {label}" for label in MINUTE_BAR_LABELS
    ])
    raw: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        data = {
            stock: per_stock[stock][metric].to_numpy()
            if stock in per_stock else np.full(len(timestamps), np.nan)
            for stock in stocks
        }
        table = pd.DataFrame(data, index=timestamps, dtype=np.float64)
        table.index.name = "datetime"
        table.columns.name = "stock"
        raw[metric] = table
    filled_prices = _fill_continuous_prices({m: raw[m] for m in PRICE_METRICS}, prev_close)
    return {
        metric: (filled_prices[metric] if metric in PRICE_METRICS else raw[metric].fillna(0.0)).astype(np.float64)
        for metric in METRICS
    }
