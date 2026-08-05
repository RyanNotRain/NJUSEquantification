"""分钟频降采样:从一天的 tick 算出 237 个分钟 bar × 11 个指标。

分钟 bar 覆盖(共 237 个,见 config.TOTAL_BARS):
    9:30, 9:31, ..., 11:30(120 个)
    13:00, 13:01, ..., 14:56(117 个)

输出格式(readme):
    minute/{metric}/{YYYYMMDD}.csv
    每个 metric 一个文件夹,内含每日一张表。
    表内行=分钟标签(9:30, 9:31, ..., 14:56),列=股票代码。

指标定义(同日频,只是把"日"换成"分钟"):
    open/high/low/close: 该分钟 bar 内首笔/最大/最小/末笔成交价
    volume/trade_count/amount: 该分钟 bar 内汇总
    buy_volume/sell_volume/buy_amount/sell_amount: BSFlag==0/1 的子集

集合竞价(BSFlag==2)不进入分钟 bar。

填充规则(readme 第 9 条):
    连续竞价阶段(高开低收)价 用前 1 分钟收盘价填充。
    (对每只股票单独在 bar 方向 ffill)
    成交量/笔数/主买主卖额:0。
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import METRICS
from .data_loader import load_one_day
from .time_utils import MINUTE_BAR_LABELS


def calc_minute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """从单只股票单天的 tick 算出 237 × 11 的 DataFrame(行=bar idx,列=metric)。

    返回的 DataFrame 索引是 0..236,列是 METRICS。
    """
    out = pd.DataFrame(
        {m: np.nan if m in {"open", "high", "low", "close"} else 0.0
         for m in METRICS},
        index=range(len(MINUTE_BAR_LABELS)),
    )

    if df is None or df.empty:
        return out

    # 过滤集合竞价(bar == -1 的 tick)
    df = df[df["minute_bar"] >= 0]
    if df.empty:
        return out

    # 计算派生列(向量化,避免 groupby 多次扫描)
    price_real = df["Price"].astype(np.float64) / 100.0
    amount = price_real * df["Volume"].astype(np.float64)
    bsflag = df["BSFlag"].to_numpy()
    is_buy = bsflag == 0
    is_sell = bsflag == 1

    work = pd.DataFrame({
        "minute_bar": df["minute_bar"].to_numpy(),
        "price_real": price_real.to_numpy(),
        "volume": df["Volume"].to_numpy(dtype=np.float64),
        "trade_count": np.ones(len(df), dtype=np.float64),
        "amount": amount.to_numpy(),
        "buy_volume": np.where(is_buy, df["Volume"].to_numpy(), 0.0),
        "sell_volume": np.where(is_sell, df["Volume"].to_numpy(), 0.0),
        "buy_amount": np.where(is_buy, amount.to_numpy(), 0.0),
        "sell_amount": np.where(is_sell, amount.to_numpy(), 0.0),
    })

    # 按 minute_bar 分组聚合
    # 注意:为了保证"first"和"last"是 bar 内的首末笔,先按 time_sec 排
    work = work.join(df["time_sec"].reset_index(drop=True))
    work = work.sort_values(["minute_bar", "time_sec"], kind="mergesort")

    grouped = work.groupby("minute_bar", sort=True)

    agg = grouped.agg(
        open=("price_real", "first"),
        high=("price_real", "max"),
        low=("price_real", "min"),
        close=("price_real", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
        amount=("amount", "sum"),
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        buy_amount=("buy_amount", "sum"),
        sell_amount=("sell_amount", "sum"),
    )

    # 写回 out(未覆盖的位置保持 NaN/0)
    out.loc[agg.index, agg.columns] = agg
    return out


def aggregate_day_minute(date: str) -> dict[str, pd.DataFrame]:
    """处理一天的所有股票,返回 {stock_code: 237×11 DataFrame}。"""
    day_data = load_one_day(date)
    return {code: calc_minute_metrics(df) for code, df in day_data.items()}


def _apply_minute_fill_rules(
    wide: pd.DataFrame, metric: str, prev_close: float | None
) -> pd.DataFrame:
    """对单张宽表(行=分钟 bar,列=股票)应用填充规则。

    - OHLC 沿行方向 ffill(前 1 分钟 close 填充)
    - 成交类:NaN 填 0
    - 如果提供了 prev_close,先用它填 9:30 bar 的剩余 NaN
    """
    if metric in {"open", "high", "low", "close"}:
        if prev_close is not None:
            wide.iloc[0] = wide.iloc[0].fillna(prev_close)
        return wide.ffill(axis=0)
    return wide.fillna(0.0)


def build_minute_table_for_date(
    date: str,
    prev_close: dict[str, float] | None = None,
    all_stocks: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """处理一天的数据,返回 11 张宽表(行=分钟 bar,列=股票代码)。

    Args:
        date: 交易日
        prev_close: 前一交易日每只股票的收盘价(用于 9:30 bar 跨日填充)
        all_stocks: 全局股票列表(列的完整集合,保证每天列数一致)。
            缺失的股票当天该列为 NaN/0。
    """
    per_stock = aggregate_day_minute(date)

    # 决定列集合
    if all_stocks is None:
        all_stocks = sorted(per_stock.keys())
    else:
        # 把当天有数据的也并入,保证不会丢
        all_stocks = sorted(set(all_stocks) | set(per_stock.keys()))

    if not all_stocks:
        return {}

    # 每天为每个 metric 拼一张宽表
    out: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        data: dict[str, np.ndarray] = {}
        for stock in all_stocks:
            if stock in per_stock:
                data[stock] = per_stock[stock][metric].to_numpy()
            else:
                data[stock] = np.full(len(MINUTE_BAR_LABELS), np.nan)
        wide = pd.DataFrame(data, index=MINUTE_BAR_LABELS)
        wide.index.name = "minute"
        wide.columns.name = "stock"

        if metric in {"open", "high", "low", "close"}:
            if prev_close:
                for s in all_stocks:
                    if s in prev_close and prev_close[s] is not None and pd.isna(wide.iloc[0][s]):
                        wide.iloc[0, wide.columns.get_loc(s)] = prev_close[s]
            wide = wide.ffill(axis=0)
        else:
            wide = wide.fillna(0.0)
        out[metric] = wide.astype(np.float64)

    return out
