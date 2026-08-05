"""日频降采样:从一天的 tick 算出 11 个日级指标。

输出格式(readme 描述 + 业内通用):
    日频每个字段一张表(共 11 张),行=日期,列=股票代码。

指标定义:
    open   当日第一笔成交价(Price/100,真实价格)
    high   当日最高成交价
    low    当日最低成交价
    close  当日最后一笔成交价
    volume 当日总成交量(股)
    trade_count 当日成交笔数(tick 数)
    amount 当日总成交额(元,= sum(Price/100 * Volume))
    buy_volume   BSFlag==0 的成交量
    sell_volume  BSFlag==1 的成交量
    buy_amount   BSFlag==0 的成交额
    sell_amount  BSFlag==1 的成交额

填充规则(readme 第 8 条):
    日频(OHLC)价 用前 1 日收盘价填充。
    成交量/笔数/主买主卖额等:0(明确无成交)。
    第一天就停牌的股票:OHLC 留 NaN。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from .config import METRICS
from .data_loader import load_one_day


def calc_daily_metrics(df: pd.DataFrame) -> dict[str, float]:
    """从单只股票单天的 tick DataFrame 算出 11 个日级指标。

    要求 df 已经按 time_sec 升序排序。如果不是,本函数内部会排序。
    """
    if df is None or df.empty:
        return {m: (np.nan if m in {"open", "high", "low", "close"} else 0.0)
                for m in METRICS}

    # 保险:按时间排序
    df = df.sort_values("time_sec", kind="mergesort")

    price_real = df["Price"].to_numpy(dtype=np.int64) / 100.0
    volume = df["Volume"].to_numpy(dtype=np.int64)
    bsflag = df["BSFlag"].to_numpy(dtype=np.int8)
    amount = price_real * volume

    buy_mask = bsflag == 0
    sell_mask = bsflag == 1

    return {
        "open": float(price_real[0]),
        "high": float(price_real.max()),
        "low": float(price_real.min()),
        "close": float(price_real[-1]),
        "volume": float(volume.sum()),
        "trade_count": float(len(df)),
        "amount": float(amount.sum()),
        "buy_volume": float(volume[buy_mask].sum()),
        "sell_volume": float(volume[sell_mask].sum()),
        "buy_amount": float(amount[buy_mask].sum()),
        "sell_amount": float(amount[sell_mask].sum()),
    }


def aggregate_day(date: str) -> dict[str, dict[str, float]]:
    """处理一天的所有股票,返回 {stock_code: {metric: value}}。"""
    day_data = load_one_day(date)
    return {code: calc_daily_metrics(df) for code, df in day_data.items()}


def build_daily_tables(dates: Iterable[str]) -> dict[str, pd.DataFrame]:
    """处理多个交易日,产出 11 张日频宽表(行=日期,列=股票代码)。

    - 行顺序与 dates 一致
    - 列是所有出现过的股票代码的并集(并按字典序)
    - OHLC 用前一日 close 填充(ffill)
    - 成交类指标无数据的填 0
    """
    dates = list(dates)
    if not dates:
        raise ValueError("dates 不能为空")

    # 1) 按日处理,收集每个 (date, stock) 的指标
    per_date: dict[str, dict[str, dict[str, float]]] = {}
    all_stocks: set[str] = set()
    for date in dates:
        per_date[date] = aggregate_day(date)
        all_stocks.update(per_date[date].keys())

    all_stocks_sorted = sorted(all_stocks)

    # 2) 构建 11 张宽表
    raw_tables: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        data = {}
        for date in dates:
            row = {
                code: per_date[date].get(code, {}).get(metric, np.nan)
                for code in all_stocks_sorted
            }
            data[date] = row
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index.name = "date"
        df.columns.name = "stock"
        raw_tables[metric] = df

    # 3) 应用填充规则
    close = raw_tables["close"]
    filled: dict[str, pd.DataFrame] = {}
    for metric in METRICS:
        if metric in {"open", "high", "low", "close"}:
            # OHLC 用前一日 close 填充(沿日期方向 ffill)
            filled[metric] = raw_tables[metric].ffill(axis=0)
        else:
            # 成交类指标:NaN 填 0
            filled[metric] = raw_tables[metric].fillna(0.0)

    # 4) 类型整理(OHLC 用 float,成交类也用 float 方便后续 resample)
    for metric in METRICS:
        filled[metric] = filled[metric].astype(np.float64)

    # 索引排序保证日期单调
    for metric in METRICS:
        filled[metric].sort_index(inplace=True)

    return filled
