"""读取并预处理一天的逐笔成交数据。

输入布局: {DATA_DIR}/YYYYMMDD/XXXXXX.csv
每行 4 个字段: Time (HHMMSScc), Price (×100 整数), Volume, BSFlag

输出: dict[stock_code, DataFrame],每只股票一个 DataFrame,
      含以下列:
        - time_sec: 当日 0 点起的秒数(浮点,含百分秒)
        - minute_bar: 0..236 的分钟 bar 索引(集合竞价为 -1)
        - call_auction: bool,是否集合竞价窗口
        - Price: 原始整数价格(已 ×100)
        - Volume: 成交量
        - BSFlag: 买卖标志
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .time_utils import (
    assign_minute_bar,
    is_call_auction,
    parse_time_to_seconds,
)

# 显式指定 dtype,降低内存
_DTYPES = {
    "Time": "int64",
    "Price": "int32",
    "Volume": "int32",
    "BSFlag": "int8",
}


def list_trading_dates() -> list[str]:
    """列出 data/TRADE 下所有交易日(YYYYMMDD 字符串,已排序)。"""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"找不到数据目录: {DATA_DIR}")
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def list_stocks_for_date(date: str) -> list[str]:
    """列出某一天的所有股票代码(去掉 .csv 后缀)。"""
    day_dir = DATA_DIR / date
    return sorted(p.stem for p in day_dir.glob("*.csv"))


def load_one_stock(date: str, stock_code: str) -> pd.DataFrame | None:
    """读取单只股票一天的 tick,加上预解析的时间列。

    返回 None 表示该文件不存在或为空。
    """
    path = DATA_DIR / date / f"{stock_code}.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, dtype=_DTYPES)
    except pd.errors.EmptyDataError:
        return None

    if df.empty:
        return None

    time_sec = parse_time_to_seconds(df["Time"].to_numpy())
    df["time_sec"] = time_sec
    df["minute_bar"] = assign_minute_bar(time_sec)
    df["call_auction"] = is_call_auction(time_sec)
    return df


def load_one_day(date: str) -> dict[str, pd.DataFrame]:
    """读取某一天所有股票的 tick 数据。

    返回: {stock_code: DataFrame}。
    缺失的股票代码不会出现在字典里。
    """
    day_dir = DATA_DIR / date
    if not day_dir.exists():
        return {}

    result: dict[str, pd.DataFrame] = {}
    for csv_path in day_dir.glob("*.csv"):
        code = csv_path.stem
        try:
            df = pd.read_csv(csv_path, dtype=_DTYPES)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        time_sec = parse_time_to_seconds(df["Time"].to_numpy())
        df["time_sec"] = time_sec
        df["minute_bar"] = assign_minute_bar(time_sec)
        df["call_auction"] = is_call_auction(time_sec)
        result[code] = df
    return result
