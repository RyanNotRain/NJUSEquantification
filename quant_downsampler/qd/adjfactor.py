"""复权因子加载与应用。

readme 补丁说明:
    股票会有分红配股等事件对股价产生较大影响,故为一定程度上保持
    股价的连续性,须使用复权因子对价格进行修正,复权因子见
    adjfactor.pkl。

本项目:
    复权价格 = 原始价格 × (当日 adjfactor / 上市首日 adjfactor)

    adjfactor.pkl 是一个 pandas DataFrame:
        - index: DatetimeIndex(交易日)
        - columns: 股票代码(形如 600705.SH,302 个)
        - values: 复权因子(float,>0)

注意(数据混淆):
    data/TRADE/ 中的股票代码是 6 位数字(如 000012),
    adjfactor.pkl 中的代码是真实代码(如 600705.SH),
    二者无法直接对应。需要用户在使用前提供映射。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from .config import (
    APPLY_ADJFACTOR,
    ADJFACTOR_PATH,
    _STOCK_CODE_MAPPING,
)


def load_adjfactor(path: Path | None = None) -> pd.DataFrame:
    """读取 adjfactor.pkl,返回 DataFrame。"""
    p = path or ADJFACTOR_PATH
    with open(p, "rb") as f:
        adj = pickle.load(f)  # type: ignore[assignment]
    if not isinstance(adj, pd.DataFrame):
        raise TypeError(f"adjfactor.pkl 应为 DataFrame,实际是 {type(adj)}")
    return adj


def get_adjust_ratio(
    adj: pd.DataFrame,
    stock_code: str,
    date: pd.Timestamp,
) -> float | None:
    """返回某只股票某一天的复权调整系数(adj_today / adj_first)。

    - stock_code 应是 adjfactor 里的代码(如 600705.SH)
    - date 应是 adjfactor.index 里的日期
    - 返回 None 表示该股票/日期没有复权因子

    注:实际计算时,基准 adj_first 是该股票在数据范围第一天的 adjfactor。
    """
    if stock_code not in adj.columns:
        return None
    series = adj[stock_code].dropna()
    if series.empty or date not in series.index:
        return None
    base = series.iloc[0]
    cur = series.loc[date]
    if base == 0:
        return None
    return float(cur / base)


def adjust_price(
    price_raw: float,
    stock_code: str,
    date: pd.Timestamp,
    adj: pd.DataFrame,
) -> float:
    """对单只股票某天的价格做复权处理。

    如果 APPLY_ADJFACTOR=False 或没有映射/因子,直接返回原价。
    """
    if not APPLY_ADJFACTOR:
        return price_raw
    real_code = _STOCK_CODE_MAPPING.get(stock_code, stock_code)
    ratio = get_adjust_ratio(adj, real_code, date)
    if ratio is None:
        return price_raw
    return price_raw * ratio
