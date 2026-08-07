"""Adjustment-factor loading, code mapping, and normalized ratios."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ADJFACTOR_PATH, APPLY_ADJFACTOR, NORMALIZE_ADJFACTOR


@lru_cache(maxsize=2)
def load_adjfactor(path: str | Path = ADJFACTOR_PATH) -> pd.DataFrame:
    adj = pd.read_pickle(Path(path))
    if not isinstance(adj, pd.DataFrame):
        raise TypeError(f"adjfactor.pkl must contain a DataFrame, got {type(adj)}")
    adj = adj.copy()
    adj.index = pd.to_datetime(adj.index).normalize()
    adj.columns = adj.columns.astype(str)
    return adj.sort_index()


@lru_cache(maxsize=2)
def code_mapping(path: str | Path = ADJFACTOR_PATH) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for full_code in load_adjfactor(path).columns:
        base = full_code.split(".", 1)[0].zfill(6)
        if base in mapping:
            raise ValueError(f"duplicate adjustment-factor base code: {base}")
        mapping[base] = full_code
    return mapping


def full_stock_code(data_code: str) -> str:
    base = str(data_code).split(".", 1)[0].zfill(6)
    try:
        return code_mapping()[base]
    except KeyError as exc:
        raise KeyError(f"no adjfactor mapping for data code {data_code}") from exc


@lru_cache(maxsize=100_000)
def get_adjust_ratio(data_code: str, date: str) -> float:
    if not APPLY_ADJFACTOR:
        return 1.0
    adj = load_adjfactor()
    full_code = full_stock_code(data_code)
    series = adj[full_code].dropna()
    ts = pd.Timestamp(date).normalize()
    if ts not in series.index:
        raise KeyError(f"missing adjfactor for {full_code} on {ts.date()}")
    current = float(series.loc[ts])
    if not np.isfinite(current) or current <= 0:
        raise ValueError(f"invalid adjfactor for {full_code} on {ts.date()}: {current}")
    if not NORMALIZE_ADJFACTOR:
        return current
    base = float(series.iloc[0])
    if not np.isfinite(base) or base <= 0:
        raise ValueError(f"invalid base adjfactor for {full_code}: {base}")
    return current / base


def adjust_prices(prices: np.ndarray, data_code: str, date: str) -> np.ndarray:
    return np.asarray(prices, dtype=np.float64) * get_adjust_ratio(data_code, date)
