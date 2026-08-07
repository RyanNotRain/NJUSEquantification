"""Factor construction from the adjusted daily tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR

ADDITIONAL_FACTORS = (
    "momentum_5d",
    "buy_sell_imbalance",
    "intraday_range",
)
FACTOR_NAMES = ("example_factor",) + ADDITIONAL_FACTORS


def load_daily_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir or (OUTPUT_DIR / "daily"))
    metrics = (
        "open", "high", "low", "close", "volume", "amount",
        "buy_volume", "sell_volume", "trade_count",
    )
    result: dict[str, pd.DataFrame] = {}
    for metric in metrics:
        path = data_dir / f"{metric}.csv"
        if not path.exists():
            raise FileNotFoundError(f"missing daily table: {path}")
        table = pd.read_csv(path, index_col=0)
        table.index = pd.to_datetime(table.index)
        table.columns = table.columns.astype(str)
        result[metric] = table.astype(np.float64).sort_index()
    reference = result["close"]
    for metric, table in result.items():
        if not table.index.equals(reference.index) or not table.columns.equals(reference.columns):
            raise ValueError(f"daily table axes do not match close.csv: {metric}")
    return result


def compute_example_factor(amount: pd.DataFrame) -> pd.DataFrame:
    """log(std([MA1, MA5, MA10, MA20] of daily turnover))."""
    moving = [amount.rolling(w, min_periods=w).mean() for w in (1, 5, 10, 20)]
    values = np.stack([x.to_numpy() for x in moving], axis=0)
    factor = np.full(values.shape[1:], np.nan, dtype=np.float64)
    valid = np.isfinite(values).all(axis=0)
    factor[valid] = np.std(values[:, valid], axis=0, ddof=1)
    factor[~np.isfinite(factor) | (factor <= 0)] = np.nan
    return pd.DataFrame(np.log(factor), index=amount.index, columns=amount.columns)


def compute_momentum_5d(close: pd.DataFrame) -> pd.DataFrame:
    out = close.pct_change(5, fill_method=None)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_buy_sell_imbalance(
    buy_volume: pd.DataFrame,
    sell_volume: pd.DataFrame,
) -> pd.DataFrame:
    denominator = (buy_volume + sell_volume).replace(0, np.nan)
    return (buy_volume - sell_volume) / denominator


def compute_intraday_range(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    return (high - low) / close.replace(0, np.nan)


def compute_forward_returns(close: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    return close.shift(-period) / close - 1.0


def compute_all_factors(
    daily_data: dict[str, pd.DataFrame] | None = None,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    data = daily_data or load_daily_data(data_dir)
    factors = {
        "example_factor": compute_example_factor(data["amount"]),
        "momentum_5d": compute_momentum_5d(data["close"]),
        "buy_sell_imbalance": compute_buy_sell_imbalance(
            data["buy_volume"], data["sell_volume"]
        ),
        "intraday_range": compute_intraday_range(
            data["high"], data["low"], data["close"]
        ),
        "forward_return_1d": compute_forward_returns(data["close"]),
    }
    return factors


def save_factors(
    factors: dict[str, pd.DataFrame],
    out_dir: Path | None = None,
) -> None:
    target = Path(out_dir or (OUTPUT_DIR / "factors"))
    target.mkdir(parents=True, exist_ok=True)
    for name, table in factors.items():
        table.to_csv(target / f"{name}.csv", float_format="%.8f")
