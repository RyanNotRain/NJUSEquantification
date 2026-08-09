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

# Experimental factors are deliberately kept outside ``FACTOR_NAMES``.  The
# latter is the assignment's official four-factor universe and is consumed by
# the published evaluation tables, whose four-row definition must not drift as
# exploratory ideas are added.
EXPERIMENTAL_FACTOR_NAMES = ("illiquidity_20d",)


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


def compute_illiquidity_20d(
    close: pd.DataFrame,
    amount: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Strictly historical 20-day Amihud-style illiquidity.

    For each stock and table date ``s`` the factor is the trailing mean of
    ``abs(daily return) / amount`` using observations through the close of
    ``s``.  The backtest, like every official factor path, makes its date-``t``
    decision from the factor row dated ``t-1``; consequently no date-``t``
    amount or return enters the signal.  It is an experimental research factor
    and is not one of the assignment's three required additional factors.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if not close.index.equals(amount.index) or not close.columns.equals(amount.columns):
        raise ValueError("close and amount axes must match")
    absolute_return = close.pct_change(fill_method=None).abs()
    positive_amount = amount.where(amount > 0.0)
    daily_illiquidity = absolute_return.divide(positive_amount)
    factor = daily_illiquidity.rolling(
        window=window, min_periods=window
    ).mean()
    return factor.replace([np.inf, -np.inf], np.nan)


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
        "illiquidity_20d": compute_illiquidity_20d(
            data["close"], data["amount"]
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
