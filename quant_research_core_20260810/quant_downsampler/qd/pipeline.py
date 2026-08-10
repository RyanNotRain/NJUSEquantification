"""End-to-end adjusted daily and minute downsampling pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DAYS_PER_CHUNK, METRICS, OUTPUT_DIR, OVERWRITE, TOTAL_BARS
from .daily import build_daily_tables
from .data_loader import list_trading_dates
from .minute import build_minute_table_for_date

logger = logging.getLogger("qd")


def _save_daily_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    target = out_dir / "daily"
    target.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        path = target / f"{metric}.csv"
        tables[metric].to_csv(path, float_format="%.6f")
        logger.info("wrote %s shape=%s", path, tables[metric].shape)


def _save_minute_tables_for_day(tables: dict[str, pd.DataFrame], date: str,
                                out_dir: Path) -> None:
    for metric in METRICS:
        target = out_dir / "minute" / metric
        target.mkdir(parents=True, exist_ok=True)
        tables[metric].to_csv(target / f"{date}.csv", float_format="%.6f")


def _minute_day_complete(out_dir: Path, date: str) -> bool:
    for metric in METRICS:
        path = out_dir / "minute" / metric / f"{date}.csv"
        if not path.exists():
            return False
        try:
            if len(pd.read_csv(path, usecols=[0])) != TOTAL_BARS:
                return False
        except Exception:
            return False
    return True


def _list_missing_minute_dates(out_dir: Path, dates: Iterable[str]) -> list[str]:
    return [date for date in dates if not _minute_day_complete(out_dir, date)]


def _daily_complete(out_dir: Path, expected_dates: int) -> bool:
    for metric in METRICS:
        path = out_dir / "daily" / f"{metric}.csv"
        if not path.exists():
            return False
        try:
            if len(pd.read_csv(path, usecols=[0])) != expected_dates:
                return False
        except Exception:
            return False
    return True


def run(dates: list[str] | None = None, out_dir: Path = OUTPUT_DIR,
        days_per_chunk: int = DAYS_PER_CHUNK, overwrite: bool = OVERWRITE) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = sorted(dates or list_trading_dates())
    if not dates:
        raise RuntimeError("no trading dates found")
    if days_per_chunk < 1:
        raise ValueError("days_per_chunk must be positive")
    if overwrite or not _daily_complete(out_dir, len(dates)):
        started = time.time()
        logger.info("building adjusted daily tables for %d dates", len(dates))
        _save_daily_tables(build_daily_tables(dates), out_dir)
        logger.info("daily complete in %.1fs", time.time() - started)
    else:
        logger.info("daily outputs are complete; skipping")

    close = pd.read_csv(out_dir / "daily" / "close.csv", index_col=0)
    close.index = pd.to_datetime(close.index)
    stocks = close.columns.tolist()
    target_dates = dates if overwrite else _list_missing_minute_dates(out_dir, dates)
    logger.info("building minute tables for %d dates", len(target_dates))
    for chunk_start in range(0, len(target_dates), days_per_chunk):
        for trading_date in target_dates[chunk_start:chunk_start + days_per_chunk]:
            timestamp = pd.Timestamp(trading_date)
            prior = close.index[close.index < timestamp]
            prev_close = close.loc[prior[-1]].to_dict() if len(prior) else None
            _save_minute_tables_for_day(
                build_minute_table_for_date(trading_date, prev_close, stocks),
                trading_date, out_dir,
            )
            logger.info("minute %s complete", trading_date)
    logger.info("downsampling complete")
