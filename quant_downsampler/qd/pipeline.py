"""主流程:把逐笔成交降采样为日频和分钟频。

输出布局:
    {OUTPUT_DIR}/
        daily/
            open.csv, high.csv, low.csv, close.csv, volume.csv,
            trade_count.csv, amount.csv,
            buy_volume.csv, sell_volume.csv, buy_amount.csv, sell_amount.csv
        minute/
            open/20250401.csv, open/20250402.csv, ...
            ...
            sell_amount/20260630.csv

每张表的格式:
    日频:行=YYYY-MM-DD,列=股票代码(6 位数字)
    分钟频:行=HH:MM(237 个 bar),列=股票代码
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import (
    CACHE_DIR,
    DAYS_PER_CHUNK,
    METRICS,
    OUTPUT_DIR,
    OVERWRITE,
)
from .data_loader import list_trading_dates
from .daily import build_daily_tables
from .minute import build_minute_table_for_date

logger = logging.getLogger("qd")


def _save_daily_tables(
    tables: dict[str, pd.DataFrame], out_dir: Path, overwrite: bool = OVERWRITE
) -> None:
    daily_dir = out_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for metric, df in tables.items():
        path = daily_dir / f"{metric}.csv"
        if path.exists() and not overwrite:
            logger.info("跳过(已存在):%s", path)
            continue
        df.to_csv(path, float_format="%.4f")
        logger.info("写入:%s  shape=%s", path, df.shape)


def _save_minute_tables_for_day(
    tables: dict[str, pd.DataFrame], date: str, out_dir: Path, overwrite: bool = OVERWRITE
) -> None:
    for metric, df in tables.items():
        metric_dir = out_dir / "minute" / metric
        metric_dir.mkdir(parents=True, exist_ok=True)
        path = metric_dir / f"{date}.csv"
        if path.exists() and not overwrite:
            continue
        df.to_csv(path, float_format="%.4f")


def _list_missing_minute_dates(out_dir: Path, dates: Iterable[str]) -> list[str]:
    """列出分钟频某 metric 下尚未产出的日期(用于断点续跑)。"""
    # 简单做法:任意一个 metric 缺失都算"未完成"
    dates = list(dates)
    sample_metric_dir = out_dir / "minute" / METRICS[0]
    if not sample_metric_dir.exists():
        return dates
    existing = {p.stem for p in sample_metric_dir.glob("*.csv")}
    return [d for d in dates if d not in existing]


def run(
    dates: list[str] | None = None,
    out_dir: Path = OUTPUT_DIR,
    days_per_chunk: int = DAYS_PER_CHUNK,
) -> None:
    """执行完整降采样流程。

    Args:
        dates: 要处理的日期列表。None 表示处理 data/TRADE 下所有日期。
        out_dir: 输出根目录。
        days_per_chunk: 分块大小(同时影响日频和分钟频的内存占用)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if dates is None:
        dates = list_trading_dates()
    if not dates:
        raise RuntimeError("没有可处理的日期,请检查 data/TRADE 目录。")

    logger.info("共 %d 个交易日,输出目录:%s", len(dates), out_dir)

    # ========== 1) 日频:一次性构建(内存允许)============
    daily_path = out_dir / "daily" / "open.csv"
    if daily_path.exists() and not OVERWRITE:
        logger.info("日频表已存在,跳过(可手动删除 %s 重跑)", out_dir / "daily")
    else:
        t0 = time.time()
        logger.info("[daily] 开始构建日频 11 张表...")
        tables = build_daily_tables(dates)
        _save_daily_tables(tables, out_dir, overwrite=True)
        logger.info("[daily] 完成,耗时 %.1fs", time.time() - t0)

    # ========== 2) 分钟频:按日处理,每只股票前一天 close 用于 9:30 bar 填充 ==========
    missing = _list_missing_minute_dates(out_dir, dates)
    if not missing:
        logger.info("[minute] 所有日期已存在,跳过")
        return

    logger.info("[minute] 待处理 %d 个日期,分块大小 %d", len(missing), days_per_chunk)

    # 加载日频 close 表(用于跨日填充)
    daily_close = pd.read_csv(
        out_dir / "daily" / "close.csv", index_col=0
    ) if (out_dir / "daily" / "close.csv").exists() else None

    # 全局股票列表(以日频表的列为准,保证 daily 和 minute 列数一致)
    all_stocks_global: list[str] = (
        list(daily_close.columns) if daily_close is not None else []
    )

    for chunk_start in range(0, len(missing), days_per_chunk):
        chunk = missing[chunk_start: chunk_start + days_per_chunk]
        t0 = time.time()
        logger.info(
            "[minute] 处理分块 %d/%d  日期 %s - %s",
            chunk_start // days_per_chunk + 1,
            (len(missing) + days_per_chunk - 1) // days_per_chunk,
            chunk[0],
            chunk[-1],
        )

        # 收集本分块所有股票并集
        all_stocks: set[str] = set()
        for date in chunk:
            all_stocks.update(_list_stocks_for_date_safe(date))
        all_stocks_sorted = sorted(all_stocks)

        # 准备 prev_close
        prev_close_dict: dict[str, float] | None = None
        if daily_close is not None:
            prev_date = _previous_trading_date(chunk[0], dates)
            if prev_date is not None and prev_date in daily_close.index:
                prev_close_dict = daily_close.loc[prev_date].to_dict()

        for date in chunk:
            t1 = time.time()
            tables = build_minute_table_for_date(
                date, prev_close=prev_close_dict, all_stocks=all_stocks_global,
            )
            _save_minute_tables_for_day(tables, date, out_dir, overwrite=False)
            logger.info("  - %s 耗时 %.1fs", date, time.time() - t1)

            # 更新 prev_close_dict 为本日的 close(下一日用)
            if daily_close is not None and date in daily_close.index:
                prev_close_dict = daily_close.loc[date].to_dict()

        logger.info("[minute] 分块完成,耗时 %.1fs", time.time() - t0)

    logger.info("全部完成。")


def _list_stocks_for_date_safe(date: str) -> list[str]:
    from .data_loader import list_stocks_for_date
    try:
        return list_stocks_for_date(date)
    except FileNotFoundError:
        return []


def _previous_trading_date(date: str, all_dates: list[str]) -> str | None:
    """返回 all_dates 中严格早于 date 的最后一个交易日。"""
    for d in reversed(all_dates):
        if d < date:
            return d
    return None
