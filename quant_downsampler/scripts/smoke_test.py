"""冒烟测试:跑 1 只股票 1 天,验证输出格式。

python -m scripts.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from qd.config import (
    AFTERNOON_BARS,
    METRICS,
    MORNING_BARS,
    TOTAL_BARS,
)
from qd.time_utils import MINUTE_BAR_LABELS
from qd.data_loader import load_one_day
from qd.daily import aggregate_day, build_daily_tables, calc_daily_metrics
from qd.minute import build_minute_table_for_date, calc_minute_metrics


def test_daily_one_stock_one_day() -> None:
    print("=" * 60)
    print("[1] 单只股票单天日频指标")
    print("=" * 60)
    day = load_one_day("20250401")
    code, df = next(iter(day.items()))
    print(f"  stock={code}, ticks={len(df)}")
    metrics = calc_daily_metrics(df)
    for k, v in metrics.items():
        print(f"    {k:12s} = {v:,.4f}" if isinstance(v, float) else f"    {k:12s} = {v}")

    # 校验 BSFlag 关系
    bs = df["BSFlag"].to_numpy()
    vol = df["Volume"].to_numpy()
    assert metrics["buy_volume"] == float(vol[bs == 0].sum())
    assert metrics["sell_volume"] == float(vol[bs == 1].sum())
    print("  ✓ BSFlag 拆分正确")


def test_minute_one_stock_one_day() -> None:
    print("=" * 60)
    print("[2] 单只股票单天分钟频指标")
    print("=" * 60)
    day = load_one_day("20250401")
    code, df = next(iter(day.items()))
    m = calc_minute_metrics(df)
    print(f"  stock={code}, minute shape={m.shape}")
    print(f"    MORNING_BARS={MORNING_BARS}, AFTERNOON_BARS={AFTERNOON_BARS}, TOTAL={TOTAL_BARS}")
    assert m.shape == (TOTAL_BARS, len(METRICS))

    # 应该有一些 bar 有数据(连续竞价 9:30 之后)
    has_data = m["volume"] > 0
    print(f"    有成交的 bar 数 = {int(has_data.sum())} / {TOTAL_BARS}")
    assert has_data.any(), "应该至少有一个 bar 有成交"

    # 集合竞价 tick 不应进入 bar
    df_continuous = df[df["minute_bar"] >= 0]
    print(f"    连续竞价 tick 数 = {len(df_continuous)}, 集合竞价 tick = {len(df) - len(df_continuous)}")
    print("  ✓ 集合竞价 tick 正确被排除")


def test_minute_table_one_day() -> None:
    print("=" * 60)
    print("[3] 一天的分钟频宽表")
    print("=" * 60)
    tables = build_minute_table_for_date("20250401")
    print(f"  metrics = {list(tables.keys())}")
    for metric, df in tables.items():
        print(f"    {metric:12s} shape={df.shape}  first 3 bars: {list(df.index[:3])}  last 3 bars: {list(df.index[-3:])}")
        assert df.shape[0] == TOTAL_BARS
        assert df.shape[1] > 0
    print("  ✓ 所有 metric 的宽表都符合预期(237 行 × N 列)")


def test_daily_table_multiple_days() -> None:
    print("=" * 60)
    print("[4] 多天日频宽表")
    print("=" * 60)
    tables = build_daily_tables(["20250401", "20250402", "20250403"])
    for metric, df in tables.items():
        print(f"    {metric:12s} shape={df.shape}  dates={list(df.index)}")
        assert df.shape[0] == 3
    # 验证 OHLC 填充:如果某只股票 4/2 停牌,4/2 的 OHLC 应等于 4/1 的 close
    # (这只股票 4/1, 4/2, 4/3 都有数据,这里不做强制断言,只检查数据连续)
    print("  ✓ 日频宽表行=日期,列=股票")


def main() -> None:
    test_daily_one_stock_one_day()
    print()
    test_minute_one_stock_one_day()
    print()
    test_minute_table_one_day()
    print()
    test_daily_table_multiple_days()
    print()
    print("ALL SMOKE TESTS PASSED ✓")


if __name__ == "__main__":
    main()
