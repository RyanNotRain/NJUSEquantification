"""时间字段解析工具。

readme 数据规则:
- Time 字段是 9 位整数,格式为 HHMMSScc(centiseconds,百分秒)。
  例如 92500740 = 9:25:00.740。
- 集合竞价窗口: 9:15-9:25 和 14:57-15:00。
- 连续竞价窗口: 9:30-11:30 和 13:00-14:57。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CALL_AUCTION_WINDOWS, MORNING_BARS, AFTERNOON_BARS, TOTAL_BARS


def parse_time_to_seconds(time_int: np.ndarray | pd.Series) -> np.ndarray:
    """把 9 位 HHMMSScc 整数解析为当日 0 点起的秒数(含小数)。

    9:25:00.740 -> 9*3600 + 25*60 + 0.740 = 33900.74
    """
    t = time_int.astype(np.int64)
    h = t // 10_000_000
    m = (t // 100_000) % 100
    s = (t // 1_000) % 100
    cs = t % 1_000
    return h * 3600 + m * 60 + s + cs / 1000.0


def seconds_to_hms(seconds: float) -> str:
    """把秒数(浮点)转成 HH:MM 字符串,用于分钟 K 线标签。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"


# 预生成 237 个分钟 bar 的标签(9:30 ... 11:30, 13:00 ... 14:56)
MORNING_START_SEC = 9 * 3600 + 30 * 60       # 34200
AFTERNOON_START_SEC = 13 * 3600              # 46800


def minute_bar_labels() -> list[str]:
    """返回 237 个分钟 bar 的标签,顺序与后面 resample 一致。"""
    labels: list[str] = []
    for i in range(MORNING_BARS):           # 9:30 - 11:30
        labels.append(seconds_to_hms(MORNING_START_SEC + i * 60))
    for i in range(AFTERNOON_BARS):         # 13:00 - 14:56
        labels.append(seconds_to_hms(AFTERNOON_START_SEC + i * 60))
    return labels


MINUTE_BAR_LABELS: list[str] = minute_bar_labels()
assert len(MINUTE_BAR_LABELS) == TOTAL_BARS, (
    f"分钟 bar 数量不对: {len(MINUTE_BAR_LABELS)} vs {TOTAL_BARS}"
)


def assign_minute_bar(seconds: np.ndarray) -> np.ndarray:
    """把秒数数组映射到 0..236 的分钟 bar 索引。

    集合竞价窗口内的 tick 返回 -1,表示不属于连续竞价 bar。
    11:30 整点(34200+120*60=41400)归到最后一个 morning bar(index 119)。
    14:56 整点(54000-240=53760,实际 13:00+117*60=53820...让我重算)
    14:56 = 14*3600+56*60 = 53760
    14:57 = 14*3600+57*60 = 53820(集合竞价开始)
    """
    bar = np.full(seconds.shape, -1, dtype=np.int32)

    # 上午 9:30-11:30 -> 0..119
    morning_mask = (seconds >= MORNING_START_SEC) & (
        seconds < MORNING_START_SEC + MORNING_BARS * 60
    )
    bar[morning_mask] = ((seconds[morning_mask] - MORNING_START_SEC) // 60).astype(np.int32)

    # 下午 13:00-14:56 -> 120..236
    afternoon_mask = (seconds >= AFTERNOON_START_SEC) & (
        seconds < AFTERNOON_START_SEC + AFTERNOON_BARS * 60
    )
    bar[afternoon_mask] = (
        MORNING_BARS + (seconds[afternoon_mask] - AFTERNOON_START_SEC) // 60
    ).astype(np.int32)

    return bar


def is_call_auction(seconds: np.ndarray) -> np.ndarray:
    """判断秒数是否落在两个集合竞价窗口内。"""
    out = np.zeros(seconds.shape, dtype=bool)
    for start, end in CALL_AUCTION_WINDOWS:
        out |= (seconds >= start) & (seconds <= end)
    return out


def seconds_to_time_str(seconds: float) -> str:
    """把秒数(浮点)转成 HH:MM:SS.sss 字符串,用于日志。"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
