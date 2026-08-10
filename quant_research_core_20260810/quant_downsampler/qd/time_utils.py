"""Parsing and displayed-minute assignment for the integer trade-time field."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AFTERNOON_BARS, AFTERNOON_END_SEC, AFTERNOON_START_SEC,
    CALL_AUCTION_WINDOWS, MORNING_BARS, MORNING_END_SEC,
    MORNING_START_SEC, TOTAL_BARS,
)


def parse_time_to_seconds(time_int: np.ndarray | pd.Series) -> np.ndarray:
    """Convert HMMSSmmm/HHMMSSmmm integers to seconds after midnight."""
    values = np.asarray(time_int, dtype=np.int64)
    hour = values // 10_000_000
    minute = (values // 100_000) % 100
    second = (values // 1_000) % 100
    millis = values % 1_000
    return hour * 3600 + minute * 60 + second + millis / 1000.0


def seconds_to_hms(seconds: float) -> str:
    hour = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    second = int(seconds % 60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def minute_bar_labels() -> list[str]:
    labels = [seconds_to_hms(MORNING_START_SEC + i * 60) for i in range(MORNING_BARS)]
    labels.extend(seconds_to_hms(AFTERNOON_START_SEC + i * 60) for i in range(AFTERNOON_BARS))
    return labels


MINUTE_BAR_LABELS = minute_bar_labels()
assert len(MINUTE_BAR_LABELS) == TOTAL_BARS
assert MINUTE_BAR_LABELS[0] == "09:30:00"
assert MINUTE_BAR_LABELS[120] == "11:30:00"
assert MINUTE_BAR_LABELS[121] == "13:00:00"
assert MINUTE_BAR_LABELS[-1] == "15:00:00"


def assign_minute_bar(seconds: np.ndarray) -> np.ndarray:
    """Map trades to the 242 displayed rows; opening auction remains daily-only."""
    seconds = np.asarray(seconds, dtype=np.float64)
    bar = np.full(seconds.shape, -1, dtype=np.int32)
    morning = (seconds >= MORNING_START_SEC) & (seconds < MORNING_END_SEC + 60)
    bar[morning] = ((seconds[morning] - MORNING_START_SEC) // 60).astype(np.int32)
    afternoon = (seconds >= AFTERNOON_START_SEC) & (seconds < AFTERNOON_END_SEC + 60)
    bar[afternoon] = (
        MORNING_BARS + (seconds[afternoon] - AFTERNOON_START_SEC) // 60
    ).astype(np.int32)
    return bar


def is_call_auction(seconds: np.ndarray) -> np.ndarray:
    seconds = np.asarray(seconds, dtype=np.float64)
    result = np.zeros(seconds.shape, dtype=bool)
    for start, end in CALL_AUCTION_WINDOWS:
        result |= (seconds >= start) & (seconds <= end)
    return result


def seconds_to_time_str(seconds: float) -> str:
    hour = int(seconds // 3600)
    minute = int((seconds % 3600) // 60)
    second = seconds - hour * 3600 - minute * 60
    return f"{hour:02d}:{minute:02d}:{second:06.3f}"
