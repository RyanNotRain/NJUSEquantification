"""Parsing and bar assignment for the integer trade-time field."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    AFTERNOON_BARS,
    AFTERNOON_END_SEC,
    AFTERNOON_START_SEC,
    CALL_AUCTION_WINDOWS,
    MORNING_BARS,
    MORNING_END_SEC,
    MORNING_START_SEC,
    TOTAL_BARS,
)


def parse_time_to_seconds(time_int: np.ndarray | pd.Series) -> np.ndarray:
    """Convert HMMSSmmm/HHMMSSmmm integers to seconds after midnight."""
    t = np.asarray(time_int, dtype=np.int64)
    h = t // 10_000_000
    m = (t // 100_000) % 100
    s = (t // 1_000) % 100
    millis = t % 1_000
    return h * 3600 + m * 60 + s + millis / 1000.0


def seconds_to_hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def minute_bar_labels() -> list[str]:
    labels = [
        seconds_to_hms(MORNING_START_SEC + i * 60)
        for i in range(MORNING_BARS)
    ]
    labels.extend(
        seconds_to_hms(AFTERNOON_START_SEC + i * 60)
        for i in range(AFTERNOON_BARS)
    )
    return labels


MINUTE_BAR_LABELS = minute_bar_labels()
assert len(MINUTE_BAR_LABELS) == TOTAL_BARS
assert MINUTE_BAR_LABELS[0] == "09:30:00"
assert MINUTE_BAR_LABELS[120] == "11:30:00"
assert MINUTE_BAR_LABELS[121] == "13:00:00"
assert MINUTE_BAR_LABELS[-1] == "15:00:00"


def assign_minute_bar(seconds: np.ndarray) -> np.ndarray:
    """Map trades to the README's 242 displayed minute rows.

    Opening-auction trades (09:25) are daily-only.  Trades stamped in the
    morning/afternoon displayed ranges are assigned by their wall-clock minute,
    including the matched closing-auction trades stamped at 15:00.
    """
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
    out = np.zeros(seconds.shape, dtype=bool)
    for start, end in CALL_AUCTION_WINDOWS:
        out |= (seconds >= start) & (seconds <= end)
    return out


def seconds_to_time_str(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
