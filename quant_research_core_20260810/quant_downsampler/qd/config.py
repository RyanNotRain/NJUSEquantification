"""Project-wide paths and market-data conventions."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PACKAGE_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent

DATA_DIR = Path(os.environ.get("QD_DATA_DIR", WORKSPACE_DIR / "data" / "TRADE"))
ADJFACTOR_PATH = Path(os.environ.get("QD_ADJFACTOR_PATH", WORKSPACE_DIR / "adjfactor.pkl"))
OUTPUT_DIR = Path(os.environ.get("QD_OUTPUT_DIR", PROJECT_DIR / "output"))
CACHE_DIR = OUTPUT_DIR / "_cache"

METRICS: tuple[str, ...] = (
    "open", "high", "low", "close", "volume", "trade_count", "amount",
    "buy_volume", "sell_volume", "buy_amount", "sell_amount",
)
PRICE_METRICS = frozenset({"open", "high", "low", "close"})

# Displayed minute rows follow the supplied example: 09:30–11:30 and
# 13:00–15:00 inclusive.  Rows 14:57–14:59 are closing-auction waiting rows;
# the matched closing-auction trade is represented at 15:00.
MORNING_BARS = 121
AFTERNOON_BARS = 121
TOTAL_BARS = MORNING_BARS + AFTERNOON_BARS  # 242
MORNING_START_SEC = 9 * 3600 + 30 * 60
MORNING_END_SEC = 11 * 3600 + 30 * 60
AFTERNOON_START_SEC = 13 * 3600
AFTERNOON_END_SEC = 15 * 3600
CONTINUOUS_AFTERNOON_END_SEC = 14 * 3600 + 57 * 60

CALL_AUCTION_WINDOWS = (
    (9 * 3600 + 15 * 60, 9 * 3600 + 25 * 60 + 59.999),
    (14 * 3600 + 57 * 60, 15 * 3600 + 59.999),
)

# The 300 raw six-digit codes match the 300 adjustment-factor base codes
# exactly; adjfactor merely adds the .SH/.SZ suffix.  Normalize each series to
# its first non-null sample value so first-day price levels are preserved.
APPLY_ADJFACTOR = True
NORMALIZE_ADJFACTOR = True

DAYS_PER_CHUNK = 5
OVERWRITE = False
