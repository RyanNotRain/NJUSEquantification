"""Project-wide paths and data conventions."""

from __future__ import annotations

from pathlib import Path

# .../项目/quant_downsampler/qd/config.py
PACKAGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
WORKSPACE_DIR = Path(__file__).resolve().parents[3]

DATA_DIR = WORKSPACE_DIR / "data" / "TRADE"
ADJFACTOR_PATH = WORKSPACE_DIR / "adjfactor.pkl"
OUTPUT_DIR = PROJECT_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "_cache"

METRICS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "amount",
    "buy_volume",
    "sell_volume",
    "buy_amount",
    "sell_amount",
)
PRICE_METRICS = frozenset({"open", "high", "low", "close"})

# The README examples contain 09:30:00 through 11:30:00 and 13:00:00
# through 15:00:00.  The three call-auction waiting rows at 14:57-14:59
# remain present; the matched closing-auction trades are recorded at 15:00.
MORNING_BARS = 121
AFTERNOON_BARS = 121
TOTAL_BARS = MORNING_BARS + AFTERNOON_BARS

MORNING_START_SEC = 9 * 3600 + 30 * 60
MORNING_END_SEC = 11 * 3600 + 30 * 60
AFTERNOON_START_SEC = 13 * 3600
AFTERNOON_END_SEC = 15 * 3600
CONTINUOUS_AFTERNOON_END_SEC = 14 * 3600 + 57 * 60

CALL_AUCTION_WINDOWS = (
    (9 * 3600 + 15 * 60, 9 * 3600 + 25 * 60 + 59.999),
    (14 * 3600 + 57 * 60, 15 * 3600 + 59.999),
)

# Normalize every adjustment-factor series to the first non-null observation
# in the supplied sample.  This preserves first-day price levels and removes
# later corporate-action discontinuities.
APPLY_ADJFACTOR = True
NORMALIZE_ADJFACTOR = True

DAYS_PER_CHUNK = 5
OVERWRITE = False
