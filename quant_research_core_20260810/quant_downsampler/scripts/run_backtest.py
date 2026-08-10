"""Compatibility entry point for Task 4.

The old implementation used an inconsistent signal/return timeline.  This
command now delegates to the strict backtest so an old command cannot silently
regenerate obsolete results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qd.task4_strict import run_all_strict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict Task 4 factor backtest")
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    print("run_backtest.py now uses the strict, leakage-safe implementation.")
    run_all_strict(lookback=args.lookback, top_n=args.top_n)


if __name__ == "__main__":
    main()
