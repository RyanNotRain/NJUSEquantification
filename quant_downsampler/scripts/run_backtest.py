"""Run corrected and deliberately biased comparison backtests."""

from __future__ import annotations

import argparse
from pathlib import Path

from qd.backtest import run_full_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="IC-weighted top-N backtest")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    results = run_full_backtest(
        args.data_dir, args.out_dir, args.lookback, args.top_n
    )
    for name, result in results.items():
        print(name, result["metrics"])


if __name__ == "__main__":
    main()
