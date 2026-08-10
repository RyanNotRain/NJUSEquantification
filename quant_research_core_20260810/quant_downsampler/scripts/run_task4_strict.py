"""Run leakage-safe Task 4 strategies and the deliberately biased controls."""

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
    parser.add_argument("--lookback", type=int, default=60, help="historical IC lookback")
    parser.add_argument("--top-n", type=int, default=10, help="portfolio size")
    parser.add_argument(
        "--factor-set", choices=("required", "extended"), default="required",
        help="required = example plus three original factors; extended = all ten",
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()
    run_all_strict(
        lookback=args.lookback, top_n=args.top_n,
        factor_set=args.factor_set, out_dir=args.out,
    )


if __name__ == "__main__":
    main()
