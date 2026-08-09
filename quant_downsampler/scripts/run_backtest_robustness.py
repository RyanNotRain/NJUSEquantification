"""Run turnover controls and symmetric transaction-cost stress tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.backtest_robustness import run_backtest_robustness


def _cost_grid(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("cost grid must contain non-negative bps values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="corrected factor backtest with turnover controls and cost stress"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--buffer-n", type=int, default=20)
    parser.add_argument("--max-replacements", type=int, default=3)
    parser.add_argument(
        "--costs-bps", type=_cost_grid, default=(0.0, 5.0, 10.0, 20.0, 30.0, 50.0)
    )
    args = parser.parse_args()
    result = run_backtest_robustness(
        args.data_dir,
        args.out_dir,
        lookback=args.lookback,
        top_n=args.top_n,
        buffer_n=args.buffer_n,
        max_replacements=args.max_replacements,
        costs_bps=args.costs_bps,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(result["stress"][["total_return", "sharpe_ratio", "max_drawdown"]])


if __name__ == "__main__":
    main()

