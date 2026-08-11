"""Generate benchmark-relative Task 4 and execution-aware Task 5 results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qd.config import OUTPUT_DIR
from qd.strategy_analysis import (
    run_lstm_model_strategy_comparison,
    run_lstm_strategy_analysis,
    run_task4_benchmark_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recent-days", type=int, default=45)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument(
        "--task4-dir", default="backtest_strict",
        help="Task 4 output directory under project/output",
    )
    parser.add_argument(
        "--task4-strategy", choices=("adaptive_close", "adaptive_open", "both"),
        default="both", help="Task 4 strategy to compare; default regenerates both",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--task4-only", action="store_true")
    scope.add_argument("--task5-only", action="store_true")
    args = parser.parse_args()
    if args.recent_days <= 0:
        parser.error("--recent-days must be positive")
    if args.sell_fee_bps < 0:
        parser.error("--sell-fee-bps cannot be negative")

    result = {}
    if not args.task5_only:
        strategy_names = (
            ("adaptive_close", "adaptive_open")
            if args.task4_strategy == "both" else (args.task4_strategy,)
        )
        result["task4_benchmarks"] = {
            strategy_name: run_task4_benchmark_analysis(
                strategy_name=strategy_name,
                recent_days=args.recent_days,
                backtest_dir=args.task4_dir,
            )
            for strategy_name in strategy_names
        }
    if not args.task4_only:
        result["task5_strategy"] = run_lstm_strategy_analysis(
            sell_fee_bps=args.sell_fee_bps
        )
        baseline_metrics = OUTPUT_DIR / "lstm_baselines" / "test_metrics.csv"
        if baseline_metrics.exists():
            result["task5_model_comparison"] = run_lstm_model_strategy_comparison(
                sell_fee_bps=args.sell_fee_bps
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
