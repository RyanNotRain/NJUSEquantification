"""Run Task 4 benchmark-relative uncertainty and rolling-risk diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qd.task4_excess_significance import run_task4_excess_significance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument("--recent-days", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_task4_excess_significance(
        block_length=args.block_length,
        iterations=args.iterations,
        rolling_window=args.rolling_window,
        recent_days=args.recent_days,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "rows": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()

