"""Run validation-frozen T+1 raw-return versus market-excess target research."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qd.t1_excess_return_research import run_t1_excess_return_research


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_t1_excess_return_research(
        sell_fee_bps=args.sell_fee_bps, seed=args.seed
    )
    print(json.dumps({"status": result["status"], "test_rows": result["test_rows"]}, indent=2))


if __name__ == "__main__":
    main()

