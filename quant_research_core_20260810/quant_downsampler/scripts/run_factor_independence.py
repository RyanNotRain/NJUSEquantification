"""Run leakage-aware factor de-redundancy and strict Task 4 comparison."""
from __future__ import annotations

import argparse
import json

from qd.factor_independence import run_factor_independence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-days", type=int, default=120)
    parser.add_argument("--correlation-threshold", type=float, default=0.60)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    args = parser.parse_args()
    result = run_factor_independence(
        calibration_days=args.calibration_days,
        correlation_threshold=args.correlation_threshold,
        lookback=args.lookback,
        top_n=args.top_n,
        sell_fee_bps=args.sell_fee_bps,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
