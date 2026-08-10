"""Run factor stability, market-regime, and single-factor benchmark analysis."""

from __future__ import annotations

import argparse
import json

from qd.factor_robustness import run_factor_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--block-length", type=int, default=5)
    args = parser.parse_args()
    result = run_factor_robustness(iterations=args.iterations, block_length=args.block_length)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

