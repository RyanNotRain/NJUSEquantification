"""Run validation-frozen multi-horizon executable-return research."""

from __future__ import annotations

import argparse
import json

from qd.config import OUTPUT_DIR
from qd.tradable_return_research import run_tradable_return_research


def main() -> None:
    parser = argparse.ArgumentParser(description="screen executable return horizons and low-turnover rules")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run_tradable_return_research(
        output_dir=args.output_dir, sell_fee_bps=args.sell_fee_bps, seed=args.seed
    )
    print(json.dumps({
        "status": result["status"],
        "test_model_rows": result["test_model_rows"],
        "test_strategy_rows": result["test_strategy_rows"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
