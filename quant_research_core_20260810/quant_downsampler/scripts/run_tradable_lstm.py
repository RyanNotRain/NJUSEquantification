"""Train the compact LSTM on the validation-screened executable horizon."""

from __future__ import annotations

import argparse
import json

from qd.config import OUTPUT_DIR
from qd.tradable_lstm import run_tradable_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description="validation-screened T+1 multi-task LSTM")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    result = run_tradable_lstm(
        output_dir=args.output_dir, epochs=args.epochs,
        sell_fee_bps=args.sell_fee_bps, device=args.device,
    )
    print(json.dumps({
        "horizon": result["horizon"],
        "test_return_metrics": result["test_return_metrics"],
        "test_strategy": result["test_strategy"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
