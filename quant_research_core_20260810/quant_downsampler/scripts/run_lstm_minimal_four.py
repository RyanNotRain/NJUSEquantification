"""Train the minimal four-feature Task 5 baseline mapped from the prompt."""
from __future__ import annotations

import argparse
import json

from qd.config import OUTPUT_DIR
from qd.lstm_minimal_four import run_lstm_minimal_four


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args()
    result = run_lstm_minimal_four(
        output_dir=args.output_dir,
        epochs=args.epochs,
        sell_fee_bps=args.sell_fee_bps,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
