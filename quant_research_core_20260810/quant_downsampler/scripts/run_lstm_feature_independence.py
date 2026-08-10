"""Run Task 5 feature de-redundancy and ensemble component diversity audit."""
from __future__ import annotations

import argparse
import json

from qd.config import OUTPUT_DIR
from qd.lstm_feature_independence import run_lstm_feature_independence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--max-training-rows", type=int, default=40_000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--reuse-frozen-selection", action="store_true")
    args = parser.parse_args()
    result = run_lstm_feature_independence(
        output_dir=args.output_dir,
        correlation_threshold=args.correlation_threshold,
        max_training_rows=args.max_training_rows,
        epochs=args.epochs,
        sell_fee_bps=args.sell_fee_bps,
        device=args.device,
        reuse_frozen_selection=args.reuse_frozen_selection,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
