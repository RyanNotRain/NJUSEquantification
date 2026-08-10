"""Train and evaluate the exploratory magnitude-aware LSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_components import DEFAULT_SPLITS
from qd.lstm_magnitude import run_magnitude_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="direction + signed-return multi-task LSTM")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--return-loss-weight", type=float, default=0.25)
    parser.add_argument("--sell-fee-bps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_magnitude")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-start", default=DEFAULT_SPLITS["train"][0])
    parser.add_argument("--train-end", default=DEFAULT_SPLITS["train"][1])
    parser.add_argument("--val-start", default=DEFAULT_SPLITS["val"][0])
    parser.add_argument("--val-end", default=DEFAULT_SPLITS["val"][1])
    parser.add_argument("--test-start", default=DEFAULT_SPLITS["test"][0])
    parser.add_argument("--test-end", default=DEFAULT_SPLITS["test"][1])
    args = parser.parse_args()
    result = run_magnitude_experiment(
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        return_loss_weight=args.return_loss_weight, sell_fee_bps=args.sell_fee_bps,
        seed=args.seed, device=args.device, data_dir=args.data_dir,
        out_dir=args.out_dir, overwrite=args.overwrite,
        splits={
            "train": (args.train_start, args.train_end),
            "val": (args.val_start, args.val_end),
            "test": (args.test_start, args.test_end),
        },
    )
    print(json.dumps(result["test_metrics"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
