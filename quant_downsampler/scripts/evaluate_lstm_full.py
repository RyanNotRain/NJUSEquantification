"""Strictly reload and re-evaluate the saved full-window LSTM ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_full import evaluate_full_model


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate saved down/flat/up LSTM ensemble")
    parser.add_argument("--model", type=Path, default=OUTPUT_DIR / "lstm_full" / "model.pt")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate_full_model(
        args.model,
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
