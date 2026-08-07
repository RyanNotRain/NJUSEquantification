"""Train and evaluate the next-minute LSTM model."""

from __future__ import annotations

import argparse
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_model import run_lstm_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="next-minute LSTM")
    parser.add_argument("--stocks", default="5", help="count or comma-separated full codes")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-set", choices=("legacy", "enhanced"), default="enhanced")
    parser.add_argument("--scaler", choices=("global", "per_stock"), default="per_stock")
    parser.add_argument("--model-version", choices=("legacy", "residual"), default="residual")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-stock-id", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-threshold-calibration", action="store_true")
    parser.add_argument(
        "--target-mode",
        choices=("nonflat_binary", "three_class", "up_vs_not_up", "move_vs_flat"),
        default="nonflat_binary",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow replacing files in an existing non-empty output directory",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if "," in args.stocks:
        stock_codes = [x.strip() for x in args.stocks.split(",")]
        n_stocks = len(stock_codes)
    else:
        stock_codes = None
        n_stocks = int(args.stocks)
    out_dir = args.out_dir or (OUTPUT_DIR / "lstm_runs" / args.target_mode)
    result = run_lstm_pipeline(
        stock_codes=stock_codes,
        n_stocks=n_stocks,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        out_dir=out_dir,
        seed=args.seed,
        feature_set=args.feature_set,
        include_stock_id=not args.no_stock_id,
        scaler_mode=args.scaler,
        model_version=args.model_version,
        num_layers=args.layers,
        dropout=args.dropout,
        class_weighted=not args.no_class_weights,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        calibrate_threshold=not args.no_threshold_calibration,
        target_mode=args.target_mode,
        overwrite=args.overwrite,
    )
    print(result["test_metrics"])


if __name__ == "__main__":
    main()
