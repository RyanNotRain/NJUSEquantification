"""Run Task 5 as a minute-level down/flat/up classification problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qd.lstm_model import run_lstm_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM next-minute down/flat/up classification")
    parser.add_argument(
        "--stocks", type=str, default="5",
        help="number of stocks or a comma-separated stock-code list",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--class-weight-power", type=float, default=0.5,
        help="0=no class weighting, 0.5=square-root inverse frequency, 1=full inverse frequency",
    )
    parser.add_argument("--direction-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--model-type", choices=("direct", "hierarchical"), default="direct",
        help="formal direct three-class model or experimental two-stage model",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if "," in args.stocks:
        stock_codes = [value.strip() for value in args.stocks.split(",") if value.strip()]
        n_stocks = len(stock_codes)
    else:
        stock_codes = None
        n_stocks = int(args.stocks)

    result = run_lstm_pipeline(
        stock_codes=stock_codes, n_stocks=n_stocks, seq_len=args.seq_len,
        hidden_size=args.hidden, epochs=args.epochs, batch_size=args.batch_size,
        data_dir=args.data_dir, out_dir=args.out, seed=args.seed,
        class_weight_power=args.class_weight_power,
        direction_loss_weight=args.direction_loss_weight,
        model_type=args.model_type,
    )
    metrics = result.get("test_metrics")
    if metrics:
        print(
            f"Accuracy={metrics['accuracy']:.2%}, balanced_accuracy="
            f"{metrics['balanced_accuracy']:.2%}, macro_F1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
