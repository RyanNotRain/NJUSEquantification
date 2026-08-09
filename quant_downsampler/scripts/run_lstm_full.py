"""Train the reproducible three-component full-window LSTM ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_full_training import run_full_training
from qd.lstm_model import DEFAULT_SPLITS


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _stocks(value: str) -> tuple[list[str] | None, int]:
    if "," in value or not value.isdigit():
        stocks = [item.strip() for item in value.split(",")]
        if not stocks or any(not item for item in stocks):
            raise argparse.ArgumentTypeError("comma-separated stock codes cannot be empty")
        return stocks, len(stocks)
    count = _positive_int(value)
    return None, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "train direction + movement + joint LSTMs, select fusion on validation, "
            "then evaluate the final test split once"
        )
    )
    parser.add_argument("--stocks", default="5", help="count or comma-separated full codes")
    parser.add_argument("--seq-len", type=_positive_int, default=60)
    parser.add_argument("--hidden", type=_positive_int, default=64)
    parser.add_argument("--layers", type=_positive_int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=_positive_int, default=12)
    parser.add_argument("--direction-epochs", type=_positive_int, default=None)
    parser.add_argument("--movement-epochs", type=_positive_int, default=None)
    parser.add_argument("--joint-epochs", type=_positive_int, default=None)
    parser.add_argument("--patience", type=_positive_int, default=5)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--direction-unweighted", action="store_true")
    parser.add_argument("--movement-unweighted", action="store_true")
    parser.add_argument("--joint-weighted", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_runs" / "full")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-start", default=DEFAULT_SPLITS["train"][0])
    parser.add_argument("--train-end", default=DEFAULT_SPLITS["train"][1])
    parser.add_argument("--val-start", default=DEFAULT_SPLITS["val"][0])
    parser.add_argument("--val-end", default=DEFAULT_SPLITS["val"][1])
    parser.add_argument("--test-start", default=DEFAULT_SPLITS["test"][0])
    parser.add_argument("--test-end", default=DEFAULT_SPLITS["test"][1])
    parser.add_argument("--move-bias-min", type=float, default=-0.30)
    parser.add_argument("--move-bias-max", type=float, default=0.30)
    parser.add_argument("--move-bias-step", type=float, default=0.05)
    parser.add_argument("--joint-weight-step", type=float, default=0.05)
    parser.add_argument(
        "--fusion-objective",
        choices=("macro_f1_then_accuracy", "accuracy_then_macro_f1"),
        default="macro_f1_then_accuracy",
        help="validation-only objective used to freeze move bias and fusion weight",
    )
    parser.add_argument("--balanced-quantile", type=float, default=0.70)
    parser.add_argument("--strict-quantile", type=float, default=0.90)
    args = parser.parse_args()

    stock_codes, n_stocks = _stocks(args.stocks)
    splits = {
        "train": (args.train_start, args.train_end),
        "val": (args.val_start, args.val_end),
        "test": (args.test_start, args.test_end),
    }
    result = run_full_training(
        stock_codes=stock_codes,
        n_stocks=n_stocks,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        direction_epochs=args.direction_epochs or args.epochs,
        movement_epochs=args.movement_epochs or args.epochs,
        joint_epochs=args.joint_epochs or args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        direction_class_weighted=not args.direction_unweighted,
        movement_class_weighted=not args.movement_unweighted,
        joint_class_weighted=args.joint_weighted,
        seed=args.seed,
        splits=splits,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=args.device,
        move_bias_min=args.move_bias_min,
        move_bias_max=args.move_bias_max,
        move_bias_step=args.move_bias_step,
        joint_weight_step=args.joint_weight_step,
        fusion_objective=args.fusion_objective,
        balanced_quantile=args.balanced_quantile,
        strict_quantile=args.strict_quantile,
        overwrite=args.overwrite,
    )
    metrics = result["test_metrics"]
    print(json.dumps({
        "out_dir": result["out_dir"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "majority_baseline": metrics["majority_baseline"],
        "validation_accuracy": metrics["validation_accuracy"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
