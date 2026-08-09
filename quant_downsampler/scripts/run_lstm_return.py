"""Train and evaluate the independent direction-plus-magnitude LSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_model import DEFAULT_SPLITS
from qd.lstm_return import run_return_lstm


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _stocks(value: str) -> tuple[list[str] | None, int]:
    if "," in value or not value.isdigit():
        stocks = [item.strip() for item in value.split(",")]
        if not stocks or any(not item for item in stocks):
            raise argparse.ArgumentTypeError("stock codes cannot be empty")
        return stocks, len(stocks)
    count = _positive_int(value)
    return None, count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "train a shared-encoder down/flat/up plus absolute-return LSTM; "
            "freeze validation choices before loading test dates"
        )
    )
    parser.add_argument("--stocks", default="5", help="count or comma-separated codes")
    parser.add_argument("--seq-len", type=_positive_int, default=60)
    parser.add_argument("--hidden", type=_positive_int, default=64)
    parser.add_argument("--layers", type=_positive_int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=_positive_int, default=8)
    parser.add_argument("--patience", type=_positive_int, default=3)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--magnitude-lambdas",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 1.0],
        help="validation-selected SmoothL1 loss weights",
    )
    parser.add_argument("--class-weighted", action="store_true")
    parser.add_argument("--clip-quantile", type=float, default=0.995)
    parser.add_argument("--base-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--cost-grid-bps",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument(
        "--max-train-samples",
        type=_positive_int,
        default=None,
        help="optional reproducible training-only subsample for a quick feasibility run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_return")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-start", default=DEFAULT_SPLITS["train"][0])
    parser.add_argument("--train-end", default=DEFAULT_SPLITS["train"][1])
    parser.add_argument("--val-start", default=DEFAULT_SPLITS["val"][0])
    parser.add_argument("--val-end", default=DEFAULT_SPLITS["val"][1])
    parser.add_argument("--test-start", default=DEFAULT_SPLITS["test"][0])
    parser.add_argument("--test-end", default=DEFAULT_SPLITS["test"][1])
    args = parser.parse_args()

    stock_codes, n_stocks = _stocks(args.stocks)
    result = run_return_lstm(
        stock_codes=stock_codes,
        n_stocks=n_stocks,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        magnitude_lambdas=args.magnitude_lambdas,
        class_weighted=args.class_weighted,
        clip_quantile=args.clip_quantile,
        base_cost_bps=args.base_cost_bps,
        cost_grid_bps=args.cost_grid_bps,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
        splits={
            "train": (args.train_start, args.train_end),
            "val": (args.val_start, args.val_end),
            "test": (args.test_start, args.test_end),
        },
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=args.device,
        overwrite=args.overwrite,
    )
    metrics = result["test_metrics"]
    print(json.dumps({
        "out_dir": result["out_dir"],
        "accuracy": metrics["classification"]["accuracy"],
        "macro_f1": metrics["classification"]["macro_f1"],
        "magnitude_mae_bps": metrics["magnitude"]["mae_bps"],
        "expected_return_spearman_ic": metrics["signed_expected_return"]["spearman_ic"],
        "group_top_minus_bottom_bps": metrics["grouped_returns"]["top_minus_bottom_bps"],
        "strict_replay": result["replay_audit"]["passed"],
    }, indent=2, ensure_ascii=False))
    print(result["strategy_comparison"][[
        "comparison_signal",
        "side",
        "gross_total_return",
        "net_total_return",
        "average_daily_turnover",
        "break_even_cost_bps",
    ]].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()

