"""Train leakage-safe Logistic Regression and histogram-tree baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_baselines import run_lstm_baselines
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
            raise argparse.ArgumentTypeError("stock codes cannot be empty")
        return stocks, len(stocks)
    count = _positive_int(value)
    return None, count


def _float_list(value: str) -> list[float]:
    try:
        parsed = [float(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def _int_list(value: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 1 for item in parsed):
        raise argparse.ArgumentTypeError("all values must exceed one")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "train traditional baselines on the same legal full-window "
            "down/flat/up task as the enhanced LSTM"
        )
    )
    parser.add_argument("--stocks", default="5", help="count or comma-separated codes")
    parser.add_argument("--seq-len", type=_positive_int, default=60)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_baselines")
    parser.add_argument("--feature-set", choices=("legacy", "enhanced"), default="enhanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=_positive_int, default=None)
    parser.add_argument("--max-validation-samples", type=_positive_int, default=None)
    parser.add_argument("--logistic-c", type=_float_list, default=[0.1, 1.0])
    parser.add_argument("--logistic-max-iter", type=_positive_int, default=120)
    parser.add_argument("--tree-leaves", type=_int_list, default=[15, 31])
    parser.add_argument("--tree-max-iter", type=_positive_int, default=50)
    parser.add_argument("--tree-learning-rate", type=float, default=0.08)
    parser.add_argument("--tree-min-samples-leaf", type=_positive_int, default=50)
    parser.add_argument("--tree-l2", type=float, default=1.0)
    parser.add_argument("--train-start", default=DEFAULT_SPLITS["train"][0])
    parser.add_argument("--train-end", default=DEFAULT_SPLITS["train"][1])
    parser.add_argument("--val-start", default=DEFAULT_SPLITS["val"][0])
    parser.add_argument("--val-end", default=DEFAULT_SPLITS["val"][1])
    parser.add_argument("--test-start", default=DEFAULT_SPLITS["test"][0])
    parser.add_argument("--test-end", default=DEFAULT_SPLITS["test"][1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.tree_learning_rate <= 0 or args.tree_l2 < 0:
        parser.error("tree learning rate must be positive and L2 must be non-negative")

    stocks, n_stocks = _stocks(args.stocks)
    splits = {
        "train": (args.train_start, args.train_end),
        "val": (args.val_start, args.val_end),
        "test": (args.test_start, args.test_end),
    }
    logistic_grid = [
        {"C": value, "max_iter": args.logistic_max_iter, "tol": 1e-3}
        for value in args.logistic_c
    ]
    tree_grid = [
        {
            "learning_rate": args.tree_learning_rate,
            "max_iter": args.tree_max_iter,
            "max_leaf_nodes": leaves,
            "min_samples_leaf": args.tree_min_samples_leaf,
            "l2_regularization": args.tree_l2,
        }
        for leaves in args.tree_leaves
    ]
    result = run_lstm_baselines(
        stock_codes=stocks,
        n_stocks=n_stocks,
        seq_len=args.seq_len,
        splits=splits,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        feature_set=args.feature_set,
        logistic_grid=logistic_grid,
        tree_grid=tree_grid,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    report = result["report"]
    print(json.dumps({
        "out_dir": result["out_dir"],
        "majority_prior": report["majority_prior_baseline"]["test"],
        "logistic_regression": report["models"]["logistic_regression"]["test"],
        "hist_gradient_boosting": report["models"]["hist_gradient_boosting"]["test"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
