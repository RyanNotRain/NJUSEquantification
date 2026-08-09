"""Train leakage-safe traditional next-minute return regressors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_model import DEFAULT_SPLITS
from qd.lstm_return_baselines import run_return_baselines


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
        description="train Ridge and HistGB next-minute signed-return baselines"
    )
    parser.add_argument("--stocks", default="5", help="count or comma-separated codes")
    parser.add_argument("--seq-len", type=_positive_int, default=60)
    parser.add_argument("--base-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--cost-grid-bps",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument("--max-train-samples", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_DIR / "lstm_return_baselines"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-start", default=DEFAULT_SPLITS["train"][0])
    parser.add_argument("--train-end", default=DEFAULT_SPLITS["train"][1])
    parser.add_argument("--val-start", default=DEFAULT_SPLITS["val"][0])
    parser.add_argument("--val-end", default=DEFAULT_SPLITS["val"][1])
    parser.add_argument("--test-start", default=DEFAULT_SPLITS["test"][0])
    parser.add_argument("--test-end", default=DEFAULT_SPLITS["test"][1])
    args = parser.parse_args()

    stock_codes, n_stocks = _stocks(args.stocks)
    result = run_return_baselines(
        stock_codes=stock_codes,
        n_stocks=n_stocks,
        seq_len=args.seq_len,
        splits={
            "train": (args.train_start, args.train_end),
            "val": (args.val_start, args.val_end),
            "test": (args.test_start, args.test_end),
        },
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        base_cost_bps=args.base_cost_bps,
        cost_grid_bps=args.cost_grid_bps,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                name: {
                    "mae_bps": metrics["mae_bps"],
                    "rmse_bps": metrics["rmse_bps"],
                    "pearson": metrics["pearson"],
                    "spearman": metrics["spearman"],
                    "direction_hit_rate": metrics[
                        "direction_hit_rate_both_nonzero"
                    ],
                }
                for name, metrics in result["test_metrics"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        result["strategy_comparison"][[
            "model",
            "side",
            "validation_frozen_threshold_bps",
            "gross_total_return",
            "net_total_return",
            "average_daily_turnover",
            "break_even_cost_bps",
            "row_coverage",
        ]].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"\noutputs written to {args.out_dir}")


if __name__ == "__main__":
    main()

