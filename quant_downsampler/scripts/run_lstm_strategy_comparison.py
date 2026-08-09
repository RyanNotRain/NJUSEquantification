"""Compare LSTM, HistGB and hybrid strategies on identical minute horizons."""

from __future__ import annotations

import argparse
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_strategy_comparison import (
    DEFAULT_PROBABILITY_COLUMNS,
    DEFAULT_SOURCE_PATHS,
    run_strategy_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run a strict same-sample economic comparison of minute models"
    )
    parser.add_argument(
        "--lstm-predictions",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["original_lstm"],
    )
    parser.add_argument(
        "--histgb-predictions",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["hist_gradient_boosting"],
    )
    parser.add_argument(
        "--hybrid-predictions",
        type=Path,
        default=DEFAULT_SOURCE_PATHS["hybrid"],
    )
    parser.add_argument(
        "--magnitude-predictions",
        type=Path,
        help="optional predictions containing expected_return_bps",
    )
    parser.add_argument(
        "--magnitude-threshold-bps",
        type=float,
        default=0.0,
        help="pre-declared absolute expected-return threshold for the magnitude model",
    )
    parser.add_argument(
        "--ridge-return-predictions",
        type=Path,
        help="optional canonical Ridge predictions containing expected_return_bps",
    )
    parser.add_argument(
        "--ridge-return-threshold-bps",
        type=float,
        default=5.0,
        help="validation-frozen Ridge opening threshold",
    )
    parser.add_argument(
        "--histgb-return-predictions",
        type=Path,
        help="optional canonical HistGB-regressor predictions containing expected_return_bps",
    )
    parser.add_argument(
        "--histgb-return-threshold-bps",
        type=float,
        default=5.0,
        help="validation-frozen HistGB-regressor opening threshold",
    )
    parser.add_argument(
        "--close-dir", type=Path, default=OUTPUT_DIR / "minute" / "close"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUTPUT_DIR / "lstm_strategy_comparison"
    )
    parser.add_argument("--base-cost-bps", type=float, default=5.0)
    parser.add_argument("--side", choices=("long_short", "long_only"), default="long_short")
    parser.add_argument("--weighting", choices=("equal", "confidence"), default="confidence")
    parser.add_argument(
        "--allow-flat-argmax",
        action="store_true",
        help="allow probability-gap trades even when flat has the largest probability",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    sources = {
        "original_lstm": args.lstm_predictions,
        "hist_gradient_boosting": args.histgb_predictions,
        "hybrid": args.hybrid_predictions,
    }
    mappings = dict(DEFAULT_PROBABILITY_COLUMNS)
    modes: dict[str, str] = {}
    thresholds: dict[str, float] = {}
    if args.magnitude_predictions is not None:
        sources["magnitude_lstm"] = args.magnitude_predictions
        modes["magnitude_lstm"] = "expected_return_bps"
        thresholds["magnitude_lstm"] = args.magnitude_threshold_bps
    if args.ridge_return_predictions is not None:
        sources["ridge_return"] = args.ridge_return_predictions
        modes["ridge_return"] = "expected_return_bps"
        thresholds["ridge_return"] = args.ridge_return_threshold_bps
    if args.histgb_return_predictions is not None:
        sources["histgb_return"] = args.histgb_return_predictions
        modes["histgb_return"] = "expected_return_bps"
        thresholds["histgb_return"] = args.histgb_return_threshold_bps

    result = run_strategy_comparison(
        sources,
        args.close_dir,
        args.out_dir,
        probability_columns=mappings,
        signal_modes=modes,
        score_thresholds=thresholds,
        side=args.side,
        weighting=args.weighting,
        require_directional_argmax=not args.allow_flat_argmax,
        base_cost_bps=args.base_cost_bps,
        overwrite=args.overwrite,
    )
    columns = [
        "name",
        "kind",
        "gross_total_return",
        "net_total_return",
        "net_relative_to_market_proxy",
        "average_daily_turnover",
        "net_max_drawdown",
        "break_even_cost_bps",
        "row_coverage",
    ]
    print(
        result["summary"][columns].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print(f"\noutputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
