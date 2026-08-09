"""Turn saved full-window LSTM probabilities into an auditable strategy."""

from __future__ import annotations

import argparse
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_strategy import SIDES, TIERS, WEIGHTINGS, run_strategy_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="evaluate leakage-safe minute LSTM probability strategies"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=OUTPUT_DIR / "lstm_full" / "test_predictions.csv",
    )
    parser.add_argument("--close-dir", type=Path, default=OUTPUT_DIR / "minute" / "close")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_strategy")
    parser.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS))
    parser.add_argument(
        "--weightings", nargs="+", choices=WEIGHTINGS, default=list(WEIGHTINGS)
    )
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--balanced-confidence-threshold",
        type=float,
        help="frozen validation threshold; only needed if selected_balanced is absent",
    )
    parser.add_argument(
        "--strict-confidence-threshold",
        type=float,
        help="frozen validation threshold; only needed if selected_strict is absent",
    )
    parser.add_argument(
        "--allow-flat-argmax",
        action="store_true",
        help="allow a directional position even when flat has the largest probability",
    )
    parser.add_argument("--side", choices=SIDES, default="long_short")
    parser.add_argument("--base-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--cost-grid-bps",
        type=float,
        nargs="+",
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    thresholds = {
        key: value
        for key, value in {
            "balanced": args.balanced_confidence_threshold,
            "strict": args.strict_confidence_threshold,
        }.items()
        if value is not None
    }
    result = run_strategy_suite(
        args.predictions,
        args.close_dir,
        args.out_dir,
        tiers=args.tiers,
        weightings=args.weightings,
        score_threshold=args.score_threshold,
        confidence_thresholds=thresholds,
        require_directional_argmax=not args.allow_flat_argmax,
        side=args.side,
        base_cost_bps=args.base_cost_bps,
        cost_grid_bps=args.cost_grid_bps,
        overwrite=args.overwrite,
    )
    columns = [
        "strategy",
        "selection_rate",
        "active_signal_rate",
        "gross_total_return",
        "net_total_return",
        "average_daily_turnover",
        "break_even_cost_bps",
    ]
    print(result["summary"][columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\noutputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
