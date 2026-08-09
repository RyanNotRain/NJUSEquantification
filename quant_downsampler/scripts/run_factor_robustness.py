"""Run temporal, regime, bootstrap and factor-correlation diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from qd.factor_robustness import RobustnessConfig, run_robustness_analysis


def _parse_exposure(values: list[str], kind: str) -> dict[str, pd.DataFrame]:
    exposures: dict[str, pd.DataFrame] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} exposure must use NAME=CSV syntax: {value}")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name or not path.is_file():
            raise ValueError(f"invalid {kind} exposure: {value}")
        frame = pd.read_csv(path, index_col=0)
        frame.index = pd.to_datetime(frame.index)
        frame.columns = frame.columns.astype(str)
        exposures[name] = frame.sort_index()
    return exposures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="factor temporal stability and robustness diagnostics"
    )
    parser.add_argument("--factor-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument("--rolling-min-periods", type=int, default=20)
    parser.add_argument("--bootstrap-iterations", type=int, default=2_000)
    parser.add_argument("--bootstrap-block-length", type=int, default=5)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=20250809)
    parser.add_argument("--min-stocks", type=int, default=30)
    parser.add_argument("--direction-warmup", type=int, default=20)
    parser.add_argument(
        "--continuous-exposure",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="real date-by-stock numeric exposure; repeat as needed",
    )
    parser.add_argument(
        "--categorical-exposure",
        action="append",
        default=[],
        metavar="NAME=CSV",
        help="real date-by-stock category exposure; repeat as needed",
    )
    args = parser.parse_args()

    config = RobustnessConfig(
        rolling_window=args.rolling_window,
        rolling_min_periods=args.rolling_min_periods,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_block_length=args.bootstrap_block_length,
        bootstrap_confidence=args.bootstrap_confidence,
        random_seed=args.random_seed,
        min_stocks=args.min_stocks,
        direction_warmup=args.direction_warmup,
    )
    manifest = run_robustness_analysis(
        factor_dir=args.factor_dir,
        out_dir=args.out_dir,
        config=config,
        continuous_exposures=_parse_exposure(
            args.continuous_exposure, "continuous"
        ),
        categorical_exposures=_parse_exposure(
            args.categorical_exposure, "categorical"
        ),
    )
    analysis = manifest["analysis"]
    neutralization = manifest["neutralization"]
    print(
        "factor robustness complete: "
        f"{analysis['date_start']} to {analysis['date_end']}, "
        f"{analysis['n_stocks']} stocks; "
        f"neutralization={neutralization['status']}"
    )


if __name__ == "__main__":
    main()

