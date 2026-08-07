"""Build and evaluate the README example factor plus three new factors."""

from __future__ import annotations

import argparse
from pathlib import Path

from qd.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="factor construction and evaluation")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary, _, _ = run_evaluation(args.data_dir, args.out_dir)
    print(summary[["IC", "IR", "ICIR", "rank_IC", "rank_IR", "rank_ICIR", "n_days"]])


if __name__ == "__main__":
    main()
