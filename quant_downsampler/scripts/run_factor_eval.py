"""Task 2+3: 因子构建与评价。

用法:
    python scripts/run_factor_eval.py
    python scripts/run_factor_eval.py --save  # 保存因子到 output/factors/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qd.evaluation import run_evaluation  # noqa: E402
from qd.factors import save_factors  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="因子构建与评价")
    p.add_argument("--save", action="store_true", help="保存因子 CSV")
    p.add_argument("--data-dir", type=Path, default=None, help="日频数据目录")
    args = p.parse_args()

    summary, layerings, factors = run_evaluation(args.data_dir)

    if args.save:
        save_factors(factors)

    print("\n完成。排名前 3 的因子:")
    print(summary.head(3)[["IC", "IR", "ICIR", "rank_IC", "rank_IR", "IC_positive_ratio"]])


if __name__ == "__main__":
    main()