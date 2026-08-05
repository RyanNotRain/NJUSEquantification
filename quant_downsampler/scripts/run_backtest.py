"""Task 4: 策略回测。

用法:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --lookback 120 --top-n 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qd.backtest import run_full_backtest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="策略回测")
    p.add_argument("--data-dir", type=Path, default=None, help="日频数据目录")
    p.add_argument("--lookback", type=int, default=60, help="IC 权重回看窗口")
    p.add_argument("--top-n", type=int, default=10, help="每期选股数")
    args = p.parse_args()

    run_full_backtest(args.data_dir, lookback=args.lookback, top_n=args.top_n)


if __name__ == "__main__":
    main()