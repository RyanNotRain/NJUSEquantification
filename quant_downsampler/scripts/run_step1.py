"""入口脚本:跑第一步降采样。

用法:
    python -m scripts.run_step1                # 处理所有日期
    python -m scripts.run_step1 --dates 20250401 20250402
    python -m scripts.run_step1 --out /tmp/out  # 自定义输出目录
    python -m scripts.run_step1 --overwrite     # 覆盖已有输出
    python -m scripts.run_step1 --days-per-chunk 1  # 每块 1 天(更省内存)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 让脚本能直接以 `python scripts/run_step1.py` 运行
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qd.config import DAYS_PER_CHUNK, OUTPUT_DIR  # noqa: E402
from qd.pipeline import run  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="兆信基金 Task 1:逐笔降采样")
    p.add_argument(
        "--dates", nargs="+", default=None,
        help="要处理的日期(YYYYMMDD),默认处理 data/TRADE 下所有日期",
    )
    p.add_argument(
        "--out", type=Path, default=OUTPUT_DIR,
        help=f"输出根目录,默认 {OUTPUT_DIR}",
    )
    p.add_argument(
        "--days-per-chunk", type=int, default=DAYS_PER_CHUNK,
        help=f"分块大小(同时处理多少天),默认 {DAYS_PER_CHUNK}",
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已存在的全部日频和分钟频表",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run(
        dates=args.dates,
        out_dir=args.out,
        days_per_chunk=args.days_per_chunk,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
