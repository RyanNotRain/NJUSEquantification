"""Task 5: LSTM 分钟涨跌预测。

用法:
    python scripts/run_lstm.py
    python scripts/run_lstm.py --stocks 20 --epochs 100 --hidden 256
    python scripts/run_lstm.py --stocks 000012,000014,000019  # 指定股票
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qd.lstm_model import run_lstm_pipeline  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="LSTM 分钟涨跌预测")
    p.add_argument("--stocks", type=str, default="10",
                   help="股票代码(逗号分隔)或数量(数字)")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--seq-len", type=int, default=60, help="输入序列长度(分钟)")
    p.add_argument("--hidden", type=int, default=128, help="LSTM 隐藏层大小")
    p.add_argument("--data-dir", type=Path, default=None, help="分钟频数据目录")
    args = p.parse_args()

    # 解析 stocks 参数
    stocks_arg = args.stocks
    if "," in stocks_arg:
        stock_codes = [s.strip() for s in stocks_arg.split(",")]
        n_stocks = None
    else:
        stock_codes = None
        n_stocks = int(stocks_arg)

    result = run_lstm_pipeline(
        stock_codes=stock_codes,
        n_stocks=n_stocks or 10,
        seq_len=args.seq_len,
        hidden_size=args.hidden,
        epochs=args.epochs,
        data_dir=args.data_dir,
    )

    print(f"\n最终测试准确率: {result['test_accuracy']*100:.2f}%")

    if result.get("test_metrics"):
        m = result["test_metrics"]
        print(f"精确率: {m['precision']*100:.2f}%  召回率: {m['recall']*100:.2f}%  F1: {m['f1']:.4f}")


if __name__ == "__main__":
    main()