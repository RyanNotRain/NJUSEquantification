"""Train factor return-regression and direct-Sharpe portfolio models."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qd.factor_learning import run_factor_models, run_rolling_factor_models

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--lr", type=float, default=.03)
p.add_argument("--cost-bps", type=float, default=5.0)
p.add_argument("--out", type=Path, default=None)
p.add_argument("--rolling", action="store_true", help="walk-forward refit using only prior data")
p.add_argument("--window", type=int, default=120, help="rolling training days")
p.add_argument("--rebalance-days", type=int, default=20, help="days between refits")
a = p.parse_args()
if a.rolling:
    run_rolling_factor_models(a.window, a.rebalance_days, a.epochs, a.lr, a.cost_bps, a.out)
else:
    run_factor_models(a.epochs, a.lr, a.cost_bps, a.out)
