"""Turnover-aware and cost-stress extensions of the strict Task 4 ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import OUTPUT_DIR
from .factors import compute_all_factors, load_daily_data
from .strategy_analysis import relative_metrics
from .task4_strict import run_strict_backtest


def run_task4_robustness(output_dir: str | Path = OUTPUT_DIR) -> dict:
    root = Path(output_dir)
    target = root / "backtest_robustness"
    target.mkdir(parents=True, exist_ok=True)
    daily = load_daily_data(root / "daily")
    bundle = compute_all_factors(daily)
    factors = {name: value for name, value in bundle.items() if name != "forward_return_1d"}
    market = daily["close"].pct_change(fill_method=None).mean(axis=1, skipna=True)

    rows: list[dict] = []
    paths: dict[str, pd.DataFrame] = {}
    for policy in ("top_n", "buffered"):
        for sell_fee_bps in (0.0, 5.0, 10.0, 20.0):
            result = run_strict_backtest(
                factors,
                daily,
                execution="close",
                method="adaptive",
                sell_fee=sell_fee_bps / 10_000.0,
                selection_policy=policy,
            )
            first_execution = pd.to_datetime(result["selections"]["execution_date"]).min()
            strategy = result["daily_return"].loc[result["daily_return"].index >= first_execution]
            comparison = relative_metrics(strategy, market.reindex(strategy.index))
            label = f"{policy}_{sell_fee_bps:g}bp"
            paths[label] = pd.concat([
                result["nav"].rename("nav"),
                result["daily_return"].rename("daily_return"),
                result["turnover"].rename("sell_turnover"),
            ], axis=1)
            rows.append({
                "selection_policy": policy,
                "sell_fee_bps": sell_fee_bps,
                "strategy_total_return": result["metrics"]["total_return"],
                "from_first_execution_geometric_excess": comparison["geometric_excess_return"],
                "information_ratio": comparison["information_ratio"],
                "average_sell_turnover": result["metrics"]["average_sell_turnover"],
                "max_drawdown": result["metrics"]["max_drawdown"],
                "first_execution_date": first_execution,
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(target / "cost_stress.csv", index=False, float_format="%.10f")
    for label, path in paths.items():
        path.to_csv(target / f"{label}_daily.csv", float_format="%.10f")
    base = metrics.loc[metrics["sell_fee_bps"].eq(5.0)].set_index("selection_policy")
    summary = {
        "status": "completed",
        "ledger": "cash and shares with price-drifted weights and locked halts",
        "top_n_5bp": base.loc["top_n"].to_dict(),
        "buffered_5bp": base.loc["buffered"].to_dict(),
        "buffer_rule": "Top-20 rank buffer, at most three replacements, trailing liquidity/volatility filters",
        "cost_scope": "sell fee only; buy cost, slippage, impact, and final forced liquidation omitted",
    }
    (target / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary

