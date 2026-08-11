"""Validation-frozen T+1 research targeting market-adjusted returns directly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from .config import OUTPUT_DIR
from .lstm_magnitude import return_metrics
from .tradable_return_research import (
    MODEL_FAMILIES,
    _prepare_split,
    _strategy_path,
    fit_validation_selected_model,
    select_strategy_on_validation,
    strategy_metrics,
)


HORIZON = "t1_same_minute_open"
TARGETS = ("raw_return", "market_excess_return")


def add_market_adjusted_target(
    frame: pd.DataFrame,
    return_column: str = HORIZON,
) -> pd.DataFrame:
    """Subtract the contemporaneous equal-weight cross-sectional market return."""
    result = frame.copy()
    market_column = f"{return_column}__market_return"
    excess_column = f"{return_column}__market_excess"
    result[market_column] = result.groupby("target_time")[return_column].transform("mean")
    result[excess_column] = result[return_column] - result[market_column]
    return result


def _target_column(target: str) -> str:
    if target == "raw_return":
        return HORIZON
    if target == "market_excess_return":
        return f"{HORIZON}__market_excess"
    raise ValueError(f"unknown target {target}")


def _plot_comparison(
    metrics: pd.DataFrame,
    strategies: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    labels = [
        f"{target.replace('_return', '')}\n{family.replace('_regressor', '')}"
        for target in TARGETS for family in MODEL_FAMILIES
    ]
    metric_index = metrics.set_index(["target", "model"])
    strategy_index = strategies.set_index(["target", "model"])
    keys = [(target, family) for target in TARGETS for family in MODEL_FAMILIES]
    rank_ic = [metric_index.loc[key, "spearman_ic"] for key in keys]
    excess = [strategy_index.loc[key, "excess_vs_matched_market"] for key in keys]
    gap = [
        strategy_index.loc[key, "cumulative_return_gap_vs_matched_market"]
        for key in keys
    ]
    colors = ["#2563EB", "#D97706", "#059669", "#7C3AED"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(x, rank_ic, color=colors)
    axes[0].set_title("Test target Rank IC")
    axes[0].set_ylabel("Spearman IC")
    axes[1].bar(x, excess, color=colors)
    axes[1].set_title("T+1 geometric excess vs matched market")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].bar(x, gap, color=colors)
    axes[2].set_title("Strategy return - matched market return")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    for axis in axes:
        axis.axhline(0.0, color="black", lw=0.8)
        axis.set_xticks(x, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_t1_excess_return_research(
    output_dir: str | Path = OUTPUT_DIR,
    sell_fee_bps: float = 5.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare raw-return and direct market-excess targets on identical T+1 keys."""
    root = Path(output_dir)
    target_dir = root / "t1_excess_return_research"
    target_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(
        root / "lstm_ensemble" / "model.pt", map_location="cpu", weights_only=False
    )
    config = bundle["config"]
    stocks = list(config["stock_codes"])
    splits = {name: tuple(value) for name, value in config["splits"].items()}
    seq_len = int(config["seq_len"])
    minute_dir = root / "minute"

    print("loading train/validation T+1 targets; test remains unopened")
    train_x, train_frame = _prepare_split(stocks, splits["train"], seq_len, minute_dir)
    validation_x, validation_frame = _prepare_split(
        stocks, splits["val"], seq_len, minute_dir
    )
    train_frame = add_market_adjusted_target(train_frame)
    validation_frame = add_market_adjusted_target(validation_frame)
    models: dict[str, dict[str, Any]] = {target: {} for target in TARGETS}
    frozen: dict[str, dict[str, Any]] = {target: {} for target in TARGETS}
    model_grid_rows: list[pd.DataFrame] = []
    strategy_grid_rows: list[pd.DataFrame] = []

    for target in TARGETS:
        label_column = _target_column(target)
        train_mask = train_frame[label_column].notna().to_numpy()
        validation_mask = validation_frame[label_column].notna().to_numpy()
        train_y = train_frame.loc[train_mask, label_column].to_numpy(float) * 10_000.0
        validation_y = (
            validation_frame.loc[validation_mask, label_column].to_numpy(float) * 10_000.0
        )
        for family in MODEL_FAMILIES:
            model, selection, validation_prediction, candidates = (
                fit_validation_selected_model(
                    family,
                    train_x[train_mask], train_y,
                    validation_x[validation_mask], validation_y,
                    seed,
                )
            )
            candidates.insert(0, "target", target)
            model_grid_rows.append(candidates)
            prediction_column = f"{target}__{family}__prediction_bps"
            strategy_frame = validation_frame.loc[validation_mask].copy()
            strategy_frame[prediction_column] = validation_prediction
            strategy_selection, strategy_candidates = select_strategy_on_validation(
                strategy_frame, HORIZON, prediction_column, sell_fee_bps
            )
            strategy_candidates.insert(0, "target", target)
            strategy_candidates.insert(1, "model", family)
            strategy_grid_rows.append(strategy_candidates)
            models[target][family] = model
            frozen[target][family] = {
                "target_column": label_column,
                "model_selection": selection,
                "strategy_selection": strategy_selection,
                "validation_target_metrics": return_metrics(
                    validation_y, validation_prediction
                ),
            }

    pd.concat(model_grid_rows, ignore_index=True).to_csv(
        target_dir / "validation_model_selection.csv", index=False
    )
    pd.concat(strategy_grid_rows, ignore_index=True).to_csv(
        target_dir / "validation_strategy_selection.csv", index=False
    )
    freeze = {
        "status": "frozen_before_test",
        "stocks": stocks,
        "splits": splits,
        "horizon": HORIZON,
        "targets": {
            "raw_return": "stock T+1 return",
            "market_excess_return": "stock T+1 return minus same-minute equal-weight market return",
        },
        "selection": frozen,
        "cost": {"sell_fee_bps": sell_fee_bps, "buy_cost_bps": 0.0},
        "known_limitation": "historical test dates were inspected in earlier project research",
    }
    (target_dir / "selection_frozen_before_test.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    joblib.dump(models, target_dir / "models.joblib")

    print("all target/model/strategy choices frozen; loading test split")
    test_x, test_frame = _prepare_split(stocks, splits["test"], seq_len, minute_dir)
    test_frame = add_market_adjusted_target(test_frame)
    predictions = test_frame.copy()
    metric_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        label_column = _target_column(target)
        mask = test_frame[label_column].notna().to_numpy()
        test_y = test_frame.loc[mask, label_column].to_numpy(float) * 10_000.0
        for family in MODEL_FAMILIES:
            prediction = np.asarray(
                models[target][family].predict(test_x[mask]), dtype=np.float64
            )
            metric_rows.append({
                "target": target,
                "model": family,
                "test_rows": int(mask.sum()),
                "zero_prediction_mae_bps": float(np.mean(np.abs(test_y))),
                **return_metrics(test_y, prediction),
            })
            prediction_column = f"{target}__{family}__prediction_bps"
            predictions[prediction_column] = np.nan
            predictions.loc[mask, prediction_column] = prediction
            strategy_frame = test_frame.loc[mask].copy()
            strategy_frame[prediction_column] = prediction
            chosen = frozen[target][family]["strategy_selection"]
            path = _strategy_path(
                strategy_frame, HORIZON, prediction_column,
                float(chosen["threshold_bps"]), int(chosen["top_k"]),
                int(chosen["interval_minutes"]), sell_fee_bps,
            )
            strategy_rows.append({
                "target": target,
                "model": family,
                "threshold_bps": float(chosen["threshold_bps"]),
                "top_k": int(chosen["top_k"]),
                "interval_minutes": int(chosen["interval_minutes"]),
                "selection_source": "validation_only",
                **strategy_metrics(path, daily_sleeves=True),
            })

    metric_table = pd.DataFrame(metric_rows)
    strategy_table = pd.DataFrame(strategy_rows)
    metric_table.to_csv(
        target_dir / "test_target_metrics.csv", index=False, float_format="%.10f"
    )
    strategy_table.to_csv(
        target_dir / "test_strategy_metrics.csv", index=False, float_format="%.10f"
    )
    predictions.to_csv(
        target_dir / "test_predictions.csv", index=False, float_format="%.10f"
    )
    _plot_comparison(
        metric_table, strategy_table, target_dir / "target_and_strategy_comparison.png"
    )
    report = {
        "status": "completed",
        "selection_frozen_before_test": True,
        "test_rows": int(metric_table["test_rows"].min()),
        "target_metrics": metric_table.to_dict(orient="records"),
        "strategy_metrics": strategy_table.to_dict(orient="records"),
    }
    (target_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report

