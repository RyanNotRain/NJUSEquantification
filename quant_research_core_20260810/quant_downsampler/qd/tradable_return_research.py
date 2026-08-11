"""Validation-frozen research on executable minute and A-share T+1 returns.

This module deliberately remains separate from the official classifier.  It
uses the same causal 60-minute sequence summaries, compares linear and tree
regressors, freezes model/strategy choices on validation data, and only then
loads the test split.  Every result is long/cash and is compared with the same
exposure window of the five-stock market proxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR
from .lstm_baselines import sequence_summary
from .lstm_components import _make_split
from .lstm_magnitude import return_metrics


HORIZONS: dict[str, int | None] = {
    "open_to_close_1m": 1,
    "open_to_close_5m": 5,
    "open_to_close_15m": 15,
    "open_to_close_30m": 30,
    "t1_same_minute_open": None,
}
MODEL_FAMILIES = ("ridge", "hist_gradient_boosting_regressor")


def _date_keys(minute_dir: Path) -> list[str]:
    return sorted(path.stem for path in (minute_dir / "open").glob("*.csv"))


def attach_horizon_returns(
    metadata: pd.DataFrame,
    minute_dir: str | Path,
) -> pd.DataFrame:
    """Attach price-time-valid target returns in the input row order.

    Intraday targets enter at ``open[t+1]`` and leave at the close of the
    1st/5th/15th/30th bar, never crossing lunch.  T+1 enters at ``open[t+1]``
    and exits at the next trading day's open at the same clock minute.  The
    final date of each split is excluded from T+1 so a validation price cannot
    become a training target at the split boundary.
    """
    root = Path(minute_dir)
    frame = metadata.reset_index(drop=True).copy()
    frame["window_end"] = pd.to_datetime(frame["window_end"])
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    outputs = {name: np.full(len(frame), np.nan) for name in HORIZONS}
    keys = _date_keys(root)
    next_key = {keys[index]: keys[index + 1] for index in range(len(keys) - 1)}
    allowed_dates = set(frame["date"].astype(str).str.replace("-", "", regex=False))

    for date, date_positions in frame.groupby("date", sort=True).groups.items():
        key = str(date).replace("-", "")
        open_table = pd.read_csv(root / "open" / f"{key}.csv", index_col=0)
        close_table = pd.read_csv(root / "close" / f"{key}.csv", index_col=0)
        open_table.index = pd.to_datetime(open_table.index)
        close_table.index = pd.to_datetime(close_table.index)
        following_open = None
        following_key = next_key.get(key)
        if following_key in allowed_dates:
            following_open = pd.read_csv(
                root / "open" / f"{following_key}.csv", index_col=0
            )
            following_open.index = pd.to_datetime(following_open.index)

        subset = frame.loc[date_positions]
        for stock, positions in subset.groupby("stock", sort=False).groups.items():
            rows = frame.loc[positions]
            target_locations = open_table.index.get_indexer(rows["target_time"])
            if (target_locations < 0).any():
                raise ValueError(f"cannot align target minute for {stock} on {date}")
            entry = open_table[str(stock)].to_numpy(dtype=np.float64)[target_locations]
            position_array = np.asarray(positions, dtype=int)
            for name, bars in HORIZONS.items():
                if bars is None:
                    if following_open is None:
                        continue
                    exit_times = pd.to_datetime([
                        pd.Timestamp.combine(following_open.index[0].date(), value.time())
                        for value in rows["target_time"]
                    ])
                    exit_locations = following_open.index.get_indexer(exit_times)
                    valid = exit_locations >= 0
                    values = np.full(len(rows), np.nan)
                    if valid.any():
                        exits = following_open[str(stock)].to_numpy(dtype=np.float64)[
                            exit_locations[valid]
                        ]
                        values[valid] = exits / entry[valid] - 1.0
                    outputs[name][position_array] = values
                    continue

                exit_locations = target_locations + int(bars) - 1
                session_end = np.where(target_locations <= 120, 120, 237)
                valid = exit_locations <= session_end
                values = np.full(len(rows), np.nan)
                if valid.any():
                    exits = close_table[str(stock)].to_numpy(dtype=np.float64)[
                        exit_locations[valid]
                    ]
                    values[valid] = exits / entry[valid] - 1.0
                outputs[name][position_array] = values

    for name, values in outputs.items():
        frame[name] = values
    return frame


def scheduled_signal_mask(target_time: pd.Series, interval_minutes: int) -> np.ndarray:
    """Choose deterministic intraday signal times without crossing sessions."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    times = pd.to_datetime(target_time)
    minute = times.dt.hour * 60 + times.dt.minute
    morning_offset = minute - (9 * 60 + 30)
    afternoon_offset = minute - 13 * 60
    offset = np.where(minute <= 11 * 60 + 30, morning_offset, afternoon_offset)
    return (offset >= 60) & ((offset - 60) % interval_minutes == 0)


def _rank_correlation(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(return_metrics(true, predicted)["spearman_ic"])


def _model_candidates(family: str) -> list[dict[str, Any]]:
    if family == "ridge":
        return [{"alpha": value} for value in (1.0, 10.0, 100.0)]
    if family == "hist_gradient_boosting_regressor":
        return [
            {"max_leaf_nodes": leaves, "l2_regularization": regularization}
            for leaves in (15, 31)
            for regularization in (1.0, 5.0)
        ]
    raise ValueError(f"unknown model family {family}")


def _make_model(family: str, parameters: dict[str, Any], seed: int):
    if family == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(parameters["alpha"]))),
        ])
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=80,
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        l2_regularization=float(parameters["l2_regularization"]),
        random_state=seed,
    )


def fit_validation_selected_model(
    family: str,
    train_x: np.ndarray,
    train_y_bps: np.ndarray,
    validation_x: np.ndarray,
    validation_y_bps: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any], np.ndarray, pd.DataFrame]:
    """Select a return regressor on validation Spearman IC, then RMSE."""
    rows: list[dict[str, Any]] = []
    fitted: list[tuple[Any, np.ndarray]] = []
    for parameters in _model_candidates(family):
        model = _make_model(family, parameters, seed)
        model.fit(train_x, train_y_bps)
        prediction = np.asarray(model.predict(validation_x), dtype=np.float64)
        metrics = return_metrics(validation_y_bps, prediction)
        rows.append({"family": family, "parameters": json.dumps(parameters), **metrics})
        fitted.append((model, prediction))
    table = pd.DataFrame(rows)
    order = table.sort_values(
        ["spearman_ic", "rmse_bps"], ascending=[False, True]
    ).index
    best_index = int(order[0])
    selection = table.loc[best_index].to_dict()
    selection["parameters"] = json.loads(str(selection["parameters"]))
    return fitted[best_index][0], selection, fitted[best_index][1], table


def _strategy_path(
    frame: pd.DataFrame,
    return_column: str,
    prediction_column: str,
    threshold_bps: float,
    top_k: int,
    interval_minutes: int,
    sell_fee_bps: float,
) -> pd.DataFrame:
    valid = frame.dropna(subset=[return_column, prediction_column]).copy()
    schedule = scheduled_signal_mask(valid["target_time"], interval_minutes)
    valid = valid.loc[schedule].copy()
    rows: list[dict[str, Any]] = []
    for target_time, group in valid.groupby("target_time", sort=True):
        selected = group[group[prediction_column] >= threshold_bps].nlargest(
            top_k, prediction_column
        )
        active = not selected.empty
        gross = float(selected[return_column].mean()) if active else 0.0
        market = float(group[return_column].mean()) if active else 0.0
        rows.append({
            "date": str(group["date"].iloc[0]),
            "target_time": pd.Timestamp(target_time),
            "gross_return": gross,
            "net_return": gross - (sell_fee_bps / 10_000.0 if active else 0.0),
            "matched_market_return": market,
            "active": int(active),
            "selected_names": int(len(selected)),
        })
    return pd.DataFrame(rows)


def strategy_metrics(path: pd.DataFrame, daily_sleeves: bool) -> dict[str, float | int]:
    """Evaluate sequential intraday trades or equal-capital T+1 sleeves."""
    if path.empty:
        return {
            "periods": 0, "active_periods": 0, "coverage": 0.0,
            "gross_total_return": 0.0, "net_total_return": 0.0,
            "matched_market_total_return": 0.0,
            "cumulative_return_gap_vs_matched_market": 0.0,
            "excess_vs_matched_market": 0.0, "mean_selected_names": 0.0,
        }
    evaluated = path.copy()
    if daily_sleeves:
        evaluated = evaluated.groupby("date", sort=True)[
            ["gross_return", "net_return", "matched_market_return", "active", "selected_names"]
        ].mean()
    gross_growth = float((1.0 + evaluated["gross_return"]).prod())
    net_growth = float((1.0 + evaluated["net_return"]).prod())
    market_growth = float((1.0 + evaluated["matched_market_return"]).prod())
    return {
        "periods": int(len(evaluated)),
        "active_periods": int(path["active"].sum()),
        "coverage": float(path["active"].mean()),
        "gross_total_return": gross_growth - 1.0,
        "net_total_return": net_growth - 1.0,
        "matched_market_total_return": market_growth - 1.0,
        "cumulative_return_gap_vs_matched_market": net_growth - market_growth,
        "excess_vs_matched_market": net_growth / market_growth - 1.0,
        "mean_selected_names": float(path.loc[path["active"].eq(1), "selected_names"].mean())
        if path["active"].any() else 0.0,
    }


def _interval_candidates(horizon: str) -> tuple[int, ...]:
    return {
        "open_to_close_1m": (1, 5, 15),
        "open_to_close_5m": (5, 15, 30),
        "open_to_close_15m": (15, 30, 60),
        "open_to_close_30m": (30, 60),
        "t1_same_minute_open": (15, 30, 60),
    }[horizon]


def select_strategy_on_validation(
    frame: pd.DataFrame,
    horizon: str,
    prediction_column: str,
    sell_fee_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predicted = frame[prediction_column].dropna().to_numpy(dtype=np.float64)
    positive_quantiles = [float(np.quantile(predicted, q)) for q in (0.50, 0.70, 0.80, 0.90, 0.95)]
    # Regression shrinkage means a useful ranking score need not be calibrated
    # one-for-one in basis points.  Keep the explicit cost floor as a candidate,
    # but let validation net/excess returns decide whether a lower score cutoff
    # is worth its extra turnover.
    thresholds = sorted(set(
        [0.0, float(sell_fee_bps)]
        + [max(0.0, value) for value in positive_quantiles]
    ))
    rows: list[dict[str, Any]] = []
    for interval in _interval_candidates(horizon):
        for top_k in (1, 2):
            for threshold in thresholds:
                path = _strategy_path(
                    frame, horizon, prediction_column, threshold, top_k,
                    interval, sell_fee_bps,
                )
                metrics = strategy_metrics(path, horizon == "t1_same_minute_open")
                rows.append({
                    "horizon": horizon,
                    "prediction_column": prediction_column,
                    "threshold_bps": threshold,
                    "top_k": top_k,
                    "interval_minutes": interval,
                    **metrics,
                })
    table = pd.DataFrame(rows)
    eligible = table[(table["active_periods"] >= 10) & (table["coverage"] >= 0.05)]
    if eligible.empty:
        eligible = table[table["active_periods"] > 0]
    if eligible.empty:
        # A model that predicts no positive return is a valid no-trade outcome.
        return table.iloc[0].to_dict(), table
    best = eligible.sort_values(
        ["excess_vs_matched_market", "net_total_return", "coverage"],
        ascending=[False, False, True],
    ).iloc[0]
    return best.to_dict(), table


def _prepare_split(
    stocks: list[str],
    date_range: tuple[str, str],
    seq_len: int,
    minute_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    x, _, stock_ids, _, metadata = _make_split(
        stocks, date_range, seq_len, minute_dir, "enhanced", "three_class", True
    )
    summary = sequence_summary(x, stock_ids, len(stocks))
    del x
    returns = attach_horizon_returns(metadata, minute_dir)
    return summary, returns


def plot_research_summary(
    metric_table: pd.DataFrame,
    strategy_table: pd.DataFrame,
    path: str | Path,
) -> None:
    """Plot test return-ranking IC and validation-frozen strategy excess."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "open_to_close_1m": "1m",
        "open_to_close_5m": "5m",
        "open_to_close_15m": "15m",
        "open_to_close_30m": "30m",
        "t1_same_minute_open": "T+1",
    }
    models = list(MODEL_FAMILIES)
    colors = ["#2563EB", "#D97706"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(labels))
    width = 0.36
    for index, (model, color) in enumerate(zip(models, colors)):
        metric = metric_table[metric_table["model"].eq(model)].set_index("horizon")
        strategy = strategy_table[strategy_table["model"].eq(model)].set_index("horizon")
        offset = (index - 0.5) * width
        axes[0].bar(
            x + offset,
            [metric.at[horizon, "spearman_ic"] for horizon in labels],
            width, label=model.replace("_regressor", ""), color=color,
        )
        axes[1].bar(
            x + offset,
            [100.0 * strategy.at[horizon, "excess_vs_matched_market"] for horizon in labels],
            width, label=model.replace("_regressor", ""), color=color,
        )
    axes[0].set_title("Test return rank IC")
    axes[0].set_ylabel("Spearman IC")
    axes[1].set_title("Validation-frozen strategy excess")
    axes[1].set_ylabel("Geometric excess vs matched market (%)")
    for axis in axes:
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, list(labels.values()))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(Path(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_tradable_return_research(
    output_dir: str | Path = OUTPUT_DIR,
    sell_fee_bps: float = 5.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit all horizons on train/validation, freeze, then evaluate test once."""
    root = Path(output_dir)
    target = root / "tradable_return_research"
    target.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(root / "lstm_ensemble" / "model.pt", map_location="cpu", weights_only=False)
    config = bundle["config"]
    stocks = list(config["stock_codes"])
    splits = {name: tuple(value) for name, value in config["splits"].items()}
    seq_len = int(config["seq_len"])
    minute_dir = root / "minute"

    print("loading train/validation executable-return targets; test remains unopened")
    train_x, train_returns = _prepare_split(stocks, splits["train"], seq_len, minute_dir)
    val_x, val_returns = _prepare_split(stocks, splits["val"], seq_len, minute_dir)
    models: dict[str, dict[str, Any]] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    model_selection_rows: list[pd.DataFrame] = []
    strategy_selection_rows: list[pd.DataFrame] = []
    frozen: dict[str, Any] = {}

    for horizon in HORIZONS:
        train_mask = train_returns[horizon].notna().to_numpy()
        val_mask = val_returns[horizon].notna().to_numpy()
        train_y = train_returns.loc[train_mask, horizon].to_numpy(float) * 10_000.0
        val_y = val_returns.loc[val_mask, horizon].to_numpy(float) * 10_000.0
        models[horizon] = {}
        frozen[horizon] = {}
        for family in MODEL_FAMILIES:
            model, model_selection, val_prediction, candidates = fit_validation_selected_model(
                family, train_x[train_mask], train_y, val_x[val_mask], val_y, seed
            )
            candidates.insert(0, "horizon", horizon)
            model_selection_rows.append(candidates)
            models[horizon][family] = model
            prediction_column = f"{family}__prediction_bps"
            validation_predictions[f"{horizon}::{family}"] = val_prediction
            strategy_frame = val_returns.loc[val_mask].copy()
            strategy_frame[prediction_column] = val_prediction
            strategy_selection, strategy_candidates = select_strategy_on_validation(
                strategy_frame, horizon, prediction_column, sell_fee_bps
            )
            strategy_candidates.insert(0, "model", family)
            strategy_selection_rows.append(strategy_candidates)
            frozen[horizon][family] = {
                "model_selection": model_selection,
                "strategy_selection": strategy_selection,
                "validation_return_metrics": return_metrics(val_y, val_prediction),
            }

    selection_table = pd.concat(model_selection_rows, ignore_index=True)
    strategy_grid = pd.concat(strategy_selection_rows, ignore_index=True)
    selection_table.to_csv(target / "validation_model_selection.csv", index=False)
    strategy_grid.to_csv(target / "validation_strategy_selection.csv", index=False)
    freeze = {
        "status": "frozen_before_test",
        "splits": splits,
        "horizons": {
            "open_to_close_1m": "enter open[t+1], exit close[t+1]",
            "open_to_close_5m": "enter open[t+1], exit fifth bar close; no lunch crossing",
            "open_to_close_15m": "enter open[t+1], exit fifteenth bar close; no lunch crossing",
            "open_to_close_30m": "enter open[t+1], exit thirtieth bar close; no lunch crossing",
            "t1_same_minute_open": "enter open[t+1], exit next trading day same-minute open",
        },
        "selection": frozen,
        "cost": {"sell_fee_bps": sell_fee_bps, "buy_cost_bps": 0.0},
        "strategy_objective": "validation geometric excess vs exposure-matched five-stock market, then net return",
        "known_limitation": "the historical test dates have been inspected in earlier project research",
    }
    (target / "selection_frozen_before_test.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    joblib.dump(models, target / "models.joblib")

    print("all model and strategy choices frozen; loading test split")
    test_x, test_returns = _prepare_split(stocks, splits["test"], seq_len, minute_dir)
    prediction_export = test_returns.copy()
    metric_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        mask = test_returns[horizon].notna().to_numpy()
        test_y = test_returns.loc[mask, horizon].to_numpy(float) * 10_000.0
        for family in MODEL_FAMILIES:
            prediction = np.asarray(models[horizon][family].predict(test_x[mask]), dtype=np.float64)
            metrics = return_metrics(test_y, prediction)
            metric_rows.append({
                "horizon": horizon, "model": family, "test_rows": int(mask.sum()),
                "zero_prediction_mae_bps": float(np.mean(np.abs(test_y))), **metrics,
            })
            column = f"{horizon}__{family}__prediction_bps"
            prediction_export[column] = np.nan
            prediction_export.loc[mask, column] = prediction
            strategy_frame = test_returns.loc[mask].copy()
            strategy_frame[column] = prediction
            chosen = frozen[horizon][family]["strategy_selection"]
            path = _strategy_path(
                strategy_frame, horizon, column,
                float(chosen["threshold_bps"]), int(chosen["top_k"]),
                int(chosen["interval_minutes"]), sell_fee_bps,
            )
            strategy_rows.append({
                "horizon": horizon,
                "model": family,
                "threshold_bps": float(chosen["threshold_bps"]),
                "top_k": int(chosen["top_k"]),
                "interval_minutes": int(chosen["interval_minutes"]),
                "selection_source": "validation_only",
                **strategy_metrics(path, horizon == "t1_same_minute_open"),
            })
    metric_table = pd.DataFrame(metric_rows)
    strategy_table = pd.DataFrame(strategy_rows)
    metric_table.to_csv(target / "test_return_metrics.csv", index=False, float_format="%.10f")
    strategy_table.to_csv(target / "test_strategy_metrics.csv", index=False, float_format="%.10f")
    prediction_export.to_csv(target / "test_predictions.csv", index=False, float_format="%.10f")
    plot_research_summary(metric_table, strategy_table, target / "horizon_strategy_summary.png")
    recommendation = strategy_grid.sort_values(
        ["excess_vs_matched_market", "net_total_return"], ascending=False
    ).iloc[0]
    report = {
        "status": "completed",
        "selection_frozen_before_test": True,
        "test_model_rows": int(len(metric_table)),
        "test_strategy_rows": int(len(strategy_table)),
        "validation_recommended_configuration": recommendation.to_dict(),
        "test_return_metrics": metric_table.to_dict(orient="records"),
        "test_strategy_metrics": strategy_table.to_dict(orient="records"),
    }
    (target / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report
