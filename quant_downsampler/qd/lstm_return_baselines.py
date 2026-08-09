"""Leakage-safe traditional regressors for next-minute signed returns.

Ridge and HistGradientBoostingRegressor consume the same five-stock,
60-minute enhanced windows as :mod:`qd.lstm_return`.  Sequence summaries are
causal fixed transformations; imputers/scalers and estimators fit on train
only, hyperparameters and opening thresholds use validation only, and the
test split is first loaded after ``selection_frozen_before_test.json`` exists.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR
from .lstm_baselines import summarize_sequences
from .lstm_model import DEFAULT_SPLITS, feature_names
from .lstm_return import (
    _load_raw_split,
    _resolve_stocks,
    _subsample_indices,
    _validate_splits,
    grouped_return_report,
)
from .lstm_strategy import (
    break_even_cost_bps,
    build_portfolio_path,
    cost_sensitivity,
    strategy_statistics,
)
from .lstm_strategy_comparison import generate_signal_weights


MODEL_VERSION = 1
MODEL_NAMES = ("ridge", "hist_gradient_boosting_regressor")
DEFAULT_RIDGE_GRID: tuple[dict[str, Any], ...] = (
    {"alpha": 1.0},
    {"alpha": 10.0},
    {"alpha": 100.0},
)
DEFAULT_TREE_GRID: tuple[dict[str, Any], ...] = (
    {
        "learning_rate": 0.08,
        "max_iter": 60,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 60,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
)
KNOWN_OUTPUT_FILES = (
    "models.joblib",
    "validation_selection.csv",
    "validation_threshold_selection.csv",
    "validation_predictions.csv",
    "selection_frozen_before_test.json",
    "test_predictions.csv",
    "ridge_test_predictions.csv",
    "hist_gradient_boosting_regressor_test_predictions.csv",
    "strategy_comparison.csv",
    "strategy_cost_sensitivity.csv",
    "test_metrics.json",
    "replay_audit.json",
    "model_manifest.json",
    "return_baseline_comparison.png",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_return_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    seed: int = 42,
) -> Pipeline:
    """Create a train-fitted preprocessing and signed-return regressor."""

    params = dict(parameters)
    if model_name == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=float(params.get("alpha", 10.0)))),
        ])
    if model_name == "hist_gradient_boosting_regressor":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=float(params.get("learning_rate", 0.08)),
                    max_iter=int(params.get("max_iter", 60)),
                    max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
                    min_samples_leaf=int(params.get("min_samples_leaf", 50)),
                    l2_regularization=float(params.get("l2_regularization", 1.0)),
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ])
    raise ValueError(f"unknown return-regression model: {model_name}")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = pd.Series(np.asarray(left, dtype=np.float64)).rank(method="average")
    y = pd.Series(np.asarray(right, dtype=np.float64)).rank(method="average")
    return _pearson(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))


def return_prediction_metrics(
    realised_return_bps: np.ndarray,
    expected_return_bps: np.ndarray,
    *,
    n_groups: int = 10,
) -> dict[str, Any]:
    """Regression, rank, direction and quantile diagnostics."""

    realised = np.asarray(realised_return_bps, dtype=np.float64)
    expected = np.asarray(expected_return_bps, dtype=np.float64)
    if (
        realised.ndim != 1
        or expected.shape != realised.shape
        or not len(realised)
        or not np.isfinite(realised).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("realised and expected returns must be aligned finite vectors")
    error = expected - realised
    zero_mae = float(np.mean(np.abs(realised)))
    zero_rmse = float(np.sqrt(np.mean(realised**2)))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    realised_nonzero = realised != 0.0
    predicted_nonzero = expected != 0.0
    nonzero_mask = realised_nonzero & predicted_nonzero
    direction_all = np.sign(expected) == np.sign(realised)
    direction_nonzero = direction_all[nonzero_mask]
    return {
        "n_samples": int(len(realised)),
        "mae_bps": mae,
        "rmse_bps": rmse,
        "zero_return_prediction_mae_bps": zero_mae,
        "zero_return_prediction_rmse_bps": zero_rmse,
        "mae_improvement_vs_zero_bps": float(zero_mae - mae),
        "rmse_improvement_vs_zero_bps": float(zero_rmse - rmse),
        "outperforms_zero_prediction_mae": bool(mae < zero_mae),
        "outperforms_zero_prediction_rmse": bool(rmse < zero_rmse),
        "pearson": _pearson(expected, realised),
        "spearman": _spearman(expected, realised),
        "direction_accuracy_including_flat": float(direction_all.mean()),
        "direction_hit_rate_both_nonzero": (
            float(direction_nonzero.mean()) if len(direction_nonzero) else 0.0
        ),
        "direction_both_nonzero_n": int(nonzero_mask.sum()),
        "predicted_positive_rate": float((expected > 0.0).mean()),
        "predicted_negative_rate": float((expected < 0.0).mean()),
        "predicted_zero_rate": float((expected == 0.0).mean()),
        "realised_mean_bps": float(realised.mean()),
        "prediction_mean_bps": float(expected.mean()),
        "grouped_returns": grouped_return_report(expected, realised, n_groups=n_groups),
    }


def select_regressor_on_validation(
    model_name: str,
    parameter_grid: Sequence[Mapping[str, Any]],
    train_matrix: np.ndarray,
    train_returns_bps: np.ndarray,
    validation_matrix: np.ndarray,
    validation_returns_bps: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any], np.ndarray]:
    """Select lowest validation RMSE, then MAE, then higher Spearman."""

    if not parameter_grid:
        raise ValueError(f"empty parameter grid for {model_name}")
    train_x = np.asarray(train_matrix)
    train_y = np.asarray(train_returns_bps, dtype=np.float64)
    validation_x = np.asarray(validation_matrix)
    validation_y = np.asarray(validation_returns_bps, dtype=np.float64)
    if len(train_x) != len(train_y) or len(validation_x) != len(validation_y):
        raise ValueError("feature matrices and return targets are not aligned")
    if not len(train_y) or not len(validation_y):
        raise ValueError("train and validation splits must be non-empty")

    records: list[dict[str, Any]] = []
    best_model: Pipeline | None = None
    best_prediction: np.ndarray | None = None
    best_key: tuple[float, float, float] | None = None
    best_id = -1
    search_started = time.perf_counter()
    for candidate_id, parameters in enumerate(parameter_grid):
        estimator = make_return_estimator(model_name, parameters, seed=seed)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(train_x, train_y)
        training_seconds = time.perf_counter() - fit_started
        prediction = np.asarray(estimator.predict(validation_x), dtype=np.float64)
        metrics = return_prediction_metrics(validation_y, prediction)
        record = {
            "model": model_name,
            "candidate_id": int(candidate_id),
            "parameters": dict(parameters),
            "validation_metrics": metrics,
            "training_seconds": float(training_seconds),
            "warnings": [str(item.message) for item in caught],
        }
        records.append(record)
        key = (
            -float(metrics["rmse_bps"]),
            -float(metrics["mae_bps"]),
            float(metrics["spearman"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_model = estimator
            best_prediction = prediction
            best_id = candidate_id
    if best_model is None or best_prediction is None:
        raise RuntimeError(f"validation search failed for {model_name}")
    selection = {
        "model": model_name,
        "selection_objective": "lowest_validation_rmse_then_mae_then_higher_spearman",
        "selected_candidate_id": int(best_id),
        "selected_parameters": dict(parameter_grid[best_id]),
        "selected_validation_metrics": records[best_id]["validation_metrics"],
        "selected_training_seconds": records[best_id]["training_seconds"],
        "search_training_seconds": float(time.perf_counter() - search_started),
        "candidates": records,
    }
    return best_model, selection, best_prediction


def _load_regression_split(
    stocks: list[str],
    date_range: tuple[str, str],
    seq_len: int,
    minute_dir: Path,
) -> dict[str, Any]:
    raw = _load_raw_split(stocks, date_range, seq_len, minute_dir)
    matrix, names = summarize_sequences(
        raw["x"],
        base_feature_names=feature_names("enhanced"),
        stock_ids=raw["stock_ids"],
        stock_codes=stocks,
    )
    del raw["x"]
    gc.collect()
    raw["matrix"] = matrix
    raw["feature_names"] = names
    return raw


def _prediction_frame(
    metadata: pd.DataFrame,
    realised_return_bps: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    frame = metadata.copy().reset_index(drop=True)
    realised = np.asarray(realised_return_bps, dtype=np.float64)
    if len(frame) != len(realised):
        raise ValueError("metadata and return targets are not aligned")
    frame["realised_return_bps"] = realised
    frame["realised_return"] = realised / 10_000.0
    for name, values in predictions.items():
        expected = np.asarray(values, dtype=np.float64)
        if expected.shape != realised.shape or not np.isfinite(expected).all():
            raise ValueError(f"{name} predictions are not aligned and finite")
        frame[f"{name}_expected_return_bps"] = expected
    return frame


def select_validation_opening_thresholds(
    validation_frame: pd.DataFrame,
    prediction_columns: Mapping[str, str],
    *,
    base_cost_bps: float,
    quantiles: Sequence[float] = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Choose per-model/per-side thresholds using validation economics only."""

    if not np.isfinite(base_cost_bps) or base_cost_bps < 0.0:
        raise ValueError("base_cost_bps must be finite and non-negative")
    if not quantiles or any(not 0.0 <= float(value) < 1.0 for value in quantiles):
        raise ValueError("threshold quantiles must lie within [0, 1)")
    rows: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, float]] = {}
    for model_name, column in prediction_columns.items():
        if column not in validation_frame:
            raise ValueError(f"validation frame is missing {column}")
        prediction = pd.to_numeric(validation_frame[column], errors="coerce")
        if not np.isfinite(prediction.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{column} must be finite")
        candidate_thresholds = np.unique(
            np.maximum(
                float(base_cost_bps),
                np.quantile(
                    np.abs(prediction.to_numpy(dtype=np.float64)),
                    np.asarray(quantiles, dtype=np.float64),
                ),
            )
        )
        chosen[model_name] = {}
        model_frame = validation_frame.copy()
        model_frame["expected_return_bps"] = prediction
        for side in ("long_short", "long_only"):
            candidates: list[tuple[tuple[float, float, float], float]] = []
            for threshold in candidate_thresholds:
                weighted = generate_signal_weights(
                    model_frame,
                    signal_mode="expected_return_bps",
                    score_threshold=float(threshold),
                    side=side,
                    weighting="confidence",
                    require_directional_argmax=False,
                )
                path = build_portfolio_path(weighted)
                statistics = strategy_statistics(path, cost_bps=base_cost_bps)
                row = {
                    "model": model_name,
                    "side": side,
                    "threshold_bps": float(threshold),
                    "active_signal_rate": float(weighted["active_signal"].mean()),
                    "gross_total_return": float(statistics["gross_total_return"]),
                    "net_total_return": float(statistics["net_total_return"]),
                    "average_daily_turnover": float(
                        statistics["average_daily_turnover"]
                    ),
                }
                rows.append(row)
                key = (
                    row["net_total_return"],
                    -row["average_daily_turnover"],
                    row["threshold_bps"],
                )
                candidates.append((key, float(threshold)))
            chosen[model_name][side] = max(candidates, key=lambda item: item[0])[1]
    return chosen, pd.DataFrame(rows)


def _prepare_output_dir(target: Path, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty return-baseline directory: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    if overwrite:
        # Clean only files owned by this module.  An unknown user file is
        # deliberately preserved rather than recursively deleting the target.
        for filename in KNOWN_OUTPUT_FILES:
            path = target / filename
            if path.is_file():
                path.unlink()


def _selection_table(selections: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, selection in selections.items():
        for candidate in selection["candidates"]:
            metrics = candidate["validation_metrics"]
            rows.append({
                "model": model_name,
                "candidate_id": candidate["candidate_id"],
                "selected": candidate["candidate_id"]
                == selection["selected_candidate_id"],
                "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                "validation_mae_bps": metrics["mae_bps"],
                "validation_rmse_bps": metrics["rmse_bps"],
                "validation_pearson": metrics["pearson"],
                "validation_spearman": metrics["spearman"],
                "training_seconds": candidate["training_seconds"],
                "warnings": " | ".join(candidate["warnings"]),
            })
    return pd.DataFrame(rows)


def load_return_baseline_bundle(path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(path))
    if set(bundle) != {"models", "config"}:
        raise ValueError("return-baseline bundle must contain models and config")
    config = bundle["config"]
    required = {
        "model_version",
        "stock_codes",
        "splits",
        "seq_len",
        "feature_set",
        "feature_names",
        "selected_parameters",
        "opening_threshold_bps",
    }
    if not required.issubset(config) or config["model_version"] != MODEL_VERSION:
        raise ValueError("return-baseline config is incomplete or incompatible")
    if set(bundle["models"]) != set(MODEL_NAMES):
        raise ValueError("return-baseline bundle has an unexpected model set")
    return bundle


def evaluate_saved_return_baselines(
    model_path: str | Path,
    data_dir: str | Path,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Reload a saved bundle and deterministically recreate val/test output."""

    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    bundle = load_return_baseline_bundle(model_path)
    config = bundle["config"]
    raw = _load_regression_split(
        list(config["stock_codes"]),
        tuple(config["splits"][split]),
        int(config["seq_len"]),
        Path(data_dir),
    )
    if list(raw["feature_names"]) != list(config["feature_names"]):
        raise RuntimeError("saved and recreated summary features differ")
    predictions = {
        name: np.asarray(model.predict(raw["matrix"]), dtype=np.float64)
        for name, model in bundle["models"].items()
    }
    metrics = {
        name: return_prediction_metrics(raw["signed_return_bps"], values)
        for name, values in predictions.items()
    }
    return {
        "predictions": _prediction_frame(
            raw["metadata"], raw["signed_return_bps"], predictions
        ),
        "metrics": metrics,
        "coverage": raw["coverage"],
    }


def _write_plot(
    target: Path,
    metrics: Mapping[str, Mapping[str, Any]],
    strategies: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(MODEL_NAMES)
    display_names = ["Ridge", "HistGB regressor"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(names))
    axes[0].bar(x - 0.18, [metrics[n]["mae_bps"] for n in names], 0.36, label="MAE")
    axes[0].bar(x + 0.18, [metrics[n]["rmse_bps"] for n in names], 0.36, label="RMSE")
    axes[0].set_xticks(x, display_names)
    axes[0].set_ylabel("Basis points")
    axes[0].set_title("Test return-regression error")
    axes[0].legend(frameon=False)
    long_short = strategies[strategies["side"].eq("long_short")]
    axes[1].bar(
        np.arange(len(long_short)),
        long_short["net_total_return"] * 100.0,
        color="#4c78a8",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.7)
    axes[1].set_xticks(
        np.arange(len(long_short)), display_names
    )
    axes[1].set_ylabel("Test-period net return (%)")
    axes[1].set_title("Validation-threshold strategy at 5 bps")
    figure.tight_layout()
    figure.savefig(target / "return_baseline_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_return_baselines(
    *,
    stock_codes: Sequence[str] | None = None,
    n_stocks: int = 5,
    seq_len: int = 60,
    splits: Mapping[str, tuple[str, str]] | None = None,
    data_dir: str | Path = OUTPUT_DIR / "minute",
    out_dir: str | Path = OUTPUT_DIR / "lstm_return_baselines",
    ridge_grid: Sequence[Mapping[str, Any]] = DEFAULT_RIDGE_GRID,
    tree_grid: Sequence[Mapping[str, Any]] = DEFAULT_TREE_GRID,
    threshold_quantiles: Sequence[float] = (
        0.0,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
    ),
    base_cost_bps: float = 5.0,
    cost_grid_bps: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    max_train_samples: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train, validation-freeze, test once and persist return regressors."""

    if seq_len < 10:
        raise ValueError("seq_len must be at least 10")
    if not np.isfinite(base_cost_bps) or base_cost_bps < 0.0:
        raise ValueError("base_cost_bps must be finite and non-negative")
    if (
        not cost_grid_bps
        or any(not np.isfinite(value) or value < 0.0 for value in cost_grid_bps)
    ):
        raise ValueError("cost_grid_bps must be finite, non-negative and non-empty")
    ranges = dict(splits or DEFAULT_SPLITS)
    _validate_splits(ranges)
    minute_dir = Path(data_dir)
    target = Path(out_dir)
    _prepare_output_dir(target, overwrite)
    stocks = _resolve_stocks(stock_codes, n_stocks, minute_dir)

    # Test dates are deliberately absent above and below this comment until
    # the frozen selection file has been written.
    train = _load_regression_split(stocks, ranges["train"], seq_len, minute_dir)
    validation = _load_regression_split(stocks, ranges["val"], seq_len, minute_dir)
    if train["feature_names"] != validation["feature_names"]:
        raise RuntimeError("train and validation summary feature names differ")
    train_indices = _subsample_indices(
        train["labels"], train["stock_ids"], max_train_samples, seed
    )
    train_matrix = train["matrix"][train_indices]
    train_returns = train["signed_return_bps"][train_indices]

    models: dict[str, Pipeline] = {}
    selections: dict[str, dict[str, Any]] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for model_name, grid in (
        ("ridge", ridge_grid),
        ("hist_gradient_boosting_regressor", tree_grid),
    ):
        model, selection, prediction = select_regressor_on_validation(
            model_name,
            grid,
            train_matrix,
            train_returns,
            validation["matrix"],
            validation["signed_return_bps"],
            seed=seed,
        )
        models[model_name] = model
        selections[model_name] = selection
        validation_predictions[model_name] = prediction

    validation_frame = _prediction_frame(
        validation["metadata"],
        validation["signed_return_bps"],
        validation_predictions,
    )
    prediction_columns = {
        name: f"{name}_expected_return_bps" for name in MODEL_NAMES
    }
    thresholds, threshold_table = select_validation_opening_thresholds(
        validation_frame,
        prediction_columns,
        base_cost_bps=base_cost_bps,
        quantiles=threshold_quantiles,
    )
    config: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "stock_codes": stocks,
        "splits": ranges,
        "seq_len": int(seq_len),
        "feature_set": "enhanced",
        "feature_names": list(train["feature_names"]),
        "summary_definition": (
            "last/full-window/last-10/last-5 means and standard deviations "
            "plus stock one-hot"
        ),
        "selected_parameters": {
            name: selection["selected_parameters"]
            for name, selection in selections.items()
        },
        "opening_threshold_bps": thresholds,
        "base_cost_bps": float(base_cost_bps),
        "preprocessing_fit_scope": "training_split_only",
        "hyperparameter_selection_scope": "validation_split_only",
        "threshold_selection_scope": "validation_split_only",
        "train_rows_available": int(len(train["matrix"])),
        "train_rows_used": int(len(train_indices)),
        "validation_rows": int(len(validation["matrix"])),
    }
    model_path = target / "models.joblib"
    joblib.dump({"models": models, "config": config}, model_path)
    _selection_table(selections).to_csv(
        target / "validation_selection.csv", index=False, float_format="%.10f"
    )
    threshold_table.to_csv(
        target / "validation_threshold_selection.csv",
        index=False,
        float_format="%.10f",
    )
    validation_frame.to_csv(
        target / "validation_predictions.csv", index=False, float_format="%.10f"
    )
    frozen_at = _utc_now()
    freeze_record = {
        "frozen_before_test_load": True,
        "frozen_at_utc": frozen_at,
        "selected_parameters": config["selected_parameters"],
        "opening_threshold_bps": thresholds,
        "minimum_threshold_bps": float(base_cost_bps),
        "all_thresholds_at_least_base_cost": bool(
            all(
                value >= base_cost_bps
                for by_side in thresholds.values()
                for value in by_side.values()
            )
        ),
        "test_evaluation_count": 0,
    }
    freeze_path = target / "selection_frozen_before_test.json"
    _json_dump(freeze_path, freeze_record)

    # First and only primary test load/evaluation occurs after the freeze file.
    test_loaded_at = _utc_now()
    test = _load_regression_split(stocks, ranges["test"], seq_len, minute_dir)
    if train["feature_names"] != test["feature_names"]:
        raise RuntimeError("train and test summary feature names differ")
    test_predictions = {
        name: np.asarray(model.predict(test["matrix"]), dtype=np.float64)
        for name, model in models.items()
    }
    test_metrics = {
        name: return_prediction_metrics(test["signed_return_bps"], prediction)
        for name, prediction in test_predictions.items()
    }
    test_frame = _prediction_frame(
        test["metadata"], test["signed_return_bps"], test_predictions
    )
    test_frame.to_csv(target / "test_predictions.csv", index=False, float_format="%.10f")
    canonical_columns = [
        column
        for column in (
            "stock",
            "stock_id",
            "date",
            "window_end",
            "target_time",
            "realised_return_bps",
            "realised_return",
        )
        if column in test_frame
    ]
    for model_name in MODEL_NAMES:
        canonical = test_frame[canonical_columns].copy()
        canonical["expected_return_bps"] = test_predictions[model_name]
        canonical.to_csv(
            target / f"{model_name}_test_predictions.csv",
            index=False,
            float_format="%.10f",
        )

    # Persistence replay uses the already loaded test matrix.  It therefore
    # verifies joblib serialization without a second raw test-data load.
    reloaded_bundle = load_return_baseline_bundle(model_path)
    replay_predictions = {
        name: np.asarray(model.predict(test["matrix"]), dtype=np.float64)
        for name, model in reloaded_bundle["models"].items()
    }
    replay_differences = {
        name: float(np.max(np.abs(test_predictions[name] - replay_predictions[name])))
        for name in MODEL_NAMES
    }
    replay_maximum_difference = max(replay_differences.values())
    replay_audit = {
        "passed": bool(replay_maximum_difference <= 1e-10),
        "same_loaded_test_matrix_used": True,
        "raw_test_data_load_count": 1,
        "rows": int(len(test["matrix"])),
        "maximum_absolute_difference": replay_maximum_difference,
        "maximum_absolute_difference_by_model": replay_differences,
    }
    _json_dump(target / "replay_audit.json", replay_audit)
    if not replay_audit["passed"]:
        raise RuntimeError("saved return-regression replay changed test predictions")

    strategy_rows: list[dict[str, Any]] = []
    sensitivity_parts: list[pd.DataFrame] = []
    for model_name, prediction in test_predictions.items():
        model_frame = test_frame.copy()
        model_frame["expected_return_bps"] = prediction
        for side in ("long_short", "long_only"):
            threshold = float(thresholds[model_name][side])
            weighted = generate_signal_weights(
                model_frame,
                signal_mode="expected_return_bps",
                score_threshold=threshold,
                side=side,
                weighting="confidence",
                require_directional_argmax=False,
            )
            path = build_portfolio_path(weighted)
            statistics = strategy_statistics(path, cost_bps=base_cost_bps)
            active_intervals = weighted.groupby("_window_end_ts")["active_signal"].any()
            row = {
                "model": model_name,
                "comparison_signal": "expected_return_bps",
                "side": side,
                "validation_frozen_threshold_bps": threshold,
                "row_coverage": float(weighted["active_signal"].mean()),
                "interval_coverage": float(active_intervals.mean()),
                "break_even_cost_bps": break_even_cost_bps(path),
                **statistics,
            }
            strategy_rows.append(row)
            curve = cost_sensitivity(path, cost_grid_bps)
            curve.insert(0, "model", model_name)
            curve.insert(1, "side", side)
            curve.insert(2, "validation_frozen_threshold_bps", threshold)
            sensitivity_parts.append(curve)
    strategy_table = pd.DataFrame(strategy_rows)
    sensitivity_table = pd.concat(sensitivity_parts, ignore_index=True)
    strategy_table.to_csv(
        target / "strategy_comparison.csv", index=False, float_format="%.10f"
    )
    sensitivity_table.to_csv(
        target / "strategy_cost_sensitivity.csv", index=False, float_format="%.10f"
    )

    test_report: dict[str, Any] = {
        "models": test_metrics,
        "zero_return_prediction_baseline": {
            "mae_bps": float(np.mean(np.abs(test["signed_return_bps"]))),
            "rmse_bps": float(
                np.sqrt(np.mean(np.asarray(test["signed_return_bps"]) ** 2))
            ),
        },
        "coverage": test["coverage"],
        "audit": {
            "selection_frozen_at_utc": frozen_at,
            "test_loaded_at_utc": test_loaded_at,
            "test_loaded_after_selection_frozen": test_loaded_at >= frozen_at,
            "test_evaluation_count_per_model": 1,
            "test_used_for_hyperparameter_or_threshold_selection": False,
            "persistence_replay": replay_audit,
        },
        "limitations": [
            "The fixed ten-day test segment has been inspected elsewhere in the project.",
            "The minute strategy assumes zero latency and linear costs without impact or borrow fees.",
        ],
    }
    _json_dump(target / "test_metrics.json", test_report)
    freeze_record["test_evaluation_count"] = 1
    freeze_record["test_evaluated_at_utc"] = _utc_now()
    _json_dump(freeze_path, freeze_record)
    manifest = {
        "kind": "train_fitted_validation_selected_return_regression_baselines",
        "models_sha256": _sha256(model_path),
        "models_path": str(model_path),
        "test_predictions_path": str(target / "test_predictions.csv"),
        "test_metrics_path": str(target / "test_metrics.json"),
    }
    _json_dump(target / "model_manifest.json", manifest)
    _write_plot(target, test_metrics, strategy_table)
    return {
        "out_dir": str(target),
        "test_metrics": test_metrics,
        "strategy_comparison": strategy_table,
        "cost_sensitivity": sensitivity_table,
        "config": config,
    }
