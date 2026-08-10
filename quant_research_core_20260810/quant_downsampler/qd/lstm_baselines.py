"""Causal classical baselines and probability-quality audits for Task 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR
from .lstm_components import _make_split


PROBABILITY_COLUMNS = ("prob_down", "prob_flat", "prob_up")


def sequence_summary(x: np.ndarray, stock_ids: np.ndarray, n_stocks: int) -> np.ndarray:
    """Summarise a fixed past window without using the target minute."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 3 or len(values) != len(stock_ids):
        raise ValueError("sequences and stock_ids must align")
    parts = [
        values[:, -1, :],
        values.mean(axis=1),
        values.std(axis=1),
        values.min(axis=1),
        values.max(axis=1),
        values[:, -1, :] - values[:, 0, :],
        np.eye(n_stocks, dtype=np.float32)[np.asarray(stock_ids, dtype=np.int64)],
    ]
    result = np.concatenate(parts, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("sequence summary contains non-finite values")
    return result


def probability_metrics(labels: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Return accuracy, macro F1, multiclass Brier, NLL, and top-label ECE."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if p.shape != (len(y), 3) or not np.isfinite(p).all():
        raise ValueError("probabilities must be finite N x 3 values")
    p = np.clip(p, 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    prediction = p.argmax(axis=1)
    one_hot = np.eye(3)[y]
    confidence = p.max(axis=1)
    correct = prediction == y
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < n_bins - 1 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "nll": float(log_loss(y, p, labels=[0, 1, 2])),
        "top_label_ece": float(ece),
    }


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.log(np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def select_temperature(labels: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Select a scalar temperature on validation NLL only."""
    rows = []
    for temperature in np.linspace(0.50, 2.50, 81):
        metrics = probability_metrics(labels, apply_temperature(probability, float(temperature)))
        rows.append({"temperature": float(temperature), **metrics})
    table = pd.DataFrame(rows)
    best = table.sort_values(["nll", "brier", "top_label_ece"]).iloc[0]
    return float(best["temperature"]), table


def _sample_weight(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=3).astype(float)
    return len(labels) / (3.0 * counts[labels])


def _prediction_frame(metadata: pd.DataFrame, labels: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    frame = metadata.drop(columns="stock_id", errors="ignore").copy()
    frame["true_label"] = labels
    frame["predicted_label"] = probability.argmax(axis=1)
    for index, column in enumerate(PROBABILITY_COLUMNS):
        frame[column] = probability[:, index]
    frame["confidence"] = probability.max(axis=1)
    return frame


def run_lstm_baselines(
    output_dir: str | Path = OUTPUT_DIR,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Fit classical baselines, audit LSTM calibration, then open test once."""
    root = Path(output_dir)
    source = root / "lstm_ensemble"
    target = root / "lstm_baselines"
    target.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
    config = bundle["config"]
    stocks = list(config["stock_codes"])
    splits = {name: tuple(value) for name, value in config["splits"].items()}
    seq_len = int(config["seq_len"])
    minute_dir = root / "minute"

    print("loading train and validation splits; test remains unopened")
    train = _make_split(stocks, splits["train"], seq_len, minute_dir, "legacy", "three_class", True)
    validation = _make_split(stocks, splits["val"], seq_len, minute_dir, "legacy", "three_class", True)
    train_x, train_y, train_ids, _, _ = train
    val_x, val_y, val_ids, _, val_metadata = validation
    train_summary = sequence_summary(train_x, train_ids, len(stocks))
    val_summary = sequence_summary(val_x, val_ids, len(stocks))
    del train_x, val_x

    scaler = StandardScaler().fit(train_summary)
    train_scaled = scaler.transform(train_summary)
    val_scaled = scaler.transform(val_summary)
    candidates: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}

    for alpha in (1e-4, 1e-3):
        model = SGDClassifier(
            loss="log_loss", alpha=alpha, max_iter=60, tol=1e-3,
            class_weight="balanced", average=True, random_state=random_seed,
        ).fit(train_scaled, train_y)
        probability = model.predict_proba(val_scaled)
        metrics = probability_metrics(val_y, probability)
        model_key = f"linear_logistic_sgd::{alpha}"
        candidates.append({"family": "linear_logistic_sgd", "parameter": alpha, "model_key": model_key, **metrics})
        fitted[model_key] = model

    weights = _sample_weight(train_y)
    for leaves in (15, 31):
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=80, max_leaf_nodes=leaves,
            l2_regularization=1.0, random_state=random_seed,
        ).fit(train_summary, train_y, sample_weight=weights)
        probability = model.predict_proba(val_summary)
        metrics = probability_metrics(val_y, probability)
        model_key = f"hist_gradient_boosting::{leaves}"
        candidates.append({"family": "hist_gradient_boosting", "parameter": leaves, "model_key": model_key, **metrics})
        fitted[model_key] = model

    selection = pd.DataFrame(candidates)
    selected: dict[str, dict[str, Any]] = {}
    for family, group in selection.groupby("family"):
        row = group.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]
        selected[family] = row.to_dict()

    validation_lstm = pd.read_csv(source / "validation_predictions.csv")
    val_lstm_probability = validation_lstm.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if not np.array_equal(validation_lstm["true_label"].to_numpy(np.int64), val_y):
        raise ValueError("saved LSTM validation labels do not align with rebuilt data")
    temperature, temperature_grid = select_temperature(val_y, val_lstm_probability)
    freeze = {
        "status": "frozen_before_test",
        "selected": selected,
        "lstm_temperature": temperature,
        "selection_objective": "validation macro F1, then accuracy; calibration uses validation NLL",
        "splits": splits,
    }
    (target / "selection_frozen_before_test.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    selection.to_csv(target / "validation_selection.csv", index=False)
    temperature_grid.to_csv(target / "temperature_selection.csv", index=False)

    print("selection frozen; loading test split for the first time")
    test = _make_split(stocks, splits["test"], seq_len, minute_dir, "legacy", "three_class", True)
    test_x, test_y, test_ids, _, test_metadata = test
    test_summary = sequence_summary(test_x, test_ids, len(stocks))
    test_scaled = scaler.transform(test_summary)
    del test_x

    model_predictions: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    for family, row in selected.items():
        model = fitted[str(row["model_key"])]
        features = test_scaled if family == "linear_logistic_sgd" else test_summary
        probability = model.predict_proba(features)
        model_predictions[family] = _prediction_frame(test_metadata, test_y, probability)
        metric_rows.append({"model": family, **probability_metrics(test_y, probability)})

    last_return = test_summary[:, 3]
    last_prediction = np.where(last_return < 0, 0, np.where(last_return > 0, 2, 1))
    last_probability = np.full((len(test_y), 3), 1e-6)
    last_probability[np.arange(len(test_y)), last_prediction] = 1.0 - 2e-6
    metric_rows.append({"model": "last_observed_move", **probability_metrics(test_y, last_probability)})

    test_lstm = pd.read_csv(source / "test_predictions.csv")
    test_lstm_probability = test_lstm.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if not np.array_equal(test_lstm["true_label"].to_numpy(np.int64), test_y):
        raise ValueError("saved LSTM test labels do not align with rebuilt data")
    raw_metrics = probability_metrics(test_y, test_lstm_probability)
    calibrated_probability = apply_temperature(test_lstm_probability, temperature)
    calibrated_metrics = probability_metrics(test_y, calibrated_probability)
    metric_rows.extend([
        {"model": "lstm_ensemble_raw", **raw_metrics},
        {"model": "lstm_ensemble_temperature_scaled", **calibrated_metrics},
        {"model": "majority_flat", **probability_metrics(test_y, np.tile([1e-6, 1 - 2e-6, 1e-6], (len(test_y), 1)))},
    ])

    for name, frame in model_predictions.items():
        frame.to_csv(target / f"{name}_test_predictions.csv", index=False, float_format="%.10f")
    calibrated_frame = _prediction_frame(test_metadata, test_y, calibrated_probability)
    calibrated_frame.to_csv(target / "lstm_calibrated_test_predictions.csv", index=False, float_format="%.10f")
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(target / "test_metrics.csv", index=False, float_format="%.10f")
    joblib.dump({
        "scaler": scaler,
        "models": {family: fitted[str(row["model_key"])] for family, row in selected.items()},
        "selection": selected,
        "feature_definition": "last/mean/std/min/max/change of 60-minute legacy features plus stock one-hot",
    }, target / "models.joblib")
    report = {
        "status": "completed",
        "test_rows": int(len(test_y)),
        "selection_frozen_before_test": True,
        "metrics": metrics.set_index("model").to_dict(orient="index"),
        "lstm_temperature": temperature,
    }
    (target / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return report


def export_saved_validation_predictions(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Path]:
    """Replay frozen classical models on validation for threshold selection."""
    root = Path(output_dir)
    source = root / "lstm_ensemble"
    target = root / "lstm_baselines"
    saved = joblib.load(target / "models.joblib")
    bundle = torch.load(source / "model.pt", map_location="cpu", weights_only=False)
    config = bundle["config"]
    stocks = list(config["stock_codes"])
    split = tuple(config["splits"]["val"])
    data = _make_split(
        stocks, split, int(config["seq_len"]), root / "minute",
        "legacy", "three_class", True,
    )
    x, labels, stock_ids, _, metadata = data
    summary = sequence_summary(x, stock_ids, len(stocks))
    scaled = saved["scaler"].transform(summary)
    outputs: dict[str, Path] = {}
    for family, model in saved["models"].items():
        features = scaled if family == "linear_logistic_sgd" else summary
        frame = _prediction_frame(metadata, labels, model.predict_proba(features))
        path = target / f"{family}_validation_predictions.csv"
        frame.to_csv(path, index=False, float_format="%.10f")
        outputs[family] = path
    return outputs
