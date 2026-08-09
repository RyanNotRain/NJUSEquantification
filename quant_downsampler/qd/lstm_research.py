"""Leakage-aware diagnostics for saved minute-LSTM predictions.

This module deliberately separates three different claims:

* ``date_stability_report`` slices one already-frozen model by date.  It is a
  stability diagnostic, not a new walk-forward experiment.
* ``make_walk_forward_splits`` defines a chronological train/validation/test
  protocol for models that will actually be refit in every fold.
* ``aggregate_walk_forward_predictions`` evaluates the test predictions from
  those independently fitted folds and rejects overlapping test samples.

The functions operate on saved probabilities whenever possible, so calibration,
strong baselines, and component ablations do not require another long training
run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FULL_CLASS_NAMES = ("down", "flat", "up")


def _probability_columns(class_names: Sequence[str]) -> list[str]:
    return [f"prob_{name}" for name in class_names]


def validate_prediction_frame(
    predictions: pd.DataFrame | str | Path,
    class_names: Sequence[str] = FULL_CLASS_NAMES,
) -> pd.DataFrame:
    """Load and validate an auditable per-sample probability table.

    The returned frame is sorted chronologically and its ``predicted_label``
    and ``confidence`` columns are recomputed from the probabilities.  A stale
    supplied value is rejected rather than silently overwritten.
    """
    names = tuple(class_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("class_names must contain at least two unique names")
    frame = (
        pd.read_csv(Path(predictions))
        if isinstance(predictions, (str, Path))
        else predictions.copy()
    )
    probability_columns = _probability_columns(names)
    required = {"stock", "date", "true_label", *probability_columns}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction table is empty")

    labels_numeric = pd.to_numeric(frame["true_label"], errors="coerce")
    if labels_numeric.isna().any() or not np.equal(labels_numeric, np.floor(labels_numeric)).all():
        raise ValueError("true_label must contain finite integer class indices")
    labels = labels_numeric.to_numpy(dtype=np.int64)
    if (labels < 0).any() or (labels >= len(names)).any():
        raise ValueError("true_label contains an out-of-range class index")
    frame["true_label"] = labels

    probability = frame[probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(probability).all():
        raise ValueError("probabilities must be finite")
    if ((probability < -1e-8) | (probability > 1.0 + 1e-8)).any():
        raise ValueError("probabilities must be within [0, 1]")
    row_sums = probability.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("probability rows must sum to one")
    # CSV decimal formatting can leave harmless 1e-8 residuals.  Canonicalise
    # once so downstream sklearn versions do not emit spurious sum warnings.
    probability = probability / row_sums[:, None]
    frame.loc[:, probability_columns] = probability

    dates = pd.to_datetime(frame["date"].astype(str), errors="coerce")
    if dates.isna().any():
        raise ValueError("date contains an invalid value")
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    sort_columns = ["date"]
    sample_keys = ["stock", "date"]
    if "target_time" in frame:
        target_time = pd.to_datetime(frame["target_time"], errors="coerce")
        if target_time.isna().any():
            raise ValueError("target_time contains an invalid value")
        if not np.array_equal(
            target_time.dt.strftime("%Y-%m-%d").to_numpy(), frame["date"].to_numpy()
        ):
            raise ValueError("target_time calendar dates do not match date")
        frame["target_time"] = target_time.dt.strftime("%Y-%m-%d %H:%M:%S")
        sort_columns.append("target_time")
        sample_keys = ["stock", "target_time"]
        if "window_end" in frame:
            window_end = pd.to_datetime(frame["window_end"], errors="coerce")
            if window_end.isna().any():
                raise ValueError("window_end contains an invalid value")
            if not (window_end < target_time).all():
                raise ValueError("window_end must be earlier than target_time")
            if not np.array_equal(
                window_end.dt.strftime("%Y-%m-%d").to_numpy(),
                frame["date"].to_numpy(),
            ):
                raise ValueError("window_end calendar dates do not match date")
            frame["window_end"] = window_end.dt.strftime("%Y-%m-%d %H:%M:%S")
    elif "window_end" in frame:
        window_end = pd.to_datetime(frame["window_end"], errors="coerce")
        if window_end.isna().any():
            raise ValueError("window_end contains an invalid value")
        frame["window_end"] = window_end.dt.strftime("%Y-%m-%d %H:%M:%S")
        sort_columns.append("window_end")
        sample_keys = ["stock", "window_end"]
    else:
        # Without a timestamp, more than one row per stock-date is ambiguous.
        if frame.duplicated(sample_keys).any():
            raise ValueError("multiple rows per stock-date require target_time or window_end")
    if frame.duplicated(sample_keys).any():
        raise ValueError(f"duplicate prediction samples for key {sample_keys}")

    predicted = probability.argmax(axis=1).astype(np.int64)
    confidence = probability.max(axis=1)
    if "predicted_label" in frame:
        supplied = pd.to_numeric(frame["predicted_label"], errors="coerce").to_numpy()
        if not np.array_equal(supplied, predicted):
            raise ValueError("saved predicted_label disagrees with saved probabilities")
    if "confidence" in frame:
        supplied_confidence = pd.to_numeric(frame["confidence"], errors="coerce").to_numpy()
        if not np.allclose(supplied_confidence, confidence, atol=1e-6, rtol=1e-6):
            raise ValueError("saved confidence disagrees with saved probabilities")
    frame["predicted_label"] = predicted
    frame["confidence"] = confidence
    return frame.sort_values([*sort_columns, "stock"], kind="stable").reset_index(drop=True)


def multiclass_brier_score(labels: np.ndarray, probability: np.ndarray) -> float:
    """Return mean squared distance to the one-hot outcome (range 0..2)."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or len(y) != len(p) or not len(y):
        raise ValueError("labels and probability must be non-empty and aligned")
    if (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("labels contain an out-of-range class index")
    if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("probability contains invalid values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("probability rows must sum to one")
    p = p / p.sum(axis=1, keepdims=True)
    outcome = np.eye(p.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((p - outcome) ** 2, axis=1)))


def _binary_reliability_bins(
    outcome: np.ndarray,
    forecast: np.ndarray,
    n_bins: int,
) -> list[dict[str, float | int | None]]:
    if n_bins < 2:
        raise ValueError("n_bins must be at least two")
    y = np.asarray(outcome, dtype=np.float64)
    p = np.asarray(forecast, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("outcome and forecast must be aligned non-empty vectors")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid reliability inputs")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    result: list[dict[str, float | int | None]] = []
    total = len(p)
    for bin_id, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = indices == bin_id
        count = int(mask.sum())
        if count:
            mean_forecast = float(p[mask].mean())
            observed_rate = float(y[mask].mean())
            gap = abs(mean_forecast - observed_rate)
        else:
            mean_forecast = observed_rate = gap = None
        result.append({
            "bin": bin_id,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "share": float(count / total),
            "mean_forecast": mean_forecast,
            "observed_rate": observed_rate,
            "absolute_gap": gap,
        })
    return result


def _ece_from_bins(bins: Sequence[Mapping[str, float | int | None]]) -> tuple[float, float]:
    weighted = [
        float(row["share"]) * float(row["absolute_gap"])
        for row in bins
        if row["absolute_gap"] is not None
    ]
    gaps = [
        float(row["absolute_gap"])
        for row in bins
        if row["absolute_gap"] is not None
    ]
    return float(sum(weighted)), float(max(gaps, default=0.0))


def calibration_report(
    labels: np.ndarray,
    probability: np.ndarray,
    class_names: Sequence[str] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute multiclass Brier/NLL and top-label plus classwise calibration."""
    from sklearn.metrics import log_loss

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    # Reuse the strict validation in the Brier implementation.
    brier = multiclass_brier_score(y, p)
    names = tuple(class_names or (str(index) for index in range(p.shape[1])))
    if len(names) != p.shape[1]:
        raise ValueError("class_names does not match probability width")

    prediction = p.argmax(axis=1)
    confidence = p.max(axis=1)
    top_bins = _binary_reliability_bins((prediction == y).astype(float), confidence, n_bins)
    top_ece, top_mce = _ece_from_bins(top_bins)
    classwise: dict[str, dict[str, Any]] = {}
    class_ece: list[float] = []
    for class_id, name in enumerate(names):
        bins = _binary_reliability_bins((y == class_id).astype(float), p[:, class_id], n_bins)
        ece, mce = _ece_from_bins(bins)
        class_ece.append(ece)
        classwise[name] = {"ece": ece, "mce": mce, "bins": bins}
    clipped = np.clip(p, 1e-15, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    return {
        "brier_score": brier,
        "brier_score_half_scaled": float(brier / 2.0),
        "negative_log_likelihood": float(log_loss(y, clipped, labels=list(range(p.shape[1])))),
        "top_label_ece": top_ece,
        "top_label_mce": top_mce,
        "top_label_bins": top_bins,
        "classwise_ece": float(np.mean(class_ece)),
        "classes": classwise,
        "n_bins": int(n_bins),
    }


def probability_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    class_names: Sequence[str] | None = None,
    n_bins: int = 10,
    include_reliability: bool = True,
) -> dict[str, Any]:
    """Return aligned classification and probability-quality metrics."""
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    # Validate before computing hard predictions.
    brier = multiclass_brier_score(y, p)
    prediction = p.argmax(axis=1)
    n_classes = p.shape[1]
    names = tuple(class_names or (str(index) for index in range(n_classes)))
    if len(names) != n_classes:
        raise ValueError("class_names does not match probability width")
    result: dict[str, Any] = {
        "n": int(len(y)),
        "accuracy": float(np.mean(prediction == y)),
        "macro_precision": float(precision_score(
            y, prediction, labels=list(range(n_classes)), average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y, prediction, labels=list(range(n_classes)), average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y, prediction, labels=list(range(n_classes)), average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y, prediction, labels=list(range(n_classes))
        ).tolist(),
        "class_support": np.bincount(y, minlength=n_classes).astype(int).tolist(),
        "brier_score": brier,
    }
    calibration = calibration_report(y, p, names, n_bins=n_bins)
    if include_reliability:
        result["calibration"] = calibration
    else:
        result.update({
            key: calibration[key]
            for key in (
                "negative_log_likelihood", "top_label_ece", "classwise_ece"
            )
        })
    return result


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling when only probabilities were saved."""
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or not len(p):
        raise ValueError("probability must be a non-empty matrix")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(
        p.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("probability matrix is invalid")
    log_probability = np.log(np.clip(p, 1e-15, 1.0)) / float(temperature)
    log_probability -= log_probability.max(axis=1, keepdims=True)
    scaled = np.exp(log_probability)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    validation_labels: np.ndarray,
    validation_probability: np.ndarray,
    bounds: tuple[float, float] = (0.05, 10.0),
) -> float:
    """Fit one temperature on validation NLL; never fit on test labels."""
    from scipy.optimize import minimize_scalar

    y = np.asarray(validation_labels, dtype=np.int64)
    p = np.asarray(validation_probability, dtype=np.float64)
    multiclass_brier_score(y, p)
    low, high = bounds
    if not np.isfinite([low, high]).all() or low <= 0 or low >= high:
        raise ValueError("temperature bounds must satisfy 0 < low < high")

    def objective(log_temperature: float) -> float:
        scaled = temperature_scale(p, float(np.exp(log_temperature)))
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(y)), y], 1e-15, 1.0))))

    result = minimize_scalar(
        objective, bounds=(float(np.log(low)), float(np.log(high))), method="bounded",
        options={"xatol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def temperature_calibration_report(
    validation_labels: np.ndarray,
    validation_probability: np.ndarray,
    test_labels: np.ndarray,
    test_probability: np.ndarray,
    class_names: Sequence[str] = FULL_CLASS_NAMES,
    n_bins: int = 10,
) -> tuple[dict[str, Any], np.ndarray]:
    """Fit on validation and report before/after validation and test metrics."""
    temperature = fit_temperature(validation_labels, validation_probability)
    scaled_validation = temperature_scale(validation_probability, temperature)
    scaled_test = temperature_scale(test_probability, temperature)
    report = {
        "fit_scope": "validation_only",
        "temperature": temperature,
        "validation_before": calibration_report(
            validation_labels, validation_probability, class_names, n_bins
        ),
        "validation_after": calibration_report(
            validation_labels, scaled_validation, class_names, n_bins
        ),
        "test_before": calibration_report(test_labels, test_probability, class_names, n_bins),
        "test_after": calibration_report(test_labels, scaled_test, class_names, n_bins),
    }
    return report, scaled_test


def _hard_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score

    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(prediction, dtype=np.int64)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError("labels and predictions must be aligned non-empty vectors")
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(y == pred)),
        "macro_f1": float(f1_score(
            y, pred, labels=list(range(n_classes)), average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y, pred, labels=list(range(n_classes))
        ).tolist(),
    }


def baseline_report(
    predictions: pd.DataFrame | str | Path,
    training_class_rates: Sequence[float] | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate causal, deployment-time baselines on the full-window sample.

    ``last_move_*`` uses the immediately preceding *realized* target within the
    same stock-day.  Rows without a contiguous saved preceding target (the
    first eligible row of each session) are excluded rather than bridging a
    lunch or warm-up gap.
    """
    frame = validate_prediction_frame(predictions, FULL_CLASS_NAMES)
    labels = frame["true_label"].to_numpy(np.int64)
    n = len(frame)
    report: dict[str, Any] = {
        "flat_constant": {
            **_hard_metrics(labels, np.full(n, 1, dtype=np.int64), 3),
            "coverage": 1.0,
            "rule": "always predict flat; fixed before viewing test labels",
        }
    }
    if training_class_rates is not None:
        rates = np.asarray(training_class_rates, dtype=np.float64)
        if rates.shape != (3,) or not np.isfinite(rates).all() or (rates < 0).any() or rates.sum() <= 0:
            raise ValueError("training_class_rates must contain three non-negative rates")
        rates = rates / rates.sum()
        prior_probability = np.broadcast_to(rates, (n, 3)).copy()
        report["training_prior"] = {
            **probability_metrics(
                labels, prior_probability, FULL_CLASS_NAMES, n_bins,
                include_reliability=False,
            ),
            "coverage": 1.0,
            "rule": "constant class distribution estimated on the training split only",
            "class_rates": rates.tolist(),
        }
        report["training_majority"] = {
            **_hard_metrics(labels, np.full(n, int(rates.argmax()), dtype=np.int64), 3),
            "coverage": 1.0,
            "rule": "constant argmax of the training-split class distribution",
        }

    time_column = (
        "target_time" if "target_time" in frame
        else ("window_end" if "window_end" in frame else None)
    )
    order_columns = ["stock", "date", *([time_column] if time_column else [])]
    ordered = frame.sort_values(order_columns, kind="stable")
    groups = ordered.groupby(["stock", "date"], sort=False)
    previous = groups["true_label"].shift(1)
    available_series = previous.notna()
    # A saved table can have lunch/session gaps.  Only call the lag an
    # immediately preceding move when its target is exactly the current
    # window end; row adjacency alone is not sufficient.
    if "target_time" in ordered and "window_end" in ordered:
        previous_target = groups["target_time"].shift(1)
        available_series &= pd.to_datetime(previous_target, errors="coerce").eq(
            pd.to_datetime(ordered["window_end"], errors="coerce")
        )
    available = available_series.to_numpy()
    if available.any():
        truth = ordered.loc[available, "true_label"].to_numpy(np.int64)
        persistence = previous.loc[available].to_numpy(np.int64)
        reversal = np.array([2, 1, 0], dtype=np.int64)[persistence]
        for name, predicted, explanation in (
            (
                "last_move_persistence", persistence,
                "repeat the last observed one-minute down/flat/up state",
            ),
            (
                "last_move_reversal", reversal,
                "reverse the last observed direction while retaining flat",
            ),
        ):
            report[name] = {
                **_hard_metrics(truth, predicted, 3),
                "coverage": float(len(truth) / n),
                "rule": explanation,
                "excluded_without_contiguous_previous_target": int(n - len(truth)),
            }
    return report


def compare_conditional_binary_model(
    full_predictions: pd.DataFrame | str | Path,
    conditional_predictions: pd.DataFrame | str | Path,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compare direction models only on their common, future-conditioned scope."""
    full = validate_prediction_frame(full_predictions, FULL_CLASS_NAMES)
    binary = validate_prediction_frame(conditional_predictions, ("down", "up"))
    key = ["stock", "target_time"] if "target_time" in full and "target_time" in binary else ["stock", "window_end"]
    full_subset = full[[*key, "true_label", "prob_down", "prob_up"]].rename(
        columns={"true_label": "full_true", "prob_down": "full_down", "prob_up": "full_up"}
    )
    binary_subset = binary[[*key, "true_label", "prob_down", "prob_up"]].rename(
        columns={"true_label": "binary_true", "prob_down": "binary_down", "prob_up": "binary_up"}
    )
    aligned = binary_subset.merge(full_subset, on=key, how="left", validate="one_to_one", indicator=True)
    if not (aligned["_merge"] == "both").all():
        raise ValueError("conditional predictions contain samples absent from the full-window table")
    if (aligned["full_true"] == 1).any():
        raise ValueError("conditional prediction table unexpectedly contains flat targets")
    mapped = (aligned["full_true"].to_numpy(np.int64) == 2).astype(np.int64)
    if not np.array_equal(mapped, aligned["binary_true"].to_numpy(np.int64)):
        raise ValueError("conditional and full-window true labels disagree")
    binary_probability = aligned[["binary_down", "binary_up"]].to_numpy(float)
    full_direction = aligned[["full_down", "full_up"]].to_numpy(float)
    direction_mass = full_direction.sum(axis=1, keepdims=True)
    if (direction_mass <= 0.0).any():
        raise ValueError("full-window probabilities assign zero mass to both directions")
    full_direction = full_direction / direction_mass
    return {
        "scope": "actual_nonflat_targets_only_post_outcome_conditioning",
        "warning": (
            "This is a future-conditioned direction diagnostic, not deployable all-window coverage: "
            "whether the next minute is flat is unknown at prediction time."
        ),
        "n": int(len(aligned)),
        "share_of_all_full_windows": float(len(aligned) / len(full)),
        "conditional_binary_lstm": probability_metrics(
            mapped, binary_probability, ("down", "up"), n_bins, include_reliability=False
        ),
        "full_ensemble_direction_on_same_rows": probability_metrics(
            mapped, full_direction, ("down", "up"), n_bins, include_reliability=False
        ),
    }


def date_stability_report(
    predictions: pd.DataFrame | str | Path,
    class_names: Sequence[str] = FULL_CLASS_NAMES,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Slice one frozen model by date; this is not walk-forward refitting."""
    frame = validate_prediction_frame(predictions, class_names)
    probability_columns = _probability_columns(class_names)
    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        metrics = probability_metrics(
            group["true_label"].to_numpy(np.int64),
            group[probability_columns].to_numpy(float),
            class_names,
            n_bins,
            include_reliability=False,
        )
        rows.append({"date": date, **metrics})
    accuracies = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    macro_f1 = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
    return {
        "scope": "one_frozen_model_sliced_by_test_date_not_refitted_walk_forward",
        "n_dates": len(rows),
        "per_date": rows,
        "summary": {
            "accuracy_mean": float(accuracies.mean()),
            "accuracy_std": float(accuracies.std(ddof=0)),
            "accuracy_min": float(accuracies.min()),
            "accuracy_max": float(accuracies.max()),
            "macro_f1_mean": float(macro_f1.mean()),
            "macro_f1_std": float(macro_f1.std(ddof=0)),
            "worst_accuracy_date": rows[int(accuracies.argmin())]["date"],
            "best_accuracy_date": rows[int(accuracies.argmax())]["date"],
        },
    }


def _normalise_dates(dates: Iterable[str | pd.Timestamp]) -> list[str]:
    parsed = pd.to_datetime(pd.Series(list(dates), dtype="object"), errors="coerce")
    if parsed.isna().any():
        raise ValueError("dates contain an invalid value")
    result = sorted(set(parsed.dt.strftime("%Y%m%d").tolist()))
    if not result:
        raise ValueError("dates must be non-empty")
    return result


def make_walk_forward_splits(
    dates: Iterable[str | pd.Timestamp],
    min_train_dates: int,
    validation_dates: int,
    test_dates: int,
    step_dates: int | None = None,
    train_window_dates: int | None = None,
    max_folds: int | None = None,
) -> list[dict[str, Any]]:
    """Build ordered folds for true per-fold refitting.

    ``train_window_dates=None`` gives an expanding window.  A positive value
    gives a trailing window, while still requiring at least
    ``min_train_dates`` observations before the first validation date.
    Test blocks are non-overlapping by construction.
    """
    ordered = _normalise_dates(dates)
    values = (min_train_dates, validation_dates, test_dates)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("train, validation, and test date counts must be positive integers")
    step = test_dates if step_dates is None else step_dates
    if not isinstance(step, int) or step < test_dates:
        raise ValueError("step_dates must be an integer at least as large as test_dates")
    if train_window_dates is not None and (
        not isinstance(train_window_dates, int) or train_window_dates < min_train_dates
    ):
        raise ValueError("train_window_dates must be None or at least min_train_dates")
    if max_folds is not None and (not isinstance(max_folds, int) or max_folds <= 0):
        raise ValueError("max_folds must be None or a positive integer")

    folds: list[dict[str, Any]] = []
    train_stop = min_train_dates
    while train_stop + validation_dates + test_dates <= len(ordered):
        train_start = 0 if train_window_dates is None else max(0, train_stop - train_window_dates)
        val_start = train_stop
        val_stop = val_start + validation_dates
        test_start = val_stop
        test_stop = test_start + test_dates
        fold_id = len(folds) + 1
        folds.append({
            "fold": fold_id,
            "mode": "expanding" if train_window_dates is None else "trailing",
            "train": [ordered[train_start], ordered[train_stop - 1]],
            "validation": [ordered[val_start], ordered[val_stop - 1]],
            "test": [ordered[test_start], ordered[test_stop - 1]],
            "n_train_dates": train_stop - train_start,
            "n_validation_dates": validation_dates,
            "n_test_dates": test_dates,
        })
        if max_folds is not None and len(folds) >= max_folds:
            break
        train_stop += step
    if not folds:
        raise ValueError("not enough dates for one walk-forward fold")
    return folds


def aggregate_walk_forward_predictions(
    fold_predictions: Mapping[str, pd.DataFrame | str | Path],
    class_names: Sequence[str] = FULL_CLASS_NAMES,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate non-overlapping OOS prediction files from refitted folds."""
    if not fold_predictions:
        raise ValueError("fold_predictions must be non-empty")
    probability_columns = _probability_columns(class_names)
    frames: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    used_dates: set[str] = set()
    timestamp_key: str | None = None
    for fold, source in fold_predictions.items():
        frame = validate_prediction_frame(source, class_names)
        current_key = (
            "target_time" if "target_time" in frame
            else ("window_end" if "window_end" in frame else "date")
        )
        if timestamp_key is None:
            timestamp_key = current_key
        elif current_key != timestamp_key:
            raise ValueError("walk-forward folds use inconsistent sample timestamp keys")
        fold_dates = set(frame["date"].unique().tolist())
        date_overlap = sorted(used_dates.intersection(fold_dates))
        if date_overlap:
            raise ValueError(
                f"walk-forward test folds overlap on calendar date {date_overlap[0]}"
            )
        used_dates.update(fold_dates)
        metrics = probability_metrics(
            frame["true_label"].to_numpy(np.int64),
            frame[probability_columns].to_numpy(float),
            class_names,
            n_bins,
            include_reliability=False,
        )
        fold_reports.append({
            "fold": str(fold),
            "test_start": frame["date"].min(),
            "test_end": frame["date"].max(),
            **metrics,
        })
        tagged = frame.copy()
        tagged["fold"] = str(fold)
        frames.append(tagged)
    combined = pd.concat(frames, ignore_index=True)
    assert timestamp_key is not None
    key = ["stock", timestamp_key]
    if combined.duplicated(key).any():
        duplicate = combined.loc[combined.duplicated(key, keep=False), key].iloc[0].to_dict()
        raise ValueError(f"walk-forward test folds overlap at sample {duplicate}")
    pooled = probability_metrics(
        combined["true_label"].to_numpy(np.int64),
        combined[probability_columns].to_numpy(float),
        class_names,
        n_bins,
        include_reliability=True,
    )
    accuracies = np.asarray([fold["accuracy"] for fold in fold_reports], dtype=float)
    return {
        "scope": "pooled_non_overlapping_test_predictions_from_independently_refitted_folds",
        "n_folds": len(fold_reports),
        "folds": fold_reports,
        "pooled": pooled,
        "dispersion": {
            "fold_accuracy_mean": float(accuracies.mean()),
            "fold_accuracy_std": float(accuracies.std(ddof=0)),
            "fold_accuracy_min": float(accuracies.min()),
            "fold_accuracy_max": float(accuracies.max()),
        },
    }


def evaluate_saved_components(
    model_path: str | Path,
    data_dir: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device: str = "cpu",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Evaluate joint-only, two-stage-only, and fused saved components.

    This is a structural ablation of an already-trained bundle.  It answers
    whether each saved branch adds value, without the cost or researcher
    degrees of freedom of another training run.
    """
    from .lstm_full import (
        _component_input,
        _predict,
        blend_probabilities,
        load_full_model,
        two_stage_probabilities,
    )
    from .lstm_model import _make_split

    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    ensemble = load_full_model(model_path, device=device)
    stocks = list(ensemble.config["stock_codes"])
    date_range = tuple(ensemble.config["splits"][split])
    seq_len = int(ensemble.component_configs["joint"]["seq_len"])
    legacy = _make_split(
        stocks, date_range, seq_len, Path(data_dir), "legacy", "three_class", True
    )
    enhanced = _make_split(
        stocks, date_range, seq_len, Path(data_dir), "enhanced", "three_class", True
    )
    legacy_x, legacy_y, legacy_ids, legacy_coverage, legacy_metadata = legacy
    enhanced_x, labels, stock_ids, coverage, metadata = enhanced
    if (
        not np.array_equal(legacy_y, labels)
        or not np.array_equal(legacy_ids, stock_ids)
        or legacy_coverage != coverage
        or not legacy_metadata.equals(metadata)
    ):
        raise ValueError("legacy and enhanced full-window samples are not aligned")

    direction_x = _component_input(
        legacy_x, stock_ids, ensemble.component_configs["direction"], stocks
    )
    movement_x = _component_input(
        enhanced_x, stock_ids, ensemble.component_configs["movement"], stocks
    )
    joint_x = _component_input(
        enhanced_x, stock_ids, ensemble.component_configs["joint"], stocks
    )
    direction = _predict(ensemble.models["direction"], direction_x, batch_size, ensemble.device)
    movement = _predict(ensemble.models["movement"], movement_x, batch_size, ensemble.device)
    joint = _predict(ensemble.models["joint"], joint_x, batch_size, ensemble.device)
    two_stage = two_stage_probabilities(
        movement[:, 1], direction[:, 1], float(ensemble.config["move_bias"])
    )
    fused = blend_probabilities(joint, two_stage, float(ensemble.config["joint_weight"])).astype(float)

    nonflat = labels != 1
    direction_labels = (labels[nonflat] == 2).astype(np.int64)
    last_return = legacy_x[:, -1, 3]
    previous_state = np.full(len(last_return), 1, dtype=np.int64)
    previous_state[last_return < -1e-8] = 0
    previous_state[last_return > 1e-8] = 2
    reversal_state = np.array([2, 1, 0], dtype=np.int64)[previous_state]
    conditional_reversal = (last_return[nonflat] < -1e-8).astype(np.int64)
    metrics = {
        "joint_only": probability_metrics(
            labels, joint, FULL_CLASS_NAMES, n_bins, include_reliability=False
        ),
        "two_stage_only": probability_metrics(
            labels, two_stage, FULL_CLASS_NAMES, n_bins, include_reliability=False
        ),
        "fused_ensemble": probability_metrics(
            labels, fused, FULL_CLASS_NAMES, n_bins, include_reliability=True
        ),
        "movement_component": {
            **probability_metrics(
                (labels != 1).astype(np.int64), movement, ("flat", "move"),
                n_bins, include_reliability=False,
            ),
            "scope": "all_windows_flat_vs_move",
        },
        "direction_component": {
            **probability_metrics(
                direction_labels, direction[nonflat], ("down", "up"),
                n_bins, include_reliability=False,
            ),
            "scope": "actual_nonflat_targets_only_post_outcome_conditioning",
            "share_of_all_windows": float(nonflat.mean()),
        },
        "last_return_persistence_baseline": {
            **_hard_metrics(labels, previous_state, 3),
            "scope": "all_windows_using_the_last_return_visible_at_prediction_time",
            "coverage": 1.0,
        },
        "last_return_reversal_baseline": {
            **_hard_metrics(labels, reversal_state, 3),
            "scope": "all_windows_using_the_last_return_visible_at_prediction_time",
            "coverage": 1.0,
        },
        "conditional_direction_reversal_baseline": {
            **_hard_metrics(direction_labels, conditional_reversal, 2),
            "scope": "actual_nonflat_targets_only_post_outcome_conditioning",
            "share_of_all_windows": float(nonflat.mean()),
        },
    }
    prediction_frame = metadata.copy()
    prediction_frame["true_label"] = labels
    for prefix, probability in (
        ("joint", joint), ("two_stage", two_stage), ("fused", fused)
    ):
        for class_id, class_name in enumerate(FULL_CLASS_NAMES):
            prediction_frame[f"{prefix}_prob_{class_name}"] = probability[:, class_id]
        prediction_frame[f"{prefix}_predicted_label"] = probability.argmax(axis=1)
    prediction_frame["movement_prob_move"] = movement[:, 1]
    prediction_frame["direction_prob_up_given_move"] = direction[:, 1]
    return {
        "split": split,
        "date_range": list(date_range),
        "coverage": coverage,
        "metrics": metrics,
        "labels": labels,
        "probabilities": {
            "joint": joint,
            "two_stage": two_stage,
            "fused": fused,
            "movement": movement,
            "direction": direction,
        },
        "prediction_frame": prediction_frame,
    }
