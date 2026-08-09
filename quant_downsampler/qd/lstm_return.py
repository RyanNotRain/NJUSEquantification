"""Leakage-safe multi-task LSTM for direction and next-minute magnitude.

This module is intentionally separate from the frozen classification models.
It shares one sequence encoder between a down/flat/up classification head and
an absolute next-minute return-in-bps head.  Every transform, loss weight and
opening threshold is frozen with training/validation data before the test
split is first loaded.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR
from .lstm_model import (
    CHANNELS,
    DEFAULT_SPLITS,
    _make_split,
    classification_metrics,
    feature_names,
    set_seed,
)
from .lstm_strategy import (
    attach_realised_returns,
    break_even_cost_bps,
    build_portfolio_path,
    generate_target_weights,
    run_strategy_suite,
    strategy_statistics,
)


CLASS_NAMES = ("down", "flat", "up")
MODEL_VERSION = "return_multitask_v1"
GENERATED_ROOT_FILES = (
    "selection_frozen_before_test.json",
    "model.pt",
    "lambda_selection.csv",
    "validation_threshold_selection.csv",
    "history.json",
    "validation_predictions.csv",
    "test_predictions.csv",
    "test_metrics.json",
    "replay_audit.json",
    "strategy_comparison.csv",
)
GENERATED_STRATEGY_DIRS = tuple(
    f"{signal}_{side}"
    for signal in ("expected_return", "probability_gap")
    for side in ("long_short", "long_only")
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _prepare_output_dir(target: Path, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty return-LSTM output directory: {target}; "
            "choose another out_dir or pass overwrite=True"
        )
    target.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for filename in GENERATED_ROOT_FILES:
        path = target / filename
        if path.is_file():
            path.unlink()
    strategy_root = target / "strategies"
    for directory_name in GENERATED_STRATEGY_DIRS:
        directory = strategy_root / directory_name
        if not directory.is_dir():
            continue
        for filename in (
            "strategy_summary.csv",
            "cost_sensitivity.csv",
            "metrics.json",
            "README.md",
            "strategy_cost_sensitivity.png",
        ):
            path = directory / filename
            if path.is_file():
                path.unlink()
        for child_name in ("positions", "portfolio_paths"):
            child = directory / child_name
            if child.is_dir():
                for path in child.glob("*.csv"):
                    if path.is_file():
                        path.unlink()
                try:
                    child.rmdir()
                except OSError:
                    # Unknown user files are intentionally preserved.
                    pass
        try:
            directory.rmdir()
        except OSError:
            pass
    if strategy_root.is_dir():
        try:
            strategy_root.rmdir()
        except OSError:
            pass


def _validate_splits(splits: Mapping[str, tuple[str, str]]) -> None:
    for name in ("train", "val", "test"):
        if name not in splits or len(splits[name]) != 2:
            raise ValueError(f"missing or malformed {name!r} date split")
        start, end = splits[name]
        if not re.fullmatch(r"\d{8}", start) or not re.fullmatch(r"\d{8}", end):
            raise ValueError("split dates must use YYYYMMDD format")
        if start > end:
            raise ValueError(f"invalid {name} split: {(start, end)}")
    if not (
        splits["train"][1] < splits["val"][0]
        and splits["val"][1] < splits["test"][0]
    ):
        raise ValueError("splits must be strictly ordered and non-overlapping")


def _resolve_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _resolve_stocks(
    stock_codes: Sequence[str] | None,
    n_stocks: int,
    minute_dir: Path,
) -> list[str]:
    if stock_codes is not None:
        stocks = [str(value).strip() for value in stock_codes]
        if not stocks or any(not value for value in stocks):
            raise ValueError("stock_codes must contain non-empty values")
        if len(set(stocks)) != len(stocks):
            raise ValueError("stock_codes contains duplicates")
        return stocks
    files = sorted((minute_dir / "close").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no minute close files under {minute_dir}")
    available = pd.read_csv(files[0], nrows=1, index_col=0).columns.tolist()
    if n_stocks <= 0 or n_stocks > len(available):
        raise ValueError(
            f"n_stocks must lie within [1, {len(available)}], got {n_stocks}"
        )
    return [str(value) for value in available[:n_stocks]]


def attach_return_targets(
    metadata: pd.DataFrame,
    close_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact signed and absolute ``close[t+1]/close[t]-1`` in bps."""
    required = {"stock", "window_end", "target_time"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"metadata is missing return-alignment columns: {missing}")
    probe = metadata.loc[:, ["stock", "window_end", "target_time"]].copy()
    # Reuse the strategy's exact timestamp alignment and validation path.
    probe["prob_down"] = 1.0 / 3.0
    probe["prob_flat"] = 1.0 / 3.0
    probe["prob_up"] = 1.0 / 3.0
    attached = attach_realised_returns(probe, close_dir)
    signed = attached["realised_return"].to_numpy(dtype=np.float64) * 10_000.0
    if not np.isfinite(signed).all():
        raise ValueError("aligned next-minute returns contain non-finite values")
    return signed, np.abs(signed)


def fit_magnitude_transform(
    absolute_return_bps: np.ndarray,
    stock_ids: np.ndarray,
    n_stocks: int,
    clip_quantile: float = 0.995,
) -> dict[str, Any]:
    """Fit robust per-stock scales and a global normalized clipping cap."""
    values = np.asarray(absolute_return_bps, dtype=np.float64)
    ids = np.asarray(stock_ids, dtype=np.int64)
    if values.ndim != 1 or len(values) != len(ids) or not len(values):
        raise ValueError("magnitude targets and stock_ids must be aligned and non-empty")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("absolute return targets must be finite and non-negative")
    if n_stocks <= 0 or ids.min() < 0 or ids.max() >= n_stocks:
        raise ValueError("stock_ids are outside the declared stock range")
    if not 0.9 <= clip_quantile < 1.0:
        raise ValueError("clip_quantile must lie within [0.9, 1.0)")

    scales = np.empty(n_stocks, dtype=np.float64)
    for stock_id in range(n_stocks):
        stock_values = values[ids == stock_id]
        positive = stock_values[stock_values > 0.0]
        if not len(stock_values):
            raise ValueError(f"training data has no rows for stock_id={stock_id}")
        scales[stock_id] = float(np.median(positive)) if len(positive) else 1.0
        if not np.isfinite(scales[stock_id]) or scales[stock_id] <= 0.0:
            scales[stock_id] = 1.0
    normalized = values / scales[ids]
    cap = float(np.quantile(normalized, clip_quantile))
    if not np.isfinite(cap) or cap <= 0.0:
        cap = 1.0
    return {
        "stock_scales_bps": scales,
        "normalized_clip": cap,
        "clip_quantile": float(clip_quantile),
        "fit_sample_n": int(len(values)),
    }


def transform_magnitude_targets(
    absolute_return_bps: np.ndarray,
    stock_ids: np.ndarray,
    transform: Mapping[str, Any],
) -> np.ndarray:
    values = np.asarray(absolute_return_bps, dtype=np.float64)
    ids = np.asarray(stock_ids, dtype=np.int64)
    scales = np.asarray(transform["stock_scales_bps"], dtype=np.float64)
    cap = float(transform["normalized_clip"])
    if len(values) != len(ids) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("invalid magnitude targets")
    if len(ids) and (ids.min() < 0 or ids.max() >= len(scales)):
        raise ValueError("stock_ids are incompatible with the magnitude transform")
    return np.minimum(values / scales[ids], cap).astype(np.float32)


def inverse_magnitude_predictions(
    normalized_prediction: np.ndarray,
    stock_ids: np.ndarray,
    transform: Mapping[str, Any],
) -> np.ndarray:
    prediction = np.asarray(normalized_prediction, dtype=np.float64)
    ids = np.asarray(stock_ids, dtype=np.int64)
    scales = np.asarray(transform["stock_scales_bps"], dtype=np.float64)
    cap = float(transform["normalized_clip"])
    if len(prediction) != len(ids) or not np.isfinite(prediction).all():
        raise ValueError("normalized predictions and stock_ids are invalid")
    if len(ids) and (ids.min() < 0 or ids.max() >= len(scales)):
        raise ValueError("stock_ids are incompatible with the magnitude transform")
    return np.clip(prediction, 0.0, cap) * scales[ids]


def _fit_feature_scaler(
    train_x: np.ndarray,
    train_ids: np.ndarray,
    n_stocks: int,
) -> tuple[np.ndarray, np.ndarray]:
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for stock_id in range(n_stocks):
        stock_x = train_x[train_ids == stock_id]
        if not len(stock_x):
            raise ValueError(f"no training feature rows for stock_id={stock_id}")
        mean = stock_x.mean(axis=(0, 1))
        std = stock_x.std(axis=(0, 1))
        std[std < 1e-8] = 1.0
        means.append(mean)
        stds.append(std)
    return np.stack(means).astype(np.float32), np.stack(stds).astype(np.float32)


def _transform_features(
    values: np.ndarray,
    stock_ids: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    include_stock_id: bool = True,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    ids = np.asarray(stock_ids, dtype=np.int64)
    if x.ndim != 3 or len(x) != len(ids):
        raise ValueError("feature sequences and stock_ids are not aligned")
    expected = (len(mean), x.shape[2])
    if mean.shape != expected or std.shape != expected:
        raise ValueError("saved feature scaler has incompatible shape")
    x = (x - mean[ids, None, :]) / std[ids, None, :]
    if include_stock_id:
        one_hot = np.eye(len(mean), dtype=np.float32)[ids]
        repeated = np.broadcast_to(one_hot[:, None, :], (len(x), x.shape[1], len(mean)))
        x = np.concatenate((x, repeated), axis=2)
    return x.astype(np.float32, copy=False)


def _load_raw_split(
    stocks: list[str],
    date_range: tuple[str, str],
    seq_len: int,
    minute_dir: Path,
) -> dict[str, Any]:
    x, labels, stock_ids, coverage, metadata = _make_split(
        stocks,
        date_range,
        seq_len,
        minute_dir,
        "enhanced",
        "three_class",
        True,
    )
    signed_bps, absolute_bps = attach_return_targets(metadata, minute_dir / "close")
    return {
        "x": x,
        "labels": labels,
        "stock_ids": stock_ids,
        "coverage": coverage,
        "metadata": metadata,
        "signed_return_bps": signed_bps,
        "absolute_return_bps": absolute_bps,
    }


def _subsample_indices(
    labels: np.ndarray,
    stock_ids: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    if maximum is None or maximum >= len(labels):
        return np.arange(len(labels))
    if maximum <= 0:
        raise ValueError("max_train_samples must be positive")
    rng = np.random.default_rng(seed)
    strata = labels * (int(stock_ids.max()) + 1) + stock_ids
    selected: list[np.ndarray] = []
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        allocation = max(1, int(round(maximum * len(members) / len(labels))))
        selected.append(rng.choice(members, size=min(allocation, len(members)), replace=False))
    result = np.concatenate(selected)
    if len(result) > maximum:
        result = rng.choice(result, size=maximum, replace=False)
    elif len(result) < maximum:
        remaining = np.setdiff1d(np.arange(len(labels)), result, assume_unique=False)
        extra = rng.choice(remaining, size=min(maximum - len(result), len(remaining)), replace=False)
        result = np.concatenate((result, extra))
    return np.sort(result)


class ReturnMultiTaskLSTM(nn.Module):
    """One LSTM encoder with three-class and non-negative magnitude heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        recent_size = max(16, hidden_size // 2)
        self.recent = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, recent_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        encoded_size = hidden_size + recent_size
        self.shared = nn.Sequential(
            nn.LayerNorm(encoded_size),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(encoded_size, len(CLASS_NAMES))
        self.magnitude_head = nn.Linear(encoded_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(x)
        encoded = torch.cat((output[:, -1], self.recent(x[:, -1])), dim=1)
        shared = self.shared(encoded)
        logits = self.class_head(shared)
        magnitude = F.softplus(self.magnitude_head(shared).squeeze(1))
        return logits, magnitude


def _loader(
    x: np.ndarray,
    labels: np.ndarray,
    magnitude: np.ndarray,
    stock_ids: np.ndarray,
    signed_bps: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(np.asarray(x, dtype=np.float32)),
            torch.from_numpy(np.asarray(labels, dtype=np.int64)),
            torch.from_numpy(np.asarray(magnitude, dtype=np.float32)),
            torch.from_numpy(np.asarray(stock_ids, dtype=np.int64)),
            torch.from_numpy(np.asarray(signed_bps, dtype=np.float32)),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return float(np.corrcoef(xr, yr)[0, 1])


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def grouped_return_report(
    expected_return_bps: np.ndarray,
    realised_return_bps: np.ndarray,
    n_groups: int = 10,
) -> dict[str, Any]:
    """Summarize realized returns in equal-count predicted-return groups."""
    expected = np.asarray(expected_return_bps, dtype=np.float64)
    realised = np.asarray(realised_return_bps, dtype=np.float64)
    if len(expected) != len(realised) or not len(expected):
        raise ValueError("expected and realised returns must align and be non-empty")
    if not np.isfinite(expected).all() or not np.isfinite(realised).all():
        raise ValueError("grouping inputs must be finite")
    count = min(max(int(n_groups), 2), len(expected))
    ordered = np.argsort(expected, kind="mergesort")
    rows: list[dict[str, float | int]] = []
    for group_id, indices in enumerate(np.array_split(ordered, count), start=1):
        rows.append({
            "group": group_id,
            "n": int(len(indices)),
            "mean_prediction_bps": float(expected[indices].mean()),
            "mean_realised_bps": float(realised[indices].mean()),
        })
    group_means = np.asarray([row["mean_realised_bps"] for row in rows], dtype=float)
    group_ids = np.arange(1, len(rows) + 1, dtype=float)
    return {
        "groups": rows,
        "monotonic_spearman": _rank_correlation(group_ids, group_means),
        "top_minus_bottom_bps": float(group_means[-1] - group_means[0]),
    }


def prediction_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    predicted_absolute_bps: np.ndarray,
    expected_return_bps: np.ndarray,
    realised_return_bps: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    predicted_absolute = np.asarray(predicted_absolute_bps, dtype=np.float64)
    expected = np.asarray(expected_return_bps, dtype=np.float64)
    realised = np.asarray(realised_return_bps, dtype=np.float64)
    n = len(labels)
    if (
        probability.shape != (n, len(CLASS_NAMES))
        or len(predicted_absolute) != n
        or len(expected) != n
        or len(realised) != n
    ):
        raise ValueError("prediction arrays are not aligned")
    if not np.isfinite(
        np.column_stack((probability, predicted_absolute, expected, realised))
    ).all():
        raise ValueError("prediction arrays contain non-finite values")

    raw_classification = classification_metrics(labels, probability, threshold=None)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[labels]
    clipped_probability = np.clip(probability, 1e-12, 1.0)
    absolute_realised = np.abs(realised)
    zero_magnitude_mae = float(np.mean(absolute_realised))
    zero_magnitude_rmse = float(np.sqrt(np.mean(absolute_realised ** 2)))
    magnitude_mae = float(np.mean(np.abs(predicted_absolute - absolute_realised)))
    magnitude_rmse = float(
        np.sqrt(np.mean((predicted_absolute - absolute_realised) ** 2))
    )
    signed_mae = float(np.mean(np.abs(expected - realised)))
    signed_rmse = float(np.sqrt(np.mean((expected - realised) ** 2)))
    zero_signed_mae = float(np.mean(np.abs(realised)))
    zero_signed_rmse = float(np.sqrt(np.mean(realised ** 2)))
    nonflat = realised != 0.0
    direction_hit = (
        float((np.sign(expected[nonflat]) == np.sign(realised[nonflat])).mean())
        if nonflat.any()
        else 0.0
    )
    return {
        "n": int(n),
        "classification": {
            "accuracy": raw_classification["accuracy"],
            "macro_precision": raw_classification["precision"],
            "macro_recall": raw_classification["recall"],
            "macro_f1": raw_classification["f1"],
            "confusion_matrix": raw_classification["confusion_matrix"],
            "brier_score": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
            "negative_log_likelihood": float(
                -np.mean(np.log(clipped_probability[np.arange(n), labels]))
            ),
        },
        "magnitude": {
            "mae_bps": magnitude_mae,
            "rmse_bps": magnitude_rmse,
            "pearson": _pearson(predicted_absolute, absolute_realised),
            "spearman": _rank_correlation(predicted_absolute, absolute_realised),
            "zero_prediction_baseline_mae_bps": zero_magnitude_mae,
            "zero_prediction_baseline_rmse_bps": zero_magnitude_rmse,
            "mae_improvement_vs_zero_bps": zero_magnitude_mae - magnitude_mae,
            "rmse_improvement_vs_zero_bps": zero_magnitude_rmse - magnitude_rmse,
        },
        "signed_expected_return": {
            "mae_bps": signed_mae,
            "rmse_bps": signed_rmse,
            "zero_prediction_baseline_mae_bps": zero_signed_mae,
            "zero_prediction_baseline_rmse_bps": zero_signed_rmse,
            "mae_improvement_vs_zero_bps": zero_signed_mae - signed_mae,
            "rmse_improvement_vs_zero_bps": zero_signed_rmse - signed_rmse,
            "pearson_ic": _pearson(expected, realised),
            "spearman_ic": _rank_correlation(expected, realised),
            "nonflat_direction_hit_rate": direction_hit,
        },
        "grouped_returns": grouped_return_report(expected, realised),
    }


def _predict_loader(
    model: ReturnMultiTaskLSTM,
    loader: DataLoader,
    magnitude_transform: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    probabilities: list[np.ndarray] = []
    normalized_magnitudes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    signed: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y, _magnitude, stock_id, signed_bps in loader:
            logits, magnitude = model(x.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            normalized_magnitudes.append(magnitude.cpu().numpy())
            labels.append(y.numpy())
            ids.append(stock_id.numpy())
            signed.append(signed_bps.numpy())
    probability = np.concatenate(probabilities)
    normalized = np.concatenate(normalized_magnitudes)
    stock_ids = np.concatenate(ids)
    predicted_absolute = inverse_magnitude_predictions(
        normalized, stock_ids, magnitude_transform
    )
    expected = (probability[:, 2] - probability[:, 0]) * predicted_absolute
    return {
        "probability": probability,
        "predicted_absolute_bps": predicted_absolute,
        "expected_return_bps": expected,
        "labels": np.concatenate(labels),
        "stock_ids": stock_ids,
        "signed_return_bps": np.concatenate(signed).astype(np.float64),
    }


def _validation_objective(metrics: Mapping[str, Any]) -> float:
    classification = float(metrics["classification"]["macro_f1"])
    return_ic = float(metrics["signed_expected_return"]["spearman_ic"])
    return classification + return_ic


def _train_candidate(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: np.ndarray,
    magnitude_transform: Mapping[str, Any],
    magnitude_lambda: float,
    epochs: int,
    patience: int,
    learning_rate: float,
    class_weighted: bool,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, list[float]], dict[str, Any]]:
    if magnitude_lambda < 0.0 or not np.isfinite(magnitude_lambda):
        raise ValueError("magnitude_lambda must be finite and non-negative")
    set_seed(seed)
    model = ReturnMultiTaskLSTM(input_size, hidden_size, num_layers, dropout).to(device)
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    class_weights = len(train_labels) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    weight = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weighted
        else None
    )
    classification_loss = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    history: dict[str, list[float]] = {
        "train_total_loss": [],
        "train_classification_loss": [],
        "train_magnitude_loss": [],
        "validation_objective": [],
        "validation_macro_f1": [],
        "validation_return_spearman": [],
    }
    best_objective = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] | None = None
    stale = 0

    for epoch in range(epochs):
        model.train()
        total_sum = class_sum = magnitude_sum = 0.0
        for x, y, magnitude, _stock_id, _signed_bps in train_loader:
            x = x.to(device)
            y = y.to(device)
            magnitude = magnitude.to(device)
            optimizer.zero_grad()
            logits, predicted_magnitude = model(x)
            class_loss = classification_loss(logits, y)
            magnitude_loss = F.smooth_l1_loss(predicted_magnitude, magnitude)
            loss = class_loss + magnitude_lambda * magnitude_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_n = len(x)
            total_sum += float(loss.item()) * batch_n
            class_sum += float(class_loss.item()) * batch_n
            magnitude_sum += float(magnitude_loss.item()) * batch_n

        validation = _predict_loader(model, val_loader, magnitude_transform, device)
        metrics = prediction_metrics(
            validation["labels"],
            validation["probability"],
            validation["predicted_absolute_bps"],
            validation["expected_return_bps"],
            validation["signed_return_bps"],
        )
        objective = _validation_objective(metrics)
        train_n = len(train_loader.dataset)
        history["train_total_loss"].append(total_sum / train_n)
        history["train_classification_loss"].append(class_sum / train_n)
        history["train_magnitude_loss"].append(magnitude_sum / train_n)
        history["validation_objective"].append(objective)
        history["validation_macro_f1"].append(
            float(metrics["classification"]["macro_f1"])
        )
        history["validation_return_spearman"].append(
            float(metrics["signed_expected_return"]["spearman_ic"])
        )
        scheduler.step(objective)
        print(
            f"lambda={magnitude_lambda:g} epoch={epoch + 1:02d} "
            f"loss={history['train_total_loss'][-1]:.5f} "
            f"val_macro_f1={history['validation_macro_f1'][-1]:.4f} "
            f"val_return_ic={history['validation_return_spearman'][-1]:.4f}"
        )
        if objective > best_objective + 1e-6:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_metrics is None:
        raise RuntimeError("multi-task training did not produce a validation result")
    return (
        {key: value.detach().cpu() for key, value in best_state.items()},
        history,
        best_metrics,
    )


def _prediction_frame(
    metadata: pd.DataFrame,
    prediction: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    frame = metadata.copy().reset_index(drop=True)
    probability = prediction["probability"]
    labels = prediction["labels"].astype(np.int64)
    if len(frame) != len(labels):
        raise ValueError("metadata and predictions are not aligned")
    frame["true_label"] = labels
    frame["predicted_label"] = probability.argmax(axis=1)
    frame["prob_down"] = probability[:, 0]
    frame["prob_flat"] = probability[:, 1]
    frame["prob_up"] = probability[:, 2]
    frame["confidence"] = probability.max(axis=1)
    frame["realised_return_bps"] = prediction["signed_return_bps"]
    frame["realised_abs_return_bps"] = np.abs(prediction["signed_return_bps"])
    frame["predicted_abs_return_bps"] = prediction["predicted_absolute_bps"]
    frame["expected_return_bps"] = prediction["expected_return_bps"]
    frame["realised_return"] = frame["realised_return_bps"] / 10_000.0
    return frame


def select_opening_thresholds(
    validation_predictions: pd.DataFrame,
    base_cost_bps: float,
    quantiles: Sequence[float] = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    *,
    signal_mode: str = "expected_return",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select one opening threshold per side using validation economics only.

    Expected-return thresholds are measured in bps and cannot fall below the
    configured one-way cost.  Probability-gap thresholds are dimensionless;
    they receive their own validation search so the magnitude ablation is not
    confounded by comparing a filtered signal with an unfiltered one.
    """
    if not np.isfinite(base_cost_bps) or base_cost_bps < 0.0:
        raise ValueError("base_cost_bps must be finite and non-negative")
    if not quantiles or any(not 0.0 <= float(value) < 1.0 for value in quantiles):
        raise ValueError("threshold quantiles must lie within [0, 1)")
    if signal_mode not in {"expected_return", "probability_gap"}:
        raise ValueError("signal_mode must be expected_return or probability_gap")
    if signal_mode == "expected_return":
        score = np.abs(
            pd.to_numeric(
                validation_predictions["expected_return_bps"], errors="raise"
            ).to_numpy(dtype=np.float64)
        )
        # ``base_cost_bps`` is the configured one-way cost floor.  A full
        # round-trip can cost more; the realized turnover calculation below
        # still performs selection under the complete turnover path.
        thresholds = np.unique(np.maximum(
            float(base_cost_bps),
            np.quantile(score, np.asarray(quantiles, dtype=float)),
        ))
        threshold_unit = "bps"
    else:
        score = np.abs(
            pd.to_numeric(
                validation_predictions["prob_up"], errors="raise"
            ).to_numpy(dtype=np.float64)
            - pd.to_numeric(
                validation_predictions["prob_down"], errors="raise"
            ).to_numpy(dtype=np.float64)
        )
        thresholds = np.unique(np.concatenate((
            [0.0],
            np.quantile(score, np.asarray(quantiles, dtype=float)),
        )))
        threshold_unit = "probability_gap"
    rows: list[dict[str, Any]] = []
    chosen: dict[str, float] = {}
    for side in ("long_short", "long_only"):
        candidates: list[tuple[tuple[float, float, float], float]] = []
        for threshold in thresholds:
            weighted = generate_target_weights(
                validation_predictions,
                tier="all",
                weighting="confidence",
                score_threshold=float(threshold),
                require_directional_argmax=True,
                side=side,
                signal_mode=signal_mode,
            )
            path = build_portfolio_path(weighted)
            statistics = strategy_statistics(path, cost_bps=base_cost_bps)
            row = {
                "signal_mode": signal_mode,
                "side": side,
                "score_threshold": float(threshold),
                "threshold_unit": threshold_unit,
                "threshold_bps": (
                    float(threshold) if signal_mode == "expected_return" else np.nan
                ),
                "active_signal_rate": float(weighted["active_signal"].mean()),
                "net_total_return": float(statistics["net_total_return"]),
                "gross_total_return": float(statistics["gross_total_return"]),
                "average_daily_turnover": float(statistics["average_daily_turnover"]),
            }
            rows.append(row)
            # Prefer validation net return, then lower turnover, then a larger
            # threshold.  No test observation participates in this choice.
            selection_key = (
                row["net_total_return"],
                -row["average_daily_turnover"],
                row["score_threshold"],
            )
            candidates.append((selection_key, float(threshold)))
        chosen[side] = max(candidates, key=lambda value: value[0])[1]
    return chosen, pd.DataFrame(rows)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_return_model(
    model_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ReturnMultiTaskLSTM, dict[str, Any]]:
    target_device = _resolve_device(device)
    bundle = torch.load(Path(model_path), map_location=target_device, weights_only=False)
    if set(bundle) != {"state_dict", "config"}:
        raise ValueError("saved return-LSTM bundle must contain state_dict and config")
    config = bundle["config"]
    required = {
        "model_version",
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "stock_codes",
        "splits",
        "seq_len",
        "feature_scaler_mean",
        "feature_scaler_std",
        "magnitude_transform",
        "magnitude_lambda",
        "opening_threshold_bps",
        "state_sha256",
    }
    if not required.issubset(config) or config["model_version"] != MODEL_VERSION:
        raise ValueError("saved return-LSTM config is incomplete or incompatible")
    if _state_sha256(bundle["state_dict"]) != config["state_sha256"]:
        raise ValueError("saved return-LSTM state hash does not match its config")
    model = ReturnMultiTaskLSTM(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(target_device).eval()
    return model, config


def evaluate_saved_return_model(
    model_path: str | Path,
    data_dir: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Strictly recreate a validation/test split and replay the saved model."""
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    model, config = load_return_model(model_path, device)
    stocks = list(config["stock_codes"])
    raw = _load_raw_split(
        stocks,
        tuple(config["splits"][split]),
        int(config["seq_len"]),
        Path(data_dir),
    )
    mean = np.asarray(config["feature_scaler_mean"], dtype=np.float32)
    std = np.asarray(config["feature_scaler_std"], dtype=np.float32)
    x = _transform_features(raw["x"], raw["stock_ids"], mean, std, True)
    magnitude = transform_magnitude_targets(
        raw["absolute_return_bps"], raw["stock_ids"], config["magnitude_transform"]
    )
    loader = _loader(
        x,
        raw["labels"],
        magnitude,
        raw["stock_ids"],
        raw["signed_return_bps"],
        batch_size,
        False,
        0,
    )
    prediction = _predict_loader(
        model, loader, config["magnitude_transform"], _resolve_device(device)
    )
    metrics = prediction_metrics(
        prediction["labels"],
        prediction["probability"],
        prediction["predicted_absolute_bps"],
        prediction["expected_return_bps"],
        prediction["signed_return_bps"],
    )
    return {
        "metrics": metrics,
        "predictions": _prediction_frame(raw["metadata"], prediction),
        "coverage": raw["coverage"],
    }


def run_return_lstm(
    *,
    stock_codes: Sequence[str] | None = None,
    n_stocks: int = 5,
    seq_len: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 8,
    patience: int = 3,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    magnitude_lambdas: Sequence[float] = (0.1, 0.3, 1.0),
    class_weighted: bool = False,
    clip_quantile: float = 0.995,
    threshold_quantiles: Sequence[float] = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95),
    base_cost_bps: float = 5.0,
    cost_grid_bps: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0),
    max_train_samples: int | None = None,
    seed: int = 42,
    splits: Mapping[str, tuple[str, str]] | None = None,
    data_dir: str | Path = OUTPUT_DIR / "minute",
    out_dir: str | Path = OUTPUT_DIR / "lstm_return",
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train, freeze, test, strictly replay and strategy-test the new model."""
    ranges = dict(splits or DEFAULT_SPLITS)
    _validate_splits(ranges)
    if epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("epochs, patience and batch_size must be positive")
    if hidden_size <= 0 or num_layers <= 0 or not 0.0 <= dropout < 1.0:
        raise ValueError("invalid model architecture")
    lambdas = [float(value) for value in magnitude_lambdas]
    if not lambdas or any(not np.isfinite(value) or value < 0.0 for value in lambdas):
        raise ValueError("magnitude_lambdas must contain finite non-negative values")
    minute_dir = Path(data_dir)
    target = Path(out_dir)
    _prepare_output_dir(target, overwrite)
    target_device = _resolve_device(device)
    stocks = _resolve_stocks(stock_codes, n_stocks, minute_dir)

    print("loading training split (test remains unopened)")
    train = _load_raw_split(stocks, ranges["train"], seq_len, minute_dir)
    print("loading validation split (test remains unopened)")
    validation = _load_raw_split(stocks, ranges["val"], seq_len, minute_dir)

    feature_mean, feature_std = _fit_feature_scaler(
        train["x"], train["stock_ids"], len(stocks)
    )
    magnitude_transform = fit_magnitude_transform(
        train["absolute_return_bps"],
        train["stock_ids"],
        len(stocks),
        clip_quantile,
    )
    train_magnitude = transform_magnitude_targets(
        train["absolute_return_bps"], train["stock_ids"], magnitude_transform
    )
    validation_magnitude = transform_magnitude_targets(
        validation["absolute_return_bps"], validation["stock_ids"], magnitude_transform
    )
    train_x = _transform_features(
        train["x"], train["stock_ids"], feature_mean, feature_std, True
    )
    validation_x = _transform_features(
        validation["x"], validation["stock_ids"], feature_mean, feature_std, True
    )
    del train["x"], validation["x"]
    gc.collect()

    selected_indices = _subsample_indices(
        train["labels"], train["stock_ids"], max_train_samples, seed
    )
    train_loader = _loader(
        train_x[selected_indices],
        train["labels"][selected_indices],
        train_magnitude[selected_indices],
        train["stock_ids"][selected_indices],
        train["signed_return_bps"][selected_indices],
        batch_size,
        True,
        seed,
    )
    val_loader = _loader(
        validation_x,
        validation["labels"],
        validation_magnitude,
        validation["stock_ids"],
        validation["signed_return_bps"],
        batch_size,
        False,
        seed,
    )
    del train_x, validation_x
    gc.collect()

    candidates: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    states: dict[float, dict[str, torch.Tensor]] = {}
    for magnitude_lambda in lambdas:
        state, history, val_metrics = _train_candidate(
            input_size=len(feature_names("enhanced")) + len(stocks),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            train_loader=train_loader,
            val_loader=val_loader,
            train_labels=train["labels"][selected_indices],
            magnitude_transform=magnitude_transform,
            magnitude_lambda=magnitude_lambda,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            class_weighted=class_weighted,
            seed=seed,
            device=target_device,
        )
        states[magnitude_lambda] = state
        histories[str(magnitude_lambda)] = history
        candidates.append({
            "magnitude_lambda": magnitude_lambda,
            "validation_objective": _validation_objective(val_metrics),
            "validation_macro_f1": val_metrics["classification"]["macro_f1"],
            "validation_return_spearman_ic": val_metrics["signed_expected_return"]["spearman_ic"],
            "validation_magnitude_mae_bps": val_metrics["magnitude"]["mae_bps"],
            "best_epoch": int(np.argmax(history["validation_objective"]) + 1),
        })
    selected = max(
        candidates,
        key=lambda row: (
            float(row["validation_objective"]),
            -float(row["validation_magnitude_mae_bps"]),
            -float(row["magnitude_lambda"]),
        ),
    )
    selected_lambda = float(selected["magnitude_lambda"])
    selected_state = states[selected_lambda]
    selected_model = ReturnMultiTaskLSTM(
        len(feature_names("enhanced")) + len(stocks),
        hidden_size,
        num_layers,
        dropout,
    ).to(target_device)
    selected_model.load_state_dict(selected_state, strict=True)
    selected_model.eval()
    val_prediction = _predict_loader(
        selected_model, val_loader, magnitude_transform, target_device
    )
    validation_frame = _prediction_frame(validation["metadata"], val_prediction)
    opening_thresholds, expected_threshold_table = select_opening_thresholds(
        validation_frame,
        base_cost_bps,
        threshold_quantiles,
        signal_mode="expected_return",
    )
    probability_gap_thresholds, probability_threshold_table = (
        select_opening_thresholds(
            validation_frame,
            base_cost_bps,
            threshold_quantiles,
            signal_mode="probability_gap",
        )
    )
    threshold_table = pd.concat(
        (expected_threshold_table, probability_threshold_table),
        ignore_index=True,
    )

    state_hash = _state_sha256(selected_state)
    config: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "input_size": len(feature_names("enhanced")) + len(stocks),
        "base_feature_names": list(feature_names("enhanced")),
        "feature_names": list(feature_names("enhanced"))
        + [f"stock_id::{stock}" for stock in stocks],
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "stock_codes": stocks,
        "splits": ranges,
        "seq_len": seq_len,
        "feature_scaler": "per_stock_train_only",
        "feature_scaler_mean": feature_mean.tolist(),
        "feature_scaler_std": feature_std.tolist(),
        "magnitude_target": "abs(10000 * (close[t+1] / close[t] - 1))",
        "expected_return_signal": "(prob_up - prob_down) * predicted_abs_return_bps",
        "magnitude_transform": {
            **magnitude_transform,
            "stock_scales_bps": np.asarray(
                magnitude_transform["stock_scales_bps"]
            ).tolist(),
        },
        "magnitude_lambda": selected_lambda,
        "lambda_selection_objective": "validation_macro_f1 + validation_expected_return_spearman_ic",
        "opening_threshold_bps": opening_thresholds,
        "probability_gap_opening_threshold": probability_gap_thresholds,
        "opening_threshold_selection": f"validation net return at {base_cost_bps:g} one-way bps",
        "opening_threshold_floor_one_way_bps": float(base_cost_bps),
        "class_weighted": bool(class_weighted),
        "learning_rate": learning_rate,
        "epochs_requested": epochs,
        "best_epoch": int(selected["best_epoch"]),
        "patience": patience,
        "batch_size": batch_size,
        "seed": seed,
        "training_rows_available": int(len(train["labels"])),
        "training_rows_used": int(len(selected_indices)),
        "validation_rows": int(len(validation["labels"])),
        "state_sha256": state_hash,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(target_device),
        },
    }
    freeze_audit = {
        "frozen_at_utc": _utc_now(),
        "test_loaded_before_freeze": False,
        "test_metrics_used_for_selection": False,
        "training_date_range": ranges["train"],
        "validation_date_range": ranges["val"],
        "test_date_range_reserved": ranges["test"],
        "lambda_candidates": candidates,
        "selected_magnitude_lambda": selected_lambda,
        "validation_threshold_candidates": threshold_table.to_dict(orient="records"),
        "selected_opening_threshold_bps": opening_thresholds,
        "selected_probability_gap_threshold": probability_gap_thresholds,
        "opening_threshold_floor_one_way_bps": float(base_cost_bps),
        "feature_scaler_fit_rows": int(len(train["labels"])),
        "magnitude_transform_fit_rows": int(magnitude_transform["fit_sample_n"]),
        "state_sha256": state_hash,
    }
    # This file and the immutable model are persisted before _load_raw_split is
    # ever called with the test date range.
    _json_dump(target / "selection_frozen_before_test.json", freeze_audit)
    torch.save(
        {"state_dict": selected_state, "config": config}, target / "model.pt"
    )
    pd.DataFrame(candidates).to_csv(target / "lambda_selection.csv", index=False)
    threshold_table.to_csv(target / "validation_threshold_selection.csv", index=False)
    _json_dump(target / "history.json", histories)
    validation_frame.to_csv(target / "validation_predictions.csv", index=False)

    del train_loader, val_loader, train, validation, train_magnitude, validation_magnitude
    del states
    gc.collect()
    print("selection audit is frozen; loading test split for the first time")
    test = _load_raw_split(stocks, ranges["test"], seq_len, minute_dir)
    test_x = _transform_features(
        test["x"], test["stock_ids"], feature_mean, feature_std, True
    )
    test_magnitude = transform_magnitude_targets(
        test["absolute_return_bps"], test["stock_ids"], magnitude_transform
    )
    test_loader = _loader(
        test_x,
        test["labels"],
        test_magnitude,
        test["stock_ids"],
        test["signed_return_bps"],
        batch_size,
        False,
        seed,
    )
    test_prediction = _predict_loader(
        selected_model, test_loader, magnitude_transform, target_device
    )
    test_metrics = prediction_metrics(
        test_prediction["labels"],
        test_prediction["probability"],
        test_prediction["predicted_absolute_bps"],
        test_prediction["expected_return_bps"],
        test_prediction["signed_return_bps"],
    )
    test_metrics["coverage"] = test["coverage"]
    test_frame = _prediction_frame(test["metadata"], test_prediction)
    test_frame.to_csv(target / "test_predictions.csv", index=False)
    _json_dump(target / "test_metrics.json", test_metrics)

    # Strict replay from the saved bundle and raw minute files.
    replay = evaluate_saved_return_model(
        target / "model.pt", minute_dir, "test", batch_size, target_device
    )
    replay_frame = replay["predictions"]
    if not test_frame[["stock", "window_end", "target_time"]].equals(
        replay_frame[["stock", "window_end", "target_time"]]
    ):
        raise RuntimeError("saved return-LSTM replay changed test sample alignment")
    comparison_columns = (
        "prob_down",
        "prob_flat",
        "prob_up",
        "predicted_abs_return_bps",
        "expected_return_bps",
    )
    maximum_difference = max(
        float(
            np.max(
                np.abs(
                    test_frame[column].to_numpy(dtype=float)
                    - replay_frame[column].to_numpy(dtype=float)
                )
            )
        )
        for column in comparison_columns
    )
    replay_audit = {
        "passed": bool(maximum_difference <= 1e-6),
        "rows": int(len(test_frame)),
        "maximum_absolute_difference": maximum_difference,
        "columns": list(comparison_columns),
        "state_sha256": state_hash,
    }
    if not replay_audit["passed"]:
        raise RuntimeError(f"strict return-LSTM replay failed: {replay_audit}")
    _json_dump(target / "replay_audit.json", replay_audit)

    strategy_root = target / "strategies"
    strategy_results: list[pd.DataFrame] = []
    for signal_mode in ("expected_return", "probability_gap"):
        for side in ("long_short", "long_only"):
            threshold = (
                opening_thresholds[side]
                if signal_mode == "expected_return"
                else probability_gap_thresholds[side]
            )
            strategy_dir = strategy_root / f"{signal_mode}_{side}"
            result = run_strategy_suite(
                target / "test_predictions.csv",
                minute_dir / "close",
                strategy_dir,
                tiers=("all",),
                weightings=("confidence",),
                score_threshold=threshold,
                require_directional_argmax=True,
                side=side,
                signal_mode=signal_mode,
                base_cost_bps=base_cost_bps,
                cost_grid_bps=cost_grid_bps,
                overwrite=overwrite,
            )
            summary = result["summary"].copy()
            summary.insert(0, "comparison_signal", signal_mode)
            summary["validation_frozen_score_threshold"] = threshold
            summary["validation_frozen_threshold_unit"] = (
                "bps" if signal_mode == "expected_return" else "probability_gap"
            )
            summary["validation_frozen_threshold_bps"] = (
                threshold if signal_mode == "expected_return" else np.nan
            )
            strategy_results.append(summary)
    strategy_comparison = pd.concat(strategy_results, ignore_index=True)
    strategy_comparison.to_csv(target / "strategy_comparison.csv", index=False)

    return {
        "out_dir": str(target),
        "test_metrics": test_metrics,
        "strategy_comparison": strategy_comparison,
        "freeze_audit": freeze_audit,
        "replay_audit": replay_audit,
    }
