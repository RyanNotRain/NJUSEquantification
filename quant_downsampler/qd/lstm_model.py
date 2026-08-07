"""Leakage-safe LSTM utilities for next-minute price classification."""

from __future__ import annotations

import copy
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR

CHANNELS = (
    "open", "high", "low", "close", "volume", "trade_count", "amount",
    "buy_volume", "sell_volume", "buy_amount", "sell_amount",
)
LEGACY_FEATURE_NAMES = (
    "log_open_to_close", "log_high_to_close", "log_low_to_close",
    "close_log_return", "log1p_volume", "log1p_trade_count",
    "log1p_amount", "log1p_buy_volume", "log1p_sell_volume",
    "log1p_buy_amount", "log1p_sell_amount",
)
ENHANCED_FEATURE_NAMES = LEGACY_FEATURE_NAMES + (
    "log_high_to_low", "close_location", "buy_volume_imbalance",
    "buy_amount_imbalance", "log1p_average_trade_size",
    "volume_log_change", "amount_log_change", "close_log_return_3m",
    "close_log_return_5m", "close_log_return_10m", "return_volatility_5m",
    "has_trade", "session_progress", "is_afternoon",
)
# Backward-compatible public name for previously saved 11-feature models.
FEATURE_NAMES = LEGACY_FEATURE_NAMES
DEFAULT_SPLITS = {
    "train": ("20250401", "20260531"),
    "val": ("20260601", "20260615"),
    "test": ("20260616", "20260630"),
}
TARGET_CLASS_NAMES = {
    "nonflat_binary": ("down", "up"),
    "three_class": ("down", "flat", "up"),
    "up_vs_not_up": ("not_up", "up"),
    "move_vs_flat": ("flat", "move"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _available_dates(data_dir: Path, date_range: tuple[str, str]) -> list[str]:
    start, end = date_range
    dates = sorted(p.stem for p in (data_dir / "close").glob("*.csv"))
    return [d for d in dates if start <= d <= end]


def load_minute_data_for_stocks(
    stock_codes: list[str],
    date_range: tuple[str, str],
    data_dir: Path | None = None,
) -> dict[str, list[np.ndarray]]:
    """Read every channel once per date and return per-stock day arrays."""
    root = Path(data_dir or (OUTPUT_DIR / "minute"))
    result: dict[str, list[np.ndarray]] = {stock: [] for stock in stock_codes}
    for date in _available_dates(root, date_range):
        channel_tables = []
        reference_index = None
        for channel in CHANNELS:
            path = root / channel / f"{date}.csv"
            table = pd.read_csv(
                path,
                index_col=0,
                usecols=lambda column: column == "datetime" or column in stock_codes,
            )
            if len(table) != 242:
                raise ValueError(f"{path} has {len(table)} rows, expected 242")
            if set(table.columns) != set(stock_codes):
                raise ValueError(
                    f"{path} stock columns do not match the requested set: "
                    f"expected {stock_codes}, got {table.columns.tolist()}"
                )
            table = table.loc[:, stock_codes]
            if reference_index is None:
                reference_index = table.index
            elif not table.index.equals(reference_index):
                raise ValueError(f"{path} datetime index is misaligned with the reference channel")
            channel_tables.append(table)
        for stock in stock_codes:
            result[stock].append(np.column_stack([
                table[stock].to_numpy(dtype=np.float64) for table in channel_tables
            ]))
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def _session_slices() -> tuple[tuple[int, int], ...]:
    # Afternoon learning samples stop before the 14:57--15:00 call-auction rows.
    return ((0, 121), (121, 238))


def feature_names(feature_set: str = "enhanced") -> tuple[str, ...]:
    if feature_set == "legacy":
        return LEGACY_FEATURE_NAMES
    if feature_set == "enhanced":
        return ENHANCED_FEATURE_NAMES
    raise ValueError(f"unknown feature_set={feature_set!r}")


def target_class_names(target_mode: str) -> tuple[str, ...]:
    try:
        return TARGET_CLASS_NAMES[target_mode]
    except KeyError as exc:
        raise ValueError(f"unknown target_mode={target_mode!r}") from exc


def engineer_features(day: np.ndarray, feature_set: str = "enhanced") -> np.ndarray:
    """Convert raw levels to past-only stationary microstructure features."""
    x = np.asarray(day, dtype=np.float64)
    close = x[:, 3]
    base = np.full_like(x, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        base[:, 0] = np.log(x[:, 0] / close)
        base[:, 1] = np.log(x[:, 1] / close)
        base[:, 2] = np.log(x[:, 2] / close)
        base[1:, 3] = np.log(close[1:] / close[:-1])
    base[0, 3] = 0.0 if np.isfinite(close[0]) else np.nan
    base[:, 4:] = np.log1p(np.maximum(x[:, 4:], 0.0))
    base[~np.isfinite(base)] = np.nan
    if feature_set == "legacy":
        return base
    if feature_set != "enhanced":
        raise ValueError(f"unknown feature_set={feature_set!r}")

    # The first afternoon return must not reach across the lunch break.
    base[121, 3] = 0.0 if np.isfinite(close[121]) else np.nan

    extra = np.full((len(x), len(ENHANCED_FEATURE_NAMES) - len(base[0])), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        extra[:, 0] = np.log(x[:, 1] / x[:, 2])
    price_range = x[:, 1] - x[:, 2]
    extra[:, 1] = 2.0 * _safe_ratio(close - x[:, 2], price_range) - 1.0
    extra[price_range == 0, 1] = 0.0
    extra[:, 2] = _safe_ratio(x[:, 7] - x[:, 8], x[:, 7] + x[:, 8])
    extra[:, 3] = _safe_ratio(x[:, 9] - x[:, 10], x[:, 9] + x[:, 10])
    extra[:, 4] = np.log1p(_safe_ratio(np.maximum(x[:, 4], 0), np.maximum(x[:, 5], 0)))
    extra[:, 11] = (x[:, 5] > 0).astype(np.float64)
    extra[:, 13] = 0.0

    for session_id, (start, stop) in enumerate(_session_slices()):
        length = stop - start
        extra[start:stop, 12] = np.linspace(0.0, 1.0, length)
        extra[start:stop, 13] = float(session_id)
        log_volume = np.log1p(np.maximum(x[start:stop, 4], 0.0))
        log_amount = np.log1p(np.maximum(x[start:stop, 6], 0.0))
        extra[start, 5:7] = 0.0
        extra[start + 1:stop, 5] = np.diff(log_volume)
        extra[start + 1:stop, 6] = np.diff(log_amount)
        for column, lag in ((7, 3), (8, 5), (9, 10)):
            extra[start:start + lag, column] = 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                extra[start + lag:stop, column] = np.log(
                    close[start + lag:stop] / close[start:stop - lag]
                )
        returns = base[start:stop, 3]
        rolling = pd.Series(returns).rolling(5, min_periods=2).std(ddof=0).fillna(0.0)
        extra[start:stop, 10] = rolling.to_numpy(dtype=np.float64)

    # Rows outside the learning sessions are intentionally unused.
    extra[~np.isfinite(extra)] = np.nan
    return np.column_stack([base, extra])


def create_sequences(
    days: list[np.ndarray] | np.ndarray,
    seq_len: int = 60,
    feature_set: str = "enhanced",
    target_mode: str = "nonflat_binary",
) -> tuple[np.ndarray, np.ndarray]:
    """Use a window ending at minute t to predict close[t+1] vs close[t].

    In ``nonflat_binary`` mode, flat targets are excluded rather than
    mislabeled as down.  ``three_class`` keeps every valid down/flat/up target.
    Windows never cross trading days or the lunch break.
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    for window, label, _, _ in _iter_sequence_samples(
        days, seq_len, feature_set, target_mode
    ):
        X.append(window)
        y.append(label)
    if not X:
        return (
            np.empty((0, seq_len, len(feature_names(feature_set))), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def _iter_sequence_samples(
    days: list[np.ndarray] | np.ndarray,
    seq_len: int,
    feature_set: str,
    target_mode: str,
):
    target_class_names(target_mode)
    iterable = list(days) if isinstance(days, np.ndarray) else days
    for day_id, day in enumerate(iterable):
        features = engineer_features(day, feature_set=feature_set)
        close = day[:, 3]
        for start, stop in _session_slices():
            for end in range(start + seq_len - 1, stop - 1):
                current, following = close[end], close[end + 1]
                if not np.isfinite(current) or not np.isfinite(following):
                    continue
                delta = following - current
                if target_mode == "nonflat_binary":
                    if delta == 0:
                        continue
                    label = int(delta > 0)
                elif target_mode == "three_class":
                    label = 0 if delta < 0 else (2 if delta > 0 else 1)
                elif target_mode == "up_vs_not_up":
                    label = int(delta > 0)
                else:  # move_vs_flat, validated above
                    label = int(delta != 0)
                window = features[end - seq_len + 1:end + 1]
                if np.isfinite(window).all():
                    yield window, label, day_id, end


def _row_timestamp(date: str, row: int) -> pd.Timestamp:
    day = pd.Timestamp(date)
    if row <= 120:
        return day + pd.Timedelta(hours=9, minutes=30 + row)
    return day + pd.Timedelta(hours=13, minutes=row - 121)


def create_sequences_with_metadata(
    days: list[np.ndarray] | np.ndarray,
    dates: list[str],
    seq_len: int = 60,
    feature_set: str = "enhanced",
    target_mode: str = "nonflat_binary",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create sequences plus auditable date/window-end/target timestamps."""
    iterable = list(days) if isinstance(days, np.ndarray) else days
    if len(iterable) != len(dates):
        raise ValueError("dates must align one-to-one with day arrays")
    X: list[np.ndarray] = []
    y: list[int] = []
    records: list[dict[str, str]] = []
    for window, label, day_id, end in _iter_sequence_samples(
        iterable, seq_len, feature_set, target_mode
    ):
        window_end = _row_timestamp(dates[day_id], end)
        target_time = _row_timestamp(dates[day_id], end + 1)
        X.append(window)
        y.append(label)
        records.append({
            "date": window_end.strftime("%Y-%m-%d"),
            "window_end": window_end.strftime("%Y-%m-%d %H:%M:%S"),
            "target_time": target_time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    n_features = len(feature_names(feature_set))
    x_array = (
        np.asarray(X, dtype=np.float32)
        if X else np.empty((0, seq_len, n_features), dtype=np.float32)
    )
    return x_array, np.asarray(y, dtype=np.int64), pd.DataFrame(
        records, columns=["date", "window_end", "target_time"]
    )


def target_coverage(
    days: list[np.ndarray] | np.ndarray,
    seq_len: int = 60,
    feature_set: str = "enhanced",
) -> dict[str, int | float]:
    """Count valid prediction times before conditioning on a non-flat target."""
    iterable = list(days) if isinstance(days, np.ndarray) else days
    total = flat = up = down = 0
    for day in iterable:
        features = engineer_features(day, feature_set=feature_set)
        close = day[:, 3]
        for start, stop in _session_slices():
            for end in range(start + seq_len - 1, stop - 1):
                current, following = close[end], close[end + 1]
                window = features[end - seq_len + 1:end + 1]
                if (
                    not np.isfinite(current)
                    or not np.isfinite(following)
                    or not np.isfinite(window).all()
                ):
                    continue
                total += 1
                delta = following - current
                if delta > 0:
                    up += 1
                elif delta < 0:
                    down += 1
                else:
                    flat += 1
    nonflat = up + down
    return {
        "all_valid_windows": total,
        "nonflat_windows": nonflat,
        "flat_windows": flat,
        "up_windows": up,
        "down_windows": down,
        "nonflat_coverage": float(nonflat / total) if total else 0.0,
        "flat_rate": float(flat / total) if total else 0.0,
    }


class MinuteLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = len(CHANNELS),
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        model_version: str = "legacy",
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if model_version == "legacy":
            self.recent = None
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )
        elif model_version == "residual":
            recent_size = max(16, hidden_size // 2)
            self.recent = nn.Sequential(
                nn.LayerNorm(input_size),
                nn.Linear(input_size, recent_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size + recent_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size + recent_size, num_classes),
            )
        else:
            raise ValueError(f"unknown model_version={model_version!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        encoded = output[:, -1]
        if self.recent is not None:
            encoded = torch.cat([encoded, self.recent(x[:, -1])], dim=1)
        return self.head(encoded)


def _make_split(
    stock_codes: list[str],
    date_range: tuple[str, str],
    seq_len: int,
    data_dir: Path | None,
    feature_set: str,
    target_mode: str,
    return_metadata: bool = False,
) -> tuple:
    root = Path(data_dir or (OUTPUT_DIR / "minute"))
    dates = _available_dates(root, date_range)
    raw = load_minute_data_for_stocks(stock_codes, date_range, root)
    arrays = []
    metadata_parts = []
    for stock_id, stock in enumerate(stock_codes):
        if return_metadata:
            X, y, metadata = create_sequences_with_metadata(
                raw[stock], dates, seq_len,
                feature_set=feature_set, target_mode=target_mode,
            )
            metadata.insert(0, "stock", stock)
            metadata.insert(1, "stock_id", stock_id)
            metadata_parts.append(metadata)
        else:
            X, y = create_sequences(
                raw[stock], seq_len, feature_set=feature_set, target_mode=target_mode
            )
        arrays.append((X, y))
    coverage_parts = [
        target_coverage(raw[stock], seq_len, feature_set=feature_set)
        for stock in stock_codes
    ]
    coverage = {
        key: int(sum(int(part[key]) for part in coverage_parts))
        for key in ("all_valid_windows", "nonflat_windows", "flat_windows", "up_windows", "down_windows")
    }
    total = coverage["all_valid_windows"]
    coverage["nonflat_coverage"] = (
        float(coverage["nonflat_windows"] / total) if total else 0.0
    )
    coverage["flat_rate"] = float(coverage["flat_windows"] / total) if total else 0.0
    X_parts = [x for x, _ in arrays if len(x)]
    y_parts = [y for _, y in arrays if len(y)]
    if not X_parts:
        n_features = len(feature_names(feature_set))
        empty = (
            np.empty((0, seq_len, n_features), np.float32),
            np.empty(0, np.int64),
            np.empty(0, np.int64),
            coverage,
        )
        if return_metadata:
            return (*empty, pd.DataFrame(
                columns=["stock", "stock_id", "date", "window_end", "target_time"]
            ))
        return empty
    stock_ids = np.concatenate([
        np.full(len(y), stock_id, dtype=np.int64)
        for stock_id, (_, y) in enumerate(arrays) if len(y)
    ])
    result = (np.concatenate(X_parts), np.concatenate(y_parts), stock_ids, coverage)
    if return_metadata:
        return (*result, pd.concat(metadata_parts, ignore_index=True))
    return result


def prepare_dataloaders(
    stock_codes: list[str],
    seq_len: int = 60,
    batch_size: int = 256,
    data_dir: Path | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
    feature_set: str = "enhanced",
    include_stock_id: bool = True,
    scaler_mode: str = "per_stock",
    seed: int = 42,
    target_mode: str = "nonflat_binary",
) -> dict:
    ranges = splits or DEFAULT_SPLITS
    for name in ("train", "val", "test"):
        if name not in ranges or ranges[name][0] > ranges[name][1]:
            raise ValueError(f"invalid {name} split: {ranges.get(name)}")
    if not (ranges["train"][1] < ranges["val"][0] <= ranges["val"][1] < ranges["test"][0]):
        raise ValueError("splits must be strictly ordered and non-overlapping")
    X_train, y_train, train_stock_ids, train_coverage, train_metadata = _make_split(
        stock_codes, ranges["train"], seq_len, data_dir, feature_set, target_mode, True
    )
    X_val, y_val, val_stock_ids, val_coverage, val_metadata = _make_split(
        stock_codes, ranges["val"], seq_len, data_dir, feature_set, target_mode, True
    )
    X_test, y_test, test_stock_ids, test_coverage, test_metadata = _make_split(
        stock_codes, ranges["test"], seq_len, data_dir, feature_set, target_mode, True
    )
    if not len(X_train) or not len(X_val) or not len(X_test):
        raise ValueError("one or more LSTM data splits are empty")

    if scaler_mode == "global":
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True)
        std[std < 1e-8] = 1.0
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        saved_mean = mean.squeeze().astype(float)
        saved_std = std.squeeze().astype(float)
    elif scaler_mode == "per_stock":
        means, stds = [], []
        for stock_id in range(len(stock_codes)):
            stock_train = X_train[train_stock_ids == stock_id]
            if not len(stock_train):
                raise ValueError(f"no training sequences for stock {stock_codes[stock_id]}")
            stock_mean = stock_train.mean(axis=(0, 1), keepdims=True)
            stock_std = stock_train.std(axis=(0, 1), keepdims=True)
            stock_std[stock_std < 1e-8] = 1.0
            means.append(stock_mean)
            stds.append(stock_std)
        for X, ids in (
            (X_train, train_stock_ids), (X_val, val_stock_ids), (X_test, test_stock_ids)
        ):
            for stock_id, (stock_mean, stock_std) in enumerate(zip(means, stds)):
                mask = ids == stock_id
                X[mask] = (X[mask] - stock_mean) / stock_std
        saved_mean = np.stack([x.squeeze() for x in means]).astype(float)
        saved_std = np.stack([x.squeeze() for x in stds]).astype(float)
    else:
        raise ValueError(f"unknown scaler_mode={scaler_mode!r}")

    if include_stock_id:
        eye = np.eye(len(stock_codes), dtype=np.float32)
        X_train = np.concatenate([
            X_train,
            np.broadcast_to(eye[train_stock_ids, None, :], (len(X_train), seq_len, len(stock_codes))),
        ], axis=2)
        X_val = np.concatenate([
            X_val,
            np.broadcast_to(eye[val_stock_ids, None, :], (len(X_val), seq_len, len(stock_codes))),
        ], axis=2)
        X_test = np.concatenate([
            X_test,
            np.broadcast_to(eye[test_stock_ids, None, :], (len(X_test), seq_len, len(stock_codes))),
        ], axis=2)

    loaders = {}
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    for name, X, y, shuffle in (
        ("train", X_train, y_train, True),
        ("val", X_val, y_val, False),
        ("test", X_test, y_test, False),
    ):
        loaders[name] = DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=batch_size, shuffle=shuffle,
            generator=train_generator if shuffle else None,
        )
    return {
        "loaders": loaders,
        "mean": saved_mean,
        "std": saved_std,
        "feature_names": list(feature_names(feature_set)) + (
            [f"stock_id::{stock}" for stock in stock_codes] if include_stock_id else []
        ),
        "sizes": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "class_rates": {
            name: (
                np.bincount(y, minlength=(3 if target_mode == "three_class" else 2)).astype(float)
                / len(y)
            ).tolist()
            for name, y in (("train", y_train), ("val", y_val), ("test", y_test))
        },
        "positive_rates": {
            "train": float((y_train == (2 if target_mode == "three_class" else 1)).mean()),
            "val": float((y_val == (2 if target_mode == "three_class" else 1)).mean()),
            "test": float((y_test == (2 if target_mode == "three_class" else 1)).mean()),
        },
        "coverage": {
            "train": train_coverage,
            "val": val_coverage,
            "test": test_coverage,
        },
        "train_labels": y_train,
        "val_labels": y_val,
        "test_labels": y_test,
        "train_stock_ids": train_stock_ids,
        "val_stock_ids": val_stock_ids,
        "test_stock_ids": test_stock_ids,
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
        "test_metadata": test_metadata,
    }


def predict_class_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, actual = [], []
    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            actual.append(y.numpy())
    return np.concatenate(probabilities), np.concatenate(actual)


def predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible positive-class probabilities for binary models."""
    probability, actual = predict_class_probabilities(model, loader, device)
    if probability.shape[1] != 2:
        raise ValueError("predict_probabilities requires a binary model")
    return probability[:, 1], actual


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None = 0.5,
) -> dict[str, float | list]:
    probability, true = predict_class_probabilities(model, loader, device)
    return classification_metrics(true, probability, threshold)


def classification_metrics(
    true: np.ndarray,
    probability: np.ndarray,
    threshold: float | None = 0.5,
) -> dict[str, float | int | list | None]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    n_classes = probability.shape[1]
    pred = (
        (probability[:, 1] >= float(threshold)).astype(np.int64)
        if n_classes == 2 else probability.argmax(axis=1)
    )
    average = "binary" if n_classes == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision": float(precision_score(true, pred, average=average, zero_division=0)),
        "recall": float(recall_score(true, pred, average=average, zero_division=0)),
        "f1": float(f1_score(true, pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(range(n_classes))).tolist(),
        "threshold": float(threshold) if n_classes == 2 else None,
        "n_classes": int(n_classes),
    }


def transform_sequences(
    sequences: np.ndarray,
    stock_ids: np.ndarray,
    config: dict,
    stock_codes: list[str] | None = None,
) -> np.ndarray:
    """Apply a saved train-only scaler and optional stock one-hot features."""
    x = np.asarray(sequences, dtype=np.float32).copy()
    ids = np.asarray(stock_ids, dtype=np.int64)
    stocks = list(stock_codes or config["stock_codes"])
    if x.ndim != 3 or len(x) != len(ids):
        raise ValueError("sequences and stock_ids are not aligned")
    if x.shape[1] != int(config["seq_len"]):
        raise ValueError(
            f"expected sequence length {config['seq_len']}, got {x.shape[1]}"
        )
    if len(ids) and (ids.min() < 0 or ids.max() >= len(stocks)):
        raise ValueError("stock_ids contain an out-of-range value")
    include_stock_id = bool(config["include_stock_id"])
    input_size = int(config.get("input_size", len(config["feature_names"])))
    base_size = input_size - (len(stocks) if include_stock_id else 0)
    if x.shape[2] != base_size:
        raise ValueError(f"expected {base_size} raw features, got {x.shape[2]}")

    mean = np.asarray(config["scaler_mean"], dtype=np.float32)
    std = np.asarray(config["scaler_std"], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("saved scaler contains invalid values")
    if config["scaler_mode"] == "global":
        if mean.shape != (base_size,) or std.shape != (base_size,):
            raise ValueError("global scaler shape does not match the model")
        x = (x - mean[None, None, :]) / std[None, None, :]
    elif config["scaler_mode"] == "per_stock":
        expected = (len(stocks), base_size)
        if mean.shape != expected or std.shape != expected:
            raise ValueError("per-stock scaler shape does not match the model")
        x = (x - mean[ids, None, :]) / std[ids, None, :]
    else:
        raise ValueError(f"unknown scaler_mode={config['scaler_mode']!r}")

    if include_stock_id:
        one_hot = np.eye(len(stocks), dtype=np.float32)[ids]
        one_hot = np.broadcast_to(
            one_hot[:, None, :], (len(x), x.shape[1], len(stocks))
        )
        x = np.concatenate((x, one_hot), axis=2)
    return x.astype(np.float32, copy=False)


def load_saved_lstm(
    model_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[MinuteLSTM, dict]:
    """Strictly rebuild a single saved LSTM and return its frozen config."""
    target_device = torch.device(device)
    bundle = torch.load(Path(model_path), map_location=target_device, weights_only=False)
    if set(bundle) != {"state_dict", "config"}:
        raise ValueError("saved LSTM bundle must contain exactly state_dict and config")
    config = bundle["config"]
    required = {
        "feature_names", "hidden_size", "num_layers", "dropout",
        "model_version", "num_classes", "class_names", "seq_len", "target_mode",
        "stock_codes", "splits", "include_stock_id", "scaler_mode",
        "scaler_mean", "scaler_std",
    }
    if not required.issubset(config):
        raise ValueError("saved LSTM config is incomplete")
    model = MinuteLSTM(
        input_size=len(config["feature_names"]),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        model_version=str(config["model_version"]),
        num_classes=int(config["num_classes"]),
    )
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(target_device).eval()
    return model, config


def evaluate_saved_lstm(
    model_path: str | Path,
    data_dir: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict:
    """Replay a saved single-model artifact from raw minute tables."""
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    model, config = load_saved_lstm(model_path, device=device)
    stock_codes = list(config["stock_codes"])
    x, labels, stock_ids, coverage, metadata = _make_split(
        stock_codes,
        tuple(config["splits"][split]),
        int(config["seq_len"]),
        Path(data_dir),
        str(config["feature_set"]),
        str(config["target_mode"]),
        True,
    )
    x = transform_sequences(x, stock_ids, config, stock_codes)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=False,
    )
    probability, replayed_labels = predict_class_probabilities(
        model, loader, torch.device(device)
    )
    if not np.array_equal(labels, replayed_labels):
        raise RuntimeError("replayed labels changed inside the data loader")
    metrics = classification_metrics(
        labels, probability, config.get("decision_threshold")
    )
    counts = np.bincount(labels, minlength=int(config["num_classes"]))
    metrics["majority_baseline"] = float(counts.max() / counts.sum())
    metrics["coverage"] = coverage
    return {
        "metrics": metrics,
        "probability": probability,
        "labels": labels,
        "stock_ids": stock_ids,
        "metadata": metadata,
    }


def select_accuracy_threshold(
    probability: np.ndarray,
    true: np.ndarray,
    minimum_gain: float = 0.003,
) -> float:
    """Select a validation threshold only when it materially beats 0.5."""
    thresholds = np.linspace(0.35, 0.65, 301)
    accuracy = np.array([np.mean((probability >= threshold) == true) for threshold in thresholds])
    neutral_accuracy = float(np.mean((probability >= 0.5) == true))
    if float(accuracy.max()) < neutral_accuracy + minimum_gain:
        return 0.5
    best = np.flatnonzero(accuracy == accuracy.max())
    # Prefer the maximizer closest to the neutral 0.5 threshold.
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    train_labels: np.ndarray,
    epochs: int,
    device: torch.device,
    patience: int = 4,
    class_weighted: bool = True,
    label_smoothing: float = 0.0,
    learning_rate: float = 1e-3,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    num_classes = int(model.head[-1].out_features)
    counts = np.bincount(train_labels, minlength=num_classes)
    weights = len(train_labels) / (len(counts) * np.maximum(counts, 1))
    weight_tensor = (
        torch.tensor(weights, dtype=torch.float32, device=device) if class_weighted else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    history = {"train_loss": [], "val_accuracy": []}
    best_accuracy = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    model.to(device)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for X, y in loaders["train"]:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(X)
        loss_value = loss_sum / len(loaders["train"].dataset)
        val = evaluate(model, loaders["val"], device)
        val_accuracy = float(val["accuracy"])
        history["train_loss"].append(loss_value)
        history["val_accuracy"].append(val_accuracy)
        scheduler.step(val_accuracy)
        print(f"epoch {epoch + 1:02d}: loss={loss_value:.5f} val_acc={val_accuracy:.4f}")
        if val_accuracy > best_accuracy + 1e-5:
            best_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("early stopping")
                break
    return history, best_state


def _plot_history(history: dict[str, list[float]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Training loss")
    axes[1].plot(history["val_accuracy"])
    axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "training_history.png", dpi=160)
    plt.close(fig)


def _write_report(result: dict, out_dir: Path) -> None:
    m = result["test_metrics"]
    cfg = result["config"]
    sizes = cfg["sizes"]
    class_names = cfg["class_names"]
    class_rates = cfg["class_rates"]
    coverage = cfg["coverage"]["test"]
    conditional = cfg["target_mode"] == "nonflat_binary"
    lines = [
        "# LSTM 下一分钟分类报告",
        "",
        "本报告由当前模型训练脚本自动生成。输入窗口结束于分钟 t，标签严格比较 `close[t+1]` 与 `close[t]`。",
        "",
        "## 设置",
        "",
        f"- 股票：{', '.join(cfg['stock_codes'])}",
        f"- 特征：{len(cfg['feature_names'])} 个过去信息特征（feature_set={cfg['feature_set']}）",
        f"- 序列长度：{cfg['seq_len']} 分钟",
        f"- 模型：{cfg['num_layers']} 层 LSTM，hidden_size={cfg['hidden_size']}，model_version={cfg['model_version']}，验证集早停",
        f"- 标准化：{cfg['scaler_mode']}；股票标识特征：{cfg['include_stock_id']}；类别加权：{cfg['class_weighted']}",
        f"- 目标：{cfg['target_mode']}；类别数：{cfg['num_classes']}",
        f"- 样本数：train={sizes['train']:,}，val={sizes['val']:,}，test={sizes['test']:,}",
        "- 类别比例：" + "；".join(
            f"{split}=" + "/".join(
                f"{name} {rate:.2%}" for name, rate in zip(class_names, class_rates[split])
            )
            for split in ("train", "val", "test")
        ),
        f"- 测试覆盖：非平价 {coverage['nonflat_windows']:,}/{coverage['all_valid_windows']:,} = {coverage['nonflat_coverage']:.2%}；平价率 {coverage['flat_rate']:.2%}",
        "",
        "## 测试集",
        "",
        f"- {'Conditional non-flat accuracy' if conditional else 'All-window accuracy'}: {m['accuracy']:.4f}",
        f"- Majority baseline: {m['majority_baseline']:.4f}",
        f"- Precision: {m['precision']:.4f}",
        f"- Recall: {m['recall']:.4f}",
        f"- F1: {m['f1']:.4f}",
        f"- Macro stock accuracy: {m['macro_stock_accuracy']:.4f}",
        f"- Decision threshold: {m['threshold']:.3f}" if m["threshold"] is not None else "- Decision rule: three-class argmax",
        f"- Confusion matrix ({'/'.join(class_names)}): {m['confusion_matrix']}",
        "- Per-stock accuracy: " + ", ".join(
            f"{stock}={values['accuracy']:.2%}" for stock, values in m["per_stock"].items()
        ),
        "",
        "`model.pt` 同时保存模型结构参数、权重、标准化均值/方差、股票、特征顺序、日期切分和随机种子，可直接重载推理。",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_lstm_pipeline(
    stock_codes: list[str] | None = None,
    n_stocks: int = 5,
    seq_len: int = 60,
    hidden_size: int = 64,
    epochs: int = 12,
    batch_size: int = 256,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    seed: int = 42,
    feature_set: str = "enhanced",
    include_stock_id: bool = True,
    scaler_mode: str = "per_stock",
    model_version: str = "residual",
    num_layers: int = 2,
    dropout: float = 0.2,
    class_weighted: bool = True,
    label_smoothing: float = 0.0,
    learning_rate: float = 1e-3,
    calibrate_threshold: bool = True,
    target_mode: str = "nonflat_binary",
    overwrite: bool = False,
) -> dict:
    set_seed(seed)
    minute_dir = Path(data_dir or (OUTPUT_DIR / "minute"))
    target = Path(out_dir or (OUTPUT_DIR / "lstm"))
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty LSTM output directory: {target}; "
            "choose another out_dir or set overwrite=True"
        )
    target.mkdir(parents=True, exist_ok=True)
    if stock_codes is None:
        first = sorted((minute_dir / "close").glob("*.csv"))[0]
        stock_codes = pd.read_csv(first, nrows=1, index_col=0).columns[:n_stocks].tolist()
    data = prepare_dataloaders(
        stock_codes,
        seq_len,
        batch_size,
        minute_dir,
        feature_set=feature_set,
        include_stock_id=include_stock_id,
        scaler_mode=scaler_mode,
        seed=seed,
        target_mode=target_mode,
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device}; stocks={stock_codes}; sizes={data['sizes']}")
    class_names = target_class_names(target_mode)
    model = MinuteLSTM(
        input_size=len(data["feature_names"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_version=model_version,
        num_classes=len(class_names),
    )
    history, best_state = train_model(
        model,
        data["loaders"],
        data["train_labels"],
        epochs,
        device,
        class_weighted=class_weighted,
        label_smoothing=label_smoothing,
        learning_rate=learning_rate,
    )
    model.load_state_dict(best_state)
    model.to(device)
    if target_mode == "three_class":
        threshold = None
    else:
        val_probability, val_labels = predict_probabilities(
            model, data["loaders"]["val"], device
        )
        threshold = (
            select_accuracy_threshold(val_probability, val_labels)
            if calibrate_threshold else 0.5
        )
    test_probability, test_labels = predict_class_probabilities(
        model, data["loaders"]["test"], device
    )
    test_metrics = classification_metrics(test_labels, test_probability, threshold)
    test_counts = np.bincount(test_labels)
    test_metrics["majority_baseline"] = float(test_counts.max() / test_counts.sum())
    per_stock = {}
    for stock_id, stock in enumerate(stock_codes):
        mask = data["test_stock_ids"] == stock_id
        stock_metrics = classification_metrics(
            test_labels[mask], test_probability[mask], threshold
        )
        per_stock[stock] = {
            "n": int(mask.sum()),
            "accuracy": stock_metrics["accuracy"],
            "f1": stock_metrics["f1"],
        }
    test_metrics["per_stock"] = per_stock
    test_metrics["macro_stock_accuracy"] = float(
        np.mean([metrics["accuracy"] for metrics in per_stock.values()])
    )

    config = {
        "stock_codes": stock_codes,
        "channels": list(CHANNELS),
        "feature_names": data["feature_names"],
        "feature_set": feature_set,
        "target_mode": target_mode,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "include_stock_id": include_stock_id,
        "scaler_mode": scaler_mode,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "model_version": model_version,
        "class_weighted": class_weighted,
        "label_smoothing": label_smoothing,
        "learning_rate": learning_rate,
        "decision_threshold": threshold,
        "epochs_requested": epochs,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": int(np.argmax(history["val_accuracy"]) + 1),
        "batch_size": batch_size,
        "device": str(device),
        "pipeline_version": 2,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seed": seed,
        "splits": DEFAULT_SPLITS,
        "sizes": data["sizes"],
        "positive_rates": data["positive_rates"],
        "class_rates": data["class_rates"],
        "coverage": data["coverage"],
        "scaler_mean": data["mean"].tolist(),
        "scaler_std": data["std"].tolist(),
    }
    bundle = {"state_dict": {k: v.cpu() for k, v in best_state.items()}, "config": config}
    torch.save(bundle, target / "model.pt")
    (target / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (target / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    predicted = (
        (test_probability[:, 1] >= float(threshold)).astype(np.int64)
        if test_probability.shape[1] == 2 else test_probability.argmax(axis=1)
    )
    prediction_table = data["test_metadata"].drop(columns="stock_id").copy()
    prediction_table["true_label"] = test_labels
    prediction_table["predicted_label"] = predicted
    for class_id, class_name in enumerate(class_names):
        prediction_table[f"prob_{class_name}"] = test_probability[:, class_id]
    prediction_table.to_csv(target / "test_predictions.csv", index=False)
    result = {"model": model, "history": history, "test_metrics": test_metrics, "config": config}
    _plot_history(history, target)
    _write_report(result, target)
    return result
