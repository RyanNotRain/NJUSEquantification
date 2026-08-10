"""Leakage-safe LSTM for next-minute down/flat/up classification.

Flat next-minute returns are first-class training targets.  The primary metrics
therefore cover every valid minute window rather than conditioning on a future
price change.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR, TOTAL_BARS


CHANNELS = (
    "open", "high", "low", "close", "volume", "trade_count", "amount",
    "buy_volume", "sell_volume", "buy_amount", "sell_amount",
)
BASE_FEATURE_NAMES = (
    "log_open_to_close", "log_high_to_close", "log_low_to_close",
    "close_log_return", "log1p_volume", "log1p_trade_count", "log1p_amount",
    "log1p_buy_volume", "log1p_sell_volume", "log1p_buy_amount", "log1p_sell_amount",
)
STATE_FEATURE_NAMES = (
    "flat_fraction_5", "flat_fraction_15", "flat_fraction_60",
    "unchanged_streak", "mean_abs_return_5", "realized_volatility_15",
    "session_sin", "session_cos", "is_afternoon",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + STATE_FEATURE_NAMES
CLASS_NAMES = ("down", "flat", "up")
DEFAULT_SPLITS = {
    "train": ("20250401", "20260531"),
    "validation": ("20260601", "20260615"),
    "test": ("20260616", "20260630"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _available_dates(data_dir: Path, date_range: tuple[str, str]) -> list[str]:
    start, end = date_range
    dates = sorted(path.stem for path in (data_dir / "close").glob("*.csv"))
    return [value for value in dates if start <= value <= end]


def load_minute_data_for_stocks(
    stock_codes: list[str], date_range: tuple[str, str],
    data_dir: Path | None = None,
) -> dict[str, list[np.ndarray]]:
    """Read every channel once per date and return per-stock day arrays."""
    root = Path(data_dir or (OUTPUT_DIR / "minute"))
    result: dict[str, list[np.ndarray]] = {stock: [] for stock in stock_codes}
    for trading_date in _available_dates(root, date_range):
        channel_tables = []
        for channel in CHANNELS:
            path = root / channel / f"{trading_date}.csv"
            table = pd.read_csv(
                path, index_col=0,
                usecols=lambda column: column == "datetime" or column in stock_codes,
            )
            if len(table) != TOTAL_BARS:
                raise ValueError(f"{path} has {len(table)} rows, expected {TOTAL_BARS}")
            missing = set(stock_codes) - set(table.columns)
            if missing:
                raise ValueError(f"{path} is missing selected stocks: {sorted(missing)}")
            channel_tables.append(table)
        for stock in stock_codes:
            result[stock].append(np.column_stack([
                table[stock].to_numpy(dtype=np.float64) for table in channel_tables
            ]))
    return result


def engineer_features(day: np.ndarray, include_state_features: bool = True) -> np.ndarray:
    """Convert raw levels into causal stationary relative/log features."""
    values = np.asarray(day, dtype=np.float64)
    close = values[:, 3]
    output = np.full((len(values), len(FEATURE_NAMES)), np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        output[:, 0] = np.log(values[:, 0] / close)
        output[:, 1] = np.log(values[:, 1] / close)
        output[:, 2] = np.log(values[:, 2] / close)
    output[:, 4:11] = np.log1p(np.maximum(values[:, 4:], 0.0))
    # Reset return and liquidity-state features at each trading session.  This
    # prevents the first afternoon observation from treating the lunch break as
    # an ordinary one-minute transition.
    for session_id, (start, stop) in enumerate(((0, 121), (121, 238))):
        session_close = close[start:stop]
        if not len(session_close):
            continue
        returns = np.zeros(len(session_close), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns[1:] = np.log(session_close[1:] / session_close[:-1])
        output[start:stop, 3] = returns
        flat = np.zeros(len(session_close), dtype=np.float64)
        flat[1:] = (session_close[1:] == session_close[:-1]).astype(np.float64)
        flat_series = pd.Series(flat)
        output[start:stop, 11] = flat_series.rolling(5, min_periods=1).mean().to_numpy()
        output[start:stop, 12] = flat_series.rolling(15, min_periods=1).mean().to_numpy()
        output[start:stop, 13] = flat_series.rolling(60, min_periods=1).mean().to_numpy()
        streak = np.zeros(len(session_close), dtype=np.float64)
        for offset in range(1, len(session_close)):
            streak[offset] = min(streak[offset - 1] + 1, 60) if flat[offset] else 0
        output[start:stop, 14] = streak / 60.0
        return_series = pd.Series(returns)
        output[start:stop, 15] = return_series.abs().rolling(5, min_periods=1).mean().to_numpy()
        output[start:stop, 16] = return_series.rolling(15, min_periods=2).std(ddof=0).fillna(0).to_numpy()
        position = np.linspace(0.0, 1.0, len(session_close), endpoint=True)
        output[start:stop, 17] = np.sin(2 * np.pi * position)
        output[start:stop, 18] = np.cos(2 * np.pi * position)
        output[start:stop, 19] = float(session_id)
    output[~np.isfinite(output)] = np.nan
    return output if include_state_features else output[:, :len(BASE_FEATURE_NAMES)]


def _sequence_candidates(
    days: list[np.ndarray] | np.ndarray, seq_len: int,
    include_state_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return feature windows and signed next-minute price changes.

    The morning segment includes 09:30–11:30.  The afternoon segment uses
    13:00–14:56; closing-auction waiting/matched rows are excluded from LSTM.
    """
    iterable = list(days) if isinstance(days, np.ndarray) else days
    features_out: list[np.ndarray] = []
    changes: list[float] = []
    for day in iterable:
        features = engineer_features(day, include_state_features)
        close = day[:, 3]
        for start, stop in ((0, 121), (121, 238)):
            for end in range(start + seq_len - 1, stop - 1):
                current, following = close[end], close[end + 1]
                if not np.isfinite(current) or not np.isfinite(following):
                    continue
                window = features[end - seq_len + 1:end + 1]
                if np.isfinite(window).all():
                    features_out.append(window)
                    changes.append(float(following - current))
    if not features_out:
        return (
            np.empty((
                0, seq_len,
                len(FEATURE_NAMES) if include_state_features else len(BASE_FEATURE_NAMES),
            ), dtype=np.float32),
            np.empty(0, dtype=np.float64),
        )
    return np.asarray(features_out, dtype=np.float32), np.asarray(changes)


def create_sequences(
    days: list[np.ndarray] | np.ndarray, seq_len: int = 60,
    include_state_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict close[t+1] versus close[t] as down=0, flat=1, up=2."""
    features, changes = _sequence_candidates(days, seq_len, include_state_features)
    labels = np.where(changes < 0, 0, np.where(changes > 0, 2, 1))
    return features, labels.astype(np.int64)


class MinuteLSTM(nn.Module):
    """Direct down/flat/up classifier retained as the formal best baseline."""

    def __init__(self, input_size: int = len(BASE_FEATURE_NAMES), hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Dropout(dropout), nn.Linear(hidden_size, 3),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(values)
        return self.head(output[:, -1])


class HierarchicalMinuteLSTM(nn.Module):
    """Shared sequence encoder with movement and conditional direction heads."""

    def __init__(self, input_size: int = len(FEATURE_NAMES), hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.normalization = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.movement_head = nn.Linear(hidden_size, 2)
        self.direction_head = nn.Linear(hidden_size, 2)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(values)
        encoded = self.dropout(self.normalization(output[:, -1]))
        return self.movement_head(encoded), self.direction_head(encoded)


def _make_split(stock_codes: list[str], date_range: tuple[str, str], seq_len: int,
                data_dir: Path | None, include_state_features: bool,
                ) -> tuple[np.ndarray, np.ndarray]:
    raw = load_minute_data_for_stocks(stock_codes, date_range, data_dir)
    arrays = [
        create_sequences(days, seq_len, include_state_features) for days in raw.values()
    ]
    arrays = [(x, y) for x, y in arrays if len(x)]
    if not arrays:
        feature_count = len(FEATURE_NAMES) if include_state_features else len(BASE_FEATURE_NAMES)
        return np.empty((0, seq_len, feature_count), np.float32), np.empty(0, np.int64)
    return np.concatenate([x for x, _ in arrays]), np.concatenate([y for _, y in arrays])


def prepare_dataloaders(
    stock_codes: list[str], seq_len: int = 60, batch_size: int = 256,
    data_dir: Path | None = None,
    splits: dict[str, tuple[str, str]] | None = None,
    include_state_features: bool = False,
) -> dict:
    ranges = splits or DEFAULT_SPLITS
    x_train, y_train = _make_split(
        stock_codes, ranges["train"], seq_len, data_dir, include_state_features,
    )
    x_val, y_val = _make_split(
        stock_codes, ranges["validation"], seq_len, data_dir, include_state_features,
    )
    x_test, y_test = _make_split(
        stock_codes, ranges["test"], seq_len, data_dir, include_state_features,
    )
    if not len(x_train) or not len(x_val) or not len(x_test):
        raise ValueError("one or more LSTM data splits are empty")
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    normalized = [(values - mean) / std for values in (x_train, x_val, x_test)]
    loaders = {}
    for name, values, labels, shuffle in (
        ("train", normalized[0], y_train, True),
        ("validation", normalized[1], y_val, False),
        ("test", normalized[2], y_test, False),
    ):
        loaders[name] = DataLoader(
            TensorDataset(torch.from_numpy(values), torch.from_numpy(labels)),
            batch_size=batch_size, shuffle=shuffle,
        )
    return {
        "loaders": loaders, "mean": mean.squeeze().astype(float),
        "std": std.squeeze().astype(float),
        "sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "class_rates": {
            split: {
                name: float((labels == class_id).mean())
                for class_id, name in enumerate(CLASS_NAMES)
            }
            for split, labels in (
                ("train", y_train), ("validation", y_val), ("test", y_test),
            )
        },
        "train_labels": y_train, "test_labels": y_test,
    }


def _collect_outputs(model: nn.Module, loader: DataLoader,
                     device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    actual, movement_probabilities, up_probabilities = [], [], []
    with torch.no_grad():
        for values, labels in loader:
            movement_logits, direction_logits = model(values.to(device))
            movement_probabilities.append(
                movement_logits.softmax(1)[:, 1].cpu().numpy()
            )
            up_probabilities.append(direction_logits.softmax(1)[:, 1].cpu().numpy())
            actual.append(labels.numpy())
    return (
        np.concatenate(actual), np.concatenate(movement_probabilities),
        np.concatenate(up_probabilities),
    )


def _predict_classes(move_probability: np.ndarray, up_probability: np.ndarray,
                     move_threshold: float, direction_threshold: float) -> np.ndarray:
    prediction = np.full(len(move_probability), 1, dtype=np.int64)
    moving = move_probability >= move_threshold
    prediction[moving & (up_probability < direction_threshold)] = 0
    prediction[moving & (up_probability >= direction_threshold)] = 2
    return prediction


def _classification_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
        precision_recall_fscore_support,
    )
    precision, recall, f1_by_class, support = precision_recall_fscore_support(
        truth, prediction, labels=[0, 1, 2], zero_division=0,
    )
    true_nonflat = truth != 1
    predicted_move = prediction != 1
    move_signal = predicted_move
    movement_truth = true_nonflat.astype(np.int64)
    movement_prediction = predicted_move.astype(np.int64)
    metrics: dict[str, float | int | list | dict | str] = {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, prediction, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1, 2]).tolist(),
        "class_names": list(CLASS_NAMES),
        "per_class": {
            name: {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1_by_class[class_id]),
                "support": int(support[class_id]),
            }
            for class_id, name in enumerate(CLASS_NAMES)
        },
        "nonflat_direction_accuracy": float(
            (prediction[true_nonflat] == truth[true_nonflat]).mean()
        ) if true_nonflat.any() else 0.0,
        "true_nonflat_rate": float(true_nonflat.mean()),
        "predicted_move_rate": float(predicted_move.mean()),
        "true_nonflat_capture_rate": float(predicted_move[true_nonflat].mean()) if true_nonflat.any() else 0.0,
        "move_signal_exact_accuracy": float(
            (prediction[move_signal] == truth[move_signal]).mean()
        ) if move_signal.any() else 0.0,
        "movement_detection_accuracy": float(accuracy_score(movement_truth, movement_prediction)),
        "movement_detection_balanced_accuracy": float(
            balanced_accuracy_score(movement_truth, movement_prediction)
        ),
    }
    # Compatibility alias: in the three-class model F1 means macro F1.
    metrics["f1"] = metrics["macro_f1"]
    return metrics


def evaluate_hierarchical(model: nn.Module, loader: DataLoader,
                          device: torch.device | None = None,
                          move_threshold: float = 0.5,
                          direction_threshold: float = 0.5) -> dict[str, object]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    truth, move_probability, up_probability = _collect_outputs(model, loader, device)
    prediction = _predict_classes(
        move_probability, up_probability, move_threshold, direction_threshold,
    )
    metrics = _classification_metrics(truth, prediction)
    metrics["move_threshold"] = float(move_threshold)
    metrics["direction_threshold"] = float(direction_threshold)
    return metrics


def tune_thresholds(model: nn.Module, loader: DataLoader,
                    device: torch.device) -> dict[str, float]:
    """Calibrate both decision thresholds on validation macro F1 only."""
    truth, move_probability, up_probability = _collect_outputs(model, loader, device)
    best: tuple[tuple[float, float, float], float, float] | None = None
    for move_threshold in np.arange(0.30, 0.701, 0.02):
        for direction_threshold in np.arange(0.40, 0.601, 0.02):
            prediction = _predict_classes(
                move_probability, up_probability,
                float(move_threshold), float(direction_threshold),
            )
            metrics = _classification_metrics(truth, prediction)
            score = (
                float(metrics["macro_f1"]),
                float(metrics["balanced_accuracy"]),
                float(metrics["accuracy"]),
            )
            if best is None or score > best[0]:
                best = (score, float(move_threshold), float(direction_threshold))
    assert best is not None
    return {
        "move_threshold": best[1], "direction_threshold": best[2],
        "validation_macro_f1": best[0][0],
        "validation_balanced_accuracy": best[0][1],
        "validation_accuracy": best[0][2],
    }


def _task_weights(labels: np.ndarray, class_count: int,
                  power: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=class_count)
    weights = (len(labels) / np.maximum(counts, 1)) ** power
    return weights / weights.mean()


def evaluate_direct(model: nn.Module, loader: DataLoader,
                    device: torch.device | None = None) -> dict[str, object]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predicted, actual = [], []
    with torch.no_grad():
        for values, labels in loader:
            predicted.append(model(values.to(device)).argmax(1).cpu().numpy())
            actual.append(labels.numpy())
    return _classification_metrics(np.concatenate(actual), np.concatenate(predicted))


def train_direct_model(
    model: nn.Module, loaders: dict[str, DataLoader], train_labels: np.ndarray,
    epochs: int, device: torch.device, patience: int = 4,
    class_weight_power: float = 0.5,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            _task_weights(train_labels, 3, class_weight_power),
            dtype=torch.float32, device=device,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5,
    )
    history = {
        "train_loss": [], "val_accuracy": [], "val_balanced_accuracy": [],
        "val_macro_f1": [],
    }
    best_score, best_state, stale = -np.inf, copy.deepcopy(model.state_dict()), 0
    model.to(device)
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for values, labels in loaders["train"]:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(values), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(values)
        loss_value = loss_sum / len(loaders["train"].dataset)
        validation = evaluate_direct(model, loaders["validation"], device)
        score = float(validation["balanced_accuracy"])
        history["train_loss"].append(loss_value)
        history["val_accuracy"].append(float(validation["accuracy"]))
        history["val_balanced_accuracy"].append(score)
        history["val_macro_f1"].append(float(validation["macro_f1"]))
        scheduler.step(score)
        print(
            f"epoch {epoch + 1:02d}: loss={loss_value:.5f} "
            f"val_acc={validation['accuracy']:.4f} val_bal={score:.4f} "
            f"val_f1={validation['macro_f1']:.4f}"
        )
        if score > best_score + 1e-5:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                print("early stopping")
                break
    return history, best_state


def train_hierarchical_model(
    model: nn.Module, loaders: dict[str, DataLoader], train_labels: np.ndarray,
    epochs: int, device: torch.device, patience: int = 4,
    class_weight_power: float = 0.5, direction_loss_weight: float = 1.0,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    movement_labels = (train_labels != 1).astype(np.int64)
    direction_labels = (train_labels[train_labels != 1] == 2).astype(np.int64)
    movement_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            _task_weights(movement_labels, 2, class_weight_power),
            dtype=torch.float32, device=device,
        ),
    )
    direction_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            _task_weights(direction_labels, 2, class_weight_power),
            dtype=torch.float32, device=device,
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5,
    )
    history = {
        "train_loss": [], "movement_loss": [], "direction_loss": [],
        "val_accuracy": [], "val_balanced_accuracy": [], "val_macro_f1": [],
    }
    best_score, best_state, stale = -np.inf, copy.deepcopy(model.state_dict()), 0
    model.to(device)
    for epoch in range(epochs):
        model.train()
        loss_sum = movement_sum = direction_sum = 0.0
        for values, labels in loaders["train"]:
            values, labels = values.to(device), labels.to(device)
            optimizer.zero_grad()
            movement_logits, direction_logits = model(values)
            movement_target = (labels != 1).long()
            moving = movement_target.bool()
            movement_loss = movement_criterion(movement_logits, movement_target)
            direction_target = (labels[moving] == 2).long()
            direction_loss = direction_criterion(direction_logits[moving], direction_target)
            loss = movement_loss + direction_loss_weight * direction_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(values)
            movement_sum += float(movement_loss.item()) * len(values)
            direction_sum += float(direction_loss.item()) * len(values)
        loss_value = loss_sum / len(loaders["train"].dataset)
        validation = evaluate_hierarchical(model, loaders["validation"], device)
        score = float(validation["macro_f1"])
        history["train_loss"].append(loss_value)
        history["movement_loss"].append(movement_sum / len(loaders["train"].dataset))
        history["direction_loss"].append(direction_sum / len(loaders["train"].dataset))
        history["val_accuracy"].append(float(validation["accuracy"]))
        history["val_balanced_accuracy"].append(float(validation["balanced_accuracy"]))
        history["val_macro_f1"].append(score)
        scheduler.step(score)
        print(
            f"epoch {epoch + 1:02d}: loss={loss_value:.5f} "
            f"val_acc={validation['accuracy']:.4f} "
            f"val_bal={validation['balanced_accuracy']:.4f} val_f1={score:.4f}"
        )
        if score > best_score + 1e-5:
            best_score, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
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
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[1].plot(history["val_accuracy"], marker="o", label="Accuracy")
    axes[1].plot(history["val_balanced_accuracy"], marker="o", label="Balanced accuracy")
    axes[1].plot(history["val_macro_f1"], marker="o", label="Macro F1")
    axes[1].set_title("Validation metrics")
    axes[1].legend()
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / "training_history.png", dpi=160)
    plt.close(figure)


def _write_report(result: dict, out_dir: Path) -> None:
    metrics, config = result["test_metrics"], result["config"]
    hierarchical = config["model_type"] == "hierarchical_multitask_lstm"
    lines = [
        "# LSTM 下一分钟下跌/平盘/上涨预测报告", "",
        (
            "共享 LSTM 先学习平盘/变动，再对真实变动样本学习下跌/上涨。"
            if hierarchical else
            "单个三分类 LSTM 直接学习下跌、平盘和上涨。"
        ),
        "平盘样本参与训练，主指标覆盖全部有效分钟窗口。", "",
        "## 设置", "", f"- 股票：{', '.join(config['stock_codes'])}",
        f"- 特征：{len(config['feature_names'])} 个平稳化分钟特征",
        f"- 序列长度：{config['seq_len']} 分钟",
        f"- 样本数：{config['sizes']}",
        f"- 测试集类别占比：{config['class_rates']['test']}", "", "## 全分钟测试结果", "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Majority baseline: {metrics['majority_baseline']:.4f}", "",
        "## 交易相关诊断", "",
        f"- 真实非平盘样本方向准确率：{metrics['nonflat_direction_accuracy']:.4f}",
        f"- 模型预测发生变动的比例：{metrics['predicted_move_rate']:.4f}",
        f"- 真实非平盘样本捕获率：{metrics['true_nonflat_capture_rate']:.4f}",
        f"- 模型发出涨跌信号时的精确命中率：{metrics['move_signal_exact_accuracy']:.4f}", "",
        "`model.pt` 包含权重、结构、标准化参数、股票、特征顺序和日期切分。",
    ]
    if hierarchical:
        lines[10:10] = [
            f"- 验证集阈值：move={config['thresholds']['move_threshold']:.2f}, "
            f"direction={config['thresholds']['direction_threshold']:.2f}",
        ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_lstm_pipeline(
    stock_codes: list[str] | None = None, n_stocks: int = 5, seq_len: int = 60,
    hidden_size: int = 64, epochs: int = 12, batch_size: int = 256,
    data_dir: Path | None = None, out_dir: Path | None = None, seed: int = 42,
    class_weight_power: float = 0.5, direction_loss_weight: float = 1.0,
    model_type: str = "direct",
) -> dict:
    set_seed(seed)
    minute_dir = Path(data_dir or (OUTPUT_DIR / "minute"))
    target = Path(out_dir or (OUTPUT_DIR / "lstm_next_minute"))
    target.mkdir(parents=True, exist_ok=True)
    if stock_codes is None:
        first = sorted((minute_dir / "close").glob("*.csv"))[0]
        stock_codes = pd.read_csv(first, nrows=1, index_col=0).columns[:n_stocks].tolist()
    if model_type not in {"direct", "hierarchical"}:
        raise ValueError("model_type must be 'direct' or 'hierarchical'")
    hierarchical = model_type == "hierarchical"
    prepared = prepare_dataloaders(
        stock_codes, seq_len, batch_size, minute_dir,
        include_state_features=hierarchical,
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device={device}; stocks={stock_codes}; sizes={prepared['sizes']}")
    model: nn.Module
    if hierarchical:
        model = HierarchicalMinuteLSTM(hidden_size=hidden_size)
        history, best_state = train_hierarchical_model(
            model, prepared["loaders"], prepared["train_labels"], epochs, device,
            class_weight_power=class_weight_power,
            direction_loss_weight=direction_loss_weight,
        )
    else:
        model = MinuteLSTM(hidden_size=hidden_size)
        history, best_state = train_direct_model(
            model, prepared["loaders"], prepared["train_labels"], epochs, device,
            class_weight_power=class_weight_power,
        )
    model.load_state_dict(best_state)
    model.to(device)
    thresholds = None
    if hierarchical:
        thresholds = tune_thresholds(model, prepared["loaders"]["validation"], device)
        test_metrics = evaluate_hierarchical(
            model, prepared["loaders"]["test"], device,
            move_threshold=thresholds["move_threshold"],
            direction_threshold=thresholds["direction_threshold"],
        )
    else:
        test_metrics = evaluate_direct(model, prepared["loaders"]["test"], device)
    test_labels = prepared["test_labels"]
    test_counts = np.bincount(test_labels, minlength=3)
    test_metrics["majority_baseline"] = float(test_counts.max() / len(test_labels))
    test_metrics["scope"] = "all valid next-minute windows; down/flat/up three-class target"
    config = {
        "stock_codes": stock_codes, "channels": list(CHANNELS),
        "feature_names": list(FEATURE_NAMES if hierarchical else BASE_FEATURE_NAMES),
        "seq_len": seq_len,
        "hidden_size": hidden_size, "num_layers": 2, "num_classes": 3,
        "model_type": (
            "hierarchical_multitask_lstm" if hierarchical else "direct_three_class_lstm"
        ),
        "class_names": list(CLASS_NAMES), "class_weight_power": class_weight_power,
        "direction_loss_weight": direction_loss_weight if hierarchical else None,
        "thresholds": thresholds,
        "seed": seed,
        "splits": DEFAULT_SPLITS, "sizes": prepared["sizes"],
        "class_rates": prepared["class_rates"],
        "scaler_mean": prepared["mean"].tolist(), "scaler_std": prepared["std"].tolist(),
    }
    torch.save(
        {"state_dict": {key: value.cpu() for key, value in best_state.items()}, "config": config},
        target / "model.pt",
    )
    (target / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (target / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (target / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    result = {"model": model, "history": history, "test_metrics": test_metrics, "config": config}
    _plot_history(history, target)
    _write_report(result, target)
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    return result
