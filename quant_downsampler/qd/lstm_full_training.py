"""Reproducible training for the full-window down/flat/up LSTM ensemble.

The final test split is not loaded until all component checkpoints, fusion
parameters, and selective-confidence thresholds have been fixed using the
training and validation splits only.
"""

from __future__ import annotations

import gc
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR
from .lstm_full import blend_probabilities, two_stage_probabilities
from .lstm_model import (
    CHANNELS,
    DEFAULT_SPLITS,
    MinuteLSTM,
    _make_split,
    classification_metrics,
    feature_names as declared_feature_names,
    predict_class_probabilities,
    set_seed,
    train_model,
    transform_sequences,
)


CLASS_NAMES = ["down", "flat", "up"]
COMPONENT_CLASS_NAMES = {
    "direction": ["down", "up"],
    "movement": ["flat", "move"],
    "joint": CLASS_NAMES,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_splits(splits: dict[str, tuple[str, str]]) -> None:
    for name in ("train", "val", "test"):
        if name not in splits or len(splits[name]) != 2:
            raise ValueError(f"missing or malformed {name!r} date split")
        start, end = splits[name]
        if not re.fullmatch(r"\d{8}", start) or not re.fullmatch(r"\d{8}", end):
            raise ValueError(f"{name} dates must use YYYYMMDD format")
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
    stock_codes: list[str] | None,
    n_stocks: int,
    data_dir: Path,
) -> list[str]:
    if stock_codes is not None:
        stocks = [str(stock).strip() for stock in stock_codes]
        if not stocks or any(not stock for stock in stocks):
            raise ValueError("stock_codes must contain at least one non-empty code")
        if len(set(stocks)) != len(stocks):
            raise ValueError("stock_codes contains duplicates")
        return stocks
    if n_stocks <= 0:
        raise ValueError("n_stocks must be positive")
    files = sorted((data_dir / "close").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no minute close tables found under {data_dir}")
    columns = pd.read_csv(files[0], nrows=1, index_col=0).columns.tolist()
    if n_stocks > len(columns):
        raise ValueError(f"requested {n_stocks} stocks but only {len(columns)} are available")
    return columns[:n_stocks]


def _prepare_output_dir(target: Path, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty full-LSTM output directory: {target}; "
            "choose another --out-dir or pass --overwrite"
        )
    target.mkdir(parents=True, exist_ok=True)


def _class_rates(labels: np.ndarray, n_classes: int) -> list[float]:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return (counts / counts.sum()).tolist()


def _normalise_features(
    x_train: np.ndarray,
    train_ids: np.ndarray,
    x_val: np.ndarray,
    val_ids: np.ndarray,
    stocks: list[str],
    scaler_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scaler_mode == "global":
        mean = x_train.mean(axis=(0, 1), keepdims=True)
        std = x_train.std(axis=(0, 1), keepdims=True)
        std[std < 1e-8] = 1.0
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        return x_train, x_val, mean.squeeze().astype(float), std.squeeze().astype(float)
    if scaler_mode != "per_stock":
        raise ValueError(f"unknown scaler_mode={scaler_mode!r}")

    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for stock_id, stock in enumerate(stocks):
        stock_train = x_train[train_ids == stock_id]
        if not len(stock_train):
            raise ValueError(f"no training sequences for stock {stock}")
        mean = stock_train.mean(axis=(0, 1), keepdims=True)
        std = stock_train.std(axis=(0, 1), keepdims=True)
        std[std < 1e-8] = 1.0
        means.append(mean)
        stds.append(std)
    for array, ids in ((x_train, train_ids), (x_val, val_ids)):
        for stock_id, (mean, std) in enumerate(zip(means, stds)):
            mask = ids == stock_id
            array[mask] = (array[mask] - mean) / std
    return (
        x_train,
        x_val,
        np.stack([value.squeeze() for value in means]).astype(float),
        np.stack([value.squeeze() for value in stds]).astype(float),
    )


def _append_stock_id(
    x: np.ndarray,
    stock_ids: np.ndarray,
    n_stocks: int,
) -> np.ndarray:
    one_hot = np.eye(n_stocks, dtype=np.float32)[stock_ids]
    repeated = np.broadcast_to(one_hot[:, None, :], (len(x), x.shape[1], n_stocks))
    return np.concatenate((x, repeated), axis=2)


def _loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _prepare_train_val(
    stocks: list[str],
    splits: dict[str, tuple[str, str]],
    seq_len: int,
    batch_size: int,
    data_dir: Path,
    feature_set: str,
    target_mode: str,
    include_stock_id: bool,
    scaler_mode: str,
    seed: int,
) -> dict:
    """Prepare only train/validation data; the test date range is untouched."""
    train = _make_split(
        stocks, splits["train"], seq_len, data_dir,
        feature_set, target_mode, True,
    )
    val = _make_split(
        stocks, splits["val"], seq_len, data_dir,
        feature_set, target_mode, True,
    )
    x_train, y_train, train_ids, train_coverage, train_metadata = train
    x_val, y_val, val_ids, val_coverage, val_metadata = val
    if not len(x_train) or not len(x_val):
        raise ValueError("training or validation split contains no LSTM samples")
    x_train, x_val, mean, std = _normalise_features(
        x_train, train_ids, x_val, val_ids, stocks, scaler_mode
    )
    base_feature_count = x_train.shape[2]
    if include_stock_id:
        x_train = _append_stock_id(x_train, train_ids, len(stocks))
        x_val = _append_stock_id(x_val, val_ids, len(stocks))
    names = list(declared_feature_names(feature_set))
    if len(names) != base_feature_count:
        raise RuntimeError("engineered feature count and declared names differ")
    if include_stock_id:
        names.extend(f"stock_id::{stock}" for stock in stocks)
    n_classes = 3 if target_mode == "three_class" else 2
    return {
        "loaders": {
            "train": _loader(x_train, y_train, batch_size, True, seed),
            "val": _loader(x_val, y_val, batch_size, False, seed),
        },
        "mean": mean,
        "std": std,
        "feature_names": names,
        "sizes": {"train": len(y_train), "val": len(y_val)},
        "class_rates": {
            "train": _class_rates(y_train, n_classes),
            "val": _class_rates(y_val, n_classes),
        },
        "coverage": {"train": train_coverage, "val": val_coverage},
        "train_labels": y_train,
        "val_labels": y_val,
        "train_stock_ids": train_ids,
        "val_stock_ids": val_ids,
        "train_metadata": train_metadata,
        "val_metadata": val_metadata,
    }


def _movement_view(joint_data: dict, batch_size: int, seed: int) -> dict:
    loaders: dict[str, DataLoader] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ("train", "val"):
        x_tensor, joint_y_tensor = joint_data["loaders"][split].dataset.tensors
        movement_y = (joint_y_tensor.numpy() != 1).astype(np.int64)
        labels[split] = movement_y
        loaders[split] = _loader(
            x_tensor.numpy(), movement_y, batch_size, split == "train", seed
        )
    return {
        "loaders": loaders,
        "mean": joint_data["mean"],
        "std": joint_data["std"],
        "feature_names": joint_data["feature_names"],
        "sizes": dict(joint_data["sizes"]),
        "class_rates": {
            split: _class_rates(labels[split], 2) for split in ("train", "val")
        },
        "coverage": joint_data["coverage"],
        "train_labels": labels["train"],
        "val_labels": labels["val"],
        "train_stock_ids": joint_data["train_stock_ids"],
        "val_stock_ids": joint_data["val_stock_ids"],
        "train_metadata": joint_data["train_metadata"],
        "val_metadata": joint_data["val_metadata"],
    }


def _component_config(
    *,
    name: str,
    data: dict,
    stocks: list[str],
    splits: dict[str, tuple[str, str]],
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    feature_set: str,
    target_mode: str,
    include_stock_id: bool,
    scaler_mode: str,
    model_version: str,
    class_weighted: bool,
    label_smoothing: float,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    history: dict[str, list[float]],
    device: torch.device,
) -> dict:
    class_names = COMPONENT_CLASS_NAMES[name]
    return {
        "component": name,
        "stock_codes": stocks,
        "channels": list(CHANNELS),
        "feature_names": data["feature_names"],
        "input_size": len(data["feature_names"]),
        "feature_set": feature_set,
        "target_mode": target_mode,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "include_stock_id": include_stock_id,
        "scaler_mode": scaler_mode,
        "scaler_mean": data["mean"].tolist(),
        "scaler_std": data["std"].tolist(),
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "model_version": model_version,
        "class_weighted": class_weighted,
        "label_smoothing": label_smoothing,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "scheduler": "ReduceLROnPlateau(mode=max,patience=2,factor=0.5)",
        "early_stopping_metric": "validation_accuracy_at_component_default_rule",
        "decision_threshold": None if len(class_names) == 3 else 0.5,
        "epochs_requested": epochs,
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": int(np.argmax(history["val_accuracy"]) + 1),
        "patience": patience,
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "splits": splits,
        "sizes": dict(data["sizes"]),
        "class_rates": dict(data["class_rates"]),
        "coverage": dict(data["coverage"]),
    }


def _train_component(
    *,
    name: str,
    data: dict,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    model_version: str,
    class_weighted: bool,
    label_smoothing: float,
    learning_rate: float,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[MinuteLSTM, dict[str, torch.Tensor], dict[str, list[float]]]:
    set_seed(seed)
    model = MinuteLSTM(
        input_size=len(data["feature_names"]),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_version=model_version,
        num_classes=len(COMPONENT_CLASS_NAMES[name]),
    )
    history, state = train_model(
        model,
        data["loaders"],
        data["train_labels"],
        epochs,
        device,
        patience=patience,
        class_weighted=class_weighted,
        label_smoothing=label_smoothing,
        learning_rate=learning_rate,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, state, history


def _predict_arrays(
    model: MinuteLSTM,
    x: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _loader(x, labels, batch_size, False, 0)
    probability, replayed = predict_class_probabilities(model, loader, device)
    if not np.array_equal(labels, replayed):
        raise RuntimeError("labels changed inside an inference data loader")
    return probability


def _inclusive_grid(low: float, high: float, step: float) -> np.ndarray:
    if not np.isfinite([low, high, step]).all() or step <= 0 or low > high:
        raise ValueError("invalid fusion grid bounds")
    count = int(np.floor((high - low) / step + 1e-10)) + 1
    values = low + step * np.arange(count, dtype=np.float64)
    if values[-1] < high - 1e-10:
        values = np.append(values, high)
    return np.clip(values, low, high)


def _select_fusion(
    labels: np.ndarray,
    direction_probability: np.ndarray,
    movement_probability: np.ndarray,
    joint_probability: np.ndarray,
    move_biases: np.ndarray,
    joint_weights: np.ndarray,
    objective: str = "macro_f1_then_accuracy",
) -> tuple[dict, np.ndarray]:
    if objective not in {"macro_f1_then_accuracy", "accuracy_then_macro_f1"}:
        raise ValueError("unknown fusion objective")
    best_score: tuple[float, float, float, float] | None = None
    best: dict | None = None
    best_probability: np.ndarray | None = None
    for move_bias in move_biases:
        staged = two_stage_probabilities(
            movement_probability[:, 1], direction_probability[:, 1], float(move_bias)
        )
        for joint_weight in joint_weights:
            probability = blend_probabilities(
                joint_probability, staged, float(joint_weight)
            )
            metrics = classification_metrics(labels, probability, threshold=None)
            primary = (
                (float(metrics["f1"]), float(metrics["accuracy"]))
                if objective == "macro_f1_then_accuracy"
                else (float(metrics["accuracy"]), float(metrics["f1"]))
            )
            score = (
                *primary,
                -abs(float(joint_weight) - 0.5),
                -abs(float(move_bias)),
            )
            if best_score is None or score > best_score:
                best_score = score
                best = {
                    "objective": objective,
                    "move_bias": float(move_bias),
                    "joint_weight": float(joint_weight),
                    "two_stage_weight": float(1.0 - joint_weight),
                    "validation_accuracy": float(metrics["accuracy"]),
                    "validation_macro_f1": float(metrics["f1"]),
                }
                best_probability = probability
    if best is None or best_probability is None:
        raise RuntimeError("fusion grid did not produce a candidate")
    return best, best_probability


def _assert_aligned(
    legacy_labels: np.ndarray,
    enhanced_labels: np.ndarray,
    legacy_ids: np.ndarray,
    enhanced_ids: np.ndarray,
    legacy_metadata: pd.DataFrame,
    enhanced_metadata: pd.DataFrame,
) -> None:
    if (
        not np.array_equal(legacy_labels, enhanced_labels)
        or not np.array_equal(legacy_ids, enhanced_ids)
        or not legacy_metadata.equals(enhanced_metadata)
    ):
        raise ValueError("legacy and enhanced full-window samples are not aligned")


def _multiclass_report(true: np.ndarray, probability: np.ndarray) -> dict:
    raw = classification_metrics(true, probability, threshold=None)
    return {
        "accuracy": raw["accuracy"],
        "macro_precision": raw["precision"],
        "macro_recall": raw["recall"],
        "macro_f1": raw["f1"],
        "confusion_matrix": raw["confusion_matrix"],
        "n_classes": 3,
    }


def _cpu_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state.items()}


def _plot_histories(histories: dict[str, dict[str, list[float]]], target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        axes[0].plot(history["train_loss"], label=name)
        axes[1].plot(history["val_accuracy"], label=name)
    axes[0].set_title("Training loss")
    axes[1].set_title("Validation accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(target / "training_history.png", dpi=160)
    plt.close(figure)


def _write_readme(target: Path, config: dict, metrics: dict) -> None:
    balanced = metrics["selective_accuracy"]["balanced"]
    strict = metrics["selective_accuracy"]["strict"]
    lines = [
        "# 全窗口下一分钟跌/平/涨 LSTM",
        "",
        "该目录由正式三组件训练入口生成。所有合法窗口均参与跌/平/涨评价；测试区间只在组件早停、融合参数和置信阈值由验证集冻结后读取。",
        "",
        "## 模型",
        "",
        f"- 股票：{', '.join(config['stock_codes'])}",
        f"- 日期切分：train={config['splits']['train']}，val={config['splits']['val']}，test={config['splits']['test']}",
        f"- 序列长度：{config['seq_len']} 分钟",
        "- direction：nonflat_binary，legacy 特征，全局标准化，无股票 ID",
        "- movement：move_vs_flat，enhanced 特征，逐股票标准化，含股票 ID",
        "- joint：three_class，enhanced 特征，逐股票标准化，含股票 ID",
        f"- 验证集选择：move_bias={config['move_bias']:.4f}，joint_weight={config['joint_weight']:.4f}",
        "",
        "## 最终测试结果",
        "",
        f"- All-window Accuracy：{metrics['accuracy']:.4f}",
        f"- Majority baseline：{metrics['majority_baseline']:.4f}",
        f"- Macro Precision / Recall / F1：{metrics['macro_precision']:.4f} / {metrics['macro_recall']:.4f} / {metrics['macro_f1']:.4f}",
        f"- Macro stock accuracy：{metrics['macro_stock_accuracy']:.4f}",
        f"- Confusion matrix [down, flat, up]：{metrics['confusion_matrix']}",
        f"- balanced：accuracy={balanced['test_accuracy']:.4f}，coverage={balanced['test_coverage']:.2%}",
        f"- strict：accuracy={strict['test_accuracy']:.4f}，coverage={strict['test_coverage']:.2%}",
        "",
        "## 产物",
        "",
        "- `model.pt`：三个组件权重、各自训练集标准化参数、股票顺序、日期切分和验证集确定的融合参数",
        "- `history.json`：三个组件逐 epoch 训练损失和验证准确率",
        "- `test_metrics.json`：全窗口、逐股票、组件和选择性指标",
        "- `test_predictions.csv`：带 stock、window_end、target_time 的逐样本概率",
        "- `training_history.png`：训练曲线",
        "",
        "可使用 `python -m scripts.evaluate_lstm_full --model <本目录/model.pt>` 从原始分钟表严格重载复核。",
        "本入口保证本次运行不以测试集选择参数，但无法消除默认测试日期在以往研究中已被查看的历史；新的无偏确认仍需要后续未见日期。",
    ]
    (target / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_full_training(
    *,
    stock_codes: list[str] | None = None,
    n_stocks: int = 5,
    seq_len: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    direction_epochs: int = 12,
    movement_epochs: int = 12,
    joint_epochs: int = 12,
    patience: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.0,
    direction_class_weighted: bool = True,
    movement_class_weighted: bool = True,
    joint_class_weighted: bool = False,
    seed: int = 42,
    splits: dict[str, tuple[str, str]] | None = None,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    move_bias_min: float = -0.30,
    move_bias_max: float = 0.30,
    move_bias_step: float = 0.05,
    joint_weight_step: float = 0.05,
    fusion_objective: str = "macro_f1_then_accuracy",
    balanced_quantile: float = 0.70,
    strict_quantile: float = 0.90,
    overwrite: bool = False,
) -> dict:
    """Train, validate, freeze, and finally test the full-window ensemble."""
    if seq_len <= 0 or hidden_size <= 0 or num_layers <= 0:
        raise ValueError("seq_len, hidden_size, and num_layers must be positive")
    if any(value <= 0 for value in (direction_epochs, movement_epochs, joint_epochs)):
        raise ValueError("all component epoch counts must be positive")
    if patience <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("patience, batch_size, and learning_rate must be positive")
    if not 0.0 <= dropout < 1.0 or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("dropout and label_smoothing must be within [0, 1)")
    if not 0.0 < balanced_quantile < strict_quantile < 1.0:
        raise ValueError("require 0 < balanced_quantile < strict_quantile < 1")
    if fusion_objective not in {
        "macro_f1_then_accuracy", "accuracy_then_macro_f1"
    }:
        raise ValueError("unknown fusion_objective")
    ranges = dict(splits or DEFAULT_SPLITS)
    _validate_splits(ranges)
    minute_dir = Path(data_dir or (OUTPUT_DIR / "minute"))
    target = Path(out_dir or (OUTPUT_DIR / "lstm_runs" / "full"))
    _prepare_output_dir(target, overwrite)
    stocks = _resolve_stocks(stock_codes, n_stocks, minute_dir)
    target_device = _resolve_device(device)
    component_seeds = {
        "direction": seed,
        "movement": seed + 1,
        "joint": seed + 2,
    }
    move_biases = _inclusive_grid(move_bias_min, move_bias_max, move_bias_step)
    joint_weights = _inclusive_grid(0.0, 1.0, joint_weight_step)
    started_at = _utc_now()

    print("preparing direction train/validation data (test remains unopened)")
    direction_data = _prepare_train_val(
        stocks, ranges, seq_len, batch_size, minute_dir,
        "legacy", "nonflat_binary", False, "global", component_seeds["direction"],
    )
    direction_model, direction_state, direction_history = _train_component(
        name="direction", data=direction_data, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, model_version="legacy",
        class_weighted=direction_class_weighted, label_smoothing=label_smoothing,
        learning_rate=learning_rate, epochs=direction_epochs, patience=patience,
        seed=component_seeds["direction"], device=target_device,
    )
    direction_config = _component_config(
        name="direction", data=direction_data, stocks=stocks, splits=ranges,
        seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, feature_set="legacy", target_mode="nonflat_binary",
        include_stock_id=False, scaler_mode="global", model_version="legacy",
        class_weighted=direction_class_weighted,
        label_smoothing=label_smoothing, learning_rate=learning_rate,
        epochs=direction_epochs, patience=patience, batch_size=batch_size,
        seed=component_seeds["direction"], history=direction_history,
        device=target_device,
    )
    direction_val_probability, direction_val_labels = predict_class_probabilities(
        direction_model, direction_data["loaders"]["val"], target_device
    )
    direction_val_metrics = classification_metrics(
        direction_val_labels, direction_val_probability, threshold=0.5
    )
    del direction_data
    gc.collect()

    print("preparing enhanced train/validation data (test remains unopened)")
    joint_data = _prepare_train_val(
        stocks, ranges, seq_len, batch_size, minute_dir,
        "enhanced", "three_class", True, "per_stock", component_seeds["joint"],
    )
    movement_data = _movement_view(joint_data, batch_size, component_seeds["movement"])
    movement_model, movement_state, movement_history = _train_component(
        name="movement", data=movement_data, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, model_version="residual",
        class_weighted=movement_class_weighted, label_smoothing=label_smoothing,
        learning_rate=learning_rate, epochs=movement_epochs, patience=patience,
        seed=component_seeds["movement"], device=target_device,
    )
    movement_config = _component_config(
        name="movement", data=movement_data, stocks=stocks, splits=ranges,
        seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, feature_set="enhanced", target_mode="move_vs_flat",
        include_stock_id=True, scaler_mode="per_stock", model_version="residual",
        class_weighted=movement_class_weighted,
        label_smoothing=label_smoothing, learning_rate=learning_rate,
        epochs=movement_epochs, patience=patience, batch_size=batch_size,
        seed=component_seeds["movement"], history=movement_history,
        device=target_device,
    )
    joint_model, joint_state, joint_history = _train_component(
        name="joint", data=joint_data, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout, model_version="residual",
        class_weighted=joint_class_weighted, label_smoothing=label_smoothing,
        learning_rate=learning_rate, epochs=joint_epochs, patience=patience,
        seed=component_seeds["joint"], device=target_device,
    )
    joint_config = _component_config(
        name="joint", data=joint_data, stocks=stocks, splits=ranges,
        seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
        dropout=dropout, feature_set="enhanced", target_mode="three_class",
        include_stock_id=True, scaler_mode="per_stock", model_version="residual",
        class_weighted=joint_class_weighted,
        label_smoothing=label_smoothing, learning_rate=learning_rate,
        epochs=joint_epochs, patience=patience, batch_size=batch_size,
        seed=component_seeds["joint"], history=joint_history,
        device=target_device,
    )

    movement_val_probability, movement_val_labels = predict_class_probabilities(
        movement_model, movement_data["loaders"]["val"], target_device
    )
    joint_val_probability, joint_val_labels = predict_class_probabilities(
        joint_model, joint_data["loaders"]["val"], target_device
    )
    movement_val_metrics = classification_metrics(
        movement_val_labels, movement_val_probability, threshold=0.5
    )
    joint_val_metrics = classification_metrics(
        joint_val_labels, joint_val_probability, threshold=None
    )

    legacy_val = _make_split(
        stocks, ranges["val"], seq_len, minute_dir,
        "legacy", "three_class", True,
    )
    legacy_val_x, legacy_val_y, legacy_val_ids, _, legacy_val_metadata = legacy_val
    _assert_aligned(
        legacy_val_y, joint_data["val_labels"], legacy_val_ids,
        joint_data["val_stock_ids"], legacy_val_metadata,
        joint_data["val_metadata"],
    )
    direction_full_val_x = transform_sequences(
        legacy_val_x, legacy_val_ids, direction_config, stocks
    )
    direction_full_val_probability = _predict_arrays(
        direction_model, direction_full_val_x, legacy_val_y,
        batch_size, target_device,
    )
    fusion, validation_probability = _select_fusion(
        joint_data["val_labels"], direction_full_val_probability,
        movement_val_probability, joint_val_probability,
        move_biases, joint_weights, fusion_objective,
    )
    val_confidence = validation_probability.max(axis=1)
    selective_thresholds = {
        "balanced": float(np.quantile(val_confidence, balanced_quantile)),
        "strict": float(np.quantile(val_confidence, strict_quantile)),
    }
    frozen_at = _utc_now()
    print(
        "validation configuration frozen: "
        f"move_bias={fusion['move_bias']:.4f}, "
        f"joint_weight={fusion['joint_weight']:.4f}; loading final test split"
    )

    # No test dates are loaded before this point.
    enhanced_test = _make_split(
        stocks, ranges["test"], seq_len, minute_dir,
        "enhanced", "three_class", True,
    )
    enhanced_test_x, test_labels, test_ids, test_coverage, test_metadata = enhanced_test
    enhanced_test_input = transform_sequences(
        enhanced_test_x, test_ids, joint_config, stocks
    )
    movement_test_probability = _predict_arrays(
        movement_model, enhanced_test_input, (test_labels != 1).astype(np.int64),
        batch_size, target_device,
    )
    joint_test_probability = _predict_arrays(
        joint_model, enhanced_test_input, test_labels, batch_size, target_device
    )
    legacy_test = _make_split(
        stocks, ranges["test"], seq_len, minute_dir,
        "legacy", "three_class", True,
    )
    legacy_test_x, legacy_test_y, legacy_test_ids, legacy_coverage, legacy_metadata = legacy_test
    _assert_aligned(
        legacy_test_y, test_labels, legacy_test_ids, test_ids,
        legacy_metadata, test_metadata,
    )
    if legacy_coverage != test_coverage:
        raise ValueError("legacy and enhanced test coverage differ")
    direction_test_input = transform_sequences(
        legacy_test_x, legacy_test_ids, direction_config, stocks
    )
    direction_test_probability = _predict_arrays(
        direction_model, direction_test_input, test_labels,
        batch_size, target_device,
    )
    staged_test_probability = two_stage_probabilities(
        movement_test_probability[:, 1], direction_test_probability[:, 1],
        fusion["move_bias"],
    )
    test_probability = blend_probabilities(
        joint_test_probability, staged_test_probability, fusion["joint_weight"]
    )
    metrics = _multiclass_report(test_labels, test_probability)
    metrics["majority_baseline"] = float(
        np.bincount(test_labels, minlength=3).max() / len(test_labels)
    )
    metrics["validation_accuracy"] = fusion["validation_accuracy"]
    metrics["validation_macro_f1"] = fusion["validation_macro_f1"]
    metrics["coverage"] = test_coverage
    metrics["fusion"] = dict(fusion)
    metrics["component_validation"] = {
        "direction": direction_val_metrics,
        "movement": movement_val_metrics,
        "joint": joint_val_metrics,
    }
    nonflat = test_labels != 1
    metrics["component_test"] = {
        "direction": classification_metrics(
            (test_labels[nonflat] == 2).astype(np.int64),
            direction_test_probability[nonflat], threshold=0.5,
        ),
        "movement": classification_metrics(
            (test_labels != 1).astype(np.int64),
            movement_test_probability, threshold=0.5,
        ),
        "joint": classification_metrics(
            test_labels, joint_test_probability, threshold=None,
        ),
    }
    predicted = test_probability.argmax(axis=1)
    per_stock: dict[str, dict] = {}
    for stock_id, stock in enumerate(stocks):
        mask = test_ids == stock_id
        values = _multiclass_report(test_labels[mask], test_probability[mask])
        per_stock[stock] = {
            "n": int(mask.sum()),
            "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"],
        }
    metrics["per_stock"] = per_stock
    metrics["macro_stock_accuracy"] = float(
        np.mean([value["accuracy"] for value in per_stock.values()])
    )

    test_confidence = test_probability.max(axis=1)
    selective: dict[str, dict] = {}
    selective_masks: dict[str, np.ndarray] = {}
    for name, quantile in (
        ("balanced", balanced_quantile), ("strict", strict_quantile)
    ):
        threshold = selective_thresholds[name]
        val_selected = val_confidence >= threshold
        test_selected = test_confidence >= threshold
        selective_masks[name] = test_selected
        selective[name] = {
            "validation_confidence_quantile": quantile,
            "validation_confidence_threshold": threshold,
            "validation_coverage": float(val_selected.mean()),
            "validation_accuracy": float(
                (validation_probability[val_selected].argmax(axis=1)
                 == joint_data["val_labels"][val_selected]).mean()
            ),
            "test_coverage": float(test_selected.mean()),
            "test_accuracy": float(
                (predicted[test_selected] == test_labels[test_selected]).mean()
            ),
            "test_n": int(test_selected.sum()),
        }
    metrics["selective_accuracy"] = selective

    direction_config["sizes"]["test"] = int(nonflat.sum())
    direction_config["class_rates"]["test"] = _class_rates(
        (test_labels[nonflat] == 2).astype(np.int64), 2
    )
    direction_config["coverage"]["test"] = test_coverage
    movement_config["sizes"]["test"] = len(test_labels)
    movement_config["class_rates"]["test"] = _class_rates(
        (test_labels != 1).astype(np.int64), 2
    )
    movement_config["coverage"]["test"] = test_coverage
    joint_config["sizes"]["test"] = len(test_labels)
    joint_config["class_rates"]["test"] = _class_rates(test_labels, 3)
    joint_config["coverage"]["test"] = test_coverage

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": str(target_device),
        "started_at_utc": started_at,
        "validation_frozen_at_utc": frozen_at,
        "completed_at_utc": _utc_now(),
    }
    full_config = {
        "pipeline_version": 2,
        "training_entrypoint": "python -m scripts.run_lstm_full",
        "stock_codes": stocks,
        "channels": list(CHANNELS),
        "splits": ranges,
        "seq_len": seq_len,
        "sizes": {
            "train": joint_data["sizes"]["train"],
            "val": joint_data["sizes"]["val"],
            "test": len(test_labels),
        },
        "coverage": {
            "train": joint_data["coverage"]["train"],
            "val": joint_data["coverage"]["val"],
            "test": test_coverage,
        },
        "class_rates": {
            "train": joint_data["class_rates"]["train"],
            "val": joint_data["class_rates"]["val"],
            "test": _class_rates(test_labels, 3),
        },
        "class_names": CLASS_NAMES,
        "move_bias": fusion["move_bias"],
        "joint_weight": fusion["joint_weight"],
        "two_stage_weight": fusion["two_stage_weight"],
        "selective_thresholds": selective_thresholds,
        "fusion_selection": {
            "objective": f"validation_{fusion_objective}",
            "move_bias_grid": move_biases.tolist(),
            "joint_weight_grid": joint_weights.tolist(),
            "validation_accuracy": fusion["validation_accuracy"],
            "validation_macro_f1": fusion["validation_macro_f1"],
        },
        "component_seeds": component_seeds,
        "runtime": runtime,
    }
    histories = {
        "direction": direction_history,
        "movement": movement_history,
        "joint": joint_history,
    }
    bundle = {
        "components": {
            "direction": {
                "state_dict": _cpu_state(direction_state),
                "config": direction_config,
            },
            "movement": {
                "state_dict": _cpu_state(movement_state),
                "config": movement_config,
            },
            "joint": {
                "state_dict": _cpu_state(joint_state),
                "config": joint_config,
            },
        },
        "config": full_config,
    }
    torch.save(bundle, target / "model.pt")
    _json_dump(target / "history.json", histories)
    _json_dump(target / "test_metrics.json", metrics)
    prediction_table = test_metadata.drop(columns="stock_id").copy()
    prediction_table["true_label"] = test_labels
    prediction_table["predicted_label"] = predicted
    prediction_table["prob_down"] = test_probability[:, 0]
    prediction_table["prob_flat"] = test_probability[:, 1]
    prediction_table["prob_up"] = test_probability[:, 2]
    prediction_table["confidence"] = test_confidence
    prediction_table["selected_balanced"] = selective_masks["balanced"]
    prediction_table["selected_strict"] = selective_masks["strict"]
    prediction_table.to_csv(target / "test_predictions.csv", index=False)
    _plot_histories(histories, target)
    _write_readme(target, full_config, metrics)
    return {
        "out_dir": str(target),
        "config": full_config,
        "test_metrics": metrics,
    }
