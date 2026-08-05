"""LSTM 分钟涨跌预测模块 (PyTorch 版)。

端到端的 LSTM 模型,预测下一个分钟股价的涨跌。

模型架构:
  - 输入: 过去 N 分钟的 OHLCV 序列 (N=60)
  - LSTM: 2 层, hidden_size=128
  - 全连接: 输出 2 分类 (涨/跌)
  - 损失: CrossEntropyLoss
  - 优化器: Adam

用法:
  python scripts/run_lstm.py --stocks 10 --epochs 50
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_minute_data_for_stocks(
    stock_codes: list[str],
    date_range: tuple[str, str] | None = None,
    data_dir: Path | None = None,
    channels: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> dict[str, np.ndarray]:
    """加载多只股票的分钟数据,返回 {stock_code: (T_days, 237, n_channels)}。"""
    data_dir = data_dir or (OUTPUT_DIR / "minute")

    all_dates = sorted(p.stem for p in (data_dir / "close").glob("*.csv"))
    if date_range:
        start, end = date_range
        all_dates = [d for d in all_dates if start <= d <= end]

    result: dict[str, np.ndarray] = {}
    for stock in stock_codes:
        stock_data = []
        for date in all_dates:
            day_data = []
            for ch in channels:
                path = data_dir / ch / f"{date}.csv"
                if not path.exists():
                    day_data = []
                    break
                df = pd.read_csv(path, index_col=0)
                if stock not in df.columns:
                    day_data = []
                    break
                day_data.append(df[stock].values)
            if len(day_data) == len(channels):
                stock_data.append(np.column_stack(day_data))
        if stock_data:
            result[stock] = np.array(stock_data)  # (T, 237, C)
    return result


def create_sequences(
    data: np.ndarray,
    seq_len: int = 60,
    pred_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """创建训练序列，逐日处理以避免跨越交易日边界。

    data: (T_days, 237, n_features)
    seq_len: 输入序列长度（分钟数）
    pred_horizon: 预测未来第几个分钟（1=下一分钟）

    Returns:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)  0=跌, 1=涨
    """
    T, bars_per_day, C = data.shape
    X_list, y_list = [], []

    for day in range(T):
        day_data = data[day]  # (237, C)

        # 移除含 NaN 的行
        valid_mask = np.isfinite(day_data).all(axis=1)
        if valid_mask.sum() < seq_len + pred_horizon:
            continue
        day_data = day_data[valid_mask]
        close = day_data[:, 3]  # close 索引

        n = len(day_data)
        for t in range(seq_len, n - pred_horizon):
            seq = day_data[t - seq_len : t]
            if np.isfinite(seq).all():
                X_list.append(seq)
                y_list.append(1 if close[t + pred_horizon] > close[t] else 0)

    if not X_list:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# PyTorch 模型
# ---------------------------------------------------------------------------

class MinuteLSTM(nn.Module):
    """LSTM 分钟涨跌预测模型。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last_out)


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    epochs: int = 50,
    lr: float = 0.001,
    device: torch.device | None = None,
) -> dict[str, list[float]]:
    """训练模型。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(X_batch)

        avg_loss = total_loss / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            val_acc = val_metrics["accuracy"]
            history["val_acc"].append(val_acc)
            scheduler.step(val_acc)
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc*100:.1f}%")
        else:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

    return history


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, float]:
    """评估模型，返回准确率、精确率、召回率、F1。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# ---------------------------------------------------------------------------
# 绘图与保存
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    out_dir: Path | None = None,
) -> Path:
    """绘制训练曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = out_dir or (OUTPUT_DIR / "lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], color="#1f77b4", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    if history.get("val_acc"):
        axes[1].plot(history["val_acc"], color="#ff7f0e", linewidth=1.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    fig.tight_layout()
    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存: {path}")
    return path


def save_results(
    result: dict,
    out_dir: Path | None = None,
) -> None:
    """保存模型权重和评估结果。"""
    import json

    out_dir = out_dir or (OUTPUT_DIR / "lstm")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    model_path = out_dir / "model.pt"
    torch.save(result["model"].state_dict(), model_path)
    print(f"模型已保存: {model_path}")

    # 保存训练历史
    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(result["history"], f, indent=2)
    print(f"训练历史已保存: {history_path}")

    # 保存测试结果
    if result.get("test_metrics"):
        metrics = result["test_metrics"].copy()
        metrics_path = out_dir / "test_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"测试指标已保存: {metrics_path}")


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def prepare_dataloaders(
    stock_codes: list[str],
    seq_len: int = 60,
    train_dates: tuple[str, str] = ("20250401", "20260531"),
    val_dates: tuple[str, str] = ("20260601", "20260615"),
    test_dates: tuple[str, str] = ("20260616", "20260630"),
    batch_size: int = 256,
    data_dir: Path | None = None,
) -> dict:
    """准备 DataLoader。"""
    print(f"加载 {len(stock_codes)} 只股票...")

    train_raw = load_minute_data_for_stocks(stock_codes, train_dates, data_dir)
    val_raw = load_minute_data_for_stocks(stock_codes, val_dates, data_dir)
    test_raw = load_minute_data_for_stocks(stock_codes, test_dates, data_dir)

    print(f"  训练: {len(train_raw)} 只, 验证: {len(val_raw)} 只, 测试: {len(test_raw)} 只")

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    for data, X_l, y_l in [
        (train_raw, X_train_list, y_train_list),
        (val_raw, X_val_list, y_val_list),
        (test_raw, X_test_list, y_test_list),
    ]:
        for stock_data in data.values():
            X, y = create_sequences(stock_data, seq_len)
            if len(X) > 0:
                X_l.append(X)
                y_l.append(y)

    X_train = np.concatenate(X_train_list) if X_train_list else np.array([], dtype=np.float32)
    y_train = np.concatenate(y_train_list) if y_train_list else np.array([], dtype=np.int64)
    X_val = np.concatenate(X_val_list) if X_val_list else np.array([], dtype=np.float32)
    y_val = np.concatenate(y_val_list) if y_val_list else np.array([], dtype=np.int64)
    X_test = np.concatenate(X_test_list) if X_test_list else np.array([], dtype=np.float32)
    y_test = np.concatenate(y_test_list) if y_test_list else np.array([], dtype=np.int64)

    # 标准化
    if len(X_train) > 0:
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        if len(X_val) > 0:
            X_val = (X_val - mean) / std
        if len(X_test) > 0:
            X_test = (X_test - mean) / std

    print(f"  训练: {len(X_train)} 样本, 验证: {len(X_val)} 样本, 测试: {len(X_test)} 样本")

    if len(y_train) > 0:
        print(f"  涨跌比: {y_train.mean()*100:.1f}% 涨")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True,
    ) if len(X_train) > 0 else None

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=batch_size,
    ) if len(X_val) > 0 else None

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=batch_size,
    ) if len(X_test) > 0 else None

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "n_features": X_train.shape[2] if len(X_train) > 0 else 5,
    }


def run_lstm_pipeline(
    stock_codes: list[str] | None = None,
    n_stocks: int = 10,
    seq_len: int = 60,
    hidden_size: int = 128,
    epochs: int = 50,
    batch_size: int = 256,
    data_dir: Path | None = None,
) -> dict:
    """运行完整 LSTM 流水线。"""
    if stock_codes is None:
        data_dir = data_dir or (OUTPUT_DIR / "minute")
        available = sorted(p.stem for p in (data_dir / "close").glob("*.csv"))
        if available:
            first_file = data_dir / "close" / f"{available[0]}.csv"
            if first_file.exists():
                all_stocks = pd.read_csv(first_file, index_col=0).columns.tolist()
                stock_codes = all_stocks[:n_stocks]
                print(f"选择前 {n_stocks} 只: {stock_codes[:5]}...")

    if not stock_codes:
        raise ValueError("没有股票")

    data = prepare_dataloaders(stock_codes, seq_len, batch_size=batch_size, data_dir=data_dir)

    if data["train_loader"] is None:
        raise ValueError("无训练数据")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")

    model = MinuteLSTM(
        input_size=data["n_features"],
        hidden_size=hidden_size,
        num_layers=2,
        num_classes=2,
        dropout=0.2,
    )
    print(f"模型: LSTM(in={data['n_features']}, hidden={hidden_size}, layers=2)")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练 {epochs} 轮...\n")

    history = train_model(model, data["train_loader"], data["val_loader"], epochs, device=device)

    test_metrics = None
    if data["test_loader"] is not None:
        test_metrics = evaluate(model, data["test_loader"], device)
        test_acc = test_metrics["accuracy"]
        print(f"\n测试集结果:")
        print(f"  准确率 (Accuracy):  {test_acc*100:.2f}%")
        print(f"  精确率 (Precision): {test_metrics['precision']*100:.2f}%")
        print(f"  召回率 (Recall):    {test_metrics['recall']*100:.2f}%")
        print(f"  F1 分数:            {test_metrics['f1']:.4f}")
        print(f"  混淆矩阵:           {test_metrics['confusion_matrix']}")

        # 基准
        if data["test_loader"].dataset is not None:
            y_test = data["test_loader"].dataset.tensors[1].numpy()
            baseline = max(y_test.mean(), 1 - y_test.mean())
            print(f"  基准(多数类):       {baseline*100:.2f}%")
            print(f"  相对提升:           {(test_acc - baseline) / baseline * 100:+.1f}%")

    # 绘图与保存
    plot_training_history(history)
    save_results({
        "model": model,
        "history": history,
        "test_accuracy": test_acc if test_metrics else np.nan,
        "test_metrics": test_metrics,
    })

    return {
        "model": model,
        "history": history,
        "test_accuracy": test_acc,
        "test_metrics": test_metrics,
    }