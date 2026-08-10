"""Magnitude-aware LSTM experiment for next-minute return ranking.

The official three-component classifier is deliberately left untouched.  This
module trains a separate shared encoder with two heads: a three-class direction
head and a signed next-minute return head.  Model/threshold selection uses only
the validation split; the test split is loaded after the checkpoint and score
thresholds have been frozen.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import OUTPUT_DIR
from .lstm_baselines import probability_metrics
from .lstm_components import DEFAULT_SPLITS, _make_split, set_seed, transform_sequences
from .lstm_ensemble_training import (
    _append_stock_id,
    _normalise_features,
    _prepare_output_dir,
    _resolve_device,
    _resolve_stocks,
    _validate_splits,
)
from .strategy_analysis import (
    _load_lstm_execution_returns,
    _minute_strategy_metrics,
    aggregate_signal_strategy,
    aggregate_t1_daily_strategy,
)


def label_returns_from_metadata(metadata: pd.DataFrame, minute_dir: Path) -> np.ndarray:
    """Read close[t+1]/close[t]-1 in exactly the sequence-table row order."""
    frame = metadata.reset_index(drop=True).copy()
    frame["window_end"] = pd.to_datetime(frame["window_end"])
    frame["target_time"] = pd.to_datetime(frame["target_time"])
    result = np.full(len(frame), np.nan, dtype=np.float64)
    for date, date_positions in frame.groupby("date", sort=True).groups.items():
        key = str(date).replace("-", "")
        close = pd.read_csv(minute_dir / "close" / f"{key}.csv", index_col=0)
        close.index = pd.to_datetime(close.index)
        date_frame = frame.loc[date_positions]
        for stock, stock_positions in date_frame.groupby("stock", sort=False).groups.items():
            rows = frame.loc[stock_positions]
            current = close.loc[rows["window_end"], str(stock)].to_numpy(dtype=np.float64)
            following = close.loc[rows["target_time"], str(stock)].to_numpy(dtype=np.float64)
            result[np.asarray(stock_positions, dtype=int)] = following / current - 1.0
    if not np.isfinite(result).all():
        raise ValueError("non-finite next-minute label returns were reconstructed")
    return result


def robust_return_transform(train_returns: np.ndarray) -> dict[str, float]:
    """Fit a train-only, zero-centred scale while limiting extreme tail leverage."""
    values = np.asarray(train_returns, dtype=np.float64) * 10_000.0
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("training returns must be finite and non-empty")
    cap = max(float(np.quantile(np.abs(values), 0.995)), 1e-6)
    clipped = np.clip(values, -cap, cap)
    scale = max(float(clipped.std(ddof=0)), 1e-6)
    return {"unit": "basis_points", "center": 0.0, "scale": scale, "cap": cap}


def transform_return_target(returns: np.ndarray, transform: dict[str, float]) -> np.ndarray:
    bps = np.asarray(returns, dtype=np.float64) * 10_000.0
    clipped = np.clip(bps, -transform["cap"], transform["cap"])
    return (clipped / transform["scale"]).astype(np.float32)


class MagnitudeAwareLSTM(nn.Module):
    """Shared residual LSTM encoder with direction and signed-return heads."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        recent_size = max(16, hidden_size // 2)
        self.recent = nn.Sequential(
            nn.LayerNorm(input_size), nn.Linear(input_size, recent_size),
            nn.GELU(), nn.Dropout(dropout),
        )
        encoded_size = hidden_size + recent_size
        self.shared = nn.Sequential(nn.LayerNorm(encoded_size), nn.Dropout(dropout))
        self.class_head = nn.Linear(encoded_size, 3)
        self.return_head = nn.Sequential(
            nn.Linear(encoded_size, max(16, hidden_size // 2)), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(max(16, hidden_size // 2), 1),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, _ = self.lstm(values)
        encoded = torch.cat([output[:, -1], self.recent(values[:, -1])], dim=1)
        encoded = self.shared(encoded)
        return self.class_head(encoded), self.return_head(encoded).squeeze(1)


def magnitude_loss(
    logits: torch.Tensor,
    predicted_return: torch.Tensor,
    labels: torch.Tensor,
    return_target: torch.Tensor,
    return_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine direction CE with magnitude-weighted robust return regression."""
    classification = nn.functional.cross_entropy(logits, labels)
    raw_regression = nn.functional.smooth_l1_loss(
        predicted_return, return_target, reduction="none", beta=0.5
    )
    # Flat/tiny observations retain weight, but large economically relevant moves
    # contribute more without allowing one tail observation to dominate training.
    observation_weight = 0.25 + torch.clamp(return_target.abs(), max=3.0) / 3.0
    regression = (raw_regression * observation_weight).sum() / observation_weight.sum()
    return classification + return_loss_weight * regression, classification, regression


def return_metrics(true_bps: np.ndarray, predicted_bps: np.ndarray) -> dict[str, float]:
    true = np.asarray(true_bps, dtype=np.float64)
    predicted = np.asarray(predicted_bps, dtype=np.float64)
    if len(true) != len(predicted) or not len(true):
        raise ValueError("return metric arrays must be aligned and non-empty")
    true_rank = pd.Series(true).rank(method="average").to_numpy(dtype=np.float64)
    pred_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=np.float64)
    pearson = float(np.corrcoef(true, predicted)[0, 1]) if true.std() and predicted.std() else 0.0
    spearman = (
        float(np.corrcoef(true_rank, pred_rank)[0, 1])
        if true_rank.std() and pred_rank.std() else 0.0
    )
    nonflat = true != 0.0
    sign_accuracy = float((np.sign(true[nonflat]) == np.sign(predicted[nonflat])).mean()) if nonflat.any() else 0.0
    return {
        "mae_bps": float(np.mean(np.abs(predicted - true))),
        "rmse_bps": float(np.sqrt(np.mean((predicted - true) ** 2))),
        "pearson_ic": pearson,
        "spearman_ic": spearman,
        "nonflat_sign_accuracy": sign_accuracy,
        "predicted_mean_bps": float(predicted.mean()),
        "predicted_std_bps": float(predicted.std(ddof=0)),
    }


def _loader(
    x: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x), torch.from_numpy(labels), torch.from_numpy(returns)
        ),
        batch_size=batch_size, shuffle=shuffle, generator=generator,
    )


def _predict(
    model: MagnitudeAwareLSTM,
    loader: DataLoader,
    device: torch.device,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probabilities, predictions, labels, targets = [], [], [], []
    with torch.no_grad():
        for x, y, target in loader:
            logits, predicted = model(x.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            predictions.append(predicted.cpu().numpy() * scale)
            labels.append(y.numpy())
            targets.append(target.numpy() * scale)
    return (
        np.concatenate(probabilities), np.concatenate(predictions),
        np.concatenate(labels), np.concatenate(targets),
    )


def _train(
    model: MagnitudeAwareLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    scale: float,
    epochs: int,
    patience: int,
    learning_rate: float,
    return_loss_weight: float,
) -> tuple[dict[str, list[float]], dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    model.to(device)
    history = {
        "train_total_loss": [], "train_classification_loss": [],
        "train_return_loss": [], "validation_accuracy": [],
        "validation_return_spearman_ic": [],
    }
    best_score: tuple[float, float] | None = None
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for epoch in range(epochs):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        count = 0
        for x, labels, target in train_loader:
            x, labels, target = x.to(device), labels.to(device), target.to(device)
            optimizer.zero_grad()
            logits, predicted = model(x)
            losses = magnitude_loss(
                logits, predicted, labels, target, return_loss_weight
            )
            losses[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals += np.asarray([float(value.item()) for value in losses]) * len(x)
            count += len(x)
        probability, predicted_bps, val_labels, true_bps = _predict(
            model, val_loader, device, scale
        )
        accuracy = float((probability.argmax(axis=1) == val_labels).mean())
        spearman = return_metrics(true_bps, predicted_bps)["spearman_ic"]
        history["train_total_loss"].append(float(totals[0] / count))
        history["train_classification_loss"].append(float(totals[1] / count))
        history["train_return_loss"].append(float(totals[2] / count))
        history["validation_accuracy"].append(accuracy)
        history["validation_return_spearman_ic"].append(spearman)
        scheduler.step(spearman)
        print(
            f"epoch {epoch + 1:02d}: loss={totals[0] / count:.5f} "
            f"val_acc={accuracy:.4f} val_return_ic={spearman:.4f}"
        )
        score = (spearman, accuracy)
        if best_score is None or score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print("early stopping")
                break
    return history, {key: value.detach().cpu() for key, value in best_state.items()}


def magnitude_thresholds(validation_score: np.ndarray) -> dict[str, float]:
    score = np.asarray(validation_score, dtype=np.float64)
    return {
        "all": 0.0,
        "balanced": max(0.0, float(np.quantile(score, 0.70))),
        "strict": max(0.0, float(np.quantile(score, 0.90))),
    }


def _prediction_frame(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    true_bps: np.ndarray,
    probability: np.ndarray,
    predicted_bps: np.ndarray,
) -> pd.DataFrame:
    frame = metadata.drop(columns="stock_id").copy()
    frame["true_label"] = labels
    frame["predicted_label"] = probability.argmax(axis=1)
    frame["prob_down"] = probability[:, 0]
    frame["prob_flat"] = probability[:, 1]
    frame["prob_up"] = probability[:, 2]
    frame["confidence"] = probability.max(axis=1)
    frame["true_return_bps"] = true_bps
    frame["predicted_return_bps"] = predicted_bps
    frame["direction_score"] = probability[:, 2] - probability[:, 0]
    return frame


def _strategy_comparison(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_root: Path,
    sell_fee_bps: float,
) -> pd.DataFrame:
    official_val = pd.read_csv(output_root / "lstm_ensemble" / "validation_predictions.csv")
    official_test = pd.read_csv(output_root / "lstm_ensemble" / "test_predictions.csv")
    keys = ["stock", "window_end", "target_time", "true_label"]
    if not test[keys].equals(official_test[keys]):
        raise ValueError("magnitude-aware and official LSTM test keys are not exactly aligned")
    score_thresholds = magnitude_thresholds(validation["predicted_return_bps"].to_numpy())
    confidence_thresholds = {
        "all": 0.0,
        "balanced": float(validation["confidence"].quantile(0.70)),
        "strict": float(validation["confidence"].quantile(0.90)),
    }
    official_thresholds = {
        "all": 0.0,
        "balanced": float(official_val["confidence"].quantile(0.70)),
        "strict": float(official_val["confidence"].quantile(0.90)),
    }
    sources = {
        "official_direction_classifier": (
            official_test,
            lambda samples, tier: samples["predicted_label"].eq(2)
            & samples["confidence"].ge(official_thresholds[tier]),
            official_thresholds,
        ),
        "multitask_direction_classifier": (
            test,
            lambda samples, tier: samples["predicted_label"].eq(2)
            & samples["confidence"].ge(confidence_thresholds[tier]),
            confidence_thresholds,
        ),
        "multitask_predicted_return": (
            test,
            lambda samples, tier: samples["predicted_return_bps"].ge(score_thresholds[tier]),
            score_thresholds,
        ),
    }
    rows: list[dict] = []
    for method, (prediction, selector, thresholds) in sources.items():
        samples = _load_lstm_execution_returns(prediction, output_root / "minute")
        for tier in ("all", "balanced", "strict"):
            selected = selector(samples, tier)
            proxy = aggregate_signal_strategy(
                samples, selected, "next_minute_open_to_close_return",
                direction="long", sell_fee_bps=sell_fee_bps,
            )
            t1 = aggregate_t1_daily_strategy(samples, selected, sell_fee_bps)
            strategy_growth = float((1.0 + t1["net_return"]).prod())
            market_growth = float((1.0 + t1["five_stock_market_return"]).prod())
            matched_growth = float((1.0 + t1["exposure_matched_market_return"]).prod())
            rows.append({
                "method": method,
                "tier": tier,
                "validation_threshold": float(thresholds[tier]),
                **_minute_strategy_metrics(proxy),
                "t1_settled_days": int(len(t1)),
                "t1_net_total_return": strategy_growth - 1.0,
                "t1_full_market_total_return": market_growth - 1.0,
                "t1_exposure_matched_market_return": matched_growth - 1.0,
                "t1_excess_vs_full_market": strategy_growth / market_growth - 1.0,
                "t1_excess_vs_exposure_matched_market": strategy_growth / matched_growth - 1.0,
            })
    return pd.DataFrame(rows)


def run_magnitude_experiment(
    *,
    stock_codes: list[str] | None = None,
    n_stocks: int = 5,
    seq_len: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 12,
    patience: int = 4,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    return_loss_weight: float = 0.25,
    sell_fee_bps: float = 5.0,
    seed: int = 42,
    splits: dict[str, tuple[str, str]] | None = None,
    data_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
) -> dict:
    """Train the validation-selected experiment, then evaluate one frozen test."""
    if return_loss_weight <= 0 or sell_fee_bps < 0:
        raise ValueError("return_loss_weight must be positive and fees non-negative")
    ranges = dict(splits or DEFAULT_SPLITS)
    _validate_splits(ranges)
    output_root = OUTPUT_DIR
    minute_dir = Path(data_dir or (output_root / "minute"))
    target = Path(out_dir or (output_root / "lstm_magnitude"))
    _prepare_output_dir(target, overwrite)
    stocks = _resolve_stocks(stock_codes, n_stocks, minute_dir)
    target_device = _resolve_device(device)
    set_seed(seed)

    print("preparing magnitude train/validation data (test remains unopened)")
    train_split = _make_split(
        stocks, ranges["train"], seq_len, minute_dir, "enhanced", "three_class", True
    )
    val_split = _make_split(
        stocks, ranges["val"], seq_len, minute_dir, "enhanced", "three_class", True
    )
    x_train, y_train, train_ids, train_coverage, train_meta = train_split
    x_val, y_val, val_ids, val_coverage, val_meta = val_split
    x_train, x_val, mean, std = _normalise_features(
        x_train, train_ids, x_val, val_ids, stocks, "per_stock"
    )
    x_train = _append_stock_id(x_train, train_ids, len(stocks))
    x_val = _append_stock_id(x_val, val_ids, len(stocks))
    train_returns = label_returns_from_metadata(train_meta, minute_dir)
    val_returns = label_returns_from_metadata(val_meta, minute_dir)
    return_transform = robust_return_transform(train_returns)
    train_target = transform_return_target(train_returns, return_transform)
    val_target = transform_return_target(val_returns, return_transform)
    train_loader = _loader(x_train, y_train, train_target, batch_size, True, seed)
    val_loader = _loader(x_val, y_val, val_target, batch_size, False, seed)
    model = MagnitudeAwareLSTM(
        x_train.shape[2], hidden_size, num_layers, dropout
    )
    history, best_state = _train(
        model, train_loader, val_loader, target_device, return_transform["scale"],
        epochs, patience, learning_rate, return_loss_weight,
    )
    model.load_state_dict(best_state, strict=True)
    model.to(target_device).eval()
    val_probability, val_predicted_bps, replayed_val, val_true_bps = _predict(
        model, val_loader, target_device, return_transform["scale"]
    )
    if not np.array_equal(replayed_val, y_val):
        raise RuntimeError("validation labels changed during replay")
    frozen_thresholds = {
        "predicted_return_bps": magnitude_thresholds(val_predicted_bps),
        "confidence": {
            "all": 0.0,
            "balanced": float(np.quantile(val_probability.max(axis=1), 0.70)),
            "strict": float(np.quantile(val_probability.max(axis=1), 0.90)),
        },
    }

    print("validation checkpoint and thresholds frozen; loading final test split")
    test_split = _make_split(
        stocks, ranges["test"], seq_len, minute_dir, "enhanced", "three_class", True
    )
    x_test, y_test, test_ids, test_coverage, test_meta = test_split
    transform_config = {
        "stock_codes": stocks, "seq_len": seq_len, "include_stock_id": True,
        "input_size": x_train.shape[2], "feature_names": [None] * x_train.shape[2],
        "scaler_mode": "per_stock", "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
    }
    x_test = transform_sequences(x_test, test_ids, transform_config, stocks)
    test_returns = label_returns_from_metadata(test_meta, minute_dir)
    test_target = transform_return_target(test_returns, return_transform)
    test_loader = _loader(x_test, y_test, test_target, batch_size, False, seed)
    test_probability, test_predicted_bps, replayed_test, test_true_bps = _predict(
        model, test_loader, target_device, return_transform["scale"]
    )
    if not np.array_equal(replayed_test, y_test):
        raise RuntimeError("test labels changed during replay")

    validation_frame = _prediction_frame(
        val_meta, y_val, val_true_bps, val_probability, val_predicted_bps
    )
    test_frame = _prediction_frame(
        test_meta, y_test, test_true_bps, test_probability, test_predicted_bps
    )
    validation_frame.to_csv(target / "validation_predictions.csv", index=False)
    test_frame.to_csv(target / "test_predictions.csv", index=False)
    comparison = _strategy_comparison(
        validation_frame, test_frame, output_root, sell_fee_bps
    )
    comparison.to_csv(target / "strategy_comparison.csv", index=False, float_format="%.10f")

    validation_metrics = {
        "classification": probability_metrics(y_val, val_probability),
        "return": return_metrics(val_true_bps, val_predicted_bps),
    }
    test_metrics = {
        "classification": probability_metrics(y_test, test_probability),
        "return": return_metrics(test_true_bps, test_predicted_bps),
    }
    config = {
        "experiment": "shared_lstm_direction_plus_signed_return",
        "status": "exploratory_not_official_model",
        "stock_codes": stocks,
        "splits": ranges,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs_requested": epochs,
        "best_epoch_by_validation_return_ic": int(
            np.argmax(history["validation_return_spearman_ic"]) + 1
        ),
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "return_loss_weight": return_loss_weight,
        "return_observation_weight": "0.25 + min(abs(standardized_return), 3) / 3",
        "return_transform": return_transform,
        "selection": "validation return Spearman IC, accuracy as tie-break",
        "thresholds_frozen_on_validation": frozen_thresholds,
        "sell_fee_bps": sell_fee_bps,
        "coverage": {"train": train_coverage, "val": val_coverage, "test": test_coverage},
        "seed": seed,
    }
    bundle = {"state_dict": best_state, "config": config, "scaler_mean": mean, "scaler_std": std}
    torch.save(bundle, target / "model.pt")
    (target / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    result = {
        "config": config,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "strategy_rows": comparison.to_dict(orient="records"),
    }
    (target / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
