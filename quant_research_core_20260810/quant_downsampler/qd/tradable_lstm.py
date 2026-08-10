"""Exploratory T+1 LSTM selected after validation-only horizon screening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .config import OUTPUT_DIR
from .lstm_baselines import probability_metrics
from .lstm_components import ENHANCED_FEATURE_NAMES, _make_split, set_seed, transform_sequences
from .lstm_ensemble_training import _append_stock_id, _normalise_features, _resolve_device
from .lstm_magnitude import (
    MagnitudeAwareLSTM,
    _loader,
    _predict,
    _train,
    return_metrics,
    robust_return_transform,
    transform_return_target,
)
from .tradable_return_research import (
    _strategy_path,
    attach_horizon_returns,
    select_strategy_on_validation,
    strategy_metrics,
)


def _screened_horizon(root: Path) -> str:
    grid = pd.read_csv(
        root / "tradable_return_research" / "validation_strategy_selection.csv"
    )
    eligible = grid[(grid["active_periods"] >= 10) & (grid["coverage"] >= 0.05)]
    if eligible.empty:
        raise ValueError("validation horizon screen contains no eligible strategy")
    return str(eligible.sort_values(
        ["excess_vs_matched_market", "net_total_return"],
        ascending=[False, False],
    ).iloc[0]["horizon"])


def _raw_split(
    stocks: list[str],
    date_range: tuple[str, str],
    seq_len: int,
    minute_dir: Path,
) -> tuple:
    return _make_split(
        stocks, date_range, seq_len, minute_dir,
        "enhanced", "three_class", True,
    )


def run_tradable_lstm(
    output_dir: str | Path = OUTPUT_DIR,
    epochs: int = 8,
    patience: int = 3,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.15,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    return_loss_weight: float = 0.30,
    sell_fee_bps: float = 5.0,
    seed: int = 42,
    device: str = "cpu",
    selected_feature_names: Sequence[str] | None = None,
    target_subdir: str = "tradable_lstm",
    model_name: str = "compact_multitask_lstm",
    feature_selection_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train a compact shared-head LSTM on the validation-screened horizon."""
    root = Path(output_dir)
    target = root / target_subdir
    target.mkdir(parents=True, exist_ok=True)
    horizon = _screened_horizon(root)
    ensemble = torch.load(
        root / "lstm_ensemble" / "model.pt", map_location="cpu", weights_only=False
    )["config"]
    stocks = list(ensemble["stock_codes"])
    splits = {name: tuple(value) for name, value in ensemble["splits"].items()}
    seq_len = int(ensemble["seq_len"])
    minute_dir = root / "minute"
    target_device = _resolve_device(device)
    set_seed(seed)

    print(f"screened horizon={horizon}; loading train/validation only")
    train = _raw_split(stocks, splits["train"], seq_len, minute_dir)
    validation = _raw_split(stocks, splits["val"], seq_len, minute_dir)
    train_x, train_labels, train_ids, _, train_metadata = train
    val_x, val_labels, val_ids, _, val_metadata = validation
    declared_features = list(ENHANCED_FEATURE_NAMES)
    chosen_features = list(selected_feature_names or declared_features)
    if len(chosen_features) != len(set(chosen_features)) or not set(chosen_features).issubset(declared_features):
        raise ValueError("selected_feature_names must be unique enhanced feature names")
    feature_indices = [declared_features.index(name) for name in chosen_features]
    train_x = train_x[:, :, feature_indices]
    val_x = val_x[:, :, feature_indices]
    train_returns = attach_horizon_returns(train_metadata, minute_dir)
    val_returns = attach_horizon_returns(val_metadata, minute_dir)
    train_mask = train_returns[horizon].notna().to_numpy()
    val_mask = val_returns[horizon].notna().to_numpy()
    train_x, val_x, mean, std = _normalise_features(
        train_x, train_ids, val_x, val_ids, stocks, "per_stock"
    )
    train_x = _append_stock_id(train_x[train_mask], train_ids[train_mask], len(stocks))
    val_x = _append_stock_id(val_x[val_mask], val_ids[val_mask], len(stocks))
    train_labels = train_labels[train_mask]
    val_labels = val_labels[val_mask]
    train_fraction = train_returns.loc[train_mask, horizon].to_numpy(float)
    val_fraction = val_returns.loc[val_mask, horizon].to_numpy(float)
    return_transform = robust_return_transform(train_fraction)
    train_target = transform_return_target(train_fraction, return_transform)
    val_target = transform_return_target(val_fraction, return_transform)
    train_loader = _loader(
        train_x, train_labels, train_target, batch_size, True, seed
    )
    val_loader = _loader(val_x, val_labels, val_target, batch_size, False, seed)
    model = MagnitudeAwareLSTM(
        train_x.shape[2], hidden_size, num_layers, dropout
    )
    history, state = _train(
        model, train_loader, val_loader, target_device,
        return_transform["scale"], epochs, patience, learning_rate,
        return_loss_weight,
    )
    model.load_state_dict(state, strict=True)
    model.to(target_device).eval()
    val_probability, val_prediction, replayed_val, val_true = _predict(
        model, val_loader, target_device, return_transform["scale"]
    )
    if not np.array_equal(replayed_val, val_labels):
        raise RuntimeError("validation replay changed labels")
    val_frame = val_returns.loc[val_mask].copy()
    prediction_column = "tradable_lstm_prediction_bps"
    val_frame[prediction_column] = val_prediction
    strategy_selection, strategy_grid = select_strategy_on_validation(
        val_frame, horizon, prediction_column, sell_fee_bps
    )
    strategy_grid.to_csv(target / "validation_strategy_selection.csv", index=False)
    freeze = {
        "status": "frozen_before_test",
        "horizon_source": "validation-only classical horizon screen",
        "horizon": horizon,
        "strategy_selection": strategy_selection,
        "checkpoint_selection": "validation return Spearman IC, accuracy tie-break",
        "return_transform": return_transform,
        "architecture": {
            "hidden_size": hidden_size, "num_layers": num_layers,
            "dropout": dropout, "epochs_requested": epochs,
        },
        "input_feature_names": chosen_features,
        "input_feature_count": len(chosen_features),
        "feature_selection": dict(feature_selection_metadata or {
            "method": "all enhanced features",
            "selection_split": "none",
        }),
        "splits": splits,
        "known_limitation": "the fixed historical test dates were inspected in earlier research",
    }
    (target / "selection_frozen_before_test.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("T+1 LSTM checkpoint and strategy frozen; loading test split")
    test = _raw_split(stocks, splits["test"], seq_len, minute_dir)
    test_x, test_labels, test_ids, _, test_metadata = test
    test_x = test_x[:, :, feature_indices]
    test_returns = attach_horizon_returns(test_metadata, minute_dir)
    test_mask = test_returns[horizon].notna().to_numpy()
    transform_config = {
        "stock_codes": stocks, "seq_len": seq_len, "include_stock_id": True,
        "input_size": len(chosen_features) + len(stocks), "feature_names": chosen_features,
        "scaler_mode": "per_stock", "scaler_mean": mean.tolist(),
        "scaler_std": std.tolist(),
    }
    test_x = transform_sequences(test_x, test_ids, transform_config, stocks)
    test_x = test_x[test_mask]
    test_labels = test_labels[test_mask]
    test_fraction = test_returns.loc[test_mask, horizon].to_numpy(float)
    test_target = transform_return_target(test_fraction, return_transform)
    test_loader = _loader(test_x, test_labels, test_target, batch_size, False, seed)
    test_probability, test_prediction, replayed_test, test_true = _predict(
        model, test_loader, target_device, return_transform["scale"]
    )
    if not np.array_equal(replayed_test, test_labels):
        raise RuntimeError("test replay changed labels")
    test_frame = test_returns.loc[test_mask].copy()
    test_frame[prediction_column] = test_prediction
    chosen = strategy_selection
    path = _strategy_path(
        test_frame, horizon, prediction_column,
        float(chosen["threshold_bps"]), int(chosen["top_k"]),
        int(chosen["interval_minutes"]), sell_fee_bps,
    )
    lstm_strategy = {
        "horizon": horizon,
        "model": model_name,
        "threshold_bps": float(chosen["threshold_bps"]),
        "top_k": int(chosen["top_k"]),
        "interval_minutes": int(chosen["interval_minutes"]),
        **strategy_metrics(path, horizon == "t1_same_minute_open"),
    }
    baselines = pd.read_csv(
        root / "tradable_return_research" / "test_strategy_metrics.csv"
    )
    comparison = pd.concat([
        baselines[baselines["horizon"].eq(horizon)],
        pd.DataFrame([lstm_strategy]),
    ], ignore_index=True, sort=False)
    comparison.to_csv(target / "strategy_comparison.csv", index=False, float_format="%.10f")
    prediction_export = test_frame.drop(columns="stock_id", errors="ignore").copy()
    prediction_export["true_label"] = test_labels
    prediction_export["predicted_label"] = test_probability.argmax(axis=1)
    prediction_export["prob_down"] = test_probability[:, 0]
    prediction_export["prob_flat"] = test_probability[:, 1]
    prediction_export["prob_up"] = test_probability[:, 2]
    prediction_export.to_csv(target / "test_predictions.csv", index=False)
    result = {
        "status": "completed",
        "horizon": horizon,
        "model": model_name,
        "input_feature_names": chosen_features,
        "input_feature_count": len(chosen_features),
        "validation_return_metrics": return_metrics(val_true, val_prediction),
        "test_classification_metrics": probability_metrics(test_labels, test_probability),
        "test_return_metrics": return_metrics(test_true, test_prediction),
        "test_strategy": lstm_strategy,
        "comparison": comparison.to_dict(orient="records"),
    }
    (target / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (target / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    torch.save({
        "state_dict": state, "config": freeze,
        "scaler_mean": mean, "scaler_std": std,
    }, target / "model.pt")
    return result
