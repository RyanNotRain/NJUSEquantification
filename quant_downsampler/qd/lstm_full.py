"""Strict loading and inference for the full-window LSTM ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .lstm_model import MinuteLSTM, _make_split, classification_metrics


REQUIRED_COMPONENTS = ("direction", "movement", "joint")


def _shift_move_probability(probability: np.ndarray, bias: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)) - bias
    return 1.0 / (1.0 + np.exp(-logit))


def two_stage_probabilities(
    move_probability: np.ndarray,
    up_given_move_probability: np.ndarray,
    move_bias: float = 0.0,
) -> np.ndarray:
    """Return down/flat/up probabilities from movement and direction models."""
    move = _shift_move_probability(move_probability, move_bias)
    up = np.asarray(up_given_move_probability, dtype=np.float64)
    if move.shape != up.shape:
        raise ValueError("movement and direction probability shapes differ")
    if not np.isfinite(up).all() or ((up < 0.0) | (up > 1.0)).any():
        raise ValueError("direction probabilities must be finite and within [0, 1]")
    result = np.column_stack((move * (1.0 - up), 1.0 - move, move * up))
    return result.astype(np.float32)


def blend_probabilities(
    joint_probability: np.ndarray,
    two_stage_probability: np.ndarray,
    joint_weight: float,
) -> np.ndarray:
    """Blend two aligned three-class probability matrices."""
    joint = np.asarray(joint_probability, dtype=np.float64)
    staged = np.asarray(two_stage_probability, dtype=np.float64)
    if joint.shape != staged.shape or joint.ndim != 2 or joint.shape[1] != 3:
        raise ValueError("both probability matrices must have aligned shape (n, 3)")
    if not 0.0 <= joint_weight <= 1.0:
        raise ValueError("joint_weight must be within [0, 1]")
    result = joint_weight * joint + (1.0 - joint_weight) * staged
    row_sum = result.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or (row_sum <= 0.0).any():
        raise ValueError("invalid component probabilities")
    return (result / row_sum).astype(np.float32)


def _component_input(
    sequences: np.ndarray,
    stock_ids: np.ndarray,
    config: dict,
    stock_codes: list[str],
) -> np.ndarray:
    x = np.asarray(sequences, dtype=np.float32).copy()
    ids = np.asarray(stock_ids, dtype=np.int64)
    if x.ndim != 3 or len(x) != len(ids):
        raise ValueError("sequences must have shape (n, seq_len, features) and align with stock_ids")
    if x.shape[1] != int(config["seq_len"]):
        raise ValueError(
            f"component expects sequence length {config['seq_len']}, got {x.shape[1]}"
        )
    if len(ids) and (ids.min() < 0 or ids.max() >= len(stock_codes)):
        raise ValueError("stock_ids contain an out-of-range value")

    include_stock_id = bool(config["include_stock_id"])
    input_size = int(config.get("input_size", len(config["feature_names"])))
    base_size = input_size - (len(stock_codes) if include_stock_id else 0)
    if x.shape[2] != base_size:
        raise ValueError(
            f"component expects {base_size} raw features, got {x.shape[2]}"
        )
    mean = np.asarray(config["scaler_mean"], dtype=np.float32)
    std = np.asarray(config["scaler_std"], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("component scaler is invalid")
    if config["scaler_mode"] == "global":
        if mean.shape != (base_size,) or std.shape != (base_size,):
            raise ValueError("global scaler shape does not match the component")
        x = (x - mean[None, None, :]) / std[None, None, :]
    elif config["scaler_mode"] == "per_stock":
        expected = (len(stock_codes), base_size)
        if mean.shape != expected or std.shape != expected:
            raise ValueError("per-stock scaler shape does not match the component")
        x = (x - mean[ids, None, :]) / std[ids, None, :]
    else:
        raise ValueError(f"unknown scaler_mode={config['scaler_mode']!r}")

    if include_stock_id:
        one_hot = np.eye(len(stock_codes), dtype=np.float32)[ids]
        one_hot = np.broadcast_to(one_hot[:, None, :], (len(x), x.shape[1], len(stock_codes)))
        x = np.concatenate((x, one_hot), axis=2)
    return x.astype(np.float32, copy=False)


def _predict(model: MinuteLSTM, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False
    )
    result: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch,) in loader:
            result.append(torch.softmax(model(batch.to(device)), dim=1).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, model.head[-1].out_features))


@dataclass
class FullLSTMEnsemble:
    """Loaded three-component ensemble with saved preprocessing parameters."""

    models: dict[str, MinuteLSTM]
    component_configs: dict[str, dict]
    config: dict
    device: torch.device

    def predict_from_features(
        self,
        legacy_sequences: np.ndarray,
        enhanced_sequences: np.ndarray,
        stock_ids: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Predict aligned down/flat/up probabilities from unscaled features."""
        stocks = list(self.config["stock_codes"])
        direction_x = _component_input(
            legacy_sequences, stock_ids, self.component_configs["direction"], stocks
        )
        movement_x = _component_input(
            enhanced_sequences, stock_ids, self.component_configs["movement"], stocks
        )
        joint_x = _component_input(
            enhanced_sequences, stock_ids, self.component_configs["joint"], stocks
        )
        direction = _predict(self.models["direction"], direction_x, batch_size, self.device)
        movement = _predict(self.models["movement"], movement_x, batch_size, self.device)
        joint = _predict(self.models["joint"], joint_x, batch_size, self.device)
        staged = two_stage_probabilities(
            movement[:, 1], direction[:, 1], float(self.config["move_bias"])
        )
        return blend_probabilities(joint, staged, float(self.config["joint_weight"]))


def load_full_model(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> FullLSTMEnsemble:
    """Load every component strictly; fail on a stale or incompatible bundle."""
    target_device = torch.device(device)
    bundle = torch.load(Path(path), map_location=target_device, weights_only=False)
    if set(bundle) != {"components", "config"}:
        raise ValueError("full LSTM bundle must contain exactly components and config")
    components = bundle["components"]
    if set(components) != set(REQUIRED_COMPONENTS):
        raise ValueError(f"full LSTM components must be {REQUIRED_COMPONENTS}")
    config = bundle["config"]
    if config.get("class_names") != ["down", "flat", "up"]:
        raise ValueError("full LSTM class order must be down/flat/up")

    models: dict[str, MinuteLSTM] = {}
    component_configs: dict[str, dict] = {}
    for name in REQUIRED_COMPONENTS:
        component = components[name]
        if set(component) != {"state_dict", "config"}:
            raise ValueError(f"component {name!r} is incomplete")
        cfg = component["config"]
        required = {
            "feature_names", "feature_set", "include_stock_id", "scaler_mode", "scaler_mean",
            "scaler_std", "hidden_size", "num_layers", "dropout",
            "model_version", "num_classes", "class_names", "seq_len", "target_mode",
        }
        if not required.issubset(cfg):
            raise ValueError(f"component {name!r} config is incomplete")
        input_size = int(cfg.get("input_size", len(cfg["feature_names"])))
        model = MinuteLSTM(
            input_size=input_size,
            hidden_size=int(cfg["hidden_size"]),
            num_layers=int(cfg["num_layers"]),
            dropout=float(cfg["dropout"]),
            model_version=str(cfg["model_version"]),
            num_classes=int(cfg["num_classes"]),
        )
        model.load_state_dict(component["state_dict"], strict=True)
        model.to(target_device).eval()
        models[name] = model
        component_configs[name] = cfg
    expected_semantics = {
        "direction": ("nonflat_binary", "legacy", 2),
        "movement": ("move_vs_flat", "enhanced", 2),
        "joint": ("three_class", "enhanced", 3),
    }
    for name, (target_mode, feature_set, num_classes) in expected_semantics.items():
        cfg = component_configs[name]
        if (
            cfg["target_mode"] != target_mode
            or cfg["feature_set"] != feature_set
            or int(cfg["num_classes"]) != num_classes
        ):
            raise ValueError(f"component {name!r} has incompatible target/features/classes")
    sequence_lengths = {
        int(component_configs[name]["seq_len"]) for name in REQUIRED_COMPONENTS
    }
    if len(sequence_lengths) != 1:
        raise ValueError("full LSTM components use different sequence lengths")
    if component_configs["direction"]["num_classes"] != 2:
        raise ValueError("direction component must be binary")
    if component_configs["movement"]["num_classes"] != 2:
        raise ValueError("movement component must be binary")
    if component_configs["joint"]["num_classes"] != 3:
        raise ValueError("joint component must have three classes")
    return FullLSTMEnsemble(models, component_configs, config, target_device)


def evaluate_full_model(
    model_path: str | Path,
    data_dir: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> dict:
    """Recreate an out-of-time split and evaluate the saved ensemble."""
    ensemble = load_full_model(model_path, device=device)
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    stocks = list(ensemble.config["stock_codes"])
    date_range = tuple(ensemble.config["splits"][split])
    seq_len = int(ensemble.component_configs["joint"]["seq_len"])
    legacy_x, legacy_y, legacy_ids, legacy_coverage, legacy_metadata = _make_split(
        stocks, date_range, seq_len, Path(data_dir), "legacy", "three_class", True
    )
    enhanced_x, enhanced_y, enhanced_ids, enhanced_coverage, enhanced_metadata = _make_split(
        stocks, date_range, seq_len, Path(data_dir), "enhanced", "three_class", True
    )
    if (
        not np.array_equal(legacy_y, enhanced_y)
        or not np.array_equal(legacy_ids, enhanced_ids)
        or legacy_coverage != enhanced_coverage
        or not legacy_metadata.equals(enhanced_metadata)
    ):
        raise ValueError("legacy and enhanced full-window samples are not aligned")
    probability = ensemble.predict_from_features(
        legacy_x, enhanced_x, enhanced_ids, batch_size=batch_size
    )
    metrics = classification_metrics(enhanced_y, probability, threshold=None)
    counts = np.bincount(enhanced_y, minlength=3)
    metrics["majority_baseline"] = float(counts.max() / counts.sum())
    metrics["coverage"] = enhanced_coverage
    return {
        "metrics": metrics,
        "probability": probability,
        "labels": enhanced_y,
        "stock_ids": enhanced_ids,
        "metadata": enhanced_metadata,
    }
