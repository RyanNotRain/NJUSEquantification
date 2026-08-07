"""Validate all generated artifacts against the README invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qd.adjfactor import code_mapping
from qd.config import METRICS, OUTPUT_DIR, PRICE_METRICS
from qd.data_loader import list_trading_dates, load_one_stock
from qd.factors import FACTOR_NAMES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_daily(output_dir: Path, dates: list[str]) -> dict:
    tables = {
        metric: pd.read_csv(output_dir / "daily" / f"{metric}.csv", index_col=0)
        for metric in METRICS
    }
    reference = tables["close"]
    if reference.shape != (len(dates), 300):
        raise AssertionError(f"daily shape {reference.shape}")
    if any("." not in code for code in reference.columns):
        raise AssertionError("daily columns must use full market-suffixed codes")
    for metric, table in tables.items():
        if not table.index.equals(reference.index) or not table.columns.equals(reference.columns):
            raise AssertionError(f"daily axes mismatch: {metric}")
    zero = tables["volume"] == 0
    previous_close = tables["close"].shift(1)
    for metric in PRICE_METRICS:
        mismatch = zero & previous_close.notna() & ~np.isclose(
            tables[metric], previous_close, equal_nan=True, rtol=0, atol=1e-6
        )
        if mismatch.to_numpy().any():
            where = np.argwhere(mismatch.to_numpy())[0]
            raise AssertionError(f"daily suspension fill mismatch {metric} at {where}")
    return {
        "shape": list(reference.shape),
        "zero_volume_cells": int(zero.to_numpy().sum()),
        "columns_with_suffix": 300,
    }


def validate_minute(output_dir: Path, dates: list[str], full: bool) -> dict:
    for metric in METRICS:
        files = sorted((output_dir / "minute" / metric).glob("*.csv"))
        if len(files) != len(dates):
            raise AssertionError(f"minute/{metric} file count {len(files)}")
        for path in files:
            if len(pd.read_csv(path, usecols=[0])) != 242:
                raise AssertionError(f"minute row count: {path}")

    check_dates = dates if full else sorted(set(
        [dates[0], dates[len(dates) // 2], dates[-1]]
    ))
    checked = 0
    for date in check_dates:
        tables = {
            metric: pd.read_csv(
                output_dir / "minute" / metric / f"{date}.csv", index_col=0
            )
            for metric in (*PRICE_METRICS, "volume")
        }
        if tables["close"].shape != (242, 300):
            raise AssertionError(f"minute shape on {date}: {tables['close'].shape}")
        idx = pd.to_datetime(tables["close"].index)
        if idx[0].strftime("%H:%M:%S") != "09:30:00" or idx[-1].strftime("%H:%M:%S") != "15:00:00":
            raise AssertionError(f"minute time range on {date}")
        if not tables["close"].iloc[-4:-1].isna().all().all():
            raise AssertionError(f"call-auction waiting prices must be NaN on {date}")
        if not (tables["volume"].iloc[-4:-1] == 0).all().all():
            raise AssertionError(f"call-auction waiting volume must be zero on {date}")
        continuous = list(range(0, 121)) + list(range(121, 238))
        zero = tables["volume"].iloc[continuous] == 0
        previous = tables["close"].shift(1)
        previous.iloc[121] = tables["close"].iloc[120]
        for metric in PRICE_METRICS:
            lhs = tables[metric].iloc[continuous]
            rhs = previous.iloc[continuous]
            mismatch = zero & rhs.notna() & ~np.isclose(
                lhs, rhs, equal_nan=True, rtol=0, atol=1e-6
            )
            if mismatch.to_numpy().any():
                raise AssertionError(f"minute fill mismatch {metric} on {date}")
        checked += 1
    return {"files_per_metric": len(dates), "shape": [242, 300], "fully_checked_dates": checked}


def validate_raw_sample(output_dir: Path) -> dict:
    date, data_code, full_code = "20250401", "000012", "000012.SZ"
    raw = load_one_stock(date, data_code)
    assert raw is not None
    daily_volume = pd.read_csv(output_dir / "daily" / "volume.csv", index_col=0)
    minute_volume = pd.read_csv(
        output_dir / "minute" / "volume" / f"{date}.csv", index_col=0
    )
    opening = raw.loc[raw["time_sec"] < 9 * 3600 + 30 * 60, "Volume"].sum()
    if daily_volume.loc["2025-04-01", full_code] != raw["Volume"].sum():
        raise AssertionError("daily raw-volume sample mismatch")
    expected_minute = raw["Volume"].sum() - opening
    if minute_volume[full_code].sum() != expected_minute:
        raise AssertionError("minute raw-volume sample mismatch")
    return {"date": date, "stock": full_code, "opening_auction_excluded": int(opening)}


def validate_factors(output_dir: Path) -> dict:
    root = output_dir / "factors"
    summary = pd.read_csv(root / "evaluation_summary.csv", index_col=0)
    if set(summary.index) != set(FACTOR_NAMES) or len(summary) != 4:
        raise AssertionError("factor summary must contain example + exactly three new factors")
    required = {"IC", "IR", "ICIR", "rank_IC", "rank_IR", "rank_ICIR", "n_days"}
    if not required.issubset(summary.columns):
        raise AssertionError("factor summary metrics missing")
    if not np.allclose(summary["IR"], summary["ICIR"] * np.sqrt(252)):
        raise AssertionError("IR/ICIR definition mismatch")
    for name in (*FACTOR_NAMES, "forward_return_1d"):
        table = pd.read_csv(root / f"{name}.csv", index_col=0)
        if table.shape != (302, 300):
            raise AssertionError(f"factor shape mismatch: {name}")
    return {"factor_count": 4, "summary_columns": sorted(required)}


def validate_backtest(output_dir: Path) -> dict:
    root = output_dir / "backtest"
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    strategies = {"corrected_close", "corrected_open", "naive_close", "naive_open"}
    if set(metrics) != strategies:
        raise AssertionError("backtest strategy set mismatch")
    volume = pd.read_csv(output_dir / "daily" / "volume.csv", index_col=0)
    volume.index = pd.to_datetime(volume.index)
    locked_counts = {}
    for name in strategies:
        selections = pd.read_csv(root / f"selections_{name}.csv")
        selections["execution_date"] = pd.to_datetime(selections["execution_date"])
        new_positions = selections[~selections["locked"].astype(bool)]
        for row in new_positions.itertuples():
            if volume.loc[row.execution_date, row.stock] <= 0:
                raise AssertionError(f"non-tradeable new position in {name}")
        locked_counts[name] = int(selections["locked"].astype(bool).sum())
    return {"strategies": sorted(strategies), "locked_rows": locked_counts}


def validate_lstm(output_dir: Path) -> dict:
    import torch
    from qd.lstm_model import MinuteLSTM, evaluate_saved_lstm

    root = output_dir / "lstm"
    bundle = torch.load(root / "model.pt", map_location="cpu", weights_only=False)
    if set(bundle) != {"state_dict", "config"}:
        raise AssertionError("LSTM model bundle is incomplete")
    config = bundle["config"]
    required = {
        "stock_codes", "channels", "feature_names", "feature_set", "seq_len",
        "hidden_size", "num_layers", "dropout", "model_version", "num_classes",
        "class_names", "target_mode", "splits",
        "scaler_mean", "scaler_std", "scaler_mode", "include_stock_id", "seed",
        "sizes", "coverage", "decision_threshold", "pipeline_version",
    }
    if not required.issubset(config):
        raise AssertionError("LSTM config is incomplete")
    metrics = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
    for key in (
        "accuracy", "majority_baseline", "precision", "recall", "f1",
        "confusion_matrix", "threshold",
    ):
        if key not in metrics:
            raise AssertionError(f"LSTM metric missing: {key}")
    input_size = len(config["feature_names"])
    model = MinuteLSTM(
        input_size=input_size,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        model_version=config["model_version"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, config["seq_len"], input_size))
    if tuple(logits.shape) != (2, config["num_classes"]):
        raise AssertionError(f"unexpected LSTM output shape: {tuple(logits.shape)}")
    base_features = input_size - (
        len(config["stock_codes"]) if config["include_stock_id"] else 0
    )
    mean = np.asarray(config["scaler_mean"])
    std = np.asarray(config["scaler_std"])
    expected_scaler_shape = (
        (len(config["stock_codes"]), base_features)
        if config["scaler_mode"] == "per_stock" else (base_features,)
    )
    if mean.shape != expected_scaler_shape or std.shape != expected_scaler_shape:
        raise AssertionError(
            f"scaler shape mismatch: mean={mean.shape}, std={std.shape}, "
            f"expected={expected_scaler_shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise AssertionError("LSTM scaler contains invalid values")
    confusion_total = int(np.asarray(metrics["confusion_matrix"]).sum())
    if confusion_total != int(config["sizes"]["test"]):
        raise AssertionError("confusion matrix total does not match test size")
    coverage = config["coverage"]["test"]
    expected_test_size = (
        coverage["nonflat_windows"]
        if config["target_mode"] == "nonflat_binary"
        else coverage["all_valid_windows"]
    )
    if int(expected_test_size) != int(config["sizes"]["test"]):
        raise AssertionError("test size does not match target coverage")
    if metrics["threshold"] != config["decision_threshold"]:
        raise AssertionError("saved threshold and reported threshold differ")
    predictions = pd.read_csv(root / "test_predictions.csv")
    metadata_columns = ["stock", "date", "window_end", "target_time"]
    if not set(metadata_columns).issubset(predictions.columns):
        raise AssertionError("LSTM predictions lack auditable sample timestamps")
    if len(predictions) != int(config["sizes"]["test"]):
        raise AssertionError("LSTM prediction row count does not match test size")
    probability_columns = [
        column for column in predictions.columns if column.startswith("prob_")
    ]
    expected_probability_columns = [f"prob_{name}" for name in config["class_names"]]
    if probability_columns != expected_probability_columns:
        raise AssertionError("LSTM probability column names do not match class semantics")
    probability = predictions[probability_columns].to_numpy(dtype=float)
    if probability.shape[1] != config["num_classes"]:
        raise AssertionError("LSTM prediction probability columns are incomplete")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("LSTM prediction probabilities do not sum to one")
    observed_accuracy = float(
        (predictions["true_label"] == predictions["predicted_label"]).mean()
    )
    if not np.isclose(observed_accuracy, metrics["accuracy"], atol=1e-12):
        raise AssertionError("LSTM prediction file and reported accuracy differ")
    if "high_confidence" in metrics:
        if "high_confidence_nonflat" not in predictions:
            raise AssertionError("high-confidence metric lacks row-level flags")
        selected = predictions["high_confidence_nonflat"].astype(bool)
        high = metrics["high_confidence"]
        if high.get("scope") != "nonflat_targets_only":
            raise AssertionError("conditional high-confidence scope is not explicit")
        if int(selected.sum()) != int(high["n"]):
            raise AssertionError("high-confidence count is inconsistent")
        selected_accuracy = float(
            (
                predictions.loc[selected, "true_label"]
                == predictions.loc[selected, "predicted_label"]
            ).mean()
        )
        if not np.isclose(selected_accuracy, high["accuracy"], atol=1e-12):
            raise AssertionError("high-confidence accuracy is inconsistent")
    replay = evaluate_saved_lstm(
        root / "model.pt", output_dir / "minute", split="test", device="cpu"
    )
    if not np.array_equal(replay["labels"], predictions["true_label"].to_numpy()):
        raise AssertionError("replayed LSTM labels differ from the prediction artifact")
    replay_metadata = replay["metadata"].drop(columns="stock_id").reset_index(drop=True)
    if not replay_metadata.equals(predictions[metadata_columns].reset_index(drop=True)):
        raise AssertionError("replayed LSTM sample timestamps differ")
    if not np.allclose(replay["probability"], probability, atol=1e-6, rtol=1e-6):
        raise AssertionError("replayed LSTM probabilities differ from the model artifact")
    return {
        "stocks": config["stock_codes"],
        "features": len(config["feature_names"]),
        "test_accuracy": metrics["accuracy"],
        "majority_baseline": metrics["majority_baseline"],
        "nonflat_coverage": coverage["nonflat_coverage"],
        "model_sha256": sha256_file(root / "model.pt"),
        "predictions_sha256": sha256_file(root / "test_predictions.csv"),
    }


def validate_lstm_full(output_dir: Path) -> dict:
    from qd.lstm_full import evaluate_full_model, load_full_model

    root = output_dir / "lstm_full"
    ensemble = load_full_model(root / "model.pt", device="cpu")
    config = ensemble.config
    required = {
        "pipeline_version", "stock_codes", "channels", "splits", "sizes",
        "coverage", "class_rates", "class_names", "move_bias",
        "joint_weight", "two_stage_weight", "selective_thresholds", "runtime",
    }
    if not required.issubset(config):
        raise AssertionError("full-window LSTM config is incomplete")
    if not np.isclose(config["joint_weight"] + config["two_stage_weight"], 1.0):
        raise AssertionError("full-window ensemble weights must sum to one")
    if config["class_names"] != ["down", "flat", "up"]:
        raise AssertionError("full-window class order is invalid")

    direction_cfg = ensemble.component_configs["direction"]
    enhanced_cfg = ensemble.component_configs["joint"]
    if int(direction_cfg["seq_len"]) != int(enhanced_cfg["seq_len"]):
        raise AssertionError("full-window components use different sequence lengths")
    n_stocks = len(config["stock_codes"])
    direction_input = int(direction_cfg.get("input_size", len(direction_cfg["feature_names"])))
    direction_base = direction_input - (
        n_stocks if direction_cfg["include_stock_id"] else 0
    )
    enhanced_input = int(enhanced_cfg.get("input_size", len(enhanced_cfg["feature_names"])))
    enhanced_base = enhanced_input - (
        n_stocks if enhanced_cfg["include_stock_id"] else 0
    )
    seq_len = int(direction_cfg["seq_len"])
    ids = np.array([0, min(1, n_stocks - 1)], dtype=np.int64)
    probability = ensemble.predict_from_features(
        np.zeros((2, seq_len, direction_base), dtype=np.float32),
        np.zeros((2, seq_len, enhanced_base), dtype=np.float32),
        ids,
        batch_size=2,
    )
    if probability.shape != (2, 3) or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-6
    ):
        raise AssertionError("full-window ensemble forward pass is invalid")

    metrics = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
    for key in (
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "confusion_matrix", "majority_baseline", "validation_accuracy",
        "coverage", "per_stock", "macro_stock_accuracy",
        "selective_accuracy",
    ):
        if key not in metrics:
            raise AssertionError(f"full-window LSTM metric missing: {key}")
    confusion_total = int(np.asarray(metrics["confusion_matrix"]).sum())
    expected = int(config["sizes"]["test"])
    if confusion_total != expected:
        raise AssertionError("full-window confusion matrix total does not match test size")
    if expected != int(config["coverage"]["test"]["all_valid_windows"]):
        raise AssertionError("full-window model does not cover every valid test window")
    if metrics["coverage"] != config["coverage"]["test"]:
        raise AssertionError("full-window metric and model coverage differ")

    predictions = pd.read_csv(root / "test_predictions.csv")
    metadata_columns = ["stock", "date", "window_end", "target_time"]
    if not set(metadata_columns).issubset(predictions.columns):
        raise AssertionError("full-window predictions lack auditable sample timestamps")
    if len(predictions) != expected:
        raise AssertionError("full-window prediction row count does not match test size")
    probability_columns = ["prob_down", "prob_flat", "prob_up"]
    saved_probability = predictions[probability_columns].to_numpy(dtype=float)
    if not np.allclose(saved_probability.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("full-window probabilities do not sum to one")
    if not np.array_equal(saved_probability.argmax(axis=1), predictions["predicted_label"]):
        raise AssertionError("full-window labels do not match saved probabilities")
    observed_accuracy = float(
        (predictions["true_label"] == predictions["predicted_label"]).mean()
    )
    if not np.isclose(observed_accuracy, metrics["accuracy"], atol=1e-12):
        raise AssertionError("full-window prediction file and accuracy differ")
    for name in ("balanced", "strict"):
        flag = f"selected_{name}"
        if flag not in predictions or name not in metrics["selective_accuracy"]:
            raise AssertionError(f"full-window selective tier missing: {name}")
        selected = predictions[flag].astype(bool)
        values = metrics["selective_accuracy"][name]
        if int(selected.sum()) != int(values["test_n"]):
            raise AssertionError(f"full-window selective count differs: {name}")
        if not np.isclose(float(selected.mean()), values["test_coverage"], atol=1e-12):
            raise AssertionError(f"full-window selective coverage differs: {name}")
        selected_accuracy = float(
            (
                predictions.loc[selected, "true_label"]
                == predictions.loc[selected, "predicted_label"]
            ).mean()
        )
        if not np.isclose(selected_accuracy, values["test_accuracy"], atol=1e-12):
            raise AssertionError(f"full-window selective accuracy differs: {name}")
        if not np.isclose(
            config["selective_thresholds"][name],
            values["validation_confidence_threshold"],
            atol=1e-12,
        ):
            raise AssertionError(f"full-window selective threshold differs: {name}")
    replay = evaluate_full_model(
        root / "model.pt", output_dir / "minute", split="test", device="cpu"
    )
    if not np.array_equal(replay["labels"], predictions["true_label"].to_numpy()):
        raise AssertionError("replayed full-window labels differ from predictions")
    replay_metadata = replay["metadata"].drop(columns="stock_id").reset_index(drop=True)
    if not replay_metadata.equals(predictions[metadata_columns].reset_index(drop=True)):
        raise AssertionError("replayed full-window sample timestamps differ")
    if not np.allclose(
        replay["probability"], saved_probability, atol=1e-6, rtol=1e-6
    ):
        raise AssertionError("replayed full-window probabilities differ from model artifact")
    return {
        "stocks": config["stock_codes"],
        "components": sorted(ensemble.models),
        "test_size": expected,
        "test_accuracy": metrics["accuracy"],
        "majority_baseline": metrics["majority_baseline"],
        "macro_f1": metrics["macro_f1"],
        "model_sha256": sha256_file(root / "model.pt"),
        "predictions_sha256": sha256_file(root / "test_predictions.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--full-minute", action="store_true")
    args = parser.parse_args()
    dates = list_trading_dates()
    if len(code_mapping()) != 300:
        raise AssertionError("adjfactor mapping is not 300/300")
    report = {
        "daily": validate_daily(args.output_dir, dates),
        "minute": validate_minute(args.output_dir, dates, args.full_minute),
        "raw_sample": validate_raw_sample(args.output_dir),
        "factors": validate_factors(args.output_dir),
        "backtest": validate_backtest(args.output_dir),
        "lstm": validate_lstm(args.output_dir),
        "lstm_full": validate_lstm_full(args.output_dir),
    }
    target = args.output_dir / "validation_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
