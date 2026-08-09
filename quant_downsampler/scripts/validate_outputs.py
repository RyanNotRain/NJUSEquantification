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


def _validate_standard_probability_artifact(
    root: Path,
    reported_accuracy: float,
) -> dict:
    predictions = pd.read_csv(root / "test_predictions.csv")
    probability_columns = ["prob_down", "prob_flat", "prob_up"]
    required = {"true_label", "predicted_label", *probability_columns}
    if not required.issubset(predictions.columns):
        raise AssertionError(f"probability artifact is incomplete: {root}")
    probability = predictions[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not np.allclose(
        probability.sum(axis=1), 1.0, atol=1e-6
    ):
        raise AssertionError(f"invalid saved probabilities: {root}")
    if not np.array_equal(
        probability.argmax(axis=1), predictions["predicted_label"].to_numpy()
    ):
        raise AssertionError(f"saved labels do not match probabilities: {root}")
    observed = float(
        (predictions["true_label"] == predictions["predicted_label"]).mean()
    )
    if not np.isclose(observed, reported_accuracy, atol=1e-12):
        raise AssertionError(f"reported accuracy differs from predictions: {root}")
    return {"n": int(len(predictions)), "accuracy": observed}


def validate_research_enhancements(output_dir: Path) -> dict:
    """Validate optional robustness/model artifacts when they are present."""

    report: dict[str, dict] = {}

    factor_root = output_dir / "factor_robustness"
    if factor_root.exists():
        bootstrap = pd.read_csv(factor_root / "ic_bootstrap_ci.csv")
        if set(bootstrap["factor"]) != set(FACTOR_NAMES):
            raise AssertionError("factor robustness bootstrap set differs")
        if not (
            np.isfinite(
                bootstrap[["mean_ic", "ci_lower", "ci_upper"]].to_numpy(float)
            ).all()
            and (bootstrap["ci_lower"] <= bootstrap["mean_ic"]).all()
            and (bootstrap["mean_ic"] <= bootstrap["ci_upper"]).all()
        ):
            raise AssertionError("factor robustness confidence intervals are invalid")
        manifest = json.loads(
            (factor_root / "analysis_manifest.json").read_text(encoding="utf-8")
        )
        neutralization = manifest.get("neutralization", {})
        if neutralization.get("status") not in {"completed", "skipped"}:
            raise AssertionError("factor neutralization status is not explicit")
        report["factor_robustness"] = {
            "factors": int(len(bootstrap)),
            "neutralization": neutralization.get("status"),
        }

    backtest_root = output_dir / "backtest_robustness"
    if backtest_root.exists():
        stress = pd.read_csv(backtest_root / "cost_stress.csv")
        if not stress["one_way_cost_bps"].is_monotonic_increasing:
            raise AssertionError("backtest cost grid is not increasing")
        if (stress["total_return"].diff().dropna() > 1e-12).any():
            raise AssertionError("backtest net return increases with transaction cost")
        summary = json.loads((backtest_root / "summary.json").read_text(encoding="utf-8"))
        break_even = float(summary["symmetric_break_even_one_way_bps"])
        if not np.isfinite(break_even) or break_even < 0.0:
            raise AssertionError("backtest break-even cost is invalid")
        report["backtest_robustness"] = {
            "cost_scenarios": int(len(stress)),
            "break_even_one_way_bps": break_even,
        }

        benchmark_path = backtest_root / "benchmark_metrics.json"
        if benchmark_path.exists():
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if benchmark["official_factor_names"] != list(FACTOR_NAMES):
                raise AssertionError("market report changed the official factor universe")
            experimental = benchmark.get("experimental_factor_names", [])
            if experimental != ["illiquidity_20d"]:
                raise AssertionError("experimental factor scope is missing or ambiguous")
            if not benchmark["benchmark_definition"].get("not_an_index_claim"):
                raise AssertionError("sample market proxy is mislabeled as an index")
            periods = pd.read_csv(backtest_root / "benchmark_periods.csv")
            required_columns = {
                "date", "strategy", "strategy_net_return", "market_return",
                "strategy_nav", "benchmark_nav", "relative_wealth",
                "n_market_stocks",
            }
            if not required_columns.issubset(periods.columns):
                raise AssertionError("factor market-period table is incomplete")
            for strategy_name, group in periods.groupby("strategy", sort=False):
                group = group.copy()
                dates = pd.to_datetime(group["date"], errors="coerce")
                if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
                    raise AssertionError(f"factor market dates are invalid: {strategy_name}")
                strategy_nav = (1.0 + group["strategy_net_return"]).cumprod()
                market_nav = (1.0 + group["market_return"]).cumprod()
                if not np.allclose(strategy_nav, group["strategy_nav"], atol=1e-8):
                    raise AssertionError(f"strategy NAV identity differs: {strategy_name}")
                if not np.allclose(market_nav, group["benchmark_nav"], atol=1e-8):
                    raise AssertionError(f"market NAV identity differs: {strategy_name}")
                if not np.allclose(
                    strategy_nav / market_nav, group["relative_wealth"], atol=1e-8
                ):
                    raise AssertionError(f"relative wealth identity differs: {strategy_name}")
                if (group["n_market_stocks"] != 300).any():
                    raise AssertionError("daily market proxy does not use all 300 sample stocks")
            single = pd.read_csv(
                backtest_root / "single_factor_market_metrics.csv", index_col=0
            )
            if set(single.index) != {*FACTOR_NAMES, "illiquidity_20d"}:
                raise AssertionError("single-factor market report has the wrong factor set")
            if single.loc["illiquidity_20d", "factor_scope"] != "experimental":
                raise AssertionError("illiquidity factor was promoted into the official set")
            illiquidity = pd.read_csv(
                backtest_root / "illiquidity_20d.csv", index_col=0
            )
            if illiquidity.shape != (302, 300):
                raise AssertionError("experimental illiquidity factor shape differs")
            report["factor_market_benchmark"] = {
                "strategies": int(periods["strategy"].nunique()),
                "official_factors": len(FACTOR_NAMES),
                "experimental_factors": len(experimental),
                "market_stock_count": 300,
            }

    baseline_root = output_dir / "lstm_baselines"
    if baseline_root.exists():
        metrics = json.loads(
            (baseline_root / "test_metrics.json").read_text(encoding="utf-8")
        )
        predictions = pd.read_csv(baseline_root / "test_predictions.csv")
        for prefix, name in (
            ("logistic", "logistic_regression"),
            ("hist_tree", "hist_gradient_boosting"),
        ):
            probability = predictions[
                [f"{prefix}_prob_down", f"{prefix}_prob_flat", f"{prefix}_prob_up"]
            ].to_numpy(float)
            if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
                raise AssertionError(f"{name} probabilities do not sum to one")
            observed = float(
                (probability.argmax(axis=1) == predictions["true_label"]).mean()
            )
            expected = float(metrics["models"][name]["test"]["accuracy"])
            if not np.isclose(observed, expected, atol=1e-12):
                raise AssertionError(f"{name} prediction accuracy differs")
        audit = metrics["audit"]
        if not audit.get("test_loaded_after_selection_frozen"):
            raise AssertionError("baseline test was not loaded after frozen selection")
        report["lstm_baselines"] = {"n": int(len(predictions)), "models": 2}

    adjusted_root = output_dir / "lstm_adjusted"
    if adjusted_root.exists():
        metrics = json.loads(
            (adjusted_root / "test_metrics.json").read_text(encoding="utf-8")
        )
        values = _validate_standard_probability_artifact(
            adjusted_root, float(metrics["accuracy"])
        )
        frozen = json.loads(
            (adjusted_root / "selection_frozen_before_test.json").read_text(
                encoding="utf-8"
            )
        )
        if int(frozen.get("test_evaluation_count", -1)) != 1:
            raise AssertionError("adjusted LSTM test evaluation count differs")
        report["lstm_adjusted"] = values

    hybrid_root = output_dir / "lstm_hybrid"
    if hybrid_root.exists():
        metrics = json.loads(
            (hybrid_root / "test_metrics.json").read_text(encoding="utf-8")
        )
        values = _validate_standard_probability_artifact(
            hybrid_root, float(metrics["hybrid"]["accuracy"])
        )
        selection = metrics["selection"]
        if not np.isclose(
            float(selection["lstm_weight"]) + float(selection["tree_weight"]), 1.0
        ):
            raise AssertionError("hybrid model weights do not sum to one")
        if not metrics["audit"].get("test_loaded_after_selection_frozen"):
            raise AssertionError("hybrid test was not loaded after frozen selection")
        report["lstm_hybrid"] = values

    research_root = output_dir / "lstm_research"
    if research_root.exists():
        metrics = json.loads(
            (research_root / "research_metrics.json").read_text(encoding="utf-8")
        )
        folds = pd.read_csv(research_root / "walk_forward_manifest.csv")
        if len(folds) != len(folds["fold"].unique()) or len(folds) == 0:
            raise AssertionError("walk-forward manifest fold identifiers are invalid")
        if not metrics["methodology"].get("date_stability_is_not_walk_forward"):
            raise AssertionError("static date slicing is mislabeled as walk-forward")
        report["lstm_research"] = {"planned_walk_forward_folds": int(len(folds))}

    for directory in (
        "lstm_strategy",
        "lstm_strategy_long_only",
        "lstm_hybrid_strategy",
        "lstm_hybrid_strategy_long_only",
    ):
        strategy_root = output_dir / directory
        if not strategy_root.exists():
            continue
        metrics = json.loads(
            (strategy_root / "metrics.json").read_text(encoding="utf-8")
        )
        if not np.isclose(
            float(metrics["label_return_alignment_audit"]["agreement"]), 1.0
        ):
            raise AssertionError(f"strategy return labels are misaligned: {directory}")
        sensitivity = pd.read_csv(strategy_root / "cost_sensitivity.csv")
        for _, group in sensitivity.groupby("strategy"):
            ordered = group.sort_values("cost_bps")
            if (ordered["net_total_return"].diff().dropna() > 1e-12).any():
                raise AssertionError(f"strategy return increases with cost: {directory}")
        report[directory] = {
            "strategies": int(sensitivity["strategy"].nunique()),
            "test_days": int(sensitivity["n_days"].iloc[0]),
        }

    return_root = output_dir / "lstm_return"
    if return_root.exists():
        predictions = pd.read_csv(return_root / "test_predictions.csv")
        required = {
            "stock", "window_end", "target_time", "true_label",
            "prob_down", "prob_flat", "prob_up", "predicted_abs_return_bps",
            "expected_return_bps", "realised_return_bps",
        }
        if not required.issubset(predictions.columns) or len(predictions) != 5_900:
            raise AssertionError("return-LSTM prediction table is incomplete")
        probability = predictions[["prob_down", "prob_flat", "prob_up"]].to_numpy(float)
        if not np.isfinite(probability).all() or not np.allclose(
            probability.sum(axis=1), 1.0, atol=1e-5
        ):
            raise AssertionError("return-LSTM probabilities are invalid")
        magnitude = predictions["predicted_abs_return_bps"].to_numpy(float)
        expected = predictions["expected_return_bps"].to_numpy(float)
        realised = predictions["realised_return_bps"].to_numpy(float)
        if not np.isfinite(np.column_stack((magnitude, expected, realised))).all() or (
            magnitude < 0.0
        ).any():
            raise AssertionError("return-LSTM magnitude outputs are invalid")
        if not np.allclose(
            expected, (probability[:, 2] - probability[:, 0]) * magnitude, atol=1e-6
        ):
            raise AssertionError("return-LSTM expected-return identity differs")
        expected_labels = np.where(realised < 0.0, 0, np.where(realised > 0.0, 2, 1))
        if not np.array_equal(expected_labels, predictions["true_label"].to_numpy(int)):
            raise AssertionError("return-LSTM labels and signed returns differ")
        metrics = json.loads(
            (return_root / "test_metrics.json").read_text(encoding="utf-8")
        )
        signed = metrics["signed_expected_return"]
        if not np.isclose(
            float(np.mean(np.abs(expected - realised))), float(signed["mae_bps"]),
            atol=1e-8,
        ):
            raise AssertionError("return-LSTM signed MAE differs")
        freeze = json.loads(
            (return_root / "selection_frozen_before_test.json").read_text(
                encoding="utf-8"
            )
        )
        if freeze.get("test_loaded_before_freeze") or freeze.get(
            "test_metrics_used_for_selection"
        ):
            raise AssertionError("return-LSTM selection used the test split")
        floor = float(freeze["opening_threshold_floor_one_way_bps"])
        if any(
            float(value) < floor
            for value in freeze["selected_opening_threshold_bps"].values()
        ):
            raise AssertionError("return-LSTM bps threshold is below the cost floor")
        probability_thresholds = freeze.get("selected_probability_gap_threshold", {})
        if set(probability_thresholds) != {"long_short", "long_only"} or any(
            not 0.0 <= float(value) <= 1.0
            for value in probability_thresholds.values()
        ):
            raise AssertionError("return-LSTM probability-gap threshold is invalid")
        strategy_table = pd.read_csv(return_root / "strategy_comparison.csv")
        expected_rows = strategy_table["comparison_signal"].eq("expected_return")
        gap_rows = strategy_table["comparison_signal"].eq("probability_gap")
        if not expected_rows.any() or not gap_rows.any():
            raise AssertionError("return-LSTM strategy ablation is incomplete")
        if not strategy_table.loc[
            expected_rows, "validation_frozen_threshold_unit"
        ].eq("bps").all() or not strategy_table.loc[
            expected_rows, "validation_frozen_threshold_bps"
        ].ge(floor).all():
            raise AssertionError("return-LSTM bps threshold units are invalid")
        if not strategy_table.loc[
            gap_rows, "validation_frozen_threshold_unit"
        ].eq("probability_gap").all() or not strategy_table.loc[
            gap_rows, "validation_frozen_threshold_bps"
        ].isna().all():
            raise AssertionError("return-LSTM probability threshold is mislabeled as bps")
        replay = json.loads((return_root / "replay_audit.json").read_text(encoding="utf-8"))
        if not replay.get("passed") or float(replay["maximum_absolute_difference"]) > 1e-6:
            raise AssertionError("return-LSTM persisted replay failed")
        report["lstm_return"] = {
            "n": int(len(predictions)),
            "magnitude_mae_bps": float(metrics["magnitude"]["mae_bps"]),
            "expected_return_spearman": float(signed["spearman_ic"]),
            "strict_replay": True,
        }

    return_baseline_root = output_dir / "lstm_return_baselines"
    if return_baseline_root.exists():
        metrics = json.loads(
            (return_baseline_root / "test_metrics.json").read_text(encoding="utf-8")
        )
        audit = metrics["audit"]
        if not audit.get("test_loaded_after_selection_frozen") or audit.get(
            "test_used_for_hyperparameter_or_threshold_selection"
        ):
            raise AssertionError("return-regression baselines used test information")
        replay = audit.get("persistence_replay", {})
        if not replay.get("passed") or replay.get("raw_test_data_load_count") != 1:
            raise AssertionError("return-regression baseline replay failed")
        predictions = pd.read_csv(return_baseline_root / "test_predictions.csv")
        if len(predictions) != 5_900:
            raise AssertionError("return-regression test row count differs")
        for model_name in ("ridge", "hist_gradient_boosting_regressor"):
            canonical = pd.read_csv(
                return_baseline_root / f"{model_name}_test_predictions.csv"
            )
            if not predictions[["stock", "window_end", "target_time"]].equals(
                canonical[["stock", "window_end", "target_time"]]
            ):
                raise AssertionError(f"canonical return keys differ: {model_name}")
            source_column = f"{model_name}_expected_return_bps"
            if not np.allclose(
                predictions[source_column], canonical["expected_return_bps"], atol=1e-9
            ):
                raise AssertionError(f"canonical return predictions differ: {model_name}")
        sensitivity = pd.read_csv(
            return_baseline_root / "strategy_cost_sensitivity.csv"
        )
        for _, group in sensitivity.groupby(["model", "side"]):
            ordered = group.sort_values("cost_bps")
            if (ordered["net_total_return"].diff().dropna() > 1e-12).any():
                raise AssertionError("return-regression strategy improves with cost")
        report["lstm_return_baselines"] = {
            "n": int(len(predictions)),
            "models": 2,
            "strict_replay": True,
        }

    for directory in (
        "lstm_strategy_comparison_full",
        "lstm_strategy_comparison_full_long_only",
    ):
        comparison_root = output_dir / directory
        if not comparison_root.exists():
            continue
        metrics = json.loads(
            (comparison_root / "metrics.json").read_text(encoding="utf-8")
        )
        alignment = metrics["alignment_audit"]
        if not (
            alignment.get("all_sample_keys_exactly_equal")
            and alignment.get("full_stock_cross_section_at_every_interval")
            and int(alignment.get("n_rows", 0)) == 5_900
        ):
            raise AssertionError(f"unified strategy alignment failed: {directory}")
        methodology = metrics["methodology"]
        if methodology.get("benchmark_used_for_model_or_threshold_selection"):
            raise AssertionError(f"market benchmark affected selection: {directory}")
        summary = pd.read_csv(comparison_root / "strategy_comparison.csv")
        expected_names = {
            "original_lstm", "hist_gradient_boosting", "hybrid",
            "magnitude_lstm", "ridge_return", "histgb_return",
            "equal_weight_market_proxy",
            "equal_weight_buy_and_hold_observed_horizons",
        }
        if set(summary["name"]) != expected_names:
            raise AssertionError(f"unified strategy model set differs: {directory}")
        market = float(
            summary.loc[
                summary["name"].eq("equal_weight_market_proxy"),
                "gross_total_return",
            ].iloc[0]
        )
        relative = (1.0 + summary["net_total_return"]) / (1.0 + market) - 1.0
        if not np.allclose(relative, summary["net_relative_to_market_proxy"], atol=1e-8):
            raise AssertionError(f"unified relative-return identity differs: {directory}")
        report[directory] = {
            "models": int(summary["kind"].eq("model").sum()),
            "benchmarks": int(summary["kind"].eq("benchmark").sum()),
            "n": int(alignment["n_rows"]),
        }
    return report


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
        "research_enhancements": validate_research_enhancements(args.output_dir),
    }
    target = args.output_dir / "validation_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
