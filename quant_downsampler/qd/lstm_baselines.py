"""Leakage-safe traditional ML baselines for the full-window LSTM task.

The baseline models consume exactly the same legal down/flat/up windows as
the enhanced LSTM.  Each 60-minute sequence is converted to a fixed-width
vector using only values at or before the window end.  Candidate models and
all preprocessing steps are fitted on the training split, hyperparameters are
selected on validation, and the final test split is not loaded until that
selection has been persisted.
"""

from __future__ import annotations

import gc
import json
import platform
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR
from .lstm_model import DEFAULT_SPLITS, _make_split, feature_names


CLASS_NAMES = ("down", "flat", "up")
SUMMARY_STATISTICS = (
    "last",
    "mean_window",
    "std_window",
    "mean_last10",
    "std_last10",
    "mean_last5",
    "std_last5",
)
DEFAULT_LOGISTIC_GRID: tuple[dict[str, Any], ...] = (
    {"C": 0.1, "max_iter": 120, "tol": 1e-3},
    {"C": 1.0, "max_iter": 120, "tol": 1e-3},
)
DEFAULT_TREE_GRID: tuple[dict[str, Any], ...] = (
    {
        "learning_rate": 0.08,
        "max_iter": 50,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
    {
        "learning_rate": 0.08,
        "max_iter": 50,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
    },
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_splits(splits: Mapping[str, tuple[str, str]]) -> None:
    for name in ("train", "val", "test"):
        if name not in splits or len(splits[name]) != 2:
            raise ValueError(f"missing or malformed {name!r} split")
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


def _resolve_stocks(
    data_dir: Path,
    stock_codes: Sequence[str] | None,
    n_stocks: int,
) -> list[str]:
    if stock_codes is not None:
        stocks = [str(value).strip() for value in stock_codes]
        if not stocks or any(not value for value in stocks):
            raise ValueError("stock_codes must contain non-empty values")
        if len(set(stocks)) != len(stocks):
            raise ValueError("stock_codes contains duplicates")
        return stocks
    if n_stocks <= 0:
        raise ValueError("n_stocks must be positive")
    close_files = sorted((data_dir / "close").glob("*.csv"))
    if not close_files:
        raise FileNotFoundError(f"no minute close tables found under {data_dir}")
    available = pd.read_csv(close_files[0], nrows=1, index_col=0).columns.tolist()
    if n_stocks > len(available):
        raise ValueError(
            f"requested {n_stocks} stocks but only {len(available)} are available"
        )
    return [str(value) for value in available[:n_stocks]]


def summarize_sequences(
    sequences: np.ndarray,
    base_feature_names: Sequence[str],
    stock_ids: np.ndarray | None = None,
    stock_codes: Sequence[str] | None = None,
    chunk_size: int = 8_192,
) -> tuple[np.ndarray, list[str]]:
    """Convert past-only sequences into compact traditional-ML features.

    For each original feature this emits the last value, full-window mean and
    standard deviation, and mean/standard deviation over the last 10 and 5
    minutes.  Optional stock one-hot columns match the identity information
    supplied to the full LSTM.  Population standard deviation (``ddof=0``) is
    used throughout for deterministic fixed-window summaries.
    """

    x = np.asarray(sequences)
    if x.ndim != 3:
        raise ValueError("sequences must have shape (samples, time, features)")
    n_samples, sequence_length, n_features = x.shape
    names = [str(value) for value in base_feature_names]
    if len(names) != n_features:
        raise ValueError("base_feature_names do not match the sequence feature axis")
    if sequence_length < 10:
        raise ValueError("sequence length must be at least 10 minutes")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    include_stock = stock_ids is not None or stock_codes is not None
    if include_stock and (stock_ids is None or stock_codes is None):
        raise ValueError("stock_ids and stock_codes must be supplied together")
    stock_names = [
        str(value) for value in ([] if stock_codes is None else stock_codes)
    ]
    ids = np.asarray(stock_ids, dtype=np.int64) if stock_ids is not None else None
    if ids is not None:
        if ids.shape != (n_samples,):
            raise ValueError("stock_ids are not aligned with sequences")
        if len(ids) and (ids.min() < 0 or ids.max() >= len(stock_names)):
            raise ValueError("stock_ids contain an out-of-range value")

    summary_names = [
        f"{statistic}::{name}"
        for statistic in SUMMARY_STATISTICS
        for name in names
    ]
    if ids is not None:
        summary_names.extend(f"stock_id::{stock}" for stock in stock_names)
    output = np.empty((n_samples, len(summary_names)), dtype=np.float32)

    numeric_columns = len(SUMMARY_STATISTICS) * n_features
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        block = x[start:stop]
        statistics = (
            block[:, -1, :],
            block.mean(axis=1),
            block.std(axis=1, ddof=0),
            block[:, -10:, :].mean(axis=1),
            block[:, -10:, :].std(axis=1, ddof=0),
            block[:, -5:, :].mean(axis=1),
            block[:, -5:, :].std(axis=1, ddof=0),
        )
        output[start:stop, :numeric_columns] = np.concatenate(statistics, axis=1)
        if ids is not None:
            output[start:stop, numeric_columns:] = np.eye(
                len(stock_names), dtype=np.float32
            )[ids[start:stop]]
    return output, summary_names


def make_estimator(
    model_name: str,
    parameters: Mapping[str, Any],
    seed: int = 42,
) -> Pipeline:
    """Build a train-fitted preprocessing and classifier pipeline."""

    params = dict(parameters)
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 120)),
            tol=float(params.get("tol", 1e-3)),
            class_weight=params.get("class_weight"),
            solver="lbfgs",
            random_state=seed,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])
    if model_name == "hist_gradient_boosting":
        classifier = HistGradientBoostingClassifier(
            learning_rate=float(params.get("learning_rate", 0.08)),
            max_iter=int(params.get("max_iter", 50)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 31)),
            min_samples_leaf=int(params.get("min_samples_leaf", 50)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            early_stopping=False,
            random_state=seed,
        )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ])
    raise ValueError(f"unknown baseline model: {model_name}")


def evaluate_probabilities(
    labels: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    """Return comparable classification and probabilistic metrics."""

    y = np.asarray(labels, dtype=np.int64)
    raw = np.asarray(probability, dtype=np.float64)
    if raw.ndim != 2 or len(raw) != len(y) or raw.shape[1] < 2:
        raise ValueError("probability must align with labels and contain >=2 classes")
    if len(y) == 0 or y.min() < 0 or y.max() >= raw.shape[1]:
        raise ValueError("labels are empty or outside the probability class range")
    if not np.isfinite(raw).all() or (raw < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sum = raw.sum(axis=1, keepdims=True)
    if (row_sum <= 0).any():
        raise ValueError("each probability row must have positive mass")
    normalized = raw / row_sum
    clipped = np.clip(normalized, 1e-15, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    predicted = normalized.argmax(axis=1)
    one_hot = np.eye(normalized.shape[1], dtype=np.float64)[y]
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "brier_score": float(np.mean(np.sum((normalized - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(
            log_loss(y, clipped, labels=np.arange(normalized.shape[1]))
        ),
        "confusion_matrix": confusion_matrix(
            y, predicted, labels=np.arange(normalized.shape[1])
        ).tolist(),
        "class_rates": (
            np.bincount(y, minlength=normalized.shape[1]).astype(float) / len(y)
        ).tolist(),
        "n_samples": int(len(y)),
    }


def _aligned_probability(estimator: Pipeline, matrix: np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(matrix)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    probability = np.zeros((len(matrix), len(CLASS_NAMES)), dtype=np.float64)
    if len(classes) != raw.shape[1] or (classes < 0).any() or (classes >= 3).any():
        raise ValueError("estimator classes are incompatible with down/flat/up")
    probability[:, classes] = raw
    return probability


def select_model_on_validation(
    model_name: str,
    parameter_grid: Sequence[Mapping[str, Any]],
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_labels: np.ndarray,
    seed: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit candidates on train and choose by validation accuracy, then Macro-F1."""

    if not parameter_grid:
        raise ValueError(f"empty parameter grid for {model_name}")
    best_model: Pipeline | None = None
    best_record: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    records: list[dict[str, Any]] = []
    search_started = time.perf_counter()
    for candidate_id, candidate in enumerate(parameter_grid):
        estimator = make_estimator(model_name, candidate, seed=seed)
        fit_started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(train_matrix, train_labels)
        training_seconds = time.perf_counter() - fit_started
        validation_probability = _aligned_probability(estimator, validation_matrix)
        metrics = evaluate_probabilities(validation_labels, validation_probability)
        record = {
            "model": model_name,
            "candidate_id": int(candidate_id),
            "parameters": dict(candidate),
            "training_seconds": float(training_seconds),
            "validation_metrics": metrics,
            "warnings": [str(item.message) for item in caught],
        }
        records.append(record)
        key = (
            float(metrics["accuracy"]),
            float(metrics["macro_f1"]),
            -float(metrics["negative_log_likelihood"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_model = estimator
            best_record = record
    assert best_model is not None and best_record is not None
    return best_model, {
        "selection_objective": "validation_accuracy_then_macro_f1_then_nll",
        "selected_candidate_id": best_record["candidate_id"],
        "selected_parameters": best_record["parameters"],
        "selected_training_seconds": best_record["training_seconds"],
        "selected_validation_metrics": best_record["validation_metrics"],
        "search_training_seconds": float(
            sum(record["training_seconds"] for record in records)
        ),
        "search_wall_seconds": float(time.perf_counter() - search_started),
        "candidates": records,
    }


def _prior_baseline(
    train_labels: np.ndarray,
    evaluation_labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    prior = counts / counts.sum()
    probability = np.broadcast_to(prior, (len(evaluation_labels), len(prior))).copy()
    metrics = evaluate_probabilities(evaluation_labels, probability)
    metrics["training_class_prior"] = prior.tolist()
    metrics["predicted_class"] = CLASS_NAMES[int(prior.argmax())]
    return probability, metrics


def _stratified_subsample(
    matrix: np.ndarray,
    labels: np.ndarray,
    stock_ids: np.ndarray,
    maximum: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maximum is None or maximum >= len(labels):
        return matrix, labels, stock_ids
    if maximum <= 0:
        raise ValueError("sample cap must be positive")
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    strata = labels * (int(stock_ids.max()) + 1) + stock_ids
    selected, _ = train_test_split(
        indices,
        train_size=maximum,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    selected.sort()
    return matrix[selected], labels[selected], stock_ids[selected]


def _load_summary_split(
    stocks: list[str],
    date_range: tuple[str, str],
    seq_len: int,
    data_dir: Path,
    feature_set: str,
    return_metadata: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame | None, list[str]]:
    split = _make_split(
        stocks,
        date_range,
        seq_len,
        data_dir,
        feature_set,
        "three_class",
        return_metadata,
    )
    if return_metadata:
        sequences, labels, stock_ids, coverage, metadata = split
    else:
        sequences, labels, stock_ids, coverage = split
        metadata = None
    if not len(labels):
        raise ValueError(f"no valid full-window samples for date range {date_range}")
    matrix, names = summarize_sequences(
        sequences,
        base_feature_names=feature_names(feature_set),
        stock_ids=stock_ids,
        stock_codes=stocks,
    )
    del sequences
    gc.collect()
    return matrix, labels, stock_ids, coverage, metadata, names


def _prepare_output_dir(target: Path, overwrite: bool) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty baseline output directory: {target}; "
            "choose another --out-dir or pass --overwrite"
        )
    target.mkdir(parents=True, exist_ok=True)


def _selection_rows(selections: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, selection in selections.items():
        selected_id = int(selection["selected_candidate_id"])
        for record in selection["candidates"]:
            metrics = record["validation_metrics"]
            rows.append({
                "model": model_name,
                "candidate_id": record["candidate_id"],
                "selected": bool(record["candidate_id"] == selected_id),
                "parameters": json.dumps(record["parameters"], sort_keys=True),
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_brier_score": metrics["brier_score"],
                "validation_negative_log_likelihood": metrics[
                    "negative_log_likelihood"
                ],
                "training_seconds": record["training_seconds"],
                "warnings": " | ".join(record["warnings"]),
            })
    return pd.DataFrame(rows)


def _write_output_readme(target: Path, report: Mapping[str, Any]) -> None:
    config = report["config"]
    sizes = report["sizes"]
    majority = report["majority_prior_baseline"]["test"]
    model_rows = []
    for name in ("logistic_regression", "hist_gradient_boosting"):
        result = report["models"][name]
        validation = result["validation"]
        test = result["test"]
        model_rows.append(
            "| "
            + " | ".join([
                name,
                f"{validation['accuracy']:.4%}",
                f"{validation['macro_f1']:.4%}",
                f"{test['accuracy']:.4%}",
                f"{test['macro_f1']:.4%}",
                f"{test['brier_score']:.6f}",
                f"{test['negative_log_likelihood']:.6f}",
                f"{result['selected_training_seconds']:.3f}",
                f"{result['search_training_seconds']:.3f}",
            ])
            + " |"
        )
    lines = [
        "# 全窗口三分类传统机器学习基线",
        "",
        "本目录由 `python -m scripts.run_lstm_baselines` 自动生成。",
        "",
        "## 无泄漏实验口径",
        "",
        f"- 股票：{', '.join(config['stock_codes'])}",
        (
            "- 日期："
            f"train={config['splits']['train']}，"
            f"validation={config['splits']['val']}，"
            f"test={config['splits']['test']}"
        ),
        (
            f"- 样本：train={sizes['train_used']:,}，"
            f"validation={sizes['validation_used']:,}，test={sizes['test']:,}"
        ),
        (
            f"- 输入：{config['seq_len']} 分钟 × "
            f"{config['base_sequence_feature_count']} 个{config['feature_set']}特征，汇总为 "
            f"{config['summary_feature_count']} 维（每个特征的最后值、全窗口均值/"
            "标准差、最近 10 分钟均值/标准差、最近 5 分钟均值/标准差，"
            f"另加 {len(config['stock_codes'])} 维股票 one-hot）。"
        ),
        "- 缺失值中位数与标准化参数只在训练集拟合。",
        "- 超参数只按验证集准确率、Macro-F1、NLL 的顺序选择。",
        "- `selection_frozen_before_test.json` 写入后才加载测试日期；每个最终模型只评价一次测试集。",
        "",
        "## 最终结果",
        "",
        (
            "训练集先验多数类基线（始终预测 flat）："
            f"测试 Accuracy={majority['accuracy']:.4%}，"
            f"Macro-F1={majority['macro_f1']:.4%}，"
            f"Brier={majority['brier_score']:.6f}，"
            f"NLL={majority['negative_log_likelihood']:.6f}。"
        ),
        "",
        "| 模型 | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test Brier | Test NLL | 入选模型训练秒数 | 网格总训练秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *model_rows,
        "",
        "Brier 与 NLL 越低越好。训练耗时不含分钟数据读取和固定维度特征汇总。",
        "所有候选参数和验证指标见 `validation_selection.csv`，逐窗口测试概率见 `test_predictions.csv`。",
    ]
    (target / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_lstm_baselines(
    stock_codes: Sequence[str] | None = None,
    n_stocks: int = 5,
    seq_len: int = 60,
    splits: Mapping[str, tuple[str, str]] | None = None,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    feature_set: str = "enhanced",
    logistic_grid: Sequence[Mapping[str, Any]] = DEFAULT_LOGISTIC_GRID,
    tree_grid: Sequence[Mapping[str, Any]] = DEFAULT_TREE_GRID,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train, select and evaluate traditional baselines without test leakage."""

    if seq_len < 10:
        raise ValueError("seq_len must be at least 10")
    ranges = dict(splits or DEFAULT_SPLITS)
    _validate_splits(ranges)
    minute_dir = Path(data_dir or (OUTPUT_DIR / "minute"))
    target = Path(out_dir or (OUTPUT_DIR / "lstm_baselines"))
    _prepare_output_dir(target, overwrite)
    stocks = _resolve_stocks(minute_dir, stock_codes, n_stocks)
    started_at = _utc_now()

    # Deliberately load train and validation only.  The test range is not
    # passed to _make_split until selected settings have been written below.
    train_matrix, train_labels, train_ids, train_coverage, _, names = _load_summary_split(
        stocks, ranges["train"], seq_len, minute_dir, feature_set, False
    )
    train_matrix, train_labels, train_ids = _stratified_subsample(
        train_matrix, train_labels, train_ids, max_train_samples, seed
    )
    validation_matrix, validation_labels, validation_ids, validation_coverage, _, validation_names = _load_summary_split(
        stocks, ranges["val"], seq_len, minute_dir, feature_set, False
    )
    validation_matrix, validation_labels, validation_ids = _stratified_subsample(
        validation_matrix,
        validation_labels,
        validation_ids,
        max_validation_samples,
        seed + 1,
    )
    if names != validation_names:
        raise RuntimeError("train and validation summary feature names differ")

    models: dict[str, Pipeline] = {}
    selections: dict[str, dict[str, Any]] = {}
    for model_name, grid in (
        ("logistic_regression", logistic_grid),
        ("hist_gradient_boosting", tree_grid),
    ):
        model, selection = select_model_on_validation(
            model_name,
            grid,
            train_matrix,
            train_labels,
            validation_matrix,
            validation_labels,
            seed,
        )
        models[model_name] = model
        selections[model_name] = selection

    _, majority_validation = _prior_baseline(train_labels, validation_labels)
    validation_frozen_at = _utc_now()
    frozen = {
        "frozen_before_test_load": True,
        "frozen_at_utc": validation_frozen_at,
        "selection_objective": "validation_accuracy_then_macro_f1_then_nll",
        "selected_parameters": {
            name: selection["selected_parameters"]
            for name, selection in selections.items()
        },
        "preprocessing_fit_scope": "training_split_only",
    }
    _json_dump(target / "selection_frozen_before_test.json", frozen)
    _selection_rows(selections).to_csv(
        target / "validation_selection.csv", index=False, float_format="%.8f"
    )

    test_loaded_at = _utc_now()
    test_matrix, test_labels, test_ids, test_coverage, test_metadata, test_names = _load_summary_split(
        stocks, ranges["test"], seq_len, minute_dir, feature_set, True
    )
    if names != test_names:
        raise RuntimeError("train and test summary feature names differ")
    assert test_metadata is not None

    _, majority_test = _prior_baseline(train_labels, test_labels)
    test_results: dict[str, dict[str, Any]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        probability = _aligned_probability(model, test_matrix)
        probabilities[model_name] = probability
        test_results[model_name] = evaluate_probabilities(test_labels, probability)

    prediction_table = test_metadata.copy()
    prediction_table["true_label"] = test_labels
    prediction_table["true_class"] = [CLASS_NAMES[value] for value in test_labels]
    for model_name, probability in probabilities.items():
        predicted = probability.argmax(axis=1)
        prefix = "logistic" if model_name == "logistic_regression" else "hist_tree"
        prediction_table[f"{prefix}_prediction"] = predicted
        prediction_table[f"{prefix}_prob_down"] = probability[:, 0]
        prediction_table[f"{prefix}_prob_flat"] = probability[:, 1]
        prediction_table[f"{prefix}_prob_up"] = probability[:, 2]
    prediction_table.to_csv(
        target / "test_predictions.csv", index=False, float_format="%.8f"
    )

    configuration = {
        "pipeline_version": 1,
        "training_entrypoint": "python -m scripts.run_lstm_baselines",
        "stock_codes": stocks,
        "splits": ranges,
        "seq_len": int(seq_len),
        "feature_set": feature_set,
        "target_mode": "three_class",
        "class_names": list(CLASS_NAMES),
        "sequence_summary_statistics": list(SUMMARY_STATISTICS),
        "summary_feature_count": int(len(names)),
        "base_sequence_feature_count": int(len(feature_names(feature_set))),
        "summary_feature_names": names,
        "preprocessing_fit_scope": "training_split_only",
        "selection_objective": "validation_accuracy_then_macro_f1_then_nll",
        "max_train_samples": max_train_samples,
        "max_validation_samples": max_validation_samples,
        "seed": int(seed),
    }
    report: dict[str, Any] = {
        "config": configuration,
        "sizes": {
            "train_used": int(len(train_labels)),
            "validation_used": int(len(validation_labels)),
            "test": int(len(test_labels)),
        },
        "coverage": {
            "train_full": train_coverage,
            "validation_full": validation_coverage,
            "test": test_coverage,
        },
        "majority_prior_baseline": {
            "validation": majority_validation,
            "test": majority_test,
        },
        "models": {
            name: {
                "selected_parameters": selections[name]["selected_parameters"],
                "selected_training_seconds": selections[name][
                    "selected_training_seconds"
                ],
                "search_training_seconds": selections[name][
                    "search_training_seconds"
                ],
                "validation": selections[name]["selected_validation_metrics"],
                "test": test_results[name],
            }
            for name in models
        },
        "audit": {
            "started_at_utc": started_at,
            "validation_selection_frozen_at_utc": validation_frozen_at,
            "test_loaded_at_utc": test_loaded_at,
            "completed_at_utc": _utc_now(),
            "test_loaded_after_selection_frozen": test_loaded_at >= validation_frozen_at,
            "test_evaluation_count_per_model": 1,
            "test_data_policy": (
                "The test split was passed to _make_split only after validation "
                "selection was persisted; its probabilities were computed once per model."
            ),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }
    _json_dump(target / "test_metrics.json", report)
    _write_output_readme(target, report)
    joblib.dump(
        {"models": models, "config": configuration},
        target / "models.joblib",
        compress=3,
    )
    return {
        "out_dir": str(target),
        "report": report,
        "models": models,
        "test_predictions": prediction_table,
    }
