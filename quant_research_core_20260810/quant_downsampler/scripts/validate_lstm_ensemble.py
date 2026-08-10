"""Strictly replay the saved ensemble and compare every test probability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qd.config import OUTPUT_DIR
from qd.lstm_ensemble import evaluate_full_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="reload the three-component checkpoint and audit saved predictions"
    )
    parser.add_argument("--run-dir", type=Path, default=OUTPUT_DIR / "lstm_ensemble")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--atol", type=float, default=1e-4,
        help="GPU/CPU and batch-size inference tolerance for saved probabilities",
    )
    args = parser.parse_args()

    model_path = args.run_dir / "model.pt"
    prediction_path = args.run_dir / "test_predictions.csv"
    metrics_path = args.run_dir / "test_metrics.json"
    for path in (model_path, prediction_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(path)

    replay = evaluate_full_model(
        model_path, args.data_dir, split="test",
        batch_size=args.batch_size, device=args.device,
    )
    saved = pd.read_csv(prediction_path)
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    probability = np.asarray(replay["probability"], dtype=float)
    labels = np.asarray(replay["labels"], dtype=int)
    metadata = replay["metadata"].drop(columns="stock_id").reset_index(drop=True)

    if len(saved) != len(labels):
        raise AssertionError(f"row count differs: saved={len(saved)}, replay={len(labels)}")
    for column in ("stock", "date", "window_end", "target_time"):
        if not saved[column].astype(str).equals(metadata[column].astype(str)):
            raise AssertionError(f"metadata column differs: {column}")
    if not np.array_equal(saved["true_label"].to_numpy(dtype=int), labels):
        raise AssertionError("saved labels differ from replayed labels")
    saved_probability = saved[["prob_down", "prob_flat", "prob_up"]].to_numpy(float)
    maximum_probability_error = float(np.max(np.abs(saved_probability - probability)))
    if maximum_probability_error > args.atol:
        raise AssertionError(
            f"probabilities differ by {maximum_probability_error:.3g}, atol={args.atol}"
        )
    predicted = probability.argmax(axis=1)
    if not np.array_equal(saved["predicted_label"].to_numpy(dtype=int), predicted):
        raise AssertionError("saved predictions differ from replayed argmax")

    replay_metrics = replay["metrics"]
    for name, saved_name in (("accuracy", "accuracy"), ("f1", "macro_f1")):
        if not np.isclose(replay_metrics[name], saved_metrics[saved_name], atol=1e-12):
            raise AssertionError(f"metric differs: {saved_name}")
    if replay_metrics["confusion_matrix"] != saved_metrics["confusion_matrix"]:
        raise AssertionError("confusion matrix differs")

    report = {
        "status": "passed",
        "run_dir": str(args.run_dir.resolve()),
        "model_reloaded": True,
        "rows_replayed": int(len(labels)),
        "metadata_exact": True,
        "labels_exact": True,
        "predicted_classes_exact": True,
        "probabilities_within_tolerance": True,
        "maximum_probability_error": maximum_probability_error,
        "probability_tolerance": args.atol,
        "accuracy": float(replay_metrics["accuracy"]),
        "macro_f1": float(replay_metrics["f1"]),
        "confusion_matrix": replay_metrics["confusion_matrix"],
    }
    (args.run_dir / "replay_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
