"""Build a validation-frozen LSTM and HistGB probability hybrid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_hybrid import run_lstm_hybrid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lstm-model", type=Path, default=OUTPUT_DIR / "lstm_full" / "model.pt")
    parser.add_argument(
        "--baseline-model",
        type=Path,
        default=OUTPUT_DIR / "lstm_baselines" / "models.joblib",
    )
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_hybrid")
    parser.add_argument("--weight-step", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=(
            "macro_f1_then_accuracy_then_nll",
            "accuracy_then_macro_f1_then_nll",
        ),
        default="macro_f1_then_accuracy_then_nll",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run_lstm_hybrid(
        args.lstm_model,
        args.baseline_model,
        args.data_dir,
        args.out_dir,
        weight_step=args.weight_step,
        objective=args.objective,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    hybrid = result["report"]["hybrid"]
    print(json.dumps({
        "out_dir": result["out_dir"],
        "selection": result["report"]["selection"],
        "test_accuracy": hybrid["accuracy"],
        "test_macro_f1": hybrid["macro_f1"],
        "test_brier": hybrid["brier_score"],
        "test_nll": hybrid["calibration"]["negative_log_likelihood"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

