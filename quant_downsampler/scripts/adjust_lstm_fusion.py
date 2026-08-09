"""Retune saved LSTM fusion on validation only, then evaluate once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qd.config import OUTPUT_DIR
from qd.lstm_adjustment import adjust_saved_fusion


def main() -> None:
    parser = argparse.ArgumentParser(description="validation-only LSTM fusion adjustment")
    parser.add_argument("--model", type=Path, default=OUTPUT_DIR / "lstm_full" / "model.pt")
    parser.add_argument("--data-dir", type=Path, default=OUTPUT_DIR / "minute")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "lstm_adjusted")
    parser.add_argument(
        "--objective",
        choices=("macro_f1_then_accuracy", "accuracy_then_macro_f1"),
        default="macro_f1_then_accuracy",
    )
    parser.add_argument("--move-bias-min", type=float, default=-0.30)
    parser.add_argument("--move-bias-max", type=float, default=0.30)
    parser.add_argument("--move-bias-step", type=float, default=0.01)
    parser.add_argument("--joint-weight-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = adjust_saved_fusion(
        args.model,
        args.data_dir,
        args.out_dir,
        objective=args.objective,
        move_bias_min=args.move_bias_min,
        move_bias_max=args.move_bias_max,
        move_bias_step=args.move_bias_step,
        joint_weight_step=args.joint_weight_step,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "out_dir": result["out_dir"],
        "selection": result["selection"],
        "test_accuracy": result["test_metrics"]["accuracy"],
        "test_macro_f1": result["test_metrics"]["macro_f1"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

