"""Minimal Task 5 baseline mapped from the prompt's four raw trade fields."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import OUTPUT_DIR
from .lstm_components import ENHANCED_FEATURE_NAMES
from .tradable_lstm import run_tradable_lstm


PROMPT_FIELD_TO_FEATURE = {
    "Time": "session_progress",
    "Price": "close_log_return",
    "Volume": "log1p_volume",
    "BSFlag": "buy_volume_imbalance",
}
PROMPT_FOUR_FEATURES = tuple(PROMPT_FIELD_TO_FEATURE.values())


def _comparison_row(label: str, summary: dict[str, Any], feature_count: int) -> dict[str, object]:
    return {
        **summary["test_strategy"],
        "model": label,
        "input_feature_count": feature_count,
        "classification_accuracy": summary["test_classification_metrics"]["accuracy"],
        "classification_macro_f1": summary["test_classification_metrics"]["macro_f1"],
        "return_spearman_ic": summary["test_return_metrics"]["spearman_ic"],
    }


def run_lstm_minimal_four(
    output_dir: str | Path = OUTPUT_DIR,
    epochs: int = 8,
    sell_fee_bps: float = 5.0,
    device: str = "cpu",
) -> dict[str, object]:
    """Train the raw-field-mapped four-feature model and compare like-for-like."""
    if len(PROMPT_FOUR_FEATURES) != 4 or not set(PROMPT_FOUR_FEATURES).issubset(ENHANCED_FEATURE_NAMES):
        raise RuntimeError("prompt four-feature mapping is invalid")
    root = Path(output_dir)
    metadata = {
        "status": "fixed_before_training",
        "selection_split": "none",
        "method": "one stationary causal feature mapped to each raw prompt field",
        "prompt_field_to_feature": PROMPT_FIELD_TO_FEATURE,
        "interpretation_note": (
            "the prompt specifies four raw tick fields, not four named LSTM factors; "
            "this is the minimal deployable mapping"
        ),
    }
    result = run_tradable_lstm(
        output_dir=root,
        epochs=epochs,
        sell_fee_bps=sell_fee_bps,
        device=device,
        selected_feature_names=PROMPT_FOUR_FEATURES,
        target_subdir="lstm_minimal_four",
        model_name="prompt_raw_four_multitask_lstm",
        feature_selection_metadata=metadata,
    )
    full = json.loads((root / "tradable_lstm" / "summary.json").read_text(encoding="utf-8"))
    pruned = json.loads(
        (root / "lstm_feature_independence" / "summary.json").read_text(encoding="utf-8")
    )
    comparison = pd.DataFrame([
        _comparison_row("all_25", full, 25),
        _comparison_row("cluster_pruned_19", pruned, 19),
        _comparison_row("prompt_raw_four", result, 4),
    ])
    target = root / "lstm_minimal_four"
    comparison.to_csv(target / "model_comparison.csv", index=False, float_format="%.10f")
    summary = {
        "status": "completed",
        "prompt_field_to_feature": PROMPT_FIELD_TO_FEATURE,
        "feature_names": list(PROMPT_FOUR_FEATURES),
        "feature_count": 4,
        "horizon": result["horizon"],
        "test_classification_metrics": result["test_classification_metrics"],
        "test_return_metrics": result["test_return_metrics"],
        "test_strategy": result["test_strategy"],
        "comparison": comparison.to_dict(orient="records"),
        "known_limitation": "the fixed validation/test dates were already inspected; exploratory only",
    }
    (target / "minimal_four_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
