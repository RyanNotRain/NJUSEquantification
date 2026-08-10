"""Train-only minute-feature de-redundancy and LSTM component diversity audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from .config import OUTPUT_DIR
from .factor_independence import correlation_clusters, independence_metrics
from .lstm_baselines import probability_metrics
from .lstm_components import ENHANCED_FEATURE_NAMES
from .tradable_lstm import _raw_split, _screened_horizon, run_tradable_lstm
from .tradable_return_research import attach_horizon_returns


SUMMARY_VIEWS = ("last", "mean", "std", "change")


def causal_summary_views(sequences: np.ndarray) -> dict[str, np.ndarray]:
    """Four fixed-window summaries that never use the target minute."""
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] < 2:
        raise ValueError("sequences must have shape (samples, time>=2, features)")
    return {
        "last": values[:, -1, :],
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "change": values[:, -1, :] - values[:, 0, :],
    }


def training_feature_statistics(
    sequences: np.ndarray,
    target_return: np.ndarray,
    feature_names: list[str],
    max_rows: int = 40_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return mean-absolute view correlation and train-only target Rank IC."""
    x = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(target_return, dtype=np.float64)
    if len(x) != len(y) or x.shape[2] != len(feature_names):
        raise ValueError("features, names, and target must align")
    sample_count = min(len(x), int(max_rows))
    positions = np.linspace(0, len(x) - 1, sample_count, dtype=int)
    sampled_x = x[positions]
    sampled_y = pd.Series(y[positions], name="target")
    views = causal_summary_views(sampled_x)
    correlations: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    for view_name, values in views.items():
        frame = pd.DataFrame(values, columns=feature_names)
        correlations.append(frame.corr(method="spearman", min_periods=100).abs())
        for feature in feature_names:
            pair = pd.concat([frame[feature], sampled_y], axis=1).replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            ic = (
                float(pair[feature].corr(pair["target"], method="spearman"))
                if len(pair) >= 100 and pair[feature].nunique() > 1 else 0.0
            )
            quality_rows.append({"feature": feature, "view": view_name, "rank_ic": ic})
    mean_absolute = (sum(matrix.fillna(0.0) for matrix in correlations) / len(correlations)).copy()
    for feature in feature_names:
        mean_absolute.loc[feature, feature] = 1.0
    detail = pd.DataFrame(quality_rows)
    quality = detail.groupby("feature", sort=False).agg(
        maximum_absolute_target_rank_ic=("rank_ic", lambda values: float(np.abs(values).max())),
        best_view=("rank_ic", lambda values: str(detail.loc[values.abs().idxmax(), "view"])),
        signed_rank_ic_at_best_view=("rank_ic", lambda values: float(values.loc[values.abs().idxmax()])),
    )
    quality["sample_rows"] = sample_count
    return mean_absolute, quality, detail


def select_feature_representatives(
    clusters: list[list[str]],
    quality: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame]:
    """Select one feature in each frozen cluster by train-only target Rank IC."""
    selected: list[str] = []
    rows: list[dict[str, object]] = []
    for cluster_id, members in enumerate(clusters, 1):
        representative = sorted(
            members,
            key=lambda name: (-float(quality.loc[name, "maximum_absolute_target_rank_ic"]), name),
        )[0]
        selected.append(representative)
        for member in members:
            rows.append({
                "cluster_id": cluster_id,
                "feature": member,
                "representative": representative,
                "selected": member == representative,
                **quality.loc[member].to_dict(),
            })
    return selected, pd.DataFrame(rows)


def _probability(frame: pd.DataFrame, prefix: str) -> np.ndarray:
    if prefix == "blend":
        columns = ["prob_down", "prob_flat", "prob_up"]
    else:
        columns = [f"{prefix}_prob_down", f"{prefix}_prob_flat", f"{prefix}_prob_up"]
    return frame[columns].to_numpy(dtype=np.float64)


def component_complementarity(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare joint, staged, and frozen blend probabilities on aligned rows."""
    labels = frame["true_label"].to_numpy(dtype=np.int64)
    probabilities = {name: _probability(frame, name) for name in ("joint", "staged", "blend")}
    rows = [{"probability_source": name, **probability_metrics(labels, values)}
            for name, values in probabilities.items()]
    joint_prediction = probabilities["joint"].argmax(axis=1)
    staged_prediction = probabilities["staged"].argmax(axis=1)
    blend_prediction = probabilities["blend"].argmax(axis=1)
    disagreement = joint_prediction != staged_prediction
    joint_correct = joint_prediction == labels
    staged_correct = staged_prediction == labels
    clipped_joint = np.clip(probabilities["joint"], 1e-12, 1.0)
    clipped_staged = np.clip(probabilities["staged"], 1e-12, 1.0)
    midpoint = (clipped_joint + clipped_staged) / 2.0
    js = 0.5 * np.sum(clipped_joint * np.log(clipped_joint / midpoint), axis=1)
    js += 0.5 * np.sum(clipped_staged * np.log(clipped_staged / midpoint), axis=1)
    correct_corr = (
        float(np.corrcoef(joint_correct.astype(float), staged_correct.astype(float))[0, 1])
        if joint_correct.std() and staged_correct.std() else 0.0
    )
    summary = {
        "joint_staged_prediction_agreement": float((~disagreement).mean()),
        "joint_staged_correctness_correlation": correct_corr,
        "disagreement_rate": float(disagreement.mean()),
        "mean_jensen_shannon_divergence": float(js.mean()),
        "joint_accuracy_on_disagreement": float(joint_correct[disagreement].mean()) if disagreement.any() else 0.0,
        "staged_accuracy_on_disagreement": float(staged_correct[disagreement].mean()) if disagreement.any() else 0.0,
        "blend_accuracy_on_disagreement": float((blend_prediction[disagreement] == labels[disagreement]).mean()) if disagreement.any() else 0.0,
    }
    return pd.DataFrame(rows), summary


def component_signal_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    signals = pd.DataFrame({
        "direction_signed": 2.0 * frame["direction_prob_up"] - 1.0,
        "movement_probability": frame["movement_prob_move"],
        "joint_direction": frame["joint_prob_up"] - frame["joint_prob_down"],
    })
    return signals.corr(method="spearman")


def _plot_feature_correlation(correlation: pd.DataFrame, selected: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    for axis, matrix, title in (
        (axes[0], correlation, "All 25 enhanced features (train only)"),
        (axes[1], correlation.loc[selected, selected], f"Cluster representatives ({len(selected)})"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_title(title)
        axis.set_xticks(range(len(matrix)), labels=matrix.columns, rotation=90, fontsize=6)
        axis.set_yticks(range(len(matrix)), labels=matrix.index, fontsize=6)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.82, pad=.04, label="Mean absolute Spearman")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_lstm_feature_independence(
    output_dir: str | Path = OUTPUT_DIR,
    correlation_threshold: float = 0.85,
    max_training_rows: int = 40_000,
    epochs: int = 8,
    sell_fee_bps: float = 5.0,
    device: str = "cpu",
    reuse_frozen_selection: bool = False,
) -> dict[str, object]:
    """Freeze train-only feature representatives and train a compact T+1 LSTM."""
    root = Path(output_dir)
    target = root / "lstm_feature_independence"
    target.mkdir(parents=True, exist_ok=True)
    ensemble = torch.load(root / "lstm_ensemble" / "model.pt", map_location="cpu", weights_only=False)["config"]
    stocks = list(ensemble["stock_codes"])
    splits = {name: tuple(value) for name, value in ensemble["splits"].items()}
    seq_len = int(ensemble["seq_len"])
    horizon = _screened_horizon(root)
    minute_dir = root / "minute"

    feature_names = list(ENHANCED_FEATURE_NAMES)
    frozen_path = target / "feature_selection_frozen.json"
    if reuse_frozen_selection:
        if not frozen_path.exists():
            raise FileNotFoundError("cannot reuse a missing frozen feature selection")
        selection_metadata = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (
            selection_metadata.get("status") != "frozen_before_validation_and_test_training"
            or not set(selection_metadata.get("selected_features", [])).issubset(feature_names)
        ):
            raise ValueError("saved feature selection is invalid")
        selected = list(selection_metadata["selected_features"])
        cluster_count = int(pd.read_csv(target / "feature_clusters.csv")["cluster_id"].nunique())
        print("reusing the already frozen train-only feature selection")
    else:
        print("loading training split for minute-feature selection; validation/test remain unopened")
        train_x, _, _, _, train_metadata = _raw_split(stocks, splits["train"], seq_len, minute_dir)
        train_returns = attach_horizon_returns(train_metadata, minute_dir)
        mask = train_returns[horizon].notna().to_numpy()
        target_fraction = train_returns.loc[mask, horizon].to_numpy(dtype=float)
        correlation, quality, quality_detail = training_feature_statistics(
            train_x[mask], target_fraction, feature_names, max_rows=max_training_rows
        )
        clusters = correlation_clusters(correlation, correlation_threshold)
        cluster_count = len(clusters)
        selected, cluster_table = select_feature_representatives(clusters, quality)
        del train_x

        correlation.to_csv(target / "training_feature_correlation.csv", float_format="%.10f")
        quality.reset_index().to_csv(target / "training_feature_quality.csv", index=False, float_format="%.10f")
        quality_detail.to_csv(target / "training_feature_quality_by_view.csv", index=False, float_format="%.10f")
        cluster_table.to_csv(target / "feature_clusters.csv", index=False, float_format="%.10f")
        selected_correlation = correlation.loc[selected, selected]
        pd.DataFrame([
            {"feature_set": "all_25", **independence_metrics(correlation)},
            {"feature_set": "cluster_representatives", **independence_metrics(selected_correlation)},
        ]).to_csv(target / "feature_independence_metrics.csv", index=False, float_format="%.10f")
        _plot_feature_correlation(correlation, selected, target / "feature_correlation.png")

        selection_metadata = {
            "status": "frozen_before_validation_and_test_training",
            "selection_split": "train",
            "method": "connected components of mean absolute Spearman across last/mean/std/change summaries",
            "correlation_threshold": correlation_threshold,
            "representative_score": "maximum absolute train T+1 Rank IC across four causal summaries",
            "training_sample_rows": int(quality["sample_rows"].iloc[0]),
            "selected_features": selected,
        }
        frozen_path.write_text(
            json.dumps(selection_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    result = run_tradable_lstm(
        output_dir=root,
        epochs=epochs,
        sell_fee_bps=sell_fee_bps,
        device=device,
        selected_feature_names=selected,
        target_subdir="lstm_feature_independence",
        model_name="cluster_pruned_multitask_lstm",
        feature_selection_metadata=selection_metadata,
    )

    component_rows: list[pd.DataFrame] = []
    complementarity: dict[str, Mapping[str, float]] = {}
    for split, filename in (("validation", "validation_predictions.csv"), ("test", "test_predictions.csv")):
        frame = pd.read_csv(root / "lstm_ensemble" / filename)
        metrics, diagnostic = component_complementarity(frame)
        metrics.insert(0, "split", split)
        component_rows.append(metrics)
        complementarity[split] = diagnostic
        component_signal_correlation(frame).to_csv(
            target / f"component_signal_correlation_{split}.csv", float_format="%.10f"
        )
    pd.concat(component_rows, ignore_index=True).to_csv(
        target / "component_ablation_metrics.csv", index=False, float_format="%.10f"
    )
    pd.DataFrame.from_dict(complementarity, orient="index").rename_axis("split").to_csv(
        target / "component_complementarity.csv", float_format="%.10f"
    )

    full_summary = json.loads((root / "tradable_lstm" / "summary.json").read_text(encoding="utf-8"))
    model_comparison = pd.DataFrame([
        {
            **full_summary["test_strategy"],
            "model": "compact_multitask_lstm_all_25",
            "input_feature_count": int(full_summary.get("input_feature_count", 25)),
            "classification_accuracy": full_summary["test_classification_metrics"]["accuracy"],
            "classification_macro_f1": full_summary["test_classification_metrics"]["macro_f1"],
            "return_spearman_ic": full_summary["test_return_metrics"]["spearman_ic"],
        },
        {
            **result["test_strategy"],
            "model": "cluster_pruned_multitask_lstm_19",
            "input_feature_count": len(selected),
            "classification_accuracy": result["test_classification_metrics"]["accuracy"],
            "classification_macro_f1": result["test_classification_metrics"]["macro_f1"],
            "return_spearman_ic": result["test_return_metrics"]["spearman_ic"],
        },
    ])
    model_comparison.to_csv(target / "full_vs_pruned_lstm.csv", index=False, float_format="%.10f")

    summary = {
        "status": "completed",
        "horizon": horizon,
        "feature_selection": selection_metadata,
        "cluster_count": cluster_count,
        "original_feature_count": len(feature_names),
        "selected_feature_count": len(selected),
        "selected_features": selected,
        "full_vs_pruned": model_comparison.to_dict(orient="records"),
        "component_complementarity": complementarity,
        "known_limitation": "fixed validation/test dates have already been inspected; exploratory only",
    }
    (target / "feature_independence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary
