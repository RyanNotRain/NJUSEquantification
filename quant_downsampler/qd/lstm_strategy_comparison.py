"""Common-horizon economic comparison for minute prediction models.

This module compares saved LSTM, HistGradientBoosting and hybrid
probabilities under one pre-declared trading rule.  Every source must contain
the exact same stock/window-end/target-time keys.  Realised returns are loaded
once from exact minute closes and then shared by all models, so a model cannot
benefit from a different sample or timestamp convention.

The primary reference is a five-stock equal-weight market proxy evaluated on
the same one-minute forecast horizons.  A second equal-weight buy-and-hold
reference compounds only those observed horizons.  Neither benchmark is used
to select a model, threshold, or position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .lstm_strategy import (
    PROBABILITY_COLUMNS,
    attach_realised_returns,
    break_even_cost_bps,
    build_portfolio_path,
    strategy_statistics,
    validate_predictions,
)


KEY_COLUMNS = ("stock", "window_end", "target_time")
SIGNAL_MODES = ("probability_gap", "expected_return_bps")
DEFAULT_SOURCE_PATHS = {
    "original_lstm": OUTPUT_DIR / "lstm_full" / "test_predictions.csv",
    "hist_gradient_boosting": OUTPUT_DIR
    / "lstm_baselines"
    / "test_predictions.csv",
    "hybrid": OUTPUT_DIR / "lstm_hybrid" / "test_predictions.csv",
}
DEFAULT_PROBABILITY_COLUMNS: dict[str, dict[str, str]] = {
    "original_lstm": {name: name for name in PROBABILITY_COLUMNS},
    "hist_gradient_boosting": {
        "prob_down": "hist_tree_prob_down",
        "prob_flat": "hist_tree_prob_flat",
        "prob_up": "hist_tree_prob_up",
    },
    "hybrid": {name: name for name in PROBABILITY_COLUMNS},
}
GENERATED_ROOT_FILES = (
    "strategy_comparison.csv",
    "wealth_curves.csv",
    "aligned_sample_returns.csv",
    "metrics.json",
    "strategy_comparison.png",
)
BENCHMARK_NAMES = (
    "equal_weight_market_proxy",
    "equal_weight_buy_and_hold_observed_horizons",
)


def _required_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _normalise_prediction_source(
    frame: pd.DataFrame,
    *,
    source_name: str,
    signal_mode: str,
    probability_columns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return canonical keys and signal columns for one saved model."""

    if signal_mode not in SIGNAL_MODES:
        raise ValueError(f"signal_mode must be one of {SIGNAL_MODES}")
    _required_columns(frame, KEY_COLUMNS)
    result = frame.copy().reset_index(drop=True)
    result["stock"] = result["stock"].astype(str)
    result["_window_end_ts"] = pd.to_datetime(result["window_end"], errors="coerce")
    result["_target_time_ts"] = pd.to_datetime(result["target_time"], errors="coerce")
    if result[["_window_end_ts", "_target_time_ts"]].isna().any().any():
        raise ValueError(f"{source_name} contains an invalid timestamp")
    if not (result["_target_time_ts"] > result["_window_end_ts"]).all():
        raise ValueError(f"{source_name} contains a non-forward target")
    if result[["stock", "_window_end_ts", "_target_time_ts"]].duplicated().any():
        raise ValueError(f"{source_name} contains duplicate sample keys")

    mapping = dict(probability_columns or {})
    if signal_mode == "probability_gap":
        if not mapping:
            mapping = {name: name for name in PROBABILITY_COLUMNS}
        _required_columns(result, list(mapping.values()))
        if sorted(mapping) != sorted(PROBABILITY_COLUMNS):
            raise ValueError(
                f"{source_name} probability mapping must define {PROBABILITY_COLUMNS}"
            )
        for canonical, source in mapping.items():
            result[canonical] = result[source]
        # Central validation enforces finite [0, 1] values summing to one.
        result = validate_predictions(result)
    else:
        _required_columns(result, ("expected_return_bps",))
        values = pd.to_numeric(result["expected_return_bps"], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{source_name} expected_return_bps must be finite")
        result["expected_return_bps"] = values
        # Probability columns are optional for a future pure return model.  If
        # present, retain them for audit but do not use them to filter trades.
        if mapping:
            _required_columns(result, list(mapping.values()))
            for canonical, source in mapping.items():
                result[canonical] = result[source]

    result["window_end"] = result["_window_end_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    result["target_time"] = result["_target_time_ts"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    return result.sort_values(
        ["_window_end_ts", "stock", "_target_time_ts"]
    ).reset_index(drop=True)


def align_prediction_sources(
    sources: Mapping[str, pd.DataFrame],
    *,
    signal_modes: Mapping[str, str] | None = None,
    probability_columns: Mapping[str, Mapping[str, str]] | None = None,
    expected_stock_count: int | None = 5,
    expected_day_count: int | None = 10,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Strictly align saved predictions without intersecting away bad rows.

    Alignment uses equality of the complete sorted key table.  It deliberately
    refuses an inner join: silently dropping a source's missing rows would
    give the compared models different economic opportunities.
    """

    if len(sources) < 2:
        raise ValueError("at least two prediction sources are required")
    modes = dict(signal_modes or {})
    mappings = dict(probability_columns or {})
    aligned: dict[str, pd.DataFrame] = {}
    for name, source in sources.items():
        mode = modes.get(name, "probability_gap")
        aligned[name] = _normalise_prediction_source(
            source,
            source_name=name,
            signal_mode=mode,
            probability_columns=mappings.get(name),
        )

    reference_name = next(iter(aligned))
    reference = aligned[reference_name]
    key_columns = ["stock", "_window_end_ts", "_target_time_ts"]
    reference_keys = reference[key_columns]
    for name, frame in aligned.items():
        if len(frame) != len(reference) or not frame[key_columns].equals(
            reference_keys
        ):
            reference_index = pd.MultiIndex.from_frame(reference_keys)
            candidate_index = pd.MultiIndex.from_frame(frame[key_columns])
            missing = reference_index.difference(candidate_index)
            extra = candidate_index.difference(reference_index)
            raise ValueError(
                f"{name} sample keys do not exactly match {reference_name}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )

    stocks = sorted(reference["stock"].unique().tolist())
    days = sorted(reference["_window_end_ts"].dt.strftime("%Y-%m-%d").unique())
    if expected_stock_count is not None and len(stocks) != expected_stock_count:
        raise ValueError(
            f"expected {expected_stock_count} stocks but aligned data contain {len(stocks)}"
        )
    if expected_day_count is not None and len(days) != expected_day_count:
        raise ValueError(
            f"expected {expected_day_count} days but aligned data contain {len(days)}"
        )
    cross_sections = reference.groupby("_window_end_ts", sort=False)["stock"].agg(
        lambda values: tuple(sorted(values))
    )
    expected_cross_section = tuple(stocks)
    if not cross_sections.map(lambda value: value == expected_cross_section).all():
        raise ValueError("at least one forecast timestamp lacks the full stock universe")
    targets_per_time = reference.groupby("_window_end_ts")["_target_time_ts"].nunique()
    if not targets_per_time.eq(1).all():
        raise ValueError("stocks at the same window_end have inconsistent targets")

    label_sources = {
        name: pd.to_numeric(frame["true_label"], errors="coerce").to_numpy()
        for name, frame in aligned.items()
        if "true_label" in frame
    }
    if label_sources:
        first_label_name = next(iter(label_sources))
        first_labels = label_sources[first_label_name]
        if not np.isfinite(first_labels).all():
            raise ValueError(f"{first_label_name} contains invalid true labels")
        for name, labels in label_sources.items():
            if not np.isfinite(labels).all() or not np.array_equal(labels, first_labels):
                raise ValueError(f"{name} true labels do not match {first_label_name}")

    audit = {
        "reference_source": reference_name,
        "n_rows": int(len(reference)),
        "n_intervals": int(reference["_window_end_ts"].nunique()),
        "n_stocks": int(len(stocks)),
        "stocks": stocks,
        "n_days": int(len(days)),
        "days": days,
        "all_sample_keys_exactly_equal": True,
        "full_stock_cross_section_at_every_interval": True,
        "labels_equal_where_available": True,
    }
    return aligned, audit


def generate_signal_weights(
    predictions: pd.DataFrame,
    *,
    signal_mode: str = "probability_gap",
    score_threshold: float = 0.0,
    side: str = "long_short",
    weighting: str = "confidence",
    require_directional_argmax: bool = True,
) -> pd.DataFrame:
    """Build one common strategy from a prediction-time signal only.

    ``expected_return_bps`` is an intentionally separate signal mode for the
    new magnitude model.  It uses the same portfolio and cost engine, while
    keeping its threshold in basis-point units.
    """

    if signal_mode not in SIGNAL_MODES:
        raise ValueError(f"signal_mode must be one of {SIGNAL_MODES}")
    if side not in ("long_short", "long_only"):
        raise ValueError("side must be long_short or long_only")
    if weighting not in ("equal", "confidence"):
        raise ValueError("weighting must be equal or confidence")
    if not np.isfinite(score_threshold) or score_threshold < 0.0:
        raise ValueError("score_threshold must be finite and non-negative")
    if signal_mode == "probability_gap" and score_threshold > 1.0:
        raise ValueError("probability-gap threshold cannot exceed one")

    frame = predictions.copy().reset_index(drop=True)
    _required_columns(frame, KEY_COLUMNS)
    if "_window_end_ts" not in frame:
        frame["_window_end_ts"] = pd.to_datetime(frame["window_end"], errors="coerce")
    if "_target_time_ts" not in frame:
        frame["_target_time_ts"] = pd.to_datetime(frame["target_time"], errors="coerce")
    if frame[["_window_end_ts", "_target_time_ts"]].isna().any().any():
        raise ValueError("prediction timestamps are invalid")

    if signal_mode == "probability_gap":
        frame = validate_predictions(frame)
        score = frame["prob_up"] - frame["prob_down"]
        active = score.abs().ge(float(score_threshold)) & score.ne(0.0)
        if require_directional_argmax:
            active &= frame[["prob_up", "prob_down"]].max(axis=1) > frame["prob_flat"]
    else:
        _required_columns(frame, ("expected_return_bps",))
        score = pd.to_numeric(frame["expected_return_bps"], errors="coerce")
        if not np.isfinite(score.to_numpy(dtype=np.float64)).all():
            raise ValueError("expected_return_bps must be finite")
        active = score.abs().ge(float(score_threshold)) & score.ne(0.0)
    if side == "long_only":
        active &= score > 0.0

    direction = np.sign(score.to_numpy(dtype=np.float64))
    raw = direction if weighting == "equal" else score.to_numpy(dtype=np.float64)
    raw = np.where(active.to_numpy(), raw, 0.0)
    frame["signal_mode"] = signal_mode
    frame["signal_score"] = score
    frame["active_signal"] = active
    frame["raw_weight"] = raw
    gross = frame.groupby("_window_end_ts", sort=False)["raw_weight"].transform(
        lambda values: values.abs().sum()
    ).to_numpy(dtype=np.float64)
    frame["target_weight"] = np.divide(
        raw,
        gross,
        out=np.zeros(len(frame), dtype=np.float64),
        where=gross > 0.0,
    )
    frame["score_threshold"] = float(score_threshold)
    frame["side"] = side
    frame["weighting"] = weighting
    return frame.sort_values(["_window_end_ts", "stock"]).reset_index(drop=True)


def build_equal_weight_market_proxy(realised: pd.DataFrame) -> pd.DataFrame:
    """Return a same-horizon, cross-sectionally rebalanced market proxy."""

    _required_columns(realised, (*KEY_COLUMNS, "realised_return"))
    frame = realised.copy()
    counts = frame.groupby("window_end")["stock"].transform("nunique")
    if (counts <= 0).any():
        raise ValueError("market proxy contains an empty cross-section")
    frame["target_weight"] = 1.0 / counts.to_numpy(dtype=np.float64)
    return build_portfolio_path(frame)


def build_equal_weight_buy_and_hold(realised: pd.DataFrame) -> pd.DataFrame:
    """Buy and hold the five names over the *observed forecast horizons*.

    The reference starts with equal notional and never rebalances.  Because
    model returns cover only legal one-minute targets, unobserved warm-up,
    lunch and overnight returns are intentionally excluded here too.  This
    keeps the economic horizon identical rather than giving the benchmark
    additional holding periods.
    """

    _required_columns(realised, (*KEY_COLUMNS, "realised_return"))
    frame = realised.copy()
    frame["_window_end_ts"] = pd.to_datetime(frame["window_end"], errors="coerce")
    frame["_target_time_ts"] = pd.to_datetime(frame["target_time"], errors="coerce")
    if frame[["_window_end_ts", "_target_time_ts"]].isna().any().any():
        raise ValueError("benchmark timestamps are invalid")
    stocks = sorted(frame["stock"].astype(str).unique().tolist())
    if not stocks:
        raise ValueError("buy-and-hold reference has no stocks")
    wealth = np.ones(len(stocks), dtype=np.float64) / len(stocks)
    stock_location = {stock: index for index, stock in enumerate(stocks)}
    records: list[dict[str, Any]] = []
    for window_end, group in frame.groupby("_window_end_ts", sort=True):
        if sorted(group["stock"].astype(str).tolist()) != stocks:
            raise ValueError("buy-and-hold reference requires a complete stock panel")
        targets = group["_target_time_ts"].unique()
        if len(targets) != 1:
            raise ValueError("stocks at one window_end have inconsistent targets")
        returns = np.empty(len(stocks), dtype=np.float64)
        for row in group.itertuples():
            returns[stock_location[str(row.stock)]] = float(row.realised_return)
        if not np.isfinite(returns).all() or (returns <= -1.0).any():
            raise ValueError("buy-and-hold reference contains an invalid return")
        weights = wealth / wealth.sum()
        gross_return = float(np.dot(weights, returns))
        records.append({
            "window_end": pd.Timestamp(window_end),
            "target_time": pd.Timestamp(targets[0]),
            "gross_return": gross_return,
            "gross_exposure": 1.0,
            "net_exposure": 1.0,
            "active_names": len(stocks),
        })
        wealth *= 1.0 + returns
    if not records:
        raise ValueError("buy-and-hold reference has no intervals")
    path = pd.DataFrame(records)
    path["turnover"] = 0.0
    path.loc[path.index[0], "turnover"] += 1.0
    path.loc[path.index[-1], "turnover"] += 1.0
    path["date"] = path["window_end"].dt.strftime("%Y-%m-%d")
    return path


def _relative_return(strategy_return: float, benchmark_return: float) -> float:
    if strategy_return <= -1.0 or benchmark_return <= -1.0:
        raise ValueError("relative return requires terminal wealth above zero")
    return float((1.0 + strategy_return) / (1.0 + benchmark_return) - 1.0)


def _summarise_path(
    name: str,
    kind: str,
    path: pd.DataFrame,
    *,
    cost_bps: float,
    row_coverage: float,
    interval_coverage: float,
) -> dict[str, Any]:
    statistics = strategy_statistics(path, cost_bps=cost_bps)
    return {
        "name": name,
        "kind": kind,
        "row_coverage": float(row_coverage),
        "interval_coverage": float(interval_coverage),
        "break_even_cost_bps": break_even_cost_bps(path),
        **statistics,
    }


def _prepare_output_directory(
    target: Path,
    overwrite: bool,
    generated_path_names: Sequence[str] = (),
) -> None:
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty comparison directory: {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    path_directory = target / "portfolio_paths"
    path_directory.mkdir(exist_ok=True)
    if overwrite:
        owned_names = set(map(str, generated_path_names))
        previous_metrics = target / "metrics.json"
        if previous_metrics.is_file():
            try:
                previous = json.loads(previous_metrics.read_text(encoding="utf-8"))
                owned_names.update(map(str, previous.get("sources", {}).keys()))
                owned_names.update(BENCHMARK_NAMES)
            except (json.JSONDecodeError, OSError, TypeError):
                # A malformed unknown file is preserved; it is not authority to
                # delete any portfolio path.
                pass
        for filename in GENERATED_ROOT_FILES:
            path = target / filename
            if path.is_file():
                path.unlink()
        for name in owned_names:
            if Path(name).name != name:
                continue
            path = path_directory / f"{name}.csv"
            if path.is_file():
                path.unlink()


def _write_plot(
    target: Path,
    summary: pd.DataFrame,
    curves: pd.DataFrame,
    *,
    base_cost_bps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    labels = {
        "original_lstm": "Direction LSTM",
        "hist_gradient_boosting": "HistGB classifier",
        "hybrid": "Classifier hybrid",
        "magnitude_lstm": "Magnitude LSTM",
        "ridge_return": "Ridge return",
        "histgb_return": "HistGB return",
        "equal_weight_market_proxy": "Equal-weight market proxy",
        "equal_weight_buy_and_hold_observed_horizons": "Observed-horizon buy & hold",
    }
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for name, group in curves.groupby("name", sort=False):
        width = 2.1 if name == "equal_weight_market_proxy" else 1.25
        axes[0].plot(
            pd.to_datetime(group["target_time"]),
            group["net_wealth"],
            label=labels.get(name, name),
            linewidth=width,
        )
    axes[0].axhline(1.0, color="black", linewidth=0.7)
    axes[0].set_title(f"Same-horizon net wealth ({base_cost_bps:g} bps one-way)")
    axes[0].set_ylabel("Wealth")
    axes[0].grid(alpha=0.25)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].set_xlabel("Target date (2026)")
    axes[0].legend(fontsize=7.2, ncol=2, loc="lower left")

    display = summary[summary["kind"].eq("model")]
    y = np.arange(len(display))
    axes[1].barh(
        y - 0.18,
        display["net_total_return"] * 100.0,
        height=0.36,
        label="Net return",
    )
    axes[1].barh(
        y + 0.18,
        display["net_relative_to_market_proxy"] * 100.0,
        height=0.36,
        label="Relative to market proxy",
    )
    axes[1].axvline(0.0, color="black", linewidth=0.7)
    axes[1].set_yticks(
        y, [labels.get(name, name) for name in display["name"]]
    )
    axes[1].set_xlabel("Test-period return (%)")
    axes[1].set_title("Model strategy economics")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(target / "strategy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_strategy_comparison(
    source_paths: Mapping[str, str | Path] = DEFAULT_SOURCE_PATHS,
    close_dir: str | Path = OUTPUT_DIR / "minute" / "close",
    out_dir: str | Path = OUTPUT_DIR / "lstm_strategy_comparison",
    *,
    probability_columns: Mapping[str, Mapping[str, str]] = DEFAULT_PROBABILITY_COLUMNS,
    signal_modes: Mapping[str, str] | None = None,
    score_thresholds: Mapping[str, float] | None = None,
    side: str = "long_short",
    weighting: str = "confidence",
    require_directional_argmax: bool = True,
    base_cost_bps: float = 5.0,
    expected_stock_count: int | None = 5,
    expected_day_count: int | None = 10,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the fixed-rule model comparison and persist auditable artifacts."""

    if len(source_paths) < 2:
        raise ValueError("at least two source paths are required")
    if not np.isfinite(base_cost_bps) or base_cost_bps < 0.0:
        raise ValueError("base_cost_bps must be finite and non-negative")
    modes = {name: "probability_gap" for name in source_paths}
    modes.update(dict(signal_modes or {}))
    thresholds = {name: 0.0 for name in source_paths}
    thresholds.update({key: float(value) for key, value in (score_thresholds or {}).items()})
    unknown_modes = sorted(set(modes).difference(source_paths))
    unknown_thresholds = sorted(set(thresholds).difference(source_paths))
    if unknown_modes or unknown_thresholds:
        raise ValueError(
            f"configuration names must match source paths: "
            f"unknown_modes={unknown_modes}, unknown_thresholds={unknown_thresholds}"
        )

    paths = {name: Path(path) for name, path in source_paths.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} prediction file not found: {path}")
    raw_sources = {name: pd.read_csv(path) for name, path in paths.items()}
    aligned, alignment_audit = align_prediction_sources(
        raw_sources,
        signal_modes=modes,
        probability_columns=probability_columns,
        expected_stock_count=expected_stock_count,
        expected_day_count=expected_day_count,
    )
    reference_name = alignment_audit["reference_source"]
    if modes[reference_name] != "probability_gap":
        probability_references = [
            name for name, mode in modes.items() if mode == "probability_gap"
        ]
        if not probability_references:
            raise ValueError("one probability source is required to attach closes")
        reference_name = probability_references[0]
    realised_reference = attach_realised_returns(aligned[reference_name], close_dir)
    realised_returns = realised_reference["realised_return"].to_numpy(dtype=np.float64)
    for frame in aligned.values():
        frame["realised_return"] = realised_returns

    target = Path(out_dir)
    _prepare_output_directory(
        target,
        overwrite,
        (*source_paths.keys(), *BENCHMARK_NAMES),
    )
    path_directory = target / "portfolio_paths"
    summary_rows: list[dict[str, Any]] = []
    paths_by_name: dict[str, pd.DataFrame] = {}

    for name, frame in aligned.items():
        weighted = generate_signal_weights(
            frame,
            signal_mode=modes[name],
            score_threshold=thresholds[name],
            side=side,
            weighting=weighting,
            require_directional_argmax=require_directional_argmax,
        )
        path = build_portfolio_path(weighted)
        active_intervals = weighted.groupby("_window_end_ts")["active_signal"].any()
        row = _summarise_path(
            name,
            "model",
            path,
            cost_bps=base_cost_bps,
            row_coverage=float(weighted["active_signal"].mean()),
            interval_coverage=float(active_intervals.mean()),
        )
        row["signal_mode"] = modes[name]
        row["score_threshold"] = thresholds[name]
        summary_rows.append(row)
        paths_by_name[name] = path

    market_path = build_equal_weight_market_proxy(realised_reference)
    buy_hold_path = build_equal_weight_buy_and_hold(realised_reference)
    for name, path in (
        ("equal_weight_market_proxy", market_path),
        ("equal_weight_buy_and_hold_observed_horizons", buy_hold_path),
    ):
        summary_rows.append(
            _summarise_path(
                name,
                "benchmark",
                path,
                cost_bps=base_cost_bps,
                row_coverage=1.0,
                interval_coverage=1.0,
            )
        )
        paths_by_name[name] = path

    summary = pd.DataFrame(summary_rows)
    market_return = float(
        summary.loc[
            summary["name"].eq("equal_weight_market_proxy"), "gross_total_return"
        ].iloc[0]
    )
    summary["gross_relative_to_market_proxy"] = summary["gross_total_return"].map(
        lambda value: _relative_return(float(value), market_return)
    )
    summary["net_relative_to_market_proxy"] = summary["net_total_return"].map(
        lambda value: _relative_return(float(value), market_return)
    )
    summary["gross_excess_return_percentage_points"] = (
        summary["gross_total_return"] - market_return
    )
    summary["net_excess_return_percentage_points"] = (
        summary["net_total_return"] - market_return
    )
    summary.to_csv(target / "strategy_comparison.csv", index=False, float_format="%.10f")

    curve_parts: list[pd.DataFrame] = []
    for name, path in paths_by_name.items():
        export = path.copy()
        export["net_return"] = (
            export["gross_return"]
            - base_cost_bps / 10_000.0 * export["turnover"]
        )
        export["gross_wealth"] = (1.0 + export["gross_return"]).cumprod()
        export["net_wealth"] = (1.0 + export["net_return"]).cumprod()
        export.insert(0, "name", name)
        export.to_csv(path_directory / f"{name}.csv", index=False, float_format="%.10f")
        curve_parts.append(export)
    curves = pd.concat(curve_parts, ignore_index=True)
    curves.to_csv(target / "wealth_curves.csv", index=False, float_format="%.10f")

    key_audit = realised_reference[
        ["stock", "window_end", "target_time", "realised_return"]
    ].copy()
    key_audit.to_csv(target / "aligned_sample_returns.csv", index=False, float_format="%.10f")

    report: dict[str, Any] = {
        "methodology": {
            "comparison_scope": "same five stocks, dates, stock-minute keys and realised returns",
            "rule": (
                "probability-gap models use P(up)-P(down), confidence-proportional "
                "unit-gross weights, and no trade when flat is argmax"
            ),
            "cost_bps_one_way": float(base_cost_bps),
            "benchmark_selection_role": "none",
            "benchmark_used_for_model_or_threshold_selection": False,
            "threshold_selection": (
                "pre-declared command-line values; defaults are zero and no result is tuned "
                "against test returns or the benchmark"
            ),
            "primary_relative_benchmark": "equal_weight_market_proxy gross return",
        },
        "alignment_audit": alignment_audit,
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "signal_mode": modes[name],
                "score_threshold": thresholds[name],
            }
            for name, path in paths.items()
        },
        "summary": summary.set_index("name").to_dict(orient="index"),
        "benchmark_definitions": {
            "equal_weight_market_proxy": (
                "Arithmetic mean of the five realised stock returns at every legal "
                "forecast horizon; evaluated only on model timestamps."
            ),
            "equal_weight_buy_and_hold_observed_horizons": (
                "Starts each stock at equal notional, never rebalances, and compounds "
                "only model-observed one-minute horizons; lunch/overnight/warm-up returns "
                "are excluded for identical horizon coverage."
            ),
        },
        "limitations": [
            "The comparison covers the same ten previously inspected test trading days, not a fresh blind sample.",
            "The long-short model portfolios and long-only market references have different exposure structures.",
            "Execution assumes zero latency; linear costs omit impact, fills and stock-borrow fees.",
            "The market proxy is a sample-five-stock reference, not a broad investable index.",
        ],
    }
    (target / "metrics.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _write_plot(target, summary, curves, base_cost_bps=base_cost_bps)
    return {
        "summary": summary,
        "curves": curves,
        "report": report,
        "out_dir": str(target),
    }
