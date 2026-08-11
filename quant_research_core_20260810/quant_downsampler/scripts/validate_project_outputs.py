"""Lightweight cross-task validation for the final project outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from qd.config import METRICS, OUTPUT_DIR, TOTAL_BARS
from qd.factors import REQUIRED_FACTOR_NAMES


def _shape(path: Path) -> tuple[int, int]:
    frame = pd.read_csv(path, index_col=0)
    return int(frame.shape[0]), int(frame.shape[1])


def main() -> None:
    checks: dict[str, object] = {}

    daily_shapes = {metric: _shape(OUTPUT_DIR / "daily" / f"{metric}.csv") for metric in METRICS}
    if set(daily_shapes.values()) != {(302, 300)}:
        raise AssertionError(f"daily shapes differ: {daily_shapes}")
    checks["task1_daily"] = {"status": "passed", "shapes": daily_shapes}

    minute_summary = {}
    for metric in METRICS:
        files = sorted((OUTPUT_DIR / "minute" / metric).glob("*.csv"))
        if len(files) != 302:
            raise AssertionError(f"{metric}: expected 302 minute files, got {len(files)}")
        sample_shapes = {_shape(files[0]), _shape(files[-1])}
        if sample_shapes != {(TOTAL_BARS, 300)}:
            raise AssertionError(f"{metric}: sample shapes differ: {sample_shapes}")
        minute_summary[metric] = {
            "files": len(files), "first": files[0].stem, "last": files[-1].stem,
            "sample_shape": [TOTAL_BARS, 300],
        }
    checks["task1_minute"] = {"status": "passed", "channels": minute_summary}

    factor_summary = pd.read_csv(OUTPUT_DIR / "evaluation" / "factor_summary.csv")
    required_factor_columns = {
        "factor", "IC", "IR", "ICIR", "rank_IC", "rank_IR", "rank_ICIR"
    }
    if len(factor_summary) != 10 or not required_factor_columns.issubset(factor_summary.columns):
        raise AssertionError("factor summary is incomplete")
    if not np.allclose(
        factor_summary["IR"], factor_summary["ICIR"] * np.sqrt(252.0), atol=2e-5
    ):
        raise AssertionError("IR is not annualized ICIR")
    checks["tasks2_3"] = {
        "status": "passed", "factor_count": int(len(factor_summary)),
        "ir_definition": "sqrt(252) * ICIR",
    }
    required_summary = pd.read_csv(
        OUTPUT_DIR / "evaluation" / "required_factor_summary.csv"
    )
    layer_daily = pd.read_csv(
        OUTPUT_DIR / "evaluation" / "factor_layer_daily_returns.csv"
    )
    layer_nav = pd.read_csv(OUTPUT_DIR / "evaluation" / "factor_layer_nav.csv")
    monotonicity = pd.read_csv(
        OUTPUT_DIR / "evaluation" / "factor_layer_monotonicity.csv"
    )
    layer_columns = {"factor", "signal_date", "Q1", "Q2", "Q3", "Q4", "Q5", "Q1_minus_Q5"}
    if (
        tuple(required_summary["factor"]) != REQUIRED_FACTOR_NAMES
        or set(monotonicity["factor"]) != set(factor_summary["factor"])
        or not layer_columns.issubset(layer_daily.columns)
        or not layer_columns.issubset(layer_nav.columns)
        or not (OUTPUT_DIR / "evaluation" / "required_factor_layering_nav.png").exists()
    ):
        raise AssertionError("prompt-required factor set or five-layer NAV outputs are incomplete")
    checks["tasks2_3"]["prompt_required_set"] = list(REQUIRED_FACTOR_NAMES)
    checks["tasks2_3"]["five_layer_nav"] = {
        "factor_count": int(monotonicity["factor"].nunique()),
        "groups": ["Q1", "Q2", "Q3", "Q4", "Q5"],
        "daily_rows": int(len(layer_daily)),
    }
    robustness_root = OUTPUT_DIR / "factor_robustness"
    factor_robustness = json.loads(
        (robustness_root / "summary.json").read_text(encoding="utf-8")
    )
    bootstrap = pd.read_csv(robustness_root / "bootstrap_ic.csv")
    regimes = pd.read_csv(robustness_root / "ic_by_market_regime.csv")
    single_factor = pd.read_csv(robustness_root / "single_factor_market_metrics.csv")
    if (
        factor_robustness.get("status") != "completed"
        or len(bootstrap) != 10
        or len(single_factor) != 10
        or set(regimes["regime_type"]) != {"target_market_direction", "known_market_trend"}
    ):
        raise AssertionError("factor robustness or market-regime outputs are incomplete")
    checks["tasks2_3"]["robustness"] = {
        "bootstrap_factor_count": int(len(bootstrap)),
        "latest_quarter": factor_robustness["latest_quarter"],
        "latest_quarter_positive_factor_count": factor_robustness[
            "latest_quarter_positive_factor_count"
        ],
        "single_factor_market_rows": int(len(single_factor)),
    }

    backtest = pd.read_csv(OUTPUT_DIR / "backtest_strict" / "metrics.csv")
    required_strategies = {
        f"{method}_{execution}"
        for method in ("rolling", "expanding", "hybrid", "adaptive", "naive")
        for execution in ("close", "open")
    }
    if set(backtest["strategy"]) != required_strategies:
        raise AssertionError("Task4 strategy set is incomplete")
    checks["task4"] = {
        "status": "passed", "strategy_count": int(len(backtest)),
        "best_strict_by_sharpe": str(
            backtest.loc[~backtest["strategy"].str.startswith("naive")]
            .sort_values("sharpe_ratio", ascending=False).iloc[0]["strategy"]
        ),
    }
    required_backtest_root = OUTPUT_DIR / "backtest_required"
    required_backtest = pd.read_csv(required_backtest_root / "metrics.csv")
    required_metadata = json.loads(
        (required_backtest_root / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        set(required_backtest["strategy"]) != required_strategies
        or required_metadata.get("factor_set") != "required"
        or tuple(required_metadata.get("factor_names", [])) != REQUIRED_FACTOR_NAMES
        or not required_backtest["factor_count"].eq(4).all()
    ):
        raise AssertionError("prompt-required four-factor Task4 backtest is incomplete")
    required_benchmarks = {}
    required_spreads = {}
    for strategy_name, execution in (("adaptive_close", "close"), ("adaptive_open", "open")):
        item = json.loads(
            (required_backtest_root / f"benchmark_metrics_{strategy_name}.json")
            .read_text(encoding="utf-8")
        )
        if item.get("strategy") != strategy_name or item.get("execution") != execution:
            raise AssertionError(f"mismatched Task4 benchmark for {strategy_name}")
        comparison = pd.read_csv(
            required_backtest_root
            / f"benchmark_comparison_{strategy_name}_daily.csv"
        )
        required_columns = {
            "strategy_cumulative_return", "benchmark_cumulative_return",
            "cumulative_return_spread", "geometric_excess_return",
        }
        if (
            "cumulative_return_spread" not in item["full_period"]
            or not required_columns.issubset(comparison.columns)
            or not (required_backtest_root / f"benchmark_comparison_{strategy_name}.png").exists()
        ):
            raise AssertionError(
                f"Task4 arithmetic/geometric benchmark paths are incomplete for {strategy_name}"
            )
        required_benchmarks[strategy_name] = item["full_period"]["geometric_excess_return"]
        required_spreads[strategy_name] = item["full_period"]["cumulative_return_spread"]
    checks["task4"]["prompt_required_four_factor_backtest"] = {
        "factor_names": list(REQUIRED_FACTOR_NAMES),
        "strategy_count": int(len(required_backtest)),
        "full_period_geometric_excess": required_benchmarks,
        "full_period_cumulative_return_spread": required_spreads,
    }
    benchmark = json.loads(
        (OUTPUT_DIR / "backtest_strict" / "benchmark_metrics_adaptive_close.json")
        .read_text(encoding="utf-8")
    )
    if benchmark.get("strategy") != "adaptive_close":
        raise AssertionError("Task4 benchmark analysis does not target adaptive_close")
    if benchmark["full_period"]["geometric_excess_return"] <= 0:
        raise AssertionError("adaptive_close no longer has positive full-period excess return")
    if benchmark["from_first_execution"]["geometric_excess_return"] <= 0:
        raise AssertionError("adaptive_close has no positive excess after its first execution")
    if "last_45_days" not in benchmark:
        raise AssertionError("Task4 recent relative analysis is missing")
    checks["task4"]["benchmark_analysis"] = {
        "full_excess_return": benchmark["full_period"]["geometric_excess_return"],
        "post_warmup_excess_return": benchmark["from_first_execution"][
            "geometric_excess_return"
        ],
        "last_45_day_excess_return": benchmark["last_45_days"]["geometric_excess_return"],
    }
    significance_root = OUTPUT_DIR / "task4_excess_significance"
    significance_summary = json.loads(
        (significance_root / "summary.json").read_text(encoding="utf-8")
    )
    significance = pd.read_csv(significance_root / "bootstrap_summary.csv")
    required_open_active = significance[
        significance["factor_set"].eq("required_4")
        & significance["strategy"].eq("adaptive_open")
        & significance["period"].eq("from_first_execution")
    ].iloc[0]
    required_close_active = significance[
        significance["factor_set"].eq("required_4")
        & significance["strategy"].eq("adaptive_close")
        & significance["period"].eq("from_first_execution")
    ].iloc[0]
    if (
        significance_summary.get("status") != "completed"
        or len(significance) != 12
        or required_open_active["geometric_excess_ci_low"] <= 0
        or required_open_active["one_sided_pvalue_mean_daily_excess_gt_zero"] >= 0.05
        or not (
            required_close_active["geometric_excess_ci_low"] < 0
            < required_close_active["geometric_excess_ci_high"]
        )
        or not (significance_root / "bootstrap_excess_ci.png").exists()
        or not (significance_root / "required_rolling_excess_and_beta.png").exists()
    ):
        raise AssertionError("Task4 excess significance or rolling-risk outputs are incomplete")
    checks["task4"]["excess_significance"] = {
        "method": significance_summary["method"],
        "rows": int(len(significance)),
        "required_open_post_warmup_geometric_excess": float(
            required_open_active["geometric_excess_return"]
        ),
        "required_open_post_warmup_ci": [
            float(required_open_active["geometric_excess_ci_low"]),
            float(required_open_active["geometric_excess_ci_high"]),
        ],
        "required_open_one_sided_pvalue": float(
            required_open_active["one_sided_pvalue_mean_daily_excess_gt_zero"]
        ),
        "required_close_ci_crosses_zero": True,
    }
    cost_stress = pd.read_csv(OUTPUT_DIR / "backtest_robustness" / "cost_stress.csv")
    if (
        len(cost_stress) != 8
        or set(cost_stress["selection_policy"]) != {"top_n", "buffered"}
        or set(cost_stress["sell_fee_bps"]) != {0.0, 5.0, 10.0, 20.0}
    ):
        raise AssertionError("Task4 turnover/cost robustness grid is incomplete")
    checks["task4"]["turnover_cost_robustness"] = {
        "rows": int(len(cost_stress)),
        "buffered_5bp_total_return": float(
            cost_stress.loc[
                cost_stress["selection_policy"].eq("buffered")
                & np.isclose(cost_stress["sell_fee_bps"], 5.0),
                "strategy_total_return",
            ].iloc[0]
        ),
    }
    independence_root = OUTPUT_DIR / "factor_independence"
    independence_summary = json.loads(
        (independence_root / "summary.json").read_text(encoding="utf-8")
    )
    independence_metrics = pd.read_csv(independence_root / "independence_metrics.csv")
    independence_backtest = pd.read_csv(independence_root / "backtest_metrics.csv")
    clusters = pd.read_csv(independence_root / "clusters.csv")
    required_factor_sets = {"raw_10", "cluster_pruned", "orthogonalized_10"}
    if (
        independence_summary.get("status") != "completed"
        or independence_summary.get("selection_status")
        != "frozen_before_post_calibration_backtest"
        or set(independence_metrics["factor_set"]) != required_factor_sets
        or set(independence_backtest["factor_set"]) != required_factor_sets
        or int(clusters["selected"].sum()) != int(independence_summary["cluster_count"])
    ):
        raise AssertionError("factor independence outputs are incomplete")
    independence_by_set = independence_metrics.set_index("factor_set")
    if not (
        independence_by_set.loc["cluster_pruned", "max_abs_off_diagonal_correlation"]
        < independence_by_set.loc["raw_10", "max_abs_off_diagonal_correlation"]
        and independence_by_set.loc["orthogonalized_10", "max_abs_off_diagonal_correlation"]
        < independence_by_set.loc["cluster_pruned", "max_abs_off_diagonal_correlation"]
    ):
        raise AssertionError("factor de-redundancy did not reduce post-freeze dependence")
    checks["task4"]["factor_independence"] = {
        "calibration_days": int(independence_summary["calibration_days"]),
        "cluster_count": int(independence_summary["cluster_count"]),
        "selected_factors": independence_summary["selected_factors"],
        "raw_max_abs_correlation": float(
            independence_by_set.loc["raw_10", "max_abs_off_diagonal_correlation"]
        ),
        "pruned_max_abs_correlation": float(
            independence_by_set.loc["cluster_pruned", "max_abs_off_diagonal_correlation"]
        ),
        "orthogonalized_max_abs_correlation": float(
            independence_by_set.loc["orthogonalized_10", "max_abs_off_diagonal_correlation"]
        ),
        "best_post_freeze_by_geometric_excess": independence_summary[
            "best_post_freeze_by_geometric_excess"
        ],
    }

    lstm_metrics = json.loads(
        (OUTPUT_DIR / "lstm_ensemble" / "test_metrics.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (OUTPUT_DIR / "lstm_ensemble" / "replay_validation.json").read_text(encoding="utf-8")
    )
    if replay.get("status") != "passed" or replay.get("rows_replayed") != 5900:
        raise AssertionError("LSTM strict replay has not passed")
    strategy_summary = json.loads(
        (OUTPUT_DIR / "lstm_ensemble" / "strategy_summary.json").read_text(encoding="utf-8")
    )
    strategy_metrics = pd.read_csv(OUTPUT_DIR / "lstm_ensemble" / "strategy_metrics.csv")
    price_time_proxy = strategy_metrics[
        (strategy_metrics["execution"] == "next_minute_open_to_close")
        & (strategy_metrics["position_type"] == "long")
    ]
    expected_pairs = {
        (strategy, fee)
        for strategy in ("all_up", "balanced_up", "strict_up")
        for fee in (0.0, 5.0)
    }
    actual_pairs = set(zip(price_time_proxy["strategy"], price_time_proxy["sell_fee_bps"]))
    if strategy_summary.get("status") != "completed" or actual_pairs != expected_pairs:
        raise AssertionError("LSTM execution-aware strategy analysis is incomplete")
    t1_metrics = pd.read_csv(OUTPUT_DIR / "lstm_ensemble" / "t1_strategy_metrics.csv")
    if set(zip(t1_metrics["strategy"], t1_metrics["sell_fee_bps"])) != expected_pairs:
        raise AssertionError("LSTM T+1 strategy analysis is incomplete")
    if not t1_metrics["settled_signal_days"].eq(9).all():
        raise AssertionError("LSTM T+1 analysis does not have the expected nine settled days")
    t1_path = OUTPUT_DIR / "lstm_ensemble" / "t1_strategy_comparison_daily.csv"
    t1_plot = OUTPUT_DIR / "lstm_ensemble" / "t1_strategy_nav_and_return_gap.png"
    t1_comparison = pd.read_csv(t1_path)
    if (
        "cumulative_return_gap_vs_exposure_matched_market" not in t1_metrics.columns
        or not any("return_gap_vs_matched_market" in column for column in t1_comparison.columns)
        or not t1_plot.exists()
    ):
        raise AssertionError("LSTM T+1 market-subtraction path or plot is missing")
    checks["task5"] = {
        "status": "passed", "test_windows": 5900,
        "accuracy": lstm_metrics["accuracy"],
        "balanced_accuracy": lstm_metrics["balanced_accuracy"],
        "macro_f1": lstm_metrics["macro_f1"],
        "strict_replay": True,
        "strategy_analysis": {
            "timing": "next_minute_open_to_close",
            "five_stock_market_total_return": strategy_summary["five_stock_market_total_return"],
            "strict_up_net_return_5bp": float(
                price_time_proxy.loc[
                    (price_time_proxy["strategy"] == "strict_up")
                    & np.isclose(price_time_proxy["sell_fee_bps"], 5.0),
                    "net_total_return",
                ].iloc[0]
            ),
            "balanced_up_t1_excess_vs_matched_market_5bp": float(
                t1_metrics.loc[
                    (t1_metrics["strategy"] == "balanced_up")
                    & np.isclose(t1_metrics["sell_fee_bps"], 5.0),
                    "excess_vs_exposure_matched_market",
                ].iloc[0]
            ),
        },
    }
    baseline_root = OUTPUT_DIR / "lstm_baselines"
    frozen = json.loads(
        (baseline_root / "selection_frozen_before_test.json").read_text(encoding="utf-8")
    )
    baseline_metrics = pd.read_csv(baseline_root / "test_metrics.csv")
    strategy_comparison = pd.read_csv(baseline_root / "strategy_comparison.csv")
    required_models = {
        "hist_gradient_boosting", "linear_logistic_sgd", "last_observed_move",
        "lstm_ensemble_raw", "lstm_ensemble_temperature_scaled", "majority_flat",
    }
    if frozen.get("status") != "frozen_before_test" or set(baseline_metrics["model"]) != required_models:
        raise AssertionError("Task5 baseline selection or probability metrics are incomplete")
    if len(strategy_comparison) != 18 or set(strategy_comparison["model"]) != {
        "lstm_ensemble", "hist_gradient_boosting", "linear_logistic_sgd"
    }:
        raise AssertionError("Task5 same-key strategy comparison is incomplete")
    raw_ece = float(
        baseline_metrics.loc[baseline_metrics["model"].eq("lstm_ensemble_raw"), "top_label_ece"].iloc[0]
    )
    calibrated_ece = float(
        baseline_metrics.loc[
            baseline_metrics["model"].eq("lstm_ensemble_temperature_scaled"), "top_label_ece"
        ].iloc[0]
    )
    if calibrated_ece >= raw_ece:
        raise AssertionError("validation temperature scaling did not improve test ECE")
    checks["task5"]["strong_baselines_and_calibration"] = {
        "models": sorted(required_models),
        "raw_lstm_ece": raw_ece,
        "temperature_scaled_lstm_ece": calibrated_ece,
        "same_key_strategy_rows": int(len(strategy_comparison)),
    }

    magnitude_root = OUTPUT_DIR / "lstm_magnitude"
    magnitude = json.loads((magnitude_root / "metrics.json").read_text(encoding="utf-8"))
    magnitude_predictions = pd.read_csv(magnitude_root / "test_predictions.csv")
    magnitude_strategy = pd.read_csv(magnitude_root / "strategy_comparison.csv")
    official_predictions = pd.read_csv(OUTPUT_DIR / "lstm_ensemble" / "test_predictions.csv")
    magnitude_keys = ["stock", "window_end", "target_time", "true_label"]
    if not magnitude_predictions[magnitude_keys].equals(official_predictions[magnitude_keys]):
        raise AssertionError("magnitude-aware LSTM test keys do not match the official LSTM")
    if (
        len(magnitude_predictions) != 5900
        or len(magnitude_strategy) != 9
        or set(magnitude_strategy["method"]) != {
            "official_direction_classifier",
            "multitask_direction_classifier",
            "multitask_predicted_return",
        }
        or set(magnitude_strategy["tier"]) != {"all", "balanced", "strict"}
    ):
        raise AssertionError("magnitude-aware LSTM outputs are incomplete")
    return_ic = float(magnitude["test_metrics"]["return"]["spearman_ic"])
    if not np.isfinite(return_ic):
        raise AssertionError("magnitude-aware LSTM return IC is not finite")
    checks["task5"]["magnitude_aware_experiment"] = {
        "test_rows": int(len(magnitude_predictions)),
        "test_return_spearman_ic": return_ic,
        "strategy_rows": int(len(magnitude_strategy)),
        "same_test_keys_as_official_lstm": True,
    }

    tradable_root = OUTPUT_DIR / "tradable_return_research"
    tradable_freeze = json.loads(
        (tradable_root / "selection_frozen_before_test.json").read_text(encoding="utf-8")
    )
    tradable_metrics = pd.read_csv(tradable_root / "test_return_metrics.csv")
    tradable_strategy = pd.read_csv(tradable_root / "test_strategy_metrics.csv")
    required_horizons = {
        "open_to_close_1m", "open_to_close_5m", "open_to_close_15m",
        "open_to_close_30m", "t1_same_minute_open",
    }
    required_return_models = {"ridge", "hist_gradient_boosting_regressor"}
    if (
        tradable_freeze.get("status") != "frozen_before_test"
        or len(tradable_metrics) != 10
        or len(tradable_strategy) != 10
        or set(tradable_metrics["horizon"]) != required_horizons
        or set(tradable_metrics["model"]) != required_return_models
        or set(tradable_strategy["horizon"]) != required_horizons
    ):
        raise AssertionError("multi-horizon executable-return research is incomplete")
    tradable_lstm_root = OUTPUT_DIR / "tradable_lstm"
    tradable_lstm = json.loads(
        (tradable_lstm_root / "summary.json").read_text(encoding="utf-8")
    )
    tradable_lstm_predictions = pd.read_csv(
        tradable_lstm_root / "test_predictions.csv"
    )
    tradable_lstm_comparison = pd.read_csv(
        tradable_lstm_root / "strategy_comparison.csv"
    )
    if (
        tradable_lstm.get("status") != "completed"
        or tradable_lstm.get("horizon") != "t1_same_minute_open"
        or len(tradable_lstm_predictions) != 5310
        or set(tradable_lstm_comparison["model"]) != {
            "ridge", "hist_gradient_boosting_regressor", "compact_multitask_lstm"
        }
    ):
        raise AssertionError("validation-screened T+1 LSTM comparison is incomplete")
    ridge_t1 = tradable_strategy[
        tradable_strategy["horizon"].eq("t1_same_minute_open")
        & tradable_strategy["model"].eq("ridge")
    ].iloc[0]
    checks["task5"]["tradable_return_horizons"] = {
        "horizons": sorted(required_horizons),
        "models": sorted(required_return_models),
        "test_metric_rows": int(len(tradable_metrics)),
        "ridge_t1_excess_vs_matched_market": float(ridge_t1["excess_vs_matched_market"]),
        "compact_t1_lstm_return_spearman_ic": float(
            tradable_lstm["test_return_metrics"]["spearman_ic"]
        ),
        "compact_t1_lstm_excess_vs_matched_market": float(
            tradable_lstm["test_strategy"]["excess_vs_matched_market"]
        ),
    }

    excess_target_root = OUTPUT_DIR / "t1_excess_return_research"
    excess_target_freeze = json.loads(
        (excess_target_root / "selection_frozen_before_test.json").read_text(
            encoding="utf-8"
        )
    )
    excess_target_summary = json.loads(
        (excess_target_root / "summary.json").read_text(encoding="utf-8")
    )
    excess_target_metrics = pd.read_csv(excess_target_root / "test_target_metrics.csv")
    excess_target_strategy = pd.read_csv(excess_target_root / "test_strategy_metrics.csv")
    expected_targets = {"raw_return", "market_excess_return"}
    if (
        excess_target_freeze.get("status") != "frozen_before_test"
        or excess_target_summary.get("status") != "completed"
        or excess_target_summary.get("test_rows") != 5310
        or len(excess_target_metrics) != 4
        or len(excess_target_strategy) != 4
        or set(excess_target_metrics["target"]) != expected_targets
        or set(excess_target_metrics["model"]) != required_return_models
        or set(excess_target_strategy["target"]) != expected_targets
        or not excess_target_metrics["test_rows"].eq(5310).all()
        or not (excess_target_root / "target_and_strategy_comparison.png").exists()
    ):
        raise AssertionError("direct T+1 market-excess target research is incomplete")
    raw_ridge = excess_target_strategy[
        excess_target_strategy["target"].eq("raw_return")
        & excess_target_strategy["model"].eq("ridge")
    ].iloc[0]
    excess_hist = excess_target_strategy[
        excess_target_strategy["target"].eq("market_excess_return")
        & excess_target_strategy["model"].eq("hist_gradient_boosting_regressor")
    ].iloc[0]
    checks["task5"]["direct_market_excess_target"] = {
        "selection_frozen_before_test": True,
        "test_rows": int(excess_target_summary["test_rows"]),
        "target_metric_rows": int(len(excess_target_metrics)),
        "strategy_rows": int(len(excess_target_strategy)),
        "raw_ridge_geometric_excess": float(raw_ridge["excess_vs_matched_market"]),
        "market_excess_histgb_geometric_excess": float(
            excess_hist["excess_vs_matched_market"]
        ),
        "market_excess_target_spearman_ic": {
            str(row["model"]): float(row["spearman_ic"])
            for _, row in excess_target_metrics[
                excess_target_metrics["target"].eq("market_excess_return")
            ].iterrows()
        },
    }

    feature_root = OUTPUT_DIR / "lstm_feature_independence"
    feature_summary = json.loads(
        (feature_root / "feature_independence_summary.json").read_text(encoding="utf-8")
    )
    pruned_lstm_summary = json.loads(
        (feature_root / "summary.json").read_text(encoding="utf-8")
    )
    feature_metrics = pd.read_csv(feature_root / "feature_independence_metrics.csv")
    feature_clusters = pd.read_csv(feature_root / "feature_clusters.csv")
    full_vs_pruned = pd.read_csv(feature_root / "full_vs_pruned_lstm.csv")
    component_ablation = pd.read_csv(feature_root / "component_ablation_metrics.csv")
    if (
        feature_summary.get("status") != "completed"
        or feature_summary.get("original_feature_count") != 25
        or feature_summary.get("selected_feature_count") != 19
        or pruned_lstm_summary.get("status") != "completed"
        or pruned_lstm_summary.get("input_feature_count") != 19
        or int(feature_clusters["selected"].sum()) != 19
        or set(feature_metrics["feature_set"]) != {"all_25", "cluster_representatives"}
        or set(full_vs_pruned["model"]) != {
            "compact_multitask_lstm_all_25", "cluster_pruned_multitask_lstm_19"
        }
        or set(component_ablation["split"]) != {"validation", "test"}
        or set(component_ablation["probability_source"]) != {"joint", "staged", "blend"}
    ):
        raise AssertionError("Task5 feature/component independence outputs are incomplete")
    feature_by_set = feature_metrics.set_index("feature_set")
    if not (
        feature_by_set.loc["cluster_representatives", "max_abs_off_diagonal_correlation"]
        < feature_by_set.loc["all_25", "max_abs_off_diagonal_correlation"]
    ):
        raise AssertionError("Task5 feature pruning did not reduce maximum dependence")
    checks["task5"]["feature_and_component_independence"] = {
        "selection_split": feature_summary["feature_selection"]["selection_split"],
        "original_feature_count": 25,
        "selected_feature_count": 19,
        "all_feature_max_abs_correlation": float(
            feature_by_set.loc["all_25", "max_abs_off_diagonal_correlation"]
        ),
        "selected_feature_max_abs_correlation": float(
            feature_by_set.loc["cluster_representatives", "max_abs_off_diagonal_correlation"]
        ),
        "pruned_lstm_test_accuracy": float(
            pruned_lstm_summary["test_classification_metrics"]["accuracy"]
        ),
        "pruned_lstm_return_spearman_ic": float(
            pruned_lstm_summary["test_return_metrics"]["spearman_ic"]
        ),
        "pruned_lstm_excess_vs_matched_market": float(
            pruned_lstm_summary["test_strategy"]["excess_vs_matched_market"]
        ),
        "validation_joint_staged_agreement": float(
            feature_summary["component_complementarity"]["validation"][
                "joint_staged_prediction_agreement"
            ]
        ),
    }

    minimal_root = OUTPUT_DIR / "lstm_minimal_four"
    minimal_summary = json.loads(
        (minimal_root / "minimal_four_summary.json").read_text(encoding="utf-8")
    )
    minimal_comparison = pd.read_csv(minimal_root / "model_comparison.csv")
    if (
        minimal_summary.get("status") != "completed"
        or minimal_summary.get("feature_count") != 4
        or set(minimal_summary.get("prompt_field_to_feature", {}))
        != {"Time", "Price", "Volume", "BSFlag"}
        or set(minimal_comparison["model"])
        != {"all_25", "cluster_pruned_19", "prompt_raw_four"}
    ):
        raise AssertionError("Task5 prompt-four baseline outputs are incomplete")
    checks["task5"]["prompt_raw_four_baseline"] = {
        "field_mapping": minimal_summary["prompt_field_to_feature"],
        "test_accuracy": float(minimal_summary["test_classification_metrics"]["accuracy"]),
        "test_macro_f1": float(minimal_summary["test_classification_metrics"]["macro_f1"]),
        "return_spearman_ic": float(minimal_summary["test_return_metrics"]["spearman_ic"]),
        "excess_vs_matched_market": float(
            minimal_summary["test_strategy"]["excess_vs_matched_market"]
        ),
    }

    report = {"status": "passed", "checks": checks}
    (OUTPUT_DIR / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
