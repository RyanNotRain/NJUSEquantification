"""Leakage-safe strategy evaluation for saved minute LSTM probabilities.

The position for ``window_end -> target_time`` is a pure function of the
three probabilities available at ``window_end``.  Realised prices and labels
are attached only after target weights have been fixed and are never used to
select observations or determine positions.

The backtest is deliberately a research bridge rather than an execution
simulator: it assumes a position can be established at the window-end close
and unwound/rebalanced at subsequent minute closes.  Transaction costs are
one-way costs applied to absolute portfolio weight changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PROBABILITY_COLUMNS = ("prob_down", "prob_flat", "prob_up")
TIERS = ("all", "balanced", "strict")
WEIGHTINGS = ("equal", "confidence")
SIDES = ("long_short", "long_only")


def _required_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _parse_boolean(series: pd.Series, name: str) -> pd.Series:
    """Parse booleans without treating the string ``'False'`` as truthy."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        values = pd.to_numeric(series, errors="coerce")
        if values.isna().any() or not values.isin((0, 1)).all():
            raise ValueError(f"{name} must contain only booleans or 0/1")
        return values.astype(bool)
    normalised = series.astype("string").str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    parsed = normalised.map(mapping)
    if parsed.isna().any():
        bad = sorted(normalised[parsed.isna()].dropna().unique().tolist())
        raise ValueError(f"{name} contains invalid boolean values: {bad}")
    return parsed.astype(bool)


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy with parsed timestamps and model confidence."""
    _required_columns(
        predictions,
        ("stock", "window_end", "target_time", *PROBABILITY_COLUMNS),
    )
    frame = predictions.copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("prediction table is empty")
    if frame[["stock", "window_end"]].duplicated().any():
        raise ValueError("prediction table contains duplicate stock/window_end rows")
    frame["stock"] = frame["stock"].astype(str)
    frame["_window_end_ts"] = pd.to_datetime(frame["window_end"], errors="coerce")
    frame["_target_time_ts"] = pd.to_datetime(frame["target_time"], errors="coerce")
    if frame[["_window_end_ts", "_target_time_ts"]].isna().any().any():
        raise ValueError("window_end or target_time contains an invalid timestamp")
    if not (frame["_target_time_ts"] > frame["_window_end_ts"]).all():
        raise ValueError("every target_time must be later than window_end")

    probability = frame.loc[:, PROBABILITY_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    values = probability.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("probability columns must be finite")
    if (values < -1e-8).any() or (values > 1.0 + 1e-8).any():
        raise ValueError("probabilities must lie within [0, 1]")
    row_sum = values.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-4, rtol=0.0):
        raise ValueError("down/flat/up probabilities must sum to one")
    frame.loc[:, PROBABILITY_COLUMNS] = values
    frame["model_confidence"] = values.max(axis=1)
    return frame


def attach_realised_returns(
    predictions: pd.DataFrame,
    close_dir: str | Path,
    *,
    return_column: str = "realised_return",
) -> pd.DataFrame:
    """Attach exact window-end-to-target simple returns from minute closes.

    Files are expected at ``close_dir/YYYYMMDD.csv`` with a ``datetime``
    index and stock-code columns, matching the project's minute output.
    """
    frame = validate_predictions(predictions)
    root = Path(close_dir)
    realised = pd.Series(np.nan, index=frame.index, dtype=np.float64)

    day_key = frame["_window_end_ts"].dt.strftime("%Y%m%d")
    for day, row_index in day_key.groupby(day_key).groups.items():
        subset = frame.loc[row_index]
        if not (
            subset["_window_end_ts"].dt.strftime("%Y%m%d").eq(day).all()
            and subset["_target_time_ts"].dt.strftime("%Y%m%d").eq(day).all()
        ):
            raise ValueError("overnight targets are not supported by daily close files")
        path = root / f"{day}.csv"
        if not path.exists():
            raise FileNotFoundError(f"minute close table not found: {path}")
        stocks = sorted(subset["stock"].unique().tolist())
        table = pd.read_csv(
            path,
            index_col=0,
            usecols=lambda column: column == "datetime" or column in stocks,
        )
        table.index = pd.to_datetime(table.index, errors="coerce")
        if table.index.isna().any() or table.index.duplicated().any():
            raise ValueError(f"{path} has an invalid or duplicate datetime index")
        missing_stocks = sorted(set(stocks).difference(table.columns))
        if missing_stocks:
            raise ValueError(f"{path} is missing stock columns: {missing_stocks}")

        current_values: list[float] = []
        target_values: list[float] = []
        for _, row in subset.iterrows():
            try:
                current_values.append(
                    float(table.at[row["_window_end_ts"], row["stock"]])
                )
                target_values.append(
                    float(table.at[row["_target_time_ts"], row["stock"]])
                )
            except KeyError as exc:
                raise ValueError(
                    f"{path} cannot align {row['stock']} at the requested timestamps"
                ) from exc
        current = np.asarray(current_values, dtype=np.float64)
        following = np.asarray(target_values, dtype=np.float64)
        if (
            not np.isfinite(current).all()
            or not np.isfinite(following).all()
            or (current <= 0.0).any()
        ):
            raise ValueError(f"{path} contains an invalid close used by the strategy")
        realised.loc[row_index] = following / current - 1.0

    if realised.isna().any():
        raise RuntimeError("some prediction rows did not receive a realised return")
    frame[return_column] = realised
    return frame


def label_alignment(predictions: pd.DataFrame) -> dict[str, float | int] | None:
    """Audit saved class labels against attached returns; never affects trades."""
    if "true_label" not in predictions or "realised_return" not in predictions:
        return None
    returns = pd.to_numeric(predictions["realised_return"], errors="coerce").to_numpy()
    labels = pd.to_numeric(predictions["true_label"], errors="coerce").to_numpy()
    if not np.isfinite(returns).all() or not np.isfinite(labels).all():
        raise ValueError("true_label/realised_return audit columns must be finite")
    return_labels = np.where(returns < 0.0, 0, np.where(returns > 0.0, 2, 1))
    matches = return_labels == labels.astype(np.int64)
    return {"n": int(len(matches)), "agreement": float(matches.mean())}


def _tier_mask(
    frame: pd.DataFrame,
    tier: str,
    confidence_threshold: float | None,
) -> pd.Series:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    if tier == "all":
        return pd.Series(True, index=frame.index)
    column = f"selected_{tier}"
    if column in frame:
        return _parse_boolean(frame[column], column)
    if confidence_threshold is None:
        raise ValueError(
            f"{column} is absent; supply a frozen validation confidence threshold"
        )
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must lie within [0, 1]")
    return frame["model_confidence"] >= float(confidence_threshold)


def generate_target_weights(
    predictions: pd.DataFrame,
    *,
    tier: str = "all",
    weighting: str = "equal",
    score_threshold: float = 0.0,
    confidence_threshold: float | None = None,
    require_directional_argmax: bool = True,
    side: str = "long_short",
) -> pd.DataFrame:
    """Create cross-sectional target weights using prediction-time data only.

    ``signal_score`` is ``P(up) - P(down)``.  Equal weighting assigns equal
    absolute weight to active names; confidence weighting uses the absolute
    directional probability gap.  Both are normalised to unit gross exposure
    at each prediction timestamp.  By default, a flat argmax means no trade.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    if not np.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must lie within [0, 1]")

    frame = validate_predictions(predictions)
    tier_selected = _tier_mask(frame, tier, confidence_threshold)
    score = frame["prob_up"] - frame["prob_down"]
    active = tier_selected & score.abs().ge(float(score_threshold)) & score.ne(0.0)
    if require_directional_argmax:
        active &= frame[["prob_up", "prob_down"]].max(axis=1) > frame["prob_flat"]
    if side == "long_only":
        active &= score > 0.0

    direction = np.sign(score.to_numpy(dtype=np.float64))
    if weighting == "equal":
        raw = direction
    else:
        raw = score.to_numpy(dtype=np.float64)
    raw = np.where(active.to_numpy(), raw, 0.0)
    frame["signal_score"] = score
    frame["tier_selected"] = tier_selected
    frame["active_signal"] = active
    frame["raw_weight"] = raw
    gross = frame.groupby("_window_end_ts", sort=False)["raw_weight"].transform(
        lambda values: values.abs().sum()
    )
    frame["target_weight"] = np.divide(
        raw,
        gross.to_numpy(dtype=np.float64),
        out=np.zeros(len(frame), dtype=np.float64),
        where=gross.to_numpy(dtype=np.float64) > 0.0,
    )
    frame["tier"] = tier
    frame["weighting"] = weighting
    frame["side"] = side
    frame["require_directional_argmax"] = bool(require_directional_argmax)
    frame["score_threshold"] = float(score_threshold)
    return frame.sort_values(["_window_end_ts", "stock"]).reset_index(drop=True)


def build_portfolio_path(
    weighted_predictions: pd.DataFrame,
    *,
    return_column: str = "realised_return",
) -> pd.DataFrame:
    """Aggregate stock weights/returns and calculate executable turnover.

    A continuous pair of forecast intervals is rebalanced directly.  At a
    lunch/overnight/warm-up gap the old portfolio is closed and the new one is
    entered.  The final portfolio is also closed, so reported turnover includes
    every entry and exit.
    """
    _required_columns(
        weighted_predictions,
        ("stock", "window_end", "target_time", "target_weight", return_column),
    )
    frame = weighted_predictions.copy()
    if "_window_end_ts" not in frame or "_target_time_ts" not in frame:
        frame["_window_end_ts"] = pd.to_datetime(frame["window_end"], errors="coerce")
        frame["_target_time_ts"] = pd.to_datetime(frame["target_time"], errors="coerce")
    if frame[["_window_end_ts", "_target_time_ts"]].isna().any().any():
        raise ValueError("portfolio timestamps are invalid")
    if frame[["stock", "_window_end_ts"]].duplicated().any():
        raise ValueError("weighted predictions contain duplicate stock/timestamp rows")
    weights = pd.to_numeric(frame["target_weight"], errors="coerce")
    returns = pd.to_numeric(frame[return_column], errors="coerce")
    if not np.isfinite(weights).all() or not np.isfinite(returns).all():
        raise ValueError("target weights and realised returns must be finite")
    frame["target_weight"] = weights
    frame[return_column] = returns
    stocks = sorted(frame["stock"].astype(str).unique().tolist())
    stock_location = {stock: index for index, stock in enumerate(stocks)}

    records: list[dict[str, object]] = []
    weight_vectors: list[np.ndarray] = []
    for window_end, group in frame.groupby("_window_end_ts", sort=True):
        targets = group["_target_time_ts"].unique()
        if len(targets) != 1:
            raise ValueError("stocks at one window_end have inconsistent target times")
        vector = np.zeros(len(stocks), dtype=np.float64)
        for row in group.itertuples():
            vector[stock_location[str(row.stock)]] = float(row.target_weight)
        gross_return = float(
            np.dot(
                group["target_weight"].to_numpy(dtype=np.float64),
                group[return_column].to_numpy(dtype=np.float64),
            )
        )
        weight_vectors.append(vector)
        records.append({
            "window_end": pd.Timestamp(window_end),
            "target_time": pd.Timestamp(targets[0]),
            "gross_return": gross_return,
            "gross_exposure": float(np.abs(vector).sum()),
            "net_exposure": float(vector.sum()),
            "active_names": int(np.count_nonzero(vector)),
        })
    if not records:
        raise ValueError("weighted prediction table is empty")

    turnover = np.zeros(len(records), dtype=np.float64)
    turnover[0] += np.abs(weight_vectors[0]).sum()
    for index in range(1, len(records)):
        previous = records[index - 1]
        current = records[index]
        if previous["target_time"] == current["window_end"]:
            turnover[index] += np.abs(
                weight_vectors[index] - weight_vectors[index - 1]
            ).sum()
        else:
            turnover[index - 1] += np.abs(weight_vectors[index - 1]).sum()
            turnover[index] += np.abs(weight_vectors[index]).sum()
    turnover[-1] += np.abs(weight_vectors[-1]).sum()

    path = pd.DataFrame(records)
    path["turnover"] = turnover
    path["date"] = path["window_end"].dt.strftime("%Y-%m-%d")
    return path


def _total_return(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if not np.isfinite(values).all() or (values <= -1.0).any():
        raise ValueError("strategy returns must be finite and greater than -100%")
    return float(np.expm1(np.log1p(values).sum()))


def _annualised_sharpe(returns: np.ndarray, periods_per_day: float) -> float:
    values = np.asarray(returns, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(values.mean() / std * np.sqrt(periods_per_day * 252.0))


def _max_drawdown(returns: np.ndarray) -> float:
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + np.asarray(returns))))
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def break_even_cost_bps(portfolio_path: pd.DataFrame) -> float:
    """Return the one-way cost in bps that reduces terminal return to zero."""
    _required_columns(portfolio_path, ("gross_return", "turnover"))
    gross = portfolio_path["gross_return"].to_numpy(dtype=np.float64)
    turnover = portfolio_path["turnover"].to_numpy(dtype=np.float64)
    if not np.isfinite(gross).all() or not np.isfinite(turnover).all():
        raise ValueError("gross returns and turnover must be finite")
    if (turnover < 0.0).any():
        raise ValueError("turnover cannot be negative")
    if _total_return(gross) <= 0.0:
        return 0.0
    positive = turnover > 0.0
    if not positive.any():
        return float("inf")

    # This upper bound keeps every per-period net return just above -100%.
    upper = float(np.min((1.0 + gross[positive]) / turnover[positive]) * (1.0 - 1e-12))

    def log_wealth(cost_rate: float) -> float:
        net = gross - cost_rate * turnover
        return float(np.log1p(net).sum())

    if log_wealth(upper) > 0.0:
        return float("inf")
    lower = 0.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if log_wealth(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return float((lower + upper) / 2.0 * 10_000.0)


def strategy_statistics(
    portfolio_path: pd.DataFrame,
    *,
    cost_bps: float = 5.0,
) -> dict[str, float | int]:
    """Summarise gross and transaction-cost-adjusted performance."""
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    gross = portfolio_path["gross_return"].to_numpy(dtype=np.float64)
    turnover = portfolio_path["turnover"].to_numpy(dtype=np.float64)
    net = gross - float(cost_bps) / 10_000.0 * turnover
    if (net <= -1.0).any():
        raise ValueError("cost assumption produces a per-period loss of at least 100%")
    number_days = int(portfolio_path["date"].nunique())
    periods_per_day = len(portfolio_path) / number_days
    daily_turnover = portfolio_path.groupby("date")["turnover"].sum()
    daily_gross = portfolio_path.groupby("date")["gross_return"].apply(
        lambda values: _total_return(values.to_numpy(dtype=np.float64))
    )
    daily_net_frame = portfolio_path.assign(_net_return=net)
    daily_net = daily_net_frame.groupby("date")["_net_return"].apply(
        lambda values: _total_return(values.to_numpy(dtype=np.float64))
    )
    return {
        "n_intervals": int(len(portfolio_path)),
        "n_days": number_days,
        "cost_bps": float(cost_bps),
        "gross_total_return": _total_return(gross),
        "net_total_return": _total_return(net),
        "gross_intraday_annualised_sharpe": _annualised_sharpe(
            gross, periods_per_day
        ),
        "net_intraday_annualised_sharpe": _annualised_sharpe(net, periods_per_day),
        "gross_daily_annualised_sharpe": _annualised_sharpe(
            daily_gross.to_numpy(dtype=np.float64), 1.0
        ),
        "net_daily_annualised_sharpe": _annualised_sharpe(
            daily_net.to_numpy(dtype=np.float64), 1.0
        ),
        "gross_max_drawdown": _max_drawdown(gross),
        "net_max_drawdown": _max_drawdown(net),
        "total_turnover": float(turnover.sum()),
        "average_daily_turnover": float(daily_turnover.mean()),
        "mean_gross_exposure": float(portfolio_path["gross_exposure"].mean()),
        "mean_absolute_net_exposure": float(portfolio_path["net_exposure"].abs().mean()),
        "active_interval_rate": float((portfolio_path["gross_exposure"] > 0.0).mean()),
        "mean_active_names": float(portfolio_path["active_names"].mean()),
        "gross_positive_day_rate": float((daily_gross > 0.0).mean()),
        "net_positive_day_rate": float((daily_net > 0.0).mean()),
        "arithmetic_transaction_cost": float(turnover.sum() * cost_bps / 10_000.0),
    }


def cost_sensitivity(
    portfolio_path: pd.DataFrame,
    cost_grid_bps: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
) -> pd.DataFrame:
    """Evaluate an unchanged position path across one-way cost assumptions."""
    if not cost_grid_bps:
        raise ValueError("cost_grid_bps must not be empty")
    rows = [strategy_statistics(portfolio_path, cost_bps=float(cost)) for cost in cost_grid_bps]
    return pd.DataFrame(rows)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _prepare_output_directory(target: Path, overwrite: bool) -> None:
    """Refuse accidental replacement and remove only known generated files."""
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty strategy output directory: {target}; "
            "choose another out_dir or pass overwrite=True"
        )
    target.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for filename in ("strategy_summary.csv", "cost_sensitivity.csv", "metrics.json", "README.md"):
        path = target / filename
        if path.is_file():
            path.unlink()
    for directory_name in ("positions", "portfolio_paths"):
        directory = target / directory_name
        if directory.is_dir():
            for path in directory.glob("*.csv"):
                if path.is_file():
                    path.unlink()


def run_strategy_suite(
    predictions_path: str | Path,
    close_dir: str | Path,
    out_dir: str | Path,
    *,
    tiers: Sequence[str] = TIERS,
    weightings: Sequence[str] = WEIGHTINGS,
    score_threshold: float = 0.0,
    confidence_thresholds: Mapping[str, float] | None = None,
    require_directional_argmax: bool = True,
    side: str = "long_short",
    base_cost_bps: float = 5.0,
    cost_grid_bps: Sequence[float] = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0),
    overwrite: bool = False,
) -> dict[str, object]:
    """Run all requested tier/weighting combinations and persist audit tables."""
    source = Path(predictions_path)
    target = Path(out_dir)
    _prepare_output_directory(target, overwrite)
    positions_dir = target / "positions"
    paths_dir = target / "portfolio_paths"
    positions_dir.mkdir(exist_ok=True)
    paths_dir.mkdir(exist_ok=True)

    predictions = pd.read_csv(source)
    realised = attach_realised_returns(predictions, close_dir)
    alignment = label_alignment(realised)
    thresholds = dict(confidence_thresholds or {})
    summary_rows: list[dict[str, object]] = []
    sensitivity_parts: list[pd.DataFrame] = []
    details: dict[str, object] = {}

    for tier in tiers:
        for weighting in weightings:
            slug = f"{tier}_{weighting}_{side}"
            weighted = generate_target_weights(
                realised,
                tier=tier,
                weighting=weighting,
                score_threshold=score_threshold,
                confidence_threshold=thresholds.get(tier),
                require_directional_argmax=require_directional_argmax,
                side=side,
            )
            path = build_portfolio_path(weighted)
            statistics = strategy_statistics(path, cost_bps=base_cost_bps)
            break_even = break_even_cost_bps(path)
            selection_rate = float(weighted["tier_selected"].mean())
            active_signal_rate = float(weighted["active_signal"].mean())
            active = weighted["active_signal"]
            active_nonflat = active & weighted["realised_return"].ne(0.0)
            if active_nonflat.any():
                direction_hit_rate = float((
                    np.sign(weighted.loc[active_nonflat, "signal_score"])
                    == np.sign(weighted.loc[active_nonflat, "realised_return"])
                ).mean())
            else:
                direction_hit_rate = 0.0
            row = {
                "strategy": slug,
                "tier": tier,
                "weighting": weighting,
                "side": side,
                "selection_rate": selection_rate,
                "active_signal_rate": active_signal_rate,
                "active_signal_n": int(active.sum()),
                "active_nonflat_n": int(active_nonflat.sum()),
                "active_nonflat_direction_hit_rate": direction_hit_rate,
                "break_even_cost_bps": break_even,
                **statistics,
            }
            summary_rows.append(row)
            curve = cost_sensitivity(path, cost_grid_bps)
            curve.insert(0, "strategy", slug)
            sensitivity_parts.append(curve)
            details[slug] = row

            export_positions = weighted.drop(
                columns=["_window_end_ts", "_target_time_ts"], errors="ignore"
            )
            export_positions.to_csv(positions_dir / f"{slug}.csv", index=False)
            path["net_return_at_base_cost"] = (
                path["gross_return"] - base_cost_bps / 10_000.0 * path["turnover"]
            )
            path["gross_wealth"] = (1.0 + path["gross_return"]).cumprod()
            path["net_wealth_at_base_cost"] = (
                1.0 + path["net_return_at_base_cost"]
            ).cumprod()
            path.to_csv(paths_dir / f"{slug}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    sensitivity = pd.concat(sensitivity_parts, ignore_index=True)
    summary.to_csv(target / "strategy_summary.csv", index=False)
    sensitivity.to_csv(target / "cost_sensitivity.csv", index=False)
    limitations = [
        "research execution assumes zero latency at the window-end close",
        "cost is linear in turnover and does not model market impact or fill constraints",
    ]
    if side == "long_short":
        limitations.append(
            "long_short results assume shorting is available and exclude stock-borrow costs"
        )
    report: dict[str, object] = {
        "predictions_path": str(source),
        "close_dir": str(Path(close_dir)),
        "signal_definition": "prob_up - prob_down",
        "score_threshold": float(score_threshold),
        "require_directional_argmax": bool(require_directional_argmax),
        "side": side,
        "base_cost_bps_one_way": float(base_cost_bps),
        "cost_grid_bps_one_way": [float(value) for value in cost_grid_bps],
        "label_return_alignment_audit": alignment,
        "portfolio_construction": {
            "cross_section": (
                "At each window_end, active stock weights are normalised to unit gross "
                "exposure; the portfolio return is sum(weight_i * stock_return_i)."
            ),
            "compounding": (
                "Cross-sectional portfolio returns are compounded chronologically. "
                "Stock-minute rows are never concatenated and compounded as separate periods."
            ),
            "flat_policy": (
                "No position when flat has the largest probability, unless explicitly overridden."
            ),
            "gap_policy": (
                "Positions are closed before lunch/overnight/warm-up gaps and re-entered after them."
            ),
            "turnover": "sum(abs(new_weight - old_weight)), including entries and final exits",
        },
        "metric_definitions": {
            "gross_total_return": (
                "Non-annualised cumulative return over the saved test trading days."
            ),
            "net_total_return": (
                "Non-annualised test-period cumulative return after linear one-way cost times turnover."
            ),
            "intraday_annualised_sharpe": (
                "Mean/std of chronological portfolio-minute returns scaled by "
                "sqrt(observed intervals per day * 252); zeros when the strategy is inactive remain included."
            ),
            "daily_annualised_sharpe": (
                "Mean/std of within-day compounded portfolio returns scaled by sqrt(252)."
            ),
            "sharpe_warning": (
                "Both Sharpe estimates use only the saved ten-day test period and are diagnostic, "
                "not reliable long-run estimates."
            ),
            "break_even_cost_bps": (
                "One-way bps cost at which chronologically compounded terminal net return equals zero."
            ),
        },
        "strategies": details,
        "limitations": limitations,
    }
    (target / "metrics.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (target / "README.md").write_text(
        "\n".join([
            "# LSTM 概率策略回测口径",
            "",
            "- 每个 `window_end` 先在股票横截面上构造组合：活跃股票权重按绝对值归一化，",
            "  该分钟组合收益为 `sum(weight_i * stock_return_i)`。",
            "- 然后仅沿时间轴复利这些“组合分钟收益”；没有把股票—分钟行串联复利。",
            "- `gross_total_return` 和 `net_total_return` 是当前测试期的累计收益，不是年化收益。",
            "- `intraday_annualised_sharpe` 按实际每日预测区间数与 252 天缩放；",
            "  `daily_annualised_sharpe` 先复利为日收益再按 252 天缩放。",
            "- 两种 Sharpe 都只基于 10 个已查看的测试交易日，仅作诊断，不能解读为长期稳定性。",
            "- 成本是单边 bps × 组合绝对权重变化，包括每段入场、换仓和最终退场。",
            "- 仓位只由当时可见概率及验证集冻结的档位标记生成；真实收益/标签只用于事后评价。",
            "",
            "详细定义见 `metrics.json`，策略汇总见 `strategy_summary.csv`，成本压力曲线见 `cost_sensitivity.csv`。",
            "",
        ]),
        encoding="utf-8",
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for strategy, group in sensitivity.groupby("strategy", sort=False):
        axes[0].plot(
            group["cost_bps"], group["net_total_return"] * 100.0,
            marker="o", label=strategy,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title(f"{side.replace('_', '-')} cost sensitivity")
    axes[0].set_xlabel("One-way cost (bps)")
    axes[0].set_ylabel("Test-period net return (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    display = summary.sort_values("break_even_cost_bps")
    axes[1].barh(display["strategy"], display["break_even_cost_bps"])
    axes[1].axvline(base_cost_bps, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Break-even one-way cost")
    axes[1].set_xlabel("bps")
    axes[1].grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(target / "strategy_cost_sensitivity.png", dpi=170)
    plt.close(figure)
    return {"summary": summary, "cost_sensitivity": sensitivity, "report": report}
