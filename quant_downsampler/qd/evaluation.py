"""Reproducible IC and quantile-layer evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import OUTPUT_DIR
from .factors import FACTOR_NAMES, compute_all_factors, load_daily_data, save_factors


def _markdown_table(table: pd.DataFrame, percent: bool = False) -> str:
    frame = table.copy()
    formatter = (lambda x: f"{x:.6%}") if percent else (lambda x: f"{x:.6f}")
    headers = [str(frame.index.name or "factor"), *map(str, frame.columns)]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for idx, row in frame.iterrows():
        values = [str(idx)] + [formatter(float(v)) if pd.notna(v) else "NaN" for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def compute_ic_series(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    method: str = "pearson",
    min_stocks: int = 30,
) -> pd.Series:
    dates = factor.index.intersection(forward_return.index)
    stocks = factor.columns.intersection(forward_return.columns)
    values: dict[pd.Timestamp, float] = {}
    for date in dates:
        pair = pd.concat(
            [factor.loc[date, stocks], forward_return.loc[date, stocks]], axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < min_stocks or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
            continue
        if method == "pearson":
            corr = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
        elif method == "spearman":
            corr = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
        else:
            raise ValueError("method must be pearson or spearman")
        if np.isfinite(corr):
            values[pd.Timestamp(date)] = float(corr)
    return pd.Series(values, name=method, dtype=np.float64).sort_index()


def _series_stats(
    series: pd.Series,
    mean_key: str,
    ir_key: str,
    icir_key: str,
) -> dict[str, float]:
    x = series.dropna()
    if len(x) < 2:
        return {mean_key: np.nan, f"{mean_key}_std": np.nan,
                icir_key: np.nan, ir_key: np.nan}
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    icir = mean / std if std > 0 else np.nan
    annual_ir = icir * np.sqrt(252) if np.isfinite(icir) else np.nan
    return {
        mean_key: mean,
        f"{mean_key}_std": std,
        icir_key: icir,
        ir_key: annual_ir,
    }


def evaluate_all_factors(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    ic_table: dict[str, pd.Series] = {}
    rank_table: dict[str, pd.Series] = {}
    for name in FACTOR_NAMES:
        ic = compute_ic_series(factors[name], forward_return, "pearson")
        rank_ic = compute_ic_series(factors[name], forward_return, "spearman")
        ic_table[name] = ic
        rank_table[name] = rank_ic
        ic_stats = _series_stats(ic, "IC", "IR", "ICIR")
        rank_stats = _series_stats(
            rank_ic, "rank_IC", "rank_IR", "rank_ICIR"
        )
        rows.append({
            "factor": name,
            **ic_stats,
            **rank_stats,
            "IC_positive_ratio": float((ic > 0).mean()),
            "rank_IC_positive_ratio": float((rank_ic > 0).mean()),
            "n_days": int(len(ic)),
        })
    summary = pd.DataFrame(rows).set_index("factor")
    summary["abs_ICIR"] = summary["ICIR"].abs()
    summary = summary.sort_values("abs_ICIR", ascending=False)
    return summary, pd.DataFrame(ic_table), pd.DataFrame(rank_table)


def factor_layering_daily(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    dates = factor.index.intersection(forward_return.index)
    stocks = factor.columns.intersection(forward_return.columns)
    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for date in dates:
        pair = pd.concat(
            [factor.loc[date, stocks].rename("factor"),
             forward_return.loc[date, stocks].rename("return")], axis=1
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < n_groups * 6 or pair["factor"].nunique() < n_groups:
            continue
        ranks = pair["factor"].rank(method="first")
        groups = pd.qcut(ranks, n_groups, labels=False) + 1
        rows[pd.Timestamp(date)] = {
            f"Q{group}": float(pair.loc[groups == group, "return"].mean())
            for group in range(1, n_groups + 1)
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _plot_results(
    ic_daily: pd.DataFrame,
    layer_means: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ic_daily.fillna(0).cumsum().plot(ax=ax)
    ax.set_title("Cumulative daily IC")
    ax.set_ylabel("Cumulative IC")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_ic.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, name in zip(axes.ravel(), FACTOR_NAMES):
        layer_means.loc[name].plot.bar(ax=ax)
        ax.set_title(name)
        ax.set_ylabel("Mean next-day return")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "factor_layers.png", dpi=160)
    plt.close(fig)


def _write_report(summary: pd.DataFrame, layer_means: pd.DataFrame, out_dir: Path) -> None:
    display = summary[["IC", "IR", "ICIR", "rank_IC", "rank_IR", "rank_ICIR", "n_days"]]
    lines = [
        "# 因子构建与评价报告",
        "",
        "本报告由 `scripts/run_factor_eval.py` 根据当前复权日频数据自动生成。",
        "",
        "## 因子",
        "",
        "- 示例因子：`log(std(MA1, MA5, MA10, MA20 of amount))`",
        "- 另构建三个因子：5 日动量、主买主卖失衡、日内振幅。",
        "",
        "## IC/IR/ICIR",
        "",
        "`ICIR = mean(IC)/std(IC)`；`IR = sqrt(252) × ICIR`。负 IC 表示反向有效，排名按 `|ICIR|`。",
        "",
        _markdown_table(display),
        "",
        "## 五分层平均次日收益",
        "",
        _markdown_table(layer_means, percent=True),
        "",
        "详细的每日 IC、Rank IC、分层收益和图表均保存在本目录，可直接复核。",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(
    data_dir: Path | None = None,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    target = Path(out_dir or (OUTPUT_DIR / "factors"))
    target.mkdir(parents=True, exist_ok=True)
    data = load_daily_data(data_dir)
    factors = compute_all_factors(data)
    forward = factors["forward_return_1d"]
    summary, ic_daily, rank_ic_daily = evaluate_all_factors(factors, forward)
    layerings = {
        name: factor_layering_daily(factors[name], forward) for name in FACTOR_NAMES
    }
    layer_means = pd.DataFrame({name: x.mean() for name, x in layerings.items()}).T

    save_factors(factors, target)
    summary.to_csv(target / "evaluation_summary.csv", float_format="%.8f")
    ic_daily.to_csv(target / "ic_daily.csv", float_format="%.8f")
    rank_ic_daily.to_csv(target / "rank_ic_daily.csv", float_format="%.8f")
    layer_means.to_csv(target / "layer_mean_returns.csv", float_format="%.8f")
    layer_dir = target / "layering_daily"
    layer_dir.mkdir(exist_ok=True)
    for name, table in layerings.items():
        table.to_csv(layer_dir / f"{name}.csv", float_format="%.8f")
    _plot_results(ic_daily, layer_means, target)
    _write_report(summary, layer_means, target)
    return summary, layerings, factors
