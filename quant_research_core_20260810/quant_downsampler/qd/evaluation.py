"""因子评价模块。

计算:
- IC (Information Coefficient): 因子值与前向收益率的 Pearson 相关系数
- ICIR: mean(IC) / std(IC)
- IR (annualized Information Ratio): sqrt(252) * ICIR
- rank_IC: 因子值与前向收益率的 Spearman 秩相关系数
- rank_ICIR: mean(rank_IC) / std(rank_IC)
- rank_IR: sqrt(252) * rank_ICIR
- 因子分层效果: 按因子值分 N 组,计算各组平均收益率

输出:
- 表格: 每个因子的 IC/IR/ICIR/rank_IC/rank_IR/rank_ICIR
- 图表: 累计 IC 曲线、分层收益柱状图
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import OUTPUT_DIR
from .factors import (
    REQUIRED_FACTOR_NAMES,
    compute_all_factors,
    load_daily_data,
    select_required_factors,
)


def compute_ic_series(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    method: str = "pearson",
) -> pd.Series:
    """计算每日截面 IC 序列。

    每天对横截面上的因子值和前向收益率求相关系数。

    Args:
        factor: 因子值,行=日期,列=股票
        forward_return: 前向收益率,行=日期,列=股票
        method: "pearson" 或 "spearman"

    Returns:
        Series,索引=日期,值=IC
    """
    common_dates = factor.index.intersection(forward_return.index)
    common_stocks = factor.columns.intersection(forward_return.columns)

    if len(common_dates) == 0 or len(common_stocks) == 0:
        return pd.Series(dtype=float)

    f = factor.loc[common_dates, common_stocks]
    r = forward_return.loc[common_dates, common_stocks]

    ic_values = []
    ic_dates = []

    for date in common_dates:
        f_row = f.loc[date].dropna()
        r_row = r.loc[date].dropna()
        common = f_row.index.intersection(r_row.index)
        if len(common) < 10:  # 至少 10 只股票才计算
            continue
        f_vals = f_row[common].values
        r_vals = r_row[common].values

        # 移除 inf
        mask = np.isfinite(f_vals) & np.isfinite(r_vals)
        if mask.sum() < 10:
            continue

        # scipy emits warnings and returns NaN for a constant cross-section;
        # such a day has no ranking information and should be skipped.
        if np.ptp(f_vals[mask]) == 0 or np.ptp(r_vals[mask]) == 0:
            continue

        if method == "pearson":
            corr, _ = stats.pearsonr(f_vals[mask], r_vals[mask])
        else:
            corr, _ = stats.spearmanr(f_vals[mask], r_vals[mask])

        if np.isfinite(corr):
            ic_values.append(corr)
            ic_dates.append(date)

    return pd.Series(ic_values, index=ic_dates, name=method)


def compute_ic_stats(ic_series: pd.Series) -> dict[str, float]:
    """从 IC 序列计算统计指标。

    Returns:
        dict with keys: IC_mean, IC_std, IR, ICIR, IC_positive_ratio, IC_t_stat
    """
    if len(ic_series) < 2:
        return {"IC_mean": np.nan, "IC_std": np.nan, "IR": np.nan,
                "ICIR": np.nan, "IC_positive_ratio": np.nan, "IC_t_stat": np.nan}

    ic = ic_series.dropna()
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    icir = mean_ic / std_ic if std_ic > 0 else np.nan
    ir = np.sqrt(252.0) * icir if np.isfinite(icir) else np.nan
    positive_ratio = (ic > 0).mean()
    t_stat = mean_ic / (std_ic / np.sqrt(len(ic))) if std_ic > 0 else np.nan

    return {
        "IC_mean": mean_ic,
        "IC_std": std_ic,
        "IR": ir,
        "ICIR": icir,
        "IC_positive_ratio": positive_ratio,
        "IC_t_stat": t_stat,
    }


def evaluate_single_factor(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    factor_name: str = "",
) -> dict[str, float]:
    """评价单个因子,返回所有指标。"""
    ic_pearson = compute_ic_series(factor, forward_return, method="pearson")
    ic_spearman = compute_ic_series(factor, forward_return, method="spearman")

    pearson_stats = compute_ic_stats(ic_pearson)
    spearman_stats = compute_ic_stats(ic_spearman)

    return {
        "factor": factor_name,
        "IC": pearson_stats["IC_mean"],
        "IR": pearson_stats["IR"],
        "ICIR": pearson_stats["ICIR"],
        "IC_positive_ratio": pearson_stats["IC_positive_ratio"],
        "IC_t_stat": pearson_stats["IC_t_stat"],
        "rank_IC": spearman_stats["IC_mean"],
        "rank_IR": spearman_stats["IR"],
        "rank_ICIR": spearman_stats["ICIR"],
        "rank_IC_positive_ratio": spearman_stats["IC_positive_ratio"],
        "n_days": len(ic_pearson),
    }


def evaluate_all_factors(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> pd.DataFrame:
    """评价所有因子,返回汇总表。"""
    factor_names = [k for k in factors if k != "forward_return_1d"]
    rows = []
    for name in factor_names:
        stats = evaluate_single_factor(factors[name], forward_return, name)
        rows.append(stats)
    return pd.DataFrame(rows).set_index("factor").sort_values("ICIR", ascending=False)


def evaluate_factor_stability(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate chronological train/validation/test IC without reusing test labels."""
    dates = forward_return.index
    train_end = int(len(dates) * 0.70)
    val_end = int(len(dates) * 0.85)
    slices = {
        "train": dates[:train_end],
        "validation": dates[train_end:val_end],
        "test": dates[val_end:],
    }
    rows = []
    for split, split_dates in slices.items():
        table = evaluate_all_factors(
            factors,
            forward_return.reindex(split_dates),
        ).reset_index()
        table.insert(0, "split", split)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# 因子分层分析
# ---------------------------------------------------------------------------

def factor_layering(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """因子分层分析。

    每天按因子值将股票分成 n_groups 组,计算每组平均前向收益率。
    返回每组的平均收益率(时间序列均值)。

    Returns:
        DataFrame, columns=[group, mean_return, t_stat]
    """
    daily = factor_layering_daily_returns(factor, forward_return, n_groups)

    result_rows = []
    for g in range(n_groups):
        column = f"Q{g + 1}"
        arr = daily[column].dropna().to_numpy(dtype=float) if column in daily else np.array([])
        if len(arr) > 1:
            mean_ret = arr.mean()
            t_stat = mean_ret / (arr.std(ddof=1) / np.sqrt(len(arr))) if arr.std() > 0 else 0
            result_rows.append({
                "group": g + 1,
                "label": f"Q{g + 1}",
                "mean_return": mean_ret,
                "t_stat": t_stat,
                "n_days": len(arr),
            })

    if not result_rows:
        return pd.DataFrame(
            columns=["label", "mean_return", "t_stat", "n_days"],
        ).rename_axis("group")
    return pd.DataFrame(result_rows).set_index("group")


def factor_layering_daily_returns(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """Return the complete daily Q1..Qn equal-weight return series.

    The index is the signal date and each value is the following trading
    day's close-to-close return.  Keeping the daily series is essential: five
    time averages alone cannot produce or verify a five-curve layer backtest.
    """
    if n_groups < 2:
        raise ValueError("n_groups must be at least two")
    common_dates = factor.index.intersection(forward_return.index)
    common_stocks = factor.columns.intersection(forward_return.columns)
    f = factor.loc[common_dates, common_stocks]
    r = forward_return.loc[common_dates, common_stocks]
    rows: list[dict[str, float | pd.Timestamp]] = []

    for date in common_dates:
        f_row = f.loc[date]
        r_row = r.loc[date]
        valid = f_row.notna() & r_row.notna()
        f_vals = f_row[valid].replace([np.inf, -np.inf], np.nan).dropna()
        r_vals = r_row.reindex(f_vals.index).replace([np.inf, -np.inf], np.nan)
        valid_index = r_vals.dropna().index
        f_vals = f_vals.reindex(valid_index)
        r_vals = r_vals.reindex(valid_index)
        if len(valid_index) < n_groups * 3 or f_vals.nunique() < n_groups:
            continue
        try:
            labels = pd.qcut(
                f_vals.rank(method="first"), n_groups, labels=False,
            )
        except ValueError:
            continue
        row: dict[str, float | pd.Timestamp] = {"signal_date": date}
        for group in range(n_groups):
            group_return = r_vals[labels == group]
            row[f"Q{group + 1}"] = float(group_return.mean()) if len(group_return) else np.nan
        rows.append(row)

    columns = [f"Q{i}" for i in range(1, n_groups + 1)]
    if not rows:
        return pd.DataFrame(columns=columns, dtype=float)
    return pd.DataFrame(rows).set_index("signal_date")[columns].sort_index()


def layering_daily_all_factors(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    n_groups: int = 5,
) -> dict[str, pd.DataFrame]:
    """Build daily layer returns for every actual factor."""
    return {
        name: factor_layering_daily_returns(factor, forward_return, n_groups)
        for name, factor in factors.items()
        if name != "forward_return_1d"
    }


def layering_all_factors(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    n_groups: int = 5,
) -> dict[str, pd.DataFrame]:
    """对所有因子做分层分析。"""
    results = {}
    factor_names = [k for k in factors if k != "forward_return_1d"]
    for name in factor_names:
        results[name] = factor_layering(factors[name], forward_return, n_groups)
    return results


def _layer_performance(
    daily_return: pd.Series,
) -> dict[str, float | int]:
    values = daily_return.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {
            "mean_daily_return": np.nan, "annualized_return": np.nan,
            "annualized_volatility": np.nan, "sharpe_ratio": np.nan,
            "max_drawdown": np.nan, "final_nav": np.nan,
            "t_stat": np.nan, "n_days": 0,
        }
    nav = (1.0 + values).cumprod()
    std = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    years = len(values) / 252.0
    annualized_return = float(nav.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    annualized_volatility = std * np.sqrt(252.0) if np.isfinite(std) else np.nan
    sharpe = float(values.mean() / std * np.sqrt(252.0)) if std > 0 else np.nan
    drawdown = nav.div(nav.cummax()).sub(1.0)
    t_stat = float(values.mean() / (std / np.sqrt(len(values)))) if std > 0 else np.nan
    return {
        "mean_daily_return": float(values.mean()),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "final_nav": float(nav.iloc[-1]),
        "t_stat": t_stat,
        "n_days": int(len(values)),
    }


def save_layering_backtests(
    daily_layerings: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Save five layer curves, long-short diagnostics, and monotonicity tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_rows: list[pd.DataFrame] = []
    nav_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    monotonicity_rows: list[dict[str, float | int | str]] = []
    nav_by_factor: dict[str, pd.DataFrame] = {}

    for factor, raw_daily in daily_layerings.items():
        if raw_daily.empty:
            continue
        daily = raw_daily.copy()
        daily["Q1_minus_Q5"] = daily["Q1"] - daily["Q5"]
        nav = (1.0 + daily).cumprod()
        nav_by_factor[factor] = nav

        daily_item = daily.reset_index()
        daily_item.insert(0, "factor", factor)
        daily_rows.append(daily_item)
        nav_item = nav.reset_index()
        nav_item.insert(0, "factor", factor)
        nav_rows.append(nav_item)

        for group in daily.columns:
            metric_rows.append({
                "factor": factor, "group": group,
                **_layer_performance(daily[group]),
            })

        group_means = daily[[f"Q{i}" for i in range(1, 6)]].mean()
        rho = stats.spearmanr(np.arange(1, 6), group_means.to_numpy()).statistic
        adjacent_decreases = int(sum(
            group_means.iloc[i] >= group_means.iloc[i + 1] for i in range(4)
        ))
        spread_stats = _layer_performance(daily["Q1_minus_Q5"])
        monotonicity_rows.append({
            "factor": factor,
            "layer_spearman": float(rho),
            "adjacent_decreases_out_of_4": adjacent_decreases,
            "strictly_decreasing": bool(adjacent_decreases == 4),
            "q1_minus_q5_mean_daily_return": spread_stats["mean_daily_return"],
            "q1_minus_q5_t_stat": spread_stats["t_stat"],
            "q1_minus_q5_final_nav": spread_stats["final_nav"],
            "n_days": spread_stats["n_days"],
        })

    pd.concat(daily_rows, ignore_index=True).to_csv(
        out_dir / "factor_layer_daily_returns.csv", index=False, float_format="%.8f",
    )
    pd.concat(nav_rows, ignore_index=True).to_csv(
        out_dir / "factor_layer_nav.csv", index=False, float_format="%.8f",
    )
    pd.DataFrame(metric_rows).to_csv(
        out_dir / "factor_layer_metrics.csv", index=False, float_format="%.8f",
    )
    pd.DataFrame(monotonicity_rows).to_csv(
        out_dir / "factor_layer_monotonicity.csv", index=False, float_format="%.8f",
    )
    _plot_layering_nav(nav_by_factor, out_dir)


def _plot_layering_nav(
    nav_by_factor: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2166ac", "#67a9cf", "#bdbdbd", "#ef8a62", "#b2182b"]
    for factor, nav in nav_by_factor.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for index, group in enumerate([f"Q{i}" for i in range(1, 6)]):
            ax.plot(nav.index, nav[group], label=group, color=colors[index], linewidth=1.35)
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        ax.set(title=f"Five-layer NAV: {factor}", xlabel="Signal date", ylabel="NAV")
        ax.grid(alpha=0.25); ax.legend(ncol=5); fig.tight_layout()
        fig.savefig(out_dir / f"layering_nav_{factor}.png", dpi=160)
        plt.close(fig)

    required = [name for name in REQUIRED_FACTOR_NAMES if name in nav_by_factor]
    if required:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
        for ax, factor in zip(axes.flat, required):
            nav = nav_by_factor[factor]
            for index, group in enumerate([f"Q{i}" for i in range(1, 6)]):
                ax.plot(nav.index, nav[group], label=group, color=colors[index], linewidth=1.0)
            ax.axhline(1.0, color="black", linewidth=0.7, linestyle=":")
            ax.set_title(factor); ax.grid(alpha=0.25)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=5)
        fig.suptitle("Prompt-required factors: five-layer NAV", y=0.98)
        fig.tight_layout(rect=(0, 0.05, 1, 0.96))
        fig.savefig(out_dir / "required_factor_layering_nav.png", dpi=170)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def print_evaluation_report(
    summary: pd.DataFrame,
    layerings: dict[str, pd.DataFrame],
) -> None:
    """打印因子评价报告。"""
    print("\n" + "=" * 80)
    print("因子评价报告")
    print("=" * 80)

    print("\n--- IC/IR/ICIR 汇总 ---")
    cols = ["IC", "IR", "ICIR", "rank_IC", "rank_IR", "rank_ICIR",
            "IC_positive_ratio", "n_days"]
    print(summary[cols].to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- 因子分层效果 (Top vs Bottom) ---")
    for name, layering in layerings.items():
        if layering.empty:
            continue
        top = layering.loc[layering.index.max(), "mean_return"]
        bot = layering.loc[layering.index.min(), "mean_return"]
        spread = top - bot
        print(f"  {name:<30s}  Q1={bot:+.6f}  Q{layering.index.max()}={top:+.6f}  spread={spread:+.6f}")

    # 综合排名
    print("\n--- 综合排名(按ICIR降序) ---")
    for i, (idx, row) in enumerate(summary.iterrows()):
        print(f"  {i+1:2d}. {idx:<30s}  IC={row['IC']:+.4f}  IR={row['IR']:+.3f}  rank_IC={row['rank_IC']:+.4f}")


def save_evaluation_results(
    summary: pd.DataFrame,
    layerings: dict[str, pd.DataFrame],
    out_dir: Path | None = None,
) -> Path:
    """Save Task 3 tables and a readable report instead of terminal-only output."""
    out_dir = out_dir or (OUTPUT_DIR / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "factor_summary.csv", float_format="%.6f")

    rows: list[pd.DataFrame] = []
    for factor, table in layerings.items():
        if table.empty:
            continue
        item = table.reset_index().copy()
        item.insert(0, "factor", factor)
        rows.append(item)
    layering_table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    layering_table.to_csv(out_dir / "factor_layering.csv", index=False, float_format="%.6f")

    required_summary = summary.reindex(REQUIRED_FACTOR_NAMES)
    required_summary.to_csv(
        out_dir / "required_factor_summary.csv", float_format="%.6f",
    )
    required_layering = layering_table[
        layering_table["factor"].isin(REQUIRED_FACTOR_NAMES)
    ] if not layering_table.empty else layering_table
    required_layering.to_csv(
        out_dir / "required_factor_layering.csv", index=False, float_format="%.6f",
    )

    lines = [
        "# Factor evaluation (Task 3)", "",
        "## Prompt-required set: example + three original factors", "",
        "```csv", required_summary.to_csv(float_format="%.6f").rstrip(), "```", "",
        "The ten-factor table is an extension and is reported after the required set.", "",
        "## Prompt-required layering", "",
    ]
    for factor in REQUIRED_FACTOR_NAMES:
        table = layerings.get(factor, pd.DataFrame())
        if not table.empty:
            lines.extend(["", f"### {factor}", "", "```csv", table.to_csv(float_format="%.6f").rstrip(), "```"])
    lines.extend(["", "## Extended ten-factor IC summary", "", "```csv", summary.to_csv(float_format="%.6f").rstrip(), "```", "", "## Extended layering"])
    for factor, table in layerings.items():
        if not table.empty:
            lines.extend(["", f"### {factor}", "", "```csv", table.to_csv(float_format="%.6f").rstrip(), "```"])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Evaluation results saved: {out_dir}")
    return out_dir


def run_evaluation(data_dir: Path | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """运行完整因子评价流程。

    Returns:
        (summary_df, layerings_dict, factors_dict)
    """
    data_dir = data_dir or (OUTPUT_DIR / "daily")
    daily_data = load_daily_data(data_dir)

    print("计算因子...")
    factors = compute_all_factors(daily_data)

    forward_return = factors.pop("forward_return_1d")
    # A factor observation is evaluated only when the stock traded on both
    # signal day and realization day.  This prevents forward-filled halted
    # prices from contributing artificial zero returns to IC and layering.
    volume = daily_data["volume"]
    tradable = (volume > 0) & (volume.shift(-1) > 0)
    forward_return = forward_return.where(tradable)

    print("评价因子...")
    summary = evaluate_all_factors(factors, forward_return)

    print("分层分析...")
    layerings = layering_all_factors(factors, forward_return)

    print_evaluation_report(summary, layerings)

    # 把 forward_return 放回去，方便调用方直接保存
    factors["forward_return_1d"] = forward_return

    return summary, layerings, factors
