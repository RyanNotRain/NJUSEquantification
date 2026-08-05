"""策略回测模块。

实现:
1. IC 加权信号选股(每日选 top 10,等权配置)
2. 收盘价调仓(卖出手续费万5,买入免费)
3. 开盘价调仓(同上)
4. 净值曲线、年化波动率、最大回撤、夏普比率

关键问题分析:
  使用"当日因子值与次日收益率"计算 IC 来加权因子,存在**前视偏差**。
  因为在 t 日你无法知道 t→t+1 的收益率,也就无法计算当天的 IC 权重。
  正确做法:使用滚动历史窗口计算 IC 权重,例如用过去 60 天的 IC 均值作为权重。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .factors import compute_all_factors, load_daily_data


# ---------------------------------------------------------------------------
# 核心: IC 加权信号生成
# ---------------------------------------------------------------------------

def compute_rolling_ic_weights(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    lookback: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算每个因子的滚动 IC 权重(无前视偏差)。

    关键修正: 在 t 日,我们只能知道 t-1 及之前的收益率。
    因此 IC 计算为: corr(factor[t-1], forward_return[t-1])
    其中 forward_return[t-1] = (close[t] - close[t-1]) / close[t-1]
    这个收益率在 t 日开盘前就已经知道了。

    然后对过去 lookback 天的 IC 求均值,作为 t 日各因子的权重。

    Returns:
        (ic_weights, ic_direction)
        - ic_weights: DataFrame, 行=日期, 列=因子名, 值=abs(IC) 权重
        - ic_direction: DataFrame, 行=日期, 列=因子名, 值=+1或-1(IC 方向)
    """
    from scipy import stats

    factor_names = [k for k in factors if k != "forward_return_1d"]
    dates = sorted(forward_return.index)
    common_stocks = forward_return.columns

    # 预先计算每个因子的每日 IC(无前视偏差版本)
    # IC[t] = corr(factor[t-1], forward_return[t-1])
    # 其中 forward_return[t-1] = (close[t] - close[t-1]) / close[t-1]
    daily_ic: dict[str, pd.Series] = {}
    for name in factor_names:
        f = factors[name]
        ic_series = []
        ic_dates = []
        for i in range(1, len(dates)):
            date = dates[i]       # 当前日期 t
            prev_date = dates[i - 1]  # 前一日 t-1

            if prev_date not in f.index or date not in forward_return.index:
                continue

            # 用 t-1 的因子值 和 t-1→t 的收益率(即 forward_return[t-1])
            f_row = f.loc[prev_date].reindex(common_stocks).dropna()
            r_row = forward_return.loc[prev_date].reindex(common_stocks).dropna()

            common = f_row.index.intersection(r_row.index)
            if len(common) < 30:
                continue
            f_vals = f_row[common].values
            r_vals = r_row[common].values
            mask = np.isfinite(f_vals) & np.isfinite(r_vals)
            if mask.sum() < 30:
                continue
            corr, _ = stats.pearsonr(f_vals[mask], r_vals[mask])
            if np.isfinite(corr):
                ic_series.append(corr)
                ic_dates.append(date)  # IC 归于 t 日(此时已知)
        daily_ic[name] = pd.Series(ic_series, index=ic_dates)

    # 对每个日期计算滚动 IC 均值
    ic_weights = pd.DataFrame(index=dates, columns=factor_names, dtype=np.float64)
    ic_direction = pd.DataFrame(index=dates, columns=factor_names, dtype=np.float64)

    for name in factor_names:
        ic = daily_ic[name]
        rolling = ic.rolling(window=lookback, min_periods=max(10, lookback // 3)).mean()
        ic_weights[name] = rolling.abs().reindex(dates)   # 绝对 IC 作为权重
        ic_direction[name] = np.sign(rolling).reindex(dates)  # IC 方向

    return ic_weights, ic_direction


def compute_naive_ic_weights(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算 naive IC 权重(存在前视偏差)。

    题目描述的原始版本:
      IC[t] = corr(factor[t], forward_return[t])
      其中 forward_return[t] = (close[t+1] - close[t]) / close[t]

    问题: 在 t 日收盘时, forward_return[t] 还未发生(需要 t+1 日收盘价),
          因此无法在 t 日计算出这个 IC 权重。这导致回测结果严重高估。

    Returns:
        (ic_weights, ic_direction)
    """
    from scipy import stats

    factor_names = [k for k in factors if k != "forward_return_1d"]
    dates = sorted(forward_return.index)
    common_stocks = forward_return.columns

    ic_weights = pd.DataFrame(index=dates, columns=factor_names, dtype=np.float64)
    ic_direction = pd.DataFrame(index=dates, columns=factor_names, dtype=np.float64)

    for name in factor_names:
        f = factors[name]
        ic_series = []
        ic_dates = []
        for date in dates:
            if date not in f.index or date not in forward_return.index:
                continue
            f_row = f.loc[date].reindex(common_stocks).dropna()
            r_row = forward_return.loc[date].reindex(common_stocks).dropna()
            common = f_row.index.intersection(r_row.index)
            if len(common) < 30:
                continue
            f_vals = f_row[common].values
            r_vals = r_row[common].values
            mask = np.isfinite(f_vals) & np.isfinite(r_vals)
            if mask.sum() < 30:
                continue
            corr, _ = stats.pearsonr(f_vals[mask], r_vals[mask])
            if np.isfinite(corr):
                ic_series.append(corr)
                ic_dates.append(date)
        ic = pd.Series(ic_series, index=ic_dates)
        ic_weights[name] = ic.abs().reindex(dates)
        ic_direction[name] = np.sign(ic).reindex(dates)

    return ic_weights, ic_direction


def compute_combined_signal(
    factors: dict[str, pd.DataFrame],
    ic_weights: pd.DataFrame,
    ic_direction: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.Series:
    """计算某一天的组合信号(无前视偏差)。

    信号 = sum( direction_i * zscore(factor_i) * abs_weight_i )

    每只股票在每个因子上先做截面 z-score 标准化,避免量纲差异。
    然后用 abs(IC) 加权,用 sign(IC) 决定方向。

    Returns:
        Series, index=股票代码, values=信号值
    """
    factor_names = [k for k in factors if k != "forward_return_1d"]
    if date not in ic_weights.index:
        return pd.Series(dtype=float)

    w = ic_weights.loc[date].dropna()
    d = ic_direction.loc[date].dropna()
    if len(w) == 0:
        return pd.Series(dtype=float)

    active_factors = [n for n in factor_names if n in w.index and n in d.index]
    if not active_factors:
        return pd.Series(dtype=float)

    # 归一化权重
    w_active = w[active_factors]
    total = w_active.sum()
    if total == 0:
        return pd.Series(dtype=float)
    w_norm = w_active / total

    # 加权组合(每个因子先 z-score 标准化)
    signal = None
    for name in active_factors:
        f = factors[name]
        if date not in f.index:
            continue
        f_row = f.loc[date].dropna()
        if len(f_row) < 10:
            continue

        # 截面 z-score 标准化
        f_z = (f_row - f_row.mean()) / (f_row.std(ddof=1) + 1e-12)

        # 方向调整 + 加权
        contrib = f_z * d[name] * w_norm[name]

        if signal is None:
            signal = contrib
        else:
            signal = signal.add(contrib, fill_value=0)

    if signal is None:
        return pd.Series(dtype=float)
    return signal


# ---------------------------------------------------------------------------
# 回测引擎
# ---------------------------------------------------------------------------

def run_backtest(
    factors: dict[str, pd.DataFrame],
    forward_return: pd.DataFrame,
    close: pd.DataFrame,
    open_price: pd.DataFrame | None = None,
    lookback: int = 60,
    top_n: int = 10,
    sell_fee: float = 0.0005,  # 万5
    buy_fee: float = 0.0,
    initial_capital: float = 10_000_000,  # 1千万
    rebalance_at: str = "close",  # "close" 或 "open"
    use_naive_ic: bool = False,  # True=含前视偏差的原始版本, False=修正版本
) -> dict:
    """运行策略回测。

    use_naive_ic=False (默认/修正版):
      用滚动历史窗口计算 IC 权重,无前视偏差。

    use_naive_ic=True (题目原始描述):
      用当日因子值与次日收益率直接计算 IC 权重,存在前视偏差。
      该版本仅用于对比分析,展示前视偏差的影响。
    """
    dates = sorted(close.index)

    if use_naive_ic:
        print(f"  计算 Naive IC 权重(含前视偏差)...")
        ic_weights, ic_direction = compute_naive_ic_weights(factors, forward_return)
    else:
        print(f"  计算滚动 IC 权重 (lookback={lookback})...")
        ic_weights, ic_direction = compute_rolling_ic_weights(factors, forward_return, lookback)

    # 确定可交易日期(需要有 IC 权重和因子数据)
    factor_names = [k for k in factors if k != "forward_return_1d"]
    valid_dates = []
    for d in dates:
        if d not in ic_weights.index:
            continue
        w = ic_weights.loc[d].dropna()
        if len(w) == 0:
            continue
        # 至少有 top_n 只股票有因子数据
        has_enough = False
        for name in factor_names:
            if name in w.index and d in factors[name].index:
                if factors[name].loc[d].notna().sum() >= top_n:
                    has_enough = True
                    break
        if has_enough:
            valid_dates.append(d)

    if len(valid_dates) < lookback + 5:
        raise ValueError(f"可交易日期不足: {len(valid_dates)}")

    # 回测: 从 lookback 天开始,等权每日调仓
    start_idx = lookback
    nav = [initial_capital]
    nav_dates = [valid_dates[start_idx]]
    daily_ret_list = []
    turnover_list = []
    prev_top_stocks: set[str] = set()
    top_stocks_history: list[tuple[pd.Timestamp, list[str]]] = []  # 记录每日选股

    for i in range(start_idx, len(valid_dates) - 1):
        date = valid_dates[i]
        next_date = valid_dates[i + 1]

        # 1) 计算信号,选股
        signal = compute_combined_signal(factors, ic_weights, ic_direction, date)
        if signal.empty or signal.notna().sum() < top_n:
            nav.append(nav[-1])
            nav_dates.append(next_date)
            daily_ret_list.append(0.0)
            turnover_list.append(0.0)
            continue

        top_stocks = set(signal.nlargest(top_n).index.tolist())
        top_stocks_history.append((date, sorted(top_stocks)))

        # 2) 计算换手率
        if prev_top_stocks:
            sold = prev_top_stocks - top_stocks
            turnover = len(sold) / top_n
        else:
            turnover = 1.0  # 首日全部买入
        turnover_list.append(turnover)

        # 3) 计算持仓收益
        if rebalance_at == "close":
            # 以 t 日收盘买入, t+1 日收盘卖出
            returns = []
            for s in top_stocks:
                if s in close.columns and s in forward_return.columns:
                    if date in forward_return.index:
                        ret = forward_return.loc[date, s]
                        if pd.notna(ret) and np.isfinite(ret):
                            returns.append(ret)
            if returns:
                portfolio_ret = np.mean(returns)
            else:
                portfolio_ret = 0.0
        else:
            # 以 t+1 日开盘买入, t+2 日开盘卖出
            returns = []
            if open_price is not None:
                for s in top_stocks:
                    if s in open_price.columns:
                        if next_date in open_price.index:
                            buy_p = open_price.loc[next_date, s]
                            # 找 t+2 的日期
                            next_idx = valid_dates.index(next_date)
                            if next_idx + 1 < len(valid_dates):
                                t2_date = valid_dates[next_idx + 1]
                                if t2_date in open_price.index:
                                    sell_p = open_price.loc[t2_date, s]
                                    if pd.notna(buy_p) and pd.notna(sell_p) and buy_p > 0:
                                        returns.append((sell_p - buy_p) / buy_p)
            if returns:
                portfolio_ret = np.mean(returns)
            else:
                portfolio_ret = 0.0

        # 4) 扣除交易成本
        cost = turnover * sell_fee  # 卖出部分收手续费
        portfolio_ret -= cost

        # 5) 更新净值
        new_nav = nav[-1] * (1 + portfolio_ret)
        nav.append(new_nav)
        nav_dates.append(next_date)
        daily_ret_list.append(portfolio_ret)
        prev_top_stocks = top_stocks

    # 构建净值序列
    nav_series = pd.Series(nav, index=nav_dates, name="nav")
    daily_returns = pd.Series(daily_ret_list, index=nav_dates[1:], name="daily_return")
    turnover_series = pd.Series(turnover_list, index=nav_dates[1:], name="turnover")

    # 计算指标
    annual_factor = np.sqrt(252)
    annual_vol = daily_returns.std() * annual_factor if len(daily_returns) > 1 else np.nan
    annual_ret = daily_returns.mean() * 252 if len(daily_returns) > 1 else np.nan
    sharpe = (annual_ret - 0.02) / annual_vol if annual_vol > 0 else np.nan

    # 最大回撤
    running_max = nav_series.cummax()
    drawdown = (nav_series - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "nav_series": nav_series,
        "daily_returns": daily_returns,
        "annual_return": annual_ret,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "turnover_series": turnover_series,
        "final_nav": nav_series.iloc[-1],
        "total_return": (nav_series.iloc[-1] / initial_capital - 1),
        "top_stocks_history": top_stocks_history,
    }


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

def plot_nav_curve(
    result_close: dict,
    result_open: dict | None,
    result_naive_close: dict | None = None,
    result_naive_open: dict | None = None,
    out_dir: Path | None = None,
) -> Path:
    """绘制净值曲线和回撤图。

    Returns:
        保存的图片路径
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    out_dir = out_dir or (OUTPUT_DIR / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # 图1: 净值曲线
    ax = axes[0]
    for label, result, color, ls in [
        ("Corrected-Close", result_close, "#1f77b4", "-"),
        ("Corrected-Open", result_open, "#ff7f0e", "-"),
        ("Naive-Close (bias)", result_naive_close, "#d62728", "--"),
        ("Naive-Open (bias)", result_naive_open, "#9467bd", "--"),
    ]:
        if result is None:
            continue
        nav = result["nav_series"]
        ax.plot(nav.index, nav.values / 1e6, label=label, color=color, linestyle=ls, linewidth=1.5)

    ax.set_ylabel("NAV (Million)", fontsize=12)
    ax.set_title("Strategy NAV Curve", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=10, color="gray", linestyle=":", alpha=0.5)

    # 图2: 回撤
    ax2 = axes[1]
    for label, result, color, ls in [
        ("Corrected-Close", result_close, "#1f77b4", "-"),
        ("Corrected-Open", result_open, "#ff7f0e", "-"),
        ("Naive-Close (bias)", result_naive_close, "#d62728", "--"),
        ("Naive-Open (bias)", result_naive_open, "#9467bd", "--"),
    ]:
        if result is None:
            continue
        nav = result["nav_series"]
        running_max = nav.cummax()
        drawdown = (nav - running_max) / running_max * 100
        ax2.fill_between(nav.index, 0, drawdown.values, label=label,
                         color=color, alpha=0.3, linewidth=0.5)

    ax2.set_ylabel("Drawdown (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_title("Drawdown", fontsize=14)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    path = out_dir / "nav_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nNAV chart saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 选股记录
# ---------------------------------------------------------------------------

def save_top_stocks(
    result: dict,
    label: str,
    out_dir: Path | None = None,
) -> Path:
    """保存每日选股记录到 CSV,并打印出现频率最高的股票。

    Returns:
        CSV 文件路径
    """
    from collections import Counter

    out_dir = out_dir or (OUTPUT_DIR / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = result.get("top_stocks_history", [])
    if not history:
        return None

    # 保存 CSV
    rows = []
    for date, stocks in history:
        for rank, stock in enumerate(stocks, 1):
            rows.append({"date": date.strftime("%Y-%m-%d"), "rank": rank, "stock": stock})
    df = pd.DataFrame(rows)
    safe_label = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = out_dir / f"top_stocks_{safe_label}.csv"
    df.to_csv(path, index=False)
    print(f"  选股记录已保存: {path}")

    # 统计出现频率
    all_stocks = [s for _, stocks in history for s in stocks]
    counter = Counter(all_stocks)
    print(f"\n  [{label}] 出现频率最高的 20 只股票:")
    for rank, (stock, count) in enumerate(counter.most_common(20), 1):
        pct = count / len(history) * 100
        print(f"    {rank:2d}. {stock}  {count}次 ({pct:.1f}%)")

    return path


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def print_backtest_report(
    result_close: dict,
    result_open: dict | None,
    result_naive_close: dict | None = None,
    result_naive_open: dict | None = None,
) -> None:
    """打印回测报告。"""
    print("\n" + "=" * 80)
    print("策略回测报告")
    print("=" * 80)

    for label, result in [
        ("修正版-收盘价调仓(无前视偏差)", result_close),
        ("修正版-开盘价调仓(无前视偏差)", result_open),
        ("Naive版-收盘价调仓(含前视偏差)", result_naive_close),
        ("Naive版-开盘价调仓(含前视偏差)", result_naive_open),
    ]:
        if result is None:
            continue
        print(f"\n--- {label} ---")
        print(f"  年化收益率:     {result['annual_return']*100:+.2f}%")
        print(f"  年化波动率:     {result['annual_volatility']*100:.2f}%")
        print(f"  夏普比率:       {result['sharpe_ratio']:+.3f}")
        print(f"  最大回撤:       {result['max_drawdown']*100:.2f}%")
        print(f"  最终净值:       {result['final_nav']:,.0f}")
        print(f"  总收益率:       {result['total_return']*100:+.2f}%")
        print(f"  日均换手率:     {result['turnover_series'].mean()*100:.2f}%")
        print(f"  交易天数:       {len(result['daily_returns'])}")

    # 前视偏差对比
    if result_naive_close is not None and result_close is not None:
        print(f"\n--- 前视偏差影响分析 ---")
        naive_ret = result_naive_close["annual_return"]
        fixed_ret = result_close["annual_return"]
        bias = naive_ret - fixed_ret
        print(f"  Naive 版年化收益:  {naive_ret*100:+.2f}%")
        print(f"  修正版年化收益:    {fixed_ret*100:+.2f}%")
        print(f"  前视偏差高估:      {bias*100:+.2f}%")
        naive_nav = result_naive_close["final_nav"]
        fixed_nav = result_close["final_nav"]
        print(f"  Naive 版最终净值:  {naive_nav:,.0f}")
        print(f"  修正版最终净值:    {fixed_nav:,.0f}")
        print(f"  净值高估幅度:      {(naive_nav/fixed_nav-1)*100:+.1f}%")


def run_full_backtest(
    data_dir: Path | None = None,
    lookback: int = 60,
    top_n: int = 10,
) -> tuple[dict, dict | None, dict | None]:
    """运行完整回测(收盘价 + 开盘价 + naive 对比)。"""
    data_dir = data_dir or (OUTPUT_DIR / "daily")
    daily_data = load_daily_data(data_dir)

    print("计算因子...")
    factors = compute_all_factors(daily_data)
    forward_return = factors.pop("forward_return_1d")
    close = daily_data["close"]
    open_price = daily_data.get("open")

    # --- 修正版(无前视偏差) ---
    print("\n>>> 修正版-收盘价调仓(无前视偏差) <<<")
    result_close = run_backtest(
        factors, forward_return, close, open_price=None,
        lookback=lookback, top_n=top_n, rebalance_at="close",
        use_naive_ic=False,
    )

    result_open = None
    if open_price is not None:
        print("\n>>> 修正版-开盘价调仓(无前视偏差) <<<")
        result_open = run_backtest(
            factors, forward_return, close, open_price=open_price,
            lookback=lookback, top_n=top_n, rebalance_at="open",
            use_naive_ic=False,
        )
    else:
        print("\n⚠ 未加载 open 数据,跳过开盘价调仓")

    # --- Naive 版(含前视偏差,仅用于对比) ---
    print("\n>>> Naive版-收盘价调仓(含前视偏差,仅用于对比) <<<")
    result_naive_close = run_backtest(
        factors, forward_return, close, open_price=None,
        lookback=lookback, top_n=top_n, rebalance_at="close",
        use_naive_ic=True,
    )

    result_naive_open = None
    if open_price is not None:
        print("\n>>> Naive版-开盘价调仓(含前视偏差,仅用于对比) <<<")
        result_naive_open = run_backtest(
            factors, forward_return, close, open_price=open_price,
            lookback=lookback, top_n=top_n, rebalance_at="open",
            use_naive_ic=True,
        )

    print_backtest_report(result_close, result_open, result_naive_close, result_naive_open)

    # 保存选股记录
    print("\n" + "=" * 80)
    print("选股记录")
    print("=" * 80)
    save_top_stocks(result_close, "修正版-收盘价调仓")
    save_top_stocks(result_open, "修正版-开盘价调仓") if result_open else None
    save_top_stocks(result_naive_close, "Naive版-收盘价调仓")
    save_top_stocks(result_naive_open, "Naive版-开盘价调仓") if result_naive_open else None

    # 绘图
    plot_nav_curve(result_close, result_open, result_naive_close, result_naive_open)

    print(FORWARD_BIAS_ANALYSIS)

    return result_close, result_open, result_naive_close, result_naive_open


# ---------------------------------------------------------------------------
# 前视偏差分析
# ---------------------------------------------------------------------------

FORWARD_BIAS_ANALYSIS = """
=== 前视偏差分析 ===

问题: 使用"当日各因子值与次日收益率计算的 IC"对因子进行加权,存在严重的前视偏差。

具体来说:
  在 t 日收盘后,我们只能看到 t 日及之前的因子值和收益率。
  t→t+1 的收益率要到 t+1 日收盘后才能知道。
  因此,在 t 日无法计算"因子_t 与 收益率_{t→t+1} 的 IC"。

如果用这个 IC 做权重,就相当于:
  → 用 t+1 日的信息来决定 t 日的交易
  → 回测结果会高估实际收益(因为用了未来信息)
  → 实盘中无法复现

修正方案:
  1. 滚动窗口法(本代码已采用):
     在 t 日,用过去 [t-lookback, t-1] 的因子与收益率计算 IC 均值,
     以此作为 t 日各因子的权重。
     完全避免前视偏差。

  2. 衰减加权法:
     类似滚动窗口,但给近期 IC 更高权重(指数衰减)。

  3. 样本外测试:
     将数据分为训练期和测试期,在训练期确定因子权重,
     在测试期固定权重不变,完全不使用测试期信息。

  4. 纯横截面法:
     不在时间维度上滚动 IC,而是用因子本身的统计特性
     (如 t 值的绝对值)作为权重,完全不依赖未来收益率。

本回测采用方案 1(滚动窗口),权重完全基于历史数据。
"""