"""因子计算模块。

从日频数据计算多个量化因子,支持:
- 示例因子: [1,5,10,20]日成交额均值的标准差的对数
- 动量因子: N日价格动量
- 波动率因子: N日收益率标准差
- 买卖失衡因子: (主买-主卖)/总成交量
- 日内振幅因子: (最高-最低)/收盘价
- 量价相关因子: 成交量变化与价格变化的相关性

输出格式: 每张表行=日期,列=股票代码,和日频数据格式一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .config import OUTPUT_DIR


# The prompt asks for the supplied example factor plus one to three original
# factors.  Keep this four-factor set as the formal answer; the remaining
# factors produced by ``compute_all_factors`` are explicitly extensions.
REQUIRED_FACTOR_NAMES: tuple[str, ...] = (
    "example_factor",
    "momentum_20d",
    "buy_sell_imbalance",
    "volume_price_corr_20d",
)


def load_daily_data(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """加载日频数据,返回 {metric_name: DataFrame}。

    DataFrame 索引为日期,列为股票代码。
    """
    data_dir = data_dir or (OUTPUT_DIR / "daily")
    metrics = ["close", "open", "volume", "amount", "high", "low",
               "buy_volume", "sell_volume"]
    result = {}
    for m in metrics:
        path = data_dir / f"{m}.csv"
        if not path.exists():
            raise FileNotFoundError(f"找不到日频数据: {path}")
        df = pd.read_csv(path, index_col=0)
        # Accept both our compact YYYYMMDD files and the teammate pipeline's
        # ISO YYYY-MM-DD files.  Normalizing here keeps every downstream task
        # independent of the CSV writer's display format.
        df.index = pd.to_datetime(df.index.astype(str), format="mixed")
        result[m] = df.astype(np.float64)
    return result


# ---------------------------------------------------------------------------
# 因子 1: 示例因子 — [1,5,10,20]日成交额均值的标准差的对数
# ---------------------------------------------------------------------------

def compute_example_factor(amount: pd.DataFrame) -> pd.DataFrame:
    """计算示例因子。

    对每只股票:
      - 计算 amount 的 1/5/10/20 日移动平均
      - 对这四个 MA 值取标准差 (按行)
      - 取对数

    返回: DataFrame, 行=日期, 列=股票代码
    """
    windows = [1, 5, 10, 20]
    ma_list = []
    for w in windows:
        if w == 1:
            ma_list.append(amount)
        else:
            # A "20-day mean" is only defined after 20 observations.  Using
            # min_periods=1 silently turned the first 19 rows into shorter,
            # differently defined factors.
            ma_list.append(amount.rolling(window=w, min_periods=w).mean())

    # 将 4 个 MA 沿时间轴堆叠,计算每行每列的标准差
    # ma_list[i] 形状: (T, N_stocks)
    # 堆叠: (4, T, N_stocks) → std over axis=0 → (T, N_stocks)
    stacked = np.stack([ma.values for ma in ma_list], axis=0)
    std_val = np.std(stacked, axis=0, ddof=1)

    # 避免 log(0) 或负数
    std_val = np.maximum(std_val, 1e-12)

    factor = pd.DataFrame(
        np.log(std_val),
        index=amount.index,
        columns=amount.columns,
    )
    return factor


def select_required_factors(
    factors: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return the example factor and exactly three prompt-compliant additions."""
    missing = [name for name in REQUIRED_FACTOR_NAMES if name not in factors]
    if missing:
        raise KeyError(f"missing required factors: {missing}")
    return {name: factors[name] for name in REQUIRED_FACTOR_NAMES}


# ---------------------------------------------------------------------------
# 因子 2: 动量因子 — (close_t - close_{t-N}) / close_{t-N}
# ---------------------------------------------------------------------------

def compute_momentum(close: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20)) -> dict[str, pd.DataFrame]:
    """计算 N 日动量因子。

    momentum_N = (close_t - close_{t-N}) / close_{t-N}

    返回: {f"momentum_{N}d": DataFrame, ...}
    """
    results = {}
    for w in windows:
        mom = (close - close.shift(w)) / close.shift(w)
        mom = mom.replace([np.inf, -np.inf], np.nan)
        results[f"momentum_{w}d"] = mom
    return results


# ---------------------------------------------------------------------------
# 因子 3: 波动率因子 — N 日收益率标准差
# ---------------------------------------------------------------------------

def compute_volatility(close: pd.DataFrame, windows: tuple[int, ...] = (5, 10, 20)) -> dict[str, pd.DataFrame]:
    """计算 N 日波动率因子。

    volatility_N = std(daily_returns, N)

    返回: {f"volatility_{N}d": DataFrame, ...}
    """
    ret = close.pct_change(fill_method=None)
    results = {}
    for w in windows:
        vol = ret.rolling(window=w, min_periods=max(2, w // 2)).std(ddof=1)
        results[f"volatility_{w}d"] = vol
    return results


# ---------------------------------------------------------------------------
# 因子 4: 买卖失衡因子 — (buy_volume - sell_volume) / (buy_volume + sell_volume)
# ---------------------------------------------------------------------------

def compute_buy_sell_imbalance(
    buy_volume: pd.DataFrame, sell_volume: pd.DataFrame
) -> pd.DataFrame:
    """计算买卖失衡因子。

    BSI = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    值域 [-1, 1],正值表示买方主导,负值表示卖方主导。
    """
    total = buy_volume + sell_volume
    # 避免除零
    total = total.replace(0, np.nan)
    bsi = (buy_volume - sell_volume) / total
    return bsi


# ---------------------------------------------------------------------------
# 因子 5: 日内振幅因子 — (high - low) / close
# ---------------------------------------------------------------------------

def compute_intraday_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """计算日内振幅因子。

    range = (high - low) / close

    值域 >= 0,越大表示日内波动越剧烈。
    """
    close_safe = close.replace(0, np.nan)
    rng = (high - low) / close_safe
    return rng


# ---------------------------------------------------------------------------
# 因子 6: 量价相关因子
# ---------------------------------------------------------------------------

def compute_volume_price_corr(
    close: pd.DataFrame, volume: pd.DataFrame, window: int = 20
) -> pd.DataFrame:
    """计算量价相关性因子。

    对每只股票,计算过去 window 日内成交量变化率与价格变化率的相关性。
    正值表示量价同向(放量上涨/缩量下跌),负值表示量价背离。

    返回: DataFrame, 行=日期, 列=股票代码
    """
    ret = close.pct_change(fill_method=None)
    vol_chg = volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

    def rolling_corr(
        ret_col: pd.Series, vol_col: pd.Series, w: int
    ) -> pd.Series:
        return ret_col.rolling(window=w, min_periods=max(5, w // 2)).corr(vol_col)

    result = pd.DataFrame(index=close.index, columns=close.columns, dtype=np.float64)
    for col in close.columns:
        result[col] = rolling_corr(ret[col], vol_chg[col], window)
    return result


# ---------------------------------------------------------------------------
# 辅助: 前向收益率(用于因子评价)
# ---------------------------------------------------------------------------

def compute_forward_returns(close: pd.DataFrame, period: int = 1) -> pd.DataFrame:
    """计算前向收益率。

    forward_return = (close_{t+period} - close_t) / close_t

    Args:
        close: 收盘价 DataFrame
        period: 前向天数(1 = 次日)

    返回: DataFrame, 行=日期, 列=股票代码
    """
    fwd = close.shift(-period)
    ret = (fwd - close) / close
    return ret


# ---------------------------------------------------------------------------
# 汇总: 一次性计算所有因子
# ---------------------------------------------------------------------------

def compute_all_factors(
    daily_data: dict[str, pd.DataFrame] | None = None,
    data_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """计算所有因子,返回 {factor_name: DataFrame}。

    因子列表:
      - example_factor: 示例因子
      - momentum_5d, momentum_10d, momentum_20d: 动量因子
      - volatility_5d, volatility_10d, volatility_20d: 波动率因子
      - buy_sell_imbalance: 买卖失衡因子
      - intraday_range: 日内振幅因子
      - volume_price_corr_20d: 量价相关因子
      - forward_return_1d: 前向收益率(非因子,用于评价)

    总计: 1 + 3 + 3 + 1 + 1 + 1 + 1 = 11 列
    """
    if daily_data is None:
        daily_data = load_daily_data(data_dir)

    close = daily_data["close"]
    amount = daily_data["amount"]
    volume = daily_data["volume"]
    high = daily_data["high"]
    low = daily_data["low"]
    buy_vol = daily_data["buy_volume"]
    sell_vol = daily_data["sell_volume"]

    print("计算示例因子...")
    factors = {"example_factor": compute_example_factor(amount)}

    print("计算动量因子...")
    factors.update(compute_momentum(close))

    print("计算波动率因子...")
    factors.update(compute_volatility(close))

    print("计算买卖失衡因子...")
    factors["buy_sell_imbalance"] = compute_buy_sell_imbalance(buy_vol, sell_vol)

    print("计算日内振幅因子...")
    factors["intraday_range"] = compute_intraday_range(high, low, close)

    print("计算量价相关因子...")
    factors["volume_price_corr_20d"] = compute_volume_price_corr(close, volume)

    print("计算前向收益率...")
    factors["forward_return_1d"] = compute_forward_returns(close)

    return factors


def save_factors(
    factors: dict[str, pd.DataFrame],
    out_dir: Path | None = None,
) -> None:
    """保存因子到 CSV。"""
    out_dir = out_dir or (OUTPUT_DIR / "factors")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in factors.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, float_format="%.6f")
        print(f"  保存: {path}  shape={df.shape}")
