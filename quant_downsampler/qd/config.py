"""路径与全局配置。

数据来源布局:
    {BASE_DIR}/data/TRADE/YYYYMMDD/XXXXXX.csv
    {BASE_DIR}/adjfactor.pkl
    {BASE_DIR}/readme.md

输出布局(见 readme):
    {OUTPUT_DIR}/daily/{metric}.csv          # 11 张日频宽表
    {OUTPUT_DIR}/minute/{metric}/{YYYYMMDD}.csv  # 11 个文件夹,内含每日宽表
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
# __file__ = .../项目/quant_downsampler/qd/config.py
# parents[0] = qd/, parents[1] = quant_downsampler/, parents[2] = 项目/
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "TRADE"
ADJFACTOR_PATH = BASE_DIR / "adjfactor.pkl"
OUTPUT_DIR = BASE_DIR / "output"

# 中间缓存(可重复运行,减少 IO)
CACHE_DIR = OUTPUT_DIR / "_cache"

# ---------------------------------------------------------------------------
# 指标定义
# ---------------------------------------------------------------------------
# 日频 + 分钟频共 11 个指标。
# 名称与 readme 描述的中文指标一一对应。
METRICS: tuple[str, ...] = (
    "open",       # 开盘价(实际价格,不是 ×100 后的整数)
    "high",       # 最高价
    "low",        # 最低价
    "close",      # 收盘价
    "volume",     # 成交量(股)
    "trade_count",# 成交笔数(tick 数)
    "amount",     # 成交额(元,Price/100 * Volume)
    "buy_volume", # 主买量(BSFlag==0 的成交量)
    "sell_volume",# 主卖量(BSFlag==1 的成交量)
    "buy_amount", # 主买额
    "sell_amount",# 主卖额
)

# ---------------------------------------------------------------------------
# 时间段定义(readme 第 6、7 条)
# ---------------------------------------------------------------------------
# 集合竞价:
#   - 开盘: 9:15:00 - 9:25:00(9:25 这一刻产生开盘价)
#   - 尾盘: 14:57:00 - 14:59:59(14:57 这一刻产生收盘价)
# 连续竞价:
#   - 上午: 9:30:00 - 11:30:00
#   - 下午: 13:00:00 - 14:57:00
#
# 分钟 K 线(连续竞价)覆盖:
#   - 9:30, 9:31, ..., 11:30(120 个 bar)
#   - 13:00, 13:01, ..., 14:56(117 个 bar)
# 共 237 个 bar。
# 11:30 这一刻的 tick 归入 11:30 那个 bar(11:30:00-11:30:59)。
# 14:56 这一刻的 tick 归入 14:56 那个 bar(14:56:00-14:56:59)。
# 14:57 起的 tick 全部视为集合竞价(BSFlag==2),不进入连续竞价 bar。

MORNING_BARS = 120   # 9:30 - 11:30
AFTERNOON_BARS = 117 # 13:00 - 14:56
TOTAL_BARS = MORNING_BARS + AFTERNOON_BARS  # 237

# 集合竞价窗口(用于过滤 tick)
CALL_AUCTION_WINDOWS = (
    (9 * 3600 + 15 * 60,           9 * 3600 + 25 * 60),          # 9:15-9:25
    (14 * 3600 + 57 * 60,          15 * 3600 + 0 * 60),          # 14:57-15:00
)

# ---------------------------------------------------------------------------
# 复权因子(readme 补丁说明)
# ---------------------------------------------------------------------------
# 复权价格 = 原始价格 × (当日 adjfactor / 上市首日 adjfactor)
# 简称:adj_price = price * (adj_today / adj_first)
#
# 但!本数据集的股票代码在 data/TRADE/ 中是 6 位数字(如 000012),
# 而在 adjfactor.pkl 中是真实代码(如 600705.SH)——readme 已说明
# "股票代码已做混淆处理"。所以我们无法直接建立映射。
#
# 默认不在本步骤应用复权(APPLY_ADJFACTOR=False),输出未复权价格。
# 如果用户拿到了代码映射,可以在 config_override 中开启,
# 并提供 _STOCK_CODE_MAPPING 字典把 data 目录的代码映射到 adjfactor 的代码。
APPLY_ADJFACTOR = False
_STOCK_CODE_MAPPING: dict[str, str] = {}  # data_code -> adjfactor_code

# ---------------------------------------------------------------------------
# 运行参数
# ---------------------------------------------------------------------------
# 一次处理的日期数(防止内存爆炸)。300 只股票 × 1 天 ≈ 30 MB tick。
DAYS_PER_CHUNK = 5
# 写盘时是否覆盖已有输出
OVERWRITE = False
