# 兆信基金 Task 1:逐笔成交降采样

> 把 300 只股票 × ~300 个交易日的逐笔成交数据(Tick)降采样为日频和分钟频 K 线,
> 含 11 个基础指标(OHLCV + 成交笔数/额 + 主买主卖)。

---

## 1. 项目结构

```
项目/
├── quant_downsampler/
│   ├── qd/
│   │   ├── __init__.py
│   │   ├── config.py        # 路径、时间段、指标定义
│   │   ├── time_utils.py    # 9 位时间字段解析
│   │   ├── data_loader.py   # CSV 读取与预处理
│   │   ├── adjfactor.py     # 复权因子
│   │   ├── daily.py         # 日频降采样
│   │   ├── minute.py        # 分钟频降采样
│   │   └── pipeline.py      # 主流程
│   └── scripts/
│       ├── run_step1.py     # 入口脚本
│       └── smoke_test.py    # 冒烟测试
└── output/                  # 运行后生成
    ├── daily/               # 11 张日频宽表
    └── minute/              # 11 个文件夹 × ~300 个日期
```

## 2. 安装依赖

```bash
pip install pandas numpy
```

## 3. 运行

```bash
# 跑全部数据
cd /Users/ryan/Desktop/大一暑/兆信基金-材料上传/项目
python -m scripts.run_step1

# 只跑某几天
python -m scripts.run_step1 --dates 20250401 20250402 20250403

# 跑冒烟测试(1 只股票 1 天)
python -m scripts.smoke_test
```

## 4. 输出格式

### 日频(daily/ 共 11 张 CSV)

行 = `YYYY-MM-DD`,列 = 股票代码(6 位数字,共 300 列)。

| 文件 | 含义 |
|---|---|
| `open.csv` | 当日第一笔成交价 |
| `high.csv` | 当日最高成交价 |
| `low.csv` | 当日最低成交价 |
| `close.csv` | 当日最后一笔成交价 |
| `volume.csv` | 当日总成交量(股) |
| `trade_count.csv` | 当日成交笔数 |
| `amount.csv` | 当日总成交额(元) |
| `buy_volume.csv` | BSFlag==0 的成交量 |
| `sell_volume.csv` | BSFlag==1 的成交量 |
| `buy_amount.csv` | BSFlag==0 的成交额 |
| `sell_amount.csv` | BSFlag==1 的成交额 |

OHLC 已按 readme 第 8 条用前一日 close 填充(停牌日)。
成交类指标无数据时填 0。

### 分钟频(minute/ 共 11 个文件夹)

每个 metric 一个文件夹,内含 `YYYYMMDD.csv` 形式的日文件。
每张表行 = 分钟 bar 标签(9:30, 9:31, ..., 11:30, 13:00, ..., 14:56,共 237 行),
列 = 股票代码(300 列)。

```
minute/
├── open/20250401.csv
├── open/20250402.csv
├── ...
├── volume/20250401.csv
├── ...
└── sell_amount/20250401.csv
```

连续竞价 tick 按所在分钟归入对应 bar,集合竞价 tick (BSFlag==2) 不进入 bar。
OHLC 已按 readme 第 9 条用前 1 分钟 close 填充(9:30 缺失则用上一交易日 close)。

## 5. 设计要点

### 5.1 时间字段

`Time` 字段是 9 位整数,格式 `HHMMSScc`(百分秒):
- `92500740` = 9:25:00.740
- `150001110` = 15:00:01.110

### 5.2 集合竞价 vs 连续竞价

| 窗口 | 时间 | 性质 | BSFlag |
|---|---|---|---|
| 集合竞价 | 9:15-9:25 | 统一价成交 | 2 |
| 连续竞价 | 9:30-11:30 | 逐笔撮合 | 0/1 |
| 连续竞价 | 13:00-14:57 | 逐笔撮合 | 0/1 |
| 集合竞价 | 14:57-15:00 | 统一价成交 | 2 |

分钟 K 线只覆盖连续竞价 237 个 bar(9:30-11:30 共 120,13:00-14:56 共 117)。
集合竞价的 tick 用作当日 open/close(因为它就是"第一笔/最后一笔"成交),
但不进入分钟 bar。

### 5.3 复权因子

`adjfactor.pkl` 是 300 × 302 的 DataFrame(columns=真实股票代码,index=日期)。
**但本项目的数据股票代码是 6 位数字混淆版,无法与 adjfactor 直接对应**。

因此默认 **不应用复权**(`qd/config.py: APPLY_ADJFACTOR = False`)。
如果用户拿到了代码映射,可以在 `config.py` 填入:

```python
_STOCK_CODE_MAPPING = {
    "000012": "600705.SH",  # data 代码 -> adjfactor 代码
    ...
}
APPLY_ADJFACTOR = True
```

### 5.4 内存与性能

- 全量数据约 4.68 GB,300 只股票 × ~300 个交易日
- 单只股票单日 ≈ 2-3000 tick(约 100 KB)
- 一天所有股票 ≈ 30 MB
- 推荐 `--days-per-chunk 5`(默认),单进程内存约 200-300 MB
- 全量跑完约 30-60 分钟(取决于机器)

## 6. 已知限制

1. **股票代码混淆**:无法自动应用复权,需要用户自行维护代码映射。
2. **9:30 缺失 bar 跨日填充**:用上一交易日 close 填充,但跨日时只覆盖 close,
   不会跨日回溯早于 9:30 的 tick。
3. **集合竞价的 close 处理**:14:57 集合竞价的 tick 不会进入 14:56 bar,
   也不会进入任何连续竞价 bar——它只参与日级 close 聚合。

## 7. 后续步骤(本任务未实现)

- Task 2:因子构建(本任务输出可直接用)
- Task 3:因子评价(IC/IR/ICIR)
- Task 4-6:策略回测、LSTM 预测

## 8. 验收清单

- [ ] `python -m scripts.smoke_test` 全部通过
- [ ] `output/daily/open.csv` 行数 = 处理日期数
- [ ] `output/daily/close.csv` 列数 = 处理日期内所有出现过的股票数
- [ ] `output/minute/volume/20250401.csv` 形状 = (237, N)
- [ ] BSFlag 拆分后:buy_volume + sell_volume ≤ volume(差额=BSFlag==2 的量)
