# Quant Downsampler：完整量化研究流水线

## 1. 任务覆盖

1. 将 300 只股票、302 个交易日的逐笔成交降采样为日频和分钟频 11 个字段。
2. 构建任务书示例因子。
3. 另构建三个因子并计算分层、IC、IR、ICIR、Rank IC、Rank IR、Rank ICIR。
4. 以 1,000 万元构建每日 Top 10 组合，分别进行收盘价和次日开盘价调仓，并对比含前视偏差的原始方法与修正版。
5. 使用分钟数据训练端到端 LSTM：主模型预测下一分钟跌/平/涨；同时保留非平价条件下的方向二分类作为诊断。

## 2. 数据口径

### 2.1 股票代码和复权

数据文件名是六位代码，`adjfactor.pkl` 列名是带 `.SH/.SZ` 的完整代码。二者去掉市场后缀后 300 只全部一一对应。

价格复权公式：

```text
adjusted_price[date] = raw_price / 100 × adjfactor[date] / adjfactor[first_valid_date]
```

归一化使样本首日价格保持原值，同时修正后续分红配股造成的跳变。成交额按真实成交价计算，不进行复权。

### 2.2 日频

每个字段一张 302×300 宽表，行是 `YYYY-MM-DD`，列是完整股票代码：

- `open/high/low/close`
- `volume/trade_count/amount`
- `buy_volume/sell_volume/buy_amount/sell_amount`

停牌日的 OHLC 全部使用前一交易日 close；成交类字段填 0。

### 2.3 分钟频

每个字段一个文件夹，每个交易日一张 242×300 表。索引为完整时间戳：

- `09:30:00`–`11:30:00`：121 行
- `13:00:00`–`15:00:00`：121 行

连续竞价无成交分钟的 OHLC 全部使用前一分钟 close。`14:57–14:59` 是尾盘集合竞价等待行，价格保持 NaN、成交字段为 0；实际撮合成交记录在 `15:00`。开盘集合竞价只参与日频 open，不进入从 09:30 开始的分钟表。

## 3. 环境

要求 Python 3.11 或更高版本。

```bash
python -m pip install -r requirements.txt
```

依赖已完整列入 `requirements.txt` 和 `pyproject.toml`：NumPy、pandas、SciPy、Matplotlib、scikit-learn、PyTorch。

## 4. 运行与验收

所有命令均从本目录执行。

```bash
# 快速验收，不写全量输出
python -m scripts.smoke_test

# 全量重建日频和分钟频
python -m scripts.run_step1 --overwrite --days-per-chunk 5

# 断点续跑：自动检查 11 个分钟字段和 242 行完整性
python -m scripts.run_step1

# 因子构建与评价
python -m scripts.run_factor_eval

# 四组回测：修正版/Naive × 收盘/开盘
python -m scripts.run_backtest --lookback 60 --top-n 10

# 单模型研究训练；默认选前 5 只股票，安全写入 output/lstm_runs/nonflat_binary
python -m scripts.run_lstm --stocks 5 --epochs 12 --seed 42

# 正式三组件训练；融合参数冻结后才读取测试日期
python -m scripts.run_lstm_full --stocks 5 --epochs 12 --device cpu

# 严格重载三组件全窗口模型，并从原始分钟表复算测试结果
python -m scripts.evaluate_lstm_full --split test

# 验收全部产物；LSTM 会逐样本重放并核对概率
python -m scripts.validate_outputs
```

也可安装为命令行工具：

```bash
python -m pip install -e .
qd-downsample --overwrite
qd-factors
qd-backtest
qd-lstm --stocks 5
qd-lstm-full --stocks 5 --epochs 12 --device cpu
qd-lstm-full-eval --split test
```

## 5. 因子

- 示例因子：`log(std([MA1, MA5, MA10, MA20] of amount))`
- 5 日动量
- 主买主卖失衡：`(buy_volume-sell_volume)/(buy_volume+sell_volume)`
- 日内振幅：`(high-low)/close`

仅构建三个新增因子，严格满足“另构建 1 至 3 个因子”。每日截面 IC 使用下一交易日复权收盘收益。`ICIR=mean(IC)/std(IC)`，`IR=sqrt(252)×ICIR`；Rank 指标同理。负 IC 表示因子反向有效，综合排序使用 `|ICIR|`。

## 6. 回测信息时序

Naive 版本严格复现题目所指出的问题：用尚未发生的次日收益计算当日 IC，因此仅作为前视偏差对照。

修正版在交易日 t 使用 t-1 因子，并只使用截至 t-2 已实现收益计算的过去 60 日平均 IC。新建仓要求执行日成交量大于 0；持仓后停牌的股票不能假设卖出，而是标记为 locked。首次建仓不收卖出手续费。

## 7. LSTM

输入覆盖全部 11 个分钟字段。基础分支使用相对价格、收益率和 `log1p` 成交特征；增强分支再加入价差、收盘位置、主买主卖失衡、滚动收益/波动、成交变化与时段特征。所有特征只依赖窗口结束时及以前的信息，窗口严格限制在单个交易日及同一连续竞价时段内。

若输入窗口结束于分钟 t，标签严格比较 `close[t+1]` 与 `close[t]`。训练/验证/测试按日期顺序切分，标准化仅拟合训练集；固定随机种子，使用验证早停，并保存模型结构、权重、类别语义、标准化参数、特征顺序、日期切分与运行库版本。

### 7.1 两种评价口径

- `output/lstm/`：只在下一分钟价格确实变化的样本上评价 down/up。测试准确率为 68.10%，多数类基线 52.21%，但只覆盖 3,323/5,900 = 56.32% 的合法测试窗口。由于“下一分钟是否平价”预测时未知，这个结果必须称为条件方向准确率。
- `output/lstm_full/`：在全部 5,900 个合法测试窗口上预测 down/flat/up，不用未来标签筛样本。这里的分母是 5 只股票 × 10 个测试交易日 × 每日 118 个完成 60 分钟预热且特征有效的窗口，不是全部 242 行。测试准确率 46.88%，Macro F1 45.87%，多数类基线 43.68%。

两种准确率的标签空间和分母不同，不能直接比较。全窗口模型还保存验证集定义的选择性预测档位：

| 档位 | 验证集规则 | 测试覆盖 | 测试准确率 |
|---|---:|---:|---:|
| 全部窗口 | 不筛选 | 100.00% | 46.88% |
| balanced | 置信度位于验证集前 30% | 30.85% | 55.60% |
| strict | 置信度位于验证集前 10% | 9.68% | 64.10% |

选择性档位只使用预测时可见的最大类别概率，不使用未来涨跌或平价标签。准确率提高的代价是覆盖率下降。

### 7.2 复现与审计

新训练默认写入 `output/lstm_runs/<target_mode>/`；若目录非空会拒绝覆盖，只有显式传入 `--overwrite` 才会替换。复训已发布的条件二分类配置时，建议写入新目录：

```bash
python -m scripts.run_lstm \
  --stocks 5 --epochs 12 --seed 42 \
  --target-mode nonflat_binary \
  --feature-set legacy --scaler global --model-version legacy --no-stock-id \
  --out-dir ../output/lstm_reproduced
```

`test_predictions.csv` 包含股票、日期、窗口结束分钟、目标分钟、真实标签、预测标签和逐类概率。`scripts.validate_outputs` 不只检查 JSON：它会严格加载权重，从当前分钟表重建相同窗口并逐样本核对概率。

正式全模型训练入口是 `python -m scripts.run_lstm_full`。它先仅加载训练/验证日期，分别训练 direction、movement、joint 三个组件；随后在验证集网格选择 move bias 与融合权重、固定 balanced/strict 置信阈值，最后才首次加载测试日期并生成最终结果。默认写入 `output/lstm_runs/full/`，不会覆盖发布模型。

本轮开发过程中固定测试日期段已经被多次查看，因此这些数字是统一口径下的样本外诊断，不再是严格盲测。后续无偏性能确认必须使用 2026-06-30 之后的新数据或预先锁定的滚动样本外区间。

## 8. 输出

```text
output/
├── daily/                 # 11 张日频表
├── minute/                # 11 × 302 张分钟表
├── factors/               # 因子、每日 IC、分层结果、图表、自动报告
├── backtest/              # 净值、收益、换手、选股、指标、图表、自动报告
├── lstm/                  # 非平价条件方向模型及逐样本结果
├── lstm_full/             # 全窗口三组件模型、选择性预测档位及逐样本结果
└── validation_report.json # 全工程严格验收结果
```

所有研究报告由对应脚本根据当前数据自动生成，避免手工报告与产物不一致。
