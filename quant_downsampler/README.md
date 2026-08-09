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

依赖已完整列入 `requirements.txt` 和 `pyproject.toml`：NumPy、pandas、SciPy、Matplotlib、scikit-learn、Joblib、PyTorch。

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

# 因子时间稳定性、区块 bootstrap、波动状态与相关性
python -m scripts.run_factor_robustness

# 四组回测：修正版/Naive × 收盘/开盘
python -m scripts.run_backtest --lookback 60 --top-n 10

# 修正版组合的换手约束与对称双边成本压力
python -m scripts.run_backtest_robustness

# 单模型研究训练；默认选前 5 只股票，安全写入 output/lstm_runs/nonflat_binary
python -m scripts.run_lstm --stocks 5 --epochs 12 --seed 42

# 正式三组件训练；融合参数冻结后才读取测试日期
python -m scripts.run_lstm_full --stocks 5 --epochs 12 --device cpu

# 严格重载三组件全窗口模型，并从原始分钟表复算测试结果
python -m scripts.evaluate_lstm_full --split test

# 同窗口 Logistic Regression / HistGradientBoosting 强基线
python -m scripts.run_lstm_baselines --stocks 5

# 只在验证集选择 LSTM/HistGradientBoosting 概率融合权重
python -m scripts.run_lstm_hybrid --overwrite

# 因果基线、组件消融、概率校准与 walk-forward 协议
python -m scripts.run_lstm_research --evaluate-components --calibrate --overwrite

# 不重训组件，只在验证集重新选择融合参数
python -m scripts.adjust_lstm_fusion --overwrite

# 将冻结概率接入下一分钟收益并做成本压力
python -m scripts.run_lstm_strategy --side long_short
python -m scripts.run_lstm_strategy \
  --side long_only --out-dir ../output/lstm_strategy_long_only

# 验收全部产物；LSTM 会逐样本重放并核对概率
python -m scripts.validate_outputs

# 当前共有 56 项单元测试
python -m unittest discover -s tests -v
```

也可安装为命令行工具：

```bash
python -m pip install -e .
qd-downsample --overwrite
qd-factors
qd-backtest
qd-factor-robustness
qd-backtest-robustness
qd-lstm --stocks 5
qd-lstm-full --stocks 5 --epochs 12 --device cpu
qd-lstm-full-eval --split test
qd-lstm-baselines --stocks 5
qd-lstm-hybrid --overwrite
qd-lstm-research --evaluate-components --calibrate --overwrite
qd-lstm-adjust --overwrite
qd-lstm-strategy --side long_short
```

## 5. 因子

- 示例因子：`log(std([MA1, MA5, MA10, MA20] of amount))`
- 5 日动量
- 主买主卖失衡：`(buy_volume-sell_volume)/(buy_volume+sell_volume)`
- 日内振幅：`(high-low)/close`

仅构建三个新增因子，严格满足“另构建 1 至 3 个因子”。每日截面 IC 使用下一交易日复权收盘收益。`ICIR=mean(IC)/std(IC)`，`IR=sqrt(252)×ICIR`；Rank 指标同理。负 IC 表示因子反向有效，综合排序使用 `|ICIR|`。

### 5.1 因子稳健性与方向漂移

`run_factor_robustness` 额外输出月度、季度、半年和 60 日滚动 IC/Rank IC、5 日区块 bootstrap 置信区间、高低目标期截面波动状态、因子相关矩阵，以及只用历史 IC 推断方向的 prequential 命中率。

| 因子 | 全样本平均 IC | 区块 bootstrap 95% CI | 2026Q2 平均 IC |
|---|---:|---:|---:|
| 示例因子 | -0.0780 | [-0.0959, -0.0585] | 0.0227 |
| 5 日动量 | -0.0276 | [-0.0445, -0.0099] | 0.0299 |
| 主买主卖失衡 | -0.0479 | [-0.0610, -0.0336] | -0.0320 |
| 日内振幅 | -0.0736 | [-0.0927, -0.0541] | 0.0056 |

全样本显著不等于方向永久稳定：最近季度已有三个因子转正，截至 2026-06-29 的 60 日滚动 IC 也转正。组合应使用严格滞后的滚动 IC 动态定向，并在接近 0 或翻转时降权。因子间平均截面 Spearman 相关性的最大绝对值为 0.318，当前四因子不存在特别严重的重复暴露。

现有数据没有真实行业或市值暴露表，因此中性化被明确标记为 `skipped`；模块提供连续/分类暴露接口，但不会用股票代码或随机分组伪造行业数据。波动状态来自已经实现的目标期收益，只是事后压力诊断，不能作为交易时可见特征。

## 6. 回测信息时序

Naive 版本严格复现题目所指出的问题：用尚未发生的次日收益计算当日 IC，因此仅作为前视偏差对照。

修正版在交易日 t 使用 t-1 因子，并只使用截至 t-2 已实现收益计算的过去 60 日平均 IC。新建仓要求执行日成交量大于 0；持仓后停牌的股票不能假设卖出，而是标记为 locked。首次建仓不收卖出手续费。

### 6.1 换手约束与成本压力

原修正版在题目规定的卖出 5 bps、买入 0 口径下累计收益 37.06%，但日均卖出换手为 72.04%。换手约束版本使用 Top 20 缓冲区、每日最多替换 3 只，并只允许历史 20 日成交额非底部 20%、历史波动非顶部 10% 的股票新建仓；这些过滤变量全部滞后一天。

`output/backtest/` 保留了题目原始的期末持仓市值口径，不强制在样本末清仓，所以其中累计收益为 37.13%。`backtest_robustness` 为了让不同成本情景可比，补计末期全部退出并用标准的日均收益/日波动计算 Sharpe，因此下表采用 37.06% 和 Sharpe 1.51。两者差异只来自末期退出费用。

| 方案 | 题目费用口径累计收益 | 日均卖出换手 | 最大回撤 | 对称单边盈亏平衡成本 |
|---|---:|---:|---:|---:|
| 原修正版 | 37.06% | 72.04% | -20.42% | 10.32 bps |
| 换手约束 | 22.82% | 30.04% | -18.01% | 14.73 bps |

对称双边成本为 10 bps 时，两者累计收益分别为 1.30% 和 8.28%；20 bps 时分别降至 -32.35% 和 -8.49%。成本包含初始建仓、调仓及样本末全部退出；这里固定持仓路径后线性重定价，没有模拟冲击、涨跌停、成交容量或滑点，不能解释为实盘可实现收益。

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

### 7.2 强基线与显著性

传统模型使用与全窗口 LSTM 完全相同的 5 只股票、日期切分和 5,900 个测试窗口。每个 60 分钟 × 25 特征序列被汇总为 180 维：最后时点、全窗口/最近 10 分钟/最近 5 分钟的均值与标准差，再加 5 维股票 one-hot。缺失值处理和标准化只拟合训练集；候选参数只在验证集选择，冻结后才读取测试集。

| 模型 | Accuracy | Macro F1 | Brier | NLL |
|---|---:|---:|---:|---:|
| 训练先验多数类 | 43.68% | 20.27% | 0.6507 | 1.0755 |
| Logistic Regression | 46.58% | 42.83% | 0.6213 | 1.0238 |
| HistGradientBoosting | 46.80% | 46.02% | 0.6050 | 0.9927 |
| 原冻结 LSTM | 46.88% | 45.87% | 0.6079 | 0.9979 |
| LSTM + HistGB hybrid | 47.31% | 46.39% | 0.6032 | 0.9906 |

原 LSTM 只比树模型高 0.085 个百分点 Accuracy，而树模型的 Macro F1、Brier 和 NLL 略优。逐样本 McNemar 精确检验 `p=0.894`，差异不显著。因此项目只能说 LSTM 优于简单多数类和因果规则，不能声称已经证明时序网络优于强树基线。

跨模型 hybrid 只在验证集选择线性概率权重，冻结为 LSTM 0.49、HistGradientBoosting 0.51 后才读取测试集。它的 balanced/strict 档准确率—覆盖率为 58.28%/29.37% 和 66.95%/9.90%；不过 hybrid 相对原 LSTM 和树模型的配对检验分别为 `p=0.269` 和 `p=0.165`，改善仍不显著。固定测试段此前也已被查看，因此这只是统一口径诊断，不是新盲测。

### 7.3 组件消融、校准与融合调整

冻结组件的结构消融结果如下。它复用已经训练的 joint/two-stage 分支，不是删除特征后重新训练的消融：

| 结构 | Accuracy | Macro F1 |
|---|---:|---:|
| joint only | 46.61% | 44.79% |
| two-stage only | 46.47% | 46.11% |
| 原融合 | 46.88% | 45.87% |

验证集温度缩放得到 `T=1.32915`，测试 top-label ECE 从 1.93% 降至 1.52%，NLL 从 0.997853 略降至 0.997734，但 Brier 从 0.607875 略升至 0.608260。因此校准概率作为可选产物保存，没有覆盖原概率。

针对原发布包，还进行了不重训组件的验证集融合调整：以 Macro F1 优先选择时，move bias 从 -0.05 调至 -0.16，joint weight 从 0.50 调至 0.34。测试 Accuracy 为 46.97%，Macro F1 为 46.76%，但 Brier 恶化为 0.6117；与 HistGradientBoosting 的配对检验仍不显著（`p=0.770`）。这说明调整改善了类别均衡性，但没有建立 LSTM 的显著结构优势。新的 `run_lstm_full` 默认也使用 `macro_f1_then_accuracy` 作为验证集融合目标。

### 7.4 概率信号到策略闭环

`run_lstm_strategy` 用预测时可见的 `P(up)-P(down)` 和验证集冻结档位生成仓位，按股票横截面归一化后，精确连接 `window_end → target_time` 的下一分钟收益；真实标签和收益只用于事后评价。组合先在同一分钟跨股票聚合，再沿时间复利，换手包含入场、调仓、数据间断平仓和最终退出。

下表选取等权/置信加权结果接近的置信加权版本，成本为单边 5 bps：

| 档位与方向 | 10 日净收益 | 日均总换手 | 盈亏平衡单边成本 |
|---|---:|---:|---:|
| all long-short | -13.27% | 165.16 | 4.14 bps |
| balanced long-short | -5.82% | 114.00 | 4.47 bps |
| strict long-short | 5.67% | 40.58 | 6.36 bps |
| all long-only | -9.03% | 129.51 | 4.27 bps |
| balanced long-only | -0.41% | 54.46 | 4.92 bps |
| strict long-only | 2.54% | 14.92 | 6.68 bps |

只有 strict 档在该线性成本假设下仍为正，说明 flat/低置信时不交易有实际价值；同时也说明优势对成本非常敏感。测试期只有已经查看过的 10 个交易日，且没有冲击、成交约束和融券成本，不能把这些累计收益或由其得到的年化 Sharpe 外推为长期表现。

更好的分类分数不必然带来更好的策略：hybrid 的 strict 置信加权 long-short/long-only 在同一单边 5 bps 口径下仅为 0.09%/1.70%，低于原 LSTM 的 5.67%/2.54%。因此最终选择模型必须同时看概率质量、换手和成本后收益，不能只按 Accuracy 排名。

### 7.5 Walk-forward 与单折重训边界

项目已按“180 个训练日 + 10 个验证日 + 10 个测试日、步长 10 日”的扩展窗口生成 11 折协议。每一折都必须独立重训标准化、三个组件、融合参数和置信阈值，静态模型按日期切片不算 walk-forward。

目前只完成第 1 折严格重训示范：训练 2025-04-01—2025-12-23，验证 2025-12-24—2026-01-08，测试 2026-01-09—2026-01-22。该折 5,900 个测试窗口的 Accuracy 为 54.46%、Macro F1 为 46.54%、Brier 为 0.5421、NLL 为 0.8930，多数类基线为 46.95%。这证明协议可以实际执行，不证明跨时期稳定；其余 10 折尚未独立训练，而且相关历史日期在此前研究中可能已经被研究者接触。

### 7.6 复现与审计

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

本轮开发过程中固定测试日期段已经被多次查看，因此这些数字是统一口径下的样本外诊断，不再是严格盲测。第 1 个 walk-forward 折虽保证模型训练流程的时间隔离，也不能消除研究者过去接触相应历史日期的事实。后续更有力的确认应完成全部预定义折，并最终使用 2026-06-30 之后的新数据。

## 8. 输出

```text
output/
├── daily/                 # 11 张日频表
├── minute/                # 11 × 302 张分钟表
├── factors/               # 因子、每日 IC、分层结果、图表、自动报告
├── factor_robustness/     # 滚动/分段 IC、bootstrap、状态与方向漂移
├── backtest/              # 净值、收益、换手、选股、指标、图表、自动报告
├── backtest_robustness/   # 换手约束、固定持仓成本压力与盈亏平衡成本
├── lstm/                  # 非平价条件方向模型及逐样本结果
├── lstm_full/             # 全窗口三组件模型、选择性预测档位及逐样本结果
├── lstm_baselines/        # Logistic/HistGradientBoosting 强基线
├── lstm_hybrid/           # 验证集冻结的 LSTM/HistGradientBoosting 概率融合
├── lstm_research/         # 因果基线、消融、校准与 walk-forward 协议
├── lstm_adjusted/         # 仅使用验证集重新选择的融合包
├── lstm_strategy/         # long-short 概率策略与成本压力
├── lstm_strategy_long_only/ # long-only 概率策略与成本压力
├── lstm_hybrid_strategy/  # hybrid long-short 策略诊断
├── lstm_hybrid_strategy_long_only/ # hybrid long-only 策略诊断
├── lstm_walk_forward/     # 已独立重训的折及 OOS 聚合
└── validation_report.json # 全工程严格验收结果
```

所有研究报告由对应脚本根据当前数据自动生成，避免手工报告与产物不一致。
