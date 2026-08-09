# 文件导览与协作说明

这份文档帮助新队友快速定位代码。项目采用“核心逻辑放在 `qd/`，命令入口放在 `scripts/`，验收逻辑放在 `tests/`”的结构。

如果目标是准备答辩而不是立即改代码，先读 [`TEAM_DEFENSE_GUIDE.md`](TEAM_DEFENSE_GUIDE.md)。它用非代码视角说明项目主线、口径、关键数字和队内分工。

## 核心模块

| 文件 | 主要职责 | 主要输入 | 主要输出 |
|---|---|---|---|
| `qd/config.py` | 统一管理路径、11 个字段、交易时段和复权开关 | 项目目录结构 | 供其他模块导入的配置常量 |
| `qd/data_loader.py` | 枚举交易日与股票、加载单只股票逐笔数据 | `data/TRADE/YYYYMMDD/*.csv` | 规范化的 pandas DataFrame |
| `qd/adjfactor.py` | 股票代码映射、读取复权因子、计算归一化复权比例 | `adjfactor.pkl` | 完整股票代码和复权价格比例 |
| `qd/time_utils.py` | 解析整数时间、生成 242 个分钟标签、判断集合竞价 | 逐笔成交时间 | 秒数、分钟标签和竞价标记 |
| `qd/daily.py` | 生成日频 OHLC、量、额、笔数和主买主卖字段 | 单日全部逐笔数据 | 11 张日频宽表的单日数据 |
| `qd/minute.py` | 生成分钟频 11 字段，处理无成交分钟和尾盘集合竞价 | 单日逐笔数据 | 每日 242×股票数的分钟表 |
| `qd/pipeline.py` | 组织降采样流程、断点续跑和落盘 | 日期列表、原始数据 | `output/daily/` 和 `output/minute/` |
| `qd/factors.py` | 构建四个因子及下一日收益 | 日频表 | 因子矩阵和 forward return |
| `qd/evaluation.py` | 因子分层、Pearson/Spearman IC 与统计报告 | 因子、未来收益 | CSV、图表和因子报告 |
| `qd/factor_robustness.py` | 分段/滚动 IC、区块 bootstrap、波动状态、相关性和方向稳定性；提供真实暴露中性化接口 | 保存的因子与 forward return | `output/factor_robustness/` |
| `qd/backtest.py` | 历史 IC 加权、Top 10 组合、费用、停牌锁仓和绩效 | 因子、开收盘价、成交量 | 净值、收益、换手、选股与指标 |
| `qd/backtest_robustness.py` | Top 20 缓冲、替换上限、滞后流动性/波动过滤及固定持仓成本重定价 | 日频与因子产物 | 换手约束路径、成本曲线和盈亏平衡成本 |
| `qd/lstm_model.py` | 分钟序列特征、样本构造、基础 LSTM 训练与评价 | 分钟频表 | 模型、概率、指标和训练历史 |
| `qd/lstm_full.py` | 加载三组件模型并完成全窗口融合预测 | 保存的组件权重与分钟表 | down/flat/up 概率和评价结果 |
| `qd/lstm_full_training.py` | 训练 direction、movement、joint 并在验证集选融合参数 | 训练/验证日期分钟表 | 三组件模型、阈值与完整配置 |
| `qd/lstm_baselines.py` | 将同一 60 分钟窗口汇总为 180 维，训练 Logistic/HistGradientBoosting | 与 LSTM 相同的合法窗口 | 强基线模型、验证选择和测试概率 |
| `qd/lstm_hybrid.py` | 只在验证集选择 LSTM/HistGradientBoosting 线性概率权重，冻结后读取测试集 | 两个已保存模型及同一合法窗口 | 融合概率、显著性检验与选择性档位 |
| `qd/lstm_research.py` | 因果规则基线、概率指标、可靠性、温度校准、结构消融及 walk-forward 聚合 | 保存的模型与逐样本概率 | `output/lstm_research/` 诊断产物 |
| `qd/lstm_adjustment.py` | 不重训组件，只在验证集重选 move bias、融合权重与置信阈值 | 原三组件模型与分钟表 | 冻结的调整后模型及测试结果 |
| `qd/lstm_strategy.py` | 将三分类概率转成 long-short/long-only 仓位，严格对齐下一分钟收益并计入换手成本 | 测试概率、分钟 close | 组合路径、成本敏感性和策略指标 |

## 命令入口

| 命令 | 对应脚本 | 什么时候用 |
|---|---|---|
| `python -m scripts.smoke_test` | `scripts/smoke_test.py` | 快速确认基本数据口径和主要函数可用 |
| `python -m scripts.run_step1` | `scripts/run_step1.py` | 生成日频、分钟频数据；支持断点续跑 |
| `python -m scripts.run_factor_eval` | `scripts/run_factor_eval.py` | 构建因子并输出评价结果 |
| `python -m scripts.run_factor_robustness` | `scripts/run_factor_robustness.py` | 检查因子方向漂移、置信区间、状态和相关性 |
| `python -m scripts.run_backtest` | `scripts/run_backtest.py` | 运行四组 IC 加权回测 |
| `python -m scripts.run_backtest_robustness` | `scripts/run_backtest_robustness.py` | 运行换手约束和 0–50 bps 对称成本压力 |
| `python -m scripts.run_lstm` | `scripts/run_lstm.py` | 训练单一研究模型或条件方向模型 |
| `python -m scripts.run_lstm_full` | `scripts/run_lstm_full.py` | 从头训练全窗口三组件模型 |
| `python -m scripts.evaluate_lstm_full` | `scripts/evaluate_lstm_full.py` | 重载模型，复算验证集或测试集指标 |
| `python -m scripts.run_lstm_baselines` | `scripts/run_lstm_baselines.py` | 在相同样本上训练传统机器学习强基线 |
| `python -m scripts.run_lstm_hybrid` | `scripts/run_lstm_hybrid.py` | 用验证集冻结 LSTM/HistGradientBoosting 概率融合 |
| `python -m scripts.run_lstm_research` | `scripts/run_lstm_research.py` | 生成消融、校准、逐日稳定性和 walk-forward 协议/聚合 |
| `python -m scripts.adjust_lstm_fusion` | `scripts/adjust_lstm_fusion.py` | 用验证集调整已保存三组件的融合参数 |
| `python -m scripts.run_lstm_strategy` | `scripts/run_lstm_strategy.py` | 将冻结概率接入收益并进行成本后策略诊断 |
| `python -m scripts.validate_outputs` | `scripts/validate_outputs.py` | 对全工程产物做严格一致性验收 |

## 最重要的五个口径

### 1. 价格和成交字段

原始价格先除以 100，再乘以“当日复权因子 / 样本首日复权因子”。成交量、成交笔数和成交额保持真实成交口径，成交额不做复权。

### 2. 回测信息时序

修正版在交易日 `t` 使用 `t-1` 的因子，并只用截至 `t-2` 已实现收益所形成的历史 IC。Naive 版本故意使用未来收益，是用来说明前视偏差会把回测结果夸大到什么程度的反例，不是可交易策略。

### 3. LSTM 的分母

条件方向模型只评价下一分钟确实发生价格变化的样本，因此 68.10% 必须与 56.32% 的覆盖率一起报告。全窗口模型对全部合法窗口预测 down/flat/up，46.88% 才是完整三分类口径。两者不能直接比较。

### 4. 强基线与模型调整

全窗口 LSTM 的 46.88% Accuracy 与 HistGradientBoosting 的 46.80% 没有显著差异（原模型配对检验 `p=0.894`）。验证集驱动的组件融合调整把 Macro F1 从 45.87% 提高到 46.76%，但 Brier 变差，且相对树基线仍不显著（`p=0.770`）。跨模型 hybrid 的 Accuracy/Brier/NLL 改善到 47.31%/0.6032/0.9906，但相对原 LSTM/树模型仍不显著（`p=0.269/0.165`），策略表现也未同步改善。汇报时不能只对比 43.68% 的多数类基线，也不能把验证集调整表述为组件重新训练。

### 5. 稳健性结果的边界

- 2026Q2 有三个因子的 IC 方向相对全样本翻转，固定全样本负号不可取。
- 换手约束提高了盈亏平衡成本，但没有模拟市场冲击和容量。
- 概率策略只有 strict 档在当前 10 日、单边 5 bps 假设下为正；宽覆盖方案成本后为负。
- 11 折 walk-forward 目前只独立重训了第 1 折，不能用单折代表完整滚动表现。

## 推荐阅读顺序

如果只想理解项目：

1. 根目录 `README.md`；
2. `quant_downsampler/README.md`；
3. `qd/config.py` 和 `qd/pipeline.py`；
4. 根据分工阅读 `factors/evaluation`、`backtest` 或 `lstm_*`。

如果要修改项目：

1. 先找到对应的 `scripts/` 入口；
2. 顺着入口导入关系定位 `qd/` 核心函数；
3. 查看 `tests/` 中与模块同名的测试；
4. 修改后先跑相关测试，再运行全部 56 项测试和完整产物验收。

## 建议分工

- 数据与口径：`data_loader.py`、`adjfactor.py`、`daily.py`、`minute.py`。
- 因子与评价：`factors.py`、`evaluation.py`、`factor_robustness.py`。
- 策略与回测：`backtest.py`、`backtest_robustness.py`、`lstm_strategy.py`。
- 模型与评价：`lstm_model.py`、`lstm_full.py`、`lstm_full_training.py`、`lstm_baselines.py`、`lstm_hybrid.py`、`lstm_research.py`、`lstm_adjustment.py`。
- 复现与展示：`scripts/validate_outputs.py`、README、图表和 presentation。
