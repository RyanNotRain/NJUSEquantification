# 兆信基金量化研究项目

本目录包含题目 Task 1–5 的可复现实现。原始逐笔数据位于工作区 `data/TRADE`，正式结果统一写入 `project/output`；路径由 `qd/config.py` 相对项目位置自动解析，也可通过环境变量覆盖。

## 当前正式结论

- Task 1：完成 302 个交易日、300 只股票的日频与分钟频降采样。日频和分钟频均包含 OHLC、成交量、成交笔数、成交额、主买/主卖量额共 11 个字段；OHLC 已使用 `adjfactor.pkl` 复权。
- Task 2–3：正式答题集合严格限定为题目示例因子加 3 个自建因子（20 日动量、主买卖失衡、20 日量价相关）；其余 6 个因子只作扩展。示例因子的 5/10/20 日均值必须满足完整窗口。除 IC、年化 IR、ICIR 和 Rank 指标外，现已保存 Q1–Q5 逐日收益、五条累计净值曲线、分层绩效和单调性指标。
- Task 4：先用上述正式 4 因子完成收盘/次日开盘两套严格历史 IC 回测。`adaptive_close` 总收益 19.85%，相对同口径等权市场全期几何超额 12.00%；`adaptive_open` 总收益 52.76%，相对开盘等权市场超额 40.64%。两者最近 45 日均略跑输市场，且整个样本已被反复查看，不能视为盲测。10 因子扩展版 `adaptive_close` 总收益为 9.25%，缓冲版为 11.33%。冻结后聚类精选的几何超额为 21.46%，仍需 walk-forward 确认。
- 因子学习：收益率回归和直接夏普优化在全滚动期为正，但最近 45 日夏普均小于 -3，说明因子方向在近期失效。
- Task 5：正式模型为三个独立 LSTM 的概率融合。可交易 T+1 基线中 Ridge 的同覆盖超额为 +0.69%，原25特征紧凑 LSTM 为 -8.08%。训练集聚类精选19个输入后，收益 Rank IC 从 -0.017 提高到 +0.065、同覆盖超额改善至 -0.67%。另将原题 `Time/Price/Volume/BSFlag` 映射为四个平稳特征重训，Accuracy 45.54%、Rank IC -0.121、同覆盖超额 -2.40%；它好于原25特征的相对收益，但明显弱于19特征，说明问题是重复输入而非所有新增信息都有害。

## 目录与入口

| 模块 | 入口 | 正式输出 |
|---|---|---|
| Task 1 | `scripts/run_step1.py` | `../output/daily`、`../output/minute` |
| Task 2–3 | `scripts/run_factor_eval.py` | `../output/factors`、`../output/evaluation`（含 Q1–Q5 五条净值曲线） |
| 因子稳健性 | `scripts/run_factor_robustness.py` | `../output/factor_robustness` |
| 因子去冗余 | `scripts/run_factor_independence.py` | `../output/factor_independence` |
| Task 4 正式 4 因子 | `scripts/run_task4_strict.py --factor-set required` | `../output/backtest_required` |
| Task 4 扩展 10 因子 | `scripts/run_task4_strict.py --factor-set extended` | `../output/backtest_strict` |
| Task 4 换手/成本 | `scripts/run_task4_robustness.py` | `../output/backtest_robustness` |
| Task 4/5 相对收益 | `scripts/run_strategy_analysis.py` | 基准、超额收益与 LSTM 策略诊断 |
| 因子学习 | `scripts/run_factor_models.py` | `../output/factor_models_*` |
| Task 5 基线 | `scripts/run_lstm.py` | `../output/lstm_next_minute` |
| Task 5 三组件 | `scripts/run_lstm_ensemble.py` | `../output/lstm_ensemble` |
| Task 5 强基线/校准 | `scripts/run_lstm_baselines.py` | `../output/lstm_baselines` |
| Task 5 涨跌幅多任务 | `scripts/run_lstm_magnitude.py` | `../output/lstm_magnitude` |
| Task 5 多期限可交易收益 | `scripts/run_tradable_return_research.py` | `../output/tradable_return_research` |
| Task 5 T+1 紧凑 LSTM | `scripts/run_tradable_lstm.py` | `../output/tradable_lstm` |
| Task 5 特征/组件独立性 | `scripts/run_lstm_feature_independence.py` | `../output/lstm_feature_independence` |
| Task 5 原题四字段基线 | `scripts/run_lstm_minimal_four.py` | `../output/lstm_minimal_four` |
| Task 5 重放 | `scripts/validate_lstm_ensemble.py` | `../output/lstm_ensemble/replay_validation.json` |
| 总体验收 | `scripts/validate_project_outputs.py` | `../output/validation_report.json` |

## Task 5 方法

输入窗口终止于分钟 t，标签严格比较 `close[t+1]` 与 `close[t]`，定义 down/flat/up 三类。窗口不能跨交易日或午休，也不使用 14:57–15:00 的收盘集合竞价等待和撮合行。

三组件分别承担不同任务：

1. direction：在真实非平盘训练样本上学习 down/up，使用 11 个基础特征、全局训练集标准化。
2. movement：在全部样本上学习 flat/move，使用 25 个增强特征、逐股票标准化和股票 ID。
3. joint：在全部样本上直接学习 down/flat/up，特征与 movement 相同。
4. validation：仅在验证集搜索 movement 概率偏置和 joint 融合权重，并冻结 70%/90% 置信分位阈值。
5. test：所有参数冻结以后才首次加载测试日期；保存每个样本、时间戳、真实标签、最终概率和三个组件概率。

正式 seed 42 模型在 5,900 个测试窗口上取得 46.46% Accuracy；高置信度档位为：

| 口径 | 测试覆盖率 | 测试准确率 |
|---|---:|---:|
| 全窗口 | 100.00% | 46.46% |
| balanced | 30.51% | 54.44% |
| strict | 9.78% | 62.56% |

选择性准确率的提升以放弃大量低置信度样本为代价，必须同时报告覆盖率。固定测试日期在研究过程中已经被查看，因此属于统一口径的样本外诊断，不是全新盲测。

策略诊断进一步区分三种时点：`close[t]→close[t+1]` 是不可交易的标签收益上限；`open[t+1]→close[t+1]` 只检验价格时点衰减，因 A 股 T+1 不能作为正式交易结果；`open[t+1]→下一交易日同分钟 open` 满足 T+1，并同时比较全仓市场和相同持仓覆盖的市场。三者共同说明不能用分类准确率或低市场暴露下的相对收益直接证明选股能力。

涨跌幅实验不把方向标签细分为更多档，而是在同一 LSTM 编码器上增加 signed-return 回归头。分类损失保留所有平盘样本；回归损失对大波动给予较高但有上限的权重，并只用训练集拟合缩放和截尾。验证集按收益 Rank IC 选择 checkpoint 并冻结收益阈值，测试集仍只打开一次。该模型只作探索性增强，不替换正式三组件模型。

可交易收益研究另外构造 `open[t+1]` 入场的 1/5/15/30 分钟标签和次交易日同分钟开盘退出的 T+1 标签。Ridge/HistGB 的超参数、Top-K、最低预测收益和再平衡间隔均在验证集冻结；测试表完整保留全部期限，不按测试收益挑选。验证推荐的 T+1 目标随后用于训练紧凑多任务 LSTM，结果弱于 Ridge，因此不替换简单模型。

## 快速运行

在本目录执行：

```powershell
.\run_all.bat
.\run_all.bat --with-lstm
```

详细命令见 [RUN.md](RUN.md)。
