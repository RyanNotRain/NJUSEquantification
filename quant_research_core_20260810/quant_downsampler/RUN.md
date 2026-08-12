# 运行说明

以下命令均在 `project/quant_downsampler` 目录运行。请先进入自己机器上的项目目录：

```powershell
cd "<项目所在位置>\project\quant_downsampler"
```

## 一键运行

```powershell
# Task 1–4 与因子学习
.\run_all.bat

# 额外运行 Task 5 基线、三组件模型和严格重放
.\run_all.bat --with-lstm
```

## 分步运行

```powershell
# 依赖
py -3 -m pip install -r requirements.txt

# 代码与单元测试
py -3 -m compileall -q qd scripts tests
py -3 -m unittest discover -v

# Task 1：日频和 242 行分钟频
py -3 scripts\run_step1.py
py -3 scripts\smoke_test.py

# Task 2–3：正式4因子＋扩展10因子、IC/Rank指标、Q1–Q5五条净值曲线
py -3 scripts\run_factor_eval.py --save
py -3 -m scripts.run_factor_robustness --iterations 2000 --block-length 5

# 因子相关性聚类、代表因子冻结、截面正交化与同区间严格回测
py -3 -m scripts.run_factor_independence --calibration-days 120 --correlation-threshold 0.60

# 数据质量抽样审计、正式4因子多期限衰减、Task4多配置四折稳定性
py -3 -m scripts.run_comprehensive_extensions --sections data factor task4

# Task 4：先跑题目正式4因子，再跑10因子扩展对照
py -3 scripts\run_task4_strict.py --factor-set required --lookback 60 --top-n 10 --out ..\output\backtest_required
py -3 scripts\run_task4_strict.py --factor-set extended --lookback 60 --top-n 10 --out ..\output\backtest_strict

# Task 4：同执行价格的等权市场基准、累计收益率差、几何超额和最近45日表现
# 默认一次同时重建 adaptive_close 与 adaptive_open
py -3 -m scripts.run_strategy_analysis --task4-only --task4-dir backtest_required --task4-strategy both --recent-days 45
py -3 -m scripts.run_strategy_analysis --task4-only --task4-dir backtest_strict --task4-strategy both --recent-days 45

# Task 4：成对5日移动区块 Bootstrap 与60日滚动超额、Beta、IR
py -3 -m scripts.run_task4_excess_significance --iterations 5000 --block-length 5 --rolling-window 60

py -3 -m scripts.run_task4_robustness

# 因子收益率回归和直接夏普优化
py -3 scripts\run_factor_models.py --out ..\output\factor_models_aligned
py -3 scripts\run_factor_models.py --rolling --out ..\output\factor_models_rolling_aligned

# Task 5：原始直接三分类基线
py -3 -m scripts.run_lstm --stocks 5 --epochs 12 --seq-len 60 --out ..\output\lstm_next_minute

# Task 5：正式三组件融合；CPU 环境可将 cuda 改为 cpu
py -3 -m scripts.run_lstm_ensemble --stocks 5 --epochs 12 --seq-len 60 --device cuda --out-dir ..\output\lstm_ensemble --overwrite

# 从分钟表严格重载模型，逐行核对概率和标签
py -3 -m scripts.validate_lstm_ensemble --run-dir ..\output\lstm_ensemble --data-dir ..\output\minute --device cuda

# 训练因果 Logistic/HistGB 强基线，并在验证集完成温度校准
py -3 -m scripts.run_lstm_baselines

# 探索性方向＋涨跌幅双头 LSTM；按验证集收益 Rank IC 选择 checkpoint
py -3 -m scripts.run_lstm_magnitude --epochs 12 --return-loss-weight 0.25 --sell-fee-bps 5 --device cpu --overwrite

# 1/5/15/30 分钟与 T+1 可交易收益标签、强回归基线和低频 Top-K
py -3 -m scripts.run_tradable_return_research --sell-fee-bps 5

# 同一 T+1 策略引擎下比较原始收益标签与直接市场超额标签
py -3 -m scripts.run_t1_excess_return_research --sell-fee-bps 5

# 在验证集筛出的 T+1 目标上训练紧凑多任务 LSTM
py -3 -m scripts.run_tradable_lstm --epochs 8 --sell-fee-bps 5 --device cpu

# 训练集分钟特征聚类、三组件互补性审计和精选特征 T+1 LSTM
py -3 -m scripts.run_lstm_feature_independence --correlation-threshold 0.85 --epochs 8 --device cpu

# 原题 Time/Price/Volume/BSFlag 映射的四特征最小基线
py -3 -m scripts.run_lstm_minimal_four --epochs 8 --sell-fee-bps 5 --device cpu

# Task5按交易日整块Bootstrap：分类指标与T+1超额置信区间
py -3 -m scripts.run_comprehensive_extensions --sections task5

# 检查扩展融合网格；参数只在验证集选择
py -3 -m scripts.analyze_lstm_fusion_grid --run-dir ..\output\lstm_ensemble

# 汇总 seed 42/43/44
py -3 -m scripts.summarize_lstm_runs

# 用冻结的预测结果回测 LSTM 的可执行收益；卖出费 5 bp
py -3 -m scripts.run_strategy_analysis --task5-only --sell-fee-bps 5

# Task1–5 最终产物验收
py -3 -m scripts.validate_project_outputs
```

## 路径覆盖

默认路径由 `qd/config.py` 解析。如需更换位置：

```powershell
$env:QD_DATA_DIR = "D:\\your_data\\TRADE"
$env:QD_ADJFACTOR_PATH = "D:\\your_data\\adjfactor.pkl"
$env:QD_OUTPUT_DIR = "D:\\your_output"
```

正式结果默认写入项目相对路径 `../output`。
