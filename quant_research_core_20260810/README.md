# 兆信基金量化研究项目

这是当前核心研究仓库，覆盖逐笔成交降采样、日频因子、严格历史 IC 回测、因子去冗余、分钟级 LSTM、可交易收益预测和统一策略评价。

> 仅用于课程与研究展示，不构成投资建议。原始行情、复权因子、模型权重和大体积逐样本输出不上传。

## 当前结论

- Task1：完成 302 个交易日、300 只股票、11 个日频/分钟频字段的降采样与复权。
- Task2–4：构建 10 个日频因子，修复未来 IC 前视偏差；严格 `adaptive_close` 总收益 15.38%。
- 因子独立性：训练期聚类将 10 个因子精选为 6 个，冻结后最大相关性由 0.837 降至 0.597；正交化后为 0.096。
- Task5 正式三组件模型：测试 Accuracy 46.46%，高于 43.68% 多数类基线；校准后 ECE 为 0.91%。
- 可交易 T+1：Ridge 同覆盖超额 +0.69%；25 特征 LSTM 为 -8.08%，训练集精选 19 特征后改善至 -0.67%。
- 原题 `Time/Price/Volume/BSFlag` 四字段映射基线同覆盖超额 -2.40%，说明四项信息不足、25 项存在冗余，19 项目前最均衡。
- 当前共有 36 个自动测试，全项目产物验收为 `passed`。

## 入口

- [完整中期 Markdown 报告](兆信基金量化研究中期报告_最终版.md)
- [核心代码说明](quant_downsampler/README.md)
- [详细运行命令](quant_downsampler/RUN.md)
- [项目状态](quant_downsampler/PROJECT_STATUS.md)
- `quant_downsampler/qd/`：核心实现
- `quant_downsampler/scripts/`：可执行入口
- `quant_downsampler/tests/`：自动测试
- `output/`：仅提交小体积结果摘要和报告图片

## 快速检查

```powershell
cd quant_downsampler
python -m compileall -q qd scripts tests
python -m unittest discover -v
python -m scripts.validate_project_outputs
```

完整流水线：

```powershell
.\run_all.bat
.\run_all.bat --with-lstm
```

## 未上传内容

- `data/` 原始逐笔行情；
- `adjfactor.pkl`；
- 日频/分钟频全量 CSV；
- `.pt/.pth/.joblib` 模型文件；
- 大体积逐样本预测和缓存。

这些文件均可由代码在本地重新生成，仓库只保留复现所需代码、文档、测试和关键结果摘要。
