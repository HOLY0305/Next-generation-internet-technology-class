# 下一代互联网技术课程项目

本仓库整理《下一代互联网技术》课程中 AI 集群网络与分布式 MoE 动态放置研究使用的论文图表、PPT 素材和可重复运行的合成 Trace 仿真代码。

## 目录

- `paper/`：论文图表、图片绘制 PPT、研究报告模板、最终 v16 PDF、完整 LaTeX 工程及 LaTeX 模板压缩包。
- `experiments/placement_trace_sim/`：范围受控的 CPU 合成路由轨迹仿真，包括代码、输入 Trace、结果和审计记录。

最终论文 PDF：`paper/MoE动态放置_个人课程报告_v16.pdf`；对应 LaTeX 工程：`paper/latex_CjC_单栏_v16_最终仓库说明修正版/`。

## 运行仿真

进入 `experiments/placement_trace_sim/`，使用 Python 3 和 NumPy 运行：

```powershell
python simulate.py
```

实验比较 Keep、sample placement、expert replication 和 Cost-aware 四种策略。详细参数、场景、成本模型和结果边界见该目录的 `README.md`。

## 结果边界

该实验是相对成本模型下的机制仿真，不是 CUDA/NCCL、GPU Step Time、网络吞吐、Token Dropping、精度或收敛时间的实测。论文中使用时应保留这一限定。

## 课程报告

论文主题为“分布式 MoE 训练中的动态专家与样本放置机制”。报告比较专家拉取、专家复制、样本放置和通信调度的收益来源与成本，并保留可核验的文献和实验条件。

仓库地址：<https://github.com/HOLY0305/Next-generation-internet-technology-class>
