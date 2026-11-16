# 合成路由轨迹放置机制仿真

本目录中的实验是范围受控的 CPU 仿真，用于检查“局部性、专家热点持续性和重配置成本如何影响放置选择”这一机制判断。它不是 CUDA/NCCL/MoE 框架实现，也不是 GPU、网络吞吐、Token Dropping、精度或 Time-to-Convergence 的实测。

## 运行

环境：Python 3、NumPy。运行：

```powershell
python simulate.py
```

脚本固定使用 3 个 seed：`11、23、37`，每个场景 12 个窗口、32 个样本、每个样本 16 个 Top-1 Token。所有策略共享同一场景和 seed 生成的 Trace；`traces.csv` 保存了样本—专家计数，便于检查共同输入。

## 逻辑拓扑与基线

- 2 个 Node，每个 Node 2 张逻辑 GPU；
- 8 个 Expert，固定基线映射为每张 GPU 承载 2 个 Expert；
- 初始样本归属按样本编号交替放在两个 Node；
- 机内与跨节点通信成本分别设为 `0.20` 和 `1.0` 个相对单位；
- 专家计算成本为每个 Token `0.80` 个相对单位；所有参数都是模型假设，不对应任何真实 A100、NVLink、InfiniBand 或 Ethernet 测量。

## 场景

- `A_high_locality_balanced`：单个样本集中访问少数 Expert，但全局 Expert 计数近似均衡；
- `B_low_locality_stable_hotspot`：样本内路由局部性低，Expert 0 在全部窗口持续热门；
- `C_fast_hotspot_drift`：热点 Expert 按窗口快速变化，上一窗口信息容易过时。

## 策略与决策约束

- `Keep`：固定样本归属和 Expert 映射；
- `Sample placement`：根据上一窗口的样本—Expert 计数选择下一窗口 Node，成本只计求解、映射及现有 Gather/下一层 Scatter 重定向，不添加一次独立 Token 搬迁；
- `Expert replication`：根据上一窗口的热门 Expert 在其他 GPU 增加一个副本，Token 工作总量不变，计入副本激活、移除和同步成本；
- `Cost-aware`：在 Keep、样本放置和专家复制之间选择，预测只使用上一窗口，下一窗口只用于评估。

## 成本代理

实验使用不模拟 overlap 的可检查代理：

```text
T_proxy = 2 × (跨节点 Token × 1.0 + 机内 Token × 0.20)
          + 峰值 GPU Token × 0.80
          + 未隐藏调整成本
```

其中 `2` 表示 Scatter 与 Gather 两个方向。`results.csv` 中的四个核心指标分别是跨节点 Token、峰值 GPU 负载、平均活动副本数、未隐藏重配置成本和总成本代理；聚合值为每个窗口均值。

## 解释边界

结果只能说明在这组合成 Trace 和相对成本模型下的条件化机制趋势。它不能证明真实系统加速，也不能替代端到端训练、网络队列、GPU kernel、显存字节、梯度协议或收敛实验。`results_summary.md` 给出反例和是否适合写入论文的判断。
