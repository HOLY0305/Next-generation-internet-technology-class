from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import statistics

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
SEEDS = [11, 23, 37]
SCENARIOS = [
    "A_high_locality_balanced",
    "B_low_locality_stable_hotspot",
    "C_fast_hotspot_drift",
]
POLICIES = ["Keep", "Sample placement", "Expert replication", "Cost-aware"]

N_NODES = 2
GPUS_PER_NODE = 2
N_GPUS = N_NODES * GPUS_PER_NODE
N_EXPERTS = 8
N_WINDOWS = 12
N_SAMPLES = 32
TOKENS_PER_SAMPLE = 16
TOP_K = 1

# These are relative model units, not measurements of A100, NVLink, InfiniBand,
# RoCE, or any other real platform.
INTER_NODE_COST_PER_TOKEN = 1.0
INTRA_NODE_COST_PER_TOKEN = 0.20
EXPERT_COMPUTE_COST_PER_TOKEN = 0.80
SAMPLE_SOLVE_COST = 4.0
SAMPLE_MAPPING_COST_PER_CHANGED_SAMPLE = 0.08
REPLICA_ACTIVATE_COST = 55.0
REPLICA_REMOVE_COST = 10.0
REPLICA_SYNC_BASE_COST = 6.0
REPLICA_GRAD_COST_PER_TOKEN = 0.015
HOTSPOT_THRESHOLD = 1.40


def expert_gpu(expert: int) -> int:
    return expert // 2


def gpu_node(gpu: int) -> int:
    return gpu // GPUS_PER_NODE


def initial_sample_nodes() -> np.ndarray:
    return np.arange(N_SAMPLES, dtype=np.int64) % N_NODES


def make_trace(scenario: str, seed: int) -> np.ndarray:
    """Generate one shared [window, sample, token] Top-1 routing trace."""
    rng = np.random.default_rng(seed + 1000 * SCENARIOS.index(scenario))
    trace = np.empty((N_WINDOWS, N_SAMPLES, TOKENS_PER_SAMPLE), dtype=np.int64)
    drift_hotspots = [0, 4, 2, 6]

    for window in range(N_WINDOWS):
        for sample in range(N_SAMPLES):
            if scenario == "A_high_locality_balanced":
                primary = sample % N_EXPERTS
                secondary = (primary + 1) % N_EXPERTS
                experts = np.array([primary] * 12 + [secondary] * 4, dtype=np.int64)
            elif scenario == "B_low_locality_stable_hotspot":
                non_hot = rng.integers(1, N_EXPERTS, size=TOKENS_PER_SAMPLE // 2)
                experts = np.concatenate([np.zeros(TOKENS_PER_SAMPLE // 2, dtype=np.int64), non_hot])
            elif scenario == "C_fast_hotspot_drift":
                hot = drift_hotspots[window % len(drift_hotspots)]
                other_experts = np.array([e for e in range(N_EXPERTS) if e != hot])
                non_hot = rng.choice(other_experts, size=TOKENS_PER_SAMPLE // 2, replace=True)
                experts = np.concatenate([np.full(TOKENS_PER_SAMPLE // 2, hot, dtype=np.int64), non_hot])
            else:
                raise ValueError(f"unknown scenario: {scenario}")
            rng.shuffle(experts)
            trace[window, sample] = experts
    return trace


def sample_expert_counts(window_trace: np.ndarray) -> np.ndarray:
    counts = np.zeros((N_SAMPLES, N_EXPERTS), dtype=np.int64)
    for sample in range(N_SAMPLES):
        counts[sample] = np.bincount(window_trace[sample], minlength=N_EXPERTS)
    return counts


def plan_sample_nodes(previous_counts: np.ndarray, current_nodes: np.ndarray) -> np.ndarray:
    """Use only the previous window to choose the next sample node."""
    node_demand = np.zeros((N_SAMPLES, N_NODES), dtype=np.int64)
    for expert in range(N_EXPERTS):
        node_demand[:, gpu_node(expert_gpu(expert))] += previous_counts[:, expert]
    planned = current_nodes.copy()
    for sample in range(N_SAMPLES):
        if node_demand[sample, 0] > node_demand[sample, 1]:
            planned[sample] = 0
        elif node_demand[sample, 1] > node_demand[sample, 0]:
            planned[sample] = 1
    return planned


def plan_replica_targets(previous_counts: np.ndarray, previous_gpu_loads: np.ndarray, current_targets: dict[int, int]) -> dict[int, int]:
    """React to the previous window's hottest expert; never inspect the next window."""
    global_counts = previous_counts.sum(axis=0)
    mean_load = float(global_counts.mean())
    hot_expert = int(np.argmax(global_counts))
    if mean_load <= 0 or float(global_counts[hot_expert]) < HOTSPOT_THRESHOLD * mean_load:
        return {}

    if hot_expert in current_targets:
        target_gpu = current_targets[hot_expert]
    else:
        base = expert_gpu(hot_expert)
        candidates = [g for g in range(N_GPUS) if g != base]
        target_gpu = min(candidates, key=lambda g: (float(previous_gpu_loads[g]), g))
    return {hot_expert: target_gpu}


def transition_base_cost(
    previous_nodes: np.ndarray | None,
    next_nodes: np.ndarray,
    previous_targets: dict[int, int],
    next_targets: dict[int, int],
) -> tuple[float, int, int, int]:
    if previous_nodes is None:
        return 0.0, 0, 0, 0
    changed_samples = int(np.count_nonzero(previous_nodes != next_nodes))
    added = len(set(next_targets) - set(previous_targets))
    removed = len(set(previous_targets) - set(next_targets))
    cost = 0.0
    if changed_samples:
        # This represents solver + mapping/Gather-Scatter redirection cost,
        # not an independent token relocation.
        cost += SAMPLE_SOLVE_COST + SAMPLE_MAPPING_COST_PER_CHANGED_SAMPLE * changed_samples
    cost += REPLICA_ACTIVATE_COST * added + REPLICA_REMOVE_COST * removed
    return cost, changed_samples, added, removed


def evaluate_window(
    window_trace: np.ndarray,
    sample_nodes: np.ndarray,
    replica_targets: dict[int, int],
    previous_nodes: np.ndarray | None,
    previous_targets: dict[int, int],
) -> dict[str, float]:
    gpu_loads = np.zeros(N_GPUS, dtype=np.int64)
    cross_node_tokens = 0
    replica_tokens = 0
    for sample in range(N_SAMPLES):
        source_node = int(sample_nodes[sample])
        for expert in window_trace[sample]:
            expert = int(expert)
            base_gpu = expert_gpu(expert)
            candidates = [base_gpu]
            if expert in replica_targets:
                candidates.append(replica_targets[expert])
            chosen_gpu = min(candidates, key=lambda g: (int(gpu_loads[g]), g))
            gpu_loads[chosen_gpu] += 1
            if gpu_node(chosen_gpu) != source_node:
                cross_node_tokens += 1
            if chosen_gpu != base_gpu:
                replica_tokens += 1

    transition, changed_samples, added, removed = transition_base_cost(
        previous_nodes, sample_nodes, previous_targets, replica_targets
    )
    sync_cost = REPLICA_SYNC_BASE_COST * len(replica_targets) + REPLICA_GRAD_COST_PER_TOKEN * replica_tokens
    reconfig_cost = transition + sync_cost
    intra_node_tokens = N_SAMPLES * TOKENS_PER_SAMPLE - cross_node_tokens
    communication_proxy = 2.0 * (
        INTER_NODE_COST_PER_TOKEN * cross_node_tokens + INTRA_NODE_COST_PER_TOKEN * intra_node_tokens
    )
    expert_proxy = EXPERT_COMPUTE_COST_PER_TOKEN * int(gpu_loads.max())
    return {
        "cross_node_tokens": float(cross_node_tokens),
        "intra_node_tokens": float(intra_node_tokens),
        "peak_expert_or_gpu_load": float(gpu_loads.max()),
        "replica_count": float(len(replica_targets)),
        "replica_tokens": float(replica_tokens),
        "reconfig_cost": float(reconfig_cost),
        "communication_proxy": float(communication_proxy),
        "expert_proxy": float(expert_proxy),
        "cost_proxy": float(communication_proxy + expert_proxy + reconfig_cost),
        "changed_samples": float(changed_samples),
        "added_replicas": float(added),
        "removed_replicas": float(removed),
        "gpu_loads": gpu_loads,
    }


def state_action(previous_nodes: np.ndarray, next_nodes: np.ndarray, previous_targets: dict[int, int], next_targets: dict[int, int]) -> str:
    node_changed = bool(np.any(previous_nodes != next_nodes))
    replica_changed = previous_targets != next_targets
    if node_changed and replica_changed:
        return "Sample placement + replication"
    if node_changed:
        return "Sample placement"
    if replica_changed:
        return "Expert replication"
    return "Keep"


def run_policy(trace: np.ndarray, policy: str) -> list[dict[str, float | str]]:
    nodes = initial_sample_nodes()
    replicas: dict[int, int] = {}
    previous_nodes: np.ndarray | None = None
    previous_trace: np.ndarray | None = None
    previous_counts: np.ndarray | None = None
    previous_gpu_loads = np.zeros(N_GPUS, dtype=np.int64)
    rows: list[dict[str, float | str]] = []

    for window in range(N_WINDOWS):
        action = "Initial"
        if window > 0:
            assert previous_trace is not None and previous_counts is not None and previous_nodes is not None
            if policy == "Keep":
                next_nodes, next_replicas = nodes.copy(), dict(replicas)
            elif policy == "Sample placement":
                next_nodes, next_replicas = plan_sample_nodes(previous_counts, nodes), dict(replicas)
            elif policy == "Expert replication":
                next_nodes, next_replicas = nodes.copy(), plan_replica_targets(previous_counts, previous_gpu_loads, replicas)
            elif policy == "Cost-aware":
                candidates = [
                    ("Keep", nodes.copy(), dict(replicas)),
                    ("Sample placement", plan_sample_nodes(previous_counts, nodes), dict(replicas)),
                    ("Expert replication", nodes.copy(), plan_replica_targets(previous_counts, previous_gpu_loads, replicas)),
                ]
                scored = []
                for label, candidate_nodes, candidate_replicas in candidates:
                    predicted = evaluate_window(
                        previous_trace, candidate_nodes, candidate_replicas, nodes, replicas
                    )
                    scored.append((predicted["cost_proxy"], 0 if label == "Keep" else 1, label, candidate_nodes, candidate_replicas))
                _, _, action, next_nodes, next_replicas = min(scored, key=lambda item: (item[0], item[1]))
            else:
                raise ValueError(policy)
            if policy != "Cost-aware":
                action = state_action(nodes, next_nodes, replicas, next_replicas)
            nodes, replicas = next_nodes, next_replicas

        metrics = evaluate_window(trace[window], nodes, replicas, previous_nodes, replicas if window == 0 else previous_replicas)
        if window == 0:
            metrics["reconfig_cost"] = 0.0
            metrics["cost_proxy"] = metrics["communication_proxy"] + metrics["expert_proxy"]
        metrics.update({"window": window, "action": action})
        rows.append(metrics)
        previous_nodes = nodes.copy()
        previous_replicas = dict(replicas)
        previous_trace = trace[window]
        previous_counts = sample_expert_counts(previous_trace)
        previous_gpu_loads = metrics["gpu_loads"].copy()
    return rows


def aggregate(scenario: str, seed: int, policy: str, rows: list[dict[str, float | str]]) -> dict[str, float | str | int]:
    numeric_fields = [
        "cross_node_tokens",
        "peak_expert_or_gpu_load",
        "replica_count",
        "reconfig_cost",
        "cost_proxy",
        "communication_proxy",
        "expert_proxy",
        "changed_samples",
        "replica_tokens",
    ]
    out: dict[str, float | str | int] = {"scenario": scenario, "seed": seed, "policy": policy, "windows": N_WINDOWS, "evidence_type": "synthetic_trace_model_proxy"}
    for field in numeric_fields:
        out[field] = round(float(statistics.mean(float(row[field]) for row in rows)), 4)
    out["action_count_keep"] = sum(row["action"] == "Keep" for row in rows[1:])
    out["action_count_sample"] = sum(row["action"] == "Sample placement" for row in rows[1:])
    out["action_count_replication"] = sum(row["action"] == "Expert replication" for row in rows[1:])
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_summary(path: Path, results: list[dict]) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results:
        grouped[(str(row["scenario"]), str(row["policy"]))].append(row)
    lines = [
        "# 合成路由轨迹小实验结果摘要",
        "",
        "本实验已实际运行并保存 `results.csv`。所有策略共享同一场景、同一 seed 生成的路由 Trace；结果是相对成本模型下的估计量，不是 GPU、网络或真实 MoE 训练实测。",
        "",
        "## 结果表（跨 3 个 seed 的均值）",
        "",
        "| 场景 | 策略 | 跨节点 Token/窗口 | 峰值 GPU 负载/窗口 | 平均副本数 | 重配置成本/窗口 | 总成本代理/窗口 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for policy in POLICIES:
            rows = grouped[(scenario, policy)]
            mean = lambda field: statistics.mean(float(r[field]) for r in rows)
            lines.append(
                f"| {scenario} | {policy} | {fmt(mean('cross_node_tokens'))} | {fmt(mean('peak_expert_or_gpu_load'))} | {fmt(mean('replica_count'))} | {fmt(mean('reconfig_cost'))} | {fmt(mean('cost_proxy'))} |"
            )

    lines += [
        "",
        "## 机制方向与反例",
        "",
        "- 场景 A 的 Trace 让单个样本偏向少数专家，但全局专家计数近似均衡；因此样本放置通常能降低跨节点 Token 量，而专家复制没有稳定的热点触发条件。",
        "- 场景 B 把大部分 Token 稳定地送往同一专家；复制可降低峰值 GPU 负载，但会引入副本激活和同步成本。样本放置即使降低通信量，也不能消除热门专家的总计算工作。",
        "- 场景 C 每个窗口快速更换热点；依赖上一窗口的复制或样本调整可能追逐过时信号，重配置成本会抵消部分收益。该场景用于检验“保持现状”是否可能优于盲目调整。",
        "- 总成本代理采用 `T_proxy = 2 × (跨节点 Token × 1.0 + 机内 Token × 0.20) + 峰值 GPU Token × 0.80 + 未隐藏调整成本`。这里的 2 表示 Scatter 与 Gather 两个方向；该公式不模拟真实通信—计算 overlap。",
        "- 样本放置的成本只计入求解、映射以及现有 Gather/下一层 Scatter 重定向，不额外添加一次独立的 Token 搬迁；复制成本包含副本激活、移除和副本同步。",
        "",
        "## 是否写入论文 4.3",
        "",
        "本次结果具有机制分析价值：它在统一合成 Trace 和透明成本模型下呈现了局部性、热点持续性与重配置成本之间的条件化权衡，并包含快速热点漂移这一反例。但本轮不写入 v7 的 4.3。原因是当前 v6 正文主体按中文字符计数已略超课程要求的 3000–5000 字区间，且该实验仍是合成 Trace 的模型代理，新增正文会进一步增加篇幅并容易被误读为真实系统验证。实验结果、代码和限制独立保留在本目录；后续若压缩正文并明确标注“合成 Trace 的模型代理结果”，再考虑加入一段不超过 150–250 字的补充。",
        "",
        "## 限制",
        "",
        "- 只有 2 节点 × 2 GPU、8 个专家、32 个样本、每样本 16 个 Top-1 Token，不能外推到具体 GPU 或集群规模。",
        "- 复制只改变同一请求在原专家和副本之间的分配；没有减少专家总计算工作，也没有模拟参数张量大小、真实梯度通信协议或显存字节数。",
        "- 成本代理采用相加形式，没有建模 CUDA kernel、NCCL、交换机队列、异步重叠和训练收敛。",
        "- Cost-aware 只使用过去一个窗口的 Trace 做决策，下一窗口只用于评估；因此结果不是 oracle 上界，也不等同于已实现的在线控制器。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    all_results: list[dict] = []
    all_trace_rows: list[dict] = []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            trace = make_trace(scenario, seed)
            # Store compact sample→expert counts so the common input is inspectable.
            for window in range(N_WINDOWS):
                counts = sample_expert_counts(trace[window])
                for sample in range(N_SAMPLES):
                    for expert in range(N_EXPERTS):
                        all_trace_rows.append({
                            "scenario": scenario,
                            "seed": seed,
                            "window": window,
                            "sample_id": sample,
                            "expert_id": expert,
                            "token_count": int(counts[sample, expert]),
                        })
            policy_rows = {}
            for policy in POLICIES:
                policy_rows[policy] = run_policy(trace, policy)
                all_results.append(aggregate(scenario, seed, policy, policy_rows[policy]))
            totals = [sum(int(x["token_count"]) for x in all_trace_rows if x["scenario"] == scenario and x["seed"] == seed)]
            if totals[0] != N_WINDOWS * N_SAMPLES * TOKENS_PER_SAMPLE:
                raise AssertionError("trace token total mismatch")

    write_csv(OUT_DIR / "results.csv", all_results)
    write_csv(OUT_DIR / "traces.csv", all_trace_rows)
    write_summary(OUT_DIR / "results_summary.md", all_results)
    print(f"wrote {len(all_results)} result rows and {len(all_trace_rows)} trace rows")


if __name__ == "__main__":
    main()
