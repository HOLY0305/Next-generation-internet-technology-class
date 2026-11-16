# Next-generation Internet Technology Class Project

This repository collects the figures, slide materials, and reproducible synthetic-trace simulation used in the course project on AI-cluster networking and dynamic placement for distributed Mixture-of-Experts (MoE) training.

## Contents

- `paper/`: paper figures, the editable figure-drawing PPT, report templates, the final v16 PDF, the complete LaTeX project, and the CjC LaTeX template archive.
- `experiments/placement_trace_sim/`: a controlled CPU simulation of synthetic routing traces, including the script, input traces, results, and audit records.

Final PDF: `paper/MoE动态放置_个人课程报告_v16.pdf`; corresponding LaTeX project: `paper/latex_CjC_单栏_v16_最终仓库说明修正版/`.

## Running the simulation

Enter `experiments/placement_trace_sim/` and run the following command with Python 3 and NumPy:

```powershell
python simulate.py
```

The experiment compares four strategies: Keep, sample placement, expert replication, and Cost-aware placement. The local README documents the parameters, scenarios, cost model, and interpretation boundaries.

## Interpretation boundary

This is a mechanism-level simulation under a relative-cost model. It is not a measurement of CUDA/NCCL execution, GPU step time, network throughput, token dropping, accuracy, or time-to-convergence. These limitations must remain explicit when the results are cited.

## Course report

The report studies dynamic expert and sample placement in distributed MoE training. It compares expert fetch, expert replication, sample placement, and communication scheduling by their benefit sources, adjustment costs, and applicable workloads.

Repository URL: <https://github.com/HOLY0305/Next-generation-internet-technology-class>
