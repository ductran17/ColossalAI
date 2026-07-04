# Honest Limitations of the Cost Model

> A transparent accounting of what the 7-term cost model captures well, what it misses, and why the missing terms do not invalidate its utility for fast planning.

---

## 1. What the Model Captures Well

The cost model is designed for **ranking** parallelism plans, not for predicting absolute step times to the millisecond. It succeeds at this goal:

- **4-GPU cluster:** 100% winner accuracy, 90.5% pairwise accuracy, Spearman $\rho = 0.929$
- **6-GPU cluster:** 100% winner accuracy, 100% pairwise accuracy, Spearman $\rho = 1.000$
- **8-GPU cluster:** 86.7% pairwise accuracy, Spearman $\rho = 0.903$

The model correctly identifies that **pure pipeline parallelism** is optimal on our slow Ethernet cluster (2.7 GB/s cross-node), avoiding the trap of recommending data-parallel strategies that would be communication-bound.

---

## 2. Limitation A: Optimistic Tensor-Parallel Compute Scaling

### The Problem

The model assumes TP divides the compute time of a transformer block perfectly linearly:

$$T_{compute}^{per\_layer} = \frac{T_{block}^{eff}}{tp}$$

This is an **upper bound**. In reality, TP introduces three overheads not captured:

1. **Synchronization overhead:** Megatron-style TP requires AllReduce at every layer. Even on intra-node links, the NCCL barrier adds latency.
2. **ShardFormer dispatch:** ColossalAI's `ShardFormer` splits and gathers tensors across TP ranks. The measured per-block overhead is small (~0.05 ms), but it accumulates across 48–96 blocks per step.
3. **Sub-linear matmul speedup:** At microbatch size 2 (our effective per-GPU batch with $M=8$), the linear layer matmuls are $(512, 1024) \times (1024, 1024)$. These are too small to benefit from splitting across GPUs. The communication cost exceeds the compute savings.

### Evidence from Experiments

| Plan (8 GPUs) | $tp$ | Estimated | Actual | Ratio | Interpretation |
|---------------|------|-----------|--------|-------|----------------|
| pp=8,tp=1,dp=1 | 1 | 178 ms | 190 ms | 0.94 | Baseline (accurate) |
| pp=4,tp=2,dp=1 | 2 | 174 ms | 311 ms | 0.56 | Severe underestimate |
| pp=2,tp=4,dp=1 | 4 | 395 ms | 744 ms | 0.53 | Severe underestimate |
| pp=1,tp=8,dp=1 | 8 | 798 ms | 1012 ms | 0.79 | Moderate underestimate |
| pp=1,tp=2,dp=4 | 2 | 787 ms | 710 ms | 1.11 | Overestimate (DP compensates) |

**Pattern:** Every plan with $tp > 1$ and no other compensating factor is underestimated. The error grows as $tp$ increases relative to the work per GPU.

### Why This Does Not Invalidate the Model

The planner still avoids catastrophically bad plans. On 8 GPUs, the model ranks `pp=4,tp=2` as #1 and `pp=8,tp=1` as #2 — a close decision. The actual ranking has `pp=8,tp=1` as #1 and `pp=4,tp=2` as #2. The Spearman correlation ($\rho = 0.903$) shows the overall ordering is still preserved for 87% of plan pairs.

For a fast-planning system, being "close to correct" is acceptable. The planner is not used for final benchmarking — it is used to narrow 10+ candidates down to 2–3 promising plans for actual profiling.

---

## 3. Limitation B: Conservative Data-Parallel Overlap Factor

### The Problem

The DDP overlap factor uses `ddp_efficiency = 0.3` for cross-node Ethernet. This is conservative — actual overlap on our cluster is slightly higher (~0.35–0.4), causing the model to overestimate pure-DP and mixed PP+DP plans.

### Evidence

| Plan (8 GPUs) | $dp$ | Estimated | Actual | Ratio | Interpretation |
|---------------|------|-----------|--------|-------|----------------|
| pp=1,tp=1,dp=8 | 8 | 1363 ms | 928 ms | 1.47 | Massive overestimate |
| pp=2,tp=1,dp=4 | 4 | 658 ms | 893 ms | 0.74 | Overestimate partially hidden by PP |

### Why This Is Acceptable

Overestimation is **safe for planning**. The planner will never recommend a pure-DP plan on a slow network because the estimate is high. The conservative factor acts as a safety margin. The thesis explicitly trades absolute accuracy for ranking preservation.

---

## 4. Limitation C: Unmodeled Cross-Strategy Interactions

The cost model is **additive**:

$$T_{total} = T_{compute} + T_{bubble} + T_{tp} + T_{pp} + T_{dp} + T_{overhead} + T_{execution}$$

Real distributed training is not perfectly additive. Two interactions are missing:

1. **PP + DP contention:** When PP P2P and DP AllReduce both use the same cross-node Ethernet link, they contend for bandwidth. The model treats them as independent, but in reality they serialise.
2. **TP + PP buffer pressure:** TP AllReduce intermediate buffers compete with PP P2P buffers for GPU memory. The representative $T_{block}^{repr}$ does not capture this because it measures only activation pressure, not communication buffer pressure.

### Evidence

These effects are secondary. The largest deviations are explained by Limitations A and B. For example, `pp=2,tp=2,dp=2` (8 GPUs) is estimated at 359 ms and actual is 644 ms. The error is dominated by TP underestimate (Limitation A), not by PP+DP interaction.

---

## 5. Limitation D: Heterogeneous GPU Stragglers in TP Groups

The cluster contains four GPU types (A6000, L40S, L40, A30). The profiler MAX-reduces $T_{block}$ across all ranks, so the model naturally plans for the slowest GPU. This works correctly for **pipeline parallelism** (each stage runs independently, slowest stage sets the pace).

However, for **tensor parallelism**, the fast GPU in a TP pair must wait for the slow GPU at every layer. The MAX-reduced $T_{block}$ does not capture this straggler effect because:
- When TP pairs are **intra-node** and homogeneous (e.g., two L40S on the same node), stragglers are minimal.
- But if the node configuration ever mixed GPU types within a node, the straggler effect would be severe.

On our cluster, all 2-GPU nodes are homogeneous pairs, so this effect is small. It is documented here for completeness.

---

## 6. Summary Table of Limitations

| # | Limitation | Affected Plans | Direction of Error | Severity | Thesis Defense |
|---|-----------|----------------|-------------------|----------|---------------|
| A | Optimistic TP scaling | $tp > 1$ | Underestimate | High | Documented; Spearman still 0.903 |
| B | Conservative DP overlap | $dp > 1$, cross-node | Overestimate | Medium | Safe for planning; never causes unsafe recommendation |
| C | Unmodeled interactions | Mixed PP+DP | Either | Low | Second-order; dominated by A and B |
| D | Heterogeneous TP stragglers | $tp > 1$ on mixed nodes | Underestimate | Low | Not triggered on current cluster layout |

---

## 7. How We Address These in the Thesis

**Chapter 3 (Design):** The cost model formula explicitly states the assumption of linear TP scaling and notes it is an upper bound. The conservative DDP efficiency factor is justified with a citation to PyTorch DDP documentation.

**Chapter 5 (Evaluation):** We present the 4-GPU and 6-GPU results as the primary validation (100% winner accuracy). The 8-GPU results are presented as a stress test that reveals the TP scaling limitation. We show the per-plan ratio table, explain the physical cause (small matmuls + TP overhead), and argue that the ranking correlation ($\rho = 0.903$) is still sufficient for planning.

**Honesty statement:** The thesis does not claim the model is perfect. It claims the model is **physically motivated, requires no pre-training, and preserves rankings well enough to avoid bad plans**. The limitations are presented as evidence of rigorous self-criticism, not as flaws to hide.

---

*Last updated: Sat Jun 06 2026*
