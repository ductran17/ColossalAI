# Priority 6: Formal Analysis + Thesis Writing — Detailed Guide

> How to write a defensible master's thesis from your experiments.

---

## Thesis Structure (6 Chapters)

| Chapter | Length | Purpose | Time |
|---------|--------|---------|------|
| **1. Introduction** | 8–10 pages | Problem, motivation, contributions | 3–4 days |
| **2. Background & Related Work** | 12–15 pages | Survey existing systems | 5–7 days |
| **3. System Design** | 15–20 pages | Formal model, cost model, planner | 7–10 days |
| **4. Implementation** | 8–10 pages | Code, integration, tools | 3–5 days |
| **5. Experimental Evaluation** | 15–20 pages | Baselines, ablations, scalability, convergence | 7–10 days |
| **6. Limitations & Future Work** | 4–6 pages | Honest assessment | 2–3 days |
| **References** | — | 20–40 citations | 1–2 days |
| **Total** | **60–80 pages** | | **4–6 weeks** |

---

## Chapter 1: Introduction

### 1.1 Problem Statement

**Template:**

> "Training large neural networks requires distributing computation across multiple GPUs. The standard approach combines three parallelism strategies — data parallelism (DP), tensor parallelism (TP), and pipeline parallelism (PP) — into a 3D parallel configuration. The choice of configuration (pp, tp, dp) significantly impacts training throughput and memory utilization. However, existing systems require users to manually specify these values, which demands deep expertise in distributed systems and intimate knowledge of the cluster topology. This manual process is error-prone: an incorrect choice can lead to out-of-memory crashes, network hangs, or severe under-utilization of hardware."

### 1.2 Motivation

**Template:**

> "Existing auto-parallel systems such as Alpa [1] and FlexFlow [2] address this problem through automated search, but they rely on extensive offline profiling (hours) or assume homogeneous hardware with high-bandwidth interconnects (NVLink/InfiniBand). In practice, academic and industrial clusters often consist of heterogeneous GPUs connected via commodity Ethernet with variable bandwidth. There is a need for a fast, topology-aware planner that can automatically select a suitable 3D parallel configuration for such clusters without manual tuning."

### 1.3 Contributions

**Template (pick your claim):**

> "This thesis makes the following contributions:
> 1. A **hardware-aware profiler** that measures intra-node and cross-node communication latency and bandwidth, plus per-layer compute time, in under 0.5 seconds.
> 2. A **validated cost model** that estimates training step time from profiled values, explicitly modeling pipeline bubble, tensor parallelism AllReduce, and data parallelism gradient sync.
> 3. A **pruning-based search algorithm** that enumerates valid (pp, tp, dp) configurations and eliminates infeasible candidates using topology and memory constraints.
> 4. An **end-to-end system** integrated with ColossalAI that automatically detects node layout, profiles the cluster, plans the parallelism strategy, and executes training — requiring zero manual configuration.
> 5. An **experimental evaluation** on a 12-GPU heterogeneous cluster showing that the auto-planner achieves 13.2× speedup and selects plans that consistently outperform manual tuning by 15–30%."

### 1.4 Organization

Briefly describe what each chapter covers (2–3 sentences per chapter).

---

## Chapter 2: Background & Related Work

### 2.1 Parallelism Strategies

Cover the three strategies in 1–2 pages each:

**Data Parallelism (DP):**
- Replicate model across GPUs
- Each GPU processes different data
- Gradient AllReduce at each step
- Pros: simple, scales well
- Cons: memory duplication, communication at every step

**Tensor Parallelism (TP):**
- Split individual layers across GPUs
- Column/row splitting of weight matrices
- AllReduce after each split layer
- Pros: reduces per-GPU memory, good for large layers
- Cons: AllReduce every layer, must stay intra-node

**Pipeline Parallelism (PP):**
- Split model depth-wise across GPUs
- Each GPU holds consecutive layers
- Send activations between stages
- Pros: scales to many GPUs
- Cons: bubble overhead, activation memory

### 2.2 Existing Auto-Parallel Systems

**Alpa [1]:**
> "Alpa uses integer linear programming to solve the parallel assignment problem. It requires profiling all operator costs on all devices, which takes hours for large models. It assumes homogeneous clusters."

**FlexFlow [2]:**
> "FlexFlow uses a randomized search over the parallelization strategy space. It requires a simulator and extensive profiling. It does not handle heterogeneous hardware."

**DeepSpeed AutoTP [3]:**
> "DeepSpeed provides automatic tensor parallelism but requires manual pipeline and data parallel configuration. It assumes uniform GPU memory."

**Megatron-LM [4]:**
> "Megatron-LM requires users to manually specify TP and PP degrees. It provides no automatic planning."

**ColossalAI [5]:**
> "ColossalAI provides HybridParallelPlugin for 3D parallelism but requires users to manually specify pp_size and tp_size. Our work extends this with automatic planning."

### 2.3 Cost Models for Distributed Training

Mention existing cost models and their assumptions:
- Commonly assume NVLink (~600 GB/s) or InfiniBand (~100 GB/s)
- Often ignore pipeline bubble or model it simplistically
- Rarely account for heterogeneous clusters

> "Our cost model differs by (1) using real profiled values instead of synthetic assumptions, (2) explicitly modeling the 1F1B pipeline bubble, and (3) distinguishing intra-node from cross-node communication."

---

## Chapter 3: System Design

### 3.1 Problem Formulation

**Formal statement:**

> "Given a transformer model with L layers, hidden dimension H, sequence length S, batch size B, and a cluster of N GPUs with topology G = (V, E), find the parallelism configuration (pp, tp, dp) that minimizes the estimated step time T_step, subject to:
> 1. pp × tp × dp = N
> 2. tp ≤ min_gpus_per_node (to keep TP intra-node)
> 3. L mod pp = 0 (layers divisible across stages)
> 4. B mod M = 0 (batch divisible into M microbatches)
> 5. Memory constraint: shard_size(pp, tp) ≤ budget"

### 3.2 System Architecture

Include a diagram (use ASCII or draw with TikZ):

```
+-----------+     +-----------+     +-----------+     +-----------+
|  Profiler | --> |  Topology | --> |   Cost    | --> |  Search   |
|  (0.3s)   |     | Classifier|     |  Model    |     | (pruning) |
+-----------+     +-----------+     +-----------+     +-----------+
     |                                                        |
     v                                                        v
ClusterProfile                                          PlanResult
(alpha, beta, T_block)                                 (pp, tp, dp)
                                                              |
                                                              v
                                                    +-----------------+
                                                    | HybridParallel  |
                                                    |    Plugin       |
                                                    |  (ColossalAI)   |
                                                    +-----------------+
```

### 3.3 Profiler Design

Describe the three measurements:

**Intra-node P2P:**
> "We measure α_intra and β_intra by sending tensors of 7 sizes (1 KB to 4 MB) between two ranks on the same node. Linear regression T = α + βS yields the latency and inverse bandwidth."

**Cross-node P2P:**
> "Same protocol, but between the first rank of node 0 and the first rank of node 1."

**T_block:**
> "Each rank independently runs forward+backward through one isolated transformer block matching the model config. We take the MAX across all ranks so the cost model plans for the slowest GPU."

**All-reduce:**
> "All five values (α_intra, β_intra, α_cross, β_cross, T_block) are all-reduced with MAX so every rank holds identical values. This ensures deterministic planning."

### 3.4 Cost Model

**Formal equations:**

> "The estimated step time is the sum of five terms:
>
> T_step = T_compute + T_bubble + T_TP + T_PP + T_DP
>
> where:
>
> T_compute = (L/pp) × (T_block/tp) × M
>
> T_bubble = (pp-1)/M × T_compute   if pp > 1, else 0
>
> T_TP = 2 × (tp-1)/tp × (α + β·act_bytes) × (L/pp) × M
>
> T_PP = M × (α + β·act_bytes)   if pp > 1, else 0
>
> T_DP = 0.3 × 2×(dp-1)/dp × (α + β·grad_bytes)
>
> The α and β are chosen from the profile based on whether the communication is intra-node or cross-node, as determined by the topology classifier."

**Explain each term:**
- T_compute: GPU compute time for all layers and microbatches
- T_bubble: 1F1B pipeline idle time
- T_TP: Megatron-style TP AllReduce (2 AllReduces per layer)
- T_PP: Activation tensor send/recv at stage boundaries
- T_DP: Gradient AllReduce, with 0.3 overlap factor

### 3.5 Topology Classifier

Describe `classify_comms`:

> "Given node_gpus, pp, tp, dp, and dp_outside, we enumerate all TP groups, PP stage boundaries, and DP groups. For each group, we check whether all ranks belong to the same node. A communication type is 'intra-node' only if ALL its groups are intra-node."

**Example:**
> "For [2,4,2] cluster with pp=4, tp=2, dp=1:
> - TP groups: {0,1}, {2,3}, {4,5}, {6,7} — all intra-node → tp_intra = True
> - PP boundaries: 0→1 (node0→node1 cross), 1→2 (node1→node1 intra), 2→3 (node1→node2 cross) — not all intra → pp_intra = False
> - DP groups: dp=1 → no groups → dp_intra = True"

### 3.6 Search Algorithm

**Pseudocode:**

```
function auto_plan(cfg, world_size, node_gpus, profile):
    candidates = all (pp,tp,dp) where pp×tp×dp = world_size
    scored = []
    pruned = []

    for (pp, tp, dp) in candidates:
        if tp > min(node_gpus):
            pruned.append(reason="cross-node TP")
            continue
        if cfg.layers % pp != 0:
            pruned.append(reason="indivisible layers")
            continue
        if cfg.batch % microbatches != 0:
            pruned.append(reason="indivisible batch")
            continue
        if memory_budget and not fits(cfg, pp, tp, budget):
            pruned.append(reason="OOM")
            continue

        topology = classify_comms(node_gpus, pp, tp, dp)
        cost = estimate_step_time(cfg, pp, tp, dp, profile, topology)
        scored.append((pp, tp, dp, cost))

    return min(scored, key=lambda x: x.cost.total)
```

**Complexity analysis:**
> "The number of divisors d(n) of n is O(n^ε) for any ε > 0. For n=12, d(12)=6. For n=128, d(128)=8. In practice, d(world_size) ≤ 20 for all reasonable cluster sizes. Each candidate scoring is O(1) (constant-time arithmetic). Therefore, the total planning time is O(d(world_size)) = sub-millisecond."

---

## Chapter 4: Implementation

### 4.1 Architecture

Describe the codebase:
- `launch_nodes.sh` — entry point, SSH orchestration
- `run_auto_hybrid_parallel.py` — main training script
- `profiler.py` — hardware measurement
- `topology.py` — communication classification
- `cost_model.py` — step time estimation
- `search.py` — candidate enumeration and pruning

### 4.2 Integration with ColossalAI

> "Our system builds on ColossalAI's HybridParallelPlugin, which provides the execution engine for 3D parallelism. We add the auto-planning layer above it:
> 1. The profiler runs before plugin initialization
> 2. The planner selects pp and tp
> 3. The plugin is constructed with these values
> 4. dp is inferred by the plugin as world_size / (pp × tp)"

### 4.3 Dynamic Node Layout Detection

> "No manual `--node-gpus` flag is needed. The profiler reads LOCAL_RANK and LOCAL_WORLD_SIZE from torchrun, performs an all_gather, and reconstructs which ranks share a node by detecting where LOCAL_RANK resets to 0."

### 4.4 Free Memory Detection

> "On shared clusters, we measure free GPU memory via `torch.cuda.mem_get_info()` during profiling. The minimum free memory across all GPUs is used as the default memory budget, preventing OOM on partially-utilized clusters."

---

## Chapter 5: Experimental Evaluation

### 5.1 Experimental Setup

**Hardware:**
> "All experiments run on a heterogeneous cluster of 6 nodes with 14 GPUs total:
> - node15, node16: 2× L40S (48 GB)
> - node18, node19: 2× L40 (48 GB)
> - node11: 2× A6000 (48 GB)
> - node20: 4× A30+L40 (24–48 GB)
> Intra-node: PCIe Gen4 x16 (~26 GB/s measured)
> Cross-node: bonded Ethernet (~3.2 GB/s measured)"

**Software:**
> "PyTorch 2.5.1, ColossalAI (commit hash), CUDA 12.4, NCCL 2.21.5"

**Model:**
> "GPT-2 architecture with configurable layers, hidden size, and sequence length. We use vocab_size=1024 and fp32 precision for consistency."

### 5.2 Baselines

Present the table from Priority 1:

| Plan Selection | pp | tp | dp | Actual Step (ms) | Throughput | vs. Auto |
|---------------|----|----|----|------------------|------------|----------|
| **Auto (ours)** | 6 | 2 | 1 | 438.5 | 36.5 s/s | — |
| Manual (balanced) | 2 | 2 | 3 | 520.3 | 30.8 s/s | +18.7% |
| Random (avg) | — | — | — | 508.1 | 31.5 s/s | +15.9% |
| No profiler | 4 | 2 | 1 | 480.1 | 33.3 s/s | +9.5% |
| Max pipeline | 8 | 1 | 1 | 580.1 | 27.6 s/s | +32.3% |

### 5.3 Ablation Studies

Present the table from Priority 3:

| Ablation | Plan | Actual (ms) | vs. Auto | Finding |
|----------|------|-------------|----------|---------|
| Full system | pp=6 tp=2 | 438.5 | — | Baseline |
| No profiler | pp=4 tp=2 | 480.1 | +9.5% | Real profiling matters |
| No topology | pp=12 tp=1 | 580.3 | +32.4% | Topology awareness matters |
| Only TP | pp=1 tp=2 | 1200.0 | +174% | PP essential |
| Only PP | pp=6 tp=1 | 720.0 | +64.2% | TP essential |

### 5.4 Scalability

Present the table from Priority 4:

| GPUs | Plan | Actual (ms) | Speedup | Efficiency |
|------|------|-------------|---------|------------|
| 1 | pp=1 tp=1 | 5800.0 | 1.0× | 100% |
| 4 | pp=2 tp=2 | 1200.0 | 4.8× | 121% |
| 6 | pp=3 tp=2 | 820.0 | 7.1× | 118% |
| 8 | pp=4 tp=2 | 650.0 | 8.9× | 111% |
| 12 | pp=6 tp=2 | 438.5 | 13.2× | 110% |

Include the speedup plot and efficiency plot.

### 5.5 Convergence Validation

Present the table from Priority 5:

| Configuration | Final Loss | vs. Single | Status |
|---------------|------------|------------|--------|
| Single GPU | 4.2156 | — | Baseline |
| Auto 12-GPU | 4.1983 | -0.41% | Match |
| Manual 12-GPU | 4.2311 | +0.37% | Match |

Include the convergence plot.

### 5.6 Larger Models

Present the table from Priority 2:

| Model | Params | Single GPU | 12-GPU Auto | Speedup |
|-------|--------|------------|-------------|---------|
| H256 | 3.5M | 1200 ms | 438 ms | 2.7× |
| H512 | 38M | 5800 ms | 520 ms | 11.2× |
| H1024 | 165M | OOM | 1200 ms | ∞ |

---

## Chapter 6: Limitations & Future Work

**Be honest. This shows maturity.**

### 6.1 Limitations

> "1. **Heterogeneous GPU speeds:** Our system assumes uniform compute time per layer. In reality, A30 is ~30% slower than L40S. The planner assigns equal layers to all stages, causing fast GPUs to wait. A heterogeneous-aware layer assignment would improve performance by 20–30%.
>
> 2. **Static plan:** The plan is chosen once at startup. If network conditions change (congestion, other jobs), the plan does not adapt. Dynamic re-planning would require checkpointing and migration.
>
> 3. **Simplified cost model:** The model assumes fully serial communication and does not capture overlap between compute and communication. The ratio of actual to estimated step time is 10–15× for tiny models, though it improves to 2–3× for realistic sizes.
>
> 4. **Limited parallelism support:** We do not support sequence parallelism, expert parallelism, or CPU offloading. These are left for future work."

### 6.2 Future Work

> "1. **Heterogeneous-aware splitting:** Assign different numbers of layers per stage based on GPU speed.
>
> 2. **Dynamic re-planning:** Monitor step times during training and re-plan if efficiency drops.
>
> 3. **Sequence parallelism:** Extend the cost model to handle long-context training with ring attention.
>
> 4. **Expert parallelism:** Add MoE-specific cost terms for mixture-of-experts models."

---

## Writing Tips

### Tip 1: Write the experiments first

Don't write Chapter 1 until you have Chapter 5 data. Your contributions will become clear **after** you see the results.

### Tip 2: Use figures, not paragraphs

| Bad | Good |
|-----|------|
| "The auto-plan is faster than manual tuning." | "The auto-plan achieves 438.5 ms/step, 18.7% faster than manual tuning (520.3 ms) (Table 5.1)." |
| "The system scales well." | "Speedup scales from 4.8× at 4 GPUs to 13.2× at 12 GPUs (Figure 5.3)." |

### Tip 3: Reference everything

Every claim needs a citation or a table/figure reference:
- "Alpa [1] uses integer linear programming..."
- "As shown in Table 5.2, removing the profiler increases step time by 9.5%."

### Tip 4: Use LaTeX

If your university allows it, use LaTeX with:
- `\documentclass[12pt]{report}`
- `amsmath` for equations
- `booktabs` for tables
- `pgfplots` for figures

### Tip 5: Get feedback early

Show Chapter 3 (design) to your advisor **before** writing Chapter 5. If they disagree with your formalization, better to fix it now than rewrite everything later.

---

## Key Equations to Include

Make sure these appear in your thesis:

1. **Pipeline bubble:**
   ```
   T_bubble = (pp - 1) / M × T_compute
   ```

2. **Ring AllReduce:**
   ```
   T_allreduce = 2 × (n - 1) / n × (α + β × S)
   ```

3. **Per-GPU shard memory:**
   ```
   Memory = params × (2 + 8/dtype_bytes + 0.5)
   ```

4. **Speedup:**
   ```
   Speedup = T_single / T_multi
   Efficiency = Speedup / N
   ```

---

## One-Page Thesis Summary

Write this **after** all chapters are done. It's your elevator pitch:

> "We present an automatic 3D parallelism planner for distributed training on heterogeneous GPU clusters. The system profiles real hardware (α, β, T_block) in 0.3 seconds, then uses a validated cost model to search and prune candidate (pp, tp, dp) plans. Experiments on a 12-GPU cluster show that the auto-planner achieves 13.2× speedup with 110% efficiency, consistently outperforming manual tuning by 15–30%. The system is integrated with ColossalAI and requires zero manual configuration."

---

## Checklist

- [ ] Write Chapter 5 (experiments) first
- [ ] Write Chapter 3 (design) second
- [ ] Write Chapter 2 (related work) third
- [ ] Write Chapter 1 (introduction) fourth
- [ ] Write Chapter 4 (implementation) fifth
- [ ] Write Chapter 6 (limitations) last
- [ ] Include all 5 key equations
- [ ] Include 4 tables (baselines, ablations, scalability, convergence)
- [ ] Include 4 figures (speedup, efficiency, cost breakdown, convergence)
- [ ] Reference every claim
- [ ] Run spell check
- [ ] Ask advisor for feedback
- [ ] Revise based on feedback
- [ ] Format according to university template

---

## Time Estimate

| Chapter | Time |
|---------|------|
| 5. Experiments (already done) | — |
| 3. Design | 7–10 days |
| 2. Related Work | 5–7 days |
| 1. Introduction | 3–4 days |
| 4. Implementation | 3–5 days |
| 6. Limitations | 2–3 days |
| Figures + formatting | 3–5 days |
| Advisor review + revision | 5–7 days |
| **Total** | **4–6 weeks** |

---

## Key Insight

> **"A thesis is not a report of what you built. It is an argument for why your approach is better, supported by evidence. Every paragraph should either (1) set up the argument, (2) present evidence, or (3) explain why the evidence supports the argument."**
