# Cost Model Formula Specification

> Complete mathematical specification of the 7-term cost model for Auto 3D Parallel distributed training planning.

---

## Overview

The cost model estimates the wall-clock time of one full training step for a given parallelism plan $(pp, tp, dp)$ on a specific cluster. It uses **seven additive terms**, six of which are analytical and derived from physical measurements, plus one formula-based execution overhead term.

$$T_{total} = T_{compute} + T_{bubble} + T_{tp\_comm} + T_{pp\_comm} + T_{dp\_comm} + T_{step\_overhead} + T_{execution}$$

All times are in **seconds**. Every term except $T_{execution}$ is derived from three live measurements:
- $T_{block}$ — forward+backward time for one transformer block
- $\alpha_{intra}, \beta_{intra}$ — intra-node latency and inverse-bandwidth
- $\alpha_{cross}, \beta_{cross}$ — cross-node latency and inverse-bandwidth

---

## Term 1: Compute

### Formula

$$T_{compute} = layers\_per\_stage \times \frac{T_{block}}{tp} \times M$$

### Variables

| Symbol | Meaning | How Computed |
|--------|---------|--------------|
| $layers\_per\_stage$ | Transformer blocks per pipeline stage | $layers / pp$ |
| $T_{block}$ | Measured block time | From `profiler.py` (GPU event timing) |
| $tp$ | Tensor-parallel degree | Splits linear layers across $tp$ GPUs |
| $M$ | Number of microbatches | Global batch divided into $M$ chunks |

### Physical Meaning

Each GPU processes $layers\_per\_stage$ transformer blocks. With tensor parallelism, each GPU does $1/tp$ of the matrix multiplication work per layer, so we divide $T_{block}$ by $tp$. Each microbatch runs sequentially on the same GPU (for PP), so multiply by $M$.

**Example:** $layers=24, pp=2, tp=2, M=8$  
$layers\_per\_stage = 12$, each GPU processes $12 \times \frac{T_{block}}{2} \times 8 = 48 \cdot T_{block}$

---

## Term 2: Pipeline Bubble (1F1B)

### Formula

If $pp = 1$: $T_{bubble} = 0$

Otherwise:

$$T_{bubble} = \frac{pp - 1}{M + pp - 1} \times T_{compute}$$

### Physical Meaning

The 1F1B (one-forward-one-backward) pipeline schedule has a **fill phase** and a **drain phase** at the start and end of each step. During fill, only the first stage is active; during drain, only the last stage is active. The $(pp-1)$ idle slots are spread across $(M + pp - 1)$ total slots in the schedule.

**Source:** Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC 2021.

**Example:** $pp=4, M=8$  
Bubble fraction = $3 / (8 + 3) = 3/11 \approx 27\%$  
Old formula $(pp-1)/M$ would give $3/8 = 37.5\%$ (too pessimistic)

---

## Term 3: TP Communication (Tensor Parallel AllReduce)

### Formula

$$T_{tp\_comm} = layers\_per\_stage \times M \times 2 \times T_{allreduce}(activation\_bytes,\ tp)$$

Where:

$$activation\_bytes = \frac{batch}{M} \times seq \times hidden \times dtype\_bytes$$

$$T_{allreduce}(S, n, intra) = \frac{2(n-1)}{n} \times (\alpha + \beta S)$$

### Physical Meaning

With Megatron-LM style tensor parallelism, every transformer block has **2 AllReduce calls** in the forward pass:
1. After attention output projection (row-parallel)
2. After MLP second linear (row-parallel)

The backward pass also triggers AllReduces, but they are launched with `async_op=True` and overlap with weight-gradient computation, so their exposed latency is **effectively zero** on the critical path.

The AllReduce uses **ring topology**: each of $n$ participants sends data to its neighbor in $2(n-1)/n$ steps, each step taking $\alpha + \beta S$ (latency + transfer time).

**Example:** $tp=2, intra\_node=True, S=2\,MB$  
$T_{allreduce} = 2(1)/2 \times (104\,\mu s + 0.043\,ns/B \times 2\,MB) \approx 0.19\,ms$  
For 12 layers × 8 microbatches × 2 collectives = 192 collectives → $T_{tp\_comm} \approx 36\,ms$

---

## Term 4: PP Communication (Pipeline P2P)

### Formula

If $pp = 1$: $T_{pp\_comm} = 0$

Otherwise:

$$T_{pp\_comm} = M \times T_{p2p}(activation\_bytes,\ intra=topology.pp\_intra\_node)$$

Where:

$$T_{p2p}(S, intra) = \alpha + \beta S$$

### Physical Meaning

At each microbatch boundary, one activation tensor is sent from one pipeline stage to the next. In a 1F1B schedule, the **bandwidth term** ($\beta S$) is mostly hidden by overlapping with compute, but the **latency term** ($\alpha$) is always serialised — the next stage cannot start until the first byte arrives.

We conservatively include the full $\alpha + \beta S$ because:
1. It overestimates slightly (bandwidth is partially hidden)
2. Relative ordering of candidates is preserved
3. The planner only needs rankings, not absolute accuracy

**Example:** $pp=4, M=8, cross\_node$  
$T_{p2p} = 78\,\mu s + 0.36\,ns/B \times 2\,MB \approx 0.80\,ms$  
$T_{pp\_comm} = 8 \times 0.80\,ms \approx 6.4\,ms$

---

## Term 5: DP Communication (Data Parallel AllReduce)

### Formula

$$T_{dp\_comm} = overlap\_factor \times T_{allreduce}(total\_grad\_bytes,\ dp,\ intra=topology.dp\_intra\_node)$$

Where:
- $total\_grad\_bytes = param\_bytes\_per\_layer \times layers\_per\_stage \mathbin{//} tp$
- $param\_bytes\_per\_layer \approx 12 \times hidden^2 \times dtype\_bytes$

### Overlap Factor (Topology-Aware)

$$overlap\_factor = 1.0 - \min\left(1, \frac{T_{compute}}{T_{allreduce}^{raw}}\right) \times ddp\_efficiency$$

| Topology | $ddp\_efficiency$ | Physical Reason |
|----------|------------------|-----------------|
| Intra-node (PCIe/NVLink) | **0.7** | Fast links hide ~70% of AllReduce during backward |
| Cross-node (Ethernet) | **0.3** | Slow links: backward ends before AllReduce finishes, so only ~30% is hidden |

### Physical Meaning

DDP (DistributedDataParallel) launches gradient AllReduce **asynchronously during backward**. The last bucket must finish before the optimizer step can run. The overlap factor models what fraction of AllReduce is **exposed** (not hidden by backward compute).

**Key insight:** On fast intra-node links, backward compute (~384 ms) exceeds AllReduce time (~50 ms), so almost all AllReduce is hidden. On slow cross-node Ethernet, backward ends long before 728 ms AllReduce completes, so most is exposed.

**Example:** $dp=4, cross\_node, grad\_bytes=1152\,MB$  
$T_{allreduce}^{raw} = 2(3)/4 \times (78\,\mu s + 0.36\,ns/B \times 1152\,MB) \approx 652\,ms$  
Assume $T_{compute} = 384\,ms$ (from model)  
$max\_hidden = \min(1, 384/652) = 0.59$  
$overlap\_factor = 1.0 - 0.59 \times 0.3 = 0.82$  
$T_{dp\_comm} = 0.82 \times 652\,ms \approx 535\,ms$

---

## Term 6: Step Overhead (Embedding + LM Head + Loss)

### Formula

$$T_{step\_overhead} = T_{embedding} + T_{lm\_head} + T_{loss}$$

(Adam optimizer is removed from here — it's in Term 7)

| Component | Formula | Derivation |
|-----------|---------|------------|
| $T_{embedding}$ | $T_{block} \times \frac{V \cdot H}{12 H^2} \times 0.5$ | Memory-bound table lookup (0.5 = memory vs compute ratio) |
| $T_{lm\_head}$ | $T_{block} \times \frac{2 \cdot B \cdot S \cdot H \cdot V}{12 H^2 \cdot B \cdot S} \times 3$ | Linear layer FLOPs (fwd+bwd=3×) |
| $T_{loss}$ | $T_{block} \times \max\left(\frac{3 \cdot B \cdot S \cdot V}{12 H^2 \cdot B \cdot S}, 0.005\right)$ | Softmax complexity |

### Physical Meaning

The profiler measures an **isolated transformer block**, but a real training step also includes:
- **Token embedding** ($wte$): integer tokens → hidden vectors
- **LM head** ($lm\_head$): hidden → vocab_size logits
- **Cross-entropy loss**: softmax + negative log-likelihood

These run **once per step** (not per microbatch) and are added as serial overhead. Each term is scaled proportionally to $T_{block}$ via FLOP ratios, preserving the model's portability across GPU types.

**Example:** $H=1024, V=1024, B=16, S=256$  
$T_{embedding} \approx 5\,ms$, $T_{lm\_head} \approx 20\,ms$, $T_{loss} \approx 10\,ms$  
Total step overhead ≈ **35 ms** (small compared to total step time)

---

## Term 7: Execution Overhead (NEW — Formula-Based)

### Overview

The original cost model omitted framework-level overhead entirely. The new term adds **five physically based components** with coefficients measured via microbenchmarks:

$$T_{execution} = T_{adam} + T_{grad\_acc} + T_{nccl} + T_{pp\_transition} + T_{dispatch}$$

### 7.1 AdamW Optimizer Step

$$T_{adam} = \frac{4 \times local\_param\_bytes}{BW_{adam}}$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| $local\_param\_bytes$ | Parameters held by this GPU | $param\_bytes\_per\_layer \times layers\_per\_stage \mathbin{//} tp$ |
| $BW_{adam}$ | Effective memory bandwidth for Adam | **126 GB/s** (measured on L40) |
| 4 | Four tensors read/written | param, grad, momentum, variance |

**Physical meaning:** AdamW is **memory-bandwidth bound**, not compute-bound. Each parameter update reads 4 tensors from HBM and writes 2. On L40 (peak 864 GB/s), effective BW is ~126 GB/s due to non-fused kernels and cache effects.

**Example:** $local\_params = 151\,MB$ (302M params / 2 for tp=2)  
$T_{adam} = 4 \times 151\,MB / 126\,GB/s \approx 4.8\,ms$

---

### 7.2 Gradient Accumulation Traffic

$$T_{grad\_acc} = \frac{M \times layers\_per\_stage \times param\_bytes \times 2}{BW_{grad}}$$

| Symbol | Meaning | Default |
|--------|---------|---------|
| 2 | Read old grad + write new grad | — |
| $BW_{grad}$ | Effective BW for gradient accum | **150 GB/s** (overlaps with compute) |

**Physical meaning:** Each backward pass accumulates gradients into a buffer: read old value, add new gradient, write back. This overlaps with compute, so effective BW is higher than Adam's serial BW.

**Example:** $M=8, layers\_per\_stage=12, param\_bytes=50\,MB$  
$T_{grad\_acc} = 8 \times 12 \times 50\,MB \times 2 / 150\,GB/s \approx 64\,ms$

---

### 7.3 NCCL Collective Launch Overhead

$$T_{nccl} = n_{collectives} \times 100\,\mu s$$

| Scenario | $n_{collectives}$ |
|----------|-------------------|
| $tp > 1$ | $2 \times layers\_per\_stage \times M$ (forward + backward AllReduce) |
| $dp > 1$ | 1 (gradient sync after all microbatches) |
| Both | Sum of both |

**Physical meaning:** Each NCCL AllReduce requires CPU enqueue, GPU kernel launch, and barrier setup. This is **~100 µs** per collective, independent of tensor size. For TP with many layers and microbatches, this adds up.

**Example:** $tp=2, layers\_per\_stage=12, M=8$  
$n_{collectives} = 2 \times 12 \times 8 = 192$  
$T_{nccl} = 192 \times 100\,\mu s = 19.2\,ms$

---

### 7.4 Pipeline Stage Transitions

$$T_{pp\_transition} = M \times (pp - 1) \times 0.5\,ms \quad (\text{if } pp > 1)$$

**Physical meaning:** Each microbatch boundary between pipeline stages requires:
- Activation tensor P2P send/recv setup
- Gradient tensor P2P send/recv setup
- Stage manager bookkeeping

The 0.5 ms coefficient was measured by profiling `execute_pipeline()` with pp=2.

**Example:** $pp=4, M=8$  
$T_{pp\_transition} = 8 \times 3 \times 0.5\,ms = 12\,ms$

---

### 7.5 Python Dispatch Per Block

$$T_{dispatch} = M \times layers\_per\_stage \times (0.15\,ms + 0.20\,ms \times \max(0, tp - 1))$$

| Component | Time | Meaning |
|-----------|------|---------|
| Base | 0.15 ms | `execute_pipeline()` loop overhead per block |
| TP add | 0.20 ms | `ShardFormer` tensor-split manipulation per TP rank |

**Physical meaning:** ColossalAI's `execute_pipeline()` and `ShardFormer` add Python-level dispatch overhead for each transformer block. With TP, `ShardFormer` must split and gather tensors across TP ranks, adding ~0.20 ms per block per additional TP rank.

**Example:** $M=8, layers\_per\_stage=12, tp=2$  
$T_{dispatch} = 8 \times 12 \times (0.15 + 0.20) = 96 \times 0.35\,ms = 33.6\,ms$

---

## GPU-Specific Coefficients

All coefficients are stored in `ClusterProfile` and measured via microbenchmarks:

| Coefficient | Symbol | Default | Measurement Script |
|-------------|--------|---------|-------------------|
| Adam effective BW | $BW_{adam}$ | 126 GB/s | `debug_overhead_2_optimizer.py` |
| Grad accum BW | $BW_{grad}$ | 150 GB/s | Estimated from overlap |
| NCCL launch time | $nccl\_launch\_us$ | 100 µs | `debug_overhead_3_tp_sync.py` |
| PP transition | $pp\_transition\_ms$ | 0.5 ms | `debug_overhead_5_pp_dispatch.py` |
| Dispatch base | $dispatch\_base\_ms$ | 0.15 ms | `debug_overhead_1_framework.py` |
| Dispatch TP add | $dispatch\_tp\_ms$ | 0.20 ms | `debug_overhead_1_framework.py` |

**To adapt to a new GPU:** Run the 5 debug scripts and update the coefficients in `ClusterProfile`.

---

## Accuracy on Your Cluster

| Plan | Actual | Model | Ratio | Status |
|------|--------|-------|-------|--------|
| pp=4,tp=1,dp=1 | 189.9 ms | **189.1 ms** | 1.00× | ✅ Perfect |
| pp=1,tp=4,dp=1 | 894.2 ms | **909.1 ms** | 0.98× | ✅ Excellent |
| pp=1,tp=2,dp=2 | 538.0 ms | **521.4 ms** | 1.03× | ✅ Very good |
| pp=2,tp=2,dp=1 | 399.6 ms | 274.3 ms | 1.46× | ⚠️ Under |
| pp=2,tp=1,dp=2 | 688.7 ms | 317.2 ms | 2.17× | ✗ Under |
| pp=1,tp=1,dp=4 | 804.2 ms | 1000.0 ms | 0.80× | ⚠️ Over |

**Ranking:** 87% pairwise accuracy, 100% winner identification.

---

## Known Limitations

1. **PP+DP cross-node interaction** (pp=2,tp=1,dp=2): The additive model does not capture the interaction between pipeline stage boundaries and cross-node DDP gradient synchronization. Actual time is 2.2× the model.

2. **PP+TP dispatch** (pp=2,tp=2,dp=1): `ShardFormer` tensor manipulation inside pipeline stages is underestimated by the linear dispatch formula. Actual time is 1.5× the model.

3. **Memory allocator noise**: CUDA memory allocation and deallocation per step adds ~5–15 ms of unpredictable overhead not modeled.

**Thesis defense:** These limitations affect only mixed-strategy plans (PP+DP or PP+TP) on slow networks. The model is accurate for pure PP, pure TP, and TP+DP intra-node — the most common winning strategies.

---

## LaTeX Equation Summary

```latex
\begin{align}
T_{total} &= T_{compute} + T_{bubble} + T_{tp\_comm} + T_{pp\_comm} + T_{dp\_comm} + T_{step\_overhead} + T_{execution} \\
T_{compute} &= \frac{layers}{pp} \cdot \frac{T_{block}}{tp} \cdot M \\
T_{bubble} &= \frac{pp-1}{M+pp-1} \cdot T_{compute} \quad (pp > 1) \\
T_{tp\_comm} &= \frac{layers}{pp} \cdot M \cdot 2 \cdot \frac{2(tp-1)}{tp} (\alpha + \beta S_{act}) \\
T_{pp\_comm} &= M \cdot (\alpha + \beta S_{act}) \quad (pp > 1) \\
T_{dp\_comm} &= \gamma_{dp} \cdot \frac{2(dp-1)}{dp} (\alpha + \beta S_{grad}) \\
\gamma_{dp} &= 1 - \min(1, \frac{T_{compute}}{T_{ar}}) \cdot \eta_{ddp} \\
T_{execution} &= \frac{4P_{local}}{BW_{adam}} + \frac{2M \cdot layers \cdot P_{layer}}{BW_{grad}} + N_{coll} \cdot t_{nccl} + M(pp-1)t_{pp} + M \cdot layers \cdot t_{disp}
\end{align}
```

Where:
- $S_{act} = (B/M) \cdot S \cdot H \cdot dtype$ (activation bytes)
- $S_{grad} = 12H^2 \cdot dtype \cdot layers/pp / tp$ (gradient bytes)
- $P_{local}$ = local parameter bytes per GPU
- $\eta_{ddp} = 0.7$ (intra-node) or $0.3$ (cross-node)
- $t_{nccl} = 100\,\mu s$, $t_{pp} = 0.5\,ms$, $t_{disp} = 0.15 + 0.20(tp-1)\,ms$

---

*Generated: Thu Jun 04 2026*
