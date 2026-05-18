# Auto 3D Parallel Solution — Supported Scenarios & Requirements

> What works, what doesn't, and when to use the auto-planner.

---

## 1. Supported Models

Your solution uses `HybridParallelPlugin` with `ShardFormer` for tensor parallelism. It works for any model that has a **registered ShardFormer policy**.

### ✅ Fully Supported (Out of the Box)

| Architecture | Example Models | Notes |
|-------------|----------------|-------|
| **Decoder-only Transformers** (Causal LM) | GPT-2, LLaMA/LLaMA 2, Mistral, Mixtral, Falcon, GPT-J, Bloom, OPT, Qwen2/3, Command, DeepSeek/DeepSeek-V3, ChatGLM2 | Standard autoregressive LLMs. These are the primary target. |
| **Encoder-Decoder** (Seq2Seq) | T5, Whisper, BLIP-2 | Cross-attention supported. T5 pipeline splits encoder and decoder blocks. |
| **Encoder-only** | BERT, ViT | Bidirectional or vision transformers. |

### ⚠️ Partially Supported

| Architecture | Status | Limitation |
|-------------|--------|------------|
| **Mixture of Experts (MoE)** | Policies exist for Mixtral/DeepSeek-V3 | Cost model assumes dense FFN. Doesn't account for expert routing/all-to-all overhead. |
| **Long-context / Ring Attention** | Plugin supports `sequence_parallelism_mode="ring_attn"` | Auto-planner cost model doesn't include sequence parallelism terms. |

### ❌ Not Supported

| Architecture | Why Not |
|-------------|---------|
| Custom `nn.Module` transformers | No ShardFormer policy. You'd need to write one. |
| CNNs (ResNet, ConvNeXt) | Can't pipeline spatial layers. |
| Diffusion models (Stable Diffusion U-Net) | Spatial dependencies + timestep conditioning don't map to PP. |
| RNNs / LSTMs | Sequential dependency prevents layer-wise splitting. |
| State-space models (Mamba, S4) | Single recurrent layer, can't pipeline across depth. |

---

## 2. Supported GPU Cluster Hardware

### ✅ What Works

| Requirement | Detail |
|-------------|--------|
| **Multi-node** | 2–100+ nodes via Ethernet/RoCE/InfiniBand. Tested on 5–6 nodes. |
| **Multi-GPU per node** | Any mix: [2,2,2,2,4], [8,8,8], [4,4,4,4], etc. |
| **PCIe Gen4 x16 intra-node** | Measured BW ~26 GB/s. Works well. |
| **Fast cross-node** | 3–12 GB/s+ tested. Slower networks work but favor PP over DP. |
| **Mixed GPU types** | A6000, L40S, L40, A30, etc. Profiler captures slowest GPU via MAX. **Safe but suboptimal.** |
| **Shared clusters** | GPUs partially in use by other jobs. Profiler measures free memory (if Option 3 added) or manual `--memory-gb` budget. |

### ⚠️ Works But Not Optimized

| Scenario | Issue |
|----------|-------|
| **Heterogeneous GPU speeds** | L40S + A30 in same cluster. Plugin assigns equal layers/shards to all. Fast GPUs wait for slow ones. **~20–30% performance loss.** |
| **Heterogeneous GPU memory** | One GPU has 24 GB, another 48 GB. Plugin assumes all have same memory. `--memory-gb` must be conservative (use smallest). |
| **Fat nodes** (e.g., node20 with 4 GPUs vs others with 2) | Works correctly. Topology classifier places PP boundaries intra-node when possible. |

### ❌ What Breaks

| Scenario | Why |
|----------|-----|
| **NVLink-only clusters** | Works, but conservative TP pruner (`tp > min_gpus`) may reject valid cross-node NVLink TP plans. |
| **Single-GPU nodes with `tp > 1`** | Pruned by Rule 1. TP must stay intra-node. |
| **Nodes with 0 free memory** | If memory pruning is on, all plans get rejected. If off, OOM at runtime. |
| **CPU-only nodes** | `torchrun` requires CUDA. No CPU fallback. |

---

## 3. Supported Configurations

### Parallelism Strategies

| Strategy | Supported | Notes |
|----------|-----------|-------|
| **Tensor Parallelism (TP)** | ✅ Yes | Splits Linear layers column/row-wise. Requires `heads % tp == 0`. |
| **Pipeline Parallelism (PP)** | ✅ Yes | Splits layers across stages. Requires `layers % pp == 0`. Uses 1F1B schedule. |
| **Data Parallelism (DP)** | ✅ Yes | Replicates model, syncs gradients. Inferred: `dp = world_size / (pp × tp)`. |
| **3D Parallel (TP + PP + DP)** | ✅ Yes | All three combined. Default target of auto-planner. |
| **ZeRO (FSDP)** | ⚠️ Partial | `zero_stage` 0 or 1 supported with PP. Stage 2+ not supported with PP. |
| **Expert Parallelism (EP)** | ❌ No | MoE models use standard TP/PP, not EP. |
| **Sequence Parallelism** | ⚠️ Partial | Ring attention supported in plugin but not in cost model. |

### Model Sizes

| Size Range | Status | Typical Config |
|------------|--------|----------------|
| **Tiny (< 100M params)** | ✅ Excellent for testing | hidden=256, layers=8–12. Cost model ratio ~10–15× (normal). |
| **Small (100M–1B params)** | ✅ Good | hidden=1024–2048, layers=24–48. Cost model ratio ~3–5×. |
| **Medium (1B–10B params)** | ✅ Good | hidden=4096, layers=32–80. Cost model ratio ~1.5–2×. |
| **Large (10B–100B params)** | ⚠️ Requires care | hidden=8192+, layers=80+. May need manual `num_layers_per_stage` or memory pruning. |
| **Very Large (> 100B params)** | ❌ Not tested | Likely requires additional optimizations (CPU offload, activation checkpointing) not in auto-planner. |

### Precision

| Precision | Status | Notes |
|-----------|--------|-------|
| **FP32** | ✅ Default | Stable. Memory-hungry. |
| **FP16 / BF16** | ⚠️ Supported by plugin | Cost model assumes `dtype_bytes=4`. You'd need to update `dtype_bytes=2` in `run_auto_hybrid_parallel.py`. |
| **FP8** | ❌ Not in cost model | Plugin may support it but profiler doesn't measure FP8 throughput. |

---

## 4. Requirements for Successful Training

### Hard Requirements (Must Satisfy)

1. **PyTorch + NCCL installed on all nodes** (same version)
2. **Passwordless SSH** from master node to all worker nodes
3. **Network interface `bond-local`** (or edit `NCCL_SOCKET_IFNAME` in launch script)
4. **Same ColossalAI code** on all nodes (shared filesystem or synced)
5. **`layers % pp == 0`** — pipeline stages must hold whole layers
6. **`batch % microbatches == 0`** — integer sequences per microbatch
7. **`heads % tp == 0`** — ShardFormer requires divisible attention heads

### Soft Requirements (Recommended)

1. **Same GPU type across cluster** — for optimal performance
2. **All GPUs have similar free memory** — avoids conservative planning
3. **Cross-node bandwidth > 1 GB/s** — makes PP with many stages viable
4. **Intra-node bandwidth > 20 GB/s** — makes TP efficient
5. **`microbatches >= pp`** — 1F1B scheduler requirement

---

## 5. Best-Use Scenarios

### Scenario A: Dedicated Homogeneous Cluster ⭐ IDEAL

```
Cluster: 8× A100 80GB, single node or NVLink-connected
Model:   LLaMA-7B equivalent (layers=32, hidden=4096)
Result:  Auto-planner finds pp=4 tp=2 dp=1 or similar. Near-optimal.
```

**Why ideal:** All GPUs identical, full memory available, fast intra-node. Cost model is most accurate here.

### Scenario B: Shared Heterogeneous Cluster ✅ WORKS

```
Cluster: [2×L40S, 2×L40S, 2×L40, 2×L40, 4×A30] (your cluster)
Model:   GPT-2-small (layers=12, hidden=512)
Result:  Auto-planner finds pp=6 tp=2 dp=1. Safe, ~20-30% suboptimal due to A30 bottleneck.
```

**Trade-off:** Works correctly but fast GPUs (L40S) wait for slow ones (A30). For small models, the absolute loss is small (~10 ms). For large models, consider excluding slow GPUs.

### Scenario C: Ethernet-Only Cluster ✅ WORKS

```
Cluster: 4 nodes, 2 GPUs each, 1 Gbps Ethernet
Model:   Any transformer
Result:  Auto-planner strongly favors dp=1 (no gradient sync). 
         Winner is typically pp=N tp=2 dp=1.
```

**Why it works:** The cost model correctly penalizes cross-node DP AllReduce. Even on slow networks, the winner eliminates DP entirely.

### Scenario D: Fat Node + Thin Nodes ✅ WORKS

```
Cluster: [2, 2, 2, 2, 4] (your topology)
Model:   Any transformer
Result:  Topology classifier knows node20 has 4 GPUs. 
         PP boundaries prefer to stay within node20 (intra-node = free).
```

**Advantage:** The auto-planner exploits the fat node for free PP communication.

---

## 6. Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Equal layer splitting** | Slow GPUs bottleneck pipeline | Exclude slow GPUs; use smaller models |
| **Equal tensor sharding** | Fast GPUs wait in TP AllReduce | Reduce TP degree; increase PP |
| **Static plan** | Doesn't adapt to changing network load | Re-run planner if network conditions change |
| **Cost model ratio > 10× for tiny models** | Absolute time estimates are wrong | Use estimates for ranking only; ignore absolute ms |
| **No expert parallelism** | MoE models suboptimal | Use DeepSpeed-MoE or Megatron-EP separately |
| **Memory pruning is coarse** | One threshold for all GPUs | Use per-node free memory (Option 3) or conservative manual budget |
| **No CPU offload** | Large models may OOM | Reduce model size or use ZeRO-Offload separately |
| **1F1B only** | No zero-bubble or interleaved pipeline | Plugin supports ZBV but auto-planner cost model doesn't |

---

## 7. Quick Decision Matrix

| Your Situation | Should You Use Auto-Planner? |
|----------------|------------------------------|
| GPT/LLaMA/Mistral-style decoder-only model | ✅ Yes — ideal fit |
| T5/Whisper encoder-decoder | ✅ Yes — supported |
| BERT encoder-only | ✅ Yes — supported |
| Homogeneous cluster (all same GPU) | ✅ Yes — optimal |
| Heterogeneous cluster (mixed GPU types) | ✅ Yes — safe but ~20-30% suboptimal |
| Shared cluster (partial GPU usage) | ✅ Yes — with `--memory-gb` or free-mem detection |
| Single node, 8 GPUs | ✅ Yes — finds pure TP or hybrid plan |
| Model > 100B parameters | ⚠️ Maybe — not tested, may need manual tuning |
| Custom architecture (non-transformer) | ❌ No — write ShardFormer policy first |
| CNN, Diffusion, RNN, Mamba | ❌ No — not supported |
| Need FP8 or expert parallelism | ❌ No — not in cost model |

---

## 8. Bottom Line

> **Your auto-planner + HybridParallelPlugin works best for:**
>
> **Standard decoder-only transformers (GPT, LLaMA, Mistral, etc.) on clusters of 2–20 nodes with 2–8 GPUs each, connected by Ethernet or InfiniBand, where the primary goal is automatically finding a good (pp, tp, dp) split without manual tuning.**
>
> It is **safe and functional** for heterogeneous clusters, but **not performance-optimal** when GPU speeds differ significantly. For production training of large models (>10B params) on mixed hardware, consider homogeneous node groups or manual stage assignment.
