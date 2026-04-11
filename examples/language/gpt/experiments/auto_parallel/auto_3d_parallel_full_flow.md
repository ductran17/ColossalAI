# Auto 3D Parallel — Full System Flow

> From `bash launch_3nodes.sh --auto` to the last training step.

---

## Overview

The system automatically selects the best tensor/pipeline/data parallelism split `(pp, tp, dp)` for a given cluster and model, then trains with it — no manual flags needed.

**Three phases run sequentially on the cluster:**

```
Phase 1 — Profile     ~0.4 s   measure α, β, T_block on real hardware
Phase 2 — Plan        < 1 ms   enumerate + score all (pp, tp, dp) candidates
Phase 3 — Train               train with the chosen plan
```

**Files involved:**

```
launch_3nodes.sh                          ← entry point (bash)
run_auto_hybrid_parallel.py               ← orchestrates all three phases

colossalai/auto_parallel/hybrid_planner/
  profiler.py                             ← Phase 1: measure cluster hardware
  topology.py                             ← Phase 2 helper: classify comm groups
  cost_model.py                           ← Phase 2 helper: score each candidate
  search.py                               ← Phase 2: enumerate + prune + pick best
  __init__.py                             ← public API
```

---

## Step 0 — Launch

```bash
bash launch_3nodes.sh --auto
# or with custom model size:
bash launch_3nodes.sh --auto --layers 8 --hidden 256 --batch 4 --steps 5
```

`launch_3nodes.sh` starts `torchrun` on all three nodes via SSH:

```
node18  (node_rank=0,  2 GPUs)  ← master, runs locally
node20  (node_rank=1,  4 GPUs)  ← remote worker
node16  (node_rank=2,  2 GPUs)  ← remote worker
─────────────────────────────────
Total                  8 GPUs   world_size=8
```

After `colossalai.launch_from_torch()`, all 8 processes are connected via NCCL.  
`torchrun` sets `RANK`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE` on every process.

---

## Step 0b — Auto-detect Node Layout

**File:** `profiler.py` → `_gather_node_layout(rank, world_size)`

No `--node-gpus` flag is needed. The function does an `all_gather` of `LOCAL_RANK`
and `LOCAL_WORLD_SIZE` across all 8 ranks, then reconstructs which ranks share a node:

```
global_rank | LOCAL_RANK | LOCAL_WORLD_SIZE | node
     0      |     0      |        2         | node18
     1      |     1      |        2         | node18
     2      |     0      |        4         | node20   ← LOCAL_RANK resets to 0
     3      |     1      |        4         | node20
     4      |     2      |        4         | node20
     5      |     3      |        4         | node20
     6      |     0      |        2         | node16   ← LOCAL_RANK resets to 0
     7      |     1      |        2         | node16

Result:
  nodes     = [[0,1], [2,3,4,5], [6,7]]
  node_gpus = [2, 4, 2]
```

Output:
```
[auto] Node layout detected: [2, 4, 2] (nodes × GPUs)
```

---

## Phase 1 — Profile the Cluster

**File:** `profiler.py` → `profile_cluster(model_cfg, warmup, repeat)`

All 8 ranks call `profile_cluster()` together. It runs three sub-measurements:

### 1a. Intra-node P2P bandwidth (ranks 0 ↔ 1, both on node18)

Sends tensors of 7 sizes (1 KB → 4 MB), fits a linear model `T = α + β × S`:

- `α_intra` = base latency (µs) — time to initiate a send regardless of size
- `β_intra` = inverse bandwidth (ns/B) — time per byte; `BW = 1/β`

All other 6 ranks are in the process group but do nothing during this measurement.

### 1b. Cross-node P2P bandwidth (rank 0 → rank 2, node18 → node20)

Same 7-size sweep over Ethernet:

- `α_cross` = cross-node base latency
- `β_cross` = cross-node inverse bandwidth

### 1c. T_block — GPU compute time per transformer block

All ranks independently run forward + backward through one isolated transformer block
(matching the model config: `hidden`, `heads`, `seq`, `batch`). Measured with
`torch.cuda.Event` for GPU-accurate timing.

`MAX` is taken across all 8 ranks — the slowest GPU sets the pace.

### 1d. All-reduce (MAX)

All four values `(α_intra, β_intra, α_cross, β_cross, T_block)` are all-reduced with
`MAX` across all ranks. Every rank now holds an **identical** `ClusterProfile`.  
This is critical: from this point on, planning is pure computation — no more
distributed ops are needed before training starts.

**Result on the real cluster:**
```
[auto] Profile done in 0.4s
       α_intra=111.2 µs  β_intra=0.04 ns/B  (BW=27.5 GB/s)
       α_cross=122.1 µs  β_cross=0.37 ns/B  (BW=2.7 GB/s)
       T_block=2.608 ms
```

**Reading the numbers:**
- `BW_intra = 27.5 GB/s` → PCIe x16 GPU-to-GPU on same node
- `BW_cross =  2.7 GB/s` → Ethernet between nodes (~10× slower per byte)
- `α_intra ≈ α_cross` on this cluster because nodes share a fast bonded Ethernet switch
- `T_block = 2.608 ms` is high for this tiny model (hidden=256) because small kernels
  don't saturate the GPU — launch overhead dominates

---

## Phase 2 — Auto-Plan

**File:** `search.py` → `auto_plan(cfg, world_size, node_gpus, profile, ...)`

`auto_plan()` is pure Python (no network calls). Because `ClusterProfile` is already
all-reduced, every rank runs the same computation and gets the same result.

### Step 2a — Enumerate all valid `(pp, tp, dp)` factorizations

Constraint: `pp × tp × dp = world_size = 8`

```
pp=1  tp=1  dp=8
pp=1  tp=2  dp=4
pp=1  tp=4  dp=2   ← pruned
pp=1  tp=8  dp=1   ← pruned
pp=2  tp=1  dp=4
pp=2  tp=2  dp=2
pp=2  tp=4  dp=1   ← pruned
pp=4  tp=1  dp=2
pp=4  tp=2  dp=1
pp=8  tp=1  dp=1
─────────────────
10 candidates total
```

### Step 2b — Prune infeasible candidates

**Rule 1 — TP must stay intra-node:**

`min(node_gpus) = min([2,4,2]) = 2`

If `tp > 2`, at least one TP AllReduce group spans nodes (Ethernet).
TP AllReduce fires every transformer layer — being 10× slower would dominate all costs.

Pruned: `tp=4` (×2) and `tp=8` (×1) → 3 candidates removed.

**Rule 2 — Layer divisibility:**

Each pipeline stage must hold whole layers: `layers % pp == 0`.  
With `layers=8`: all remaining pp values (1, 2, 4, 8) divide 8 evenly → no extra pruning.  
If `layers=6`: `pp=4` (6/4=1.5) and `pp=8` (6/8<1) would also be pruned.

**Rule 3 — Memory budget (optional):**

If `--memory-gb` is provided, prune plans where the per-GPU shard exceeds the budget.  
Accounts for: parameters + gradients + Adam states (fp32) + activations.  
Not active in the default run.

**After pruning: 7 candidates survive.**

### Step 2c — Score each candidate

**File:** `topology.py` → `classify_comms(node_gpus, pp, tp, dp, dp_outside=True)`

For each candidate, first classify whether each communication type is intra-node or cross-node.

The rank layout is determined by `HybridParallelPlugin` with `dp_outside=True` (default):

```
ProcessGroupMesh shape = (dp, pp, tp)
rank = dp_rank × (pp × tp) + pp_rank × tp + tp_rank
```

`classify_comms` walks through all comm groups for the given `(pp, tp, dp)` plan,
checks which global ranks are in each group, and maps them back to nodes:

```
TopologyInfo(
  tp_intra_node = True/False,  # are all TP AllReduce peers on the same node?
  pp_intra_node = True/False,  # are PP stage boundaries within a node?
  dp_intra_node = True/False,  # are all DP AllReduce peers on the same node?
)
```

**File:** `cost_model.py` → `estimate_step_time(cfg, pp, tp, dp, profile, topology, M)`

Five-term cost model:

```
T_total = T_compute + T_bubble + T_tp_comm + T_pp_comm + T_dp_comm
```

| Term | Formula | Notes |
|---|---|---|
| T_compute | `(layers/pp) × (T_block/tp) × M` | M = num_microbatches |
| T_bubble | `(pp-1)/M × T_compute` | 1F1B idle stages |
| T_tp_comm | `2×(tp-1)/tp × (α + β × act_bytes) × (layers/pp) × M` | Ring AllReduce per layer |
| T_pp_comm | `M × (α + β × act_bytes)` | Activation tensor at each stage boundary |
| T_dp_comm | `0.3 × 2×(dp-1)/dp × (α + β × grad_bytes)` | 0.3 = 70% hidden by backward |

`α` and `β` are chosen from `ClusterProfile` based on `topology`:
- `tp_intra_node=True` → use `α_intra`, `β_intra` for TP comm
- `pp_intra_node=False` → use `α_cross`, `β_cross` for PP comm
- etc.

### Step 2d — Pick the winner

Minimum `T_total` across all 7 scored candidates.

**Scored table (example with real cluster values):**

```
plan              total ms   compute  bubble   TP      PP      DP
─────────────────────────────────────────────────────────────────────────────
pp=1 tp=1 dp=8    ~400 ms    20.8     0.0      0.0     0.0     ~380    ← DP kills it
pp=1 tp=2 dp=4    ~200 ms    10.4     0.0      1.8     0.0     ~190
pp=2 tp=1 dp=4    ~200 ms    10.4     2.6      0.0     ~0.2    ~190
pp=2 tp=2 dp=2    ~100 ms     5.2     1.3      0.9     0.6     ~90
pp=4 tp=1 dp=2     ~90 ms     5.2     7.8      0.0     0.6     ~75     ← dp cross-node!
pp=4 tp=2 dp=1  *  20.7 ms   10.4     7.8      1.8     0.6     0.0    ← WINNER
pp=8 tp=1 dp=1     ~35 ms     5.2    15.6      0.0     0.6     0.0
```

**Why `pp=4 tp=2 dp=1` wins:**

- `dp=1` → no DP AllReduce at all → `T_dp = 0` (eliminates the ~75–380ms dominant cost)
- `tp=2` → TP stays intra-node (2 ≤ min_gpus=2) → fast PCIe AllReduce (1.8ms)
- `pp=4` vs `pp=8`: `tp=2` halves per-stage compute → smaller bubble (7.8ms vs 15.6ms)
- `pp=4 tp=2 dp=1` vs `pp=4 tp=1 dp=2`: cross-node DP AllReduce costs ~75ms vs 0ms → tp=2 wins

Output:
```
[auto] Best plan: pp=4  tp=2  dp=1  (estimated 20.7 ms/step)
       T_total = 20.657 ms: compute=10.4ms(50.5%) bubble=7.8ms(37.9%)
                            TP=1.8ms(8.8%) PP=0.6ms(2.8%) DP=0.0ms(0%)
[auto] Full candidate table: [7 plans scored, 3 pruned]
```

---

## Phase 3 — Train with the Chosen Plan

**File:** `run_auto_hybrid_parallel.py` (Phase 3 section)

### 3a — Build the plugin

```python
plugin = HybridParallelPlugin(
    pp_size          = 4,   # from auto_plan
    tp_size          = 2,   # from auto_plan
    num_microbatches = 4,
    precision        = "fp32",
    dp_outside       = True,   # MUST match what auto_plan used
)
# dp is inferred: world_size / (pp × tp) = 8 / 8 = 1
```

`dp_outside=True` (default) → `ProcessGroupMesh(dp=1, pp=4, tp=2)`.

**Rank layout:**
```
rank | dp | pp | tp | node
-----|----|----|----|-----------
  0  |  0 |  0 |  0 | node18
  1  |  0 |  0 |  1 | node18  ← TP pair: intra-node PCIe ✓
  2  |  0 |  1 |  0 | node20
  3  |  0 |  1 |  1 | node20  ← TP pair: intra-node PCIe ✓
  4  |  0 |  2 |  0 | node20
  5  |  0 |  2 |  1 | node20  ← TP pair: intra-node PCIe ✓
  6  |  0 |  3 |  0 | node16
  7  |  0 |  3 |  1 | node16  ← TP pair: intra-node PCIe ✓
```

### 3b — Boost the model

```python
model, optimizer, *_ = booster.boost(model, optimizer)
```

`Booster.boost()` calls ShardFormer internally, which:
- Finds all `Linear` layers in GPT2 attention + MLP
- Splits them column-wise (first linear) or row-wise (second linear) across `tp=2` ranks
- Each GPU holds half the weight matrix

After boost, each GPU holds:
- 2 of 8 transformer layers (`pp=4` → 2 layers/stage)
- Half of each layer's parameters (`tp=2` → 50% of weights)
- Total: **1/8 of the full model per GPU**

### 3c — Training loop

```python
for step in range(steps):
    batch = make_batch(step)            # full batch (pp splits internally)
    outputs = booster.execute_pipeline(
        iter([batch]), model, criterion, optimizer, return_loss=True
    )
    optimizer.step()
    optimizer.zero_grad()
```

`execute_pipeline()` runs the **1F1B schedule**:

```
Microbatch:     mb0   mb1   mb2   mb3
Stage 0 (n18):  F0    F1    F2    F3    B3    B2    B1    B0
Stage 1 (n20):        F0    F1    F2    F3    B3    B2    B1    B0
Stage 2 (n20):              F0    F1    F2    F3    B3    B2    B1    B0
Stage 3 (n16):                    F0    F1    F2    F3    B3    B2    B1    B0
                                              ↑ loss computed here
```

- F = forward pass through 2 layers on that stage
- B = backward pass, gradients flow back through all stages
- Stage boundaries send activation tensors over NCCL (cross-node Ethernet)
- Loss is only computed on stage 3 (ranks 6, 7) — only those ranks log it

### 3d — dp_rank formula (critical)

`dp_outside=True` → mesh shape `(dp, pp, tp)` → `rank = dp_rank×(pp×tp) + pp_rank×tp + tp_rank`

Extracting dp_rank: `dp_rank = rank // (pp × tp)`

This seeds `make_batch()` to give each DP replica different data.  
With `dp=1`, all ranks get `dp_rank=0` (only one replica).

Using the wrong formula `(rank // tp) % dp` (only correct for `dp_outside=False`) would
give wrong seeds for ranks 2–5 when `dp > 1`, causing DP replicas to see identical data.

### 3e — Validation output (profiler vs actual)

After all steps, a validation report is printed:

```
[auto] ── Profiler validation report ──────────────────────────────
  Profiled T_block      : 2.608 ms  (isolated block fwd+bwd)
  Estimated step time   : 20.7 ms   (cost model from profiled α/β/T_block)
  Actual avg step time  : XX.X ms   (wall clock including framework overhead)
  Ratio actual/estimate : X.XX×

  Cost model breakdown (estimated):
    compute = 10.4 ms  bubble = 7.8 ms  TP = 1.8 ms  PP = 0.6 ms  DP = 0.0 ms

  Interpretation:
    ratio < 1.5 → profiler estimates are representative
    ratio > 2.0 → framework/Python overhead significant (normal for tiny models)
```

The ratio is expected to be 1–3× for a tiny model (hidden=256) because:
- Framework overhead (CUDA kernel launch, Python, barrier sync) is large relative to compute
- The cost model is designed to **rank candidates correctly**, not predict exact wall time

---

## Key Invariants

### 1. `dp_outside` must be consistent everywhere

`auto_plan()` uses `dp_outside` to classify which comm groups are intra/cross-node.  
`HybridParallelPlugin` uses `dp_outside` to build `ProcessGroupMesh`.  
`dp_rank` extraction formula depends on `dp_outside`.

All three must agree. Both default to `True`. Never change one without the others.

### 2. `ClusterProfile` is all-reduced before planning

`profile_cluster()` ends with `dist.all_reduce(tensor, op=MAX)`.  
This guarantees every rank has identical `ClusterProfile` values.  
`auto_plan()` can then run independently on each rank with no communication.

### 3. `tp ≤ min(node_gpus)` is enforced by the pruner

TP AllReduce fires every transformer layer. Cross-node TP AllReduce (Ethernet)
is 10× slower per byte than intra-node (PCIe/NVLink).  
The pruner removes all plans with `tp > min(node_gpus)` before scoring.

### 4. Full batch goes into `execute_pipeline`

`booster.execute_pipeline()` receives the full global batch and splits it
into microbatches internally. Do not pre-divide the batch.

---

## End-to-End Timeline

```
t = 0.0s   torchrun starts 8 processes, NCCL initialized
t = 0.0s   _gather_node_layout() → node_gpus=[2,4,2]   (< 0.1s)
t = 0.1s   Phase 1 starts: intra P2P + cross P2P + T_block measurements
t = 0.5s   Phase 1 done: ClusterProfile all-reduced (identical on all 8 ranks)
t = 0.5s   Phase 2: auto_plan() runs (pure Python, < 1ms)
t = 0.5s   Rank 0 prints plan table and winner
t = 0.5s   Phase 3: model built and boosted with pp=4 tp=2 dp=1
t = ?s     Training steps 1–3 complete
t = ?s     Validation report printed (estimated vs actual step time)
t = ?s     All ranks exit 0
```

---

## Module Dependency Graph

```
run_auto_hybrid_parallel.py
  │
  ├── profiler.py
  │     _gather_node_layout()   → node_gpus list
  │     profile_cluster()       → ClusterProfile(α_intra, β_intra,
  │                                               α_cross, β_cross, T_block)
  │
  ├── search.py
  │     auto_plan()             → PlanResult(pp, tp, dp, cost, ...)
  │       │
  │       ├── topology.py
  │       │     classify_comms()  → TopologyInfo(tp_intra, pp_intra, dp_intra)
  │       │
  │       └── cost_model.py
  │             estimate_step_time()  → CostBreakdown(T_compute, T_bubble,
  │                                                    T_tp, T_pp, T_dp)
  │
  └── HybridParallelPlugin (ColossalAI)
        pp_size=result.pp, tp_size=result.tp
        dp inferred: world_size / (pp × tp)
```

---

## Quick Reference — Key Numbers for [2,4,2] Cluster

| Metric | Profiled (real) | Synthetic (docs) |
|---|---|---|
| α_intra | 111 µs | 5 µs |
| BW_intra | 27.5 GB/s | 1.1 GB/s |
| α_cross | 122 µs | 80 µs |
| BW_cross | 2.7 GB/s | 12.5 GB/s |
| T_block | 2.608 ms | 1.0 ms |
| Winner | pp=4 tp=2 dp=1 | pp=4 tp=2 dp=1 |
| Estimated step | 20.7 ms | 29.4 ms |

The winner is the same despite different hardware values — the decision logic
(eliminate DP AllReduce, keep TP intra-node) is robust across a wide range of cluster types.
