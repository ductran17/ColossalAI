# Phase 1 Auto 3D Parallel — Before vs After

## Overview

Phase 1 adds automatic PP+TP+DP planning on top of ColossalAI's existing auto TP+DP.
This document shows what changed and why.

---

## Before Phase 1: Manual Parallel Training

### What the user had to do

```python
# User had to manually decide the parallel strategy
plugin = HybridParallelPlugin(
    tp_size=2,     # manually chosen — user must know the model
    pp_size=2,     # manually chosen — user must benchmark first
    num_microbatches=4,
)
booster = Booster(plugin=plugin)
model, optimizer, _, _, _ = booster.boost(model, optimizer)
```

**Problems with this approach:**

1. **Manual guessing** — user picks `tp_size` and `pp_size` based on intuition or trial-and-error.
2. **Architecture-specific Policy classes** — `HybridParallelPlugin` requires a hand-written
   `GPT2Policy`, `LlamaPolicy`, etc. for each model. If your model doesn't have one, it won't work.
3. **No cost-aware optimization** — there is no search over alternatives. If the user picks
   PP=2, TP=2, that's what runs — even if PP=4, TP=1 would be 30% faster.
4. **No profiling** — communication costs (α, β) are not measured. The plan ignores
   actual inter-GPU bandwidth.
5. **Fixed strategy for all layers** — every layer gets the same sharding strategy regardless
   of which layers are compute-heavy vs communication-heavy.

### Code path before Phase 1

```
User code
   │
   ▼
HybridParallelPlugin(tp_size=N, pp_size=M)
   │
   ├─▶ GPT2Policy.module_policy() → hand-coded per-layer sharding rules
   │
   ├─▶ PipelineStageManager (fixed pp_size)
   │
   └─▶ ShardFormer → applies policy rules to model
          │
          └─▶ ModuleWrapper (with TP sharding baked in)
```

**No planning, no profiling, no optimization. Strategy = what the user typed.**

---

## After Phase 1: Auto 3D Parallel Planning

### What the user does now

```python
# User only provides model structure and cluster — no strategy needed
stage_module, stage_manager, plan = autoparallelize_with_pp(
    layers=layers,          # the transformer blocks
    meta_args=meta_args,    # tensor shapes only (no real data)
    num_microbatches=4,
    uniform_tp_degree=None, # None = auto-search all TP degrees
)

# The plan tells you what was chosen
print(plan.pp_size, plan.tp_size, plan.dp_size)
print(plan.stage_layer_ranges)  # which layers go to which GPU
```

### New code path

```
User code
   │
   ▼
autoparallelize_with_pp()
   │
   ├─▶ Step 1: AlphaBetaProfiler
   │      Sends probe tensors between all GPU pairs
   │      Measures α (latency) and β (inverse bandwidth) per link
   │      Result: mesh_alpha, mesh_beta — real hardware numbers
   │
   ├─▶ Step 2: _infer_devices_per_host()
   │      Detects cluster topology from α/β profile
   │      NVLink pairs (intra-node) have 10× lower β than InfiniBand (cross-node)
   │      Result: num_hosts, devices_per_host
   │
   ├─▶ Step 3: build_pipeline_plan()
   │      │
   │      ├─▶ get_submesh_choices(num_hosts, devices_per_host)
   │      │      Returns all valid (rows × cols) submesh shapes
   │      │      e.g. for 4 GPUs: [(1,1), (1,2), (1,4), (2,2)]
   │      │
   │      ├─▶ For each TP degree candidate:
   │      │      get_compute_cost()
   │      │         For each stage_size s and submesh m:
   │      │           - Build a _StageModule(layers[0:s])
   │      │           - Run initialize_model() → ILP solver (PuLP)
   │      │           - Record ILP objective as cost[k, k+s, m]
   │      │           - Broadcast to all (k, k+s, m) — homogeneous assumption
   │      │         Returns: cost table shape (K, K+1, M, 1)
   │      │
   │      └─▶ alpa_dp(cost table)
   │             Dynamic programming over (num_stages, layers, devices)
   │             f[s, k, d] = min cost: s stages, layers k..K-1, d devices
   │             Finds globally optimal layer assignment + submesh per stage
   │             Returns: PipelinePlan(pp=P, tp=T, dp=D, stage_layer_ranges=...)
   │
   ├─▶ Step 4: ProcessGroupMesh(pp_size, tp_size, dp_size)
   │      Creates all PP / TP / DP process groups
   │      Axis 0=PP, Axis 1=TP, Axis 2=DP
   │      PipelineStageManager wraps this for P2P routing
   │
   ├─▶ Step 5: DeviceMesh for ALL stages (all ranks participate)
   │      Required by PyTorch: every rank must call dist.new_group()
   │      for every group created anywhere in the job
   │
   └─▶ Step 6: initialize_model() for this rank's stage only
              Applies the chosen TP+DP sharding strategy per operator
              Returns: ModuleWrapper (TP+DP sharded, ready to train)
```

---

## Side-by-Side Comparison

| Aspect | Before (Manual) | After (Auto Phase 1) |
|--------|-----------------|----------------------|
| Strategy selection | User manually sets `tp_size`, `pp_size` | Planner searches all valid (PP, TP, DP) combinations |
| Communication costs | Ignored | Profiled with AlphaBetaProfiler before planning |
| Model architecture knowledge | Requires hand-written Policy class per model | Only needs layer list + input tensor shapes (`meta_args`) |
| Layer assignment to stages | Fixed: PP divides layers evenly | Optimal: alpa_dp minimizes pipeline makespan |
| DP degree | User sets explicitly | Auto-inferred: `dp = world_size / (pp × tp)` |
| Cost model | None | ILP solver (PuLP/COIN-BC) estimates per-operator compute cost |
| Topology awareness | None | Detects NVLink vs InfiniBand from β profile |
| Cached plans | None | Cost tables cached to disk (`--cache` flag) |
| Works with any model | No (needs Policy class) | Yes (any list of `nn.Module` layers) |

---

## Concrete Example: 2 GPUs, `--tp 2` vs auto-search

### With `--tp 2` (forced, Phase 1)

```
Planner evaluates only TP=2 submesh:
  submesh_choices filtered to: [(1, 2)]

get_compute_cost():
  stage_size=1: cost = ILP_objective(1 layer, tp=2 mesh)
  stage_size=2: cost = ILP_objective(2 layers, tp=2 mesh)
  stage_size=3: cost = ILP_objective(3 layers, tp=2 mesh)
  stage_size=4: cost = ILP_objective(4 layers, tp=2 mesh)

alpa_dp():
  Only 1 valid submesh (1,2), 2 devices total
  Only 1 stage fits (1 stage × 2 devices = 2 total)
  f[1, 0, 2] = cost[0, 4, 0] = ILP cost of all 4 layers on tp=2

Result: pp=1, tp=2, dp=1
  Stage 0: layers[0:4] on both GPUs (TP sharded)
```

### With auto-search (Phase 1, no `--tp`)

```
Planner evaluates both TP=1 and TP=2:

For TP=1:
  submesh_choices filtered to: [(1, 1)]
  2 stages possible: each stage gets 1 device
  alpa_dp evaluates:
    pp=2, tp=1, dp=1:
      stage_cost = ILP(2 layers, 1 GPU)
      total_cost = stage_cost + (B-1) × stage_cost = B × stage_cost
                 = 2 × ILP(2 layers, 1 GPU)   [B=2 microbatches]

For TP=2:
  submesh_choices filtered to: [(1, 2)]
  Only 1 stage possible: uses both devices
  alpa_dp evaluates:
    pp=1, tp=2, dp=1:
      stage_cost = ILP(4 layers, 2 GPUs)
      total_cost = stage_cost   [no bubble, pp=1]

Comparison:
  TP=2: total_cost = ILP(4 layers, 2 GPUs)
  PP=2: total_cost = 2 × ILP(2 layers, 1 GPU)

With B=2 microbatches, PP=2 has 50% bubble overhead.
TP=2 wins → planner outputs pp=1, tp=2, dp=1.
```

### Why PP=2 would win with more microbatches (B=16)

```
PP=2: total_cost = ILP(2 layers, 1 GPU) + (16-1) × ILP(2 layers, 1 GPU)
                 = 16 × ILP(2 layers, 1 GPU)
                 ≈ 16 × (ILP(4 layers, 2 GPUs) / 2)   [half layers = ~half cost]
                 ≈ 8 × ILP(4 layers, 2 GPUs)

TP=2: total_cost = ILP(4 layers, 2 GPUs)  [no bubble regardless of B]

Still TP=2 wins. PP=2 only wins at 4+ GPUs where it enables
more parallelism than TP can provide with the available device count.
```

---

## New Files Written for Phase 1

```
colossalai/auto_parallel/pipeline_shard/
├── __init__.py            ← exports PipelinePlan, build_pipeline_plan,
│                              autoparallelize_with_pp, get_compute_cost
├── compute_cost.py        ← _StageModule, get_compute_cost()
│                              Fills cost table via ILP solver per submesh
└── orchestrator.py        ← PipelinePlan dataclass
                               build_pipeline_plan() — pure Python planner
                               autoparallelize_with_pp() — distributed entry point
                               _infer_devices_per_host() — topology detection

colossalai/auto_parallel/tensor_shard/initialize.py  (modified)
    └── solve_solution() now returns (solution, objective) tuple
        so the ILP cost is available to the caller

colossalai/device/calc_pipeline_strategy.py  (3 bug fixes)
    ├── alpa_dp: shape assertion off-by-one (K vs K+1)
    ├── alpa_dp_impl: wrong DP recurrence variable (k → i)
    └── alpa_dp_impl: math.pow() returns float → cast to int for indexing
```

---

## What Phase 2 Would Add (Not Yet Implemented)

Phase 1 uses **uniform TP**: all pipeline stages share the same TP degree.
At stage boundaries, sharding matches exactly — no resharding needed.

Phase 2 allows **heterogeneous TP**: different TP per stage (e.g., Stage 0 uses
TP=4, Stage 1 uses TP=2). This needs:

- `BoundaryReshardingModule` — all_gather / split at stage transitions
- `create_fused_boundary_groups()` — custom process groups for the resharding collectives
- Extended cost model — boundary communication cost added to `alpa_dp` inputs
- Wider search space — `alpa_dp` considers all submeshes simultaneously, not
  filtered by a single TP degree

Expected gain of Phase 2 over Phase 1: 5–15% throughput depending on model.
