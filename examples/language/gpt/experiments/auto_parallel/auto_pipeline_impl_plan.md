# Auto Pipeline Parallelism — Implementation Plan for ColossalAI

## Context

ColossalAI has ported Alpa's inter-op DP solver (`alpa_dp`) and all supporting
components, but the pipeline planning is not wired together. The goal is to
implement the missing `pipeline_shard/` module so that `autoparallelize()` can
automatically find the best pipeline stage assignment in addition to the
intra-op (TP/DP) sharding it already does.

---

## What Already Exists (do not re-implement)

| Component | File | Status |
|---|---|---|
| `alpa_dp()` / `alpa_dp_impl()` | `colossalai/device/calc_pipeline_strategy.py` | ✅ done |
| `get_submesh_choices()` | same file | ✅ done |
| `AlphaBetaProfiler` | `colossalai/device/alpha_beta_profiler.py` | ✅ done |
| `DeviceMesh` | `colossalai/device/device_mesh.py` | ✅ done |
| `initialize_model()` (intra-op ILP) | `colossalai/auto_parallel/tensor_shard/initialize.py` | ✅ done |
| `GraphAnalyser` (liveness/memory) | `colossalai/auto_parallel/tensor_shard/solver/graph_analysis.py` | ✅ done |
| `meta_profiler` (per-op cost model) | `colossalai/auto_parallel/meta_profiler/` | ✅ done |
| Pipeline runtime (1F1B, interleaved, ZBV) | `colossalai/pipeline/schedule/` | ✅ done |
| `PipelineStageManager` | `colossalai/pipeline/stage_manager.py` | ✅ done |

## What Is Missing (must build)

```
colossalai/auto_parallel/pipeline_shard/
├── __init__.py               ← currently empty, needs exports
├── compute_cost.py           ← Step 1: fill the cost table
├── layer_partition.py        ← Step 2: FX graph → layer list
└── orchestrator.py           ← Step 3: wire everything together
```

And a new top-level entry point:
```
colossalai/auto_parallel/tensor_shard/initialize.py
    autoparallelize_with_pp()  ← Step 4: new public API
```

And a test script:
```
examples/language/gpt/experiments/auto_parallel/test_auto_pipeline.py
```

---

## Data Flow (target)

```
autoparallelize_with_pp(model, meta_args, num_pp_stages, num_devices)
        │
        ▼
[Step A] layer_partition.py
  ColoTracer traces model → FX graph
  Identify "pipeline-able" layer boundaries (e.g. transformer blocks)
  Split graph into K candidate layer slices
        │
        ▼
[Step B] compute_cost.py  get_compute_cost()
  For each (start_layer, end_layer, submesh_shape):
      sub_graph = slice FX graph [start..end]
      sub_mesh  = DeviceMesh carved from full mesh
      cost[start, end, mesh_id] = run initialize_model(sub_graph, sub_mesh)
                                    → read TrainCycleItem.total from solution
        │
        ▼
[Step C] alpa_dp()  [already in calc_pipeline_strategy.py]
  Input : cost table (num_layers, num_layers, num_submeshes, 1)
  Output: list of ((start, end), submesh_id, config_id) per stage
        │
        ▼
[Step D] orchestrator.py  build_pipeline_plan()
  Assign each stage's layer range to a set of ranks
  Build PipelineStageManager with the chosen stage assignment
  Call initialize_model() on each stage's sub-graph with its assigned submesh
        │
        ▼
  Return: list of ModuleWrapper (one per stage) + PipelineStageManager
```

---

## Step-by-Step Implementation

---

### Step 1 — `pipeline_shard/layer_partition.py`

**Purpose**: convert a traced FX graph into a flat list of "layers"
(coarse-grained pipeline granularity units, equivalent to Alpa's
`AutoLayerOption`).

**Key function**:

```python
def get_pipeline_layers(
    gm: ColoGraphModule,
    split_points: List[str],          # node names where to cut (e.g. module boundaries)
) -> List[ColoGraphModule]:
    """
    Split gm at split_points and return a list of sub-GraphModules,
    one per pipeline layer candidate.

    split_points can be derived automatically by scanning for repeated
    block patterns in the FX graph (e.g. every nn.TransformerEncoderLayer).
    """
```

**Implementation notes**:

1. Use `torch.fx.passes.split_module` (already vendored in Merak, also in PyTorch)
   to split the graph at module boundaries.
2. For transformer models: detect repeated submodule names
   (`model.layers.0`, `model.layers.1`, ...) and use each as one layer.
3. Return a `List[ColoGraphModule]` where `layers[0]` takes the original
   `meta_args` as input and `layers[i+1]` takes the output of `layers[i]`.
4. Also return `input_shapes: List[torch.Size]` — the activation tensor shape
   at each boundary, needed for cost estimation.

**Automatic split-point detection** (if no manual split_points given):

```python
def detect_split_points(gm: ColoGraphModule) -> List[str]:
    """
    Walk the FX graph and identify repeating submodule call sites.
    Return the node name just before each new block starts.
    """
```

---

### Step 2 — `pipeline_shard/compute_cost.py`

**Purpose**: populate the `compute_cost` table that `alpa_dp` needs.

**Key function**:

```python
def get_compute_cost(
    layers: List[ColoGraphModule],        # from Step 1
    input_shapes: List[torch.Size],       # activation shape at each boundary
    submesh_choices: List[Tuple[int,int]],# from get_submesh_choices()
    full_device_mesh: DeviceMesh,         # the entire cluster mesh
    num_microbatches: int,
    memory_budget: float = -1.0,
    cache_path: str = None,               # if set, save/load .pt cache file
) -> np.ndarray:
    """
    Returns:
        compute_cost: np.ndarray shape (K, K, M, 1)
            K = len(layers),  M = len(submesh_choices)
        compute_cost[k, i, m, 0] = estimated time to run layers[k:i]
                                    on submesh m with best intra-op plan
    """
```

**Implementation notes**:

1. **Cache check**: if `cache_path` exists, `np.load()` and return early.

2. **Enumerate**: nested loops over `(k, i, m)` —
   `k` = start layer index, `i` = end layer index (k < i), `m` = submesh index.
   Total combinations: `O(K² × M)`.  For GPT-2 with K=12 layers and M=4
   submesh choices that is 12×12×4/2 ≈ 288 calls — manageable.

3. **Sub-graph construction**: merge `layers[k:i]` into one `nn.Sequential`
   (or re-stitch the FX graphs) so `initialize_model()` can trace it.

4. **Sub-mesh construction**: carve a sub-DeviceMesh from `full_device_mesh`
   using `np.prod(submesh_choices[m])` devices.

5. **Call `initialize_model()`**:
   ```python
   from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
   _, solution = initialize_model(
       sub_model, meta_args_for_slice,
       sub_mesh,
       memory_budget=memory_budget,
       return_solution=True,
   )
   ```
   Extract the total cost from the solver's objective value:
   ```python
   cost = solver.last_objective  # float, sum of compute + comm costs
   compute_cost[k, i, m, 0] = cost
   ```

6. **Fallback — analytical estimate via meta_profiler**: if `initialize_model()`
   fails (e.g. sub-graph too small, or ILP infeasible), fall back to summing
   per-node `TrainCycleItem.total` from the meta-profiler without solving the ILP.

7. **Parallel execution** (optional optimisation): since each `(k,i,m)` call is
   independent, use `concurrent.futures.ProcessPoolExecutor` to run multiple
   ILP solves in parallel on CPU.

8. **Save cache**:
   ```python
   np.save(cache_path, compute_cost)
   ```

---

### Step 3 — `pipeline_shard/orchestrator.py`

**Purpose**: wire `get_compute_cost()` → `alpa_dp()` → runtime setup.

**Key function**:

```python
def build_pipeline_plan(
    model: nn.Module,
    meta_args: Dict[str, Any],
    num_devices: int,
    num_microbatches: int,
    num_hosts: int = 1,
    num_devices_per_host: int = None,   # defaults to num_devices
    memory_budget: float = -1.0,
    cache_path: str = None,
    mode: str = "new",                  # submesh enumeration mode
) -> "PipelinePlan":
    """
    Returns a PipelinePlan dataclass containing:
      - stage_layer_ranges: List[Tuple[int,int]]  e.g. [(0,4),(4,8),(8,12)]
      - stage_submesh_ids: List[int]
      - stage_device_groups: List[List[int]]       rank lists per stage
      - best_cost: float
    """
```

**`PipelinePlan` dataclass**:

```python
@dataclass
class PipelinePlan:
    stage_layer_ranges: List[Tuple[int, int]]  # (start, end) layer indices
    stage_submesh_ids: List[int]               # index into submesh_choices
    stage_device_groups: List[List[int]]       # physical rank IDs per stage
    submesh_choices: List[Tuple[int, int]]
    best_cost: float
```

**Implementation**:

```python
def build_pipeline_plan(...) -> PipelinePlan:
    # 1. Trace model and get layers
    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)
    K = len(layers)

    # 2. Enumerate submesh choices
    ndph = num_devices_per_host or num_devices
    submesh_choices = get_submesh_choices(num_hosts, ndph, mode=mode)

    # 3. Build cost table
    full_mesh = initialize_device_mesh(world_size=num_devices)
    cost_table = get_compute_cost(
        layers, input_shapes, submesh_choices, full_mesh,
        num_microbatches, memory_budget, cache_path
    )

    # 4. Run DP solver
    best_cost, solution = alpa_dp(
        num_layers=K,
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        submesh_choices=submesh_choices,
        num_autosharding_configs=1,
        compute_cost=cost_table,
    )

    # 5. Parse solution into PipelinePlan
    stage_layer_ranges = []
    stage_submesh_ids = []
    stage_device_groups = []
    device_cursor = 0
    for (start, end), mesh_id, _ in solution:
        n_devs = int(np.prod(submesh_choices[mesh_id]))
        ranks = list(range(device_cursor, device_cursor + n_devs))
        stage_layer_ranges.append((start, end))
        stage_submesh_ids.append(mesh_id)
        stage_device_groups.append(ranks)
        device_cursor += n_devs

    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        stage_submesh_ids=stage_submesh_ids,
        stage_device_groups=stage_device_groups,
        submesh_choices=submesh_choices,
        best_cost=best_cost,
    )
```

---

### Step 4 — New public API in `tensor_shard/initialize.py`

Add `autoparallelize_with_pp()` alongside the existing `autoparallelize()`:

```python
def autoparallelize_with_pp(
    model: nn.Module,
    meta_args: Dict[str, Any],
    num_pp_stages: int,
    num_devices: int,
    num_microbatches: int,
    schedule_type: str = "1f1b",        # "1f1b" | "interleaved" | "zero_bubble"
    memory_budget: float = -1.0,
    cache_path: str = None,
) -> Tuple[List[ModuleWrapper], PipelineStageManager]:
    """
    End-to-end auto 3D parallelism including pipeline stages.

    Returns:
        stage_modules: one ModuleWrapper per pipeline stage
                       (only the module for the current rank's stage is needed)
        stage_manager: configured PipelineStageManager for use in training loop
    """
    from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
    from colossalai.auto_parallel.pipeline_shard.layer_partition import get_pipeline_layers
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_model, initialize_device_mesh
    from colossalai.pipeline.stage_manager import PipelineStageManager

    # 1. Find optimal pipeline plan (runs on all ranks, deterministic)
    plan = build_pipeline_plan(
        model, meta_args, num_devices, num_microbatches,
        memory_budget=memory_budget, cache_path=cache_path,
    )

    # 2. Determine current rank's stage
    rank = dist.get_rank()
    my_stage = _rank_to_stage(rank, plan)

    # 3. Apply intra-op sharding only to current rank's stage sub-graph
    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, _ = get_pipeline_layers(gm)
    start, end = plan.stage_layer_ranges[my_stage]
    stage_sub_model = _merge_layers(layers[start:end])

    sub_mesh = _build_submesh(plan, my_stage)
    stage_module = initialize_model(
        stage_sub_model, meta_args, sub_mesh,
        memory_budget=memory_budget,
    )

    # 4. Build PipelineStageManager
    pg_mesh = ProcessGroupMesh(num_pp_stages, ...)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)

    return stage_module, stage_manager
```

**Helper functions to implement** (in the same file or `orchestrator.py`):

- `_rank_to_stage(rank, plan) -> int` — maps a global rank to its pipeline stage
- `_merge_layers(layers: List[ColoGraphModule]) -> ColoGraphModule` — stitches
  a slice of layer sub-graphs back into one traceable module
- `_build_submesh(plan, stage_id) -> DeviceMesh` — creates the sub-DeviceMesh
  for a given stage using the ranks in `plan.stage_device_groups[stage_id]`

---

### Step 5 — `pipeline_shard/__init__.py`

```python
from .orchestrator import build_pipeline_plan, PipelinePlan
from .compute_cost import get_compute_cost
from .layer_partition import get_pipeline_layers, detect_split_points

__all__ = [
    "build_pipeline_plan",
    "PipelinePlan",
    "get_compute_cost",
    "get_pipeline_layers",
    "detect_split_points",
]
```

---

## Step 6 — Test Script

**File**: `examples/language/gpt/experiments/auto_parallel/test_auto_pipeline.py`

### Test 1 — Unit test: `alpa_dp` with synthetic cost table

```python
# No GPU needed. Verify the DP solver produces a valid partition.
def test_alpa_dp_synthetic():
    import numpy as np
    from colossalai.device.calc_pipeline_strategy import alpa_dp, get_submesh_choices

    K = 8           # layers
    N = 4           # devices
    B = 4           # microbatches
    submesh_choices = [(1, 1), (1, 2), (1, 4)]
    M = len(submesh_choices)

    # uniform cost: every (k,i,m) slice costs proportional to (i-k)
    cost = np.zeros((K, K, M, 1), dtype=np.float32)
    for k in range(K):
        for i in range(k+1, K+1):
            for m in range(M):
                cost[k, i-1, m, 0] = (i - k) * 1.0  # note: alpa_dp indexing

    # Actually alpa_dp expects shape (num_layers, num_layers, M, C)
    # where cost[k,i,m,c] = cost of assigning layers k..i to stage on submesh m
    cost_full = np.zeros((K, K, M, 1), dtype=np.float32)
    for k in range(K):
        for i in range(k, K):
            for m in range(M):
                cost_full[k, i, m, 0] = (i - k + 1) * 1.0

    best_cost, solution = alpa_dp(
        num_layers=K,
        num_devices=N,
        num_microbatches=B,
        submesh_choices=submesh_choices,
        num_autosharding_configs=1,
        compute_cost=cost_full,
    )
    assert solution is not None, "DP returned no solution"
    # Verify all K layers are covered
    covered = set()
    for (start, end), mesh_id, _ in solution:
        for l in range(start, end):
            covered.add(l)
    assert covered == set(range(K)), f"Not all layers covered: {covered}"
    print(f"[PASS] test_alpa_dp_synthetic: cost={best_cost:.2f}, stages={len(solution)}")
```

### Test 2 — Unit test: layer partition on GPT-2

```python
# Requires: pip install transformers
# No GPU needed (CPU meta-device tracing).
def test_layer_partition_gpt2():
    from transformers import GPT2Model, GPT2Config
    from colossalai.auto_parallel.pipeline_shard.layer_partition import (
        get_pipeline_layers, detect_split_points
    )
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer

    config = GPT2Config(n_layer=4, n_head=4, n_embd=64)
    model = GPT2Model(config)
    meta_args = {"input_ids": torch.zeros(1, 16, dtype=torch.long)}

    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)

    assert len(layers) > 1, "Should produce more than 1 pipeline layer"
    assert len(layers) == len(input_shapes)
    print(f"[PASS] test_layer_partition_gpt2: {len(layers)} layers detected")
    for i, (layer, shape) in enumerate(zip(layers, input_shapes)):
        print(f"  layer {i}: input_shape={shape}")
```

### Test 3 — Integration test: `get_compute_cost` on GPT-2 (requires 1+ GPU)

```python
# Run with: torchrun --nproc_per_node=2 test_auto_pipeline.py --test compute_cost
def test_compute_cost_gpt2():
    import os
    import torch.distributed as dist
    from transformers import GPT2Model, GPT2Config
    from colossalai.auto_parallel.pipeline_shard.layer_partition import get_pipeline_layers
    from colossalai.auto_parallel.pipeline_shard.compute_cost import get_compute_cost
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_device_mesh
    from colossalai.device.calc_pipeline_strategy import get_submesh_choices
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()

    config = GPT2Config(n_layer=4, n_head=4, n_embd=64)
    model = GPT2Model(config)
    meta_args = {"input_ids": torch.zeros(1, 16, dtype=torch.long)}

    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)

    submesh_choices = get_submesh_choices(
        num_hosts=1, num_devices_per_host=world_size, mode="new"
    )
    full_mesh = initialize_device_mesh(world_size=world_size)
    cost_table = get_compute_cost(
        layers, input_shapes, submesh_choices, full_mesh,
        num_microbatches=4,
        cache_path="/tmp/gpt2_cost_cache.npy",
    )

    K = len(layers)
    M = len(submesh_choices)
    assert cost_table.shape == (K, K, M, 1), f"Wrong shape: {cost_table.shape}"
    assert not np.all(np.isinf(cost_table)), "All costs are inf — solver failed"
    print(f"[PASS] test_compute_cost_gpt2: cost_table shape={cost_table.shape}")
    print(f"  min finite cost = {np.nanmin(cost_table[cost_table < np.inf]):.4f}")
```

### Test 4 — End-to-end: `build_pipeline_plan` on GPT-2 (requires 2+ GPUs)

```python
# Run with: torchrun --nproc_per_node=4 test_auto_pipeline.py --test e2e
def test_e2e_pipeline_plan():
    from transformers import GPT2Model, GPT2Config
    from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan

    dist.init_process_group("nccl")

    config = GPT2Config(n_layer=8, n_head=4, n_embd=128)
    model = GPT2Model(config)
    meta_args = {"input_ids": torch.zeros(1, 32, dtype=torch.long)}

    plan = build_pipeline_plan(
        model=model,
        meta_args=meta_args,
        num_devices=dist.get_world_size(),
        num_microbatches=4,
        cache_path="/tmp/gpt2_e2e_cache.npy",
    )

    if dist.get_rank() == 0:
        print(f"[PASS] Pipeline plan found: cost={plan.best_cost:.4f}")
        for s, ((start, end), mesh_id, ranks) in enumerate(zip(
            plan.stage_layer_ranges, plan.stage_submesh_ids, plan.stage_device_groups
        )):
            mesh = plan.submesh_choices[mesh_id]
            print(f"  Stage {s}: layers [{start},{end}) | mesh={mesh} | ranks={ranks}")
```

### Test 5 — Full training loop test (requires 4 GPUs)

```python
# torchrun --nproc_per_node=4 test_auto_pipeline.py --test training
def test_training_loop():
    from transformers import GPT2LMHeadModel, GPT2Config
    from colossalai.auto_parallel.tensor_shard.initialize import autoparallelize_with_pp
    from colossalai.pipeline.schedule.one_f_one_b import OneForwardOneBackwardSchedule

    dist.init_process_group("nccl")
    rank = dist.get_rank()

    config = GPT2Config(n_layer=8, n_head=4, n_embd=128)
    model = GPT2LMHeadModel(config)
    meta_args = {
        "input_ids": torch.zeros(2, 32, dtype=torch.long),
        "labels":    torch.zeros(2, 32, dtype=torch.long),
    }

    stage_module, stage_manager = autoparallelize_with_pp(
        model=model,
        meta_args=meta_args,
        num_pp_stages=2,
        num_devices=4,
        num_microbatches=4,
        schedule_type="1f1b",
    )

    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)
    schedule = OneForwardOneBackwardSchedule(stage_manager, num_microbatches=4)

    # Fake data
    batch = {
        "input_ids": torch.randint(0, 50257, (2, 32)).cuda(),
        "labels":    torch.randint(0, 50257, (2, 32)).cuda(),
    }

    def fwd_fn(batch):
        return stage_module(**batch)

    schedule.forward_backward_step(
        model=stage_module,
        data_iter=iter([batch]),
        criterion=lambda outputs, _: outputs.loss,
        optimizer=optimizer,
        return_loss=True,
    )
    optimizer.step()
    optimizer.zero_grad()

    if rank == 0:
        print("[PASS] test_training_loop: one step completed")
```

---

## Implementation Order

Implement in this sequence — each step is independently testable:

```
1. layer_partition.py       → Test 2 (CPU only, no distributed)
2. compute_cost.py          → Test 3 (needs 1+ GPU, uses existing initialize_model)
3. orchestrator.py          → Test 4 (needs 2+ GPUs)
4. autoparallelize_with_pp  → Test 5 (needs 4 GPUs, full training)
```

---

## Key Design Decisions

### A. How to represent `compute_cost[k, i, m, 0]`

`alpa_dp` indexing: `compute_cost[k, i, m, c]` means the cost of running
**layers k through i-1** (i.e. `layers[k:i]`). The diagonal `k==i` should be
`np.inf` (cannot have zero-layer stage). Stages must cover all K layers exactly.

### B. Cost value to use from `initialize_model()`

`solve_solution()` in `initialize.py` calls `Solver` whose `.last_objective` is
the ILP objective = sum of `compute_cost + communication_cost` across all nodes
in the sub-graph, for the chosen strategy. Use this as the stage cost.

Alternatively, sum `strategy.compute_cost.total + strategy.communication_cost.total`
over all nodes manually after the solution is applied.

### C. Sub-graph merging for `initialize_model()`

`initialize_model()` expects an `nn.Module` it can trace. For a stage covering
`layers[k:i]`, create a thin wrapper:

```python
class StageSlice(nn.Module):
    def __init__(self, layers):
        super().__init__()
        for j, l in enumerate(layers):
            setattr(self, f"layer_{j}", l)
    def forward(self, x):
        for j in range(len(self._modules)):
            x = getattr(self, f"layer_{j}")(x)
        return x
```

### D. Sub-mesh carving

For submesh `(n_hosts, n_devs_per_host)`, create a sub-DeviceMesh by selecting
the appropriate ranks from the full physical mesh:

```python
sub_physical_ids = torch.tensor(plan.stage_device_groups[stage_id])
sub_mesh = DeviceMesh(
    physical_mesh_id=sub_physical_ids,
    mesh_shape=torch.Size(submesh_choices[mesh_id]),
    mesh_alpha=full_mesh.mesh_alpha,
    mesh_beta=full_mesh.mesh_beta,
)
```

### E. Cost table symmetry / triangle

Only fill the upper triangle `i > k`. Set the lower triangle and diagonal to
`np.inf`. This matches the semantics of `alpa_dp_impl` which iterates
`for i in range(num_layers, k, -1)`.

---

## Files to Create / Modify

| Action | File |
|---|---|
| **Create** | `colossalai/auto_parallel/pipeline_shard/layer_partition.py` |
| **Create** | `colossalai/auto_parallel/pipeline_shard/compute_cost.py` |
| **Create** | `colossalai/auto_parallel/pipeline_shard/orchestrator.py` |
| **Modify** | `colossalai/auto_parallel/pipeline_shard/__init__.py` |
| **Modify** | `colossalai/auto_parallel/tensor_shard/initialize.py` (add `autoparallelize_with_pp`) |
| **Create** | `examples/language/gpt/experiments/auto_parallel/test_auto_pipeline.py` |

---

## Running the Tests

```bash
# Test 1 — CPU only, no torch.distributed needed
python test_auto_pipeline.py --test dp_synthetic

# Test 2 — CPU only, needs transformers
python test_auto_pipeline.py --test layer_partition

# Test 3 — needs 2 GPUs
torchrun --nproc_per_node=2 test_auto_pipeline.py --test compute_cost

# Test 4 — needs 4 GPUs
torchrun --nproc_per_node=4 test_auto_pipeline.py --test pipeline_plan

# Test 5 — needs 4 GPUs, full training loop
torchrun --nproc_per_node=4 test_auto_pipeline.py --test training
```

---

## Open Questions / Risks

| Issue | Notes |
|---|---|
| `initialize_model()` may fail on very small sub-graphs (1-2 ops) | Add a try/except and fall back to meta-profiler sum |
| Sub-graph stitching may break FX tracing | Test each `StageSlice` independently with `ColoTracer` before solving |
| `alpa_dp` expects `num_layers == total layers`, but after splitting the count may differ from `num_pp_stages` | The DP auto-selects the number of stages — no need to fix `num_pp_stages` upfront |
| `DeviceMesh` sub-carving requires that process groups for the sub-ranks exist | Call `DeviceMesh(..., init_process_group=True)` after `dist.init_process_group()` |
| Cost table computation is `O(K²×M)` ILP solves — can be slow for large K | Cache aggressively; run in parallel on CPU cores |