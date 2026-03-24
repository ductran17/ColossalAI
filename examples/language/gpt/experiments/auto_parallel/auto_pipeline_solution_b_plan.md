# Solution B — Uniform TP Auto Pipeline: Implementation Plan

## Context

This plan implements **Phase 1 (heuristic)** 3D auto-parallelism for ColossalAI: automatic
pipeline stage assignment with a uniform TP degree constraint across all stages.

The constraint `TP_stage0 == TP_stage1 == ... == TP_stageN` eliminates the cross-stage
tensor resharding problem entirely, so no new collective communication primitives are needed.
The result is still strictly better than current ColossalAI (`autoparallelize()`) because:
- `alpa_dp` automatically finds the optimal number of pipeline stages (not manual)
- `alpa_dp` assigns layers to stages to minimize the slowest-stage bottleneck (not equal split)
- `initialize_model` independently optimizes TP/DP within each stage

This plan depends on the original `auto_pipeline_impl_plan.md` structure. All files from that
plan are still created; this plan documents the **changes and additions** on top of it.

---

## Prerequisites: Two Upstream Bug Fixes

These must land before any Solution B code can work.

### Fix 1 — Expose `last_objective` from `solve_solution()`

**File**: `colossalai/auto_parallel/tensor_shard/initialize.py`

```python
# BEFORE (line 129-132):
def solve_solution(gm, strategy_constructor, memory_budget=-1.0):
    cost_graph = CostGraph(strategy_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategy_constructor, cost_graph, memory_budget=memory_budget)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])
    return solution                          # ← objective lost here

# AFTER:
def solve_solution(gm, strategy_constructor, memory_budget=-1.0):
    cost_graph = CostGraph(strategy_constructor.leaf_strategies)
    cost_graph.simplify_graph()
    solver = Solver(gm.graph, strategy_constructor, cost_graph, memory_budget=memory_budget)
    ret = solver.call_solver_serialized_args()
    solution = list(ret[0])
    objective = solver.last_objective        # ret[2] also holds it
    return solution, objective
```

Update the one caller inside `initialize_model()`:
```python
# initialize.py ~line 280:
# BEFORE:
solution = solve_solution(gm, strategies_constructor, memory_budget)
# AFTER:
solution, _ = solve_solution(gm, strategies_constructor, memory_budget)
```

### Fix 2 — Correct `alpa_dp_impl` array indexing

**File**: `colossalai/device/calc_pipeline_strategy.py`

The inner loop accesses `compute_cost[k, i, m]` with `i` reaching `num_layers`, but the
assertion enforces shape `(num_layers, num_layers, ...)`. The second dimension must be
`num_layers + 1` to hold the index `num_layers`.

```python
# BEFORE (line 103-108):
assert np.shape(compute_cost) == (
    num_layers, num_layers, len(submesh_choices), num_autosharding_configs
), "Cost shape wrong."

# AFTER:
assert np.shape(compute_cost) == (
    num_layers, num_layers + 1, len(submesh_choices), num_autosharding_configs
), "Cost shape wrong. Expected (K, K+1, M, C)."
```

The `compute_cost` array filled by `get_compute_cost()` (Step 2 below) must use
shape `(K, K+1, M, 1)` accordingly. The diagonal and upper-right corner are `np.inf`.

---

## What Already Exists (do not re-implement)

Same as `auto_pipeline_impl_plan.md` — all entries still valid.

---

## What Is New / Changed

```
colossalai/auto_parallel/pipeline_shard/
├── __init__.py               ← update exports
├── compute_cost.py           ← NEW (replaces original): uniform-TP filtered
├── layer_partition.py        ← same as original plan
└── orchestrator.py           ← NEW: accepts uniform_tp_degree param

colossalai/auto_parallel/tensor_shard/initialize.py
    solve_solution()          ← fix: return objective (Fix 1 above)
    initialize_model()        ← fix: unpack (solution, _) from solve_solution
    autoparallelize_with_pp() ← NEW: add uniform_tp_degree param
```

---

## Step-by-Step Implementation

---

### Step 1 — `pipeline_shard/layer_partition.py`

**No change from original plan.** Implement exactly as described in
`auto_pipeline_impl_plan.md` Step 1. Both solutions share this module.

---

### Step 2 — `pipeline_shard/compute_cost.py` (Solution B version)

**Key difference from original plan**: submesh choices are filtered to only include
submeshes whose TP degree (second element) equals `uniform_tp_degree`.

```python
def get_compute_cost(
    layers: List[ColoGraphModule],
    input_shapes: List[torch.Size],
    submesh_choices: List[Tuple[int, int]],
    full_device_mesh: DeviceMesh,
    num_microbatches: int,
    uniform_tp_degree: int,              # ← NEW: Solution B constraint
    memory_budget: float = -1.0,
    cache_path: str = None,
) -> np.ndarray:
    """
    Returns:
        compute_cost: np.ndarray shape (K, K+1, M, 1)   ← note K+1
            K = len(layers), M = len(filtered_submesh_choices)
        compute_cost[k, i, m, 0] = estimated time to run layers[k:i]
                                    on submesh m (all submeshes have same TP degree)
    """
```

**Implementation**:

```python
def get_compute_cost(..., uniform_tp_degree: int, ...):
    # 1. Cache check
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    # 2. Filter submesh choices to uniform TP — Solution B key constraint
    #    submesh_choice = (n_hosts, n_devs_per_host)
    #    TP degree = n_devs_per_host (the inner-node dimension)
    filtered = [
        (i, s) for i, s in enumerate(submesh_choices)
        if s[1] == uniform_tp_degree
    ]
    assert len(filtered) > 0, (
        f"No submesh with TP={uniform_tp_degree} found in {submesh_choices}. "
        f"Available TP degrees: {set(s[1] for s in submesh_choices)}"
    )
    filtered_indices, filtered_submeshes = zip(*filtered)
    M = len(filtered_submeshes)
    K = len(layers)

    # 3. Build cost table — shape (K, K+1, M, 1) to match alpa_dp_impl indexing
    compute_cost = np.full((K, K + 1, M, 1), np.inf, dtype=np.float32)

    for m, (orig_idx, submesh) in enumerate(zip(filtered_indices, filtered_submeshes)):
        n_submesh_devices = int(np.prod(submesh))
        if n_submesh_devices > dist.get_world_size():
            continue  # skip submeshes larger than available devices

        for k in range(K):
            for i in range(k + 1, K + 1):   # i in [k+1, K] inclusive
                try:
                    cost = _estimate_stage_cost(
                        layers, k, i, submesh, full_device_mesh,
                        input_shapes, memory_budget,
                    )
                except Exception:
                    cost = _fallback_meta_profiler_cost(layers, k, i)
                compute_cost[k, i, m, 0] = cost

    # 4. Save cache
    if cache_path:
        np.save(cache_path, compute_cost)

    return compute_cost, list(filtered_submeshes)   # return filtered list too


def _estimate_stage_cost(layers, k, i, submesh, full_mesh, input_shapes, memory_budget):
    """Run initialize_model on layers[k:i] and return the ILP objective."""
    from colossalai.auto_parallel.tensor_shard.initialize import (
        initialize_model, initialize_device_mesh,
    )
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer

    # Build a thin sequential wrapper for layers[k:i] that ColoTracer can trace
    stage_model = _build_stage_module(layers[k:i])

    # Build sub-mesh using the first n_submesh_devices ranks
    n_devs = int(np.prod(submesh))
    sub_physical_ids = torch.arange(n_devs)
    sub_mesh = DeviceMesh(
        physical_mesh_id=sub_physical_ids,
        mesh_shape=torch.Size(submesh),
        mesh_alpha=full_mesh.mesh_alpha[:len(submesh)],
        mesh_beta=full_mesh.mesh_beta[:len(submesh)],
    )

    # Build meta_args for this stage from its input shape
    stage_input = torch.zeros(input_shapes[k])
    stage_meta_args = {"x": stage_input}

    # Run the ILP solver — Solution B: this is shared between all stage configs
    _, objective = solve_solution_with_objective(
        stage_model, stage_meta_args, sub_mesh, memory_budget
    )
    return float(objective)


def _build_stage_module(layer_list):
    """
    Wrap layers[k:i] into a simple nn.Sequential-compatible module.
    Uses torch.fx.passes.split_module output directly where possible.
    Falls back to nn.Sequential for purely sequential layers.
    """
    class StageModule(nn.Module):
        def __init__(self, layers):
            super().__init__()
            for j, layer in enumerate(layers):
                self.add_module(f"layer_{j}", layer)
            self._n = len(layers)
        def forward(self, x):
            for j in range(self._n):
                x = getattr(self, f"layer_{j}")(x)
            return x
    return StageModule(layer_list)


def solve_solution_with_objective(model, meta_args, device_mesh, memory_budget):
    """Helper: run build_strategy_constructor + solve_solution, return (solution, objective)."""
    from colossalai.auto_parallel.tensor_shard.initialize import (
        build_strategy_constructor, solve_solution,
    )
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer
    from colossalai._analyzer.fx.graph_module import ColoGraphModule
    from colossalai._analyzer.fx.passes import shape_prop_pass
    from colossalai._analyzer.fx.codegen import ActivationCheckpointCodeGen

    tracer = ColoTracer(trace_act_ckpt=True, bias_addition_split=True)
    graph = tracer.trace(root=model, meta_args=meta_args)
    graph.set_codegen(ActivationCheckpointCodeGen())
    gm = ColoGraphModule(model, graph, model.__class__.__name__)
    shape_prop_pass(gm, *meta_args.values())
    gm.recompile()

    strategies_constructor = build_strategy_constructor(
        graph, device_mesh,
        solver_preference="standard",
        dataloader_option="replicated",
        shard_option="standard",
    )
    solution, objective = solve_solution(gm, strategies_constructor, memory_budget)
    return solution, objective
```

---

### Step 3 — `pipeline_shard/orchestrator.py` (Solution B version)

```python
@dataclass
class PipelinePlan:
    stage_layer_ranges: List[Tuple[int, int]]
    stage_submesh_ids: List[int]
    stage_device_groups: List[List[int]]
    submesh_choices: List[Tuple[int, int]]   # filtered (uniform TP)
    uniform_tp_degree: int                   # ← NEW
    best_cost: float


def build_pipeline_plan(
    model: nn.Module,
    meta_args: Dict[str, Any],
    num_devices: int,
    num_microbatches: int,
    uniform_tp_degree: int = None,          # ← NEW: None = auto-select best
    num_hosts: int = 1,
    num_devices_per_host: int = None,
    memory_budget: float = -1.0,
    cache_path: str = None,
    mode: str = "new",
) -> PipelinePlan:
    from colossalai.auto_parallel.pipeline_shard.layer_partition import get_pipeline_layers
    from colossalai.auto_parallel.pipeline_shard.compute_cost import get_compute_cost
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_device_mesh
    from colossalai.device.calc_pipeline_strategy import alpa_dp, get_submesh_choices
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer
    from colossalai._analyzer.fx.graph_module import ColoGraphModule

    ndph = num_devices_per_host or num_devices
    all_submesh_choices = get_submesh_choices(num_hosts, ndph, mode=mode)

    # 1. If uniform_tp_degree not given, try all available TP degrees and pick best
    if uniform_tp_degree is None:
        available_tp = sorted(set(s[1] for s in all_submesh_choices))
    else:
        available_tp = [uniform_tp_degree]

    # 2. Trace model and get layer list
    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)
    K = len(layers)

    full_mesh = initialize_device_mesh(world_size=num_devices)

    best_overall_cost = np.inf
    best_plan = None

    for tp in available_tp:
        tp_cache = f"{cache_path}_tp{tp}.npy" if cache_path else None

        # 3. Get cost table filtered to this TP degree
        cost_table, filtered_submeshes = get_compute_cost(
            layers, input_shapes, all_submesh_choices, full_mesh,
            num_microbatches, uniform_tp_degree=tp,
            memory_budget=memory_budget, cache_path=tp_cache,
        )
        M = len(filtered_submeshes)

        # 4. Run DP solver
        best_cost, solution = alpa_dp(
            num_layers=K,
            num_devices=num_devices,
            num_microbatches=num_microbatches,
            submesh_choices=filtered_submeshes,
            num_autosharding_configs=1,
            compute_cost=cost_table,
        )

        if solution is not None and best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_plan = (solution, filtered_submeshes, tp)

    assert best_plan is not None, "alpa_dp found no valid solution for any TP degree."
    solution, filtered_submeshes, chosen_tp = best_plan

    # 5. Parse solution into PipelinePlan
    stage_layer_ranges = []
    stage_submesh_ids = []
    stage_device_groups = []
    device_cursor = 0
    for (start, end), mesh_id, _ in solution:
        n_devs = int(np.prod(filtered_submeshes[mesh_id]))
        ranks = list(range(device_cursor, device_cursor + n_devs))
        stage_layer_ranges.append((start, end))
        stage_submesh_ids.append(mesh_id)
        stage_device_groups.append(ranks)
        device_cursor += n_devs

    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        stage_submesh_ids=stage_submesh_ids,
        stage_device_groups=stage_device_groups,
        submesh_choices=filtered_submeshes,
        uniform_tp_degree=chosen_tp,
        best_cost=best_overall_cost,
    )
```

---

### Step 4 — `autoparallelize_with_pp()` (Solution B version)

**File**: `colossalai/auto_parallel/tensor_shard/initialize.py`

```python
def autoparallelize_with_pp(
    model: nn.Module,
    meta_args: Dict[str, Any],
    num_devices: int,
    num_microbatches: int,
    uniform_tp_degree: int = None,
    schedule_type: str = "1f1b",
    memory_budget: float = -1.0,
    cache_path: str = None,
) -> Tuple[ModuleWrapper, "PipelineStageManager"]:
    """
    Auto 3D parallelism with uniform TP constraint (Solution B).
    All pipeline stages use the same TP degree.
    No cross-stage resharding is needed.

    Returns:
        stage_module: ModuleWrapper for this rank's pipeline stage
        stage_manager: configured PipelineStageManager
    """
    from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
    from colossalai.auto_parallel.pipeline_shard.layer_partition import get_pipeline_layers
    from colossalai.cluster import ProcessGroupMesh
    from colossalai.pipeline.stage_manager import PipelineStageManager
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer

    rank = dist.get_rank()

    # 1. Find optimal pipeline plan (runs on all ranks identically — deterministic)
    plan = build_pipeline_plan(
        model, meta_args, num_devices, num_microbatches,
        uniform_tp_degree=uniform_tp_degree,
        memory_budget=memory_budget, cache_path=cache_path,
    )

    # 2. Determine this rank's pipeline stage
    my_stage = _rank_to_stage(rank, plan)
    num_stages = len(plan.stage_layer_ranges)

    # 3. Get this stage's layer sub-graph
    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)
    start, end = plan.stage_layer_ranges[my_stage]
    stage_meta_args = {"x": torch.zeros(input_shapes[start])}

    # 4. Build sub-mesh for this stage
    sub_mesh = _build_submesh(plan, my_stage)

    # 5. Apply intra-op (TP/DP) sharding to this stage only
    from colossalai.auto_parallel.pipeline_shard.compute_cost import _build_stage_module
    stage_model = _build_stage_module(layers[start:end])
    stage_module = initialize_model(
        stage_model, stage_meta_args, sub_mesh,
        memory_budget=memory_budget,
    )

    # 6. Build PipelineStageManager
    #    Layout: PP_AXIS=0, TP_AXIS=1
    tp = plan.uniform_tp_degree
    pg_mesh = ProcessGroupMesh(num_stages, tp)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)

    return stage_module, stage_manager


def _rank_to_stage(rank: int, plan: "PipelinePlan") -> int:
    for stage_idx, ranks in enumerate(plan.stage_device_groups):
        if rank in ranks:
            return stage_idx
    raise ValueError(f"Rank {rank} not found in any stage: {plan.stage_device_groups}")


def _build_submesh(plan: "PipelinePlan", stage_id: int) -> DeviceMesh:
    from colossalai.device.alpha_beta_profiler import AlphaBetaProfiler
    ranks = plan.stage_device_groups[stage_id]
    submesh_shape = plan.submesh_choices[plan.stage_submesh_ids[stage_id]]
    physical_ids = torch.tensor(ranks)
    ab_profiler = AlphaBetaProfiler(ranks)
    mesh_alpha, mesh_beta = ab_profiler.extract_alpha_beta_for_device_mesh()
    return DeviceMesh(
        physical_mesh_id=physical_ids,
        mesh_shape=torch.Size(submesh_shape),
        mesh_alpha=mesh_alpha,
        mesh_beta=mesh_beta,
        init_process_group=True,
    )
```

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

## Files to Create / Modify

| Action | File | Notes |
|---|---|---|
| **Modify** | `colossalai/auto_parallel/tensor_shard/initialize.py` | Fix `solve_solution()`, add `autoparallelize_with_pp()`, `_rank_to_stage()`, `_build_submesh()` |
| **Modify** | `colossalai/device/calc_pipeline_strategy.py` | Fix `alpa_dp` shape assertion to `(K, K+1, M, C)` |
| **Create** | `colossalai/auto_parallel/pipeline_shard/layer_partition.py` | Same as original plan |
| **Create** | `colossalai/auto_parallel/pipeline_shard/compute_cost.py` | Uniform-TP filtered version |
| **Create** | `colossalai/auto_parallel/pipeline_shard/orchestrator.py` | With `uniform_tp_degree` param |
| **Modify** | `colossalai/auto_parallel/pipeline_shard/__init__.py` | Add exports |
| **Create** | `examples/language/gpt/experiments/auto_parallel/test_auto_pipeline_b.py` | Tests below |

---

## Test Plan

### Test 1 — Verify alpa_dp shape fix (CPU, no GPU)

```python
def test_alpa_dp_shape_fix():
    """Verify that shape (K, K+1, M, C) is accepted and produces valid solution."""
    import numpy as np
    from colossalai.device.calc_pipeline_strategy import alpa_dp, get_submesh_choices

    K, N, B = 8, 4, 4
    submesh_choices = [(1, 1), (1, 2), (1, 4)]
    M = len(submesh_choices)

    # Correct shape: (K, K+1, M, 1)
    cost = np.full((K, K + 1, M, 1), np.inf, dtype=np.float32)
    for k in range(K):
        for i in range(k + 1, K + 1):  # i from k+1 to K inclusive
            for m in range(M):
                n_devs = int(np.prod(submesh_choices[m]))
                cost[k, i, m, 0] = (i - k) * float(n_devs)

    best_cost, solution = alpa_dp(K, N, B, submesh_choices, 1, cost)
    assert solution is not None, "DP returned no solution"
    covered = set()
    for (start, end), _, _ in solution:
        for l in range(start, end):
            covered.add(l)
    assert covered == set(range(K)), f"Not all layers covered: {covered}"
    print(f"[PASS] test_alpa_dp_shape_fix: cost={best_cost:.2f}, stages={len(solution)}")
```

### Test 2 — Verify solve_solution returns objective (CPU)

```python
def test_solve_solution_objective():
    """Verify solve_solution() returns (solution, objective) after the fix."""
    # (Requires a small ColossalAI-traceable model and a DeviceMesh)
    from colossalai.auto_parallel.tensor_shard.initialize import (
        build_strategy_constructor, solve_solution,
    )
    # ... setup gm and strategies_constructor ...
    result = solve_solution(gm, strategies_constructor)
    assert isinstance(result, tuple) and len(result) == 2
    solution, objective = result
    assert isinstance(solution, list)
    assert isinstance(objective, float) and objective >= 0
    print(f"[PASS] test_solve_solution_objective: objective={objective:.4f}")
```

### Test 3 — Layer partition (CPU, needs transformers)

Same as original plan Test 2.

### Test 4 — get_compute_cost with uniform TP (needs 1+ GPU)

```python
def test_compute_cost_uniform_tp():
    """Verify that compute_cost only fills entries for the specified TP degree."""
    # ... setup same as original plan Test 3 ...
    cost_table, filtered_submeshes = get_compute_cost(
        layers, input_shapes, submesh_choices, full_mesh,
        num_microbatches=4, uniform_tp_degree=2,
        cache_path="/tmp/gpt2_cost_b.npy",
    )
    K = len(layers)
    M = len(filtered_submeshes)
    assert cost_table.shape == (K, K + 1, M, 1)
    # All filtered submeshes must have TP degree == 2
    assert all(s[1] == 2 for s in filtered_submeshes), \
        f"Non-uniform TP in filtered submeshes: {filtered_submeshes}"
    print(f"[PASS] test_compute_cost_uniform_tp: shape={cost_table.shape}, "
          f"submeshes={filtered_submeshes}")
```

### Test 5 — End-to-end pipeline plan (needs 4 GPUs)

```python
# torchrun --nproc_per_node=4 test_auto_pipeline_b.py --test e2e
def test_e2e_uniform_tp():
    from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
    # ...
    plan = build_pipeline_plan(
        model=model, meta_args=meta_args,
        num_devices=4, num_microbatches=4,
        uniform_tp_degree=2,               # force TP=2 for all stages
    )
    assert plan.uniform_tp_degree == 2
    assert all(
        plan.submesh_choices[mid][1] == 2
        for mid in plan.stage_submesh_ids
    ), "Not all stages have TP=2"
    print(f"[PASS] test_e2e_uniform_tp: cost={plan.best_cost:.4f}, "
          f"stages={len(plan.stage_layer_ranges)}")
```

### Test 6 — Full training loop (needs 4 GPUs)

Same structure as original plan Test 5, calling `autoparallelize_with_pp()` with
`uniform_tp_degree=2`.

---

## Implementation Order

```
1. Fix solve_solution() → test_solve_solution_objective
2. Fix alpa_dp shape  → test_alpa_dp_shape_fix
3. layer_partition.py → test_layer_partition_gpt2 (from original plan)
4. compute_cost.py    → test_compute_cost_uniform_tp
5. orchestrator.py    → test_e2e_uniform_tp
6. autoparallelize_with_pp() → test_training_loop
```

---

## Key Design Decisions

### A. Why filter submesh_choices rather than constraining the DP?

Filtering before calling `alpa_dp` is cleaner: the DP solver never sees incompatible
submeshes, so its solution is always valid by construction. The alternative (adding a
post-hoc constraint check in `alpa_dp_impl`) would require modifying the ported solver code.

### B. Choosing `uniform_tp_degree` automatically

When `uniform_tp_degree=None`, `build_pipeline_plan` tries every available TP degree and
returns the globally cheapest plan. For large models this can be slow; cache aggressively.

### C. Cost table shape (K, K+1, M, 1) vs (K, K, M, 1)

`alpa_dp_impl` loops `for i in range(num_layers, k, -1)` and accesses `cost[k, i, m]`.
With `k=0` and `num_layers=K`, `i` takes value `K` as the first element. The table must
have second dimension `K+1` to hold index `K`. This is a bug in the original plan; Solution
B fixes it explicitly.

---

## Open Questions

| Issue | Notes |
|---|---|
| `_build_stage_module` trace failures for non-sequential layers | Use `split_module` output directly as the stage module (same fix as original plan's gap #4) |
| `AlphaBetaProfiler` re-runs NCCL profiling per sub-mesh | Cache `alpha_beta_dict` once and reuse across all stage cost calls |
| `uniform_tp_degree` search may be slow for large K | Set `cache_path` and run cost table computation once; subsequent runs load from cache |
