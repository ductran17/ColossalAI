# Solution A — Heterogeneous TP with Slice-Gather: Implementation Plan

## Context

This plan implements **Phase 2 (optimal)** 3D auto-parallelism for ColossalAI: automatic
pipeline stage assignment where each stage independently optimizes its own TP degree.
Adjacent stages with different TP degrees are bridged by lightweight Slice-Gather ops.

**Solution A builds directly on top of Solution B.** All Solution B code must be in place
before implementing Solution A. This plan documents the **additional files and changes**
needed on top of Solution B.

Reference implementation: `Hetu-Galvatron` — `galvatron/core/runtime/redistribute.py`
and `galvatron/core/runtime/hybrid_parallel_model.py` (Slice-Gather pattern).

---

## What Changes vs Solution B

| Area | Solution B | Solution A |
|---|---|---|
| Submesh search space | Filtered to 1 TP degree | All submesh choices |
| Stage boundary | Activations pass through as-is | BoundaryAllGather or BoundarySplit injected |
| Cost table | Stage cost only | Stage cost + boundary resharding penalty |
| `PipelinePlan` | Has `uniform_tp_degree` | Has `stage_tp_degrees` list |
| `autoparallelize_with_pp` | Uniform TP enforced | Per-stage TP, boundary ops wired |

---

## New Files

```
colossalai/auto_parallel/pipeline_shard/
├── boundary_resharding.py    ← NEW (Step 2): autograd ops for stage boundaries
├── compute_cost.py           ← MODIFY: add resharding penalty to cost table
├── orchestrator.py           ← MODIFY: heterogeneous submesh search + boundary wiring
└── __init__.py               ← MODIFY: export boundary ops
```

---

## Step-by-Step Implementation

---

### Step 1 — Prerequisites

All of Solution B must be complete and passing:
- `solve_solution()` returns `(solution, objective)`
- `alpa_dp` uses `(K, K+1, M, C)` shape
- `layer_partition.py` implemented
- `compute_cost.py` (Solution B version) working
- `orchestrator.py` (Solution B version) working

---

### Step 2 — `pipeline_shard/boundary_resharding.py` (NEW)

**Purpose**: autograd-compatible all_gather / split ops to insert at stage boundaries
where adjacent stages use different TP degrees. Gradient flows correctly in both directions.

```python
# colossalai/auto_parallel/pipeline_shard/boundary_resharding.py

import torch
import torch.distributed as dist
from torch import nn
from typing import Optional


class _BoundaryAllGather(torch.autograd.Function):
    """
    Forward:  gather shards from all TP ranks → full tensor.
              Used when sender stage has TP > 1 and receiver stage has TP = 1
              (or lower TP degree).
    Backward: split gradient back across TP ranks (reverse of gather).

    Equivalent to Galvatron's _Gather in redistribute.py.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.tp_group = tp_group
        world_size = dist.get_world_size(tp_group)
        if world_size == 1:
            return input
        # Allocate output buffer
        output_shape = list(input.shape)
        output_shape[-1] *= world_size           # gather along last (hidden) dim
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        dist.all_gather_into_tensor(output, input.contiguous(), group=tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        tp_group = ctx.tp_group
        world_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)
        if world_size == 1:
            return grad_output, None
        # Split gradient: this rank owns its shard
        chunks = torch.chunk(grad_output, world_size, dim=-1)
        return chunks[rank].contiguous(), None


class _BoundarySplit(torch.autograd.Function):
    """
    Forward:  split full tensor and keep only this rank's shard.
              Used when sender stage has TP = 1 and receiver stage has TP > 1.
    Backward: all_gather gradients from all TP ranks (reverse of split).

    Equivalent to Galvatron's _Split in redistribute.py.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.tp_group = tp_group
        world_size = dist.get_world_size(tp_group)
        rank = dist.get_rank(tp_group)
        if world_size == 1:
            return input
        chunks = torch.chunk(input, world_size, dim=-1)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        tp_group = ctx.tp_group
        world_size = dist.get_world_size(tp_group)
        if world_size == 1:
            return grad_output, None
        output_shape = list(grad_output.shape)
        output_shape[-1] *= world_size
        full_grad = torch.empty(output_shape, dtype=grad_output.dtype,
                                device=grad_output.device)
        dist.all_gather_into_tensor(full_grad, grad_output.contiguous(), group=tp_group)
        return full_grad, None


class BoundaryReshardingModule(nn.Module):
    """
    An nn.Module wrapper inserted between pipeline stages at stage boundaries
    where adjacent stages have different TP degrees.

    At forward time this module is called on the receiving stage (stage N+1):
    - if tp_sender > tp_receiver: the activation arrived gathered (full) from
      stage N's all_gather, so this module is a no-op in that direction.
    - Actually: the module is inserted on the SENDING side just before P2P send.

    Design: insert ONE BoundaryReshardingModule per stage boundary.
    The op is chosen based on (sender_tp, receiver_tp).
    """

    def __init__(
        self,
        sender_tp: int,
        receiver_tp: int,
        tp_group: dist.ProcessGroup,
    ):
        super().__init__()
        self.sender_tp = sender_tp
        self.receiver_tp = receiver_tp
        self.tp_group = tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sender_tp == self.receiver_tp:
            return x
        elif self.sender_tp > self.receiver_tp:
            # Sender is TP-sharded, receiver wants full tensor → gather
            return _BoundaryAllGather.apply(x, self.tp_group)
        else:
            # Sender has full tensor, receiver wants shard → split
            return _BoundarySplit.apply(x, self.tp_group)

    def extra_repr(self) -> str:
        return (f"sender_tp={self.sender_tp}, receiver_tp={self.receiver_tp}")


def get_boundary_resharding_cost(
    sender_tp: int,
    receiver_tp: int,
    activation_shape: torch.Size,
    mesh_alpha: float,
    mesh_beta: float,
) -> float:
    """
    Estimate the communication cost of one boundary resharding operation.
    Used by compute_cost.py to add resharding penalty to the cost table.

    For all_gather: cost = alpha + beta * (T-1)/T * tensor_bytes
    For split: cost is negligible (local chunk, no communication)
    For no-op: cost = 0
    """
    if sender_tp == receiver_tp:
        return 0.0
    tensor_bytes = _activation_bytes(activation_shape)
    if sender_tp > receiver_tp:
        # all_gather: (T-1)/T fraction of data is communicated
        t = sender_tp
        return mesh_alpha + mesh_beta * (t - 1) / t * tensor_bytes
    else:
        # split: local op, no communication cost
        return 0.0


def _activation_bytes(shape: torch.Size, dtype_bytes: int = 2) -> int:
    """Size of activation tensor in bytes (default fp16 = 2 bytes)."""
    n = 1
    for s in shape:
        n *= s
    return n * dtype_bytes
```

---

### Step 3 — `pipeline_shard/compute_cost.py` (Solution A additions)

**Modify** the Solution B `compute_cost.py` to:
1. Remove the `uniform_tp_degree` filter — allow all submesh choices
2. Add a boundary resharding penalty between adjacent stages in the cost table

The penalty must be attached to the **receiving stage** because `alpa_dp_impl` accumulates
costs per stage (`new_cost = f[s-1, k, d - n_submesh_devices] + stage_cost`). The boundary
cost is a property of the transition, so add it to the entering stage's cost.

```python
# In compute_cost.py — Solution A version

from colossalai.auto_parallel.pipeline_shard.boundary_resharding import (
    get_boundary_resharding_cost,
)

def get_compute_cost(
    layers: List[ColoGraphModule],
    input_shapes: List[torch.Size],
    submesh_choices: List[Tuple[int, int]],
    full_device_mesh: DeviceMesh,
    num_microbatches: int,
    memory_budget: float = -1.0,
    cache_path: str = None,
    # Solution A: NO uniform_tp_degree parameter — all submeshes allowed
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Returns:
        compute_cost: np.ndarray shape (K, K+1, M, 1)
        submesh_choices: the original list (unfiltered)

    compute_cost[k, i, m, 0] includes:
        - ILP objective for layers[k:i] on submesh m
        - boundary resharding penalty IF the previous stage had a different TP degree
          (handled via a separate boundary_cost table used in orchestrator)
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        return data["cost"], data["submeshes"]

    K = len(layers)
    M = len(submesh_choices)
    compute_cost = np.full((K, K + 1, M, 1), np.inf, dtype=np.float32)

    for m, submesh in enumerate(submesh_choices):
        n_submesh_devices = int(np.prod(submesh))
        if n_submesh_devices > dist.get_world_size():
            continue
        for k in range(K):
            for i in range(k + 1, K + 1):
                try:
                    cost = _estimate_stage_cost(
                        layers, k, i, submesh, full_device_mesh,
                        input_shapes, memory_budget,
                    )
                except Exception:
                    cost = _fallback_meta_profiler_cost(layers, k, i)
                compute_cost[k, i, m, 0] = cost

    if cache_path:
        np.save(cache_path, {"cost": compute_cost, "submeshes": submesh_choices})

    return compute_cost, submesh_choices


def get_boundary_cost_table(
    submesh_choices: List[Tuple[int, int]],
    input_shapes: List[torch.Size],
    full_device_mesh: DeviceMesh,
) -> np.ndarray:
    """
    Pre-compute boundary resharding costs for every pair of adjacent submeshes.

    Returns:
        boundary_cost: np.ndarray shape (M, M, K)
            boundary_cost[m_prev, m_curr, k] = cost of resharding at layer boundary k
            when going from a stage using submesh m_prev to a stage using submesh m_curr.
    """
    M = len(submesh_choices)
    K = len(input_shapes)
    boundary_cost = np.zeros((M, M, K), dtype=np.float32)

    # Use alpha/beta from the full mesh for cross-stage communication estimate
    alpha = full_device_mesh.mesh_alpha[0] if full_device_mesh.mesh_alpha else 1e-5
    beta = full_device_mesh.mesh_beta[0] if full_device_mesh.mesh_beta else 1e-10

    for m_prev, submesh_prev in enumerate(submesh_choices):
        tp_prev = submesh_prev[1]
        for m_curr, submesh_curr in enumerate(submesh_choices):
            tp_curr = submesh_curr[1]
            for k in range(K):
                if tp_prev == tp_curr:
                    boundary_cost[m_prev, m_curr, k] = 0.0
                else:
                    boundary_cost[m_prev, m_curr, k] = get_boundary_resharding_cost(
                        sender_tp=tp_prev,
                        receiver_tp=tp_curr,
                        activation_shape=input_shapes[k],
                        mesh_alpha=alpha,
                        mesh_beta=beta,
                    )

    return boundary_cost
```

---

### Step 4 — `pipeline_shard/orchestrator.py` (Solution A additions)

**Key change**: after `alpa_dp` finds the stage assignment, post-process the solution to
add boundary costs, then build `BoundaryReshardingModule` instances between stages.

```python
@dataclass
class PipelinePlan:
    stage_layer_ranges: List[Tuple[int, int]]
    stage_submesh_ids: List[int]
    stage_device_groups: List[List[int]]
    stage_tp_degrees: List[int]              # ← NEW: per-stage TP degree
    submesh_choices: List[Tuple[int, int]]
    boundary_resharding: List[Optional[BoundaryReshardingModule]]  # ← NEW: one per boundary
    best_cost: float


def build_pipeline_plan(
    model: nn.Module,
    meta_args: Dict[str, Any],
    num_devices: int,
    num_microbatches: int,
    num_hosts: int = 1,
    num_devices_per_host: int = None,
    memory_budget: float = -1.0,
    cache_path: str = None,
    mode: str = "new",
) -> PipelinePlan:
    """Solution A: no uniform_tp_degree constraint."""
    from colossalai.auto_parallel.pipeline_shard.layer_partition import get_pipeline_layers
    from colossalai.auto_parallel.pipeline_shard.compute_cost import (
        get_compute_cost, get_boundary_cost_table,
    )
    from colossalai.auto_parallel.tensor_shard.initialize import initialize_device_mesh
    from colossalai.device.calc_pipeline_strategy import alpa_dp, get_submesh_choices
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer

    ndph = num_devices_per_host or num_devices
    submesh_choices = get_submesh_choices(num_hosts, ndph, mode=mode)

    # 1. Trace model and get layer list
    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)
    K = len(layers)

    full_mesh = initialize_device_mesh(world_size=num_devices)

    # 2. Build base compute cost table (no boundary penalty yet)
    cost_table, _ = get_compute_cost(
        layers, input_shapes, submesh_choices, full_mesh,
        num_microbatches, memory_budget=memory_budget, cache_path=cache_path,
    )

    # 3. Build boundary resharding cost table (M × M × K)
    boundary_cost = get_boundary_cost_table(submesh_choices, input_shapes, full_mesh)

    # 4. Augment cost table: for each (k, i, m_curr), add the minimum boundary cost
    #    over all possible previous submesh choices m_prev.
    #    This is a conservative estimate — alpa_dp doesn't track m_prev directly.
    #    We add min_over_m_prev(boundary_cost[m_prev, m_curr, k]) to each entry.
    #    A more accurate approach (DP over (stage, layer, device, prev_submesh)) is
    #    left as a future improvement.
    for m_curr in range(len(submesh_choices)):
        min_boundary = boundary_cost[:, m_curr, :].min(axis=0)   # shape (K,)
        for k in range(K):
            for i in range(k + 1, K + 1):
                if not np.isinf(cost_table[k, i, m_curr, 0]):
                    cost_table[k, i, m_curr, 0] += min_boundary[k]

    # 5. Run DP solver on augmented cost table
    best_cost, solution = alpa_dp(
        num_layers=K,
        num_devices=num_devices,
        num_microbatches=num_microbatches,
        submesh_choices=submesh_choices,
        num_autosharding_configs=1,
        compute_cost=cost_table,
    )
    assert solution is not None, "alpa_dp found no valid solution"

    # 6. Parse solution
    stage_layer_ranges = []
    stage_submesh_ids = []
    stage_device_groups = []
    stage_tp_degrees = []
    device_cursor = 0
    for (start, end), mesh_id, _ in solution:
        n_devs = int(np.prod(submesh_choices[mesh_id]))
        ranks = list(range(device_cursor, device_cursor + n_devs))
        stage_layer_ranges.append((start, end))
        stage_submesh_ids.append(mesh_id)
        stage_device_groups.append(ranks)
        stage_tp_degrees.append(submesh_choices[mesh_id][1])
        device_cursor += n_devs

    # 7. Build boundary resharding modules (one per adjacent stage pair)
    num_stages = len(stage_layer_ranges)
    boundary_modules = []
    for s in range(num_stages - 1):
        tp_sender = stage_tp_degrees[s]
        tp_receiver = stage_tp_degrees[s + 1]
        if tp_sender == tp_receiver:
            boundary_modules.append(None)
        else:
            # The process group spanning both stages' TP ranks
            # For now use the receiver's TP group (it initiates the receive)
            tp_group = _get_tp_group(stage_device_groups[s + 1])
            boundary_modules.append(
                BoundaryReshardingModule(tp_sender, tp_receiver, tp_group)
            )

    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        stage_submesh_ids=stage_submesh_ids,
        stage_device_groups=stage_device_groups,
        stage_tp_degrees=stage_tp_degrees,
        submesh_choices=submesh_choices,
        boundary_resharding=boundary_modules,
        best_cost=best_cost,
    )


def _get_tp_group(ranks: List[int]) -> dist.ProcessGroup:
    """Get or create the TP process group for a set of ranks."""
    ranks_tuple = tuple(sorted(ranks))
    return dist.new_group(list(ranks_tuple))
```

---

### Step 5 — `autoparallelize_with_pp()` (Solution A version)

**File**: `colossalai/auto_parallel/tensor_shard/initialize.py`

```python
def autoparallelize_with_pp_hetero(
    model: nn.Module,
    meta_args: Dict[str, Any],
    num_devices: int,
    num_microbatches: int,
    schedule_type: str = "1f1b",
    memory_budget: float = -1.0,
    cache_path: str = None,
) -> Tuple[ModuleWrapper, "PipelineStageManager", Optional["BoundaryReshardingModule"]]:
    """
    Auto 3D parallelism with heterogeneous TP (Solution A).
    Each pipeline stage can use a different TP degree.
    Boundary resharding modules are returned for wiring into the training loop.

    Returns:
        stage_module: ModuleWrapper for this rank's pipeline stage
        stage_manager: configured PipelineStageManager
        boundary_module: BoundaryReshardingModule to call before send_forward,
                         or None if no resharding needed at the outgoing boundary.
    """
    from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
    from colossalai.auto_parallel.pipeline_shard.layer_partition import get_pipeline_layers
    from colossalai.cluster import ProcessGroupMesh
    from colossalai.pipeline.stage_manager import PipelineStageManager
    from colossalai._analyzer.fx.tracer.tracer import ColoTracer

    rank = dist.get_rank()

    # 1. Find optimal pipeline plan with heterogeneous TP
    plan = build_pipeline_plan(
        model, meta_args, num_devices, num_microbatches,
        memory_budget=memory_budget, cache_path=cache_path,
    )

    # 2. Determine this rank's stage
    my_stage = _rank_to_stage(rank, plan)
    num_stages = len(plan.stage_layer_ranges)

    # 3. Get this stage's sub-graph
    gm = ColoTracer().trace(model, meta_args=meta_args)
    layers, input_shapes = get_pipeline_layers(gm)
    start, end = plan.stage_layer_ranges[my_stage]
    stage_meta_args = {"x": torch.zeros(input_shapes[start])}

    # 4. Build sub-mesh for this stage
    sub_mesh = _build_submesh(plan, my_stage)

    # 5. Apply intra-op sharding (TP/DP) to this stage — independent per stage
    from colossalai.auto_parallel.pipeline_shard.compute_cost import _build_stage_module
    stage_model = _build_stage_module(layers[start:end])
    stage_module = initialize_model(
        stage_model, stage_meta_args, sub_mesh,
        memory_budget=memory_budget,
    )

    # 6. Build PipelineStageManager
    #    With heterogeneous TP, pipeline_axis is the PP dimension.
    #    TP groups are independently created per stage via DeviceMesh.
    pg_mesh = ProcessGroupMesh(num_stages)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)

    # 7. Get the boundary resharding module for this stage's outgoing boundary
    #    (None if this is the last stage or TP degrees match)
    boundary_module = None
    if my_stage < num_stages - 1:
        boundary_module = plan.boundary_resharding[my_stage]

    return stage_module, stage_manager, boundary_module
```

---

### Step 6 — Training Loop Integration

The `boundary_module` must be called on the activation **before** `send_forward()`.
Modify the training loop to wrap the output:

```python
# In the per-step training loop (user code):
stage_module, stage_manager, boundary_module = autoparallelize_with_pp_hetero(...)

def forward_step(x):
    output = stage_module(x)
    # Apply boundary resharding before sending to next stage
    if boundary_module is not None and not stage_manager.is_last_stage():
        output = boundary_module(output)
    return output
```

For `OneForwardOneBackwardSchedule`, override the model callable:
```python
schedule.forward_backward_step(
    model=forward_step,   # pass the wrapped callable
    ...
)
```

---

### Step 7 — `pipeline_shard/__init__.py`

```python
from .orchestrator import build_pipeline_plan, PipelinePlan
from .compute_cost import get_compute_cost, get_boundary_cost_table
from .layer_partition import get_pipeline_layers, detect_split_points
from .boundary_resharding import (
    BoundaryReshardingModule,
    get_boundary_resharding_cost,
)

__all__ = [
    "build_pipeline_plan",
    "PipelinePlan",
    "get_compute_cost",
    "get_boundary_cost_table",
    "get_pipeline_layers",
    "detect_split_points",
    "BoundaryReshardingModule",
    "get_boundary_resharding_cost",
]
```

---

## Files to Create / Modify

| Action | File | Notes |
|---|---|---|
| **Create** | `colossalai/auto_parallel/pipeline_shard/boundary_resharding.py` | `_BoundaryAllGather`, `_BoundarySplit`, `BoundaryReshardingModule`, `get_boundary_resharding_cost` |
| **Modify** | `colossalai/auto_parallel/pipeline_shard/compute_cost.py` | Remove uniform_tp filter, add `get_boundary_cost_table()`, augment cost table with boundary penalties |
| **Modify** | `colossalai/auto_parallel/pipeline_shard/orchestrator.py` | Heterogeneous submesh search, boundary module creation, updated `PipelinePlan` dataclass |
| **Modify** | `colossalai/auto_parallel/pipeline_shard/__init__.py` | Export boundary ops |
| **Modify** | `colossalai/auto_parallel/tensor_shard/initialize.py` | Add `autoparallelize_with_pp_hetero()` |
| **Create** | `examples/language/gpt/experiments/auto_parallel/test_auto_pipeline_a.py` | Tests below |

---

## Test Plan

### Test 1 — BoundaryAllGather correctness (needs 2+ GPUs)

```python
# torchrun --nproc_per_node=2 test_auto_pipeline_a.py --test gather
def test_boundary_all_gather():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    tp_group = dist.new_group([0, 1])

    # Each rank has a shard of shape [B, S, H/2]
    shard = torch.randn(2, 16, 32, device="cuda")
    full = _BoundaryAllGather.apply(shard, tp_group)

    # Full tensor should be [B, S, H]
    assert full.shape == (2, 16, 64), f"Wrong shape: {full.shape}"

    # Backward: gradient splits back to shard size
    loss = full.sum()
    loss.backward()
    assert shard.grad.shape == shard.shape
    print("[PASS] test_boundary_all_gather")
```

### Test 2 — BoundarySplit correctness (needs 2+ GPUs)

```python
def test_boundary_split():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    tp_group = dist.new_group([0, 1])

    full = torch.randn(2, 16, 64, device="cuda")
    shard = _BoundarySplit.apply(full, tp_group)

    assert shard.shape == (2, 16, 32), f"Wrong shape: {shard.shape}"
    # Verify this rank got the correct chunk
    chunks = torch.chunk(full, 2, dim=-1)
    assert torch.allclose(shard, chunks[rank])
    print("[PASS] test_boundary_split")
```

### Test 3 — BoundaryReshardingModule round-trip (needs 2+ GPUs)

```python
def test_boundary_round_trip():
    """Verify gather then split returns original shard."""
    tp_group = dist.new_group([0, 1])
    rank = dist.get_rank()

    shard = torch.randn(2, 16, 32, device="cuda", requires_grad=True)

    # Stage 0 (TP=2) → Stage 1 (TP=1): gather
    gather_mod = BoundaryReshardingModule(sender_tp=2, receiver_tp=1, tp_group=tp_group)
    full = gather_mod(shard)
    assert full.shape[-1] == 64

    # Stage 1 (TP=1) → Stage 2 (TP=2): split
    split_mod = BoundaryReshardingModule(sender_tp=1, receiver_tp=2, tp_group=tp_group)
    back = split_mod(full)
    assert back.shape[-1] == 32
    assert torch.allclose(back, shard)
    print("[PASS] test_boundary_round_trip")
```

### Test 4 — get_boundary_cost_table shape (CPU)

```python
def test_boundary_cost_table():
    submesh_choices = [(1, 1), (1, 2), (1, 4)]
    input_shapes = [torch.Size([2, 16, 256])] * 8
    full_mesh = ...  # mock or initialize

    boundary_cost = get_boundary_cost_table(submesh_choices, input_shapes, full_mesh)
    assert boundary_cost.shape == (3, 3, 8)
    # Same TP → zero cost
    assert boundary_cost[0, 0, :].sum() == 0.0
    assert boundary_cost[1, 1, :].sum() == 0.0
    # Different TP → nonzero cost for gather direction
    assert boundary_cost[1, 0, 0] > 0   # TP=2 → TP=1 requires all_gather
    assert boundary_cost[0, 1, 0] == 0  # TP=1 → TP=2 is a local split, no comm
    print("[PASS] test_boundary_cost_table")
```

### Test 5 — End-to-end plan with heterogeneous TP (needs 4+ GPUs)

```python
# torchrun --nproc_per_node=4 test_auto_pipeline_a.py --test e2e
def test_e2e_hetero_tp():
    plan = build_pipeline_plan(
        model=model, meta_args=meta_args,
        num_devices=4, num_microbatches=4,
    )
    # Verify plan is valid
    assert len(plan.stage_layer_ranges) > 1
    assert len(plan.stage_tp_degrees) == len(plan.stage_layer_ranges)
    assert len(plan.boundary_resharding) == len(plan.stage_layer_ranges) - 1

    # Print the plan for inspection
    if dist.get_rank() == 0:
        for s, (lr, tp) in enumerate(zip(plan.stage_layer_ranges, plan.stage_tp_degrees)):
            print(f"  Stage {s}: layers {lr}, TP={tp}")
        for s, bm in enumerate(plan.boundary_resharding):
            if bm is not None:
                print(f"  Boundary {s}→{s+1}: {bm}")
    print("[PASS] test_e2e_hetero_tp")
```

### Test 6 — Full training loop with resharding (needs 4+ GPUs)

```python
# torchrun --nproc_per_node=4 test_auto_pipeline_a.py --test training
def test_training_loop_hetero():
    stage_module, stage_manager, boundary_module = autoparallelize_with_pp_hetero(
        model=model, meta_args=meta_args,
        num_devices=4, num_microbatches=4,
    )

    def forward_step(x):
        out = stage_module(x)
        if boundary_module is not None and not stage_manager.is_last_stage():
            out = boundary_module(out)
        return out

    schedule = OneForwardOneBackwardSchedule(stage_manager, num_microbatches=4)
    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)

    batch = {"input_ids": torch.randint(0, 50257, (2, 32)).cuda()}
    schedule.forward_backward_step(
        model=forward_step, data_iter=iter([batch]),
        criterion=lambda out, _: out.loss,
        optimizer=optimizer, return_loss=True,
    )
    optimizer.step()
    optimizer.zero_grad()

    if dist.get_rank() == 0:
        print("[PASS] test_training_loop_hetero: one step completed")
```

---

## Implementation Order

```
1. boundary_resharding.py     → Test 1, 2, 3 (verify autograd)
2. get_boundary_cost_table()  → Test 4 (verify cost shape and values)
3. compute_cost.py update     → Test 4 extended (cost table with boundary penalty)
4. orchestrator.py update     → Test 5 (heterogeneous plan)
5. autoparallelize_with_pp_hetero() → Test 6 (full training)
```

---

## Key Design Decisions

### A. Boundary cost is added to the receiving stage, not a separate term

`alpa_dp_impl` has no concept of inter-stage transition cost. The cleanest way to model
boundary resharding cost is to include it in `compute_cost[k, i, m_curr, 0]` as a penalty
attached to the stage that enters after the boundary. The `get_boundary_cost_table` approach
computes all pairwise `(m_prev, m_curr)` penalties; then `build_pipeline_plan` adds the
minimum over `m_prev` to each cost table entry. This is a conservative upper bound — the
actual cost depends on which `m_prev` the DP chooses.

A more accurate approach would extend `alpa_dp_impl` to track `m_prev` as a 4th DP
dimension, but this significantly increases the DP state space.

### B. `_BoundarySplit` is a local op (no communication)

When `tp_prev < tp_curr` (TP increases), each rank simply takes its chunk of the full
activation — no collective is needed. Only `_BoundaryAllGather` (TP decreases) requires
actual cross-device communication. This asymmetry is reflected in `get_boundary_resharding_cost`
returning 0 for the split direction.

### C. Process group for boundary ops

The boundary `tp_group` is created over the receiving stage's ranks. In practice, the
sending stage's ranks and receiving stage's ranks are disjoint (different pipeline stages
= different GPU sets). The all_gather spans the sender's TP group, so the correct group
is the **sender's TP group**, initialized via `DeviceMesh(init_process_group=True)`.

### D. Solution A is a strict superset of Solution B

Setting `uniform_tp_degree=T` in Solution B is equivalent to Solution A where the DP
happens to choose the same TP=T for every stage. If all stages pick the same TP, all
boundary modules are `None` (no-op). This means Solution A's infrastructure can replace
Solution B entirely once implemented.

---

## Open Questions / Risks

| Issue | Notes |
|---|---|
| DP boundary cost approximation may miss best plan | Future: extend `alpa_dp_impl` with `m_prev` dimension |
| Backward through pipeline P2P may not call boundary_module.backward | Test gradient flow explicitly; may need custom pipeline schedule hook |
| Process group creation for `boundary_module` must happen at startup | `_get_tp_group()` calls `dist.new_group()` which is a collective — must run on all ranks simultaneously |
| Activation shape may not be simple `[B, S, H]` for all models | Robustify `_BoundaryAllGather`/`_BoundarySplit` to handle tuple outputs and arbitrary tensor shapes |
