# Phase 3 Plan — Variable-Size Pipeline Stages

## Goal

Remove the equal-devices-per-stage constraint so the planner can assign
different device counts to different stages, e.g. pp=2 with stage 0 using
1 GPU (tp=1) and stage 1 using 2 GPUs (tp=2) — total 3 GPUs.

This matches what Alpa does natively and is the correct generalisation of Phase 2.

---

## Why Phase 2 Cannot Do This

Phase 2 enforces `pp_size * max_tp * dp_size == world_size`, which forces
all stages to use the same number of devices (`devices_per_stage = max_tp * dp`).

The two specific blockers:

1. **`ProcessGroupMesh(pp, max_tp, dp)`** is a regular 3D grid. Every rank has
   exactly one (pp, tp, dp) coordinate. Variable-size stages cannot be expressed
   in a regular grid.

2. **`PipelineP2PCommunication`** routes P2P by a simple rank offset
   (`next_rank = current_rank + devices_per_stage`). If stages have different
   sizes, this offset is wrong.

---

## What Alpa Does (Reference)

| Component | Alpa | Our Phase 2 |
|---|---|---|
| Stage device count | Variable (`sum == world_size`) | Equal (`pp * max_tp * dp`) |
| Mesh topology | `PhysicalDeviceMeshGroup` (list of independent meshes) | `ProcessGroupMesh(pp, max_tp, dp)` |
| Cross-stage comm | `ReshardingTask` (arbitrary src/dst mesh shapes) | `PipelineP2PCommunication` (1-to-1 rank offset) |
| Boundary resharding | Built into `ReshardingTask` | `BoundaryAllGather + BoundarySplit` |

Key Alpa file: `alpa/pipeline_parallel/cross_mesh_resharding.py` —
`ReshardingTask` with `src_mesh` / `dst_mesh` of different sizes.

---

## DP Uniformity: Our Plan vs Alpa

### Alpa: No DP constraint at all

Alpa's `cross_mesh_resharding.py` uses a **tile-based resharding** system
(`ReshardingTaskSpec._look_up_dst_tile_from_src`). It maps source mesh tiles
(shards on sender) to destination mesh tiles (shards on receiver) for ANY
combination of sender/receiver mesh shapes — including different DP degrees.

Example Alpa handles natively:
```
Stage 0: tp=1, dp=4  (4 ranks, each processes batch/4)
Stage 1: tp=2, dp=2  (4 ranks, TP-pairs [0,1] and [2,3], each pair processes batch/2)
→ ReshardingTask merges 2 dp replicas from stage 0 into 1 dp replica of stage 1
```

No assertion or filter — the DP solver is free to assign any submesh to any stage.

### Phase 3 Plan: Two implementation options

**Option A — Uniform DP (simpler, sufficient for most cases)**

Add a post-filter in `_parse_solution_variable()` that rejects solutions
where `dp_per_stage` is non-uniform. This covers the most common hetero-TP
use case (same dp, different tp per stage) with simple broadcast-based P2P.

```
Stage 0: tp=1, dp=1  (1 GPU)
Stage 1: tp=2, dp=1  (2 GPUs)
Both dp=1 → uniform ✓ → simple broadcast works
```

**Option B — Full tile-based resharding (matches Alpa, more complex)**

Port Alpa's `ReshardingTaskSpec` tile mapper to handle arbitrary dp transitions.
Allows the planner to fully explore all submesh combinations.

```
Stage 0: tp=1, dp=2  (2 GPUs, each processes batch/2)
Stage 1: tp=4, dp=1  (4 GPUs, all TP on same batch)
→ Tile mapper: each dp replica of stage 0 sends to subset of stage 1 TP ranks
```

**Recommendation: Start with Option A.** It removes the equal-devices constraint
and enables the 1+2=3 GPU case. Option B is a follow-up if dp transitions are needed.

---

## Files to Change

### 1. `pipeline_shard/orchestrator.py`

#### 1a. `PipelinePlan` — new fields

```python
@dataclass
class PipelinePlan:
    # existing fields ...
    dp_per_stage: List[int] = field(default_factory=list)   # NEW: dp per stage
    rank_ranges: List[List[int]] = field(default_factory=list)  # NEW: ranks per stage
    variable_stage_sizes: bool = False                       # NEW: Phase 3 flag
```

`rank_ranges[s]` = list of global rank indices belonging to stage s.
Example for tp=[1,2], dp=1, world_size=3: `[[0], [1, 2]]`.

#### 1b. `build_pipeline_plan()` — Phase 3 path

New parameter: `variable_stage_sizes: bool = False`

```python
if variable_stage_sizes:
    # Run alpa_dp with ALL submeshes simultaneously (same as Phase 2).
    cost_table = get_compute_cost(layers, meta_args, all_submeshes, ...)
    boundary_table = get_boundary_cost_table(all_submeshes, ...)
    min_incoming = boundary_table.min(axis=0)
    for m in range(len(all_submeshes)):
        cost_table[:, :, m, 0] += min_incoming[m]

    cost, solution = alpa_dp(num_layers, num_devices, ...)

    # NEW: reject solutions with non-uniform dp across stages
    plan = _parse_solution_variable(solution, all_submeshes, num_devices, cost)
    if plan is None:
        raise RuntimeError("No valid plan with uniform dp across stages.")
    return plan
```

#### 1c. `_parse_solution_variable()` — new helper

```python
def _parse_solution_variable(solution, submesh_choices, num_devices, cost,
                              require_uniform_dp=True):
    tp_per_stage = [int(submesh_choices[m][1]) for (_, m, _) in solution by stage]
    device_counts = [int(prod(submesh_choices[m])) for (_, m, _) in solution by stage]
    dp_per_stage = [d // t for d, t in zip(device_counts, tp_per_stage)]

    # Option A: reject non-uniform dp (simpler P2P routing)
    if require_uniform_dp and len(set(dp_per_stage)) > 1:
        return None
    # Option B: allow non-uniform dp (requires tile-based CrossMeshP2P)

    # Build rank_ranges: stage s gets ranks [offset, offset + device_counts[s])
    rank_ranges = []
    offset = 0
    for n in device_counts:
        rank_ranges.append(list(range(offset, offset + n)))
        offset += n

    return PipelinePlan(
        stage_layer_ranges=...,
        submesh_per_stage=...,
        tp_per_stage=tp_per_stage,
        dp_per_stage=dp_per_stage,
        rank_ranges=rank_ranges,
        pp_size=len(solution),
        tp_size=max(tp_per_stage),
        dp_size=dp_per_stage[0] if len(set(dp_per_stage)) == 1 else -1,
        estimated_cost=cost,
        heterogeneous_tp=any(t != tp_per_stage[0] for t in tp_per_stage),
        variable_stage_sizes=True,
    )
```

#### 1d. `autoparallelize_with_pp()` — Phase 3 path

Replace the `ProcessGroupMesh` block entirely when `plan.variable_stage_sizes`:

**Step 4 (new) — PP process groups without ProcessGroupMesh:**

```python
if plan.variable_stage_sizes:
    # Create PP send/recv groups for each adjacent stage pair.
    # Each group contains ALL ranks of stage s UNION stage s+1.
    # dist.new_group() must be called by ALL ranks for every group.
    pp_groups = {}   # (s, s+1) -> ProcessGroup
    for s in range(pp_size - 1):
        group_ranks = plan.rank_ranges[s] + plan.rank_ranges[s + 1]
        g = dist.new_group(ranks=group_ranks)
        if rank in group_ranks:
            pp_groups[(s, s + 1)] = g
```

**Step 5 (new) — per-stage DeviceMesh:**

```python
all_stage_meshes = []
for s in range(pp_size):
    tp_s = tp_per_stage[s]
    dp_s = dp_per_stage[s]
    stage_ranks = torch.tensor(plan.rank_ranges[s])
    mesh = DeviceMesh(
        physical_mesh_id=stage_ranks,
        logical_mesh_id=stage_ranks.reshape(tp_s, dp_s),
        mesh_alpha=list(mesh_alpha),
        mesh_beta=list(mesh_beta),
        init_process_group=True,
    )
    all_stage_meshes.append(mesh)
```

**Step 6 (new) — custom PipelineStageManager:**

`PipelineStageManager` requires a `ProcessGroupMesh` which does not exist for variable stages. Two options:

- **Option A**: Extend `PipelineStageManager` with a `from_rank_ranges()` factory
  that accepts rank ranges directly and creates a minimal pp group per stage pair.
- **Option B**: Create `VariableStagePipelineManager` that reimplements
  `is_first_stage()`, `is_last_stage()`, `stage` using `plan.rank_ranges`.

Option B is less invasive:

```python
class VariableStagePipelineManager:
    def __init__(self, plan: PipelinePlan, rank: int):
        self.plan = plan
        self.rank = rank
        self.stage = next(
            s for s, rr in enumerate(plan.rank_ranges) if rank in rr
        )
        self.num_stages = plan.pp_size

    def is_first_stage(self): return self.stage == 0
    def is_last_stage(self):  return self.stage == self.num_stages - 1
```

**Step 7 (new) — CrossMeshP2PCommunication:**

See Section 2 below.

---

### 2. New file: `pipeline_shard/cross_mesh_p2p.py`

This is the core new component for Phase 3.

#### Design

For each boundary (stage s → stage s+1):

```
Sender stage s:   tp_s ranks, all holding the same full tensor [B, S, H]
                  (Megatron TP: output is always replicated)
Receiver stage s+1: tp_{s+1} ranks that need the same full tensor

Routing rule:
  dp replica r on stage s  →  dp replica r on stage s+1
  Within each dp replica:
    sender tp ranks  →  ALL receiver tp ranks (broadcast within dp replica)
```

The mapping per dp replica `r`:
- Sender ranks for dp-r on stage s: `stage_s_ranks[r * tp_s : (r+1) * tp_s]`
  (just 1 representative needed since all have same tensor)
- Receiver ranks for dp-r on stage s+1: `stage_{s+1}_ranks[r * tp_{s+1} : (r+1) * tp_{s+1}]`
- Communication: **broadcast** from sender dp-r representative to all receiver dp-r TP ranks

#### Implementation plan

```python
class CrossMeshP2PCommunication:
    """
    Replaces PipelineP2PCommunication for variable-size stages.

    At init, for each stage boundary (s, s+1):
      - Build send_groups[s]:  one dist.ProcessGroup per dp replica
            members = {one sender rank from dp-r} ∪ {all receiver ranks of dp-r}
      - Build recv_groups[s]:  same groups, viewed from receiver side

    send_forward(tensor, stage):
      For each dp replica r where this rank is in sender-r:
        dist.broadcast(tensor, src=sender_representative, group=send_groups[s][r])

    recv_forward(stage):
      For each dp replica r where this rank is in receiver-r:
        dist.broadcast(buffer, src=sender_representative, group=recv_groups[s][r])
      return buffer
    """

    def __init__(self, plan: PipelinePlan, rank: int):
        self.rank = rank
        self.plan = plan
        self.send_groups: Dict[int, List[ProcessGroup]] = {}  # s → [group_r0, group_r1...]
        self.recv_groups: Dict[int, List[ProcessGroup]] = {}  # s+1 → [group_r0, ...]
        self._build_groups()

    def _build_groups(self):
        # ALL ranks must call dist.new_group() for every group created.
        # Build groups for each boundary and each dp replica.
        dp = self.plan.dp_size
        for s in range(self.plan.pp_size - 1):
            tp_s    = self.plan.tp_per_stage[s]
            tp_sp1  = self.plan.tp_per_stage[s + 1]
            ranks_s   = self.plan.rank_ranges[s]
            ranks_sp1 = self.plan.rank_ranges[s + 1]

            self.send_groups[s]   = []
            self.recv_groups[s+1] = []

            for r in range(dp):
                # Sender dp replica r: pick first TP rank as broadcaster
                sender_tp_ranks = ranks_s[r * tp_s : (r + 1) * tp_s]
                sender_rep = sender_tp_ranks[0]

                # Receiver dp replica r: all TP ranks
                recv_tp_ranks = ranks_sp1[r * tp_sp1 : (r + 1) * tp_sp1]

                group_members = [sender_rep] + list(recv_tp_ranks)
                g = dist.new_group(ranks=group_members)
                # Store only if this rank participates
                if self.rank in group_members:
                    self.send_groups[s].append((r, sender_rep, g))
                    self.recv_groups[s+1].append((r, sender_rep, g))

    def send_forward(self, tensor: torch.Tensor, stage: int):
        for r, sender_rep, g in self.send_groups.get(stage, []):
            if self.rank == sender_rep:
                dist.broadcast(tensor.contiguous(), src=sender_rep, group=g)

    def recv_forward(self, stage: int, shape, dtype) -> torch.Tensor:
        buffer = torch.empty(shape, dtype=dtype, device="cuda")
        for r, sender_rep, g in self.recv_groups.get(stage, []):
            if self.rank != sender_rep:
                dist.broadcast(buffer, src=sender_rep, group=g)
        return buffer

    def send_backward(self, grad: torch.Tensor, stage: int):
        # Gradient flows back: receiver → sender (reversed direction)
        for r, sender_rep, g in self.recv_groups.get(stage, []):
            if self.rank != sender_rep:
                dist.broadcast(grad.contiguous(), src=self.rank, group=g)
                # NOTE: need a separate backward group (sender becomes receiver)
                # Full impl requires separate backward process groups.

    def recv_backward(self, stage: int, shape, dtype) -> torch.Tensor:
        buffer = torch.empty(shape, dtype=dtype, device="cuda")
        for r, sender_rep, g in self.send_groups.get(stage, []):
            if self.rank == sender_rep:
                dist.broadcast(buffer, src=???, group=g)
        return buffer
```

**Note:** The backward pass requires separate process groups where the *receiver*
stage broadcasts gradients back to the *sender* stage. The forward and backward groups
have opposite broadcast roots — build them separately at init.

#### Simpler alternative: unicast (one sender representative per dp replica)

Instead of broadcast-to-all-TP-receivers, have:
- Sender representative (rank 0 of each sender TP group) sends to receiver representative
- Receiver TP group then broadcasts within itself

This reduces cross-stage traffic:

```
send_forward:
  sender_rep[r] → recv_rep[r]  (unicast)

within receiver stage:
  recv_rep[r] broadcasts within its TP group  (intra-stage broadcast)
```

The intra-stage broadcast is a normal TP group operation and doesn't need
new process groups. This is cheaper and simpler to implement.

---

### 3. `pipeline_shard/__init__.py`

Add exports:

```python
from .cross_mesh_p2p import CrossMeshP2PCommunication
from .orchestrator import VariableStagePipelineManager
```

---

### 4. `run_3d_auto_parallel.py` — training loop update

```python
# Phase 3: use variable-stage manager and cross-mesh P2P
if plan.variable_stage_sizes:
    stage_manager = VariableStagePipelineManager(plan, rank)
    p2p = CrossMeshP2PCommunication(plan, rank)
else:
    # Phase 1/2: existing code
    pg_mesh = ProcessGroupMesh(pp_size, tp_size, dp_size)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)
    p2p = PipelineP2PCommunication(stage_manager, overlap_p2p=False)

# Training loop — forward
if is_first:
    out = stage_module(x)
    p2p.send_forward(out, stage=current_stage)  # new signature

elif is_last:
    tensor_shape = (args.batch, args.seq, args.hidden)
    recv = p2p.recv_forward(stage=current_stage, shape=tensor_shape, dtype=torch.float32)
    recv = recv.requires_grad_(True)
    out  = stage_module(recv)
    ...
```

---

## Implementation Order

### Option A — Uniform DP (recommended starting point)

| Step | Task | Complexity | Depends On |
|---|---|---|---|
| 1 | Add `dp_per_stage`, `rank_ranges`, `variable_stage_sizes` to `PipelinePlan` | Low | — |
| 2 | Add `_parse_solution_variable()` with `require_uniform_dp=True` filter | Low | Step 1 |
| 3 | Add `variable_stage_sizes=True` path in `build_pipeline_plan()` | Low | Step 2 |
| 4 | Implement `VariableStagePipelineManager` | Low | Step 1 |
| 5 | Implement `CrossMeshP2PCommunication` forward (broadcast within dp replica) | Medium | Steps 1, 4 |
| 6 | Implement `CrossMeshP2PCommunication` backward (reversed broadcast) | Medium | Step 5 |
| 7 | Update `autoparallelize_with_pp()` Phase 3 path (no `ProcessGroupMesh`) | Medium | Steps 4, 5 |
| 8 | Update training loop in `run_3d_auto_parallel.py` | Low | Steps 4, 5, 7 |
| 9 | Unit tests | Medium | Step 5 |
| 10 | End-to-end test: 3 GPUs, pp=2, tp=[1,2] | Medium | All |

### Option B — Full tile-based resharding (follow-up, matches Alpa fully)

| Step | Task | Complexity | Depends On |
|---|---|---|---|
| B1 | Port Alpa's `ReshardingTaskSpec` tile mapper to PyTorch/ColossalAI | High | Option A Step 1 |
| B2 | Implement `TileBasedCrossMeshP2P` using the tile mapper | High | B1 |
| B3 | Remove `require_uniform_dp` filter in `_parse_solution_variable()` | Low | B2 |
| B4 | Tests for dp-transition cases (e.g. dp=2 → dp=1 across stages) | High | B2, B3 |

---

## Key Risks and Open Questions

### Risk 1: `dist.new_group()` call consistency
Every rank must call `dist.new_group()` for every group, even if it doesn't
participate. In Phase 3 with variable stages, some ranks may not participate
in many groups. The `_build_groups()` method must iterate all groups and call
`dist.new_group()` universally, storing the result only when the rank participates.

### Risk 2: Backward pass group topology
Gradients flow backwards: stage s+1 → stage s. The broadcast direction is reversed.
Need separate groups for backward (or reuse the same groups with a different root).
The simplest approach: build `backward_send_groups` and `backward_recv_groups`
as mirrors of the forward groups, swapping sender and receiver roles.

### Risk 3: Tensor shape negotiation
`recv_forward()` needs to know the tensor shape in advance (for buffer allocation).
Options:
- Pass `shape` and `dtype` explicitly (as in the design above)
- Send shape metadata before the tensor (extra round-trip)
- Pre-compute shape from `meta_args` (preferred, avoids round-trip)

### Risk 4: Megatron TP output is replicated
As established in Phase 2: ColossalAI's Megatron-style TP produces a FULL
tensor on every TP rank (after AllReduce). Therefore:
- `BoundaryAllGather` is always a noop for Megatron TP
- `BoundarySplit` should NOT be applied at the receiver
- The broadcast in `CrossMeshP2PCommunication` sends the full tensor from ONE
  sender TP rank (the representative) — correct because all sender TP ranks
  hold the same full tensor

### Risk 5: dp > 1 routing
The design above handles dp > 1 by grouping sender and receiver by dp replica.
But the ProcessGroupMesh's notion of "dp replica" must be reconstructed from
`rank_ranges` and `tp_per_stage`. Verify the rank layout matches:
  - Stage s, dp replica r: `rank_ranges[s][r * tp_s : (r+1) * tp_s]`

This assumes ranks within a stage are laid out as `(dp, tp)` row-major.
Must match the DeviceMesh reshape in `autoparallelize_with_pp()`.

---

## Test Plan

### Unit tests (no distributed, single process)

1. `_parse_solution_variable()` correctly builds `rank_ranges` for tp=[1,2] on 3 GPUs
2. `_parse_solution_variable()` rejects non-uniform dp (e.g. tp=[1,3] with dp=[1,1] impossible for 4 devices)
3. `VariableStagePipelineManager.stage` correctly identifies each rank's stage

### Distributed tests (3 GPUs minimum)

4. `CrossMeshP2PCommunication.send_forward / recv_forward`: sender rank 0 sends
   tensor; receiver ranks 1,2 both receive the same tensor → `torch.allclose` ✓
5. Full forward+backward on 3 GPUs, pp=2, tp=[1,2], 2 steps, loss decreases

### Regression tests

6. All Phase 1 tests still pass (`test_phase1_pipeline.py`)
7. All Phase 2 tests still pass (`test_phase2_boundary.py`)
8. `run_3d_auto_parallel.py` on 2 GPUs (Phase 1 mode) still works

---

## Comparison: Phase 1 / 2 / 3

| Feature | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| TP per stage | Uniform | Hetero (equal devices) | Hetero (variable devices) |
| Stage device count | Equal | Equal | Variable |
| Devices topology | `ProcessGroupMesh(pp, tp, dp)` | Same | Per-stage `DeviceMesh` + custom PP groups |
| P2P communication | `PipelineP2PCommunication` | Same | `CrossMeshP2PCommunication` |
| Min GPUs for hetero | N/A | 8 (auto) | 3 (manual) / 3 (auto for 1+2) |
| Auto-finds hetero plan | No | Yes (8+ GPUs) | Yes (any world_size ≥ 3) |
| BoundaryAllGather | Not needed | Present (noop for Megatron TP) | Replaced by CrossMesh broadcast |
| BoundarySplit | Not needed | Present (breaks Megatron TP) | Removed |

---

## Files Summary

| File | Action | Description |
|---|---|---|
| `pipeline_shard/orchestrator.py` | Modify | Add `variable_stage_sizes` path, `_parse_solution_variable()`, `VariableStagePipelineManager`, Phase 3 `autoparallelize_with_pp()` |
| `pipeline_shard/cross_mesh_p2p.py` | New | `CrossMeshP2PCommunication` replacing 1-to-1 P2P |
| `pipeline_shard/__init__.py` | Modify | Export new classes |
| `run_3d_auto_parallel.py` | Modify | Branch on `plan.variable_stage_sizes` for stage manager and P2P |
| `test_phase3_variable_stages.py` | New | Unit + distributed tests |
