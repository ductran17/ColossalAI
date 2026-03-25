# Auto 3D Parallel Architecture — Complete Technical Design

## Overview

This document describes the architecture for adding automatic pipeline parallelism (PP) to
ColossalAI's existing auto tensor+data parallelism (TP+DP), forming full **3D auto-parallelism**.

Two paths are defined:
- **Phase 1 (Solution B)**: Uniform TP — same TP degree on all pipeline stages. No boundary
  resharding needed. Fully implementable with existing PyTorch primitives.
- **Phase 2 (Solution A)**: Heterogeneous TP — different TP degrees per stage. Requires boundary
  resharding using `torch.distributed._functional_collectives` and Galvatron-style fused
  process groups.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Auto 3D Parallel Planner                        │
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐     │
│  │  1. Profile   │───▶│  2. Plan PP  │───▶│  3. Plan TP+DP/stage  │    │
│  │  (α,β costs)  │    │  (alpa_dp)   │    │  (initialize_model)   │    │
│  └──────────────┘    └──────────────┘    └───────────────────────┘     │
│         │                   │                        │                  │
│         ▼                   ▼                        ▼                  │
│  AlphaBetaProfiler   compute_cost.py          ILP Solver (PuLP)        │
│  → mesh_alpha/beta   → cost per (stage,       → sharding strategy      │
│                        submesh, layers)          per operator           │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              4. Build Runtime (orchestrator.py)                   │  │
│  │                                                                  │  │
│  │  ProcessGroupMesh(pp, tp, dp) → PipelineStageManager             │  │
│  │  Per-stage DeviceMesh → initialize_model() → ModuleWrapper       │  │
│  │  [Phase 2 only] BoundaryReshardingModule at stage transitions    │  │
│  │  Pipeline schedule (1F1B / interleaved)                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Uniform TP (Solution B)

### Why This Works Without Resharding

When all stages share the same TP degree, the activation tensor at every stage boundary has
identical sharding: `Shard(dim=hidden, tp_degree=T)`. Rank i in stage N sends directly to
rank i in stage N+1 via P2P — no collective needed at boundaries.

This is the same approach used by:
- ColossalAI `HybridParallelPlugin` (single `tp_size` parameter)
- DeepSpeed `PipeModelDataParallelTopology` (uniform `num_mp`)
- Megatron-LM (uniform TP assumed throughout)

### Architecture Flow

```
                            User Code
                               │
                               ▼
                  autoparallelize_with_pp(model, meta_args,
                      num_devices, num_microbatches,
                      uniform_tp_degree=None)
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                 ▼
     ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
     │ Profile      │  │ Trace model  │  │ Get cluster     │
     │ α,β costs    │  │ via ColoFX   │  │ topology        │
     │ (once,       │  │ → FX Graph   │  │ num_hosts,      │
     │  cached)     │  │              │  │ devs_per_host   │
     └──────┬──────┘  └──────┬───────┘  └────────┬────────┘
            │                │                    │
            ▼                ▼                    ▼
     ┌──────────────────────────────────────────────────────┐
     │           build_pipeline_plan()                       │
     │                                                       │
     │  1. submesh_choices = get_submesh_choices(H, D)       │
     │     Filter: keep only submeshes where                 │
     │     submesh[1] == uniform_tp_degree                   │
     │                                                       │
     │  2. For each (num_stages K, layer_range, submesh m):  │
     │     cost[k,i,m] = run initialize_model() on           │
     │     layers[k:i] with DeviceMesh(submesh_m)            │
     │     → ILP objective value = estimated stage time       │
     │                                                       │
     │  3. best_plan = alpa_dp(num_layers, K_max,            │
     │                         submesh_choices, cost)         │
     │     → layer_assignment: which layers in which stage    │
     │     → submesh_assignment: which submesh per stage      │
     └──────────────────────┬───────────────────────────────┘
                            │
                            ▼
     ┌──────────────────────────────────────────────────────┐
     │          build_runtime()                              │
     │                                                       │
     │  1. pg_mesh = ProcessGroupMesh(pp_size, tp_size,      │
     │                                dp_size)               │
     │     ✓ prod(pp, tp, dp) == world_size                  │
     │                                                       │
     │  2. stage_mgr = PipelineStageManager(pg_mesh,         │
     │                                      pipeline_axis=0) │
     │                                                       │
     │  3. For each stage s (runs on its own ranks):         │
     │     stage_layers = model.layers[assignment[s]]        │
     │     stage_ranks = ranks_for_stage(pg_mesh, s)         │
     │     local_mesh = DeviceMesh(stage_ranks,              │
     │                    mesh_shape=(tp_size, dp_size),      │
     │                    mesh_alpha=..., mesh_beta=...)      │
     │     wrapped = initialize_model(stage_layers,           │
     │                    meta_args, local_mesh)              │
     │     → ModuleWrapper with TP+DP sharding baked in      │
     │                                                       │
     │  4. Schedule: 1F1B pipeline with                      │
     │     PipelineP2PCommunication(stage_mgr)               │
     └──────────────────────┬───────────────────────────────┘
                            │
                            ▼
              Tuple[ModuleWrapper, PipelineStageManager]
```

### Rank Layout Example

8 GPUs, 2 nodes × 4 GPUs/node, chosen plan: PP=2, TP=2, DP=2

```
ProcessGroupMesh(2, 2, 2):

              TP=0   TP=1
        DP=0  [G0]   [G1]     ← Stage 0 (layers 0-5)
  PP=0  DP=1  [G2]   [G3]

        DP=0  [G4]   [G5]     ← Stage 1 (layers 6-11)
  PP=1  DP=1  [G6]   [G7]

TP groups:  {G0,G1}, {G2,G3}, {G4,G5}, {G6,G7}
DP groups:  {G0,G2}, {G1,G3}, {G4,G6}, {G5,G7}
PP groups:  {G0,G4}, {G1,G5}, {G2,G6}, {G3,G7}

P2P at boundary: G0→G4, G1→G5, G2→G6, G3→G7 (rank-to-rank, same sharding)
```

### Auto TP Degree Search

If `uniform_tp_degree=None`, search all valid TP degrees:

```python
def build_pipeline_plan(model, meta_args, num_devices, num_microbatches,
                        uniform_tp_degree=None, ...):
    H, D = num_hosts, devices_per_host

    if uniform_tp_degree is not None:
        tp_candidates = [uniform_tp_degree]
    else:
        # Try all TP degrees that are valid submesh column counts
        tp_candidates = [s[1] for s in get_submesh_choices(H, D)]
        tp_candidates = sorted(set(tp_candidates))

    best_plan, best_cost = None, float('inf')
    for tp in tp_candidates:
        filtered_submeshes = [s for s in get_submesh_choices(H, D) if s[1] == tp]
        cost_table = get_compute_cost(layers, meta_args, filtered_submeshes, ...)
        plan = alpa_dp(num_layers, K_max, filtered_submeshes, cost_table)
        if plan.cost < best_cost:
            best_plan, best_cost = plan, plan.cost

    return best_plan
```

### Data Flow at Stage Boundary (Uniform TP)

```
Stage 0, Rank 0 (TP=2, has hidden[:H/2])     Stage 1, Rank 0 (TP=2, expects hidden[:H/2])
   │                                              ▲
   │  P2P isend(tensor, dst=stage1_rank0)         │  P2P irecv(tensor, src=stage0_rank0)
   └──────────────────────────────────────────────┘

No resharding — sharding specs match exactly.
```

### Gaps in Phase 1: None

Phase 1 has no fundamental gaps. All components exist in ColossalAI today:
- `alpa_dp` / `alpa_dp_impl` — exists (needs 2 bug fixes documented in solution_b_plan.md)
- `initialize_model()` — exists, produces `ModuleWrapper` with TP+DP
- `ProcessGroupMesh` — exists, supports ND mesh
- `PipelineStageManager` — exists, manages P2P groups
- `DeviceMesh` with `AlphaBetaProfiler` — exists

What needs to be **written**:
- `compute_cost.py` — calls `initialize_model()` per (stage, submesh) to fill the cost table
- `orchestrator.py` — wires everything together
- Bug fixes to `solve_solution()` and `alpa_dp_impl`

---

## Phase 2: Heterogeneous TP (Solution A)

### Why This Is Harder

When stage 0 has TP=4 and stage 1 has TP=2, the activation at the boundary is:
- Stage 0 output: each rank holds `hidden[rank*H/4 : (rank+1)*H/4]`  (4 shards)
- Stage 1 expects: each rank holds `hidden[rank*H/2 : (rank+1)*H/2]`  (2 shards)

A collective operation must reshape 4 shards → 2 shards (or vice versa) at the boundary.

### Three Sub-Problems

```
┌──────────────────────────────────────────────────────────────────────┐
│                  Heterogeneous TP Boundary Resharding                │
│                                                                      │
│  Sub-problem 1: The collective operation                             │
│  ─────────────────────────────────────────                           │
│  When sender_tp > receiver_tp: all_gather within sender TP group,    │
│  then split into receiver_tp chunks.                                 │
│  When sender_tp < receiver_tp: each receiver rank receives a         │
│  sub-chunk from the sender.                                          │
│                                                                      │
│  Solved by: torch.distributed._functional_collectives                │
│  (all_gather_tensor, reduce_scatter_tensor)                          │
│  These are ATen-registered ops, torch.compile-compatible.            │
│                                                                      │
│  Sub-problem 2: Process group topology (fused groups)                │
│  ────────────────────────────────────────────────────                 │
│  Which ranks form the communicator for the boundary collective?      │
│  With heterogeneous TP, the mesh is non-rectangular.                 │
│  ProcessGroupMesh cannot represent this — need custom groups.        │
│                                                                      │
│  Solved by: Galvatron's merge_redistributed_group() logic            │
│  (galvatron/core/runtime/comm_groups.py:336-357)                     │
│  Creates "fused split groups" and "fused allgather groups" based     │
│  on the ratio of adjacent stages' TP degrees.                        │
│                                                                      │
│  Sub-problem 3: P2P routing (M:N rank mapping)                       │
│  ──────────────────────────────────────────────                       │
│  With uniform TP: rank i → rank i (1:1 mapping).                     │
│  With TP=4 → TP=2: 4 sender ranks must map to 2 receiver ranks.     │
│  Galvatron's approach: split BEFORE P2P (sender side) so each        │
│  P2P transfer is still 1:1, but with a smaller tensor.               │
│                                                                      │
│  Solved by: Porting Galvatron's fused_split_allgather() pattern      │
│  (galvatron/core/runtime/redistribute.py)                            │
└──────────────────────────────────────────────────────────────────────┘
```

### Architecture Flow (Additions Over Phase 1)

```
Phase 2 adds these components (marked with ★):

                  autoparallelize_with_pp_hetero(model, ...)
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
     Profile α,β       Trace model        Get topology
              │                │                  │
              ▼                ▼                  ▼
     ┌────────────────────────────────────────────────────────┐
     │  build_pipeline_plan()                                  │
     │                                                         │
     │  submesh_choices: ALL valid submeshes (no TP filter) ★  │
     │                                                         │
     │  cost[k,i,m] = stage_compute_cost                       │
     │              + boundary_reshard_cost(m_prev, m_curr) ★   │
     │                                                         │
     │  alpa_dp() → may assign different submeshes per stage   │
     └────────────────────────┬────────────────────────────────┘
                              │
                              ▼
     ┌────────────────────────────────────────────────────────┐
     │  build_runtime()                                        │
     │                                                         │
     │  1. Create per-stage process groups (not ProcessGroup-  │
     │     Mesh — non-rectangular topology)               ★    │
     │                                                         │
     │  2. Create fused boundary groups via                    │
     │     merge_redistributed_group(tp_prev, tp_next)    ★    │
     │                                                         │
     │  3. Per stage: initialize_model() → ModuleWrapper       │
     │                                                         │
     │  4. Insert BoundaryReshardingModule between stages ★    │
     │     Uses _functional_collectives ops on fused groups     │
     │                                                         │
     │  5. Pipeline schedule with resharding hooks              │
     └────────────────────────┬────────────────────────────────┘
                              │
                              ▼
     Tuple[ModuleWrapper, PipelineStageManager,
           List[BoundaryReshardingModule]]
```

### Boundary Resharding: Detailed Data Flow

**Example: Stage 0 (TP=4) → Stage 1 (TP=2), 8 GPUs**

```
Stage 0 ranks: [G0, G1, G2, G3]  (TP=4, DP=1)
Stage 1 ranks: [G4, G5, G6, G7]  (TP=2, DP=2)

Step 1: Fused Split (within Stage 0, BEFORE P2P)
────────────────────────────────────────────────
Fused split groups (from merge_redistributed_group):
  Group A: [G0, G2]  (stride = receiver_tp = 2)
  Group B: [G1, G3]

G0 has shard[0], G1 has shard[1], G2 has shard[2], G3 has shard[3]

After fused split:
  G0 keeps shard[0]           → will send to G4
  G1 keeps shard[1]           → will send to G5
  G2 keeps shard[2]           → will send to G6
  G3 keeps shard[3]           → will send to G7

Step 2: P2P Send/Recv (1:1 mapping, each rank sends to its pipeline peer)
──────────────────────────────────────────────────────────────────────────
  G0 ──isend──▶ G4    (shard[0])
  G1 ──isend──▶ G5    (shard[1])
  G2 ──isend──▶ G6    (shard[2])
  G3 ──isend──▶ G7    (shard[3])

Step 3: Fused AllGather (within Stage 1, AFTER P2P recv)
────────────────────────────────────────────────────────
Fused allgather groups:
  Group X: [G4, G5]  (TP group of Stage 1, DP copy 0)
  Group Y: [G6, G7]  (TP group of Stage 1, DP copy 1)

G4 has shard[0], G5 has shard[1] → all_gather → G4,G5 each have shard[0]+shard[1]
G6 has shard[2], G7 has shard[3] → all_gather → G6,G7 each have shard[2]+shard[3]

Result: Stage 1 TP=2, each rank holds half the hidden dimension ✓
```

**Example: Stage 0 (TP=2) → Stage 1 (TP=4), reverse direction**

```
Step 1: No fused split needed (sender TP < receiver TP)
Step 2: P2P Send/Recv (1:1)
  G0 ──isend──▶ G4
  G1 ──isend──▶ G5
  G2 ──isend──▶ G6
  G3 ──isend──▶ G7
Step 3: Fused Split at receiver (G4-G7 split received tensors into TP=4 shards)
```

### BoundaryReshardingModule Implementation

```python
# colossalai/auto_parallel/pipeline_shard/boundary_resharding.py

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_gather_tensor

class _FusedBoundaryAllGather(torch.autograd.Function):
    """Gather shards from fused_group after P2P recv, before next stage forward."""

    @staticmethod
    def forward(ctx, input_tensor, fused_group, gather_dim=0):
        ctx.fused_group = fused_group
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(fused_group)
        # _functional_collectives: ATen-registered, torch.compile-safe
        return all_gather_tensor(input_tensor, gather_dim, fused_group)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse: split gradient back into shards
        chunks = grad_output.chunk(ctx.world_size, dim=ctx.gather_dim)
        local_rank = dist.get_rank(ctx.fused_group)
        return chunks[local_rank].contiguous(), None, None


class _FusedBoundarySplit(torch.autograd.Function):
    """Split tensor within fused_group before P2P send to next stage."""

    @staticmethod
    def forward(ctx, input_tensor, fused_group, split_dim=0):
        ctx.fused_group = fused_group
        ctx.split_dim = split_dim
        world_size = dist.get_world_size(fused_group)
        local_rank = dist.get_rank(fused_group)
        chunks = input_tensor.chunk(world_size, dim=split_dim)
        return chunks[local_rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse: gather gradient from all ranks in fused group
        return all_gather_tensor(
            grad_output, ctx.split_dim, ctx.fused_group
        ), None, None


class BoundaryReshardingModule(torch.nn.Module):
    """Inserted at pipeline stage boundaries when TP degrees differ."""

    def __init__(self, sender_tp: int, receiver_tp: int,
                 fused_group: dist.ProcessGroup, position: str):
        """
        Args:
            sender_tp: TP degree of the sending stage
            receiver_tp: TP degree of the receiving stage
            fused_group: The fused communicator from merge_redistributed_group()
            position: "before_send" or "after_recv"
        """
        super().__init__()
        self.sender_tp = sender_tp
        self.receiver_tp = receiver_tp
        self.fused_group = fused_group
        self.position = position

    def forward(self, x):
        if self.sender_tp == self.receiver_tp:
            return x  # No resharding needed
        if self.position == "before_send" and self.sender_tp > self.receiver_tp:
            return _FusedBoundarySplit.apply(x, self.fused_group)
        elif self.position == "after_recv" and self.sender_tp < self.receiver_tp:
            return _FusedBoundaryAllGather.apply(x, self.fused_group)
        elif self.position == "after_recv" and self.sender_tp > self.receiver_tp:
            return _FusedBoundaryAllGather.apply(x, self.fused_group)
        return x
```

### Fused Group Creation (Ported from Galvatron)

```python
# colossalai/auto_parallel/pipeline_shard/fused_groups.py

def create_fused_boundary_groups(
    stage_tp_degrees: List[int],
    stage_rank_lists: List[List[int]],
) -> List[Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]]:
    """
    For each pair of adjacent stages, create the fused split/allgather groups.

    Returns:
        List of (fused_split_group, fused_allgather_group) per boundary.
        One of the two is None depending on direction.

    Logic (from Galvatron merge_redistributed_group):
        If sender_tp > receiver_tp:
            fused_split_group = ranks in sender with stride = receiver_tp
            fused_allgather_group = None (use receiver's existing TP group)
        If sender_tp < receiver_tp:
            fused_split_group = None
            fused_allgather_group = ranks in receiver with stride = sender_tp
    """
    boundary_groups = []
    for s in range(len(stage_tp_degrees) - 1):
        tp_send = stage_tp_degrees[s]
        tp_recv = stage_tp_degrees[s + 1]
        ranks_send = stage_rank_lists[s]
        ranks_recv = stage_rank_lists[s + 1]

        if tp_send == tp_recv:
            boundary_groups.append((None, None))
            continue

        if tp_send > tp_recv:
            # Create fused split groups within sender
            fused_split_groups = []
            num_dp_send = len(ranks_send) // tp_send
            for dp in range(num_dp_send):
                base = dp * tp_send
                for j in range(tp_recv):
                    group_ranks = [ranks_send[base + j + k * tp_recv]
                                   for k in range(tp_send // tp_recv)]
                    fused_split_groups.append(
                        dist.new_group(group_ranks)
                    )
            boundary_groups.append((fused_split_groups, None))
        else:
            # Create fused allgather groups within receiver
            fused_ag_groups = []
            num_dp_recv = len(ranks_recv) // tp_recv
            for dp in range(num_dp_recv):
                base = dp * tp_recv
                for j in range(tp_send):
                    group_ranks = [ranks_recv[base + j + k * tp_send]
                                   for k in range(tp_recv // tp_send)]
                    fused_ag_groups.append(
                        dist.new_group(group_ranks)
                    )
            boundary_groups.append((None, fused_ag_groups))

    return boundary_groups
```

### Cost Model Extension

```python
# In compute_cost.py — Phase 2 addition

def get_boundary_cost(
    sender_submesh: Tuple[int, int],
    receiver_submesh: Tuple[int, int],
    activation_bytes: int,
    mesh_beta: float,
    mesh_alpha: float,
) -> float:
    """
    Estimate communication cost of boundary resharding.

    For sender_tp > receiver_tp:
        Cost = all_gather within fused group of size (sender_tp / receiver_tp)
        bytes_transferred = activation_bytes * (1 - receiver_tp/sender_tp)
        time = alpha + beta * bytes_transferred

    For sender_tp < receiver_tp:
        Cost = all_gather within fused group of size (receiver_tp / sender_tp)
        time = alpha + beta * activation_bytes * (1 - sender_tp/receiver_tp)
    """
    tp_send = sender_submesh[1]   # column = TP degree
    tp_recv = receiver_submesh[1]

    if tp_send == tp_recv:
        return 0.0

    ratio = max(tp_send, tp_recv) / min(tp_send, tp_recv)
    bytes_moved = activation_bytes * (1.0 - 1.0 / ratio)
    return mesh_alpha + mesh_beta * bytes_moved
```

### Gaps in Phase 2

| Gap | Severity | Description | Mitigation |
|-----|----------|-------------|------------|
| `ProcessGroupMesh` non-rectangular | Medium | Different TP per stage → mesh is not a single ND array. Cannot use `ProcessGroupMesh` directly. | Create TP/DP groups manually per stage. Use `ProcessGroupMesh` only for the PP axis (which is always well-defined). |
| Backward pass resharding | Medium | The `_FusedBoundarySplit.backward` and `_FusedBoundaryAllGather.backward` must mirror the forward correctly. Gradient flow through P2P + resharding is tricky to debug. | Unit test each autograd function with `torch.autograd.gradcheck`. |
| Non-divisible TP ratios | Low | If `tp_send=4` and `tp_recv=3`, there's no clean split. Galvatron only handles power-of-2 TP ratios where one divides the other. | Constrain `alpa_dp` to only consider submeshes where adjacent stages' TP degrees are divisible. In practice, TP is always power-of-2 (1, 2, 4, 8). |
| Pipeline bubble interaction | Low | Boundary resharding adds latency to every micro-batch's stage transition, increasing the bubble. | Include resharding cost in `alpa_dp`'s cost model so the planner avoids unnecessary TP transitions. |
| `torch.compile` on `ModuleWrapper` | Not needed | ColossalAI's `runtime_apply_pass` uses Python dicts + old `dist` APIs → graph breaks. | Don't compile `ModuleWrapper`. Only optionally compile `BoundaryReshardingModule`. The TP+DP runs in eager mode as designed. |
| Memory spike from all_gather | Low | Boundary all_gather temporarily materializes the full (un-sharded) activation tensor. | Account for this in the memory budget passed to `alpa_dp`. |

---

## Comparison: Phase 1 vs Phase 2

| Aspect | Phase 1 (Uniform TP) | Phase 2 (Heterogeneous TP) |
|--------|----------------------|----------------------------|
| Complexity | Low — no new collective ops | High — fused groups, boundary modules, cost model extension |
| Code to write | ~500 lines (3 new files) | ~1200 lines (6 new files + modifications) |
| Search space | TP fixed → only PP stages + DP vary | Full 3D — TP, PP, DP all vary per stage |
| Throughput gain over TP+DP only | 15-40% (pipeline overlaps compute) | 20-50% (+ heterogeneous TP for uneven layers) |
| Marginal gain of Phase 2 over Phase 1 | — | 5-15% (Galvatron paper, model-dependent) |
| Risk | Low — all components exist | Medium — complex group topology, gradient correctness |
| Dependencies | ColossalAI only | + Galvatron's group logic, functional collectives |

---

## Implementation Order

```
Phase 1 (Solution B):
  1. Bug fix: solve_solution() expose last_objective     ← 30 min
  2. Bug fix: alpa_dp_impl off-by-one                    ← 30 min
  3. compute_cost.py (uniform TP filter + cost table)    ← core work
  4. orchestrator.py (wire everything)                   ← core work
  5. Tests: unit + integration with GPT-2 small          ← validation
  6. Benchmark: compare auto-3D vs manual HybridParallel ← proof of value

Phase 2 (Solution A) — only after Phase 1 is validated:
  7. boundary_resharding.py (autograd functions)
  8. fused_groups.py (port Galvatron group creation)
  9. Extend compute_cost.py (boundary cost table)
  10. Extend orchestrator.py (insert boundary modules)
  11. Remove uniform TP filter from submesh_choices
  12. Tests: gradient correctness + end-to-end heterogeneous TP
```

---

## Key Design Decisions

### Why not `torch.compile` the whole pipeline?

ColossalAI's `runtime_apply_pass` injects Python-level `runtime_apply()` functions into the
FX graph that index into Python dicts (`origin_dict[node_index]`) and call old-style
`torch.distributed` ops (not `_functional_collectives`). This causes graph breaks in Dynamo.
Rewriting this to be compile-compatible would require replacing the entire sharding runtime —
out of scope for this project.

Instead: each stage's `ModuleWrapper` runs in eager mode (as designed), and `torch.compile` is
optionally used only on the boundary resharding module.

### Why not DTensor for cross-mesh resharding?

`torch.distributed.tensor._redistribute.redistribute_local_tensor()` raises
`NotImplementedError("Cross device mesh comm not supported yet!")` (PyTorch 2.5.1).
Until PyTorch implements cross-mesh DTensor redistribution, we must use raw
`_functional_collectives` for boundary ops.

### Why port Galvatron's group logic instead of inventing our own?

Galvatron is the only active PyTorch system that has production-tested fused group creation
for heterogeneous TP boundaries. Their `merge_redistributed_group()` handles the rank
arithmetic for arbitrary TP ratios (where one divides the other). Re-inventing this logic
risks subtle rank-mapping bugs that only manifest at scale.

### Why `_functional_collectives` instead of regular `torch.distributed`?

Regular `torch.distributed.all_gather` is not graph-capturable — it operates on tensors
in-place and has no ATen kernel registration. `_functional_collectives.all_gather_tensor`
returns a new tensor, is registered as a fake ATen op, and can be traced by `torch.compile`.
Even without compiling, the functional API is cleaner for autograd integration because it
avoids in-place mutation.
