# Phase 2 Components — Heterogeneous TP (Solution A)

This document explains every component added or changed in Phase 2 compared to Phase 1
(uniform TP), what it does, and why it exists.

---

## Background: What Changed Between Phase 1 and Phase 2

| | Phase 1 | Phase 2 |
|---|---|---|
| TP per pipeline stage | Same for every stage | Can differ per stage |
| Cost search | One DP run per TP degree | One DP run over all submeshes |
| Stage boundary | No resharding needed | AllGather + Split when TP differs |
| Autograd at boundary | Not applicable | Custom backward through reshard ops |

**Why Phase 2 matters:** A model's early layers (cheap, communication-bound) may prefer
TP=1 or TP=2, while later layers (expensive, compute-bound) may prefer TP=4.
Forcing all stages to the same TP (Phase 1) leaves performance on the table.
Phase 2 lets the solver pick the best TP independently per stage.

---

## New File: `boundary_resharding.py`

Path: `colossalai/auto_parallel/pipeline_shard/boundary_resharding.py`

This file is entirely new in Phase 2. It provides the three classes that handle
activation resharding at pipeline stage boundaries.

---

### `BoundaryAllGather`

**What it is:** A `torch.autograd.Function` — a custom differentiable operation.

**What it does:**

- **Forward:** Each rank in the TP group holds a shard `[B, S, H/T]` of the hidden
  tensor. `all_gather` collects all shards and `cat` concatenates them along the hidden
  dimension, giving every rank the full tensor `[B, S, H]`.
- **Backward:** The incoming gradient `[B, S, H]` is chunked back into T slices.
  This rank takes only `chunk[rank]`, producing a gradient shard `[B, S, H/T]`.

**Where it is placed:** At the **output** of the sender stage (before P2P send), when
`sender_tp > 1`.

**Why it is needed:** P2P send/recv in ColossalAI is a 1-to-1 operation — rank 0 sends
to rank 0, rank 1 sends to rank 1, etc. If the sender stage uses TP=4 and the receiver
uses TP=1, the sender's rank 0 only holds `H/4` hidden units. The receiver's rank 0
expects the full `H`. AllGather gives every sender rank the complete tensor so the
1-to-1 P2P delivers the full tensor to every receiver rank.

```
Sender rank 0: [B, S, H/4]  ─┐
Sender rank 1: [B, S, H/4]  ─┤── AllGather ──► all ranks: [B, S, H]
Sender rank 2: [B, S, H/4]  ─┤
Sender rank 3: [B, S, H/4]  ─┘
```

**Cost:** Communicates `(T-1)/T` of the tensor across the TP group
(each rank already has `1/T` locally). This cost is added to the planner's cost
table so the DP solver weighs it against the compute savings of hetero TP.

---

### `BoundarySplit`

**What it is:** A `torch.autograd.Function` — custom differentiable, but no communication.

**What it does:**

- **Forward:** The received full tensor `[B, S, H]` is split into T equal chunks along
  the hidden dimension. This rank keeps only `chunk[tp_rank]`, giving `[B, S, H/T]`.
- **Backward:** The incoming gradient shard `[B, S, H/T]` is zero-padded back to the
  full size `[B, S, H]`. Only the positions belonging to `tp_rank` are non-zero.

**Where it is placed:** At the **input** of the receiver stage (after P2P recv), when
`receiver_tp > 1`.

**Why the zero-pad backward:** The P2P layer on the receiver side needs to send a
gradient tensor back to the sender. The sender's `BoundaryAllGather.backward` expects a
full-size `[B, S, H]` gradient so it can chunk it and each sender rank extracts its
correct slice `[B, S, H/T]`. The zero-pad provides exactly that full-size tensor, with
zeros in positions this rank did not compute.

```
Received: [B, S, H]
  rank 0 keeps: [B, S, H/4]   (positions 0 .. H/4-1)
  rank 1 keeps: [B, S, H/4]   (positions H/4 .. H/2-1)
  ...
```

**Why no communication:** Every receiver rank receives the same full tensor from its
paired sender rank. The split is a pure local tensor operation — no collective needed.

---

### `BoundaryReshardingModule`

**What it is:** An `nn.Module` wrapper around `BoundaryAllGather` / `BoundarySplit`.

**What it does:** Acts as a router — it holds the configuration (mode, sender_tp,
receiver_tp, tp_group, tp_rank) and dispatches to the correct primitive in `forward`.

**Two modes:**

| Mode | Placed at | Calls | When it is a no-op |
|---|---|---|---|
| `"before_send"` | End of sender stage | `BoundaryAllGather` | `sender_tp == 1` or `sender_tp == receiver_tp` |
| `"after_recv"` | Start of receiver stage | `BoundarySplit` | `receiver_tp == 1` or `sender_tp == receiver_tp` |

**Why a module rather than bare functions:** Storing it as `nn.Module` lets it live in
`plan.send_boundary_modules` / `plan.recv_boundary_modules` and be easily moved to GPU
(`.cuda()`) or serialised. It also makes `extra_repr` print cleanly for debugging.

**When it is NOT created:** When `sender_tp == receiver_tp` (Phase 1 behaviour, or
hetero TP that happens to assign the same TP to adjacent stages), no module is inserted.
The `forward` short-circuits with `return x`.

---

## Modified File: `compute_cost.py`

Path: `colossalai/auto_parallel/pipeline_shard/compute_cost.py`

**Change:** One new function `get_boundary_cost_table()` added at the end.

### `get_boundary_cost_table()`

**What it does:** Returns an `(M, M)` float32 array where `table[m1, m2]` is the
estimated time cost of resharding activations from submesh `m1` to submesh `m2` at a
pipeline boundary.

**Formula for each cell:**

```
tp1 = submesh_choices[m1][1]   # cols = TP degree of sender
tp2 = submesh_choices[m2][1]   # cols = TP degree of receiver

if tp1 == tp2:
    cost = 0.0                              # no resharding
elif tp1 > 1:
    bytes_moved = activation_bytes * (1 - 1/tp1)
    cost = alpha + beta * bytes_moved       # AllGather time
else:  # tp1 == 1, receiver Splits locally
    cost = 0.0                              # no communication
```

**Why `(1 - 1/tp1)` bytes:** AllGather sends `tp1 - 1` shards to each rank. Each shard
is `activation_bytes / tp1`. Total bytes moved = `activation_bytes * (tp1-1)/tp1`.

**Why only the sender's TP matters:** The AllGather happens on the sender side (Phase 2).
The split on the receiver side is a local tensor op with zero communication cost.

**How it feeds into the planner:** In `build_pipeline_plan()` (Phase 2 path):

```python
min_incoming = boundary_table.min(axis=0)   # shape (M,)
for m in range(M):
    cost_table[:, :, m, 0] += min_incoming[m]
```

This adds a per-submesh penalty to every cost table entry for submesh `m`. It tells
`alpa_dp`: "if you assign any stage to submesh `m`, expect at least `min_incoming[m]`
of boundary overhead." The DP solver then naturally prefers plans where boundary cost
is low relative to compute savings.

---

## Modified File: `orchestrator.py`

Path: `colossalai/auto_parallel/pipeline_shard/orchestrator.py`

Several additions and changes:

---

### `PipelinePlan` — new fields

**Phase 1 fields (unchanged):** `stage_layer_ranges`, `submesh_per_stage`, `pp_size`,
`tp_size`, `dp_size`, `estimated_cost`.

**New fields in Phase 2:**

| Field | Type | Purpose |
|---|---|---|
| `tp_per_stage` | `List[int]` | TP degree for each stage. In Phase 1 all values are identical. In Phase 2 they may differ. |
| `heterogeneous_tp` | `bool` | `True` only when at least one pair of adjacent stages has different TP degrees. `False` even with `--hetero` if the solver chose uniform TP. |
| `send_boundary_modules` | `dict[int, BoundaryReshardingModule]` | Keyed by stage index `s`. Module applied by stage `s` before P2P send. Empty in Phase 1. |
| `recv_boundary_modules` | `dict[int, BoundaryReshardingModule]` | Keyed by stage index `s+1`. Module applied by stage `s+1` after P2P recv. Empty in Phase 1. |

**Why `tp_per_stage` instead of just `tp_size`:** In Phase 2 each stage needs to know
its own TP degree to reshape the DeviceMesh correctly. `tp_size` (max TP) is still kept
for the `ProcessGroupMesh` shape which must be consistent across all ranks.

---

### `build_pipeline_plan()` — Phase 2 path

**New parameter:** `heterogeneous_tp: bool = False`

**Phase 1 path (unchanged):** Loops over TP candidates one at a time, each candidate
constrains all submeshes to that TP degree, runs `alpa_dp`, keeps the best cost.

**Phase 2 path (new):** Single `alpa_dp` run over **all submeshes simultaneously**.
The solver is free to assign any submesh (any TP degree) to any stage.

Steps in the Phase 2 path:

1. Call `get_compute_cost()` with the full `all_submeshes` list (not filtered by TP).
2. Compute `boundary_table` via `get_boundary_cost_table()`.
3. Add `min_incoming[m]` penalty to each column `m` of the cost table.
4. Run `alpa_dp` once over all submeshes.
5. Parse the solution with `_parse_solution(..., heterogeneous_tp=True)`.

**Why one combined run instead of many:** In Phase 1 the per-TP runs are necessary
because different TP degrees cannot coexist in one `alpa_dp` call — each run's submesh
list is filtered to a single TP. In Phase 2 all submeshes are allowed, so a single DP
finds the globally optimal assignment in one pass.

---

### `_estimate_activation_bytes()` — new helper

**What it does:** Reads the first tensor in `meta_args` (typically `hidden_states`),
multiplies all dimensions together, and multiplies by 2 (float16 element size).

**Why needed:** `get_boundary_cost_table()` needs to know the activation size at the
boundary to estimate AllGather communication time. This helper extracts that size from
the same `meta_args` dict already passed to the planner.

---

### `_parse_solution()` — new helper

**What it does:** Converts the raw `alpa_dp` solution list into a `PipelinePlan`.
Fills `stage_layer_ranges`, `submesh_per_stage`, `tp_per_stage`, `pp_size`, `tp_size`,
`dp_size`, `estimated_cost`, and `heterogeneous_tp`.

**Why extracted:** Before Phase 2 the parsing was inline in the loop. With two paths
(Phase 1 and Phase 2) producing solutions that need identical post-processing, a shared
helper avoids duplication.

**`heterogeneous_tp` detection logic:**

```python
actual_hetero = heterogeneous_tp and any(
    tp_per_stage[i] != tp_per_stage[i + 1]
    for i in range(len(tp_per_stage) - 1)
)
```

Even when `--hetero` is passed, if the solver happens to choose the same TP for every
stage, `plan.heterogeneous_tp` is set to `False`. This means no boundary modules are
created and the training loop runs exactly as in Phase 1.

---

### `autoparallelize_with_pp()` — Phase 2 additions

**New parameter:** `heterogeneous_tp: bool = False`

**Step 5 change — per-stage DeviceMesh shape:**

Phase 1:
```python
stage_ranks_t.reshape(tp_size, dp_size)   # same tp for every stage
```

Phase 2:
```python
tp_s = tp_per_stage[s]
dp_s = devices_per_stage // tp_s
stage_ranks_t.reshape(tp_s, dp_s)          # per-stage tp
```

Each stage's DeviceMesh is shaped to its actual TP degree. The total device count per
stage (`tp_size * dp_size`) stays the same — only the split between TP and DP changes.

**Why all ranks create all stages' meshes:** `DeviceMesh(init_process_group=True)`
calls `dist.new_group()` for the mesh's communication groups. PyTorch requires every
rank to call `new_group()` for every group created anywhere in the job, even if a rank
does not participate in that group. Skipping this would cause hangs or group ID
mismatches.

**Step 7 — new (Phase 2 only):**

```python
if plan.heterogeneous_tp:
    _build_boundary_modules(plan, pp_size, tp_per_stage, all_stage_meshes, current_stage)
```

Only runs when the plan actually has heterogeneous TP. Has no effect in Phase 1.

---

### `_build_boundary_modules()` — new function

**What it does:** Populates `plan.send_boundary_modules` and `plan.recv_boundary_modules`
for the **current rank's stage only**.

**For each boundary between stage `s` and stage `s+1` where `tp_per_stage[s] != tp_per_stage[s+1]`:**

- If `current_stage == s`:
  - Retrieves the TP process group from `all_stage_meshes[s]` (axis 0 = TP axis).
  - Creates `BoundaryReshardingModule("before_send", ...)` and stores it in
    `plan.send_boundary_modules[s]`.

- If `current_stage == s + 1`:
  - Computes `tp_rank` of this rank within stage `s+1`:
    ```
    local_index = global_rank - stage_start
    tp_rank = local_index // dp_recv
    ```
  - Creates `BoundaryReshardingModule("after_recv", ...)` and stores it in
    `plan.recv_boundary_modules[s+1]`.

**Why compute `tp_rank` manually:** `DeviceMesh` gives us `get_process_group(axis)` but
not a simple rank-within-axis query. The rank layout is deterministic from the mesh
shape `(tp_recv, dp_recv)`, so `local_index // dp_recv` gives the row position = TP rank.

---

## Modified File: `run_3d_auto_parallel.py`

Path: `examples/language/gpt/experiments/auto_parallel/run_3d_auto_parallel.py`

**New flag:** `--hetero` — passes `heterogeneous_tp=True` to `autoparallelize_with_pp()`.

**Updated log output:**

```
Plan: pp=2, tp=2, dp=1, hetero=True, estimated_cost=1.2345s
  Stage 0: layers[0:2], tp=1
  Stage 1: layers[2:4], tp=2
```

In Phase 1 `hetero=False` and every stage prints the same `tp`. In Phase 2 stages
may show different `tp` values, making it easy to see which boundary has a TP mismatch.

**Training loop changes (pp > 1 branch):**

```python
send_mod = plan.send_boundary_modules.get(current_stage)   # None in Phase 1
recv_mod = plan.recv_boundary_modules.get(current_stage)   # None in Phase 1
```

Both are `None` in Phase 1 (the dicts are empty), so the training loop degrades to the
exact same control flow as before — no `if` branch is taken.

In Phase 2, when the current stage is a sender at a TP-mismatched boundary:

```python
out = stage_module(x)
if send_mod is not None:
    out = send_mod(out)          # AllGather: [B,S,H/T] → [B,S,H]
p2p.send_forward(out)
saved_output = out
```

When the current stage is a receiver:

```python
recv, _ = p2p.recv_forward()
recv = recv.requires_grad_(True)
act = recv_mod(recv) if recv_mod is not None else recv   # Split: [B,S,H] → [B,S,H/T]
out = stage_module(act)
saved_input = recv   # ← important: save pre-split tensor, not post-split
```

**Why `saved_input = recv` (pre-split):** During backward, `recv.grad` is the gradient
of the loss with respect to the pre-split tensor. `BoundarySplit.backward` fills
`recv.grad` with the zero-padded full-size gradient, which is exactly what the sender
expects from `p2p.send_backward(saved_input.grad)`.

---

## Modified File: `__init__.py`

Path: `colossalai/auto_parallel/pipeline_shard/__init__.py`

**Added exports:**

```python
from .boundary_resharding import BoundaryReshardingModule
from .compute_cost import get_boundary_cost_table, get_compute_cost
```

`BoundaryReshardingModule` and `get_boundary_cost_table` are Phase 2 additions. They are
exported so users can inspect or test them independently without reaching into the
internal module path.

---

## Summary: Data Flow Through Phase 2 at a Boundary

```
Stage s (tp=2)                          Stage s+1 (tp=4)
─────────────────────────────────────────────────────────
rank 0  [B,S,H/2]                       rank 0  [B,S,H/4]
rank 1  [B,S,H/2]                       rank 1  [B,S,H/4]
                                         rank 2  [B,S,H/4]
                                         rank 3  [B,S,H/4]

FORWARD:
rank 0: AllGather → [B,S,H]  ─P2P─►  rank 0: Split[0/4] → [B,S,H/4]
rank 1: AllGather → [B,S,H]  ─P2P─►  rank 1: Split[1/4] → [B,S,H/4]
                              ─P2P─►  rank 2: Split[2/4] → [B,S,H/4]  (from rank 0 or 1)
                              ─P2P─►  rank 3: Split[3/4] → [B,S,H/4]

BACKWARD:
rank 0: chunk[0] ← [B,S,H]  ◄─P2P─   rank 0: zero-pad → [B,S,H]
rank 1: chunk[1] ← [B,S,H]  ◄─P2P─   rank 1: zero-pad → [B,S,H]
                              ◄─P2P─   rank 2: zero-pad → [B,S,H]
                              ◄─P2P─   rank 3: zero-pad → [B,S,H]
```

The AllGather and Split are inverses in the gradient sense:
`AllGather.backward` = chunk, `Split.backward` = zero-pad.
Together they form a symmetric pair that gives correct gradient flow through the
pipeline P2P communication with no extra collective in the backward pass.
