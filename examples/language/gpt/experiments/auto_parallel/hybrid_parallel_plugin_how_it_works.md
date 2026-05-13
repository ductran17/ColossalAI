# HybridParallelPlugin — How It Works

> Internal flow from construction to training, explained step by step.
> Covers: ProcessGroupMesh, ShardFormer (TP), PipelineStageManager (PP), DDP/ZeRO (DP),
> execute_pipeline (1F1B schedule), and optimizer.step.

---

## What the Plugin Does

`HybridParallelPlugin` wires together three orthogonal parallelism strategies into one
coherent system. You give it `(pp_size, tp_size)` and it figures out the rest:

```
world_size = 8,  pp=4,  tp=2  →  dp = 8 / (4×2) = 1

Splits the work three ways:
  TP (tensor parallel):   each layer's weight matrix split across tp=2 ranks  → less memory/GPU
  PP (pipeline parallel): different layers on different GPUs                   → less memory/GPU
  DP (data parallel):     same model, different data, gradient sync            → more throughput
```

You never manually manage NCCL process groups, layer assignment, or microbatch scheduling —
the plugin handles all of it.

---

## Phase 1 — `__init__`: Build the rank topology

```python
plugin = HybridParallelPlugin(
    pp_size          = 4,
    tp_size          = 2,
    num_microbatches = 4,
    precision        = "fp32",
    dp_outside       = True,    # default
)
```

### 1a. Infer dp_size

```python
dp_size = world_size // (tp_size * pp_size)
# = 8 // (2 * 4) = 1
```

### 1b. Build ProcessGroupMesh

`dp_outside=True` (default):

```python
pg_mesh = ProcessGroupMesh(dp_size=1, pp_size=4, tp_size=2, sp_size=1)
# axes:  dp=0, pp=1, tp=2, sp=3

# C-order indexing (last axis fastest):
# rank = dp_rank*(pp*tp) + pp_rank*tp + tp_rank
```

This creates a 4-dimensional logical grid of ranks. For our pp=4, tp=2, dp=1 case:

```
rank | dp | pp | tp | node
-----|----|----|----|-----------
  0  |  0 |  0 |  0 | node18
  1  |  0 |  0 |  1 | node18   ← TP pair
  2  |  0 |  1 |  0 | node20
  3  |  0 |  1 |  1 | node20   ← TP pair
  4  |  0 |  2 |  0 | node20
  5  |  0 |  2 |  1 | node20   ← TP pair
  6  |  0 |  3 |  0 | node16
  7  |  0 |  3 |  1 | node16   ← TP pair
```

`dp_outside=False` would swap dp and pp axes:
```python
pg_mesh = ProcessGroupMesh(pp_size, dp_size, tp_size, sp_size)
# rank = pp_rank*(dp*tp) + dp_rank*tp + tp_rank
```

### 1c. Extract process groups from the mesh

```python
tp_group = pg_mesh.get_group_along_axis(tp_axis)   # axis 2
dp_group = pg_mesh.get_group_along_axis(dp_axis)   # axis 0
pp_group = pg_mesh.get_group_along_axis(pp_axis)   # axis 1
```

`get_group_along_axis(2)` collects all rank sets that share the same dp and pp coordinates
but differ in tp:

```
tp_group membership (pp=4, tp=2, dp=1):
  group {0,1}   — pp_rank=0, dp_rank=0  (node18)
  group {2,3}   — pp_rank=1, dp_rank=0  (node20)
  group {4,5}   — pp_rank=2, dp_rank=0  (node20)
  group {6,7}   — pp_rank=3, dp_rank=0  (node16)

dp_group membership:
  dp=1 → each rank is its own dp group (no gradient sync needed)

pp_group membership:
  All ranks with same dp_rank and tp_rank form a pipeline:
  {0,2,4,6} — tp_rank=0
  {1,3,5,7} — tp_rank=1
```

### 1d. PipelineStageManager

```python
stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=pp_axis)
```

From the mesh coordinate of this rank, `PipelineStageManager` determines:
- `stage_manager.stage` = this rank's pipeline stage index (0–3)
- `stage_manager.prev_rank` = global rank of the previous PP stage (for recv)
- `stage_manager.next_rank` = global rank of the next PP stage (for send)

For rank 2 (pp_rank=1, tp_rank=0): `prev_rank=0`, `next_rank=4`.

### 1e. Scheduler

```python
scheduler = OneForwardOneBackwardSchedule(
    stage_manager    = stage_manager,
    num_microbatches = 4,
)
```

The scheduler holds the 1F1B execution plan. It doesn't run anything yet —
it just records how many microbatches and which stage this rank is.

---

## Phase 2 — `booster.boost(model, optimizer)` → `configure()`

This is where the model is transformed and the optimizer is wrapped.

### 2a. ShardFormer: apply TP sharding

```python
shardformer = ShardFormer(shard_config)
module, shared_params = shardformer.optimize(module, policy=None)
```

ShardFormer inspects the model type (GPT2LMHeadModel) and applies the corresponding
built-in policy (`GPT2LMHeadModelPolicy`). For each transformer block it:

**Column-parallel (splits along output dimension):**
```
Original Q weight: [H, H]       → this rank holds: [H, H/tp]
Original K weight: [H, H]       → this rank holds: [H, H/tp]
Original V weight: [H, H]       → this rank holds: [H, H/tp]
Original MLP fc1:  [H, 4H]      → this rank holds: [H, 4H/tp]
```

**Row-parallel (splits along input dimension):**
```
Original attention out: [H, H]  → this rank holds: [H/tp, H]
Original MLP fc2:       [4H, H] → this rank holds: [4H/tp, H]
```

After a row-parallel layer, the partial outputs from all tp ranks must be summed.
ShardFormer inserts an **AllReduce hook** on the output of each row-parallel layer
that fires automatically during forward pass:

```python
# pseudo-code of what ShardFormer inserts:
def forward_hook(output):
    dist.all_reduce(output, group=tp_group)   # sum partial results
    return output
```

**Pipeline layer assignment:**

ShardFormer also slices the model so each rank only holds its pipeline stage's layers.
For pp=4, layers=8:

```
stage 0 (ranks 0,1): GPT2 layers 0, 1
stage 1 (ranks 2,3): GPT2 layers 2, 3
stage 2 (ranks 4,5): GPT2 layers 4, 5
stage 3 (ranks 6,7): GPT2 layers 6, 7  + lm_head (last stage only)
```

Ranks holding non-last stages have no `lm_head`. Ranks holding non-first stages
have no `token embedding`.

**Result after ShardFormer:**
```
Each GPU holds:
  - 2 of 8 transformer layers     (pp=4 → 2 layers/stage)
  - 50% of each layer's weights   (tp=2 → each GPU has half the matrix)
  Total: 1/8 of the full model
```

### 2b. Move to GPU and set precision

```python
if mixed_precision is not None:
    module = module.to(mixed_precision)   # fp16 or bf16
module = module.to(get_accelerator().get_current_device())
```

With `precision="fp32"`, no cast happens — weights stay fp32 on GPU.

### 2c. Decide DDP vs ZeRO vs nothing for DP

```python
use_ddp = (dp_size > 1 and pp_size == 1 and zero_stage == 0) or (dp_size == 1 and pp_size == 1)
```

For our case (pp=4, dp=1):
- `dp_size > 1` is False → `use_ddp = False`
- No DDP wrapper, no gradient sync needed — dp=1 means only one replica

If `dp > 1` and `pp > 1` (e.g. pp=2, tp=2, dp=2):
- `use_ddp = False` (DDP conflicts with PP microbatch accumulation)
- Gradient sync happens manually in `execute_pipeline()` after the full step

If `dp > 1` and `pp = 1`:
- `use_ddp = True` → standard PyTorch DDP wrapper applied

### 2d. Wrap the optimizer

```python
optimizer = HybridParallelNaiveOptimizer(
    optimizer,
    model,
    use_pipeline = True,   # pp > 1
    pp_process_group = pp_group,
    tp_process_group = tp_group,
)
```

`HybridParallelNaiveOptimizer` wraps the base Adam optimizer. During `.step()` it:
- Clips gradients and normalises grad norm by `tp_size` (because TP splits weights)
- Syncs shared parameters (e.g. embedding ↔ lm_head which appear on different PP stages)
- Calls the underlying `optimizer.step()`

---

## Phase 3 — `booster.execute_pipeline(...)` — The Training Step

```python
outputs = booster.execute_pipeline(
    iter([batch]),          # full batch — NOT pre-divided
    model,
    criterion = criterion,
    optimizer = optimizer,
    return_loss = True,
)
```

### 3a. Disable automatic gradient sync

```python
with model.no_sync(), model._hook_context():
    outputs = scheduler.forward_backward_step(...)
```

`model.no_sync()` disables automatic DDP gradient AllReduce during the microbatch loop.
Gradients will accumulate across all microbatches and sync once at the end.

### 3b. 1F1B schedule — `OneForwardOneBackwardSchedule.forward_backward_step()`

The scheduler drives the 1F1B execution for this rank. It uses `stage_manager` to know
what stage it is, and sends/receives activation tensors using the `pp_group`.

**Conceptual timeline for pp=4, M=4 microbatches:**

```
Time →  t0   t1   t2   t3   t4   t5   t6   t7   t8   t9  t10  t11  t12  t13
Stage 0:  F0   F1   F2   F3   -    -    -   B3   B2   B1   B0
Stage 1:       F0   F1   F2   F3   -    -    -   B3   B2   B1   B0
Stage 2:            F0   F1   F2   F3   -    -    -   B3   B2   B1   B0
Stage 3:                 F0   F1   F2   F3   B3   B2   B1   B0
                                             ↑ loss computed on stage 3
```

- `F0` = forward pass on microbatch 0
- `B0` = backward pass on microbatch 0
- `-` = idle (bubble)

**What happens on rank 2 (stage 1, tp_rank=0) during F0:**

```
1. recv activation from rank 0 (stage 0, same tp_rank=0) via pp_group NCCL P2P
2. forward through layers 2–3:
     layernorm → attention Q/K/V projection (column-parallel, local matmul)
     → QKV compute → attention output projection (row-parallel, local matmul)
     → AllReduce on tp_group {2,3}  ← TP sync fires HERE inside forward
     → MLP fc1 (column-parallel) → relu → fc2 (row-parallel)
     → AllReduce on tp_group {2,3}  ← TP sync fires HERE
3. send activation to rank 4 (stage 2, same tp_rank=0) via pp_group NCCL P2P
```

**Communication pattern:**

```
TP AllReduce:  {0,1}, {2,3}, {4,5}, {6,7}    intra-node PCIe  (fast)
PP P2P send:   0→2, 2→4, 4→6                 cross-node NCCL  (Ethernet)
               1→3, 3→5, 5→7                 cross-node NCCL  (Ethernet)
DP AllReduce:  dp=1 → none
```

**Loss computation (stage 3 only):**

Only ranks 6 and 7 (last PP stage) run `criterion(outputs, inputs)` and get a real loss value.
All other stages return `loss=None` from `execute_pipeline`.

### 3c. Gradient sync after the full step

```python
model.sync_shared_params()   # sync embedding ↔ lm_head gradients via pp_group
model.sync_sp_grads()        # no-op (sp=1)
model.sync_dp_grads()        # no-op (dp=1); if dp>1: AllReduce grads on dp_group
```

For dp=1, `sync_dp_grads` returns immediately (group size is 1).

If dp>1 and pp>1, `sync_dp_grads` does:
```python
for p in model.parameters():
    if p.grad is not None:
        dist.all_reduce(p.grad, group=dp_group)
        p.grad.div_(dp_group.size())
```

---

## Phase 4 — `optimizer.step()` and `optimizer.zero_grad()`

```python
optimizer.step()
optimizer.zero_grad()
```

`HybridParallelNaiveOptimizer.step()`:

```
1. Clip gradient norm
   - collect grad norms from all parameters on this stage
   - if tp > 1: divide by tp_size (TP splits weights, gradients are partial)
   - if pp > 1: all-reduce grad norms across pp_group to get global norm

2. Sync shared parameters (e.g. token embedding on stage 0 ↔ lm_head on stage 3)
   - for each shared param: dist.all_reduce(param.grad, group=cross_stage_group)

3. base_optimizer.step()
   - Adam update: m, v, param in-place update on each GPU's local shard

4. zero_grad(): clear accumulated gradients from all microbatches
```

---

## End-to-End Data Flow Diagram

```
User code:
  booster.execute_pipeline(iter([batch]), model, criterion, optimizer)

    │  batch = {input_ids: [4, 64], attention_mask: [4, 64], labels: [4, 64]}
    │  (4 sequences, full global batch)
    │
    ▼
  scheduler.forward_backward_step()
    │
    │  splits batch into 4 microbatches of size 1
    │  [mb0, mb1, mb2, mb3] each shape [1, 64]
    │
    ├── Microbatch 0, Stage 0 (ranks 0,1):
    │     input_ids [1,64] → token_embedding → hidden [1,64,256]
    │     → layer 0 forward:
    │         Q = x @ W_Q_local     (local matmul, tp_rank gets cols 0:128)
    │         K = x @ W_K_local
    │         V = x @ W_V_local
    │         attn = softmax(QKᵀ/√d) @ V
    │         out = attn @ W_out_local   (row-parallel)
    │         AllReduce(out, tp_group={0,1})  ← sums partial results
    │     → layer 1 forward (same pattern)
    │     → send hidden [1,64,256] to rank 2 (P2P over Ethernet)
    │
    ├── Microbatch 0, Stage 1 (ranks 2,3):
    │     recv hidden [1,64,256] from rank 0
    │     → layer 2 forward → layer 3 forward (same TP pattern)
    │     → send to rank 4
    │
    ├── Microbatch 0, Stage 2 (ranks 4,5):
    │     recv → layer 4 → layer 5 → send to rank 6
    │
    ├── Microbatch 0, Stage 3 (ranks 6,7):
    │     recv → layer 6 → layer 7 → lm_head
    │     → logits [1,64,1024]
    │     → loss = criterion(logits, labels)   ← only here
    │     → loss.backward() starts
    │     → grad_hidden sent back to rank 4 (P2P)
    │
    └── ... backward propagates back through all stages ...
        → gradients accumulate on each rank's local parameter shards

  model.sync_dp_grads()    ← AllReduce gradients if dp > 1

  optimizer.step()         ← Adam update on local shards
  optimizer.zero_grad()
```

---

## Key Properties

### TP: what is AllReduced and when

Each row-parallel layer (attention output proj, MLP fc2) produces a **partial result** on each
tp rank. The full output = sum of all tp ranks' partial outputs. ShardFormer inserts an
AllReduce hook that fires automatically inside the layer's forward pass. The tensor size
being AllReduced = `(microbatch, seq, hidden)`.

### PP: what is sent between stages

The **activation tensor** (output of the last layer on a stage) is sent to the next stage.
Shape: `(microbatch, seq, hidden)`. For our config: `1 × 64 × 256 × 4B = 64 KB` per P2P send.

### DP: when gradients are synced

If `dp > 1`:
- When `pp = 1`: DDP wrapper handles it automatically during backward
- When `pp > 1`: DDP is disabled; manual `dist.all_reduce` happens in `sync_dp_grads()`
  after all microbatches of the 1F1B step are complete

If `dp = 1`: no gradient sync at all.

### Loss availability

```
outputs = booster.execute_pipeline(...)

outputs["loss"] is not None  ← only True for ranks on the LAST PP stage
outputs["loss"] is None      ← all other stages
```

---

## Integration with Our Auto-Planner

The planner (`auto_plan`) and the plugin must agree on the same `dp_outside` value:

| Parameter | auto_plan | HybridParallelPlugin | Must match |
|---|---|---|---|
| `dp_outside` | `auto_plan(..., dp_outside=True)` | `HybridParallelPlugin(..., dp_outside=True)` | ✓ |
| Rank formula | `rank = dp*(pp*tp) + pp*tp + tp` | `ProcessGroupMesh(dp,pp,tp)` C-order | same |
| `dp_rank` | `rank // (pp*tp)` | mesh axis 0 coordinate | same |

The planner uses `dp_outside` to classify whether TP/PP/DP comm groups are intra-node or
cross-node. If `dp_outside` disagrees, the planner's comm cost estimates are wrong and it
may pick the wrong plan.

---

## Summary of What Each Component Owns

```
HybridParallelPlugin.__init__()
  └── ProcessGroupMesh         builds tp/pp/dp process groups from rank layout

booster.boost() → configure()
  └── ShardFormer              slices weight matrices (TP) + assigns layers (PP)
  └── HybridParallelModule     model wrapper: holds tp/dp/pp groups, hooks
  └── HybridParallelNaiveOptimizer  wraps Adam: handles grad norm, shared params

booster.execute_pipeline()
  └── OneForwardOneBackwardSchedule  drives 1F1B: sends/recvs activations (PP)
  └── AllReduce hooks (inside each layer)  sum partial activations (TP)
  └── model.sync_dp_grads()    AllReduce gradients after full step (DP, if dp>1)

optimizer.step()
  └── clip + normalize grad norm across PP stages
  └── sync shared embedding/lm_head params
  └── Adam update on local parameter shard
```
