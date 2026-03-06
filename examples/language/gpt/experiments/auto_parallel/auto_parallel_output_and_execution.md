# Auto-Parallel: What the ModuleWrapper Looks Like and How Training Executes

## The Transformation Pipeline

Before reaching `ModuleWrapper`, the raw `ColoGraphModule` passes through four sub-passes inside `runtime_preparation_pass` and `runtime_apply_pass`. Understanding each pass is the key to understanding what the final object contains and how it runs.

```
solution: List[int]   +   ColoGraphModule (with strategies_vector on every node)
              |
              | runtime_preparation_pass()
              |   1. solution_annotation_pass     -- stick best strategy to each node
              |   2. size_value_converting_pass   -- fix .size() calls for sharded dims
              |   3. node_args_converting_pass    -- fix shape args for view/reshape
              |   4. module_params_sharding_pass  -- physically slice weights, add grad hooks
              |
              | runtime_apply_pass()
              |   5. _shape_consistency_apply     -- insert resharding nodes between mismatched neighbors
              |   6. _comm_spec_apply             -- insert collective comm nodes (all-reduce, all-gather, etc.)
              |
              | shape_prop_pass()                 -- re-propagate shapes on the now-modified graph
              | gm.recompile()                    -- regenerate Python source for the graph
              |
              v
         ColoGraphModule (sharded weights, comm ops in graph, recompiled)
              |
              | ModuleWrapper(gm, sharding_spec_dict, origin_spec_dict, comm_actions_dict)
              v
         ModuleWrapper  ← the final output of autoparallelize()
```

---

## Pass 1: solution_annotation_pass

**What it does:** Reads the solver's `solution` (list of strategy indices) and stamps each graph node with its chosen strategy.

For every node `i`, it sets:
- `node.best_strategy` → the chosen `ShardingStrategy` object
- `node.sharding_spec` → the output `ShardingSpec` of this node under that strategy
- `node.target_sharding_specs` → list of `ShardingSpec` objects that each successor node expects as input (may differ from `node.sharding_spec` if resharding is needed)

It also builds three dictionaries that will be passed at runtime:

```python
sharding_spec_convert_dict  = { node_index: [spec_for_user_0, spec_for_user_1, ...] }
origin_node_sharding_spec_dict = { node_index: node.sharding_spec }
comm_actions_dict = { node_index: { op_data_name: CommAction } }
```

These three dicts are then **inserted into the FX graph itself as new placeholder nodes** (`sharding_spec_convert_dict`, `origin_node_sharding_spec_dict`, `comm_actions_dict`). This makes them available as named arguments during `gm.forward()` at runtime.

---

## Pass 2: size_value_converting_pass

**Problem:** If a tensor of shape `[B, S, H]` is sharded as `S0` on dim 0, each GPU holds shape `[B/2, S, H]`. Any call to `tensor.size(0)` in the graph would incorrectly return `B/2` when the downstream op (e.g., `.view()`) expects the original full size `B`.

**Fix:** For every `.size()` call node in the graph, insert a `size_processing` node immediately after it. At runtime, `size_processing` multiplies the local size back by the shard factor derived from `dim_partition_dict` and `device_mesh.shape`.

```python
# Before:
x = hidden.size(0)           # returns B/2 (wrong for downstream view)
y = hidden.view(x, -1)

# After:
x = hidden.size(0)           # still returns B/2
x = size_processing(x, dim_partition_dict, device_mesh_info, target_dim=0)  # returns B
y = hidden.view(x, -1)
```

---

## Pass 3: node_args_converting_pass

**Problem:** Similar issue for ops whose arguments are literal shape integers (`.view()`, `.reshape()`, `torch.reshape`). If the output is sharded, the shape argument must be divided by the shard factor.

**Fix:** For all nodes whose target is in `SHAPE_ARGUMENT_OPS`, divide the integer argument on each sharded dimension by the mesh size on that axis.

```python
# If output of view is sharded [B, S, H] -> sharded on dim 0 by 2:
# Before: hidden.view(B, S * H)
# After:  hidden.view(B // 2, S * H)   <- adjusted to local shard size
```

---

## Pass 4: module_params_sharding_pass

**What it does:** Physically slices every weight and buffer in the model according to the chosen strategy. This is the pass that actually changes the tensor data.

### Weight slicing

For each `call_module` node, for each named parameter in the submodule:
1. Looks up `target_sharding_spec` from `node.best_strategy.get_sharding_spec_by_name(param_name)`.
2. Calls `shape_consistency_manager.apply_for_autoparallel_runtime(param.data, origin_spec, target_spec)`.
3. Replaces the parameter in-place with the sliced shard.

Example for a column-parallel Linear with weight `[H_out, H_in]` on 2 GPUs:
- Rank 0 gets weight `[H_out/2, H_in]`
- Rank 1 gets weight `[H_out/2, H_in]`

### Gradient hooks

For parameters that need gradient synchronization (e.g., the output projection in row-parallel attention requires an all-reduce of gradients across the TP group), a `register_hook` is attached:

```python
def hook_fn(grad):
    _all_reduce(grad, comm_spec, async_op=overlap)

param.register_hook(hook_fn)
```

`CommType.HOOK` marks that the communication happens on the gradient, not the activation.

Optional `overlap=True` runs the all-reduce on a dedicated CUDA stream to overlap it with subsequent backward operations.

---

## Pass 5: _shape_consistency_apply (inside runtime_apply_pass)

**What it does:** Detects adjacent node pairs where the output sharding of node A does not match the expected input sharding of node B, and inserts a `runtime_apply` call node between them.

```python
# If node A outputs ShardingSpec([S0, R]) but node B expects ShardingSpec([R, R]):
# → need an all-gather on dim 0 between A and B

# Before graph:
#   A → B

# After graph:
#   A → runtime_apply(A, origin_dict, input_dict, idx_A, user_idx) → B
```

`runtime_apply` (executed at runtime) calls `shape_consistency_manager.apply_for_autoparallel_runtime(tensor, origin_spec, target_spec)` which executes the appropriate collective (all-gather, all-reduce, reduce-scatter, etc.).

For iterable outputs (tuples/lists of tensors), `runtime_apply_for_iterable_object` is inserted instead.

---

## Pass 6: _comm_spec_apply (inside runtime_apply_pass)

**What it does:** Inserts explicit communication nodes for operations that need a collective before or after the computation (not on the gradient, not due to spec mismatch, but inherent to the op strategy).

Two timing modes:

- **`CommType.BEFORE`**: Insert a `runtime_comm_spec_apply` node **before** the compute node, replacing the input argument.
  - Example: for a row-parallel linear, the input must be all-gathered before the matmul.

- **`CommType.AFTER`**: Insert a `runtime_comm_spec_apply` node **after** the compute node, replacing the output.
  - Example: for a column-parallel linear with partial sum output, an all-reduce is inserted after the matmul.

```python
# BEFORE example (all-gather input):
# Before:  matmul(x_sharded, weight)
# After:   x_full = runtime_comm_spec_apply(x_sharded, comm_dict, idx, "input")
#           matmul(x_full, weight)

# AFTER example (all-reduce output):
# Before:  out = matmul(x, weight_col_parallel)
# After:   out = matmul(x, weight_col_parallel)
#           out = runtime_comm_spec_apply(out, comm_dict, idx, "output")
```

`runtime_comm_spec_apply` at runtime calls `comm_spec.covert_spec_to_action(tensor)` which dispatches to the correct collective primitive from `colossalai.tensor.comm_spec`.

---

## What the Final ColoGraphModule Looks Like

After all passes and `gm.recompile()`, the graph module's forward function is regenerated as Python code. Conceptually it looks like this for a GPT-2 layer (simplified):

```python
# Generated forward code (pseudocode — actual output is a compiled Python function)
def forward(self,
            input_ids,
            attention_mask,
            sharding_spec_convert_dict,       # <-- injected by solution_annotation_pass
            origin_node_sharding_spec_dict,   # <-- injected
            comm_actions_dict):               # <-- injected

    # --- Embedding (replicated on all ranks) ---
    inputs_embeds  = self.wte(input_ids)       # full embedding on every rank
    position_embeds = self.wpe(position_ids)
    hidden_states  = inputs_embeds + position_embeds

    # --- Transformer block (column-parallel attention) ---
    hidden_states_ln = self.h_0_ln_1(hidden_states)

    # CommType.BEFORE: all-gather hidden if needed
    hidden_states_ln = runtime_comm_spec_apply(
        hidden_states_ln, comm_actions_dict, node_idx, "input")

    # Column-parallel QKV projection: weight is [3H, H/N] on this rank
    qkv = self.h_0_attn_c_attn(hidden_states_ln)   # output: [B, S, 3H/N]

    # CommType.AFTER: all-reduce partial sums from QKV proj if row-parallel
    # ... or shape_consistency node if spec mismatch with next op ...
    qkv = runtime_apply(qkv, origin_dict, input_dict, 5, 0)  # resharding

    # Attention computation (head-parallel: each rank handles H/N heads)
    query, key, value = split(qkv, ...)
    attn_output = scaled_dot_product(query, key, value)

    # Row-parallel output proj: reduces partial sums via all-reduce (HOOK on grad)
    attn_output = self.h_0_attn_c_proj(attn_output)   # output: [B, S, H] full

    # residual + LN + MLP (same pattern) ...

    # --- LM head (vocab-parallel) ---
    lm_logits = self.lm_head(hidden_states)    # weight: [H, V/N] on this rank
    lm_logits = runtime_comm_spec_apply(lm_logits, comm_actions_dict, last_idx, "output")
    # all-gather logits across TP group → [B, S, V]

    return lm_logits
```

---

## The ModuleWrapper

```python
class ModuleWrapper(nn.Module):
    def __init__(self, module, sharding_spec_dict, origin_spec_dict, comm_actions_dict):
        self.module = module                           # ColoGraphModule (recompiled)
        self.sharding_spec_dict = sharding_spec_dict  # target specs per node per user
        self.origin_spec_dict   = origin_spec_dict    # output spec per node
        self.comm_actions_dict  = comm_actions_dict   # comm actions per node

    def forward(self, *args, **kwargs):
        return self.module(
            *args,
            sharding_spec_convert_dict=self.sharding_spec_dict,
            origin_node_sharding_spec_dict=self.origin_spec_dict,
            comm_actions_dict=self.comm_actions_dict,
            **kwargs,
        )
```

Its only job is to **inject the three runtime dictionaries** into every forward call so the `runtime_apply` and `runtime_comm_spec_apply` nodes inside the graph can look up what collectives to execute for each node at this point in the graph.

The actual model weights live inside `self.module` (the `ColoGraphModule`), already sliced per-rank.

---

## How Distributed Training Executes

### Step-by-step for one training iteration

```
All ranks call gm(input_ids, attention_mask) simultaneously
                         |
          ModuleWrapper.forward()
                         |
          ColoGraphModule.forward(
              input_ids, attention_mask,
              sharding_spec_convert_dict,
              origin_node_sharding_spec_dict,
              comm_actions_dict
          )
                         |
    Execute graph nodes one-by-one:
    ┌──────────────────────────────────────────────────────────────┐
    │ node: wte(input_ids)                                         │
    │   → each rank computes same embedding (replicated weight)    │
    │                                                              │
    │ node: runtime_comm_spec_apply(hidden, ...)   [CommType.BEFORE]│
    │   → collective op (e.g., split/all-gather) on NCCL           │
    │                                                              │
    │ node: c_attn(hidden)    ← weight is [3H, H/N] on this rank  │
    │   → local matmul only                                        │
    │                                                              │
    │ node: runtime_apply(qkv, origin_dict, input_dict, ...)       │
    │   → ShapeConsistencyManager checks specs                     │
    │   → if mismatch: fires collective (all-gather / reduce-scatter)│
    │                                                              │
    │ node: c_proj(attn_out)  ← weight is [H/N, H] on this rank  │
    │   → local matmul, produces partial sum                       │
    │                                                              │
    │ node: runtime_comm_spec_apply(out, ...)   [CommType.AFTER]   │
    │   → all-reduce across TP group → full output tensor          │
    │                                                              │
    │ ... repeat for each transformer block ...                    │
    │                                                              │
    │ node: lm_head(hidden)   ← vocab-parallel weight [H, V/N]    │
    │   → local matmul                                             │
    │                                                              │
    │ node: runtime_comm_spec_apply(logits, ...) [CommType.AFTER]  │
    │   → all-gather logits → [B, S, V] full vocabulary            │
    └──────────────────────────────────────────────────────────────┘
                         |
                    lm_logits  (same on all ranks if replicated output)
                         |
              loss = criterion(lm_logits, labels)
                         |
              loss.backward()
                         |
    ┌──────────────────────────────────────────────────────────────┐
    │ Autograd runs backward through same graph in reverse         │
    │                                                              │
    │ For TP weights (e.g., c_proj row-parallel):                  │
    │   grad arrives at parameter                                  │
    │   registered hook fires:                                     │
    │     _all_reduce(grad, comm_spec)  ← sync grad across TP group│
    │                                                              │
    │ For DP weights (replicated across data-parallel ranks):      │
    │   registered hook fires:                                     │
    │     _all_reduce(grad, comm_spec)  ← sync grad across DP group│
    └──────────────────────────────────────────────────────────────┘
                         |
              optimizer.step()   ← each rank updates its local weight shard
```

### Key runtime functions

| Function | Location | Called by | Purpose |
|---|---|---|---|
| `runtime_apply` | `runtime_apply_pass.py` | Injected graph node | Reshape/reshard activation between mismatched adjacent node specs |
| `runtime_apply_for_iterable_object` | `runtime_apply_pass.py` | Injected graph node | Same as above for tuple/list outputs |
| `runtime_comm_spec_apply` | `runtime_apply_pass.py` | Injected graph node | Execute a specific collective (all-reduce, all-gather, reduce-scatter) |
| `size_processing` | `runtime_preparation_pass.py` | Injected graph node | Fix `.size()` return values to reflect unsharded global size |
| `hook_fn` (closure) | `runtime_preparation_pass.py` | `param.register_hook` | All-reduce gradients for TP/DP parameters on backward |
| `ShapeConsistencyManager.apply_for_autoparallel_runtime` | `colossalai/tensor/shape_consistency.py` | Above functions | Compute + execute the sequence of collectives to go from one ShardingSpec to another |

---

## What Each Rank Owns After autoparallelize()

For GPT-2 with 4 GPUs in a `[2, 2]` mesh (2 DP × 2 TP):

```
GPU 0 (DP=0, TP=0)          GPU 1 (DP=0, TP=1)
  wte:  full [50257, H]       wte:  full [50257, H]
  c_attn weight: [3H, H/2]   c_attn weight: [3H, H/2]  ← different columns
  c_proj weight: [H/2, H]    c_proj weight: [H/2, H]
  lm_head: [H, V/2]          lm_head: [H, V/2]          ← different vocab slice

GPU 2 (DP=1, TP=0)          GPU 3 (DP=1, TP=1)
  (same weights as GPU 0)    (same weights as GPU 1)     ← DP replica
```

At runtime:
- **TP groups** `{GPU0, GPU1}` and `{GPU2, GPU3}` exchange activations via all-reduce / all-gather for tensor-parallel layers.
- **DP groups** `{GPU0, GPU2}` and `{GPU1, GPU3}` synchronize gradients via all-reduce during backward.

The `ModuleWrapper` and the injected runtime nodes handle all of this transparently — the training loop calls `gm(input_ids, attention_mask)` exactly as it would a normal `nn.Module`.