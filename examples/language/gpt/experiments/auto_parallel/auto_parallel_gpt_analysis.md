# Auto Parallel with GPT2 — Flow, Input, Output, and Mechanism

## Overview

`auto_parallel_with_gpt.py` demonstrates ColossalAI's **tensor sharding auto-parallelism** applied to a custom GPT-2 language model. The system automatically searches for and applies an optimal tensor parallelism strategy across multiple GPUs without requiring the user to manually annotate how tensors should be distributed.

---

## Configuration Constants

| Constant      | Value  | Description                         |
|---------------|--------|-------------------------------------|
| `BATCH_SIZE`  | 16     | Number of samples per step          |
| `SEQ_LENGTH`  | 1024   | Token sequence length               |
| `HIDDEN_DIM`  | 4096   | Embedding / hidden dimension        |
| `NUM_HEADS`   | 16     | Number of attention heads           |
| `NUM_LAYERS`  | 4      | Number of transformer blocks        |
| `VOCAB_SIZE`  | 50257  | GPT-2 vocabulary size               |
| `NUM_STEPS`   | 10     | Number of training iterations       |
| `FP16`        | True   | Use half-precision (float16)        |

---

## Model Architecture (`gpt_modules.py`)

A custom GPT-2 stack is defined instead of the HuggingFace default. The reason is that the FX tracer used by ColossalAI requires a static, branch-free computation graph.

### Components

- **`GPT2MLP`**: Two-layer feed-forward block (`Conv1D` -> activation -> `Conv1D`).
- **`GPT2Attention`**: Multi-head self-attention using a combined QKV projection (`c_attn: Conv1D`) followed by output projection (`c_proj: Conv1D`). The split/view order is rearranged to match Megatron-LM conventions for better sharding compatibility.
- **`GPT2Block`**: Pre-LayerNorm transformer block (`LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual`).
- **`GPT2Model`**: Full transformer with token embedding (`wte`) + positional embedding (`wpe`), a stack of `GPT2Block` layers, and a final `LayerNorm`.
- **`GPT2LMHeadModel`**: Wraps `GPT2Model` and adds a language modelling head (`Linear: hidden_dim -> vocab_size`).
- **`GPTLMLoss`**: Cross-entropy loss with causal shift (predict token `t+1` from token `t`).

---

## End-to-End Training Flow

```
main()
  |
  +-- launch_from_torch()               # Initialize distributed process group
  |
  +-- Build GPT2LMHeadModel             # Instantiate model in FP16 on CUDA
  |
  +-- Construct meta_input_sample       # Shape-only tensors on "meta" device (no real data)
  |     input_ids:      [16, 1024] int64
  |     attention_mask: [16, 1024] int64
  |
  +-- autoparallelize(model, meta_args) # CORE: auto sharding search + model transformation
  |     |
  |     +-- initialize_device_mesh()
  |     |     - Profile inter-GPU alpha/beta (latency/bandwidth)
  |     |     - Search best logical mesh topology (e.g., 2x2 for 4 GPUs)
  |     |
  |     +-- initialize_model()
  |           |
  |           +-- ColoTracer.trace()          # Trace model into an FX computation graph
  |           +-- shape_prop_pass()           # Propagate tensor shapes through graph
  |           +-- build_strategy_constructor()
  |           |     - Walk each graph node
  |           |     - Use node-type handlers to enumerate valid sharding strategies
  |           |     - Compute compute + communication cost for each strategy
  |           |
  |           +-- solve_solution()            # ILP solver (PuLP + CBC)
  |           |     - Build cost graph (node costs + edge resharding costs)
  |           |     - Simplify edge costs
  |           |     - Minimize total cost subject to optional memory budget
  |           |
  |           +-- transform_to_sharded_model()
  |                 - runtime_preparation_pass(): shard model weights, register grad hooks
  |                 - runtime_apply_pass(): insert communication ops (all-reduce, all-gather, etc.)
  |                 - shape_prop_pass(): re-propagate shapes on sharded graph
  |                 - gm.recompile(): generate final Python code for sharded forward pass
  |
  +-- Print solution per node (rank 0 only)
  |
  +-- Training loop (10 steps)
        for each step:
          - generate random input_ids + attention_mask on GPU
          - forward pass through sharded model (gm)
          - compute GPTLMLoss
          - backward + Adam optimizer step
          - log loss, step time, TFLOPS
```

---

## Inputs

### `autoparallelize()` inputs

| Argument          | Value in script                          | Description                               |
|-------------------|------------------------------------------|-------------------------------------------|
| `model`           | `GPT2LMHeadModel` (FP16, on CUDA)        | The model to be auto-sharded              |
| `meta_args`       | `{input_ids: meta[16,1024], mask: meta[16,1024]}` | Shape descriptors on the "meta" device — no actual data, used only for tracing |
| `return_solution` | `True`                                   | Also return the chosen strategy per node  |

### Training loop inputs (per step)

| Tensor           | Shape          | Device | Description                    |
|------------------|----------------|--------|--------------------------------|
| `input_ids`      | `[16, 1024]`   | CUDA   | Random token IDs (0–50256)     |
| `attention_mask` | `[16, 1024]`   | CUDA   | All-ones (no padding)          |

---

## Outputs

### `autoparallelize()` outputs

| Return value | Type              | Description                                                 |
|--------------|-------------------|-------------------------------------------------------------|
| `gm`         | `ModuleWrapper`   | The transformed, sharded model ready for distributed training |
| `solution`   | `List[str]`       | Per-node strategy names, e.g. `"linear_1 S0S1 = S0R x RS1"` |

The `solution` list is printed on rank 0 and contains one entry per traced graph node, describing:
- The node name
- The chosen sharding strategy (notation: `S` = sharded on that axis, `R` = replicated)

### Training loop outputs (logged per step, rank 0 only)

```
[1/10] Loss: 10.823, Step time: 3.412s, TFLOPS: 0.214
[2/10] Loss: 10.761, Step time: 1.023s, TFLOPS: 0.715
...
```

| Field       | Description                                                             |
|-------------|-------------------------------------------------------------------------|
| `Loss`      | Cross-entropy LM loss (next-token prediction), expected ~10.8 at start  |
| `Step time` | Wall clock time for forward + backward + optimizer step (seconds)       |
| `TFLOPS`    | Throughput: `numel * batch * seq * 8 / 1e12 / step_time / world_size`  |

Memory usage is also logged after model initialization:
```
GPU memory usage: XXX.XX MB, CPU memory usage: XXX.XX MB
```

---

## How Auto-Parallelism Works in ColossalAI

### 1. FX Graph Tracing (`ColoTracer`)

ColossalAI uses PyTorch's FX symbolic tracer (`ColoTracer`) to convert the model's `forward()` into a **computation graph** of nodes. The meta device tensors in `meta_input_sample` allow the tracer to infer tensor shapes without executing real computation.

The custom `GPT2Attention` in `gpt_modules.py` deliberately removes conditional branches from the HuggingFace original — because FX tracing requires a single static path through the graph.

### 2. Device Mesh Construction (`initialize_device_mesh`)

The physical GPUs are arranged into a **logical 2D mesh** (e.g., `[2, 2]` for 4 GPUs):
- **Mesh axis 0** — typically used for data parallelism (DP)
- **Mesh axis 1** — typically used for tensor parallelism (TP)

The mesh topology is chosen automatically by profiling point-to-point communication bandwidth (`alpha`) and latency (`beta`) between all GPU pairs using `AlphaBetaProfiler`.

### 3. Strategy Enumeration (`StrategiesConstructor`)

For each graph node, a **node handler** enumerates all valid tensor sharding strategies. Different operation types have dedicated handlers:

| Operation type    | Handler file                    | Example strategies                         |
|-------------------|---------------------------------|--------------------------------------------|
| Linear / matmul   | `linear_handler.py`             | Row-parallel, column-parallel, replicated  |
| Attention         | `bmm_handler.py`                | Head-parallel, batch-parallel              |
| Embedding         | `embedding_handler.py`          | Vocab-parallel, replicated                 |
| LayerNorm         | `layer_norm_handler.py`         | Batch-sharded, replicated                  |
| Element-wise ops  | `binary_elementwise_handler.py` | Match input sharding                       |
| Reshape/view      | `view_handler.py`               | Propagate sharding through shape changes   |

Each strategy specifies:
- The **ShardingSpec** of every input and output tensor (which dimensions are sharded on which mesh axis)
- The **compute cost** (estimated FLOPs)
- The **communication cost** (all-reduce / all-gather / reduce-scatter operations required)

### 4. Cost Graph and ILP Solver (`Solver`)

The solver (adapted from [Alpa](https://github.com/alpa-projects/alpa)) constructs a **cost graph**:
- **Node cost**: compute + activation memory cost for each strategy choice
- **Edge cost**: resharding cost (communication) when adjacent nodes use incompatible sharding specs

It then formulates an **Integer Linear Program (ILP)** using the `PuLP` library with the `coin-or-CBC` backend:

```
minimize:  sum(node_cost[i][s_i]) + sum(edge_cost[i][j][s_i][s_j])
subject to:
  - one strategy selected per node
  - optional: peak memory <= memory_budget
```

The solution is a vector of strategy indices — one per graph node — that minimizes total cost.

### 5. Model Transformation (`transform_to_sharded_model`)

With the solution in hand, the model is physically transformed:

1. **`runtime_preparation_pass`**:
   - Slices model weight tensors according to the chosen sharding specs on the current rank
   - Registers gradient hooks for gradient synchronization (all-reduce for DP, reduce-scatter for ZeRO-like patterns)

2. **`runtime_apply_pass`**:
   - Inserts communication collective operations (all-reduce, all-gather, reduce-scatter) into the FX graph at points where adjacent nodes have incompatible sharding specs

3. **`gm.recompile()`**:
   - Regenerates executable Python code from the modified FX graph
   - The resulting `ColoGraphModule` is a standard `nn.Module` that runs correctly in a distributed context

4. **`ModuleWrapper`**:
   - Wraps the graph module to automatically inject `sharding_spec_dict`, `origin_spec_dict`, and `comm_actions_dict` into each forward call

### 6. Distributed Training Loop

After `autoparallelize()`, the model `gm` is used as a normal PyTorch model. Each GPU rank:
- Receives the same random input (replicated dataloader)
- Executes the sharded forward pass (each rank only computes its slice of tensors)
- Communication collectives in the graph synchronize partial results at the right points
- Backward pass and optimizer step are also sharded/synchronized automatically through the hooks

---

## Key Files Reference

| File | Role |
|------|------|
| `auto_parallel_with_gpt.py` | Main training script |
| `gpt_modules.py` | Custom GPT-2 model (FX-traceable, branch-free) |
| `colossalai/auto_parallel/tensor_shard/initialize.py` | Top-level API: `autoparallelize`, `initialize_model`, `initialize_device_mesh` |
| `colossalai/auto_parallel/tensor_shard/solver/solver.py` | ILP solver (PuLP/CBC), adapted from Alpa |
| `colossalai/auto_parallel/tensor_shard/solver/strategies_constructor.py` | Enumerate strategies per node |
| `colossalai/auto_parallel/tensor_shard/solver/cost_graph.py` | Build and simplify cost graph |
| `colossalai/auto_parallel/passes/runtime_preparation_pass.py` | Shard weights, add grad hooks |
| `colossalai/auto_parallel/passes/runtime_apply_pass.py` | Insert communication ops into FX graph |
| `colossalai/device/alpha_beta_profiler.py` | Profile GPU-to-GPU bandwidth/latency |
| `colossalai/device/device_mesh.py` | Logical 2D device mesh abstraction |

---

## Launch Command

```bash
colossalai run --nproc_per_node 4 auto_parallel_with_gpt.py
```

Runs with 4 GPUs. The distributed process group is initialized via `launch_from_torch()`.
