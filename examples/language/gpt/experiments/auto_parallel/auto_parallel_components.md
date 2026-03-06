# ColossalAI Auto-Parallel: Analyzer, Generator, and Solver

## Overview

The auto-parallel system in `colossalai/auto_parallel/tensor_shard/` is built around three cooperating subsystems that work in sequence:

```
Model (nn.Module)
      |
      v
[0. DEVICE MESH]  -- profile GPUs, build 2D logical mesh --> DeviceMesh (alpha/beta, process groups)
      |
      v
[1. ANALYZER]   -- trace + shape propagation --> ColoGraphModule (nodes with shapes + meta)
      |
      v
[2. GENERATOR]  -- enumerate strategies per node --> StrategiesVector per node (costs attached)
      |
      v
[3. SOLVER]     -- ILP optimization --> solution: one strategy index per node
      |
      v
  Sharded Model (weights sliced, comm ops injected, recompiled)
```

---

## Component 0: DeviceMesh

**Location:** `colossalai/device/device_mesh.py`, `colossalai/device/alpha_beta_profiler.py`

**Purpose:** Model the physical GPU cluster as a logical N-dimensional grid. Every cost estimate, every sharding decision, and every runtime collective operation references this object. It is the shared foundation that connects all other components.

### 0.1 Physical vs Logical Mesh

The `DeviceMesh` separates two views of the same set of GPUs:

```
physical_mesh_id = [0, 1, 2, 3]   ← flat list of global ranks

logical_mesh_id  = [[0, 1],        ← 2D grid, shape (2, 2)
                    [2, 3]]

                      axis 0         axis 1
                  (Data Parallel)  (Tensor Parallel)
```

Any shape is supported — `(4,)` for 1D, `(2, 2)`, `(2, 4)`, `(4, 4)`, etc. The 2D case is the most common because it naturally maps one axis to DP and one to TP.

The logical position of each GPU in the mesh is stored in `_global_to_local_rank_mapping`:
```python
# For mesh [[0,1],[2,3]]:
{
    0: [0, 0],   # GPU 0 is at row 0, col 0
    1: [0, 1],   # GPU 1 is at row 0, col 1
    2: [1, 0],   # GPU 2 is at row 1, col 0
    3: [1, 1],   # GPU 3 is at row 1, col 1
}
```

### 0.2 Alpha-Beta Profiler (`AlphaBetaProfiler`)

Before the mesh shape is chosen, `AlphaBetaProfiler` benchmarks **every pair of GPUs** with real collective calls (all-reduce or broadcast) across multiple message sizes to fit an alpha-beta communication model:

```
latency(N bytes) = alpha + beta × N
```

| Symbol | Meaning | Typical NVLink | Typical PCIe |
|---|---|---|---|
| `alpha` | Fixed startup latency per message | ~20 µs | ~50 µs |
| `beta` | Per-byte transfer time | ~4 ps/byte | ~40 ps/byte |

The profiler stores results as `alpha_beta_dict[(rank_i, rank_j)] = (alpha, beta)`.

**Mesh topology selection:** `search_best_logical_mesh()` uses the profiled values to find the mesh shape that minimises expected communication cost. GPUs with low `alpha`+`beta` (NVLink peers, same node) are grouped on the **TP axis** (axis 1) because TP requires frequent intra-layer collectives. GPUs with higher latency (cross-node InfiniBand) are placed on the **DP axis** (axis 0) since DP only needs one gradient sync per step.

### 0.3 Process Groups

`DeviceMesh.init_logical_process_group()` calls `torch.distributed.new_group()` to create one `ProcessGroup` per axis per rank:

```
mesh [[0,1],[2,3]]

axis 0 (DP) groups:   {0, 2}  and  {1, 3}   ← same column
axis 1 (TP) groups:   {0, 1}  and  {2, 3}   ← same row
```

These process groups are stored in `_process_group_dict[global_rank][axis]` and retrieved at runtime via `mesh.get_process_group(axis)`. Every NCCL collective (all-reduce, all-gather, reduce-scatter) is executed within one of these groups, not across the full world — which avoids unnecessary cross-group traffic.

### 0.4 Communication Cost Methods

`DeviceMesh` exposes cost estimation methods used by the Generator to price each strategy:

```python
mesh.all_reduce_cost(num_bytes, mesh_dim)
    # = alpha[dim] + beta[dim] × 2(D-1)/D × num_bytes

mesh.all_gather_cost(num_bytes, mesh_dim)
    # = alpha[dim] + beta[dim] × (D-1)/D × num_bytes

mesh.reduce_scatter_cost(num_bytes, mesh_dim)
    # = alpha[dim] + beta[dim] × (D-1)/D × num_bytes

mesh.all_to_all_cost(num_bytes, mesh_dim)
    # = alpha[dim] + beta[dim] × (D-1)/D²  × num_bytes × D/2
```

Where `D = mesh.shape[mesh_dim]` is the number of devices along that axis.

### 0.5 How DeviceMesh Connects to Every Other Component

```
DeviceMesh
    │
    ├── Generator (StrategiesConstructor / NodeHandler)
    │       uses mesh.all_reduce_cost / all_gather_cost
    │       to compute communication_cost for each ShardingStrategy
    │
    ├── ShardingSpec  (attached to every tensor in the graph)
    │       stores reference to mesh
    │       dim_partition_dict maps tensor dim → mesh axis
    │       e.g. {0: [0]} means dim 0 sharded on mesh axis 0 (DP)
    │
    ├── runtime_preparation_pass
    │       uses mesh.shape[axis] to compute shard slice per rank
    │       e.g. rank local on axis 1 = 0 → take weight[:, 0 : H/num_tp_gpus]
    │
    └── runtime_apply_pass (runtime_comm_spec_apply)
            uses mesh.get_process_group(axis) to run the correct
            NCCL collective on the right subset of GPUs
```

### 0.6 ShardingSpec — Tensor's View of the Mesh

Every tensor in the graph is annotated with a `ShardingSpec` that encodes **which dimension of the tensor is distributed across which axis of the mesh**:

```python
# Notation:  dim_partition_dict = {tensor_dim: [mesh_axes]}

ShardingSpec(mesh, shape=[16,1024,4096], dim_partition_dict={})
    # → Replicated: every GPU holds the full [16, 1024, 4096]

ShardingSpec(mesh, shape=[16,1024,4096], dim_partition_dict={0: [0]})
    # → S0: batch dim sharded on mesh axis 0 (DP)
    #   GPU (DP=0): holds [8, 1024, 4096]
    #   GPU (DP=1): holds [8, 1024, 4096]

ShardingSpec(mesh, shape=[16,1024,4096], dim_partition_dict={2: [1]})
    # → S1 on last dim: hidden sharded on mesh axis 1 (TP)
    #   GPU (TP=0): holds [16, 1024, 2048]
    #   GPU (TP=1): holds [16, 1024, 2048]

ShardingSpec(mesh, shape=[16,1024,4096], dim_partition_dict={0:[0], 2:[1]})
    # → S0S1: batch sharded on DP axis AND hidden sharded on TP axis
    #   GPU (DP=0,TP=0): holds [8, 1024, 2048]
```

The string notation used in `solution` output (`"RS1 = RR x RS1"`) reads each dimension left-to-right: `R` = replicated, `S0`/`S1` = sharded on mesh axis 0/1.

---

## Component 1: Analyzer

**Location:** `colossalai/_analyzer/`

**Purpose:** Convert a PyTorch `nn.Module` into a static, annotated computation graph where every node carries full shape and memory metadata — without running any real GPU computation.

### Sub-components

#### 1.1 ColoTracer (`_analyzer/fx/tracer/tracer.py`)

Extends PyTorch's `torch.fx.Tracer` to symbolically trace the model's `forward()` function into an FX `Graph`.

**Key behaviors:**
- Accepts `meta_args`: shape-only tensors on the `"meta"` device (no real data).
- Handles `torch.utils.checkpoint` regions (`trace_act_ckpt=True`) by recording checkpoint boundaries.
- Splits bias-addition ops into two separate nodes (`bias_addition_split=True`) so sharding strategies can handle weight and bias independently.
- Treats certain modules as leaf nodes (not traced into) via `_custom_leaf_module` registry.
- Produces a `ColoProxy` for each operation (a traced symbolic value).

**Output:** A raw `torch.fx.Graph` — a DAG where each node represents one operation (`placeholder`, `call_function`, `call_module`, `call_method`, `get_attr`, `output`).

#### 1.2 ColoGraphModule (`_analyzer/fx/graph_module.py`)

A `torch.fx.GraphModule` subclass that wraps the traced graph together with the original module. Supports `recompile()` to regenerate executable Python code from the graph after modifications.

Codegen is set to `ActivationCheckpointCodeGen` to produce forward code that respects activation checkpoint regions recorded during tracing.

#### 1.3 ShapeProp / shape_prop_pass (`_analyzer/fx/passes/shape_prop.py`)

Walks every node in the graph and **executes it symbolically** using `MetaTensor` — a fake tensor that performs all shape/dtype/device arithmetic but allocates zero memory.

For each node it attaches a `MetaInfo` object containing:

| Field | Description |
|---|---|
| `outputs` | Output tensor(s) with shape/dtype on meta device |
| `inputs` | Input tensors (meta) |
| `parameters` | Weight/bias tensors referenced by this node |
| `buffers` | Buffer tensors (e.g., causal mask) |
| `is_alias` | Whether the output shares storage with an input |
| `global_ctx` / `curr_ctx` | Saved-tensor hooks context for activation memory tracking |

The node's `_meta_data` field is set to the output shape, which all downstream components rely on.

**Output:** Same `ColoGraphModule`, now with every node annotated with shape and memory metadata.

#### 1.4 MetaTensor (`_analyzer/_subclasses/meta_tensor.py`)

A `torch.Tensor` subclass that intercepts all PyTorch operations. It forwards them through `MetaTensorMode`, which uses PyTorch's dispatch mechanism to compute output shapes without allocating real memory.

This is what makes it possible to analyze a multi-billion parameter model's graph on CPU with negligible memory.

### Analyzer Flow Summary

```
nn.Module + meta_args
      |
      | ColoTracer.trace(root=model, meta_args=meta_args)
      v
torch.fx.Graph  (static DAG of all ops)
      |
      | ColoGraphModule(model, graph)
      v
ColoGraphModule  (graph + module weights)
      |
      | shape_prop_pass(gm, *meta_args.values())
      v
ColoGraphModule  (every node has _meta_data, MetaInfo: shapes, params, buffers)
      |
      | gm.recompile()
      v
Executable ColoGraphModule (ready for strategy enumeration)
```

---

## Component 2: Generator

**Location:** `colossalai/auto_parallel/tensor_shard/node_handler/`

**Purpose:** For every node in the graph, enumerate all valid tensor sharding strategies and compute their costs (compute, communication, memory, resharding). **DeviceMesh is the primary input** — the Generator cannot produce cost estimates without it, because communication cost depends on the number of devices per axis and their measured bandwidth.

### Sub-components

#### 2.1 StrategiesConstructor (`solver/strategies_constructor.py`)

Orchestrates the Generator phase. Iterates over every node in the graph and dispatches to the correct `NodeHandler` based on node type.

**Node type dispatch:**

| FX node op | Handler used |
|---|---|
| `placeholder` | `PlaceholderHandler` |
| `get_attr` | `GetattrHandler` |
| `output` | `OutputHandler` |
| `call_module` | looked up from `operator_registry` by submodule type |
| `call_function` | looked up from `operator_registry` by function target |
| `call_method` | looked up from `operator_registry` by method |

After processing all nodes, it calls `generate_alias_set()` to detect repeated transformer blocks (e.g., identical attention layers) so the solver can share strategy variables across them, reducing ILP problem size.

**Output per node:** A `StrategiesVector` — a list of `ShardingStrategy` objects — attached as `node.strategies_vector`.

#### 2.2 NodeHandler (`node_handler/node_handler.py`)

Abstract base class for all operation-specific handlers. Each handler:

1. Builds an `operation_data_mapping`: maps names (`"input"`, `"weight"`, `"bias"`, `"output"`) to `OperationData` objects carrying the meta tensor and its type (`INPUT`, `PARAM`, `OUTPUT`).
2. Instantiates the appropriate `StrategyGenerator(s)` for the operation.
3. Calls `generator.collate_strategies()` to get a list of `ShardingStrategy` objects.
4. Calls `update_resharding_cost(strategy)` for each generated strategy.

**Concrete handlers include:**

| Handler | Operations covered |
|---|---|
| `LinearModuleHandler` / `LinearFunctionHandler` | `nn.Linear`, `F.linear` |
| `MatmulHandler` | `torch.matmul`, `@` operator |
| `BmmHandler` | `torch.bmm` (batched matmul — attention QK^T and AV) |
| `EmbeddingHandler` | `nn.Embedding` |
| `LayerNormHandler` | `nn.LayerNorm` |
| `ConvHandler` | `nn.Conv1d/2d/3d` |
| `BatchNormHandler` | `nn.BatchNorm*` |
| `BinaryElementwiseHandler` | `+`, `-`, `*`, `/`, `torch.add`, etc. |
| `UnaryElementwiseHandler` | `relu`, `gelu`, `softmax`, `dropout`, etc. |
| `ViewHandler` / `DefaultReshapeHandler` | `.view()`, `.reshape()`, `torch.reshape` |
| `SplitHandler` | `torch.split`, `.split()` |
| `TransposeHandler` / `PermuteHandler` | `.transpose()`, `.permute()` |
| `SoftmaxHandler` | `F.softmax` |
| `SumHandler` | `torch.sum` |
| `GetitemHandler` | `tensor[...]` indexing |
| `PlaceholderHandler` | Graph inputs |
| `OutputHandler` | Graph output |

#### 2.3 StrategyGenerator (`node_handler/strategy/strategy_generator.py`)

Abstract base for strategy generation logic. Each subclass implements `collate_strategies()` which returns a list of `ShardingStrategy` objects.

A strategy is described using **ShardingSpec notation**: `S` = sharded on that axis, `R` = replicated. The mesh axis index is the subscript.

**Example strategies for a Linear layer (column-parallel):**
```
"S0R = S0R x RR"   # input sharded on batch(S0), weight replicated, output sharded on batch
"RS1 = RR x RS1"   # input replicated, weight col-parallel(S1), output sharded on feature
"S0S1 = S0R x RS1" # both batch-sharded input and col-parallel weight
"RR = RR x RR"     # fully replicated (no parallelism)
```

Each `StrategyGenerator` calls helper methods to:
- Build `dim_partition_dict` for each operand: which tensor dimensions map to which mesh axes
- Convert to `ShardingSpec` objects
- Compute `communication_action_mapping`: which collectives (`all-reduce`, `all-gather`, `reduce-scatter`) are needed after the op

#### 2.4 ShardingStrategy (`sharding_strategy.py`)

The core data object produced by the Generator. Contains:

```python
@dataclass
class ShardingStrategy:
    name: str                                             # e.g. "S0S1 = S0R x RS1"
    sharding_specs: Dict[OperationData, ShardingSpec]     # spec for every tensor in the op
    compute_cost:       TrainCycleItem                    # FLOPs fwd/bwd/total
    communication_cost: TrainCycleItem                    # comm bytes fwd/bwd/total
    memory_cost:        TrainCycleItem                    # MemoryCost fwd/bwd/total
    communication_actions: Dict[OperationData, CommAction] # what collective to run, when
    resharding_costs: Dict[Node, List[TrainCycleItem]]    # cost[pred_node][pred_strategy_idx]
```

`TrainCycleItem` holds separate `fwd`, `bwd`, `total` values because communication patterns differ between forward (e.g., all-reduce for row-parallel output) and backward passes.

`MemoryCost` breaks down into `activation`, `parameter`, `temp`, `buffer` bytes — the Solver uses this for memory budget constraints.

#### 2.5 Resharding Cost Computation (`NodeHandler.update_resharding_cost`)

For each generated strategy and each predecessor node, the handler queries `ShapeConsistencyManager.shape_consistency(prev_spec, current_spec)` to determine what collective operations are needed to convert the predecessor's output sharding into the format this strategy expects as input.

For example: if node A outputs `S0` (sharded on dim 0) but node B needs `R` (replicated) as input, an `all-gather` is needed. Its cost is estimated using the device mesh's `alpha` (latency) and `beta` (bandwidth) values.

The result is stored as `strategy.resharding_costs[pred_node][pred_strategy_index]`.

### Generator Flow Summary

```
ColoGraphModule (nodes with _meta_data)
      |
      | StrategiesConstructor.build_strategies_and_cost()
      |
      | for each node:
      |   handler = operator_registry.get(node_type)(node, device_mesh, ...)
      |   handler.register_strategy()
      |     |-- generator.collate_strategies()     -> list of ShardingStrategy
      |     |-- update_resharding_cost(strategy)   -> fills strategy.resharding_costs
      |   node.strategies_vector = StrategiesVector([strategy_0, strategy_1, ...])
      v
Graph where every node has strategies_vector:
  node_0.strategies_vector = [S0=Replicated, S1=ColParallel, S2=RowParallel, ...]
  node_1.strategies_vector = [...]
  ...
```

---

## Component 3: Solver

**Location:** `colossalai/auto_parallel/tensor_shard/solver/`

**Purpose:** Given all node strategy options and their costs, find the globally optimal assignment of one strategy per node that minimizes total cost (compute + communication + resharding) subject to an optional memory constraint.

### Sub-components

#### 3.1 CostGraph (`solver/cost_graph.py`)

Transforms the per-node strategy data into a form suitable for ILP.

**Two responsibilities:**

**A. Build edge cost matrix:**

For every pair of adjacent nodes `(src, dst)`, builds a 2D cost matrix:

```
edge_costs[(src, dst)][(i, j)] = resharding cost when src uses strategy i
                                  and dst uses strategy j
```

This is a `|strategies_src| × |strategies_dst|` matrix per edge.

**B. Graph simplification (node merging):**

Trivial nodes — element-wise ops (relu, add, dropout), reshape/view, transpose — are *merged* into their successor nodes. The reasoning: their sharding is fully determined by their input's sharding, so they add no independent decision variables.

Merging works by:
1. Identifying mergeable `(src, dst)` pairs where `strategies_vector.check_merge()` returns True
2. Absorbing `src`'s costs + resharding costs into `dst`'s strategies via a `merge_map`
3. Rewiring edges: `src`'s parent connections are promoted to `dst`
4. Recording in `following_dict[dst] = src` so the solver can recover merged node strategies after solving

This significantly reduces the ILP problem size for models with many element-wise operations.

#### 3.2 Solver (`solver/solver.py`)

Adapted from [Alpa](https://github.com/alpa-projects/alpa/). Uses `PuLP` with the `coin-or-CBC` backend to solve an Integer Linear Program.

**ILP Formulation:**

**Variables:**

```
s[i] ∈ {0,1}^|strategies_i|     for each non-merged node i
                                  (one-hot: exactly one strategy selected)

e[i,j] ∈ {0,1}^(|s_i|×|s_j|)   for each edge (i,j)
                                  (one-hot over strategy pair combinations)
```

**Objective — minimize total cost:**

```
minimize:
  Σ_i  dot(s[i],  compute_cost[i] + communication_cost[i])   # node costs
  + Σ_(i,j)  dot(e[i,j],  resharding_cost[i,j])              # edge resharding costs
```

**Constraints:**

```
# 1. Each node selects exactly one strategy
Σ_k s[i][k] = 1   for all nodes i

# 2. Edge variable consistency with node variables
Σ_k e[i,j][k * |s_j| + l] = s[j][l]   for all edges (i,j), all j-strategies l
Σ_l e[i,j][k * |s_j| + l] = s[i][k]   for all edges (i,j), all i-strategies k

# 3. Memory budget (optional, if memory_budget > 0)
Σ_i  dot(s[i], memory_cost[i])  <=  memory_budget

# 4. Alias constraints (repeated blocks share the same strategy variable)
s[i] = s[alias[i]]   for all nodes i in alias_set
```

**Alias optimization:** Repeated transformer blocks (detected by `generate_alias_set()`) share the same `s[i]` variable, reducing binary variables dramatically for deep models.

**Follow optimization:** Merged nodes use `s[i] = s[follow[i]]` — no independent variable created.

**Output:** `solution` — a flat list of integers, one per non-trivial node, each being the index into that node's `strategies_vector`.

#### 3.3 Strategy Recovery (`_recover_merged_node_strategy`)

After the ILP returns `solution`, merged nodes have no direct entry. The Solver walks through the graph and recovers each merged node's strategy index by matching its input's chosen sharding spec against the available strategies.

#### 3.4 GraphAnalyser (`solver/graph_analysis.py`)

Performs liveness analysis on the computation graph to determine which tensors are live at each point. This is used to compute peak memory (rather than sum of all node memories). Currently not fully wired in — the solver falls back to treating all nodes as live simultaneously.

### Solver Flow Summary

```
StrategiesVector per node (from Generator)
      |
      | CostGraph(leaf_strategies)
      |   _build_cost_graph()    -> edge_costs[(src,dst)][(i,j)]
      |   simplify_graph()       -> merge trivial nodes, reduce variables
      v
CostGraph (simplified)
      |
      | Solver(graph, strategies_constructor, cost_graph, memory_budget)
      |   _prepare_data_for_solver()   -> numpy arrays: compute/comm/memory/resharding costs
      |   _call_solver_serialized_args()
      |     -> build ILP variables s[i], e[i,j]
      |     -> add constraints (one-hot, edge consistency, memory budget, alias)
      |     -> pulp.solve() via CBC
      |   _recover_merged_node_strategy()
      v
solution: List[int]   (one strategy index per node)
```

---

## End-to-End Data Flow Between Components

```
  physical GPUs ──> ┌─────────────────────────────────────────────┐
                    │               DEVICE MESH                    │
                    │                                              │
                    │  AlphaBetaProfiler.profile_ab()             │
                    │    → alpha/beta per GPU pair                 │
                    │  search_best_logical_mesh()                  │
                    │    → logical_mesh_id [[0,1],[2,3]]           │
                    │  init_logical_process_group()                │
                    │    → ProcessGroup per axis per rank          │
                    └──────────────────┬──────────────────────────┘
                                       │ DeviceMesh
                                       │   (shape, alpha, beta,
                                       │    process_group_dict)
                          ┌────────────┴────────────┐
                          │                         │
                          v                         v (passed to all)
                    ┌─────────────────────────────────────────────┐
                    │                  ANALYZER                    │
                    │                                              │
  nn.Module    ──>  │  ColoTracer.trace()                         │
  meta_args    ──>  │    → torch.fx.Graph (DAG of ops)            │
                    │  ColoGraphModule(model, graph)               │
                    │  shape_prop_pass()                           │
                    │    → node._meta_data (shapes)                │
                    │    → node MetaInfo (params, buffers, memory) │
                    └──────────────────┬──────────────────────────┘
                                       │ ColoGraphModule
                                       v
                    ┌─────────────────────────────────────────────┐
                    │                 GENERATOR                    │
                    │                                              │
                    │  StrategiesConstructor(graph, device_mesh)   │
                    │    for each node:                            │
                    │      NodeHandler(node, device_mesh, ...)     │
                    │        StrategyGenerator.collate_strategies()│
                    │          → ShardingStrategy objects          │
                    │            name: "RS1 = RR x RS1"           │
                    │            sharding_specs per tensor         │
                    │            compute_cost (FLOPs)              │
                    │            communication_cost                │
                    │              ← mesh.all_reduce_cost(bytes,1) │
                    │            memory_cost                       │
                    │        update_resharding_cost()              │
                    │          → strategy.resharding_costs[pred]   │
                    │      node.strategies_vector = [s0, s1, ...]  │
                    └──────────────────┬──────────────────────────┘
                                       │ strategies_vector per node
                                       v
                    ┌─────────────────────────────────────────────┐
                    │                   SOLVER                     │
                    │                                              │
                    │  CostGraph                                   │
                    │    build edge cost matrices                  │
                    │    merge trivial nodes                       │
                    │  Solver (ILP via PuLP/CBC)                   │
                    │    variables: s[i], e[i,j]                   │
                    │    minimize: node costs + resharding costs    │
                    │    constraint: memory budget (optional)       │
                    │    alias set: share vars for repeat blocks   │
                    └──────────────────┬──────────────────────────┘
                                       │ solution: List[int]
                                       v
                    ┌─────────────────────────────────────────────┐
                    │              TRANSFORMATION                  │
                    │                                              │
                    │  runtime_preparation_pass(gm, device_mesh)  │
                    │    shard weights: rank slice via mesh.shape  │
                    │    register gradient hooks                   │
                    │  runtime_apply_pass()                        │
                    │    insert runtime_apply nodes (resharding)   │
                    │    insert runtime_comm_spec_apply nodes      │
                    │      ← uses mesh.get_process_group(axis)     │
                    │  gm.recompile()                              │
                    │    generate final distributed forward code   │
                    └──────────────────┬──────────────────────────┘
                                       │
                                       v
                             Sharded ModuleWrapper
                          (ready for distributed training)
```

---

## Key Data Structures Reference

| Structure | Module | Description |
|---|---|---|
| `DeviceMesh` | `colossalai/device/device_mesh.py` | 2D logical GPU topology; holds shape, alpha/beta costs, process groups per axis |
| `AlphaBetaProfiler` | `colossalai/device/alpha_beta_profiler.py` | Benchmarks GPU pairs; fits alpha/beta model; selects optimal mesh shape |
| `ColoGraphModule` | `_analyzer/fx/graph_module.py` | FX GraphModule with recompile support |
| `MetaInfo` | `_analyzer/fx/node_util.py` | Per-node shape/param/buffer/memory metadata |
| `MetaTensor` | `_analyzer/_subclasses/meta_tensor.py` | Zero-cost fake tensor for shape inference |
| `StrategiesVector` | `tensor_shard/sharding_strategy.py` | List of `ShardingStrategy` for one node |
| `ShardingStrategy` | `tensor_shard/sharding_strategy.py` | One candidate strategy: specs + all costs |
| `ShardingSpec` | `colossalai/tensor/sharding_spec.py` | Per-tensor distribution descriptor: `dim_partition_dict` maps tensor dim → mesh axis; holds reference to `DeviceMesh` |
| `OperationData` | `tensor_shard/sharding_strategy.py` | Typed tensor reference (INPUT/PARAM/OUTPUT) |
| `TrainCycleItem` | `tensor_shard/sharding_strategy.py` | `fwd`/`bwd`/`total` cost triple |
| `MemoryCost` | `tensor_shard/sharding_strategy.py` | `activation`/`parameter`/`temp`/`buffer` bytes |
| `CommAction` | `tensor_shard/sharding_strategy.py` | Collective type + when to execute (BEFORE/AFTER/HOOK) + which process group axis |
| `CostGraph` | `tensor_shard/solver/cost_graph.py` | Edge cost matrices + node merge logic |
| `StrategiesConstructor` | `tensor_shard/solver/strategies_constructor.py` | Orchestrates Generator phase; receives DeviceMesh as constructor arg |
| `Solver` | `tensor_shard/solver/solver.py` | ILP solver wrapping PuLP/CBC |