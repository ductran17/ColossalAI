# Ba Module Nền Tảng của HybridParallelPlugin

> Tài liệu tổng quan về kiến trúc 3D Parallel trong ColossalAI: ShardFormer (Tensor Parallel), PipelineStageManager (Pipeline Parallel), và cơ chế Data Parallel.

---

## 1. Tổng quan kiến trúc 3D Parallel

ColossalAI `HybridParallelPlugin` cài đặt **3D Parallelism** bằng cách kết hợp đồng thờ ba chiều song song:

| Chiều | Module chính | Trách nhiệm | Phạm vi giao tiếp |
|-------|-------------|-------------|------------------|
| **Tensor Parallel (TP)** | `ShardFormer` | Chia linear layers theo chiều hidden | Intra-node (NVLink/PCIe) |
| **Pipeline Parallel (PP)** | `PipelineStageManager` | Chia model thành các stage theo chiều layer | Intra-node hoặc cross-node |
| **Data Parallel (DP)** | `sync_dp_grads()` / DDP / ZeRO | Chia batch data, đồng bộ gradient | Cross-node (Ethernet) |

Ba module này được liên kết thông qua `ProcessGroupMesh` — một lưới process group đa chiều quản lý toàn bộ topology GPU.

---

## 2. ShardFormer — Tensor Parallel Engine

### 2.1. Vai trò

`ShardFormer` là module cài đặt **Tensor Parallelism (TP)** theo phong cách Megatron-LM. Nó nhận một model transformers nguyên bản và "xẻ nhỏ" các lớp linear để thực thi song song trên nhiều GPU trong cùng một node.

### 2.2. Nguyên lý hoạt động

ShardFormer hoạt động qua hai bước chính:

**Bước 1: Auto-detect kiến trúc**
```python
# Từ model object, xác định model family
full_name = _fullname(model)  # "transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel"
policy_location = _POLICY_LIST[full_name]  # → GPT2Policy
policy = import_policy(policy_location)     # Load policy module
```

Policy định nghĩa:
- Lớp nào cần thay thế (e.g. `nn.Linear` → `ParallelLinear`)
- Chiều nào cần shard (column-parallel vs row-parallel)
- Các hook cần inject (AllReduce, AllGather)

**Bước 2: Shard model**
```python
sharder = ModelSharder(model=model, shard_config=shard_config, policy=policy)
shared_params = sharder.shard()
```

Ví dụ với GPT2 Attention:
```
Trước shard:
  c_attn: Linear(768, 2304)  # QKV projection
  c_proj: Linear(768, 768)   # Output projection

Sau shard (tp=2):
  c_attn: ColumnParallelLinear(768, 2304)  # Mỗi GPU giữ 768×1152
  c_proj: RowParallelLinear(768, 768)      # Mỗi GPU giữ 384×768
```

### 2.3. Các loại parallelism trong ShardFormer

| Loại | Cách shard | Ví dụ |
|------|-----------|-------|
| **Column Parallel** | Chia output dim | `c_attn`, MLP up-project |
| **Row Parallel** | Chia input dim | `c_proj`, MLP down-project |
| **Sequence Parallel** | Chia sequence dim | Input activations (tùy chọn) |

### 2.4. Giao tiếp

- **AllReduce**: Sau row-parallel projection (attention output, MLP output)
- **AllGather**: Khi cần full tensor cho residual connection
- **Reduce-Scatter**: Sequence parallelism (tùy chọn)

→ Giao tiếp xảy ra **liên tục** (mỗi layer forward + backward), nên **bắt buộc intra-node** (NVLink/PCIe).

---

## 3. PipelineStageManager — Pipeline Parallel Scheduler

### 3.1. Vai trò

`PipelineStageManager` là module cài đặt **Pipeline Parallelism (PP)**. Nó chia model thành nhiều **stage** theo chiều layer, mỗi stage chạy trên một nhóm GPU riêng. Module này quản lý:
- Xác định stage nào GPU hiện tại đang giữ
- Thiết lập P2P communication với stage trước/sau
- Cung cấp thông tin cho pipeline scheduler (1F1B, Interleaved, ZBV)

### 3.2. Nguyên lý hoạt động

**Bước 1: Xác định stage từ rank**
```python
# Với ProcessGroupMesh shape = (dp, pp, tp)
coord = pg_mesh.coordinate()  # e.g. (0, 2, 1) = dp=0, pp=2, tp=1
stage = coord[pipeline_axis]   # stage = 2
```

**Bước 2: Thiết lập P2P neighbors**
```python
prev_coord = coord với pipeline_axis - 1 (wrap)
next_coord = coord với pipeline_axis + 1 (wrap)
self.prev_rank = pg_mesh.ravel(prev_coord)  # Stage trước
self.next_rank = pg_mesh.ravel(next_coord)  # Stage sau
```

Ví dụ với 4 stage (pp=4), rank layout:
```
Stage 0: ranks [0, 1]     ← node 1
Stage 1: ranks [2, 3, 4, 5] ← node 2
Stage 2: ranks [6, 7]     ← node 3
Stage 3: ranks [8, 9]     ← node 4

P2P edges: 0→1→2→3→(wrap to 0)
```

**Bước 3: Chia layer cho stage**
```python
layers_per_stage = total_layers // num_stages
stage_start = stage * layers_per_stage
stage_end = stage_start + layers_per_stage
```

Ví dụ: 24 layers, pp=4
```
Stage 0: layers 0-5
Stage 1: layers 6-11
Stage 2: layers 12-17
Stage 3: layers 18-23
```

### 3.3. Pipeline Schedules

| Schedule | Mô tả | Ưu điểm | Nhược điểm |
|----------|-------|---------|-----------|
| **1F1B** (default) | Forward 1 microbatch → Backward 1 microbatch xen kẽ | Memory ổn định, đơn giản | Bubble lớn khi pp cao |
| **Interleaved** | Mỗi GPU giữ nhiều stage chunks | Bubble nhỏ hơn | Phức tạp, nhiều P2P |
| **ZBV** (Zero Bubble) | Forward + backward overlap tối đa | Bubble ~0 | Chỉ ZeRO-1, phức tạp |

### 3.4. Giao tiếp

- **P2P send/recv**: Truyền activation tensor giữa stage $i$ và $i+1$
- Mỗi microbatch trigger 2 lần P2P (forward + backward)
- Có thể intra-node hoặc cross-node tùy topology

---

## 4. Data Parallel — Gradient Synchronization

### 4.1. Vai trò

Data Parallel (DP) chia **batch dữ liệu** cho nhiều GPU replica, mỗi GPU chạy forward/backward trên một phần data. Sau đó gradient được đồng bộ để tất cả GPU có cùng model update.

### 4.2. Các chế độ DP trong HybridParallelPlugin

| Chế độ | Khi nào dùng | Cách sync | Đặc điểm |
|--------|-------------|-----------|---------|
| **PyTorch DDP** | `pp=1, zero_stage=0` | Bucketed async AllReduce | Mặc định PyTorch |
| **Manual AllReduce** | `pp>1, zero_stage=0` | `sync_dp_grads()` sau tất cả microbatches | ColossalAI tự implement |
| **ZeRO-1** | `zero_stage=1` | Shard optimizer state | Giảm memory |
| **ZeRO-2** | `zero_stage=2` | Shard optimizer + gradient | Giảm memory hơn nữa |

### 4.3. Nguyên lý hoạt động

**Với DDP (`pp=1`)**:
```python
# PyTorch tự động hook
loss.backward()          # Gradient tính xong
# DDP hook: tự động kick-off AllReduce bucket
dist.all_reduce(grad)    # Đồng bộ gradient
optimizer.step()         # Update
```

**Với Pipeline (`pp>1`)**:
```python
# Trong execute_pipeline()
for m in range(num_microbatches):
    output = forward(...)   # Không sync gradient
    loss.backward()          # Gradient accumulate

# Sau tất cả microbatches
model.sync_dp_grads()     # Manual AllReduce một lần
optimizer.step()
```

### 4.4. Giao tiếp

- **AllReduce (ring)**: Đồng bộ gradient qua DP group
- Có thể **overlap** với backward (DDP/ZeRO) hoặc **serial** (manual sync)
- Thường là **cross-node** bottleneck vì DP group thường span nhiều node

---

## 5. Tương tác giữa ba module

### 5.1. Khởi tạo trong HybridParallelPlugin

```python
# 1. Tạo ProcessGroupMesh — định nghĩa topology 3D
pg_mesh = ProcessGroupMesh(dp_size, pp_size, tp_size)

# 2. Khởi tạo PipelineStageManager từ mesh
stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=pp_axis)

# 3. Tạo ShardConfig từ stage_manager
shard_config = ShardConfig(
    tensor_parallel_process_group=tp_group,
    pipeline_stage_manager=stage_manager,
    enable_tensor_parallelism=(tp_size > 1),
)

# 4. Shard model bằng ShardFormer
shardformer = ShardFormer(shard_config)
model, shared_params = shardformer.optimize(model)

# 5. Wrap với DDP nếu cần (chỉ khi pp=1)
if use_ddp:
    model = DDP(model, process_group=dp_group)

# 6. Tạo Pipeline Scheduler
scheduler = OneForwardOneBackwardSchedule(stage_manager, ...)
```

### 5.2. Luồng dữ liệu trong 1 training step

TP xảy ra **bên trong** mỗi Pipeline Stage (mỗi layer trong stage có TP AllReduce). PP xảy ra **giữa** các Stage (P2P activation/gradient). DP xảy ra **cuối cùng** (AllReduce gradient).

```
Input batch → Chia theo DP (mỗi GPU 1 shard)
    ↓
FOR each microbatch m = 1..M:
    
    ┌─[Pipeline Stage 0]──────────────────────────────┐
    │  FOR each layer in Stage 0:                     │
    │    ├─ ShardFormer: split tensor theo TP group   │
    │    ├─ Compute: matmul, attention, MLP           │
    │    └─ AllReduce: sau row-parallel projection    │
    │  P2P send activation ────────→ Stage 1          │
    └─────────────────────────────────────────────────┘
                          ↓
    ┌─[Pipeline Stage 1]──────────────────────────────┐
    │  (tương tự Stage 0, TP cho mỗi layer)           │
    │  P2P send activation ────────→ Stage 2          │
    └─────────────────────────────────────────────────┘
                          ↓
                        ...
                          ↓
    ┌─[Pipeline Stage pp-1]───────────────────────────┐
    │  TP forward cho mỗi layer trong stage           │
    │  Loss computation                               │
    └─────────────────────────────────────────────────┘

    ← Backward microbatch m ←
    
    ┌─[Pipeline Stage pp-1]───────────────────────────┐
    │  Loss.backward()                                │
    │  TP backward (AllReduce gradient)               │
    │  P2P send gradient ←──────── Stage pp-2         │
    └─────────────────────────────────────────────────┘
                          ↑
                        ...
                          ↑
    ┌─[Pipeline Stage 0]──────────────────────────────┐
    │  TP backward cho mỗi layer                      │
    └─────────────────────────────────────────────────┘

END FOR (tất cả M microbatches xong)
    ↓
[All stages, all microbatches xong]:
    sync_dp_grads()  # DP AllReduce across DP group (nếu dp > 1)
    optimizer.step()  # AdamW update (memory-bandwidth bound)
```

**Chi tiết Tensor Parallel trong mỗi layer:**

```
Input activation [B, S, H]
    ↓
ColumnParallelLinear (QKV projection):
    Mỗi GPU giữ 1/tp của output dim → compute partial result
    Không cần AllReduce (vì output sẽ dùng cho attention local)
    ↓
Attention compute (local trên mỗi GPU):
    Softmax, matmul giữa Q_local × K_local^T
    ↓
RowParallelLinear (output projection):
    Mỗi GPU giữ 1/tp của input dim → compute partial result
    dist.all_reduce(output)  # ← TP AllReduce #1
    ↓
Residual connection (full tensor):
    output + input
    ↓
ColumnParallelLinear (MLP up-project):
    Tương tự QKV
    ↓
GELU / SwiGLU activation
    ↓
RowParallelLinear (MLP down-project):
    dist.all_reduce(output)  # ← TP AllReduce #2
    ↓
Residual connection
```

**Tóm lại phân cấp:**
- **Outermost**: PP chia theo stage (layer range)
- **Middle**: Mỗi stage chạy M microbatches (1F1B schedule)
- **Innermost**: Mỗi layer trong stage chạy TP (AllReduce giữa TP group)

### 5.3. Ví dụ cụ thể: 8 GPUs, pp=2, tp=2, dp=2

**ProcessGroupMesh**: shape `(2, 2, 2)` — (dp=2, pp=2, tp=2)

| Rank | dp | pp | tp | Stage | TP group | DP group |
|------|----|----|----|-------|----------|----------|
| 0 | 0 | 0 | 0 | 0 | {0,1} | {0,4} |
| 1 | 0 | 0 | 1 | 0 | {0,1} | {1,5} |
| 2 | 0 | 1 | 0 | 1 | {2,3} | {2,6} |
| 3 | 0 | 1 | 1 | 1 | {2,3} | {3,7} |
| 4 | 1 | 0 | 0 | 0 | {4,5} | {0,4} |
| 5 | 1 | 0 | 1 | 0 | {4,5} | {1,5} |
| 6 | 1 | 1 | 0 | 1 | {6,7} | {2,6} |
| 7 | 1 | 1 | 1 | 1 | {6,7} | {3,7} |

**ShardFormer**: Mỗi cặp {0,1}, {2,3}, {4,5}, {6,7} chia linear layers.

**PipelineStageManager**: Stage 0 (ranks 0,1,4,5) ↔ Stage 1 (ranks 2,3,6,7).

**DP**: {0,4}, {1,5}, {2,6}, {3,7} đồng bộ gradient sau backward.

---

## 6. Tóm tắt trách nhiệm

| Module | Trách nhiệm chính | Dữ liệu input | Dữ liệu output |
|--------|-------------------|---------------|----------------|
| **ShardFormer** | Shard model theo chiều hidden | Model nguyên bản + Policy | Model đã shard + shared_params |
| **PipelineStageManager** | Quản lý stage boundary + P2P | ProcessGroupMesh + pp_axis | prev_rank, next_rank, stage_indices |
| **DP (DDP/ZeRO/manual)** | Đồng bộ gradient | Gradient local | Gradient averaged |

| Module | Giao tiếp chính | Tần suất | Yêu cầu latency |
|--------|----------------|----------|----------------|
| **ShardFormer** | AllReduce (TP) | Mỗi layer, mỗi microbatch | Thấp (intra-node) |
| **PipelineStageManager** | P2P send/recv (PP) | Mỗi microbatch boundary | Trung bình |
| **DP** | AllReduce (ring) | 1 lần/step | Chấp nhận cao (cross-node) |

---

*File này cung cấp cái nhìn tổng quan để hiểu cách ba module nền tảng tương tác trong kiến trúc 3D Parallel của ColossalAI. Chi tiết cài đặt cụ thể của từng module có thể tham khảo source code tương ứng.*
