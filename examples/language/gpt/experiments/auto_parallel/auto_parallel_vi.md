# Auto-Parallel trong ColossalAI

## Auto-Parallel là gì?

Khi huấn luyện các mô hình ngôn ngữ lớn (LLM) như GPT, Llama, một GPU đơn lẻ thường không đủ bộ nhớ hoặc tốc độ. Giải pháp là chia nhỏ mô hình ra và chạy song song trên nhiều GPU.

**Auto-Parallel** là tính năng của ColossalAI giúp tự động tìm ra cách chia mô hình tối ưu nhất — người dùng **không cần tự tay** chỉ định cách phân chia từng tầng. Hệ thống sẽ phân tích mô hình, tính toán chi phí, rồi đưa ra kế hoạch song song hóa tốt nhất.

---

## Đầu vào và Đầu ra

### Đầu vào

```python
from colossalai.auto_parallel.tensor_shard import autoparallelize

gm, solution = autoparallelize(
    model,       # mô hình PyTorch bình thường (nn.Module)
    meta_args,   # dict mô tả shape của input, không cần dữ liệu thật
    return_solution=True
)
```

| Tham số | Ý nghĩa |
|---|---|
| `model` | Mô hình PyTorch (ví dụ: GPT2LMHeadModel) |
| `meta_args` | Dictionary chứa các tensor "ảo" (chỉ có shape, không có giá trị thật) |
| `solver_preference` | Ưu tiên chiến lược: `"standard"`, `"tp"` (tensor parallel), `"dp"` (data parallel) |
| `memory_budget` | Giới hạn bộ nhớ GPU (byte), mặc định là không giới hạn |

**Ví dụ `meta_args`:**
```python
meta_args = {
    "input_ids":      torch.zeros(16, 1024, dtype=torch.long, device="meta"),
    "attention_mask": torch.zeros(16, 1024, dtype=torch.long, device="meta"),
}
# device="meta" → tensor không chiếm bộ nhớ thật, chỉ dùng để phân tích shape
```

### Đầu ra

| Đầu ra | Kiểu | Ý nghĩa |
|---|---|---|
| `gm` | `ModuleWrapper` | Mô hình đã được chia nhỏ, sẵn sàng chạy phân tán |
| `solution` | `List[str]` | Danh sách chiến lược được chọn cho từng node trong đồ thị |

**Ví dụ `solution`:**
```
wte        Replica Placeholder
h_0_ln_1   RR = RR            (replicated)
h_0_c_attn RS1 = RR x RS1     (weight sharded trên TP axis)
h_0_c_proj S0R = S0R x RR     (input sharded trên DP axis)
lm_head    RS1 = RR x RS1
...
```
Ký hiệu: `S` = sharded (chia nhỏ), `R` = replicated (nhân bản), số subscript = trục của device mesh.

---

## Luồng hoạt động tổng quan

```
        Model + meta_args
              │
              ▼
    ┌─────────────────────┐
    │    0. DEVICE MESH   │  Xây dựng bản đồ GPU + đo băng thông
    └─────────┬───────────┘
              │  DeviceMesh [DP × TP] + alpha/beta mỗi trục
              ▼
    ┌─────────────────────┐
    │    1. ANALYZER      │  Phân tích cấu trúc mô hình
    └─────────┬───────────┘
              │  Đồ thị tính toán (FX Graph) + thông tin shape mỗi node
              ▼
    ┌─────────────────────┐
    │    2. GENERATOR     │  Liệt kê tất cả chiến lược song song có thể
    └─────────┬───────────┘
              │  Mỗi node có danh sách chiến lược + chi phí ước tính
              ▼
    ┌─────────────────────┐
    │    3. SOLVER        │  Chọn chiến lược tối ưu (bài toán ILP)
    └─────────┬───────────┘
              │  Kết quả: 1 chiến lược cho mỗi node
              ▼
    ┌─────────────────────┐
    │  4. TRANSFORMATION  │  Áp dụng: chia trọng số, chèn comm ops
    └─────────┬───────────┘
              │
              ▼
       ModuleWrapper
    (mô hình phân tán, dùng như nn.Module bình thường)
```

---

## Các thành phần chính

### 0. DeviceMesh — Bản đồ GPU

**Mục đích:** Trước khi làm bất cứ điều gì, hệ thống cần biết mình đang có bao nhiêu GPU, chúng kết nối với nhau như thế nào, và tốc độ truyền dữ liệu giữa các GPU là bao nhiêu. `DeviceMesh` là đối tượng lưu trữ toàn bộ thông tin đó.

**Khái niệm Physical vs Logical Mesh:**

```
Physical mesh (thực tế):  [GPU 0, GPU 1, GPU 2, GPU 3]  ← 4 GPU, mỗi GPU có global rank được đánh số bằng global rank do torch.distributed cấp:

Logical mesh (2D):        [[0, 1],
                           [2, 3]]
                           ↑    ↑
                         axis 0  axis 1
                          (DP)   (TP)
```

Từ 4 GPU vật lý, hệ thống tạo ra một **lưới 2 chiều** (2×2):
- **Axis 0 (hàng)** → Data Parallel (DP): GPU 0 và GPU 2 cùng hàng, chứa bản sao giống nhau
- **Axis 1 (cột)** → Tensor Parallel (TP): GPU 0 và GPU 1 cùng cột, chia nhỏ tensor

**Cách DeviceMesh được tạo:**

```python
# Tự động — hệ thống tự tìm cấu hình tốt nhất
device_mesh = initialize_device_mesh()
# → AlphaBetaProfiler đo băng thông thật giữa các GPU
# → Tự chọn logical_mesh_shape phù hợp nhất

# Thủ công — người dùng chỉ định
device_mesh = initialize_device_mesh(logical_mesh_shape=(2, 2))
```

**AlphaBetaProfiler — đo tốc độ mạng GPU:**

Trước khi xây DeviceMesh, hệ thống chạy các phép `all-reduce` thử nghiệm giữa mọi cặp GPU để đo:

| Tham số | Ý nghĩa | Đơn vị |
|---|---|---|
| `alpha` | Latency — độ trễ cố định mỗi lần gửi | giây |
| `beta` | Bandwidth — thời gian truyền mỗi byte | giây/byte |

Chi phí ước tính cho một `all-reduce` trên `N` bytes qua `D` GPU:
```
cost = alpha + beta × 2(D-1)/D × N
```

Các GPU cùng NVLink (cùng node) sẽ có `alpha` nhỏ và `beta` nhỏ → ưu tiên đặt TP group ở đây vì TP cần giao tiếp nhiều hơn DP.

**Process Group — nhóm giao tiếp:**

Sau khi có logical mesh, DeviceMesh tạo các **process group** cho từng trục:

```
logical_mesh = [[0, 1],
                [2, 3]]

Process groups cho axis 0 (DP):  {0,2} và {1,3}
Process groups cho axis 1 (TP):  {0,1} và {2,3}
```

Mỗi collective operation (all-reduce, all-gather...) sẽ chỉ chạy trong một process group nhất định, không phải trên toàn bộ 4 GPU — giúp giảm chi phí giao tiếp.

**DeviceMesh được dùng ở đâu trong hệ thống?**

| Ai dùng | Mục đích |
|---|---|
| **Generator** | Dùng `mesh.all_reduce_cost(bytes, axis)` để ước tính chi phí từng chiến lược |
| **Solver** | Nhận chi phí từ Generator để so sánh các chiến lược |
| **Transformation** | Dùng process group để tạo collective ops thật khi chạy |
| **ShardingSpec** | Gắn với mỗi tensor để biết tensor đó được chia trên trục nào của mesh |

---

### 1. Analyzer — Phân tích mô hình

**Mục đích:** Chuyển mô hình PyTorch thành một đồ thị tính toán tĩnh, trong đó mỗi phép toán là một node và mỗi node biết shape của tensor đầu vào/đầu ra.

**Cách hoạt động:**

```
Model.forward()
      │
      │ ColoTracer.trace()   ← "chạy thử" mô hình không có dữ liệu thật
      ▼
  FX Graph  (DAG các phép toán)
      │
      │ ShapeProp            ← lan truyền shape qua từng node
      ▼
  FX Graph với đầy đủ thông tin shape
```

- **`ColoTracer`**: Dùng kỹ thuật *symbolic tracing* của PyTorch FX. Thay vì chạy thật, nó theo dõi tất cả các phép toán được gọi và tạo ra một đồ thị. Nhờ `device="meta"`, tensor không chiếm RAM/VRAM.
- **`ShapeProp`**: Chạy lại đồ thị với `MetaTensor` (tensor ảo) để biết shape đầu ra của mỗi node.

**Tại sao cần Analyzer?**
Để Generator và Solver làm việc, chúng cần biết chính xác shape của mọi tensor trong mô hình mà không cần chạy thật với dữ liệu.

---

### 2. Generator — Sinh chiến lược song song

**Mục đích:** Với mỗi phép toán trong đồ thị, liệt kê tất cả cách có thể để phân chia nó trên nhiều GPU, kèm theo ước tính chi phí.

**Cách hoạt động:**

Mỗi loại phép toán có một **Handler** riêng:

| Phép toán | Handler | Chiến lược ví dụ |
|---|---|---|
| Linear / Matmul | `LinearHandler` | Column-parallel, Row-parallel, Replicated |
| Attention (BMM) | `BmmHandler` | Head-parallel, Batch-parallel |
| Embedding | `EmbeddingHandler` | Vocab-parallel, Replicated |
| LayerNorm | `LayerNormHandler` | Batch-sharded, Replicated |
| Element-wise (+, relu...) | `BinaryElementwiseHandler` | Theo input |

Mỗi chiến lược được biểu diễn bằng **ShardingSpec** — mô tả chiều nào của tensor được chia:

```
"RS1 = RR x RS1"
 │         │  │
 │         │  └─ weight chia theo chiều cột trên TP axis
 │         └──── input không chia (replicated)
 └──────────── output chia theo chiều cuối trên TP axis
```

Mỗi chiến lược kèm theo ba loại chi phí:
- **Compute cost**: số FLOPs ước tính
- **Communication cost**: chi phí all-reduce / all-gather
- **Memory cost**: bộ nhớ activation + tham số

Handler còn tính **resharding cost** — chi phí khi node trước dùng chiến lược A nhưng node sau cần chiến lược B (phải chạy thêm một collective operation để chuyển đổi).

---

### 3. Solver — Tìm chiến lược tối ưu

**Mục đích:** Từ hàng trăm/nghìn tổ hợp chiến lược có thể, tìm ra tổ hợp tốt nhất cho toàn bộ mô hình.

**Cách hoạt động:**

Solver xây dựng một **bài toán Integer Linear Programming (ILP)**:

```
Biến quyết định:
  s[i] = 1 nếu node i chọn chiến lược s, = 0 nếu không

Tối thiểu hóa:
  Tổng chi phí compute + communication của tất cả node
  + Tổng chi phí resharding giữa các node liền kề

Ràng buộc:
  - Mỗi node phải chọn đúng 1 chiến lược
  - Tổng bộ nhớ sử dụng ≤ memory_budget (nếu có)
  - Các block transformer lặp lại dùng cùng chiến lược
```

Bài toán được giải bằng thư viện **PuLP** với bộ giải **CBC** (coin-or).

**Tối ưu hóa thêm:**
- Các node đơn giản (relu, add, reshape) được **gộp** vào node liền kề — không cần biến quyết định riêng, giảm kích thước bài toán ILP.
- Các transformer block giống hệt nhau (layer 1, layer 2, ...) dùng **chung biến** → giảm số biến đáng kể.

**Kết quả:** Một list số nguyên, mỗi số là index của chiến lược được chọn cho node tương ứng.

---

### 4. Transformation — Áp dụng kết quả

**Mục đích:** Dựa vào kết quả của Solver, biến đổi mô hình thực tế: cắt trọng số, chèn các lệnh giao tiếp GPU.

**Bốn bước biến đổi:**

#### Bước 4a: Gán chiến lược vào đồ thị
Đánh dấu `best_strategy` và `sharding_spec` lên mỗi node. Tạo ba dictionary sẽ được dùng lúc chạy thật:
- `sharding_spec_convert_dict`: spec mỗi node phải có khi truyền cho node tiếp theo
- `origin_spec_dict`: spec đầu ra thực của mỗi node
- `comm_actions_dict`: collective operation cần chạy ở mỗi node

#### Bước 4b: Cắt trọng số (`module_params_sharding_pass`)
Mỗi GPU chỉ giữ **phần trọng số của mình**:
```
# 2 GPU, column-parallel Linear (hidden_dim=1024)
GPU 0: weight = [512, 1024]   ← nửa đầu các cột
GPU 1: weight = [512, 1024]   ← nửa sau các cột
```
Đồng thời đăng ký **gradient hook** để tự động đồng bộ gradient khi backward.

#### Bước 4c: Chèn resharding nodes (`_shape_consistency_apply`)
Khi node A xuất tensor với spec `S0` nhưng node B cần `R`, tự động chèn một node `runtime_apply` giữa chúng. Node này sẽ gọi `all-gather` lúc chạy thật.

#### Bước 4d: Chèn collective nodes (`_comm_spec_apply`)
Chèn `runtime_comm_spec_apply` trước hoặc sau các phép tính cần collective:
```
# Row-parallel linear cần all-reduce sau khi tính xong
output = linear(input, weight_shard)
output = runtime_comm_spec_apply(output, ...)  ← all-reduce giữa TP ranks
```

Cuối cùng: `gm.recompile()` — tái tạo code Python từ đồ thị đã chỉnh sửa.

---

## ModuleWrapper — Đầu ra cuối cùng

```
ModuleWrapper
├── self.module             ← ColoGraphModule đã recompile
│     ├── weights           ← đã được cắt nhỏ cho từng GPU
│     └── graph             ← chứa các node runtime_apply và runtime_comm_spec_apply
│
├── self.sharding_spec_dict ← dùng khi forward để biết resharding nào cần làm
├── self.origin_spec_dict   ← spec gốc của từng node
└── self.comm_actions_dict  ← collective nào cần chạy ở đâu
```

Mỗi lần gọi `gm(input_ids, attention_mask)`, `ModuleWrapper` tự động truyền ba dictionary này vào, đồng nghĩa với việc tất cả collective operations được thực thi đúng chỗ mà không cần người dùng làm gì thêm.

---

## Graph Node được phân phối xuống GPU như thế nào?

Đây là câu hỏi quan trọng: **node trong đồ thị tính toán không được "gán" cho một GPU cụ thể**. Thay vào đó, **tất cả GPU đều chạy tất cả các node**, nhưng mỗi GPU chỉ xử lý **phần tensor của mình**.

### Cơ chế SPMD (Single Program, Multiple Data)

Auto-Parallel của ColossalAI dùng mô hình **SPMD** — mọi GPU chạy cùng một chương trình (cùng graph), nhưng với dữ liệu khác nhau (phần tensor được chia cho GPU đó).

```
Cùng graph forward() chạy trên tất cả GPU
         │
         GPU 0                GPU 1                GPU 2                GPU 3
         rank=0               rank=1               rank=2               rank=3
         mesh pos: [0,0]      mesh pos: [0,1]      mesh pos: [1,0]      mesh pos: [1,1]
              │                    │                    │                    │
         node: wte           node: wte            node: wte            node: wte
         weight[full]        weight[full]         weight[full]         weight[full]
              │                    │                    │                    │
         node: c_attn        node: c_attn         node: c_attn         node: c_attn
         weight[:,0:H/2]     weight[:,H/2:H]      weight[:,0:H/2]      weight[:,H/2:H]
              │                    │                    │                    │
         node: all-reduce ───────────────────────────────────────────────────
```

### ShardingSpec — "Mỗi GPU đang giữ phần nào?"

Mỗi tensor trong graph được gắn một **ShardingSpec** — mô tả chính xác GPU đang cầm phần nào của tensor đó:

```
Tensor gốc: hidden_states [Batch=16, Seq=1024, Hidden=4096]

ShardingSpec {S0: [0]}  →  chia theo batch trên axis 0 của mesh (DP)
  GPU 0 (mesh[0,0]): hidden_states[0:8,  :, :]   ← batch 0..7
  GPU 1 (mesh[0,1]): hidden_states[0:8,  :, :]   ← batch 0..7  (cùng DP group với GPU 0)
  GPU 2 (mesh[1,0]): hidden_states[8:16, :, :]   ← batch 8..15
  GPU 3 (mesh[1,1]): hidden_states[8:16, :, :]   ← batch 8..15

ShardingSpec {S1: [1]}  →  chia theo hidden dim trên axis 1 của mesh (TP)
  GPU 0 (mesh[0,0]): hidden_states[:, :, 0:2048]     ← nửa đầu hidden
  GPU 1 (mesh[0,1]): hidden_states[:, :, 2048:4096]  ← nửa sau hidden
  GPU 2 (mesh[1,0]): hidden_states[:, :, 0:2048]     ← nửa đầu hidden
  GPU 3 (mesh[1,1]): hidden_states[:, :, 2048:4096]  ← nửa sau hidden

ShardingSpec {R}  →  replicated, mọi GPU giữ tensor đầy đủ
  GPU 0, 1, 2, 3: hidden_states[:, :, :]  ← giống nhau hoàn toàn
```

### Ví dụ cụ thể: 4 GPU, mesh [2×2], GPT-2

```
logical_mesh = [[GPU0, GPU1],   axis 1 = TP (tensor parallel)
                [GPU2, GPU3]]
                  axis 0 = DP (data parallel)

Sau autoparallelize(), mỗi GPU sở hữu:

┌──────────────────────────────────────────────────────────────┐
│  Tầng         │ GPU 0 (DP=0,TP=0) │ GPU 1 (DP=0,TP=1)       │
├──────────────────────────────────────────────────────────────┤
│  wte          │ weight [50257, H]  │ weight [50257, H]        │ ← replicated
│  c_attn       │ weight [3H, H/2]   │ weight [3H, H/2]         │ ← col-parallel (TP)
│  c_proj       │ weight [H/2, H]    │ weight [H/2, H]          │ ← row-parallel (TP)
│  lm_head      │ weight [H, V/2]    │ weight [H, V/2]          │ ← vocab-parallel (TP)
│  input batch  │ batch[0:8]         │ batch[0:8]               │ ← DP: GPU0,1 cùng batch
├──────────────────────────────────────────────────────────────┤
│  Tầng         │ GPU 2 (DP=1,TP=0) │ GPU 3 (DP=1,TP=1)       │
├──────────────────────────────────────────────────────────────┤
│  wte          │ weight [50257, H]  │ weight [50257, H]        │ ← replicated
│  c_attn       │ weight [3H, H/2]   │ weight [3H, H/2]         │ ← cùng shard với GPU0
│  c_proj       │ weight [H/2, H]    │ weight [H/2, H]          │ ← cùng shard với GPU1
│  lm_head      │ weight [H, V/2]    │ weight [H, V/2]          │
│  input batch  │ batch[8:16]        │ batch[8:16]              │ ← DP: GPU2,3 khác batch
└──────────────────────────────────────────────────────────────┘
```

### Khi nào cần giao tiếp giữa GPU?

Giao tiếp xảy ra khi **ShardingSpec của output node A ≠ ShardingSpec mà node B cần**. Có hai trường hợp:

**Trường hợp 1 — Resharding (tái phân phối tensor):**
```
Node A xuất: hidden [S0, R, R]   ← chia theo batch
Node B cần:  hidden [R,  R, R]   ← cần tensor đầy đủ

→ Chèn all-gather trên axis 0 (DP group) giữa A và B
```

**Trường hợp 2 — Collective cố định của chiến lược:**
```
Column-parallel Linear:
  Mỗi GPU tính output cục bộ (partial sum)
  → Cần all-reduce trên axis 1 (TP group) sau khi tính xong
  → Kết quả mỗi GPU giống nhau (replicated)

Row-parallel Linear:
  Mỗi GPU tính phần của mình
  → Cần all-reduce trên axis 1 để cộng các partial sums
```

**Tóm lại:** Node không được "gán cứng" cho GPU nào. Tất cả GPU chạy mọi node, nhưng mỗi GPU tính trên **phần tensor riêng** của mình. DeviceMesh xác định **phần đó là phần nào** (qua axis 0 hay axis 1), còn ShardingSpec gắn trực tiếp lên tensor để mọi collective operation biết phải giao tiếp với GPU nào.

---

## Quá trình huấn luyện phân tán

Sau khi có `ModuleWrapper`, training loop **không khác gì** huấn luyện thông thường:

```python
optimizer = torch.optim.Adam(gm.parameters())
criterion = GPTLMLoss()

for step in range(num_steps):
    input_ids      = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN)).cuda()
    attention_mask = torch.ones(BATCH, SEQ_LEN).cuda()

    # Forward — mỗi GPU chạy phần của mình, collective tự động khi cần
    logits = gm(input_ids, attention_mask)

    # Loss — logits đã được all-gather về full vocab trước khi ra khỏi gm
    loss = criterion(logits, input_ids)

    # Backward — gradient hook tự all-reduce qua TP/DP groups
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```

**Điều xảy ra bên trong mỗi forward pass:**

```
Tất cả GPU chạy đồng thời cùng input
         │
         ├─ Embedding: replicated → mỗi GPU tính giống nhau
         │
         ├─ Attention QKV: column-parallel
         │    → GPU 0 tính head 0..7, GPU 1 tính head 8..15
         │    → all-gather kết quả ở cuối block
         │
         ├─ MLP: column-parallel + row-parallel
         │    → GPU 0 và GPU 1 tính các phần khác nhau
         │    → all-reduce ở cuối để tổng hợp
         │
         └─ LM Head: vocab-parallel
              → mỗi GPU tính logit cho 1 phần vocab
              → all-gather để ra logit đầy đủ
         │
         ▼
     loss.backward()
         │
         └─ gradient hook: all-reduce grad qua DP group
              → đồng bộ gradient giữa các bản sao DP
```

---

## So sánh với ShardFormer

ColossalAI có hai hệ thống song song hóa khác nhau:

| | Auto-Parallel | ShardFormer |
|---|---|---|
| **Cách chọn chiến lược** | Tự động (ILP solver) | Thủ công (viết Policy) |
| **Dễ dùng** | Cao — chỉ cần gọi `autoparallelize()` | Thấp hơn — phải biết về parallelism |
| **Hỗ trợ mô hình** | Mọi mô hình traceable | Chỉ các mô hình có sẵn policy |
| **Pipeline Parallel** | Không hỗ trợ | Có hỗ trợ |
| **Sequence Parallel** | Không hỗ trợ | Có hỗ trợ |
| **Tối ưu kernel** | Không | Có (Flash Attention, Fused LayerNorm) |
| **Dùng trong production** | Nghiên cứu / thử nghiệm | Production (LLaMA, Qwen, DeepSeek...) |

**Kết luận:** Auto-Parallel phù hợp khi bạn có mô hình mới chưa có policy, hoặc muốn khám phá chiến lược tối ưu tự động. ShardFormer phù hợp hơn cho production với các mô hình phổ biến.

---

## Tóm tắt

```
┌──────────────────────────────────────────────────────────────┐
│                     AUTO-PARALLEL                            │
│                                                              │
│  ĐẦU VÀO: model (nn.Module) + meta_args (shape của input)   │
│                                                              │
│  LUỒNG XỬ LÝ:                                               │
│  1. Analyzer   → đồ thị tính toán + shape mọi tensor        │
│  2. Generator  → mọi chiến lược có thể + chi phí            │
│  3. Solver     → chiến lược tối ưu (ILP)                    │
│  4. Transform  → cắt trọng số + chèn comm ops               │
│                                                              │
│  ĐẦU RA: ModuleWrapper (dùng như nn.Module bình thường)      │
│          → trọng số đã chia theo GPU                         │
│          → collective ops chạy tự động trong forward/backward│
└──────────────────────────────────────────────────────────────┘
```
