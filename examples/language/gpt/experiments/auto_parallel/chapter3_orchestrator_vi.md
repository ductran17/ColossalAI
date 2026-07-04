## 3.2.6 Điều phối tổng thể — Orchestrator (run_auto_hybrid_parallel.py)

### Vị trí và nhiệm vụ

File `run_auto_hybrid_parallel.py` là **điều phối viên tổng thể (orchestrator)** — nó điều phối toàn bộ pipeline từ đầu đến cuối: **đo profile → lập kế hoạch → huấn luyện**, đồng thời tự động thử nhiều cấu hình cụm (world_size) và cả hai chiều lưới (dp_outside) để tìm ra chiến lược tối ưu toàn cục.

Khác với `search.py` chỉ xử lý **một** `(world_size, dp_outside)`, orchestrator quyết định:
- Có nên thử dùng **ít GPU hơn** số GPU hiện có không? (subset search)
- Có nên thử cả **`dp_outside=True`** và **`dp_outside=False`** không?
- Nếu plan tốt nhất dùng ít GPU hơn, có nên **tự động relaunch** không?

---

### Ba pha chính

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Phase 1        │     │  Phase 2        │     │  Phase 3        │
│  Profile        │ ──→ │  Plan           │ ──→ │  Train          │
│  (~2 giây)      │     │  (<1 ms)        │     │  (nhiều phút)   │
│                 │     │                 │     │                 │
│ profile_cluster │     │ auto_plan()     │     │ HybridParallel  │
│ α, β, T_block   │     │ enumerate/prune/│     │ Plugin          │
│                 │     │ score/select    │     │ execute_pipeline│
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

#### Phase 1: Đo thông số cụm (Profiling)

Mọi rank gọi `profile_cluster()` cùng lúc:
- Đo `α_intra, β_intra` qua NCCL P2P giữa 2 rank cùng node
- Đo `α_cross, β_cross` qua P2P giữa 2 rank khác node
- Đo `T_block` và `T_block_with_microbatches` trên GPU cục bộ
- `all_reduce(MAX)` để mọi rank có cùng giá trị — slowest GPU quyết định

**Kết quả:** `ClusterProfile` đồng nhất trên toàn bộ cụm.

#### Phase 2: Lập kế hoạch (Planning)

Đây là pha phức tạp nhất. Orchestrator xây dựng **không gian tìm kiếm** và gọi `search.py` cho từng combo.

##### 2.1 Xây dựng không gian tìm kiếm

**a) Chọn các world_size cần thử:**

```python
if --fixed-world-size:
    prefix_ws = [world_size]           # Chỉ thử đúng số GPU hiện có
else:
    prefix_ws = [2, 4, 6]            # Thử mọi prefix: 2, 4, 6 (cho node_gpus=[2,2,2])
    prefix_nodes = [[2], [2,2], [2,2,2]]
```

Ví dụ: cụm `[2,2,2]` (6 GPU) → thử cả `world_size=2` (1 node), `world_size=4` (2 node), `world_size=6` (3 node).

Tại sao thử cả subset? Vì trên mạng Ethernet chậm, thêm node thứ 3 (cross-node thứ 2) có thể làm DP/PP chậm đến mức **dùng ít GPU hơn còn nhanh hơn**.

**b) Thử cả hai dp_outside:**

```python
for ws in prefix_ws:
    for dp_outside_flag in (True, False):
        res = auto_plan(world_size=ws, dp_outside=dp_outside_flag, ...)
```

→ Với 3 world_size × 2 dp_outside = **6 lần gọi** `auto_plan()`.

##### 2.2 So sánh toàn bộ combo và chọn winner toàn cục

Sau khi thử xong, orchestrator so sánh `best_cost` giữa tất cả combo:

```python
best_result = None
best_cost = float('inf')

for ws, dpo, res in all_results:
    if res is not None and res.cost.total < best_cost:
        best_cost = res.cost.total
        best_result = res
        best_ws = ws
        best_dpo = dpo
```

**Ví dụ kết quả** (cụm 6 GPU `[2,2,2]`):

| world_size | dp_outside | Best plan | T_est (ms) | Nhận xét |
|-----------|-----------|-----------|-----------|----------|
| 2 | True | pp=1,tp=2,dp=1 | 350.6 | Chỉ 1 node, không cross-node |
| 2 | False | pp=1,tp=2,dp=1 | 350.6 | dp=1 → dp_outside không ảnh hưởng |
| 4 | True | pp=4,tp=1,dp=1 | 263.0 | Pure PP, nhanh hơn world_size=2 |
| 4 | False | pp=4,tp=1,dp=1 | 263.0 | dp=1 → không ảnh hưởng |
| 6 | True | pp=6,tp=1,dp=1 | **203.2** | **Winner toàn cục** |
| 6 | False | pp=6,tp=1,dp=1 | 203.2 | dp=1 → không ảnh hưởng |

→ Orchestrator chọn `world_size=6, dp_outside=True, pp=6,tp=1,dp=1`.

##### 2.3 Báo cáo so sánh (Comparison Report)

Orchestrator viết file `comparison_{world_size}gpu_{layers}L_{hidden}H.txt` gồm 2 phần:

**Section 1:** Toàn bộ candidate plans cho **mỗi combo** `(world_size, dp_outside)` — giúp người dùng thấy tại sao một plan thắng/thua.

**Section 2:** Bảng tóm tắt best plan per combo + đánh dấu `<<< BEST` cho winner toàn cục.

##### 2.4 Cơ chế tự động relaunch (Auto-relaunch)

Nếu winner dùng **ít GPU hơn** số GPU hiện có (ví dụ: winner ở world_size=4 nhưng cụm có 6 GPU), orchestrator:
1. In cảnh báo
2. Ghi file `RELAUNCH.txt` chứa số node tối ưu
3. `launch_nodes.sh` (bên ngoài) đọc `RELAUNCH.txt` và tự động relaunch với ít node hơn

Nếu không relaunch (hoặc relaunch không thành công), orchestrator **tái chọn plan** cho `world_size` đầy đủ để training vẫn chạy được.

---

### Phase 3: Huấn luyện (Training)

Sau khi có plan tốt nhất, orchestrator khởi tạo:

```python
plugin = HybridParallelPlugin(
    pp_size=pp,
    tp_size=tp,
    num_microbatches=args.microbatches,
    dp_outside=training_dp_outside,
    ...
)
```

Và chạy training loop. Hai điểm đáng chú ý:

1. **pp=1 branch:** Nếu `pp=1` (không pipeline), dùng forward/backward chuẩn thay vì `execute_pipeline()` (vì `execute_pipeline()` crash khi pp=1).

2. **Timing:** Dùng `torch.cuda.synchronize()` + `time.perf_counter()` để đo wall-clock thời gian một step chính xác.

---

### Luồng dữ liệu tổng thể

```
Người dùng chạy:
    ./launch_nodes.sh node18 node19 node15 --auto --layers 24 --hidden 1024 ...

↓

Orchestrator trên mỗi node:
    1. Auto-detect node_gpus từ LOCAL_RANK
    2. profile_cluster() → ClusterProfile
    3. Vòng lặp: for ws in [2,4,6]:
           for dpo in [True, False]:
               auto_plan(ws, dpo) → PlanResult
    4. So sánh tất cả PlanResult → chọn winner
    5. Viết comparison_*.txt
    6. (Nếu winner dùng ít GPU) → ghi RELAUNCH.txt
    7. Khởi tạo HybridParallelPlugin với winner plan
    8. Training loop (forward/backward/optimize)
    9. So sánh estimated vs actual → viết JSON report
```

---

### Mối quan hệ giữa orchestrator và các module con

| Thành phần | Nhiệm vụ | Gọi bởi |
|-----------|---------|---------|
| **Orchestrator** (`run_auto_hybrid_parallel.py`) | Quyết định thử bao nhiêu combo, so sánh kết quả, điều phối training | Người dùng |
| **Profiler** (`profiler.py`) | Đo α, β, T_block trên hardware thực | Orchestrator Phase 1 |
| **Planner** (`search.py`) | Với 1 `(world_size, dp_outside)`, enumerate/prune/score/select | Orchestrator Phase 2 |
| **Topology** (`topology.py`) | Phân loại intra/cross-node cho 1 plan | Planner (qua auto_plan) |
| **Cost Model** (`cost_model.py`) | Tính T_total cho 1 plan | Planner (qua auto_plan) |
| **HybridParallelPlugin** | Triển khai TP/PP/DP thực tế | Orchestrator Phase 3 |

---

### Ví dụ minh họa: Orchestrator với cụm 6 GPU

**Input:** `node_gpus = [2, 2, 2]`, model `24L/1024H`, batch=16, M=8

**Phase 1:** `profile_cluster()` đo được:
- `T_block = 1.96 ms` (A6000/L40S/L40 trộn lẫn, MAX-reduced)
- `α_intra = 66 µs`, `β_intra = 0.05 ns/B` (PCIe ~20 GB/s)
- `α_cross = 84 µs`, `β_cross = 0.38 ns/B` (Ethernet ~2.6 GB/s)

**Phase 2:** Thử 6 combo:

```
world_size=2, dp_outside=True   → best: (1,2,1) @ 350.6 ms
world_size=2, dp_outside=False  → best: (1,2,1) @ 350.6 ms
world_size=4, dp_outside=True   → best: (4,1,1) @ 263.0 ms
world_size=4, dp_outside=False  → best: (4,1,1) @ 263.0 ms
world_size=6, dp_outside=True   → best: (6,1,1) @ 203.2 ms  <<< BEST
world_size=6, dp_outside=False  → best: (6,1,1) @ 203.2 ms
```

**Quyết định:** Winner là `world_size=6, pp=6, tp=1, dp=1, dp_outside=True`.

**Phase 3:** Khởi tạo plugin và train 20 step. Kết quả JSON:
- Estimated: 203.2 ms
- Actual: 208.5 ms
- Ratio: 0.98 → cost model chính xác

---

### Tại sao không gộp orchestrator vào search.py?

Có thể gộp, nhưng tách ra mang lại lợi ích:

1. **Tái sử dụng:** `search.py` có thể dùng độc lập (ví dụ: script so sánh chiến lược thủ công không cần training).
2. **Kiểm thử:** `search.py` pure Python, test nhanh không cần cluster.
3. **Tính mô-đun:** Thay đổi cách tìm kiếm world_size (ví dụ: thử cả world_size lẻ) chỉ cần sửa orchestrator, không đụng planner.
4. **Nhiều chế độ:** Orchestrator hỗ trợ cả `--manual-pp/--manual-tp` (bypass search hoàn toàn) và `--fixed-world-size` (tắt subset search).

---

### Tài liệu tham khảo

- `run_auto_hybrid_parallel.py` — source code orchestrator.
- `search.py` — module planner.
- `launch_nodes.sh` — script khởi động node, đọc `RELAUNCH.txt` để auto-relaunch.
