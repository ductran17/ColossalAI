## 3.2.5 Module tìm kiếm chiến lược — Planner (search.py)

### Vị trí và nhiệm vụ

Module `search.py` đóng vai trò **bộ lập kế hoạch (planner)** — nhận đầu vào là cấu hình mô hình, profile cụm, và topology, rồi trả về chiến lược song song $(pp, tp, dp)$ tốt nhất. Đây là phần "cốt lõi" của giai đoạn lập kế hoạch (Phase 2) trong pipeline tổng thể.

**Phạm vi:** `search.py` chỉ xử lý **một cặp `(world_size, dp_outside)` cụ thể** trong một lần gọi. Việc thử nhiều `world_size` và cả hai `dp_outside` được thực hiện bên ngoài, trong file `run_auto_hybrid_parallel.py` (orchestrator).

---

### Quy trình làm việc của planner

Quy trình gồm 4 bước: **enumerate → prune → score → select**.

#### Bước 1: Liệt kê mọi ứng viên (Enumerate)

Hàm `_all_candidates(world_size)` sinh ra **mọi bộ ba** $(pp, tp, dp)$ thỏa mãn:

$$pp \times tp \times dp = \text{world_size}$$

Ví dụ với `world_size = 8`:
```
(1,1,8), (1,2,4), (1,4,2), (1,8,1),
(2,1,4), (2,2,2), (2,4,1),
(4,1,2), (4,2,1),
(8,1,1)
```

→ Tổng cộng **10 chiến lược** cần đánh giá.

#### Bước 2: Loại bỏ ứng viên bất khả thi (Prune)

Không phải mọi chiến lược đều khả thi. `search.py` áp dụng 4 quy tắc loại bỏ:

| Quy tắc | Điều kiện loại | Lý do |
|---------|---------------|-------|
| **TP cross-node** | `tp > min_gpus_per_node` | Nếu TP nhóm vượt quá số GPU trên một node, ít nhất một nhóm TP sẽ nằm trên hai node (Ethernet chậm). TP AllReduce mỗi lớp → tốc độ giảm ~10× |
| **Layers không chia hết** | `layers % pp != 0` | Mỗi stage pipeline phải có số lớp nguyên. Không thể chia 24 lớp cho pp=5 |
| **Batch không chia hết** | `batch % microbatches != 0` | Kích thước microbatch phải là số nguyên |
| **Vượt bộ nhớ GPU** | `_fits_in_memory(...) == False` | Ước lượng bộ nhớ cần thiết (param + grad + Adam state + activations) vượt quá ngân sách |

Các chiến lược bị loại được ghi lại trong `pruned_table` kèm lý do, giúp người dùng hiểu tại sao một chiến lược không xuất hiện trong bảng điểm.

**Ví dụ thực tế** (cụm 6 GPU `[2,2,2]`, `min_gpus_per_node=2`):
```
pp=1 tp=3 dp=2  → tp=3 > 2  → cross-node TP  (loại)
pp=1 tp=6 dp=1  → tp=6 > 2  → cross-node TP  (loại)
pp=2 tp=3 dp=1  → tp=3 > 2  → cross-node TP  (loại)
```

Từ 10 ứng viên, chỉ còn **7 ứng viên khả thi**.

#### Bước 3: Tính điểm (Score)

Với mỗi ứng viên còn lại, planner gọi hai module con:
1. `classify_comms(node_gpus, pp, tp, dp, dp_outside)` → xác định nhóm giao tiếp nào là intra/cross-node
2. `estimate_step_time(cfg, pp, tp, dp, profile, topology, microbatches)` → tính `T_total` theo 7 thành phần của cost model

Kết quả là một bảng gồm các cột:
```
plan | total | compute | bubble | TP comm | PP comm | DP comm | step OH | exec OH | tp_intra | pp_intra | dp_intra | dp_outside
```

#### Bước 4: Chọn chiến lược tốt nhất (Select)

Planner chọn ứng viên có `T_total` nhỏ nhất:

```python
best_row = min(scored, key=lambda r: r["cost"].total)
```

Trả về `PlanResult` chứa:
- `pp, tp, dp` của chiến lược thắng
- `cost`: breakdown chi tiết 7 thành phần
- `topology`: phân loại intra/cross-node
- `scored_table`: toàn bộ bảng điểm (để người dùng so sánh)
- `pruned_table`: danh sách bị loại (để hiểu tại sao)

---

### Tại sao không thử luôn cả dp_outside trong search.py?

`search.py` được thiết kế **đơn nhiệm** (single-responsibility):
- Input: 1 world_size + 1 dp_outside
- Output: 1 PlanResult

Lý do tách biệt:
1. **Tính độc lập:** `search.py` không cần biết toàn bộ cụm có bao nhiêu GPU. Nó chỉ cần biết node_gpus tương ứng với world_size đang xét.
2. **Tính tái sử dụng:** Có thể gọi `auto_plan()` từ script khác để so sánh riêng một cấu hình.
3. **Dễ kiểm thử:** `search.py` pure Python, không cần torch.distributed → unit-test trên laptop.

Việc lặp qua nhiều world_size và cả hai dp_outside là **nhiệm vụ của orchestrator** (`run_auto_hybrid_parallel.py`), vì nó cần biết tổng thể cụm để quyết định:
- Có nên dùng ít GPU hơn không? (subset search)
- Có nên tự động relaunch không?

---

### Ví dụ minh họa: Planner với world_size=4, dp_outside=True

Cụm: `[2, 2]` GPU. Model: `layers=24, hidden=1024, batch=16, microbatches=8`.

**Các ứng viên (pp×tp×dp = 4):**
```
(1,1,4), (1,2,2), (2,1,2), (2,2,1), (4,1,1)
```

**Sau prune:**
- `(1,2,2)`: tp=2 ≤ min_gpus=2 ✓ (intra-node)
- `(2,2,1)`: tp=2 ≤ 2 ✓
- Tất cả đều thỏa mãn layers%pp=0 và batch%M=0

**Bảng điểm (ước lượng):**

| Plan | T_total | T_compute | T_bubble | T_DP | Ghi chú |
|------|---------|-----------|----------|------|---------|
| (1,1,4) | 766 ms | 376 ms | 0 | 321 ms | Pure DP, cross-node AllReduce |
| (1,2,2) | 452 ms | 188 ms | 0 | 101 ms | TP+DP, DP cross-node |
| (2,1,2) | 538 ms | 370 ms | 41 ms | 101 ms | PP+DP, DP cross-node |
| (2,2,1) | 282 ms | 185 ms | 21 ms | 0 | PP+TP, no DP |
| (4,1,1) | **263 ms** | 185 ms | 50 ms | 0 | **Pure PP, winner** |

→ Planner chọn **pp=4, tp=1, dp=1** vì T_total nhỏ nhất.

Nếu cùng world_size=4 nhưng `dp_outside=False`, plan `(2,1,2)` sẽ có T_total khác (PP cross-node thay vì DP cross-node), có thể thay đổi thứ hạng.

---

### Tài liệu tham khảo

- `search.py` — source code planner.
- `topology.py` — module phân loại topology.
- `cost_model.py` — module tính toán chi phí 7 thành phần.
