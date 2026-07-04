## 3.3 Triển khai thực thi với HybridParallelPlugin

> **Phân loại nguồn gốc:** Toàn bộ nội dung trong mục 3.3 mô tả các thành phần **có sẵn trong framework ColossalAI**. Các module tự phát triển của luận văn (profiler, cost model, search, orchestrator) chỉ **tương tác** với HybridParallelPlugin thông qua giao diện công khai của nó, không sửa đổi core code bên trong.

---

### 3.3.1 Vai trò của HybridParallelPlugin trong hệ thống

**Mục đích.**

`HybridParallelPlugin` là một plugin (mô-đun mở rộng) trong hệ thống `Booster` của ColossalAI. Nó đóng vai trò **lớp trừu tượng hóa (abstraction layer)** cho việc triển khai đồng thờicả ba chiến lược song song: Tensor Parallel (TP), Pipeline Parallel (PP) và Data Parallel (DP). Plugin nhận đầu vào là một mô hình PyTorch thuần túy (ví dụ: `GPT2LMHeadModel` từ Hugging Face), tự động phân tách nó theo các chiều song song, và cung cấp một giao diện huấn luyện thống nhất — ngườidùng gọi `execute_pipeline()` hoặc `model.forward()` mà không cần quan tâm đến việc tensor đang nằm ở GPU nào, gradient cần đồng bộ qua nhóm nào.

**Tại sao cần nó?**

Việc tự triển khai TP+PP+DP thủ công bằng PyTorch distributed là cực kỳ phức tạp: cần tạo process group, shard tham số, đồng bộ gradient, quản lý pipeline stage, gửi activation qua P2P, xử lý microbatch... `HybridParallelPlugin` gói gọn toàn bộ logic này thành một lớp duy nhất. Luận văn không phát triển lại bánh xe này, mà **tận dụng** nó làm **môi trường thực thi** cho các chiến lược do planner tự động tìm ra.

**Giới hạn mà luận văn đặt ra.**

Theo nguyên tắc **không sửa core code**, luận văn sử dụng `HybridParallelPlugin` như một **hộp đen (black box)**: planner quyết định $(pp, tp, dp)$ và `dp_outside`, sau đó truyền các tham số này vào plugin. Không có sự can thiệp vào cách plugin shard mô hình, lập lịch pipeline, hay đồng bộ gradient.

---

### 3.3.2 Kiến trúc tổng quan

`HybridParallelPlugin` gồm ba thành phần chính hoạt động phối hợp:

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridParallelPlugin                      │
├─────────────────┬─────────────────┬─────────────────────────┤
│   ShardFormer   │ PipelineStage   │       Data Parallel     │
│   (TP sharding) │   Manager (PP)  │       (DDP/ZeRO)        │
└────────┬────────┴────────┬────────┴────────────┬────────────┘
         │                 │                     │
         ▼                 ▼                     ▼
┌─────────────────────────────────────────────────────────────┐
│              OneForwardOneBackwardSchedule                   │
│                    (1F1B Pipeline)                           │
└─────────────────────────────────────────────────────────────┘
```

**Thành phần 1 — ShardFormer (Tensor Parallelism).**

*Ngườidùng nhìn thấy:* Mô hình `GPT2LMHeadModel` bình thường.
*Thực tế bên trong:* `ShardFormer` thay thế mỗi lớp `nn.Linear` bằng các lớp song song tương ứng (`Linear1D_Col`, `Linear1D_Row`) dựa trên *policy* của từng kiến trúc. Với GPT-2, policy (`GPT2Policy`) chỉ định:
- `attn.c_attn` → `GPT2FusedLinearConv1D_Col` (chia trọng số theo chiều output)
- `attn.c_proj` → `GPT2FusedLinearConv1D_Row` (chia trọng số theo chiều input)
- `mlp.c_fc` → `GPT2FusedLinearConv1D_Col`
- `mlp.c_proj` → `GPT2FusedLinearConv1D_Row`

*Phân loại:* **Có sẵn trong ColossalAI.** `ShardFormer`, `ShardConfig`, và các policy (`GPT2Policy`, `BasePolicy`) đều là code của framework.

**Thành phần 2 — PipelineStageManager (Pipeline Parallelism).**

*Chức năng:* Xác định mỗi rank thuộc pipeline stage nào, và điều phối việc gửi/nhận activation giữa các stage qua `dist.send`/`dist.recv`. Nó tạo một *process group mesh* 3 chiều $(dp, pp, tp)$ và ánh xạ global rank → $(dp\_rank, pp\_rank, tp\_rank)$.

*Phân loại:* **Có sẵn trong ColossalAI.** `PipelineStageManager` và `ProcessGroupMesh` là core component của framework.

**Thành phần 3 — Data Parallel (DDP/ZeRO).**

*Chức năng:* Khi $dp > 1$, `HybridParallelPlugin` bọc mô hình bằng `DistributedDataParallel` (DDP) hoặc `LowLevelZeroOptimizer` (ZeRO) để đồng bộ gradient sau mỗi step. Với $dp = 1$, không có overhead DP.

*Phân loại:* **Có sẵn trong PyTorch/ColossalAI.** DDP là module của PyTorch; ZeRO là module của ColossalAI.

---

### 3.3.3 Lịch trình Pipeline: Non-interleaved 1F1B

**Thuật toán.**

`HybridParallelPlugin` hỗ trợ ba kiểu pipeline schedule: `1f1b`, `interleaved`, và `zbv` (Zero-Bubble). Trong luận văn, do không truyền tham số `pp_style`, plugin sử dụng **giá trị mặc định** `pp_style="1f1b"`, tức là lịch trình **One-Forward-One-Backward (1F1B) non-interleaved**.

**1F1B non-interleaved** hoạt động như sau:

1. **Warmup (Fill phase):** Mỗi stage liên tiếp nhận và xử lý các microbatch forward cho đến khi microbatch đầu tiên đi đến stage cuối cùng.
2. **Steady state:** Mỗi stage xen kẽ thực hiện **1 forward** cho microbatch mới và **1 backward** cho microbatch cũ (đã forward xong).
3. **Cooldown (Drain phase):** Sau khi tất cả microbatch đã forward, các stage lần lượt thực hiện backward cho các microbatch còn lại.

Pseudocode:
```
Algorithm OneForwardOneBackwardSchedule(pp, M)
    // Fill phase: forward cho pp-1 microbatch đầu
    for m = 1 to pp-1:
        forward_microbatch(m)
    
    // Steady state: 1F1B
    for m = pp to M:
        forward_microbatch(m)
        backward_microbatch(m - pp + 1)
    
    // Drain phase: backward các microbatch còn lại
    for m = M - pp + 2 to M:
        backward_microbatch(m)
```

**Tại sao chọn 1F1B non-interleaved?**

- **Đơn giản:** Không cần chia model thành nhiều chunk như interleaved.
- **Bộ nhớ ổn định:** Mỗi GPU chỉ giữ một stage liên tục, dễ dự đoán memory footprint.
- **Đủ tốt cho thesis:** Trong thực nghiệm, microbatch size = 2 (small) nên bubble không quá lớn; lợi ích của interleaved không đáng kể so với độ phức tạp thêm.

*Phân loại:* **Có sẵn trong ColossalAI.** Class `OneForwardOneBackwardSchedule` thuộc `colossalai.pipeline.schedule`. Luận văn chỉ **truyền tham số** `num_microbatches` vào, không can thiệp logic lập lịch.

---

### 3.3.4 Cách plugin nhận chiến lược từ planner

**Luồng dữ liệu:**

```
Planner (tự phát triển)
    ↓
Quyết định: pp=4, tp=2, dp=1, dp_outside=True
    ↓
Orchestrator (tự phát triển)
    ↓
Khởi tạo plugin:
    plugin = HybridParallelPlugin(
        pp_size=4,
        tp_size=2,
        num_microbatches=8,
        dp_outside=True,      // ← mặc định, phù hợp với planner
        precision="fp32",
    )
    ↓
Plugin tự động:
    - Tạo ProcessGroupMesh shape=(dp, pp, tp)
    - Shard model qua ShardFormer
    - Tạo PipelineStageManager
    - Tạo 1F1B scheduler
    ↓
Training loop gọi:
    booster.execute_pipeline(batch_iter, model, criterion, optimizer)
```

**Giao diện kết nối giữa code tự phát triển và code framework:**

| Module tự phát triển | Giao diện với framework | Tham số truyền |
|---------------------|------------------------|---------------|
| `Planner` (`search.py`) | Gọi `auto_plan()` → trả về `(pp, tp, dp)` | Không tương tác trực tiếp với plugin |
| `Orchestrator` (`run_auto_hybrid_parallel.py`) | Khởi tạo `HybridParallelPlugin(...)` | `pp_size`, `tp_size`, `num_microbatches`, `dp_outside` |
| `Training loop` | Gọi `booster.execute_pipeline(...)` | `batch_iter`, `model`, `criterion`, `optimizer` |

→ **Không có sửa đổi nào** bên trong `HybridParallelPlugin`, `ShardFormer`, hay `PipelineStageManager`. Mọi thứ đều thông qua giao diện công khai (public API).

---

### 3.3.5 Xử lý đặc biệt: pp = 1

**Vấn đề.**

Khi $pp = 1$ (không pipeline parallelism), `execute_pipeline()` của ColossalAI gặp lỗi vì nó mong đợi có ít nhất 2 stage để điều phối P2P. Đây là một **giới hạn đã biết** của framework.

**Giải pháp trong luận văn (tự phát triển).**

Trong `run_auto_hybrid_parallel.py` (dòng 587-599), orchestrator kiểm tra điều kiện $pp = 1$ và chuyển sang nhánh **forward/backward chuẩn** thay vì gọi `execute_pipeline()`:

```python
if pp == 1:
    # Không pipeline → dùng forward/backward chuẩn
    output = model(**batch_gpu)
    loss = output.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
else:
    # Có pipeline → dùng execute_pipeline
    outputs = booster.execute_pipeline(
        iter([batch]), model, criterion, optimizer, ...
    )
```

*Phân loại:* Đây là **code tự phát triển** (workaround) để xử lý giới hạn của framework. Nó không sửa plugin, mà **bypass** plugin ở tầng orchestrator.

---

### 3.3.6 Tóm tắt phân loại nguồn gốc

| Thành phần | Nguồn gốc | Vai trò trong luận văn |
|-----------|-----------|----------------------|
| `HybridParallelPlugin` | **ColossalAI** | Môi trường thực thi TP+PP+DP. Nhận $(pp, tp, dp)$ từ planner. Không sửa. |
| `ShardFormer` | **ColossalAI** | Shard model theo TP. Tự động thay thế `nn.Linear` bằng parallel layers. |
| `PipelineStageManager` | **ColossalAI** | Quản lý PP stage, P2P send/recv, process group mesh. |
| `OneForwardOneBackwardSchedule` | **ColossalAI** | Lịch trình 1F1B non-interleaved. Mặc định `pp_style="1f1b"`. |
| `ProcessGroupMesh` | **ColossalAI** | Tạo lưới process group theo thứ tự $(dp, pp, tp)$. |
| `DDP` / `ZeRO` | **PyTorch / ColossalAI** | Đồng bộ gradient khi $dp > 1$. |
| `auto_plan()` | **Tự phát triển** | Tìm $(pp, tp, dp)$ tối ưu. Truyền tham số vào plugin. |
| `pp=1 branch` | **Tự phát triển** | Workaround cho lỗi `execute_pipeline()` khi không có PP. |

---

### 3.3.7 Hình ảnh đề xuất cho luận văn

**Hình 3.X — Luồng tích hợp giữa Planner và HybridParallelPlugin.**

*Thành phần:*
- Bên trái: Module tự phát triển (profiler → cost model → search → orchestrator)
- Mũi tên: "$(pp, tp, dp), dp\_outside$" chuyển sang bên phải
- Bên phải: Box lớn "HybridParallelPlugin" chứa 3 box nhỏ: ShardFormer, PipelineStageManager, DDP/ZeRO
- Dưới cùng: "Training Loop" với nhánh `pp=1` (standard forward) và `pp>1` (execute_pipeline)

*Mục đích:* Minh họa ranh giới rõ ràng giữa code tự phát triển (quyết định chiến lược) và code framework (thực thi chiến lược).

---

## Tài liệu tham khảo

- `colossalai/booster/plugin/hybrid_parallel_plugin.py` — source code HybridParallelPlugin.
- `colossalai/pipeline/schedule/one_f_one_b.py` — source code 1F1B scheduler.
- `colossalai/shardformer/` — module ShardFormer và các policy.
- `colossalai/pipeline/stage_manager.py` — module PipelineStageManager.
