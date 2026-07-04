## 3.2.2 Đo thông số cụm — Profiler (profiler.py)

### Vấn đề đặt ra

Cost model cần bốn đại lượng để tính toán chính xác:
1. **$\alpha, \beta$ giao tiếp trong node** — độ trễ và nghịch đảo băng thông của liên kết PCIe/NVLink (nhanh, ~26 GB/s).
2. **$\alpha, \beta$ giao tiếp giữa node** — tương tự nhưng qua Ethernet (chậm, ~2,7 GB/s).
3. **$T_{block}$** — thời gian forward+backward của một khối transformer trên GPU thực tế.
4. **$T_{block}^{repr}$** — thời gian tương tự nhưng trong điều kiện có áp lực bộ nhớ từ $M$ microbatch.

Nếu các đại lượng này không được đo trên **hardware thực tế**, cost model sẽ:
- Dùng giả định lý thuyết (ví dụ: peak FLOPS từ datasheet) → sai lệch lớn vì GPU không bao giờ đạt peak.
- Không biết được tốc độ mạng thực tế của cụm → đánh giá sai chi phí DP/PP.
- Không nắm được sự chênh lệch giữa các GPU trong cụm heterogeneous.

Mục tiêu của profiler: **đo mọi thứ cần thiết trong < 3 giây**, **không cần đoạn code huấn luyện thực tế**, **tự động đồng bộ giá trị giữa mọi rank**.

---

### Kiến trúc tổng quan

Profiler chạy trên **mọi rank** cùng lúc sau khi `dist.init_process_group()` đã được gọi. Nó gồm 7 bước:

```
Step 1: Thu thập node layout từ LOCAL_RANK / LOCAL_WORLD_SIZE
Step 2: Chọn 1 cặp intra-node và 1 cặp cross-node để đo P2P
Step 3: Đo α_intra, β_intra (send/recv nhiều kích thước tensor)
Step 4: Đo α_cross, β_cross (tương tự, qua node khác)
Step 5: Đo T_block cô lập (isolated, clean cache)
Step 6: Đo T_block đại diện (với M microbatch activation + Adam state)
Step 7: Đo bộ nhớ trống tối thiểu + all_reduce(MAX) để đồng bộ
```

→ Trả về `ClusterProfile` — dataclass chứa 6 giá trị số, đồng nhất trên toàn bộ cụm.

---

### Cơ sở lý thuyết: Mô hình T = α + β × S

Thời gian truyền một tensor từ GPU này sang GPU khác tuân theo mô hình điểm-điểm đơn giản:

$$T(S) = \alpha + \beta \times S$$

Trong đó:
- **$\alpha$** (giây): độ trễ cố định mỗi lần gửi — bao gồm overhead hệ điều hành, CPU enqueue, NCCL handshake. Không phụ thuộc kích thước tensor.
- **$\beta$** (giây/byte): nghịch đảo băng thông — thời gian truyền thêm cho mỗi byte dữ liệu. $\beta = 1 / BW$.

Ví dụ: với BW = 26 GB/s → $\beta = 1/(26 \times 10^9) \approx 0{,}038$ ns/byte.

Profiler đo $T(S)$ ở **15 điểm kích thước** khác nhau (từ 256 byte đến 4 MB), sau đó dùng **hồi quy tuyến tính bình phương tối thiểu** để xác định $\alpha$ và $\beta$.

---

### Chi tiết từng phép đo

#### 1. Thu thập node layout (`_gather_node_layout`)

Mỗi rank báo cáo `LOCAL_RANK` (vị trí trong node) và `LOCAL_WORLD_SIZE` (tổng số GPU trong node). Thông tin này được `all_gather` để rank 0 (và mọi rank khác) xây dựng lại bản đồ:

```python
nodes = [[0, 1], [2, 3, 4, 5], [6, 7]]   # cụm [2,4,2]
```

→ Biết được rank nào nằm trên node nào mà **không cần file cấu hình tĩnh**.

#### 2. Chọn cặp đo (`_select_pairs`)

Vì đo tất cả $\binom{8}{2}=28$ cặp rank là lãng phí, profiler chỉ chọn:
- **Intra-node pair:** hai rank đầu tiên trên node đầu tiên có $\geq 2$ GPU.
- **Cross-node pair:** rank 0 của node 0 → rank 0 của node 1.

Đây là **đại diện đủ tốt** vì tốc độ P2P trong cùng node (PCIe) và giữa hai node (Ethernet) không phụ thuộc vào việc chọn rank nào cụ thể.

#### 3. Đo α và β (`_measure_p2p`)

**Phương pháp đo (GPU-side events, không dùng CPU timer):**

```python
# 1. Warmup: 10 lần gửi/nhận (không tính thời gian)
for _ in range(warmup):
    dist.send(tensor, dst=dst)   # nếu là src
    dist.recv(tensor, src=src)   # nếu là dst

# 2. Timed measurement: 50 lần, mỗi lần dùng torch.cuda.Event
for _ in range(repeat):
    start_evt.record()
    dist.send(...) / dist.recv(...)
    end_evt.record()
    torch.cuda.synchronize()
    dt = start_evt.elapsed_time(end_evt)   # GPU-side microseconds
```

**Tại sao dùng `torch.cuda.Event` thay vì `time.perf_counter`?**

- `time.perf_counter` đo thời gian **CPU** — bao gồm cả overhead hệ điều hành, scheduling, Python GIL. Sai số có thể > 1 ms.
- `torch.cuda.Event` đo thời gian **GPU** — chỉ tính từ lúc kernel NCCL bắt đầu đến khi kết thúc. Độ chính xác ~1 µs.

**Loại bỏ ngoại lai (IQR clip):**

GPU có thể bị jitter từ:
- Thermal throttling (giảm xung nhịp khi nóng)
- OS scheduling (CPU thread bị đẩy sang core khác)
- CUDA context switch từ process khác trên node

Profiler tính Q1, Q3, IQR của 50 lần đo, sau đó **loại bỏ các điểm ngoài khoảng [Q1 − 1,5×IQR, Q3 + 1,5×IQR]**. Kết quả lấy **median** của các điểm còn lại.

**Ví dụ số liệu thực tế** (cụm [2,2,2]):

| Kích thước tensor | Thời gian median (µs) |
|-------------------|------------------------|
| 256 B | 82 |
| 4 KB | 85 |
| 64 KB | 92 |
| 512 KB | 120 |
| 1 MB | 160 |
| 4 MB | 340 |

Hồi quy tuyến tính trên các điểm này cho:
- **Intra-node:** $\alpha_{intra} \approx 66\ \mu s$, $\beta_{intra} \approx 0{,}05\ ns/B$ → BW ≈ 20 GB/s
- **Cross-node:** $\alpha_{cross} \approx 84\ \mu s$, $\beta_{cross} \approx 0{,}38\ ns/B$ → BW ≈ 2,6 GB/s

#### 4. Đo $T_{block}$ cô lập (`_measure_T_block`)

**Mục đích:** Đo thời gian forward+backward của **một khối transformer đơn lẻ** — không có áp lực bộ nhớ từ microbatch khác.

**Phương pháp:**
1. Xây dựng một module `_OneBlock` gồm: LayerNorm → Attention (Q,K,V,Out) → LayerNorm → MLP (fc1, fc2) — tương đương một transformer block GPT-2.
2. Dùng SGD optimizer (không cần Adam state phức tạp vì chỉ đo compute time).
3. Input: `batch × seq × hidden` tensor.
4. Clear CUDA cache (`empty_cache`) trước khi đo.
5. Warmup 10 lần + timed 50 lần, dùng `torch.cuda.Event`.
6. IQR clip + lấy median.

**Tại sao đo block đơn lẻ mà không đo cả model?**

- Đo cả model 24 lớp mất thời gian dài, dễ bị OOM trên GPU nhỏ.
- Cost model cần biết thời gian **một lớp** để nhân với `layers/pp`.
- Block đơn lẻ đại diện đủ tốt vì các lớp transformer là **identical** (cùng kích thước, cùng kiến trúc).

**Ví dụ số liệu** (A6000/L40S/L40/A30 trộn lẫn):
- $T_{block} \approx 1{,}96\ ms$ (MAX-reduced, vì slowest GPU quyết định pipeline)

#### 5. Đo $T_{block}$ đại diện (`_measure_T_block_with_microbatches`)

**Vấn đề:** $T_{block}$ cô lập (bộ nhớ sạch, chỉ 1 block) **thấp hơn thực tế** khi chạy pipeline 1F1B. Lý do:
- Trong pipeline, $M$ microbatch activation **cư trú đồng thời** trong GPU memory.
- Allocator bị phân mảnh, phải tìm chunk lớn hơn.
- L2 cache bị ô nhiễm bởi activation của microbatch khác.
- CUDA stream switch overhead tăng.

**Phương pháp đo:**
1. Chạy forward **M lần** với `no_grad`, lưu M activation tensors vào buffer (giữ chúng alive).
2. Chạy 5 step AdamW thật để pre-fill optimizer state (momentum, variance) — mô phỏng bộ nhớ thực tế.
3. Đo forward+backward 1 block trong context này.

**Chênh lệch thực tế:**
- Isolated $T_{block}$: ~1,96 ms
- Representative $T_{block}^{repr}$: ~3,68 ms
- **Tỷ lệ:** 1,88× (gần gấp đôi!)

→ Nếu cost model dùng $T_{block}$ cô lập cho pp>1, nó sẽ **đánh giá thấp thời gian tính toán ~1,9×**, dẫn đến chọn nhầm chiến lược.

**Dẫn chứng:** Đo bằng `debug_overhead_8_context_mismatch.py` — chạy `execute_pipeline()` với pp=4,tp=1 và so sánh $T_{block}$ đo được với $T_{block}$ cô lập. Sai lệch 1,9–2,3× phù hợp với đại diện.

#### 6. Đo bộ nhớ trống (`min_free_memory_gb`)

```python
free_bytes, _ = torch.cuda.mem_get_info()
dist.all_reduce(free_tensor, op=MIN)
```

→ Lấy **bộ nhớ trống nhỏ nhất** trên mọi GPU. Dùng làm ngân sách mặc định cho memory pruning trong planner. Giúp tránh OOM khi một GPU bị chiếm bởi process khác.

#### 7. Đồng bộ hóa (`all_reduce MAX`)

```python
buf = [alpha_intra, beta_intra, alpha_cross, beta_cross, T_block, T_block_repr]
dist.all_reduce(buf, op=MAX)
```

**Tại sao MAX?**
- $\alpha, \beta$: các rank không đo (không thuộc cặp đo) có giá trị 0. MAX lấy giá trị thật.
- $T_{block}$: MAX chọn **GPU chậm nhất**. Lý do: trong pipeline parallelism, stage chậm nhất quyết định tốc độ toàn pipeline (bottleneck).

→ Mọi rank nhận được **cùng một ClusterProfile**, đảm bảo planner tính toán đồng nhất.

---

### Ví dụ minh họa: ClusterProfile thực tế

Output từ cụm 6 GPU `[2, 2, 2]` (node18, node19, node15):

```json
{
  "alpha_intra_us": 65.8,
  "beta_intra_ns_per_B": 0.051,     // BW ≈ 19.6 GB/s
  "alpha_cross_us": 83.8,
  "beta_cross_ns_per_B": 0.375,     // BW ≈ 2.67 GB/s
  "T_block_ms": 1.976,              // isolated
  "T_block_with_microbatches_ms": 3.682,  // representative
  "min_free_memory_gb": 35.4
}
```

**Phân tích:**
- Băng thông intra-node (19,6 GB/s) thấp hơn lý thuyết PCIe Gen4 x16 (32 GB/s) do overhead NCCL và kernel launch.
- Băng thông cross-node (2,67 GB/s) phù hợp với bonded 10 GbE Ethernet lý thuyết (2,5–3,0 GB/s).
- Chênh lệch $T_{block}$ vs $T_{block}^{repr}$ = 1,86×, xác nhận cần dùng đo lường đại diện cho pp>1.

---

### Các vấn đề thực tế đã gặp và cách khắc phục

| Vấn đề | Triệu chứng | Nguyên nhân | Khắc phục |
|--------|------------|------------|-----------|
| **NCCL hang** | Profiler đứng yên ở `dist.send` | PyTorch version mismatch (node18 dùng 2.10.0+cu128/NCCL 2.27.5, các node khác 2.5.0/NCCL 2.21.5) | Đồng bộ toàn bộ cụm lên PyTorch 2.5.1+cu124 |
| **T_block biến động** | Độ lệch chuẩn > 15% giữa các lần đo | Thermal throttling + OS jitter | Tăng repeat từ 10 → 50, thêm IQR clip |
| **Underestimate cho pp>1** | Cost model dự đoán thấp hơn thực tế 1,9× | T_block cô lập không có áp lực bộ nhớ | Thêm `_measure_T_block_with_microbatches()` |
| **CPU timer noise** | T_block dao động 0,5–2 ms giữa các run | `time.perf_counter` bị ảnh hưởng bởi scheduling | Chuyển sang `torch.cuda.Event` |
| **OOM trên A30** | `CUDA out of memory` khi profile | A30 chỉ có 24 GB, model hidden=1024 chiếm ~15 GB | Đo T_block với microbatch size nhỏ hơn, dùng `min_free_memory_gb` |

---

### Tại sao không đo thêm các hệ số khác (BW_adam, nccl_launch, dispatch)?

Profiler hiện tại chỉ đo $\alpha, \beta, T_{block}, T_{block}^{repr}$, không đo các hệ số $T_{execution}$ (AdamW BW, NCCL launch, PP transition, Python dispatch).

**Lý do:**
1. Các hệ số này cần **nhiều scenario** để tách riêng (ví dụ: so sánh tp=1 vs tp=2 để tách dispatch_tp). Tự động hóa phức tạp hơn đo P2P đơn giản.
2. Chúng chiếm tỷ lệ nhỏ trong tổng thời gian (~8–15%), nên sai số 20% chỉ gây sai lệch tổng thể ~1–3%.
3. Các giá trị default (126 GB/s, 100 µs, 0.5 ms, 0.15 ms, 0.05 ms) đã được hiệu chỉnh bằng microbenchmark một lần và lưu cứng trong code.

**Hướng cải tiến:** Trong tương lai, có thể tích hợp các microbenchmark `debug_overhead_*.py` vào profiler để tự động đo toàn bộ hệ số.

---

### Tài liệu tham khảo

- `profiler.py` — source code profiler.
- `debug_overhead_1_framework.py` — đo base Python dispatch.
- `debug_overhead_2_optimizer.py` — đo AdamW effective bandwidth.
- `debug_overhead_3_tp_sync.py` — đo NCCL launch overhead.
- `debug_overhead_5_pp_dispatch.py` — đo PP transition time.
- `debug_overhead_6_dispatch_tp.py` — đo TP-specific ShardFormer overhead.
- `debug_overhead_8_context_mismatch.py` — đo chênh lệch T_block cô lập vs thực tế.
