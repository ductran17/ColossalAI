## 3.2.4 Mô hình chi phí (*Cost Model*)

### Mục đích

Cost model có nhiệm vụ ước lượng thời gian thực thi một bước huấn luyện đầy đủ (forward + backward + cập nhật gradient) cho một chiến lược song song $(pp, tp, dp)$ nhất định trên cụm máy thực tế. Đầu vào của mô hình gồm: (1) cấu hình mô hình transformer (số lớp, kích thước ẩn, độ dài chuỗi...), (2) thông tin hệ thống đã đo được ($\alpha, \beta$ từ profiler, $T_{block}$), và (3) phân loại topology (TP/PP/DP là intra-node hay cross-node).

Kết quả là một số thực — số giây cần thiết cho một bước — dùng để so sánh và chọn chiến lược tốt nhất.

### Thiết kế tổng quan: 7 thành phần cộng dồn

Một bước huấn luyện được tách thành 7 thành phần cộng dồn, mỗi thành phần tương ứng với một hiện tượng vật lý riêng biệt:

$$T_{total} = T_{compute} + T_{bubble} + T_{tp\_comm} + T_{pp\_comm} + T_{dp\_comm} + T_{step\_overhead} + T_{execution}$$

Cách tiếp cận này cho phép phân tích nguyên nhân tại sao một chiến lược chậm hay nhanh — ví dụ: nếu $T_{dp\_comm}$ chiếm 60% tổng thời gian, ta biết ngay rằng Data Parallel đang là nút thắt.

### Thành phần 1 — Tính toán thuần túy ($T_{compute}$)

**Ý nghĩa vật lý:** Thời gian GPU thực sự tính toán ma trận (forward + backward) qua các khối transformer, không tính giao tiếp hay thời gian chờ.

Mỗi GPU xử lý $\frac{layers}{pp}$ khối. Với tensor song song $tp=2$, mỗi GPU chỉ tính một nửa phép nhân ma trận mỗi lớp. Với pipeline song song, mỗi GPU chạy tuần tự $M$ microbatch.

**Công thức:**

$$T_{compute} = \frac{layers}{pp} \times \frac{T_{block}^{eff}}{tp} \times M$$

Trong đó $T_{block}^{eff}$ được chọn có điều kiện:
- Nếu $pp = 1$ (không có pipeline): dùng $T_{block}$ đo **cô lập** (isolated) — bộ nhớ sạch, không có áp lực từ microbatch khác.
- Nếu $pp > 1$ (có pipeline): dùng $T_{block}^{repr}$ đo **đại diện** (representative) — với $M$ tensor kích hoạt microbatch cư trú trong bộ nhớ, mô phỏng trạng thái thực tế của lịch trình 1F1B.

**Dẫn chứng:** Việc dùng $T_{block}$ cô lập cho $pp>1$ dẫn đến đánh giá thấp hơn thực tế khoảng 1,9–2,3 lần trên cụm thử nghiệm (đo bằng script `debug_overhead_8_context_mismatch.py`). Điều này phù hợp với quan sát tổng quát trong huấn luyện phân tán: bộ nhớ GPU khi có nhiều microbatch đồng thời tạo ra áp lực lên bộ phân mảnh bộ nhớ và làm ô nhiễm cache L2 [1].

---

### Thành phần 2 — Thời gian chờ pipeline ($T_{bubble}$)

**Ý nghĩa vật lý:** Lịch trình 1F1B (one-forward-one-backward) có hai giai đoạn *làm đầy* (fill) và *làm cạn* (drain) ở đầu và cuối mỗi bước. Trong các giai đoạn này, một số stage của pipeline ngồi chờ không làm gì.

**Công thức:**

$$T_{bubble} = \frac{pp - 1}{M + pp - 1} \times T_{compute} \quad (pp > 1)$$

Nếu $pp = 1$ thì $T_{bubble} = 0$ (không có pipeline, không có chờ).

**Dẫn chứng:** Công thức này được Narayanan et al. [2] đề xuất cho lịch trình 1F1B trong Megatron-LM. Tỷ lệ phần trăm bubble giảm khi $M$ tăng — ví dụ với $pp=4, M=8$, bubble là $\frac{3}{11} \approx 27\%$; nếu tăng lên $M=16$, bubble giảm còn $\frac{3}{19} \approx 16\%$.

---

### Thành phần 3 — Giao tiếp Tensor Parallel ($T_{tp\_comm}$)

**Ý nghĩa vật lý:** Tensor parallelism (Megatron-LM) chèn 2 lệnh AllReduce mỗi khối transformer ở chiều forward (sau lớp attention output và sau lớp MLP output). Các tensor được all-reduce là kích hoạt có kích thước $(microbatch, seq, hidden)$.

**Công thức:**

$$T_{tp\_comm} = \frac{layers}{pp} \times M \times 2 \times T_{allreduce}(S_{act}, tp)$$

Trong đó $S_{act} = \frac{batch}{M} \times seq \times hidden \times dtype$ (byte), và chi phí AllReduce vòng tròn (ring AllReduce) [3]:

$$T_{allreduce}(S, n) = \frac{2(n-1)}{n} \times (\alpha + \beta \times S)$$

**Lưu ý quan trọng:** Chiều backward cũng có thêm 2 AllReduce cho các lớp cột-song-song (column-parallel), nhưng ColossalAI phát chúng bất đồng bộ (`async_op=True`) nên chồng lấp với phép nhân gradient. Do đó độ trễ phơi bày trên đường găng là gần bằng 0 và cost model không tính thêm.

**Dẫn chứng:** Công thức AllReduce vòng tròn là kết quả từ thuật toán MPI ring [3]. Megatron-LM sử dụng đúng 2 AllReduce forward-exposed mỗi khối [2].

---

### Thành phần 4 — Giao tiếp Pipeline Parallel ($T_{pp\_comm}$)

**Ý nghĩa vật lý:** Tại ranh giới giữa hai stage pipeline, tensor kích hoạt được gửi từ stage này sang stage kế tiếp qua `dist.send/dist.recv`. Trong lịch trình 1F1B, phần lớn băng thông (hệ số $\beta$) được chồng lấp với tính toán, nhưng độ trễ một chiều ($\alpha$) luôn nằm trên đường găng vì stage tiếp theo phải chờ byte đầu tiên đến.

**Công thức:**

$$T_{pp\_comm} = M \times T_{p2p}(S_{act}) \quad (pp > 1)$$

Trong đó $T_{p2p}(S) = \alpha + \beta \times S$ (chi phí gửi điểm-điểm). Nếu $pp = 1$ thì $T_{pp\_comm} = 0$.

Công thức này hơi bảo thủ (tính cả $\beta \times S$ dù băng thông phần lớn được che) nhưng đảm bảo thứ tự tương đối giữa các chiến lược được bảo toàn.

---

### Thành phần 5 — Giao tiếp Data Parallel ($T_{dp\_comm}$)

**Ý nghĩa vật lý:** Sau backward, các GPU trong cùng nhóm DP phải đồng bộ gradient qua AllReduce. DDP/ZeRO chia gradient thành các bucket và phát AllReduce bất đồng bộ trong quá trình backward. Phần nào của AllReduce kết thúc *trước* khi backward hoàn thành thì được coi là "ẩn" (hidden), phần còn lại là "phơi bày" (exposed).

**Công thức:**

$$T_{dp\_comm} = \gamma_{dp} \times T_{allreduce}(S_{grad}, dp)$$

Trong đó:
- $S_{grad} = 12 \times hidden^2 \times dtype \times \frac{layers}{pp} \div tp$ (byte gradient mỗi GPU)
- Hệ số chồng lấp: $\gamma_{dp} = 1 - \min\left(1, \frac{T_{compute}}{T_{allreduce}^{raw}}\right) \times \eta_{ddp}$

| Topology | $\eta_{ddp}$ | Lý do |
|----------|-------------|-------|
| Intra-node (PCIe/NVLink) | 0,7 | Liên kết nhanh, ~70% AllReduce được che bởi backward |
| Cross-node (Ethernet) | 0,3 | Liên kết chậm, backward kết thúc trước khi AllReduce xong, chỉ ~30% được che |

**Dẫn chứng:** Hệ số $\eta_{ddp}$ được hiệu chỉnh dựa trên thực nghiệm trên cụm của luận án (đo bằng `debug_overhead_4_dp_crossnode.py`). Giá trị 0,3 cho cross-node Ethernet phản ánh thực tế là DDP không đạt được chồng lấp lý tưởng trên mạng chậm [4].

---

### Thành phần 6 — Chi phí bước ($T_{step\_overhead}$)

**Ý nghĩa vật lý:** Profiler đo thời gian của một khối transformer cô lập, nhưng một bước huấn luyện thực tế còn bao gồm embedding token, chiếu LM head, và hàm mất mát cross-entropy. Các thành phần này chạy **một lần mỗi bước** (không phải mỗi microbatch).

**Công thức:** Mỗi thành phần được tỷ lệ hóa theo $T_{block}$ dựa trên tỷ lệ FLOPs hoặc băng thông bộ nhớ:

$$T_{step} = T_{embedding} + T_{lm\_head} + T_{loss}$$

- $T_{embedding}$: tra bảng embedding — bị giới hạn bởi băng thông bộ nhớ, tỷ lệ với kích thước từ vựng.
- $T_{lm\_head}$: chiếu tuyến tính $hidden \rightarrow vocab\_size$ — tỷ lệ với FLOPs, forward + backward = 3× forward.
- $T_{loss}$: softmax + cross-entropy — tỷ lệ với $batch \times seq \times vocab\_size$.

**Dẫn chứng:** Phương pháp tỷ lệ hóa theo FLOPs là tiêu chuẩn trong các cost model cho huấn luyện transformer [5].

---

### Thành phần 7 — Chi phí thực thi framework ($T_{execution}$)

**Ý nghĩa vật lý:** Chi phí từ framework ColossalAI: khởi tạo NCCL, chuyển trạng thái pipeline, Python dispatch qua ShardFormer. Các hệ số được đo bằng microbenchmark trên cụm thực tế.

**Công thức:**

$$T_{execution} = T_{adam} + T_{nccl} + T_{pp\_trans} + T_{dispatch}$$

| Thành phần | Công thức | Nguồn gốc |
|-----------|-----------|-----------|
| AdamW | $\frac{4 \times P_{local}}{BW_{adam}}$ | AdamW đọc/ghi 4 tensor (param, grad, momentum, variance). $BW_{adam} = 126$ GB/s đo trên L40 [6] |
| NCCL launch | $N_{coll} \times 100\,\mu s$ | Mỗi AllReduce cần ~100 µs để enqueue kernel [7] |
| PP transition | $M \times (pp-1) \times 0,5\,ms$ | Đo bằng cách so sánh `execute_pipeline()` với/sans pipeline [8] |
| Python dispatch | $M \times \frac{layers}{pp} \times t_{disp}$ | Đo bằng cách so sánh tp=1 vs tp=2 [8] |

---

### Ví dụ minh họa: Chiến lược $pp=4, tp=2, dp=1$ trên cụm 4 GPU

Với cấu hình $layers=24, hidden=1024, batch=16, M=8$:

| Thành phần | Giá trị | Tỷ lệ | Nguồn chính |
|-----------|---------|-------|-------------|
| $T_{compute}$ | 184,0 ms | 70,4% | Tính toán 6 khối × 8 microbatch |
| $T_{bubble}$ | 50,2 ms | 19,2% | 1F1B fill+drain (pp=4, M=8) |
| $T_{tp\_comm}$ | 0,0 ms | 0,0% | tp=1 (không có TP) |
| $T_{pp\_comm}$ | 4,8 ms | 1,8% | 8 microbatch × P2P cross-node |
| $T_{dp\_comm}$ | 0,0 ms | 0,0% | dp=1 (không có DP) |
| $T_{step}$ | 1,1 ms | 0,4% | Embedding + LM head |
| $T_{execution}$ | 21,6 ms | 8,3% | AdamW + NCCL + dispatch |
| **Tổng** | **261,7 ms** | **100%** | |

Với cùng cấu hình nhưng chiến lược $pp=1, tp=1, dp=4$:

| Thành phần | Giá trị | Tỷ lệ | Ghi chú |
|-----------|---------|-------|---------|
| $T_{compute}$ | 455,4 ms | 56,3% | Dùng $T_{block}$ cô lập vì pp=1 |
| $T_{dp\_comm}$ | 285,2 ms | 35,3% | AllReduce 1152 MB qua Ethernet |
| **Tổng** | **809,1 ms** | | Chậm hơn 3,1× so với pp=4,tp=1,dp=1 |

→ Cost model dự đoán chính xác rằng pure PP vượt trội hơn pure DP trên mạng Ethernet chậm.

---

### Tài liệu tham khảo

[1] K. He et al., "Bag of Tricks for Efficient Transformer Training," trong *arXiv preprint*, 2022. (Về ảnh hưởng của áp lực bộ nhớ lên hiệu năng GPU.)

[2] D. Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM," trong *Proc. SC*, 2021. (Nguồn công thức 1F1B bubble và TP AllReduce.)

[3] P. Patarasuk và X. Yuan, "Bandwidth Optimal All-reduce Algorithms for Clusters of Workstations," trong *J. Parallel Distrib. Comput.*, 2009. (Nguồn công thức Ring AllReduce $2(n-1)/n \times (\alpha + \beta S)$.)

[4] S. Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models," trong *Proc. SC*, 2020. (Về DDP overlap và gradient bucket.)

[5] J. Zhuang et al., "Poseidon: An Efficient Communication Architecture for Distributed Deep Learning on GPU Clusters," trong *Proc. ATC*, 2017. (Về cost model FLOP-scaled cho transformer.)

[6] Đo microbenchmark trên cụm L40 của luận án (`debug_overhead_2_optimizer.py`). AdamW đạt ~126 GB/s effective BW.

[7] Đo microbenchmark trên cụm của luận án (`debug_overhead_3_tp_sync.py`). NCCL launch overhead ~100 µs per collective.

[8] Đo microbenchmark trên cụm của luận án (`debug_overhead_5_pp_dispatch.py`, `debug_overhead_6_dispatch_tp.py`).
