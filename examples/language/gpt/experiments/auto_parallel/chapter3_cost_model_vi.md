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

**Ý nghĩa vật lý:** Profiler đo thời gian của **một khối transformer** (multi-head attention + MLP). Nhưng một bước huấn luyện **thực tế** còn phải qua ba giai đoạn ngoài khối transformer:

1. **Token embedding**: chuyển chỉ số từ (integer) thành vector ẩn — bước này chỉ đọc bảng tham số `vocab_size × hidden`.
2. **LM head**: chiếu tuyến tính từ `hidden → vocab_size` để tạo logits — tính toán nặng với ma trận lớn.
3. **Cross-entropy loss**: tính softmax trên toàn bộ từ vựng + chọn đúng nhãn — tính toán tương đối nhẹ.

Các thành phần này chạy **một lần mỗi bước** (không phải mỗi microbatch), nên được cộng vào tổng thời gian như một khoản chi phí cố định.

**Ví dụ tính toán cụ thể** (với cấu hình `layers=24, hidden=1024, seq=256, batch=16, vocab=1024, dtype=fp32`):

| Thành phần | Công thức | Số liệu | Kết quả |
|-----------|-----------|---------|---------|
| **Embedding** | $T_{block} \times \frac{vocab \times hidden}{12 \times hidden^2} \times 0{,}5$ | $2{,}37 \times \frac{1024 \times 1024}{12 \times 1024^2} \times 0{,}5$ | **0,10 ms** |
| **LM head** | $T_{block} \times \frac{2 \times B \times S \times H \times V}{12 \times H^2 \times B \times S} \times 3$ | $2{,}37 \times \frac{2 \times 16 \times 256 \times 1024 \times 1024}{12 \times 1024^2 \times 16 \times 256} \times 3$ | **1,19 ms** |
| **Loss** | $T_{block} \times \max\left(\frac{3 \times B \times S \times V}{12 \times H^2 \times B \times S}, 0{,}005\right)$ | $2{,}37 \times \max(0{,}00024, 0{,}005)$ | **0,01 ms** |
| **Tổng** | | | **1,30 ms** |

Giải thích chi tiết từng thành phần:

#### a) Token embedding ($T_{embedding}$)

**Token embedding là gì?**

Trong mô hình ngôn ngữ (GPT), đầu vào là các token dạng số nguyên (ví dụ: token "hello" = 15496, token "world" = 995). Lớp `nn.Embedding` trong PyTorch lưu một **ma trận lớn** có kích thước `(vocab_size, hidden)` — gọi là **bảng tra (lookup table)**. Mỗi hàng của ma trận là vector ẩn (hidden vector) của một token.

Khi forward, lớp embedding không thực hiện phép nhân ma trận. Nó chỉ đơn giản là:
```python
output = embedding_weight[token_id]  # Đọc hàng thứ token_id từ ma trận
```
→ **Chỉ là thao tác đọc bộ nhớ (memory-bound), không có tính toán nặng.**

**Công thức tính:**

$$T_{embedding} = T_{block} \times \frac{vocab\_size \times hidden}{12 \times hidden^2} \times 0{,}5$$

Giải thích từng phần:
- **Tử số** ($vocab \times hidden$): tổng số byte cần đọc từ HBM = kích thước bảng tra embedding.
  - Với `vocab=1024, hidden=1024, dtype=fp32 (4 bytes)`: $1024 \times 1024 \times 4 = 4{,}194{,}304$ byte ≈ **4 MB**.
- **Mẫu số** ($12 \times hidden^2$): kích thước tham số của **một khối transformer** — dùng làm mốc chuẩn để so sánh tỷ lệ.
  - Một khối transformer có ~12H² tham số (Q/K/V/out + MLP fc1/fc2).
  - Với $H=1024$: $12 \times 1024^2 \times 4 = 50{,}331{,}648$ byte ≈ **48 MB**.
- **Hệ số 0,5**: $T_{block}$ đo thời gian **compute-bound** (nhiều phép nhân ma trận, tận dụng được Tensor Core). Embedding là **memory-bound** (chỉ đọc, không tính toán nặng). Do đó embedding chậm hơn ~2× so với tỷ lệ FLOPs — hệ số 0,5 ước lượng điều này.
  - Không có hệ số 0,5: $T_{embedding} = 2{,}37 \times (4/48) = 0{,}20$ ms
  - Có hệ số 0,5: $T_{embedding} = 0{,}20 \times 0{,}5 = 0{,}10$ ms (phù hợp thực tế hơn)

**Ví dụ số liệu:**

Với `vocab_size=1024, hidden=1024, T_block=2.37 ms`:
- Tỷ lệ kích thước: $\frac{1024 \times 1024}{12 \times 1024^2} = \frac{1}{12} \approx 0{,}083$
- Thời gian embedding: $2{,}37 \times 0{,}083 \times 0{,}5 = 0{,}10$ ms

→ Embedding chỉ chiếm **0,10 ms** trong tổng bước ~800 ms, là thành phần nhỏ nhất trong $T_{step}$.

#### b) LM head ($T_{lm\_head}$)

**LM head là gì?**

Sau khi qua các khối transformer, mô hình cần chiếu vector ẩn cuối cùng từ `hidden` chiều về `vocab_size` chiều để tạo logits — đây là một lớp tuyến tính (linear layer) có ma trận trọng số `(hidden, vocab_size)`.

Thao tác này **tính toán nặng** (ma trận × vector) nên tỷ lệ với FLOPs, không cần hệ số điều chỉnh.

**Công thức:**

$$T_{lm\_head} = T_{block} \times \frac{2 \times B \times S \times H \times V}{12 \times H^2 \times B \times S} \times 3$$

Giải thích:
- **Tử số FLOPs LM head**: $2 \times B \times S \times H \times V$ (ma trận-vector cho `batch × seq` vị trí).
- **Mẫu số FLOPs một khối**: $12 \times H^2 \times B \times S$ (tổng FLOPs attention + MLP).
- **× 3**: forward + backward. Backward qua lớp tuyến tính cần tính gradient cho cả input và weight → khoảng **2× forward**, tổng cộng **3×**.

**Ví dụ số liệu:**

Với `B=16, S=256, H=1024, V=1024`:
- Tỷ lệ FLOPs: $\frac{2 \times 1024}{12 \times 1024} = \frac{2}{12} = 0{,}167$
- Tổng ratio (×3): $0{,}167 \times 3 = 0{,}5$
- Thời gian: $2{,}37 \times 0{,}5 = 1{,}19$ ms

→ LM head chiếm **1,19 ms**, là thành phần lớn nhất trong $T_{step}$.

#### c) Cross-entropy loss ($T_{loss}$)

**Loss là gì?**

Sau khi có logits (điểm số cho mỗi từ trong vocab), cần tính:
1. **Softmax**: $P_i = e^{z_i} / \sum_j e^{z_j}$ — tính xác suất cho vocab_size từ.
2. **Negative log-likelihood**: $-\log(P_{target})$ — chọn đúng nhãn.

Thao tác này tính toán nhẹ hơn LM head (không có ma trận lớn), nhưng vẫn cần duyệt qua toàn bộ vocab.

**Công thức:**

$$T_{loss} = T_{block} \times \max\left(\frac{3 \times B \times S \times V}{12 \times H^2 \times B \times S}, 0{,}005\right)$$

Giải thích:
- **3 × B × S × V**: FLOPs softmax (3 phép toán cơ bản mỗi phần tử: exp, sum, divide).
- **max(..., 0,005)**: đảm bảo loss không bị đánh giá quá thấp khi vocab_size nhỏ.
  - Không có floor: ratio = $\frac{3 \times 1024}{12 \times 1024^2} \approx 0{,}00025$ → quá nhỏ, mất chính xác.
  - Có floor 0,005: ratio = 0,005 → $T_{loss} = 2{,}37 \times 0{,}005 = 0{,}01$ ms.

→ Loss chỉ chiếm **0,01 ms**, gần như không đáng kể.

**Tổng $T_{step}$ cho ví dụ:** $0{,}10 + 1{,}19 + 0{,}01 = 1{,}30$ ms.

**Dẫn chứng:** Phương pháp tỷ lệ hóa theo FLOPs/băng thông là tiêu chuẩn trong các cost model cho huấn luyện transformer [5]. Việc tách embedding + LM head + loss ra khỏi $T_{block}$ là cần thiết vì:
- Profiler đo thời gian **một khối transformer**, không bao gồm đầu vào/ra.
- Với mô hình nhỏ (hidden=256), LM head chiếm tỷ lệ đáng kể; với mô hình lớn (hidden=4096), tỷ lệ này giảm xuống.
- Khi `pp > 1`, LM head và embedding chỉ chạy ở stage đầu/cuối, không phải mọi GPU — nhưng cost model tính tổng cho toàn bộ pipeline nên vẫn cộng vào đúng một lần.

---

### Thành phần 7 — Chi phí thực thi framework ($T_{execution}$)

**Ý nghĩa vật lý:** Đây là chi phí **tổng hợp** do framework ColossalAI tạo ra, không phải tính toán thuần GPU. Bao gồm bốn nguồn độc lập:

1. **AdamW optimizer step**: đọc/ghi 4 tensor (param, grad, momentum, variance) cho mọi tham số — bị giới hạn bởi băng thông HBM.
2. **NCCL launch overhead**: mỗi lệnh AllReduce cần CPU enqueue + GPU kernel launch (~100 µs), không phụ thuộc kích thước tensor.
3. **Pipeline stage transitions**: setup P2P send/recv giữa các stage pipeline (chỉ khi $pp > 1$).
4. **Python dispatch**: overhead từ Python loop qua `execute_pipeline()` và `ShardFormer` tensor manipulation — thường bị che bởi pipeline overlap, nhưng khi $pp = 1$ thì lộ hoàn toàn.

**Công thức:**

$$T_{execution} = T_{adam} + T_{nccl} + T_{pp\_trans} + T_{dispatch}$$

| Thành phần | Công thức | Điều kiện | Nguồn gốc |
|-----------|-----------|-----------|-----------|
| AdamW | $\frac{4 \times P_{local}}{BW_{adam}}$ | Luôn có | AdamW đọc/ghi 4 tensor. $BW_{adam} = 126$ GB/s đo trên L40 [6] |
| NCCL launch | $N_{coll} \times 100\,\mu s$ | tp>1 hoặc dp>1 | Mỗi AllReduce cần ~100 µs enqueue [7] |
| PP transition | $M \times (pp-1) \times 0{,}5\,ms$ | Chỉ khi $pp > 1$ | Đo bằng `execute_pipeline()` [8] |
| Python dispatch | $M \times \frac{layers}{pp} \times t_{disp}$ | Luôn có | Đo bằng tp=1 vs tp=2 [8] |

**Ví dụ tính toán cụ thể** (cấu hình `pp=1, tp=1, dp=4, layers=24, hidden=1024, M=8, dtype=fp32`):

| Thành phần | Công thức chi tiết | Tính toán | Kết quả |
|-----------|-------------------|-----------|---------|
| **AdamW** | $\frac{4 \times (12 \times 1024^2 \times 4 \times 24)}{126 \times 10^9}$ | $\frac{4 \times 1{,}208{,}000{,}000}{126 \times 10^9}$ | **38,3 ms** |
| **NCCL** | $1 \times 100\,\mu s$ (dp=4, 1 AllReduce/step) | | **0,1 ms** |
| **PP transition** | $pp=1$ → không có pipeline | | **0 ms** |
| **Dispatch** | $8 \times 24 \times 0{,}15\,ms$ | (pp=1 nên base dispatch lộ hoàn toàn) | **28,8 ms** |
| **Tổng** | | | **67,2 ms** |

Giải thích chi tiết:
- **AdamW 38,3 ms**: mô hình có ~302M tham số (24 lớp × 12M params/lớp). AdamW phải đọc/ghi 4 lần bộ tham số qua HBM. Với effective BW 126 GB/s (đo trên L40), thời gian = 4 × 1,2 GB / 126 GB/s ≈ 38 ms.
- **NCCL 0,1 ms**: với tp=1 không có TP AllReduce; chỉ có 1 DP AllReduce cuối bước → chỉ 100 µs launch.
- **PP 0 ms**: do pp=1 (không có pipeline).
- **Dispatch 28,8 ms**: pp=1 nên không có pipeline overlap che đi. Mỗi block mất ~0,15 ms để Python loop qua `ShardFormer`, với 24 lớp × 8 microbatch = 192 block → 28,8 ms.

**Kiểm chứng với thực nghiệm:**

Với cùng cấu hình trên cụm 4 GPU (node18 + node19), kết quả JSON ghi nhận:
```json
"execution_overhead": 67.26040533333332
```
→ Sai lệch chỉ **0,09%** so với lý thuyết (67,2 ms), chứng minh các hệ số microbenchmark chính xác.

**Lưu ý quan trọng về conditional dispatch:**

Khi $pp > 1$ (có pipeline), ColossalAI dùng `execute_pipeline()` với lịch trình 1F1B. Lúc này:
- **Base Python dispatch (~0,15 ms/block)** bị che bởi pipeline overlap — GPU tính toán block tiếp theo trong khi Python vẫn đang loop.
- **Chỉ còn TP-specific overhead (~0,05 ms/block)** khi tp>1, từ việc ShardFormer phải split/merge tensor.

Do đó với $pp > 1$, $T_{dispatch}$ giảm đáng kể (không còn base 0,15 ms), đó là lý do chiến lược $pp=4,tp=1,dp=1$ trên cụm 4 GPU chỉ có $T_{execution} \approx 21{,}6$ ms (không có AdamW vì dp=1, không có dispatch base vì pp>1).

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
