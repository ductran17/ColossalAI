## 3.1 Kiến trúc tổng thể hệ thống

### 3.1.1 Mục tiêu và phạm vi

Hệ thống được đề xuất nhằm giải quyết bài toán **lập kế hoạch song song tự động** cho huấn luyện mô hình ngôn ngữ lớn trên cụm GPU phân tán. Thay vì yêu cầu ngườidùng chỉ định thủ công các siêu tham số song song $(pp, tp, dp)$, hệ thống tự động đo thông số phần cứng, ước lượng thờ i gian thực thi của từng chiến lược, và chọn ra phương án tối ưu trong vòng vài giây.

**Phạm vi của luận văn tập trung vào:**
- Kiến trúc **decoder-only** GPT-style (self-attention + feed-forward).
- Cụm GPU **không đồng nhất** (heterogeneous), kết nối qua Ethernet thông thường.
- Ba chiến lược song song: Tensor Parallel (TP), Pipeline Parallel (PP), Data Parallel (DP).

**Nguyên tắc thiết kế quan trọng nhất:** luận văn **không sửa đổi mã nguồn lõi** của framework ColossalAI. Các thành phần tự phát triển chỉ tương tác với framework thông qua giao diện công khai (public API), đảm bảo tính module hóa và khả năng tái sử dụng.

---

### 3.1.2 Kiến trúc phân tầng

Hệ thống được tổ chức theo bốn tầng chức năng, từ trừu tượng đến cụ thể:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 1 — NGƯỜI DÙNG VÀ CẤU HÌNH                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐                      │
│  │ Mô hình     │  │ Cụm GPU     │  │ Siêu tham số    │                      │
│  │ (GPT-2)     │  │ (6 node)    │  │ (batch, M, ...) │                      │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘                      │
│         └─────────────────┴───────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🟦 TẦNG 2 — BỘ LẬP KẾ HOẠCH TỰ ĐỘNG (Đề xuất trong luận văn)               │
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│   │ Profiler    │    │ Cost Model  │    │ Planner     │                     │
│   │ (Đo lường)  │───▶│ (Dự đoán)   │◀───│ (Tìm kiếm)  │                     │
│   └─────────────┘    └─────────────┘    └─────────────┘                     │
│          │                    ▲                  │                           │
│          └────────────────────┴──────────────────┘                           │
│                  ClusterProfile + TopologyInfo                               │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ Orchestrator — Điều phối thử nhiều world_size và dp_outside        │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ (pp, tp, dp, dp_outside)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 🟩 TẦNG 3 — THỰC THI PHÂN TÁN (ColossalAI / PyTorch)                        │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│   │ShardFormer  │  │PipelineStage│  │ DDP / ZeRO  │  │ 1F1B Scheduler  │   │
│   │ (TP)        │  │ Manager(PP) │  │   (DP)      │  │ (Pipeline)      │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │
│                                                                              │
│   Tổng hợp: HybridParallelPlugin                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TẦNG 4 — PHẦN CỨNG                                                          │
│   Node11 (A6000)  Node15 (L40S)  Node16 (L40S)  Node18 (L40)  Node20 (A30) │
│   ──────────────────────── Ethernet 10GbE (bonded) ──────────────────────── │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Tầng 1 — Ngườidùng và cấu hình.** Ngườidùng chỉ cần cung cấp cấu hình mô hình (số lớp, kích thước ẩn), định nghĩa cụm GPU (qua file cấu hình hoặc biến môi trường của torchrun), và các siêu tham số huấn luyện (kích thước batch, số microbatch). Hệ thống không yêu cầu bất kỳ kiến thức nào về phân tán.

**Tầng 2 — Bộ lập kế hoạch tự động.** Đây là phần đóng góp chính của luận văn, bao gồm bốn module: (1) *Profiler* đo tốc độ tính toán và truyền thông trên phần cứng thực tế; (2) *Cost Model* ước lượng thờ i gian một bước huấn luyện từ các đại lượng đo được; (3) *Topology* phân loại giao tiếp nào nằm trong node (nhanh) và giao tiếp nào vươn qua node (chậm); (4) *Planner* tìm kiếm và đánh giá mọi chiến lược $(pp, tp, dp)$ khả thi. Orchestrator điều phối toàn bộ quy trình, tự động thử nhiều kích thước cụm con và cả hai cách sắp xếp chiều lưới (dp_outside).

**Tầng 3 — Thực thi phân tán.** Tầng này hoàn toàn dựa trên framework ColossalAI có sẵn. `HybridParallelPlugin` nhận bộ ba $(pp, tp, dp)$ do planner chọn và tự động triển khai: `ShardFormer` để shard tham số theo TP, `PipelineStageManager` để chia mô hình thành các stage PP, và `DDP` hoặc `ZeRO` để đồng bộ gradient theo DP. Lịch trình pipeline mặc định là **1F1B non-interleaved**.

**Tầng 4 — Phần cứng.** Cụm thực nghiệm gồm sáu node với bốn loại GPU khác nhau (NVIDIA A6000, L40S, L40, A30), kết nối qua bonded 10 GbE Ethernet. Sự không đồng nhất này là thách thức chính mà cost model phải đối mặt.

---

### 3.1.3 Luồng xử lý chính: ba pha

Hệ thống thực hiện theo ba pha liên tiếp, được điều phối bởi Orchestrator:

**Pha 1 — Đo lường cụm (Profiling).**

Tất cả các rank GPU thực hiện đo lường đồng thờ i. Profiler đo bốn đại lượng: (a) độ trễ và nghịch đảo băng thông của liên kết intra-node (PCIe); (b) tương tự cho liên kết cross-node (Ethernet); (c) thờ i gian forward+backward của một khối transformer trong điều kiện bộ nhớ sạch; (d) thờ i gian tương tự nhưng với áp lực bộ nhớ từ $M$ microbatch. Các giá trị được đồng bộ hóa qua `all_reduce` với toán tử `MAX` để mọi rank nhận được cùng một profile, trong đó GPU chậm nhất quyết định tốc độ toàn hệ thống. Pha này hoàn thành trong khoảng hai giây.

**Pha 2 — Lập kế hoạch (Planning).**

Orchestrator xây dựng không gian tìm kiếm bằng cách thử mọi kích thước cụm con (prefix world sizes) và cả hai giá trị `dp_outside` (True và False). Với mỗi cấu hình, Planner liệt kê tất cả các bộ ba $(pp, tp, dp)$ thỏa mãn $pp \times tp \times dp = \text{world\_size}$, loại bỏ các chiến lược bất khả thi theo bốn quy tắc (TP vượt quá số GPU trên một node, số lớp không chia hết cho số stage, batch không chia hết cho số microbatch, vượt ngân sách bộ nhớ), sau đó tính điểm từng chiến lược còn lại bằng cost model. Chiến lược có thờ i gian ước lượng nhỏ nhất được chọn làm winner. Nếu winner sử dụng ít GPU hơn số GPU hiện có, hệ thống ghi cờ tự động relaunch.

**Pha 3 — Huấn luyện (Training).**

Orchestrator khởi tạo `HybridParallelPlugin` với bộ ba $(pp, tp, dp)$ do pha 2 chọn. Plugin tự động shard mô hình, thiết lập các nhóm process, và bắt đầu huấn luyện. Sau khi chạy một số bước ổn định, hệ thống so sánh thờ i gian thực tế đo được với thờ i gian ước lượng từ cost model, xuất báo cáo dạng JSON để đánh giá độ chính xác.

---

### 3.1.4 Ranh giới giữa luận văn và framework

Một điểm quan trọng cần làm rõ là ranh giới giữa công việc tự phát triển và công cụ có sẵn. Bảng dưới đây phân loại từng thành phần:

| Thành phần | Nguồn gốc | Vai trò trong hệ thống |
|-----------|-----------|----------------------|
| **Profiler** (`profiler.py`) | 🟦 Đề xuất trong luận văn | Đo $\alpha, \beta$ (intra/cross), $T_{block}$, $T_{block}^{repr}$, bộ nhớ trống. Đồng bộ hóa giá trị giữa các rank. |
| **Topology** (`topology.py`) | 🟦 Đề xuất trong luận văn | Phân loại giao tiếp TP/PP/DP là intra-node hay cross-node dựa trên cách sắp xếp rank của `HybridParallelPlugin`. |
| **Cost Model** (`cost_model.py`) | 🟦 Đề xuất trong luận văn | Công thức 7 thành phần ước lượng thờ i gian một bước. Không dùng FLOPs giả định, chỉ dùng $T_{block}$ đo thực tế. |
| **Planner** (`search.py`) | 🟦 Đề xuất trong luận văn | Liệt kê, loại bỏ, và chấm điểm các chiến lược $(pp, tp, dp)$. Xử lý một cặp `(world_size, dp_outside)` mỗi lần gọi. |
| **Orchestrator** (`run_auto_hybrid_parallel.py`) | 🟦 Đề xuất trong luận văn | Điều phối 3 pha, thử nhiều combo, so sánh kết quả, tự động relaunch, workaround khi $pp=1$. |
| `HybridParallelPlugin` | 🟩 ColossalAI | Nhận $(pp, tp, dp)$ và triển khai TP+PP+DP. Giao diện duy nhất giữa planner và framework. |
| `ShardFormer` | 🟩 ColossalAI | Tự động shard mô hình theo TP (thay `nn.Linear` bằng `Col`/`Row` parallel layers). Hỗ trợ 20+ kiến trúc. |
| `PipelineStageManager` | 🟩 ColossalAI | Quản lý PP stage, tạo process group mesh $(dp, pp, tp)$, điều phối P2P. |
| `OneForwardOneBackwardSchedule` | 🟩 ColossalAI | Lịch trình 1F1B non-interleaved (fill → steady → drain). Mặc định `pp_style="1f1b"`. |
| DDP / ZeRO | 🟩 PyTorch / ColossalAI | Đồng bộ gradient khi $dp > 1$. |
| GPT2Config / GPT2LMHeadModel | 🟩 HuggingFace | Mô hình ngôn ngữ dùng cho thực nghiệm. |

Các module 🟦 (màu xanh dương) là phần **đóng góp khoa học** của luận văn. Các module 🟩 (màu xanh lá) là **cơ sở hạ tầng** được tận dụng. Sự tách biệt này đảm bảo rằng bộ lập kế hoạch có thể hoạt động độc lập và, trong tương lai, có thể được tích hợp với các framework khác (ví dụ: DeepSpeed, Megatron-LM) mà không cần sửa đổi lõi.

---

### 3.1.5 Dữ liệu luân chuyển giữa các module

Dữ liệu chính di chuyển trong hệ thống gồm ba đối tượng trung tâm:

1. **ModelConfig:** Mô tả kiến trúc mô hình (số lớp, kích thước ẩn, độ dài chuỗi, kích thước batch, độ chính xác). Đối tượng này đi từ ngườidùng → orchestrator → profiler → planner → cost model.

2. **ClusterProfile:** Chứa kết quả đo lường từ profiler ($\alpha_{intra}, \beta_{intra}, \alpha_{cross}, \beta_{cross}, T_{block}, T_{block}^{repr}$, bộ nhớ trống). Đây là đầu vào duy nhất mà cost model cần để tính toán — thay thế cho mọi giả định lý thuyết về FLOPs hay băng thông. ClusterProfile được all-reduce nên đồng nhất trên mọi rank.

3. **PlanResult:** Chứa chiến lược thắng $(pp, tp, dp)$ cùng với `CostBreakdown` (phân rã 7 thành phần) và `TopologyInfo` (phân loại intra/cross). Orchestrator thu thập nhiều `PlanResult` từ các combo khác nhau, so sánh, và chọn winner toàn cục.

Luồng dữ liệu có thể tóm tắt như sau:

```
ModelConfig ──► Profiler ──► ClusterProfile ──┐
                                              ├──► Planner ──► PlanResult ──► HybridParallelPlugin
ModelConfig ──► Topology ◄─── node_gpus ──────┘                (training)
```

---

### 3.1.6 Điểm đặc biệt của thiết kế

**Thứ nhất, tính độc lập với framework.** Bộ lập kế hoạch hoạt động như một **hộp đen bên ngoài** framework. Nó không can thiệp vào cách `HybridParallelPlugin` shard mô hình hay lập lịch pipeline. Điều này có hai lợi ích: (a) khi ColossalAI cập nhật phiên bản mới, plugin vẫn hoạt động bình thường; (b) bộ lập kế hoạch có thể được chuyển sang framework khác chỉ bằng cách thay đổi giao diện kết nối.

**Thứ hai, đo lường thay vì giả định.** Cost model không sử dụng peak FLOPs từ datasheet GPU hay mô hình lý thuyết về băng thông mạng. Thay vào đó, nó đo $T_{block}$ trên GPU chậm nhất trong cụm và đo $\alpha, \beta$ trên liên kết thực tế. Điều này đặc biệt quan trọng trên cụm heterogeneous, nơi mỗi loại GPU có hiệu suất khác nhau và liên kết mạng thực tế thường thấp hơn lý thuyết.

**Thứ ba, đo lường theo ngữ cảnh (context-aware).** Cost model sử dụng hai giá trị $T_{block}$ khác nhau tùy theo chiến lược: $T_{block}$ cô lập (bộ nhớ sạch) cho $pp = 1$, và $T_{block}^{repr}$ (với áp lực bộ nhớ từ $M$ microbatch) cho $pp > 1$. Điều này khắc phục hiện tượng đánh giá thấp ~1,9 lần khi dùng $T_{block}$ cô lập cho pipeline parallelism.

**Thứ tư, tìm kiếm không gian con tối ưu.** Thay vì chỉ tìm chiến lược cho toàn bộ cụm, orchestrator tự động thử các kích thước cụm con (ví dụ: 2, 4, 6 GPU trên cụm 6 GPU). Điều này giúp phát hiện trường hợp dùng ít GPU hơn lại nhanh hơn do tránh được giao tiếp cross-node đắt đỏ trên Ethernet chậm.

---

### 3.1.7 Hình ảnh đề xuất cho luận văn

**Hình 3.1 — Kiến trúc tổng thể hệ thống Auto 3D Parallel.**

Sơ đồ khối 4 tầng theo chiều dọc. Tầng 1 (màu xám nhạt): ngườidùng + cấu hình. Tầng 2 (màu xanh dương): 4 module tự phát triển (Profiler, Topology, Cost Model, Planner) nằm trong một hộp lớn có nhãn "Bộ lập kế hoạch tự động", với Orchestrator ở trên cùng. Tầng 3 (màu xanh lá): `HybridParallelPlugin` bao bọc 4 thành phần con (ShardFormer, PipelineStageManager, DDP/ZeRO, 1F1B Scheduler). Tầng 4 (màu xám đậm): các node GPU với nhãn card đồ họa. Các mũi tên chỉ luồng dữ liệu chính từ trên xuống dưới, với nhãn "$(pp, tp, dp)$" ở ranh giới giữa tầng 2 và 3.

**Hình 3.2 — Luồng xử lý ba pha.**

Sơ đồ tuần tự (horizontal flow) gồm 3 hộp chính: (1) "Pha 1: Đo lường" với icon đồng hồ bấm giờ và output `ClusterProfile`; (2) "Pha 2: Lập kế hoạch" với icon kính lúp và output `(pp, tp, dp)`; (3) "Pha 3: Huấn luyện" với icon GPU và output `Validation Report`. Các mũi tên nối liên tiếp 3 hộp. Bên dưới hộp 2, thêm một hộp nhỏ "Thử nhiều combo" với mũi tên lặp ngược lại hộp 2 để minh họa việc thử nhiều `world_size` và `dp_outside`.

**Hình 3.3 — Ranh giới trách nhiệm giữa luận văn và framework.**

Sơ đồ hai cột song song. Cột trái (xanh dương, nhãn "Luận văn đề xuất") chứa 4 khối: Hiểu (Profiler), Dự đoán (Cost Model), Chọn (Planner), Điều phối (Orchestrator). Cột phải (xanh lá, nhãn "ColossalAI Framework") chứa 4 khối: Chia (ShardFormer), Gửi (PipelineStageManager), Đồng bộ (DDP/ZeRO), Lập lịch (1F1B Scheduler). Mũi tên từ cột trái sang phải mang nhãn "$(pp, tp, dp)$" thể hiện việc luận văn chỉ **quyết định** chiến lược, framework **thực thi** chiến lược.
