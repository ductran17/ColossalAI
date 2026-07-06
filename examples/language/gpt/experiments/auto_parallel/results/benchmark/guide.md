# Kế hoạch thực nghiệm cho Hybrid Auto-Parallel Planner

## 1. Mục tiêu thực nghiệm

Phần thực nghiệm nhằm đánh giá toàn diện giải pháp Hybrid Auto-Parallel Planner theo các mục tiêu chính sau:

1. Kiểm chứng độ ổn định và tính đại diện của pha profiling.
2. Đánh giá khả năng xếp hạng các chiến lược huấn luyện của Cost Model.
3. Đánh giá chất lượng chiến lược được planner lựa chọn so với chiến lược tốt nhất thực tế.
4. Kiểm chứng giá trị của cơ chế Plan Discovery khi cho phép sử dụng ít GPU hơn tổng tài nguyên khả dụng.
5. Đánh giá hiệu quả của các quy tắc pruning.
6. Phân tích đóng góp của từng thành phần trong giải pháp thông qua ablation study.
7. Đánh giá chi phí lập kế hoạch và hiệu năng end-to-end của toàn hệ thống.

---

## 2. Bộ workload đề xuất

### 2.1. W0 — GPT-2 Pilot

Dùng để kiểm tra pipeline, debug và đối chiếu với các thử nghiệm trước đây.

```json
{
  "model_type": "gpt2",
  "layers": 24,
  "hidden": 1024,
  "heads": 16,
  "seq": 256,
  "batch": 16,
  "microbatches": 8,
  "steps": 50,
  "dtype_bytes": 4
}
```

Mục đích:

- kiểm tra toàn bộ pipeline;
- kiểm tra tính đúng đắn của profiler, planner và training launcher;
- đối chiếu với dữ liệu thử nghiệm cũ;
- không dùng làm workload chính duy nhất của luận văn.

---

### 2.2. W1 — GPT-2 Medium

```json
{
  "model_type": "gpt2",
  "layers": 24,
  "hidden": 1024,
  "heads": 16,
  "seq": 512,
  "batch": 32,
  "microbatches": 8,
  "steps": 50,
  "dtype_bytes": 4
}
```

Mục đích:

- baseline chính;
- kiểm tra ranking giữa DP, TP và PP;
- đại diện cho kiến trúc decoder-only với MHA và MLP chuẩn.

---

### 2.3. W2 — GPT-2 Large / Memory Stress

```json
{
  "model_type": "gpt2",
  "layers": 36,
  "hidden": 2048,
  "heads": 32,
  "seq": 512,
  "batch": 16,
  "microbatches": 8,
  "steps": 50,
  "dtype_bytes": 4
}
```

Mục đích:

- tạo áp lực bộ nhớ;
- kiểm chứng memory pruning;
- tạo nhu cầu thực sự cho TP và PP;
- đánh giá planner trong trường hợp model lớn hơn đáng kể so với baseline.

---

### 2.4. W3 — LLaMA Medium

```json
{
  "model_type": "llama",
  "layers": 24,
  "hidden": 1536,
  "heads": 24,
  "num_key_value_heads": 8,
  "intermediate_size": 4096,
  "seq": 512,
  "batch": 32,
  "microbatches": 8,
  "steps": 50,
  "dtype_bytes": 4
}
```

Mục đích:

- kiểm chứng generic parameter model;
- kiểm tra GQA;
- kiểm tra gated MLP;
- kiểm tra cơ chế tự động trích xuất các hệ số kiến trúc từ `model.config`.

---

### 2.5. W4 — LLaMA Large / Long Sequence

```json
{
  "model_type": "llama",
  "layers": 32,
  "hidden": 2048,
  "heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 5504,
  "seq": 1024,
  "batch": 16,
  "microbatches": 8,
  "steps": 50,
  "dtype_bytes": 4
}
```

Mục đích:

- kiểm tra workload lớn;
- kiểm tra sequence dài;
- tăng activation memory;
- tăng chi phí PP communication;
- kiểm tra representative block profiling trong điều kiện áp lực bộ nhớ lớn hơn.

---

## 3. E1 — Kiểm chứng Profiler

### 3.1. E1.1 — Kiểm chứng tham số truyền thông α/β

Đo bốn tham số:

$$
\alpha_{\text{intra}}, \quad
\beta_{\text{intra}}, \quad
\alpha_{\text{inter}}, \quad
\beta_{\text{inter}}
$$

Protocol đề xuất:

- sử dụng 7–10 kích thước payload;
- lặp 10–20 lần cho mỗi kích thước;
- thực hiện 3 independent runs.

Các chỉ số cần báo cáo:

- mean;
- standard deviation;
- coefficient of variation;
- regression fit;
- so sánh intra-node và inter-node.

Bảng kết quả đề xuất:

| Loại kết nối | α mean | α std | β mean | β std | R² |
|---|---:|---:|---:|---:|---:|
| Intra-node | | | | | |
| Inter-node | | | | | |

Mục tiêu:

- chứng minh profiler phản ánh được khác biệt giữa liên kết nội node và mạng liên node;
- chứng minh mô hình tuyến tính `T(S) = α + βS` đủ phù hợp trong dải payload được sử dụng.

---

### 3.2. E1.2 — So sánh `T_block_isolated` và `T_block_repr`

Đo:

$$
T_{\text{block\_isolated}}
$$

và:

$$
T_{\text{block\_repr}}
$$

theo các giá trị:

$$
M \in \{4,8,16\}
$$

Báo cáo:

$$
\frac{T_{\text{block\_repr}}}
{T_{\text{block\_isolated}}}
$$

Bảng đề xuất:

| Workload | M | Isolated | Representative | Ratio |
|---|---:|---:|---:|---:|
| W1 | 4 | | | |
| W1 | 8 | | | |
| W1 | 16 | | | |
| W4 | 8 | | | |

Mục tiêu:

- chứng minh representative profiling là cần thiết;
- đánh giá ảnh hưởng của pipeline memory pressure đến thời gian thực thi một Transformer block.

---

## 4. E2 — Đánh giá khả năng ranking của Cost Model

Đây là thí nghiệm cốt lõi của luận văn.

### 4.1. Cách thực hiện

Trên 8 GPU:

1. Sinh toàn bộ candidate strategy.
2. Áp dụng pruning.
3. Với mỗi candidate còn lại:
   - tính estimated step time;
   - chạy actual training;
   - đo actual step time.
4. So sánh thứ hạng estimated và actual.

Thực hiện cho:

- W1;
- W2;
- W3;
- W4.

Bảng đề xuất:

| Plan `(tp, pp, dp)` | Estimated time | Actual time | Estimated rank | Actual rank |
|---|---:|---:|---:|---:|
| | | | | |
| | | | | |
| | | | | |

### 4.2. Metric chính

#### Spearman Rank Correlation

$$
\rho_s
=
\operatorname{corr}
(
\operatorname{rank}(T_{\text{est}}),
\operatorname{rank}(T_{\text{actual}})
)
$$

Đây là metric chính của luận văn.

#### Kendall's Tau

$$
\tau
$$

Dùng để đánh giá mức độ đúng của pairwise ordering.

#### Top-1 Accuracy

$$
\text{Top1}
=
\frac{
\#\text{workload planner chọn đúng actual-best}
}{
\#\text{workload}
}
$$

#### Top-k Accuracy

Khuyến nghị báo cáo thêm Top-3.

#### MAPE

$$
\operatorname{MAPE}
=
\frac{1}{N}
\sum_i
\left|
\frac{T_i^{est}-T_i^{actual}}
{T_i^{actual}}
\right|
$$

MAPE chỉ nên là metric phụ vì mục tiêu chính của Cost Model là ranking.

---

## 5. E3 — Đánh giá chất lượng Winner Plan

### 5.1. Oracle plan

Xác định chiến lược tốt nhất thực tế:

$$
p^*
=
\arg\min_p T_{\text{actual}}(p)
$$

Planner chọn:

$$
\hat{p}
=
\arg\min_p T_{\text{est}}(p)
$$

### 5.2. Regret

$$
\text{Regret}
=
\frac{
T_{\text{actual}}(\hat{p})
-
T_{\text{actual}}(p^*)
}{
T_{\text{actual}}(p^*)
}
\times 100\%
$$

Bảng đề xuất:

| Workload | Planner plan | Oracle best | Top-1 đúng? | Planner actual | Oracle actual | Regret |
|---|---|---|---|---:|---:|---:|
| W1 | | | | | | |
| W2 | | | | | | |
| W3 | | | | | | |
| W4 | | | | | | |

Mục tiêu:

- đánh giá planner có chọn đúng best plan hay không;
- nếu không chọn đúng, đánh giá mức chênh lệch hiệu năng so với oracle.

---

## 6. E4 — Kiểm chứng Plan Discovery

Mục tiêu của thí nghiệm là chứng minh:

> Sử dụng nhiều GPU hơn không nhất thiết làm training nhanh hơn.

### 6.1. Cách chạy

Dùng toàn bộ cụm 10 GPU và cho planner thử các subset hợp lệ theo node layout thực tế.

Ví dụ:

$$
2,\ 4,\ 6,\ 8,\ 10\ \text{GPU}
$$

Khuyến nghị chạy với:

- W2 GPT-2 Large;
- W4 LLaMA Large.

Bảng đề xuất:

| GPU subset | GPU count | Best plan | Estimated time | Actual time |
|---|---:|---|---:|---:|
| | 2 | | | |
| | 4 | | | |
| | 6 | | | |
| | 8 | | | |
| Full cluster | 10 | | | |

Mục tiêu:

- kiểm tra tính không đơn điệu của step time theo số GPU;
- kiểm chứng giá trị của subset discovery;
- kiểm tra liệu có trường hợp `T*8 < T*10` hay không.

---

## 7. E5 — Đánh giá hiệu quả Pruning

### 7.1. Số lượng candidate sau từng rule

Đo:

$$
N_{\text{initial}}
\rightarrow
N_{\text{after Rule 1}}
\rightarrow
N_{\text{after Rule 2}}
\rightarrow
N_{\text{after Rule 3}}
$$

Bảng đề xuất:

| Workload | Initial | Sau TP rule | Sau layer rule | Sau memory rule | Final |
|---|---:|---:|---:|---:|---:|
| W1 | | | | | |
| W2 | | | | | |
| W3 | | | | | |
| W4 | | | | | |

### 7.2. Pruning Ratio

$$
\text{Pruning Ratio}
=
1-
\frac{N_{\text{final}}}
{N_{\text{initial}}}
$$

### 7.3. Optimal-plan Preservation

$$
\text{Optimal Recall}
=
\frac{
\#\text{workload mà oracle-best không bị prune}
}{
\#\text{workload}
}
$$

Mục tiêu:

- chứng minh pruning giảm không gian tìm kiếm;
- chứng minh pruning không loại nhầm actual-best plan.

---

## 8. E6 — Ablation Study

Mục tiêu là đánh giá đóng góp của từng thành phần trong giải pháp.

### 8.1. Các biến thể

#### Full Model

Toàn bộ giải pháp.

#### A1 — Không topology-aware

Không phân biệt:

$$
\alpha_{\text{intra}},\beta_{\text{intra}}
$$

và:

$$
\alpha_{\text{inter}},\beta_{\text{inter}}
$$

#### A2 — Không representative profiling

Luôn dùng:

$$
T_{\text{block\_isolated}}
$$

#### A3 — Không pruning

Đưa toàn bộ candidate vào bước scoring.

#### A4 — Không bubble term

Loại bỏ:

$$
T_{\text{bubble}}
$$

### 8.2. Metric

| Variant | Spearman ρ | Kendall τ | Top-1 | Mean Regret | Planning Time |
|---|---:|---:|---:|---:|---:|
| Full | | | | | |
| -Topology | | | | | |
| -Representative block | | | | | |
| -Pruning | | | | | |
| -Bubble | | | | | |

Nếu thời gian hạn chế, ưu tiên:

- Full;
- -Topology;
- -Representative block;
- -Pruning.

---

## 9. E7 — Planning Overhead và End-to-End

### 9.1. Planning Overhead

Đo riêng:

$$
T_{\text{profiling}}
$$

$$
T_{\text{discovery}}
$$

$$
T_{\text{pruning}}
$$

$$
T_{\text{topology}}
$$

$$
T_{\text{cost}}
$$

$$
T_{\text{total planning}}
$$

Bảng đề xuất:

| Thành phần | Time |
|---|---:|
| Node discovery | |
| Network profiling | |
| Block profiling | |
| Plan discovery | |
| Pruning | |
| Topology classification | |
| Cost evaluation | |
| Total planning | |

So sánh với exhaustive empirical benchmarking:

$$
T_{\text{planner}}
\ll
T_{\text{exhaustive}}
$$

Có thể báo cáo:

$$
\text{Planning Speedup}
=
\frac{
T_{\text{exhaustive}}
}{
T_{\text{planner}}
}
$$

---

### 9.2. End-to-End Performance

So sánh các phương pháp:

- DP-only;
- Use-all-GPU heuristic;
- Balanced heuristic;
- Proposed Planner;
- Oracle.

Bảng đề xuất:

| Method | Actual Step Time | Tokens/s | Gap to Oracle |
|---|---:|---:|---:|
| DP-only | | | |
| Use-all-GPU | | | |
| Balanced heuristic | | | |
| Proposed Planner | | | |
| Oracle | | | |

Các metric:

#### Step Time

$$
T_{\text{step}}
$$

#### Throughput

$$
\text{tokens/s}
=
\frac{
B_{\text{global}}
\times Seq
}{
T_{\text{step}}
}
$$

#### Speedup

$$
\text{Speedup}
=
\frac{
T_{\text{baseline}}
}{
T_{\text{proposed}}
}
$$

---

## 10. Cách tổ chức thí nghiệm trên cụm 10 GPU L40

### 10.1. Nhóm A — Main Experiment

Dùng:

$$
8\ \text{GPU}
$$

Workload:

- W1;
- W2;
- W3;
- W4.

Thực hiện:

- E2 Ranking Accuracy;
- E3 Winner Plan Quality;
- E5 Pruning;
- E6 Ablation;
- E7 Planning Overhead.

Lý do chọn 8 GPU:

$$
8 = 2 \times 2 \times 2
$$

cho phép kiểm tra đầy đủ 3D Parallelism với:

$$
(tp,pp,dp)=(2,2,2)
$$

và có không gian candidate đa dạng.

---

### 10.2. Nhóm B — Resource Subset Experiment

Dùng:

$$
2,\ 4,\ 6,\ 8,\ 10\ \text{GPU}
$$

Workload:

- W2;
- W4.

Thực hiện:

- E4 Plan Discovery.

---

### 10.3. Nhóm C — Profiler Experiment

Dùng:

- 1 GPU cho block profiling;
- 2 GPU cùng node cho intra-node profiling;
- 2 GPU khác node cho inter-node profiling.

Thực hiện:

- E1.

---

## 11. Protocol đo Actual Training Time

Với mỗi plan:

1. Khởi tạo training.
2. Warm-up.
3. Đo nhiều step liên tiếp.
4. Lặp lại độc lập.

Protocol khuyến nghị:

```text
10 warm-up steps
50 measured steps
3 independent runs
```

Giá trị actual time:

$$
T_{\text{actual}}
=
\operatorname{median}
(
T_{\text{measured steps}}
)
$$

Nên báo cáo thêm:

- mean;
- standard deviation;
- coefficient of variation.

---

## 12. Quy tắc so sánh batch giữa các plan

Trước khi chạy toàn bộ thí nghiệm cần xác định rõ semantics của:

- `batch`;
- `microbatch_size`;
- `microbatches`;
- `dp`.

### Khuyến nghị

Cố định:

$$
B_{\text{global}}
$$

và:

$$
B_{\mu}
$$

sau đó tính:

$$
M(dp)
=
\frac{
B_{\text{global}}
}{
dp \times B_{\mu}
}
$$

Điều này đảm bảo mọi plan xử lý cùng lượng dữ liệu trong một training step.

Nếu giữ `M` cố định cho mọi giá trị `dp`, global batch có thể thay đổi theo plan. Khi đó không nên chỉ so sánh raw step time mà phải báo cáo thêm:

$$
\text{tokens/s}
$$

---

## 13. Bộ thí nghiệm tối thiểu cần hoàn thành

Nếu thời gian thực nghiệm hạn chế, ưu tiên theo thứ tự sau:

| ID | Thí nghiệm | Mức độ quan trọng |
|---|---|---|
| E1 | Profiler Validation | Cần |
| E2 | Cost Model Ranking Accuracy | Cốt lõi |
| E3 | Winner Plan Quality và Regret | Cốt lõi |
| E4 | Plan Discovery / GPU Subset | Rất nên |
| E5 | Pruning Effectiveness | Cần |
| E6 | Ablation Study | Rất nên |
| E7 | Planning Overhead và End-to-End | Cốt lõi |

Trọng tâm đánh giá của luận văn nên theo thứ tự:

$$
\boxed{
\text{Rank Correlation}
\rightarrow
\text{Winner Quality}
\rightarrow
\text{Regret}
\rightarrow
\text{Planning Time}
}
$$

MAPE chỉ nên đóng vai trò metric phụ.
