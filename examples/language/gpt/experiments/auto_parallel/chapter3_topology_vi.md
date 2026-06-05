## 3.2.3 Phân loại topology liên lạc (*Topology Classification*)

### Vấn đề đặt ra

Một chiến lược song song $(pp, tp, dp)$ có thể được triển khai theo nhiều cách khác nhau trên cùng một cụm máy. Ví dụ, chiến lược $pp=2, tp=2, dp=2$ trên 8 GPU có thể xếp đặt các *rank* theo nhiều thứ tự khác nhau. Tùy thuộc vào cách xếp đặt này, một nhóm giao tiếp (ví dụ: nhóm *Tensor Parallel*) có thể nằm hoàn toàn trong một máy chủ vật lý (tốc độ PCIe ~26 GB/s) hoặc vươn qua nhiều máy chủ qua Ethernet (tốc độ ~2,7 GB/s, chậm hơn gần 10 lần).

Nếu cost model không biết được liệu một nhóm giao tiếp là *intra-node* hay *cross-node*, nó sẽ tính sai thời gian truyền thông — và do đó chọn nhầm chiến lược tối ưu.

### Giải pháp: phân loại nhóm giao tiếp

Module topology nhận đầu vào là:
- Bố trí vật lý của cụm: số GPU trên mỗi node (ví dụ: node18 có 2 GPU, node20 có 4 GPU, node16 có 2 GPU)
- Chiến lược song song $(pp, tp, dp)$
- Cách sắp xếp *rank* theo chiều ngoài cùng của lưới (mesh): `dp_outside` hay `pp_outside`

Từ đó, module xác định với từng loại giao tiếp (TP, PP, DP), các nhóm rank có nằm chung một máy vật lý hay không. Kết quả là ba giá trị boolean:

| Giá trị | Ý nghĩa |
|---------|---------|
| `tp_intra_node = True` | Mọi nhóm TP đều nằm trong cùng một máy → dùng α_intra, β_intra (tốc độ cao) |
| `pp_intra_node = False` | Ít nhất một ranh giới PP vươn qua hai máy → dùng α_cross, β_cross (tốc độ chậm) |
| `dp_intra_node = True/False` | Tùy cách xếp rank, DP có thể nằm trong máy hoặc vươn qua máy |

### Ví dụ minh họa

Xét cụm 3 node `[2, 4, 2]` GPU với chiến lược $pp=2, tp=2, dp=2$.

**Với `dp_outside=True` (mặc định của ColossalAI):**

Lưới process group có dạng $(dp, pp, tp)$. Công thức xếp rank:

$$\text{rank} = dp_{\text{rank}} \times (pp \times tp) + pp_{\text{rank}} \times tp + tp_{\text{rank}}$$

Các rank được phân bổ như sau:

| dp | pp | tp | Rank | Máy vật lý |
|:--:|:--:|:--:|:----:|:----------:|
| 0 | 0 | 0 | 0 | node18 |
| 0 | 0 | 1 | 1 | node18 |
| 0 | 1 | 0 | 2 | node20 |
| 0 | 1 | 1 | 3 | node20 |
| 1 | 0 | 0 | 4 | node20 |
| 1 | 0 | 1 | 5 | node20 |
| 1 | 1 | 0 | 6 | node16 |
| 1 | 1 | 1 | 7 | node16 |

Từ bảng này, ta suy ra:
- **Nhóm TP** (cùng dp, cùng pp): $\{0,1\}$ trên node18, $\{2,3\}$ trên node20... → **toàn bộ intra-node** ✓
- **Ranh giới PP** (cùng dp, cùng tp): $0 \rightarrow 2$ (node18→node20), $4 \rightarrow 6$ (node20→node16) → **ít nhất một nhánh cross-node** ✗
- **Nhóm DP** (cùng pp, cùng tp): $\{0,4\}$ (node18 và node20) → **cross-node** ✗

Kết quả: `tp_intra=True`, `pp_intra=False`, `dp_intra=False`.

**Điều này có ý nghĩa gì cho cost model?**

Cùng một chiến lược $pp=2, tp=2, dp=2$, nếu ta đổi sang `dp_outside=False` (lưới $(pp, dp, tp)$), cách xếp rank thay đổi hoàn toàn và các ranh giới PP/DP sẽ rơi vào các cặp node khác nhau. Do đó, cùng một bộ $(pp, tp, dp)$ có thể cho ra **ước lượng thời gian rất khác nhau** tùy thuộc cách sắp xếp lưới.

**Lưu ý quan trọng:** Cách xếp rank này không do module topology tự đặt ra, mà là do chính `HybridParallelPlugin` của ColossalAI quy định (qua lớp `ProcessGroupMesh` sử dụng `np.ravel_multi_index` theo thứ tự C). Module topology chỉ **tái tạo lại (replicate)** cách xếp rank này để dự đoán các nhóm giao tiếp sẽ rơi vào đâu. Nếu ColossalAI thay đổi cách xếp rank trong tương lai, module topology cũng phải cập nhật theo để cost model vẫn tính đúng.

### Ý nghĩa đối với cost model

Module topology không tính thời gian — nó chỉ trả lời câu hỏi: *"Giao tiếp này chạy trên PCIe nhanh hay Ethernet chậm?"*

Trả lời đúng câu hỏi này là tiền đề để cost model chọn đúng cặp $(\alpha, \beta)$ khi ước lượng $T_{tp\_comm}, T_{pp\_comm}, T_{dp\_comm}$. Nếu nhầm lẫn giữa intra-node và cross-node, sai số có thể lên đến **10 lần** (26 GB/s so với 2,7 GB/s), dẫn đến chọn sai chiến lược tối ưu.
