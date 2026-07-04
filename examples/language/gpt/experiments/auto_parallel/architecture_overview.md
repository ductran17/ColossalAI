# Kiến trúc tổng thể giải pháp Auto 3D Parallel

> **Chú thích màu sắc / phân loại:**
> - 🟦 **Xanh dương** = Code tự phát triển (đề xuất trong luận văn)
> - 🟩 **Xanh lá** = Code có sẵn trong ColossalAI / PyTorch / HuggingFace
> - 🟨 **Vàng** = Dữ liệu / cấu hình đầu vào
> - ➡️ Mũi tên liền = Luồng điều khiển (gọi hàm)
> - ➖➡️ Mũi tên nét đứt = Luồng dữ liệu (truyền object)

---

## 1. Tổng quan hệ thống (High-Level Flow)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NGƯỜI DÙNG (User)                                        │
│  ./launch_nodes.sh node18 node19 node15 --auto --layers 24 --hidden 1024 ...    │
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│  🟦 ORCHESTRATOR (run_auto_hybrid_parallel.py)                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐│
│  │ Phase 1: Profile    Phase 2: Plan Search    Phase 3: Train                  ││
│  │  (~2 giây)          (<1 ms per combo)       (nhiều phút)                    ││
│  └─────────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────┬───────────────────────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────┐      ┌──────────────────┐      ┌──────────────────────┐
│ 🟦 PROFILER  │      │ 🟦 PLANNER       │      │ 🟩 HybridParallel    │
│ (profiler.py)│      │ (search.py)      │      │    Plugin            │
│              │      │                  │      │    (ColossalAI)      │
│ ┌──────────┐ │      │ ┌──────────────┐ │      │                      │
│ │_measure_ │ │      │ │_all_candidates│ │      │ ┌─────────────────┐  │
│ │  p2p()   │ │      │ │  (enumerate)  │ │      │ │  ShardFormer    │  │
│ └────┬─────┘ │      │ └──────┬───────┘ │      │ │  (TP sharding)  │  │
│      │       │      │        │         │      │ └─────────────────┘  │
│ ┌────▼─────┐ │      │ ┌──────▼───────┐ │      │ ┌─────────────────┐  │
│ │_measure_ │ │      │ │ Prune rules   │ │      │ │PipelineStageMgr │  │
│ │T_block() │ │      │ │ (4 điều kiện) │ │      │ │   (PP stages)   │  │
│ └────┬─────┘ │      │ └──────┬───────┘ │      │ └─────────────────┘  │
│      │       │      │        │         │      │ ┌─────────────────┐  │
│ ┌────▼────────┐     │ ┌──────▼───────┐ │      │ │  1F1B Scheduler │  │
│ │_measure_    │     │ │ Cost Model    │ │      │ │  (non-interlv)  │  │
│ │T_block_repr │     │ │ (7-term eq)   │ │      │ └─────────────────┘  │
│ └─────────────┘     │ └──────┬───────┘ │      │ ┌─────────────────┐  │
│                     │        │         │      │ │   DDP / ZeRO    │  │
│  Output:            │   ┌────▼────┐    │      │ │  (DP grad sync) │  │
│  ClusterProfile     │   │ Select  │    │      │ └─────────────────┘  │
│  (α, β, T_block)    │   │  min()  │    │      └──────────────────────┘
│                     │   └────┬────┘    │
│                     │        │         │
│                     │   PlanResult     │
│                     │   (best plan)    │
└─────────────────────┘        │         └──────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  🟦 Orchestrator     │
                    │  So sánh tất cả      │
                    │  PlanResult từ       │
                    │  mọi (world_size,    │
                    │  dp_outside) combo   │
                    │  → Chọn winner       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Khởi tạo plugin với  │
                    │ pp, tp, dp, dp_outside│
                    │ từ winner plan       │
                    └──────────────────────┘
```

---

## 2. Chi tiết Phase 2: Planner Loop (nhiều combo)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    🟦 ORCHESTRATOR LOOP (trong Phase 2)                       │
│                                                                              │
│   for world_size in [2, 4, 6]:          ← prefix world sizes                │
│     for dp_outside in [True, False]:    ← cả 2 chiều lưới                  │
│                                                                              │
│       ┌──────────────────────────────────────────────────────────────────┐   │
│       │ 🟦 PLANNER (auto_plan(world_size, dp_outside))                    │   │
│       │                                                                │   │
│       │  Step 1: _all_candidates(world_size)                           │   │
│       │     → Liệt kê mọi (pp, tp, dp): pp×tp×dp = world_size        │   │
│       │                                                                │   │
│       │  Step 2: Prune (4 rules)                                       │   │
│       │     ├── tp > min_gpus_per_node  → LOẠI (cross-node TP)        │   │
│       │     ├── layers % pp != 0        → LOẠI                       │   │
│       │     ├── batch % M != 0          → LOẠI                       │   │
│       │     └── _fits_in_memory() == F  → LOẠI (OOM)                 │   │
│       │                                                                │   │
│       │  Step 3: Score từng candidate còn lại                        │   │
│       │     ┌──────────────────────┐                                 │   │
│       │     │ 🟦 TOPOLOGY          │                                 │   │
│       │     │ classify_comms()     │                                 │   │
│       │     │ Input: node_gpus, pp,│                                 │   │
│       │     │        tp, dp,       │                                 │   │
│       │     │        dp_outside    │                                 │   │
│       │     │ Output: TopologyInfo │                                 │   │
│       │     │ (tp_intra, pp_intra, │                                 │   │
│       │     │  dp_intra: bool)     │                                 │   │
│       │     └──────────┬───────────┘                                 │   │
│       │                │                                             │   │
│       │     ┌──────────▼───────────┐                                 │   │
│       │     │ 🟦 COST MODEL        │                                 │   │
│       │     │ estimate_step_time() │                                 │   │
│       │     │ Input: cfg, pp, tp,  │                                 │   │
│       │     │   dp, profile,       │                                 │   │
│       │     │   topology, M        │                                 │   │
│       │     │ Output: CostBreakdown│                                 │   │
│       │     │ (7 terms: T_compute, │                                 │   │
│       │     │  T_bubble, T_tp_comm,│                                 │   │
│       │     │  T_pp_comm,          │                                 │   │
│       │     │  T_dp_comm,          │                                 │   │
│       │     │  T_step_overhead,    │                                 │   │
│       │     │  T_execution)        │                                 │   │
│       │     └──────────┬───────────┘                                 │   │
│       │                │                                             │   │
│       │  Step 4: Select min(cost.total)                              │   │
│       │     → PlanResult (pp, tp, dp, cost, topology)              │   │
│       │                                                                │   │
│       └────────────────────────┬───────────────────────────────────────┘   │
│                                │                                            │
│       all_results.append( (world_size, dp_outside, PlanResult) )          │
│                                                                              │
│   best = min(all_results, key=cost.total)  ← Winner toàn cục              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết Phase 3: Training Loop (tương tác với ColossalAI)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    🟦 ORCHESTRATOR TRAINING LOOP                              │
│                                                                              │
│   pp = best_plan.pp                                                          │
│   tp = best_plan.tp                                                          │
│   dp = best_plan.dp                                                          │
│   dp_outside = best_dp_outside                                               │
│                                                                              │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │ 🟩 HybridParallelPlugin(pp_size=pp, tp_size=tp,                        │ │
│   │                        dp_outside=dp_outside, ...)                     │ │
│   │                                                                        │ │
│   │  Bên trong plugin (ColossalAI):                                        │ │
│   │  1. ProcessGroupMesh(shape=(dp, pp, tp))                              │ │
│   │  2. ShardFormer.apply(model, policy=GPT2Policy)                       │ │
│   │     → Thay nn.Linear bằng Col/Row parallel layers                     │ │
│   │  3. PipelineStageManager(pp_axis, enable_interleave=False)            │ │
│   │  4. OneForwardOneBackwardSchedule(num_microbatches=M)                 │ │
│   │  5. Nếu dp>1: DDP.wrap(model) hoặc ZeRO optimizer                     │ │
│   │                                                                        │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │ for step in range(num_steps):                                          │ │
│   │                                                                        │ │
│   │   if pp == 1:  ← 🟦 WORKAROUND (code tự phát triển)                   │ │
│   │      // Không pipeline → forward/backward chuẩn                       │ │
│   │      output = model(**batch)                                          │ │
│   │      loss = output.loss                                               │ │
│   │      loss.backward()                                                  │ │
│   │      optimizer.step()                                                 │ │
│   │   else:                                                               │ │
│   │      // Có pipeline → dùng 1F1B scheduler                             │ │
│   │      🟩 booster.execute_pipeline(                                     │ │
│   │           batch_iter, model, criterion, optimizer)                    │ │
│   │         │                                                             │ │
│   │         └──▶ OneForwardOneBackwardSchedule.forward_backward_step()    │ │
│   │              │                                                        │ │
│   │              ├──▶ Fill: forward M microbatch đầu                      │ │
│   │              ├──▶ Steady: 1F1B (forward+backward xen kẽ)             │ │
│   │              └──▶ Drain: backward microbatch còn lại                  │ │
│   │                                                                        │ │
│   │   optimizer.step()     ← 🟩 AdamW (tự động bởi plugin)               │ │
│   │   optimizer.zero_grad()                                               │ │
│   │                                                                        │ │
│   │   // Đo thời gian và so sánh                                          │ │
│   │   actual_time = time.perf_counter() - start                           │ │
│   │                                                                        │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
│                                │                                             │
│                                ▼                                             │
│   ┌────────────────────────────────────────────────────────────────────────┐ │
│   │ 🟦 JSON Export (validation report)                                      │ │
│   │ {                                                                      │ │
│   │   "plan": {"pp":4, "tp":1, "dp":1},                                    │ │
│   │   "estimated_step_time_ms": 261.7,    ← từ Cost Model                 │ │
│   │   "actual": {"avg_step_time_ms": 225.5}  ← từ Training loop           │ │
│   │ }                                                                      │ │
│   └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Bảng phân loại nguồn gốc tất cả module

| Module | File | Nguồn gốc | Vai trò |
|--------|------|-----------|---------|
| **Orchestrator** | `run_auto_hybrid_parallel.py` | 🟦 Tự phát triển | Điều phối 3 phase: profile → plan → train. Tự động thử nhiều world_size và dp_outside. So sánh kết quả và chọn winner. |
| **Profiler** | `profiler.py` | 🟦 Tự phát triển | Đo α_intra, β_intra, α_cross, β_cross, T_block, T_block_repr trên hardware thực. Dùng GPU events, IQR clip, linear regression. |
| **Topology** | `topology.py` | 🟦 Tự phát triển | Phân loại intra-node vs cross-node cho TP/PP/DP dựa trên cách sắp xếp rank của HybridParallelPlugin. |
| **Cost Model** | `cost_model.py` | 🟦 Tự phát triển | Tính T_total = 7 thành phần (compute, bubble, tp_comm, pp_comm, dp_comm, step_overhead, execution). Không dùng FLOPs giả định, chỉ dùng T_block đo được. |
| **Planner/Search** | `search.py` | 🟦 Tự phát triển | Enumerate → Prune (4 rules) → Score (qua topology + cost model) → Select min. Xử lý 1 (world_size, dp_outside) mỗi lần gọi. |
| **HybridParallelPlugin** | `hybrid_parallel_plugin.py` | 🟩 ColossalAI | Plugin tổng hợp TP+PP+DP. Nhận (pp, tp, dp) và dp_outside từ orchestrator. Không sửa. |
| **ShardFormer** | `shardformer/` | 🟩 ColossalAI | Tự động shard model theo TP (thay nn.Linear bằng Col/Row parallel). Policy GPT2Policy cho GPT-2. |
| **PipelineStageManager** | `pipeline/stage_manager.py` | 🟩 ColossalAI | Quản lý PP stage, tạo process group mesh, điều phối P2P. |
| **1F1B Scheduler** | `pipeline/schedule/one_f_one_b.py` | 🟩 ColossalAI | Lịch trình non-interleaved 1F1B (fill → steady → drain). Mặc định pp_style="1f1b". |
| **DDP** | `torch.nn.parallel.DistributedDataParallel` | 🟩 PyTorch | Đồng bộ gradient khi dp > 1. |
| **ZeRO** | `colossalai.zero.low_level` | 🟩 ColossalAI | Tối ưu bộ nhớ optimizer khi zero_stage > 0. |
| **GPT2Config / GPT2LMHeadModel** | `transformers` | 🟩 HuggingFace | Mô hình ngôn ngữ dùng cho thực nghiệm. |
| **pp=1 workaround** | Trong `run_auto_hybrid_parallel.py` | 🟦 Tự phát triển | Bypass execute_pipeline() khi không có PP, dùng forward/backward chuẩn. |
| **Auto-relaunch** | Trong `run_auto_hybrid_parallel.py` + `launch_nodes.sh` | 🟦 Tự phát triển | Tự động relaunch với ít node hơn nếu winner dùng subset GPU. |

---

## 5. Luồng dữ liệu chi tiết (Data Flow)

```
[Đầu vào]
  ├── ModelConfig (layers, hidden, batch, seq, ...)     🟨 Cấu hình người dùng
  ├── Node layout (node_gpus: [2,2,2])                  🟨 Auto-detect từ env
  └── CLI args (--microbatches, --steps, ...)           🟨 Người dùng truyền

         │
         ▼
[Phase 1: Profiler] 🟦
  ├── _measure_p2p(intra_pair)  →  (α_intra, β_intra)
  ├── _measure_p2p(cross_pair)  →  (α_cross, β_cross)
  ├── _measure_T_block()        →  T_block (isolated)
  ├── _measure_T_block_with_microbatches() → T_block_repr
  └── all_reduce(MAX)           →  ClusterProfile (đồng nhất mọi rank)

         │ ClusterProfile
         ▼
[Phase 2: Planner Loop] 🟦
  ├── Vòng lặp: world_size ∈ {2,4,6}, dp_outside ∈ {True, False}
  │     │
  │     ├──→ [Topology] 🟦 classify_comms() → TopologyInfo
  │     │
  │     ├──→ [Cost Model] 🟦 estimate_step_time() → CostBreakdown (7 terms)
  │     │
  │     └──→ PlanResult (best cho 1 combo)
  │
  └── So sánh tất cả PlanResult → Winner (pp, tp, dp, world_size, dp_outside)

         │ Winner Plan
         ▼
[Phase 3: Training] 🟦 (orchestrator) + 🟩 (ColossalAI)
  ├── Khởi tạo HybridParallelPlugin(pp, tp, dp, dp_outside)
  │     ├──→ ShardFormer shard model
  │     ├──→ PipelineStageManager tạo stage groups
  │     └──→ 1F1B Scheduler khởi tạo
  │
  ├── Training loop
  │     ├──→ Nếu pp=1: forward/backward chuẩn (🟦 workaround)
  │     └──→ Nếu pp>1: execute_pipeline() (🟩 ColossalAI 1F1B)
  │
  └── Đo actual step time + so sánh với estimate

         │ Validation Result
         ▼
[Đầu ra]
  ├── comparison_*.txt      🟦 Báo cáo so sánh chi tiết
  ├── auto_parallel_*.json  🟦 Kết quả benchmark (est vs actual)
  └── RELAUNCH.txt          🟦 (optional) Flag tự động relaunch
```

---

## 6. Điểm đặc biệt của kiến trúc

### 6.1 Tách biệt rõ ràng "Planner" vs "Executor"

Luận văn không sửa core code của ColossalAI. Thay vào đó, xây dựng một **lớp planner độc lập** bên ngoài:
- Planner quyết định **chiến lược** nào tốt nhất (dựa trên cost model)
- HybridParallelPlugin **thực thi** chiến lược đó (không biết planner tồn tại)

Điều này đảm bảo:
- **Tính modular:** Có thể thay planner mà không đụng đến training code
- **Tính portable:** Planner có thể dùng cho framework khác (DeepSpeed, Megatron)
- **Tính bảo trì:** Khi ColossalAI update, plugin vẫn hoạt động bình thường

### 6.2 Profiler + Cost Model không dùng FLOPs giả định

Khác với các cost model khác (ví dụ: `estimate-train-time` của NVIDIA dùng regression trên A100/GH200), cost model của luận văn:
- **Không giả định** GPU đồng nhất
- **Không dùng** peak FLOPs từ datasheet
- **Chỉ dùng** T_block đo được từ GPU chậm nhất trong cụm

→ Phù hợp cho cụm **heterogeneous** (A6000, L40S, L40, A30 trộn lẫn).

### 6.3 Conditional T_block

Cost model dùng **hai giá trị T_block** tùy theo context:
- `pp == 1`: T_block cô lập (clean cache, không áp lực bộ nhớ)
- `pp > 1`: T_block đại diện (có M microbatch activation resident)

→ Khắc phục underestimate ~1.9× khi dùng T_block cô lập cho pipeline.

---

## 7. Đề xuất hình ảnh cho Chương 3 luận văn

### Hình 3.X — Kiến trúc tổng thể hệ thống Auto 3D Parallel

**Loại:** Block diagram (có thể vẽ bằng TikZ trong LaTeX)

**Thành phần:**
- 3 cột dọc: "Code tự phát triển" (xanh dương), "Dữ liệu" (vàng), "ColossalAI Framework" (xanh lá)
- 3 hàng ngang: Phase 1 (Profile), Phase 2 (Plan), Phase 3 (Train)
- Mũi tên chỉ luồng dữ liệu giữa các block

### Hình 3.Y — Chi tiết Cost Model (7 thành phần)

**Loại:** Bar chart hoặc stacked horizontal bar

**Thành phần:**
- 1 bar cho mỗi plan (pp,tp,dp)
- Stack: T_compute | T_bubble | T_tp | T_pp | T_dp | T_step | T_exec
- So sánh giữa estimated và actual

### Hình 3.Z — Luồng 1F1B Pipeline (pp=4, M=8)

**Loại:** Timeline / Gantt chart

**Thành phần:**
- Trục dọc: 4 stage pipeline
- Trục ngang: thời gian
- Màu xanh: forward, Màu đỏ: backward
- Vùng trống (trắng): bubble (fill + drain)
