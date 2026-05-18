# Strategic Recommendation: Code vs. Experiments for Your Thesis

> Should you modify the core auto-planner code, or focus on experiments and writing?

---

## My Strong Recommendation: Focus on Experiments & Analysis

Don't modify the core code. Your current implementation is **good enough** for a thesis. What makes a thesis is **evidence and analysis**, not more features.

### The 80/20 Rule for Your Timeline

| Activity | Time | Value for Thesis |
|----------|------|-----------------|
| **Running experiments** (baselines, ablations, scalability) | 60% | **High** — this is what reviewers judge |
| **Writing** (formal analysis, related work, results) | 25% | **High** — tells the story |
| **Small code helpers** (logging, memory detection, scripts) | 10% | **Medium** — makes experiments easier |
| **Core code changes** (heterogeneous splits, new parallelism) | 5% | **Low/Risky** — high chance of breaking |

---

## Why You Should NOT Modify Core Code

### 1. It's High-Risk, Low-Reward

| Change | Effort | Risk | Thesis Value |
|--------|--------|------|-------------|
| Unequal layer splitting for heterogeneous GPUs | 4–6 weeks | **Very High** — requires modifying `PipelineStageManager` and `ShardFormer` | Medium — reviewers will ask "does it converge correctly?" |
| Expert parallelism in cost model | 3–4 weeks | **High** — MoE routing changes everything | Low — out of scope for your claim |
| Sequence parallelism | 2–3 weeks | **High** — plugin supports it, cost model doesn't | Low — not your target scenario |
| Heterogeneous tensor sharding | 4–6 weeks | **Very High** — rewrite ShardFormer policies | Medium — cool but risky |

**Reality check:** These are research-level engineering tasks. A master's thesis timeline (2–4 months of active work) is not enough to implement + validate them reliably. One bug in `PipelineStageManager` and your entire experimental pipeline is broken for weeks.

### 2. Your Current Code Already Has a Defensible Claim

Your system:
- Profiles real hardware in **0.3 seconds** (vs. hours for Alpa)
- Works on **heterogeneous clusters** (L40S + L40 + A30) safely
- Runs on **commodity Ethernet** (3 GB/s) without requiring InfiniBand
- Requires **zero manual tuning** (auto-detects node layout)

**This is already novel.** A thesis doesn't need to solve every problem — it needs to solve **one** problem well and prove it with evidence.

### 3. Reviewers Judge Experiments, Not Code Completeness

What a thesis committee asks:
- "Does your system support FP8?" — Nobody cares.
- "How do you know your plan is better than manual tuning?" — **Critical.**
- "What happens if you remove the profiler?" — **Critical.**
- "Does it scale to more GPUs?" — **Important.**
- "Does the model actually converge?" — **Critical.**

**All of these are answered by experiments, not by more code.**

---

## What You SHOULD Do (The 6 Priorities)

### Priority 1: Baselines (Highest Value)

**What:** Run your current code with different "plan selection strategies" and compare.

| Baseline | How to Run | What You Get |
|----------|-----------|-------------|
| **Manual "balanced" plan** | `bash launch_nodes.sh ... --hybrid --pp 2 --tp 2` | Show humans pick suboptimal plans |
| **Random valid plan** | Pick random (pp, tp, dp) from candidate list | Show random search is worse |
| **No profiler** | Hard-code synthetic profile (alpha=5us, beta=0.01ns/B, T=1ms) | Show real profiling matters |
| **Maximum pipeline** | `pp=12, tp=1, dp=1` (or `pp=14` if node11 included) | Show extreme plans fail or are slow |

**Effort:** 1–2 weeks. You're just running the same code with different flags.

**Deliverable:** A table showing throughput (samples/sec) for each baseline vs. your auto-plan.

### Priority 2: Larger Models (Highest Value)

**What:** Run medium-sized models that are actually meaningful.

| Model | Config | Why |
|-------|--------|-----|
| **Small** | hidden=512, layers=12, seq=128, batch=16 | Step up from your current hidden=256 |
| **Medium** | hidden=1024, layers=24, seq=512, batch=16 | Realistic small LLM |
| **Large** | hidden=2048, layers=32, seq=1024, batch=32 | If your cluster can handle it |

**Effort:** 1 week. Change command-line args. May need `--memory-gb` for larger models.

**Deliverable:** Throughput scaling with model size. Convergence curves (loss vs. step).

### Priority 3: Ablation Studies (High Value)

**What:** Disable parts of your system to show they matter.

| Ablation | How | Expected Result |
|----------|-----|-----------------|
| **No profiler** | Use hard-coded profile in `run_auto_hybrid_parallel.py` | Plans mis-rank on real hardware |
| **No pruner** | Comment out pruning rules in `search.py` | May pick cross-node TP -> crash or very slow |
| **No topology classifier** | Assume all comm is cross-node | Over-penalizes intra-node, picks suboptimal plans |
| **Only TP** | Force `pp=1, dp=1` | Shows PP+DP matter for multi-node |
| **Only PP** | Force `tp=1, dp=1` | Shows TP matters for compute efficiency |

**Effort:** 3–4 days. Small code changes (commenting out lines, hard-coding values).

**Deliverable:** Table showing ranking accuracy or throughput with each component disabled.

### Priority 4: Scalability Study (Medium-High Value)

**What:** Run on subsets of your cluster.

| GPUs | Topology | Command |
|------|----------|---------|
| 4 | `node18 node20` (2+2) | `bash launch_nodes.sh node18 node20 --auto` |
| 6 | `node18 node15 node16` (2+2+2) | `bash launch_nodes.sh node18 node15 node16 --auto` |
| 8 | `node18 node15 node16 node19` (2+2+2+2) | `bash launch_nodes.sh node18 node15 node16 node19 --auto` |
| 12 | Full cluster minus node11 | Your current benchmark |

**Effort:** 3–4 days. Just run the launch script with different node lists.

**Deliverable:** Speedup plot (speedup vs. number of GPUs). Is it linear? Where does it plateau?

### Priority 5: Convergence Validation (Medium Value)

**What:** Show that training actually works and converges correctly.

```bash
# Run for 100+ steps with a real model
bash launch_nodes.sh node18 node15 node16 node19 node20 --auto \
  --layers 12 --hidden 512 --batch 16 --microbatches 8 --steps 100
```

Compare with single-GPU training on the same model:
```bash
python run_auto_hybrid_parallel.py --layers 12 --hidden 512 --batch 16 --microbatches 8 --steps 100
# (without torchrun, just single process)
```

**Effort:** 2–3 days. Need to log loss per step.

**Deliverable:** Loss curves showing single-GPU vs. 12-GPU convergence match within 1–2%.

### Priority 6: Formal Analysis + Writing (Critical)

**What:** Write the thesis chapters.

| Chapter | Time | What to Include |
|---------|------|----------------|
| **Ch 1: Introduction** | 3 days | Problem, motivation, your claim |
| **Ch 2: Related Work** | 5 days | Alpa, Megatron, DeepSpeed, ColossalAI auto-parallel. Compare their assumptions vs. yours. |
| **Ch 3: Design** | 5 days | Formal problem formulation, profiler design, cost model equations, pruning rules |
| **Ch 4: Implementation** | 3 days | Code structure, integration with ColossalAI |
| **Ch 5: Experiments** | 7 days | Baselines, ablations, scalability, convergence, large models |
| **Ch 6: Limitations** | 2 days | Heterogeneity not optimized, static plan, tiny model overhead |

**Effort:** 3–4 weeks of writing.

**This is what makes it a thesis.** Without writing, it's just a project.

---

## Small Code Enhancements Worth Doing (10% of Time)

These are low-risk, high-utility:

### 1. Add Runtime Free Memory Detection (1–2 days)

**Why:** Makes experiments on shared clusters safer. Prevents OOM failures during thesis demo.

**What to add:**
```python
# In profiler.py, inside profile_cluster()
free_bytes, _ = torch.cuda.mem_get_info()
free_gb = free_bytes / (1024**3)
# all_gather + MIN -> min_free_memory_gb
# Add to ClusterProfile
```

**Risk:** Very low. No core algorithm changes.

### 2. Add JSON Result Output (1 day)

**Why:** Makes collecting experimental results automated instead of copy-pasting from logs.

**What to add:**
```python
# In run_auto_hybrid_parallel.py, after training:
import json
result = {
    "plan": {"pp": pp, "tp": tp, "dp": dp},
    "profile": {"alpha_intra": ..., "T_block": ...},
    "step_times": step_times_ms,
    "avg_step_time": avg_actual_ms,
    "estimated_step_time": estimated.total * 1000,
}
with open(f"results_{world_size}gpu.json", "w") as f:
    json.dump(result, f, indent=2)
```

### 3. Add Benchmark Runner Script (2–3 days)

**Why:** Automates running all baselines and collecting results.

**What to write:** A bash script that loops over:
- Different node combinations (4, 6, 8, 12 GPUs)
- Different baselines (auto, manual pp=2 tp=2, manual pp=12 tp=1)
- Different model sizes

And saves all results to a folder for analysis.

### 4. Fix Minor Issues (1 day)

- Change `MASTER_PORT` back to 29500 (I changed it to 29501 earlier)
- Add `--microbatches >= pp` check in `search.py` to prevent runtime assertion failures
- Clean up any print statements or debug logs

---

## What NOT to Do (Avoid These Traps)

| Trap | Why | What Happens |
|------|-----|--------------|
| **Rewrite PipelineStageManager for heterogeneous layers** | 4–6 weeks, high bug risk | You break training and have no experiments for 2 months |
| **Add FP16/BF16 support** | Plugin supports it, but cost model needs updating | Complexity for marginal thesis value |
| **Add sequence parallelism** | Not in cost model, requires new profiling | Scope creep — save for future work |
| **Train a model to full convergence** | Takes days/weeks | 100 steps is enough to prove correctness |
| **Support >14 GPUs (synthetic)** | You don't have the hardware | Reviewers will ask "did you actually run it?" Say no and move on |

---

## Concrete 10-Week Plan

| Week | Focus | Deliverable |
|------|-------|-------------|
| **1** | Small code helpers (free mem, JSON output, benchmark script) | Automated experiment runner |
| **2** | Run baselines on 12 GPUs (manual, random, no profiler) | Baseline results table |
| **3** | Run larger models (hidden=512, 1024) | Throughput + convergence data |
| **4** | Ablation studies (no profiler, no pruner, no topology) | Ablation results table |
| **5** | Scalability (4, 6, 8, 12 GPUs) | Speedup plot |
| **6** | Convergence validation (single GPU vs. 12 GPU) | Loss curves |
| **7** | Write Ch 1–3 (Intro, Related Work, Design) | Draft chapters |
| **8** | Write Ch 4–5 (Implementation, Experiments) | Draft chapters |
| **9** | Write Ch 6 (Limitations), polish, make figures | Complete draft |
| **10** | Advisor review, revisions, final formatting | Submission-ready thesis |

---

## Bottom Line

> **Don't modify the core code. Run experiments with the code you have.**

Your auto-planner is already novel enough for a thesis. What will make it a thesis is:
1. **Evidence** that it beats baselines
2. **Analysis** of why it works (ablations)
3. **Generality** (scales to different cluster sizes)
4. **Correctness** (convergence matches single-GPU)
5. **Writing** that frames it as a research contribution

**Spend 90% of your time running experiments and writing. Spend 10% on small code helpers.**

This is the fastest path to a defensible thesis.
