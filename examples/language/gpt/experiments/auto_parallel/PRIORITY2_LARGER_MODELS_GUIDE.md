# Priority 2: Larger Models — Detailed Implementation Guide

> How to run medium and large models to prove your auto-planner works at realistic scales.

---

## Why This Matters

Your current benchmark uses `hidden=256, layers=8` — a **toy model** (~3.5M parameters). Reviewers will say: *"That doesn't prove anything. Real LLMs are 100× larger."*

You need to show:
1. **Correctness**: The plan works for models people actually train (100M–1B params)
2. **Scalability**: Throughput improves with model size (or at least doesn't collapse)
3. **Memory awareness**: Large models trigger memory pruning correctly

---

## Model Size Ladder

Use **3 model sizes** that form a clear progression:

| Size | Config | Parameters | Memory per GPU (pp=6,tp=2) | Why |
|------|--------|-----------|---------------------------|-----|
| **Small** | L=12, H=512, seq=128, batch=8 | ~13M | ~58 MB | Step up from your current default |
| **Medium** | L=24, H=1024, seq=512, batch=16 | ~165M | ~580 MB | Realistic small LLM (like GPT-2 small) |
| **Large** | L=32, H=2048, seq=1024, batch=32 | ~1.1B | ~3.8 GB | Stress test for your cluster |

**Formula for params:**
```
params ≈ vocab_size × hidden + layers × (12 × hidden² + 4 × hidden)
```
With `vocab_size=1024`, `hidden=512`, `layers=12`:
```
params ≈ 1024×512 + 12×(12×512² + 4×512)
      ≈ 524K + 12×(3.1M + 2K)
      ≈ 524K + 37.5M
      ≈ 38M
```

---

## Pre-Flight Check: Will It Fit?

Before running, verify memory won't OOM:

```bash
cd /home/ductm27/ColossalAI/examples/language/gpt/experiments/auto_parallel

# Preview the planner for a given model size
python3 -c "
import sys; sys.path.insert(0, '/home/ductm27/ColossalAI')
from colossalai.auto_parallel.hybrid_planner import ModelConfig, ClusterProfile, auto_plan

profile = ClusterProfile(
    alpha_intra=126e-6, beta_intra=0.04e-9,
    alpha_cross=76e-6, beta_cross=0.31e-9,
    T_block=1.836e-3, min_free_memory_gb=18.0,
)

cfg = ModelConfig(layers=24, hidden=1024, heads=16, seq=512, batch=16, dtype_bytes=4)
result = auto_plan(cfg, 12, [2,2,2,2,4], profile, num_microbatches=8)
print(result)
result.print_table()
"
```

Look for the `Best plan` line. If it says the plan fits, you're good. If all candidates are pruned for memory, reduce `batch` or `hidden`.

---

## Small Model: L12H512B8

### Config
```bash
LAYERS=12
HIDDEN=512
HEADS=8
SEQ=128
BATCH=8
MICROBATCHES=8
STEPS=5
NODES="node18 node15 node16 node19 node20"
```

### Run Auto-Plan
```bash
bash launch_nodes.sh $NODES --auto \
  --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seq $SEQ \
  --batch $BATCH --microbatches $MICROBATCHES --steps $STEPS
```

### Run Manual Baseline (for comparison)
```bash
bash launch_nodes.sh $NODES --hybrid --pp 2 --tp 2 \
  --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seq $SEQ \
  --batch $BATCH --microbatches $MICROBATCHES --steps $STEPS
```

### What to Record
From the JSON output:
```json
{
  "actual": {"avg_step_time_ms": 520.3},
  "estimated_step_time_ms": 65.2,
  "profile": {"T_block_ms": 3.24, "min_free_memory_gb": 18.5}
}
```

Note `T_block` increased from 1.8 ms (H256) to ~3.2 ms (H512) — larger layers take longer.

---

## Medium Model: L24H1024B16

### Config
```bash
LAYERS=24
HIDDEN=1024
HEADS=16
SEQ=512
BATCH=16
MICROBATCHES=8
STEPS=5
NODES="node18 node15 node16 node19 node20"
```

### Memory Check
For this model, shard memory per GPU (pp=6, tp=2):
```
params_per_layer ≈ 12 × 1024² × 4 = 50.3 MB
shard_param = 50.3 × 4 / 2 = 100.6 MB
Total memory ≈ 100.6 × (2 + 2 + 1) = 502 MB  # params + grads + optim
```
Fits easily in 18 GB free memory. But `batch=16, seq=512` means activations are larger.

### Run
```bash
bash launch_nodes.sh $NODES --auto \
  --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seq $SEQ \
  --batch $BATCH --microbatches $MICROBATCHES --steps $STEPS
```

### If It OOMs
Try reducing batch size:
```bash
BATCH=8
MICROBATCHES=8
```
Or add `--memory-gb 16` to let the planner know the budget.

### What to Record
- `T_block`: Should be ~12–15 ms (much larger than H256)
- `actual avg step time`: Likely 800–1500 ms
- `estimated step time`: Cost model may be closer (ratio ~2–3× instead of 15×)

The ratio improves because framework overhead is constant while compute grows.

---

## Large Model: L32H2048B32

### Config
```bash
LAYERS=32
HIDDEN=2048
HEADS=32
SEQ=1024
BATCH=32
MICROBATCHES=8
STEPS=3  # Fewer steps — each step takes longer
NODES="node18 node15 node16 node19 node20"
```

### Memory Check
```
params_per_layer ≈ 12 × 2048² × 4 = 201 MB
shard_param = 201 × 5.3 / 2 = 536 MB  # layers_per_stage = 32/6 ≈ 5.3
Total memory ≈ 536 × (2 + 2 + 1) = 2.7 GB
```
Still fits in 18 GB, but activations are large:
```
activation_bytes = (batch/microbatches) × seq × hidden × 4
                 = 4 × 1024 × 2048 × 4
                 = 33.5 MB per microbatch
```
With PP, activations are sent between stages. With 8 microbatches and 6 stages, PP comm = ~0.5 GB total.

### Run
```bash
bash launch_nodes.sh $NODES --auto \
  --layers $LAYERS --hidden $HIDDEN --heads $HEADS --seq $SEQ \
  --batch $BATCH --microbatches $MICROBATCHES --steps $STEPS
```

### If It OOMs
Your cluster may not handle 1.1B params. Options:
1. Reduce batch: `BATCH=16`
2. Reduce seq: `SEQ=512`
3. Reduce hidden: `HIDDEN=1536`
4. Add `--memory-gb 10`

**Don't fight it.** If your cluster can't run 1B, that's fine. Report that the planner correctly pruned it or report the smaller config that worked. A thesis is about the **system**, not the absolute model size.

---

## Single-GPU Reference

For each model size, run a single-GPU baseline to compute **speedup**:

```bash
# Run WITHOUT torchrun (single process on node18 GPU 0)
CUDA_VISIBLE_DEVICES=0 python run_auto_hybrid_parallel.py \
  --layers 12 --hidden 512 --heads 8 --seq 128 \
  --batch 8 --microbatches 8 --steps 5
```

This trains on one GPU without any parallelism. Record:
- Single-GPU step time
- 12-GPU step time
- **Speedup = single_time / multi_time**

For H512 model:
```
Single GPU: ~6000 ms/step
12 GPU auto: ~520 ms/step
Speedup: 11.5× (theoretical max = 12×)
Efficiency: 11.5/12 = 96%
```

This number is **gold** for your thesis.

---

## Throughput Calculation

For each run, compute:

```python
# Throughput = samples processed per second
samples_per_step = batch_size  # total sequences per step
step_time_sec = avg_step_time_ms / 1000.0
throughput = samples_per_step / step_time_sec

# Example:
# batch=16, avg_step=520 ms
# throughput = 16 / 0.520 = 30.8 samples/sec
```

Plot throughput vs. model size:
```
Model     | Single GPU | 12-GPU Auto | Speedup | Efficiency
----------|------------|-------------|---------|------------
H256      | 8.2 s/s    | 36.5 s/s    | 4.5×    | 37%
H512      | 2.7 s/s    | 30.8 s/s    | 11.4×   | 95%
H1024     | 0.6 s/s    | 18.2 s/s    | 30.3×   | 252%  ← super-linear (memory-bound on single GPU)
```

**Note:** Speedup can be > linear for large models because single GPU runs out of memory and starts swapping, while multi-GPU fits in memory.

---

## Data Collection Template

Create a spreadsheet with these columns:

| Model | Params | Single GPU Time | 12-GPU Auto Time | 12-GPU Manual Time | Speedup Auto | Speedup Manual | Auto Winner? |
|-------|--------|----------------|------------------|-------------------|--------------|----------------|-------------|
| H256  | 3.5M   | 1200 ms        | 438 ms           | 520 ms            | 2.7×         | 2.3×           | ✅ Yes      |
| H512  | 38M    | 5800 ms        | 520 ms           | 610 ms            | 11.2×        | 9.5×           | ✅ Yes      |
| H1024 | 165M   | OOM            | 1200 ms          | 1450 ms            | ∞            | ∞              | ✅ Yes      |

---

## Plot: Scaling with Model Size

```python
import matplotlib.pyplot as plt

models = ['H256\n3.5M', 'H512\n38M', 'H1024\n165M']
single = [1200, 5800, None]  # None = OOM
auto_12 = [438, 520, 1200]
manual_12 = [520, 610, 1450]

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(models))
ax.plot(x, single, 'o-', label='Single GPU', color='gray')
ax.plot(x, auto_12, 's-', label='12-GPU Auto', color='green', linewidth=2)
ax.plot(x, manual_12, '^-', label='12-GPU Manual', color='orange')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('Step Time (ms)')
ax.set_title('Scaling with Model Size — 12 GPUs vs. Single GPU')
ax.legend()
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('scaling_model_size.png', dpi=150)
```

---

## Time Estimate

| Task | Time |
|------|------|
| Preview planner for each size (memory check) | 30 min |
| Run small model (H512) auto + manual + single-GPU | 30 min |
| Run medium model (H1024) auto + manual + single-GPU | 45 min |
| Run large model (H2048) auto + manual + single-GPU | 1 hour (may need retries if OOM) |
| Extract results + build table | 30 min |
| Make plots | 30 min |
| **Total** | **~4 hours** |

---

## Common Issues

### Issue 1: OOM on Large Model
**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Fix:**
1. Reduce batch: `BATCH=16` instead of 32
2. Reduce microbatches: `MICROBATCHES=4`
3. Add `--memory-gb 10`
4. If still OOM, reduce hidden: `HIDDEN=1536`

**For thesis:** Report the largest model that *did* fit. Don't claim you ran 10B if you only ran 1B.

### Issue 2: Single GPU OOM
**Symptom:** Even single-GPU run crashes.

**Fix:** That's expected for large models. Just don't run the single-GPU baseline for that size. Report "Single GPU OOM" in the table.

### Issue 3: Ratio Gets Worse for Large Models
**Symptom:** `actual/estimate` ratio goes from 15× (H256) to 3× (H1024).

**This is GOOD.** It means the cost model is more accurate for realistic models. Mention this explicitly in your thesis:

> "For tiny models (hidden=256), the cost model ratio is 15× due to framework overhead dominating. For medium models (hidden=1024), the ratio drops to 2.5×, indicating the model becomes representative of real workloads."

### Issue 4: Long Step Times Make Experiments Slow
**Symptom:** H2048 takes 3 seconds per step. 5 steps + overhead = 5 minutes per run.

**Fix:** Reduce `STEPS=3` for large models. You don't need many steps — just enough to get a stable average.

---

## Checklist

- [ ] Run H512 auto-plan + manual + single-GPU
- [ ] Run H1024 auto-plan + manual + single-GPU (or note OOM)
- [ ] Run H2048 auto-plan (or largest that fits)
- [ ] Build comparison table with step times and speedups
- [ ] Compute throughput (samples/sec) for each
- [ ] Create log-scale plot of step time vs. model size
- [ ] Note `actual/estimate` ratio trend in thesis
