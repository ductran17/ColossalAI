"""
Compile all overhead measurements into final report.
"""
import json, glob, os

# ── Measured Overheads ─────────────────────────────────────────────
OVERHEADS = {
    "adam_ms": 37.77,
    "framework_per_loop_ms": 1.98,
    "tp_sync_per_collective_ms": 0.04,
    "dp_intra_raw_model_ratio": 1.71,  # actual / model
    "dp_cross_raw_model_ratio": 0.75,  # actual / model (for n=4)
    "pp_dispatch_per_microbatch_ms": -1.21,  # NEGATIVE (pipeline hides overhead)
}

print("=" * 90)
print("COMPLETE OVERHEAD MEASUREMENT RESULTS")
print("=" * 90)
print()
print("1. ADAM OPTIMIZER STEP")
print(f"   Measured:        {OVERHEADS['adam_ms']:.2f} ms")
print(f"   Cost model:      ~1.0 ms (FLOP-scaled from T_block)")
print(f"   Constant gap:    +{OVERHEADS['adam_ms'] - 1.0:.2f} ms for ALL plans")
print()
print("2. FRAMEWORK PER-MICROBATCH DISPATCH")
print(f"   Measured:        {OVERHEADS['framework_per_loop_ms']:.2f} ms per iteration")
print(f"   For M=8, pp=2:   {OVERHEADS['framework_per_loop_ms'] * 8 * 2:.1f} ms total")
print(f"   Cost model:      0 ms (not modeled)")
print()
print("3. TP ALLREDUCE SYNC (intra-node, 2 GPUs)")
print(f"   Per collective:  {OVERHEADS['tp_sync_per_collective_ms']:.2f} ms")
print(f"   For tp=2,pp=2:   {OVERHEADS['tp_sync_per_collective_ms'] * 2 * 12 * 8:.1f} ms total")
print(f"   Cost model:      0 ms (assumed fully overlapped)")
print()
print("4. DP ALLREDUCE SYNC (intra-node, 2 GPUs)")
print(f"   Actual / model:  {OVERHEADS['dp_intra_raw_model_ratio']:.2f}×")
print(f"   Meaning:         Model UNDERESTIMATES by 71%")
print(f"   Exposed fraction: {OVERHEADS['dp_intra_raw_model_ratio']:.2f} (not 0.3)")
print()
print("5. DP ALLREDUCE SYNC (cross-node, 4 GPUs, node18+node19)")
print(f"   Actual / model:  {OVERHEADS['dp_cross_raw_model_ratio']:.2f}×")
print(f"   Meaning:         Model OVERESTIMATES by 25%")
print(f"   Exposed fraction: {OVERHEADS['dp_cross_raw_model_ratio']:.2f}")
print(f"   NOTE: Cross-node is SLOWER in absolute terms but DDP overlap works better than expected")
print()
print("6. PP STAGE MANAGER DISPATCH (pp=2, 2 GPUs)")
print(f"   Per microbatch:  {OVERHEADS['pp_dispatch_per_microbatch_ms']:.2f} ms")
print(f"   For M=8:         {OVERHEADS['pp_dispatch_per_microbatch_ms'] * 8:.1f} ms total")
print(f"   NOTE: NEGATIVE! Pipeline overlap HIDES framework cost")
print()
print("=" * 90)
print("GAP DECOMPOSITION FOR EACH PLAN")
print("=" * 90)

# Load actual results
results_dir = "/home/ductm27/ColossalAI/examples/language/gpt/experiments/auto_parallel/results_before_estimate"
files = sorted(glob.glob(os.path.join(results_dir, "auto_parallel_4gpu_24L_1024H_16B_*.json")))

print(f"\n{'Plan':>18} | {'Model':>7} | {'Actual':>7} | {'Gap':>7} | {'Adam':>6} | {'Framewk':>8} | {'TPsync':>7} | {'DPsync':>7} | {'SumOver':>8} | {'Model+Over':>10} | {'Error':>7} | {'RankOK'}")
print("-" * 140)

all_data = []
for f in files:
    with open(f) as fp:
        j = json.load(fp)
    p = j["plan"]
    est = j["estimated_step_time_ms"]
    act = j["actual"]["avg_step_time_ms"]
    gap = act - est
    
    pp, tp, dp = p["pp"], p["tp"], p["dp"]
    M = j["model"]["microbatches"]
    layers = j["model"]["layers"]
    layers_per_stage = layers // pp
    
    # 1. Adam
    adam = OVERHEADS["adam_ms"]
    
    # 2. Framework: per-loop overhead * total iterations
    # For pp=1: M * layers iterations
    # For pp>1: M * layers_per_stage * pp = M * layers (same total)
    total_blocks = M * layers
    framework = OVERHEADS["framework_per_loop_ms"] * total_blocks
    
    # 3. TP sync: 2 collectives/layer * layers_per_stage * M
    if tp > 1:
        n_collectives = 2 * layers_per_stage * M
        tp_sync = OVERHEADS["tp_sync_per_collective_ms"] * n_collectives
    else:
        tp_sync = 0
    
    # 4. DP sync: modeled * ratio - modeled = modeled * (ratio - 1)
    # But we need to know what the cost model predicted for DP comm
    # For simplicity, use the ratio: actual_dp = model_dp * ratio
    # So overhead = model_dp * (ratio - 1)
    # If ratio < 1, then actual < model, so "overhead" is negative
    if dp > 1:
        # Use cross-node ratio since 4GPU on 2 nodes means dp groups are cross-node
        dp_sync_ratio = OVERHEADS["dp_cross_raw_model_ratio"]
        # Need model_dp prediction - extract from JSON breakdown
        bd = j.get("estimated_breakdown_ms", {})
        model_dp = bd.get("dp_comm", 0)
        dp_sync = model_dp * (dp_sync_ratio - 1)  # can be negative!
    else:
        dp_sync = 0
        model_dp = 0
    
    # 5. PP dispatch
    if pp > 1:
        pp_dispatch = OVERHEADS["pp_dispatch_per_microbatch_ms"] * M
    else:
        pp_dispatch = 0
    
    total_overhead = adam + framework + tp_sync + dp_sync + pp_dispatch
    model_plus_overhead = est + total_overhead
    error = act - model_plus_overhead
    
    # For ranking check
    all_data.append({"pp": pp, "tp": tp, "dp": dp, "est": est, "act": act, 
                     "model_plus_overhead": model_plus_overhead, "error": error})
    
    plan_str = f"pp={pp},tp={tp},dp={dp}"
    rank_ok = "✓" if abs(error) < 100 else "✗"
    print(f"{plan_str:>18} | {est:>7.1f} | {act:>7.1f} | {gap:>7.1f} | {adam:>6.1f} | {framework:>8.1f} | {tp_sync:>7.1f} | {dp_sync:>7.1f} | {total_overhead:>8.1f} | {model_plus_overhead:>10.1f} | {error:>7.1f} | {rank_ok}")

# Check ranking preservation
print()
print("=" * 90)
print("RANKING PRESERVATION CHECK (with overhead correction)")
print("=" * 90)

# Sort by actual
all_data.sort(key=lambda x: x["act"])
for i, d in enumerate(all_data):
    est_sorted = sorted(all_data, key=lambda x: x["model_plus_overhead"])
    rank_est = next(k for k, x in enumerate(est_sorted) 
                   if x["pp"]==d["pp"] and x["tp"]==d["tp"] and x["dp"]==d["dp"])
    marker = "✓ CORRECT" if rank_est == i else f"✗ WRONG (est rank {rank_est+1})"
    plan_str = f"pp={d['pp']},tp={d['tp']},dp={d['dp']}"
    print(f"  Actual rank {i+1}: {plan_str:>15} | model+overhead: {d['model_plus_overhead']:>8.1f} ms | {marker}")

print()
print("KEY FINDINGS:")
print("- Adam step: +38 ms constant for all plans (preserves rankings)")
print("- Framework: +381 ms for all plans with M=8, layers=24 (preserves rankings)")
print("- TP sync: negligible (+7 ms for tp=2 plans)")
print("- DP cross-node: NEGATIVE overhead (-25% of model prediction)! Model overestimates.")
print("- PP dispatch: NEGATIVE (-10 ms) because pipeline overlap hides framework cost")
print("- TOTAL: model+overhead is still 100-300ms LOWER than actual")
print("- Remaining unexplained: memory allocation, gradient accumulation buffers,")
print("  ShardFormer tensor-split overhead, Python GIL contention")
print()
print("RANKINGS WITH OVERHEAD CORRECTION:")
correct = sum(1 for i, d in enumerate(all_data) 
              if next(k for k, x in enumerate(sorted(all_data, key=lambda x: x["model_plus_overhead"]))
                     if x["pp"]==d["pp"] and x["tp"]==d["tp"] and x["dp"]==d["dp"]) == i)
print(f"  {correct}/{len(all_data)} plans correctly ranked ({100*correct/len(all_data):.0f}%)")
