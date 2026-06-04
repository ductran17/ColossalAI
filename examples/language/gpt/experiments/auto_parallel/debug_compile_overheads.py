"""
Compile all overhead measurements and compare with actual training gaps.
"""
import json, glob, os

# ── Overhead measurements from debug scripts ────────────────────────
OVERHEADS = {
    "adam_ms": 37.77,           # from debug_overhead_2_optimizer.py
    "framework_per_microbatch_ms": 2.55,  # from debug_overhead_1_framework.py
    "tp_sync_per_collective_ms": 0.06,    # from debug_overhead_3_tp_sync.py
    "dp_intra_sync_per_call_ms": 35.66,   # from debug_overhead_4b (pp=1, n=2)
    # Cross-node: we estimate from alpha_cross/beta_cross vs alpha_intra/beta_intra ratio
    "dp_cross_sync_ratio": 0.36 / 0.043,  # beta_cross / beta_intra ≈ 8.4× slower
}

print("=" * 80)
print("OVERHEAD MEASUREMENT SUMMARY")
print("=" * 80)
print(f"Adam optimizer step:           {OVERHEADS['adam_ms']:.2f} ms (cost model: ~1.0 ms)")
print(f"Framework per microbatch:      {OVERHEADS['framework_per_microbatch_ms']:.2f} ms")
print(f"TP sync per collective:        {OVERHEADS['tp_sync_per_collective_ms']:.2f} ms")
print(f"DP intra-node sync overhead:   {OVERHEADS['dp_intra_sync_per_call_ms']:.2f} ms per call")
print(f"DP cross-node slowdown ratio:  {OVERHEADS['dp_cross_sync_ratio']:.1f}× vs intra-node")
print()

# ── Load actual training results ─────────────────────────────────────
results_dir = "/home/ductm27/ColossalAI/examples/language/gpt/experiments/auto_parallel/results_before_estimate"
files = sorted(glob.glob(os.path.join(results_dir, "auto_parallel_4gpu_24L_1024H_16B_*.json")))

print("=" * 80)
print("GAP DECOMPOSITION: Estimated + Overheads vs Actual")
print("=" * 80)
print(f"{'Plan':>18} | {'Model':>7} | {'Actual':>7} | {'Gap':>7} | {'Adam':>6} | {'Framewk':>8} | {'TPsync':>7} | {'DPsync':>7} | {'SumOver':>8} | {'Model+Over':>10} | {'Err':>6}")
print("-" * 120)

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
    
    # 1. Adam: same for all plans
    adam = OVERHEADS["adam_ms"]
    
    # 2. Framework: per microbatch * M * layers_per_stage * pp stages
    # Total blocks processed = M * layers
    total_blocks = M * layers
    framework = OVERHEADS["framework_per_microbatch_ms"] * total_blocks
    
    # 3. TP sync: 2 collectives/layer * layers_per_stage * M microbatches
    if tp > 1:
        n_collectives = 2 * layers_per_stage * M
        tp_sync = OVERHEADS["tp_sync_per_collective_ms"] * n_collectives
    else:
        tp_sync = 0
    
    # 4. DP sync: depends on topology (intra vs cross-node)
    # For 4 GPU on 2 nodes with 2 GPUs each:
    # dp=2: each DP group is cross-node (1 GPU from each node)
    # dp=4: all 4 GPUs are in one DP group, cross-node
    node_gpus = j["plan"]["node_gpus"]  # e.g. [2, 2]
    n_nodes = len(node_gpus)
    
    if dp > 1:
        # Determine if DP is intra-node or cross-node
        gpus_per_node = node_gpus[0]
        dp_groups_cross_node = 0
        dp_groups_intra_node = 0
        
        # Simplified: if dp > gpus_per_node, some groups are cross-node
        if dp <= gpus_per_node:
            # e.g., dp=2 on node with 4 GPUs → all intra-node
            dp_is_intra = True
        else:
            # e.g., dp=4 on 2 nodes with 2 GPUs each → cross-node
            dp_is_intra = False
        
        if dp_is_intra:
            # Base from measurement
            dp_sync = OVERHEADS["dp_intra_sync_per_call_ms"] * (1.0)  # per step, once
        else:
            # Cross-node: much slower
            dp_sync = OVERHEADS["dp_intra_sync_per_call_ms"] * OVERHEADS["dp_cross_sync_ratio"]
    else:
        dp_sync = 0
    
    total_overhead = adam + framework + tp_sync + dp_sync
    model_plus_overhead = est + total_overhead
    error = act - model_plus_overhead
    
    plan_str = f"pp={pp},tp={tp},dp={dp}"
    print(f"{plan_str:>18} | {est:>7.1f} | {act:>7.1f} | {gap:>7.1f} | {adam:>6.1f} | {framework:>8.1f} | {tp_sync:>7.1f} | {dp_sync:>7.1f} | {total_overhead:>8.1f} | {model_plus_overhead:>10.1f} | {error:>6.1f}")

print()
print("KEY FINDINGS:")
print("- Adam step alone explains ~38 ms constant gap for ALL plans")
print("- Framework overhead is HUGE: ~490 ms for all pp values (scales with total blocks)")
print("- This means estimated + overheads ≈ actual is FALSE unless we add framework fudge")
print("- However, framework is MONOTONIC: all plans get +490 ms, preserving rankings")
print("- Only DP cross-node can flip rankings (non-monotonic with topology)")
