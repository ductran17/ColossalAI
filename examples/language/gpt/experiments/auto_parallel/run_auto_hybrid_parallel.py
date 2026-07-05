"""
Auto 3D Parallel training — full pipeline: profile → plan → train.

Three phases run automatically on the cluster:
  Phase 1  profile_cluster()  ~2 s   measure α/β/T_block on real hardware
  Phase 2  auto_plan()        <1 ms  enumerate + score (pp,tp,dp) candidates
  Phase 3  HybridParallelPlugin       train with the chosen plan

No pp/tp flags needed — the planner picks them.

Usage:
  bash launch_nodes.sh --auto                       # fully automatic
  bash launch_nodes.sh --auto --layers 8 --batch 4  # specify model size
  bash launch_nodes.sh --auto --profile-repeat 100    # higher accuracy profiling

How to test without a cluster:
  Use run_hybrid_parallel.py --pp 4 --tp 2 to manually run the plan that
  auto_plan() would choose on the [2,4,2] cluster.
"""

import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.distributed as dist
import transformers

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.logging import disable_existing_loggers, get_dist_logger

from colossalai.auto_parallel.hybrid_planner import (
    profile_cluster,
    auto_plan,
    ModelConfig,
)
from colossalai.auto_parallel.hybrid_planner.profiler import _gather_node_layout


def parse_args():
    p = argparse.ArgumentParser(description="Auto 3D parallel GPT2 training")
    # Model
    p.add_argument("--model",        type=str, default="gpt2", choices=["gpt2", "llama", "llama32", "smollm", "qwen25"],
                   help="Model family: gpt2, llama, llama32, smollm, or qwen25 (default: gpt2)")
    p.add_argument("--gpt2-size",    type=str, default=None, choices=["small", "medium", "large"],
                   help="For --model=gpt2: use standard OpenAI GPT-2 size "
                        "(small=124M, medium=345M, large=774M). "
                        "If set, --layers/--hidden/--heads are ignored. "
                        "If omitted and no dims given, defaults to small.")
    p.add_argument("--model-name",   type=str, default="meta-llama/Llama-2-7b-hf",
                   help="HuggingFace model name or local path for --model=llama/llama32. "
                        "Ignored for gpt2 unless using real weights. "
                        "(default: meta-llama/Llama-2-7b-hf)")
    p.add_argument("--layers",       type=int, default=None, help="Transformer layers (default: from model config or 8 for gpt2)")
    p.add_argument("--hidden",       type=int, default=None, help="Hidden dimension (default: from model config or 256 for gpt2)")
    p.add_argument("--heads",        type=int, default=None, help="Attention heads (default: from model config or 4 for gpt2)")
    p.add_argument("--seq",          type=int, default=None, help="Sequence length (default: from model config or 64 for gpt2)")
    # Training
    p.add_argument("--batch",        type=int, default=4,   help="Global batch size (total sequences per step)")
    p.add_argument("--microbatches", type=int, default=4,   help="Microbatches for pipeline schedule")
    p.add_argument("--steps",        type=int, default=3,   help="Training steps")
    # Planner
    p.add_argument("--memory-gb",    type=float, default=None,
                   help="Per-GPU memory budget in GB for pruning (e.g. 24.0). "
                        "If omitted, memory pruning is skipped.")
    p.add_argument("--dp-outside",    dest="dp_outside", action="store_true", default=True,
                   help="dp_outside flag for HybridParallelPlugin (default True)")
    p.add_argument("--no-dp-outside", dest="dp_outside", action="store_false",
                   help="Use dp_outside=False for HybridParallelPlugin (mesh shape = (pp, dp, tp))")
    p.add_argument("--fixed-world-size", action="store_true",
                   help="Disable subset search: only evaluate the exact world_size provided, "
                        "do not consider using fewer GPUs.")
    # Manual plan override (for cost model validation / Priority 0)
    p.add_argument("--manual-pp",    type=int, default=None,
                   help="Force pipeline-parallel degree (bypass auto_plan). "
                        "If set, --manual-tp must also be set.")
    p.add_argument("--manual-tp",    type=int, default=None,
                   help="Force tensor-parallel degree (bypass auto_plan). "
                        "If set, --manual-pp must also be set.")
    # Profiler
    p.add_argument("--warmup",         type=int, default=3,   help="Profiler warmup iters")
    p.add_argument("--repeat",           type=int, default=10,  help="Profiler timed iters")
    p.add_argument("--profile-warmup",   type=int, default=10,
                   help="Warmup iterations for profiler P2P and T_block (default 10).")
    p.add_argument("--profile-repeat",   type=int, default=50,
                   help="Timed iterations for profiler P2P and T_block (default 50). "
                        "Higher = more stable but slower profiling.")
    return p.parse_args()


def _resolve_model_dims(args):
    """Ensure args.layers/hidden/heads/seq have concrete values before profiling.

    For pre-trained models (llama, llama32, smollm, qwen25) we may need to
    load the HF config early so that profile_cluster receives valid dims.
    """
    if args.model in ("llama", "llama32", "smollm", "qwen25"):
        model_name = args.model_name
        if model_name == "meta-llama/Llama-2-7b-hf":
            defaults = {
                "llama":   "meta-llama/Llama-2-7b-hf",
                "llama32": "meta-llama/Llama-3.2-1B",
                "smollm":  "HuggingFaceTB/SmolLM-360M",
                "qwen25":  "Qwen/Qwen2.5-0.5B",
            }
            model_name = defaults[args.model]
        cfg = transformers.AutoConfig.from_pretrained(model_name)
        if args.layers is None:
            args.layers = cfg.num_hidden_layers
        if args.hidden is None:
            args.hidden = cfg.hidden_size
        if args.heads is None:
            args.heads = cfg.num_attention_heads
        if args.seq is None:
            args.seq = getattr(cfg, "max_position_embeddings", 2048)
    elif args.model == "gpt2":
        gpt2_presets = {
            "small":  (12, 768, 12),
            "medium": (24, 1024, 16),
            "large":  (36, 1280, 20),
        }
        if args.gpt2_size in gpt2_presets:
            # Standard OpenAI GPT-2 size (ignore any manually passed dims)
            args.layers, args.hidden, args.heads = gpt2_presets[args.gpt2_size]
            if args.seq is None:
                args.seq = 1024
        elif args.layers is None and args.hidden is None and args.heads is None:
            # Default to GPT-2 Small when no dims and no --gpt2-size
            if args.seq is None:
                args.seq = 1024
            args.layers, args.hidden, args.heads = 12, 768, 12
        else:
            # Custom synthetic GPT-2
            if args.layers is None:
                args.layers = 8
            if args.hidden is None:
                args.hidden = 256
            if args.heads is None:
                args.heads = 4
            if args.seq is None:
                args.seq = 64


def main():
    args = parse_args()
    # Remember whether user explicitly passed dims (for GPT-2 synthetic vs real).
    args._gpt2_custom_dims = (
        args.model == "gpt2" and
        args.gpt2_size is None and
        not (args.layers is None and args.hidden is None and args.heads is None)
    )
    _resolve_model_dims(args)
    disable_existing_loggers()
    colossalai.launch_from_torch()
    logger = get_dist_logger()

    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    # ------------------------------------------------------------------
    # Step 0: auto-detect node layout from torchrun env vars.
    #
    # torchrun sets LOCAL_RANK and LOCAL_WORLD_SIZE on every rank.
    # _gather_node_layout() does an all_gather of these two values and
    # reconstructs which global ranks share the same physical node.
    # ------------------------------------------------------------------
    nodes     = _gather_node_layout(rank, world_size)
    node_gpus = [len(node) for node in nodes]

    if rank == 0:
        logger.info(f"[auto] Node layout detected: {node_gpus} (nodes × GPUs)", ranks=[0])

    # ------------------------------------------------------------------
    # Phase 1: Profile the cluster.
    #
    # All ranks call profile_cluster() together. It measures:
    #   α_intra, β_intra  — intra-node P2P latency + inverse bandwidth
    #   α_cross, β_cross  — cross-node P2P latency + inverse bandwidth
    #   T_block           — GPU time for one transformer block fwd+bwd
    #
    # profile_cluster() all-reduces with MAX so every rank ends up with
    # identical ClusterProfile values — the plan is deterministic everywhere.
    # ------------------------------------------------------------------
    model_cfg_dict = {
        "batch":  args.batch // args.microbatches,  # per-microbatch size for T_block
        "seq":    args.seq,
        "hidden": args.hidden,
        "heads":  args.heads,
    }

    if rank == 0:
        logger.info("[auto] Phase 1: profiling cluster α/β/T_block ...", ranks=[0])

    t0 = time.perf_counter()
    profile = profile_cluster(
        model_cfg=model_cfg_dict,
        warmup=args.profile_warmup,
        repeat=args.profile_repeat,
        num_microbatches=args.microbatches,
    )
    t_profile = time.perf_counter() - t0

    if rank == 0:
        logger.info(
            f"[auto] Profile done in {t_profile:.1f}s\n"
            f"       α_intra={profile.alpha_intra*1e6:.1f} µs  "
            f"β_intra={profile.beta_intra*1e9:.2f} ns/B  "
            f"(BW={1/profile.beta_intra/1e9:.1f} GB/s)\n"
            f"       α_cross={profile.alpha_cross*1e6:.1f} µs  "
            f"β_cross={profile.beta_cross*1e9:.2f} ns/B  "
            f"(BW={1/profile.beta_cross/1e9:.1f} GB/s)\n"
            f"       T_block={profile.T_block*1e3:.3f} ms\n"
            f"       T_emb+head={profile.T_embedding_lm_head*1e3:.3f} ms  "
            f"(profiled embedding+LM head)\n"
            f"       min_free_mem={profile.min_free_memory_gb:.1f} GB  "
            f"(across all GPUs at profiling time)",
            ranks=[0],
        )

    # ------------------------------------------------------------------
    # MODEL SELECTION
    # ------------------------------------------------------------------
    # Uncomment ONE block below to switch the model family.
    # The cost model auto-detects architecture coefficients
    # (intermediate_size, num_key_value_heads, mlp_gated) from the
    # resulting config object, so no other changes are required.
    #
    # NOTE: args.layers / args.hidden / args.heads should match the
    #       loaded model config (or override them after loading).
    # ------------------------------------------------------------------

    if args.model == "gpt2":
        # Three modes:
        #   1) --gpt2-size small/medium/large  → standard OpenAI config (vocab=50257)
        #   2) No dims passed                 → default to GPT-2 Small (vocab=50257)
        #   3) Custom dims passed             → synthetic model (vocab=1024)
        if args._gpt2_custom_dims:
            model_config = transformers.GPT2Config(
                n_positions = args.seq or 64,
                n_layer     = args.layers or 8,
                n_head      = args.heads or 4,
                n_embd      = args.hidden or 256,
                n_inner     = (args.hidden or 256) * 4,
                vocab_size  = 1024,
                resid_pdrop = 0.0,
                attn_pdrop  = 0.0,
            )
        else:
            model_config = transformers.GPT2Config(
                n_positions = args.seq or 1024,
                n_layer     = args.layers,
                n_embd      = args.hidden,
                n_head      = args.heads,
                n_inner     = args.hidden * 4,
                resid_pdrop = 0.0,
                attn_pdrop  = 0.0,
            )
    elif args.model == "llama":
        model_config = transformers.AutoConfig.from_pretrained(args.model_name)
        if args.layers is None:
            args.layers = model_config.num_hidden_layers
        if args.hidden is None:
            args.hidden = model_config.hidden_size
        if args.heads is None:
            args.heads  = model_config.num_attention_heads
        if args.seq is None:
            args.seq    = getattr(model_config, "max_position_embeddings", 2048)
    elif args.model == "llama32":
        model_name = args.model_name if args.model_name != "meta-llama/Llama-2-7b-hf" else "meta-llama/Llama-3.2-1B"
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        if args.layers is None:
            args.layers = model_config.num_hidden_layers
        if args.hidden is None:
            args.hidden = model_config.hidden_size
        if args.heads is None:
            args.heads  = model_config.num_attention_heads
        if args.seq is None:
            args.seq    = getattr(model_config, "max_position_embeddings", 2048)
    elif args.model == "smollm":
        model_name = args.model_name if args.model_name != "meta-llama/Llama-2-7b-hf" else "HuggingFaceTB/SmolLM-360M"
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        if args.layers is None:
            args.layers = model_config.num_hidden_layers
        if args.hidden is None:
            args.hidden = model_config.hidden_size
        if args.heads is None:
            args.heads  = model_config.num_attention_heads
        if args.seq is None:
            args.seq    = getattr(model_config, "max_position_embeddings", 2048)
    elif args.model == "qwen25":
        model_name = args.model_name if args.model_name != "meta-llama/Llama-2-7b-hf" else "Qwen/Qwen2.5-0.5B"
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        if args.layers is None:
            args.layers = model_config.num_hidden_layers
        if args.hidden is None:
            args.hidden = model_config.hidden_size
        if args.heads is None:
            args.heads  = model_config.num_attention_heads
        if args.seq is None:
            args.seq    = getattr(model_config, "max_position_embeddings", 2048)
    else:
        raise ValueError(f"Unknown --model: {args.model}")

    # ------------------------------------------------------------------
    # Extract generic architecture coefficients from the config.
    # The code below works for ANY transformers.PretrainedConfig.
    # ------------------------------------------------------------------
    intermediate_size = getattr(model_config, "n_inner", None) or getattr(model_config, "intermediate_size", None)
    num_key_value_heads = getattr(model_config, "num_key_value_heads", None) or getattr(model_config, "n_head", None)
    mlp_gated = "swiglu" in getattr(model_config, "hidden_act", "").lower()

    # Enrich model_cfg_dict with architecture coefficients so the profiler
    # builds a representative block (GQA + SwiGLU vs standard MHA + FFN).
    vocab_size = getattr(model_config, "vocab_size", 1024)
    model_cfg_dict["vocab_size"] = vocab_size
    model_cfg_dict["intermediate_size"] = intermediate_size
    model_cfg_dict["num_key_value_heads"] = num_key_value_heads
    model_cfg_dict["mlp_gated"] = mlp_gated

    # ------------------------------------------------------------------
    # Phase 2: Run the auto-planner.
    #
    # auto_plan() is pure Python (no distributed ops). Because
    # ClusterProfile is already all-reduced, every rank computes the
    # exact same plan independently — no broadcast needed.
    # ------------------------------------------------------------------
    cfg = ModelConfig(
        layers              = args.layers,
        hidden              = args.hidden,
        heads               = args.heads,
        seq                 = args.seq,
        batch               = args.batch,
        dtype_bytes         = 4,   # fp32 (plugin precision="fp32")
        vocab_size          = getattr(model_config, "vocab_size", 1024),
        intermediate_size   = intermediate_size,
        num_key_value_heads = num_key_value_heads,
        mlp_gated           = mlp_gated,
    )

    # ------------------------------------------------------------------
    # Phase 2: Plan selection (auto or manual override).
    #
    # Normal mode: auto_plan() searches and scores all candidates.
    # Manual mode (--manual-pp/--manual-tp): skip search, use forced plan.
    #   dp is derived from world_size / (pp * tp).
    # ------------------------------------------------------------------
    # Store all evaluated plans for JSON export
    all_evaluated = []

    if args.manual_pp is not None and args.manual_tp is not None:
        # Manual override mode — for cost model validation (Priority 0).
        pp = args.manual_pp
        tp = args.manual_tp
        if world_size % (pp * tp) != 0:
            raise ValueError(
                f"Manual plan pp={pp} tp={tp}: world_size={world_size} not divisible by pp*tp={pp*tp}"
            )
        dp = world_size // (pp * tp)

        # Build a minimal result object with cost estimate.
        from colossalai.auto_parallel.hybrid_planner.cost_model import estimate_step_time
        from colossalai.auto_parallel.hybrid_planner.topology import classify_comms
        from colossalai.auto_parallel.hybrid_planner.search import PlanResult

        topology = classify_comms(node_gpus, pp, tp, dp, dp_outside=args.dp_outside)
        cost = estimate_step_time(cfg, pp, tp, dp, profile, topology, args.microbatches)

        result = PlanResult(pp=pp, tp=tp, dp=dp, cost=cost, topology=topology,
                            scored_table=[], pruned_table=[])
        training_dp_outside = args.dp_outside
        all_evaluated = [{
            "world_size": world_size,
            "dp_outside": args.dp_outside,
            "best_plan": {"pp": pp, "tp": tp, "dp": dp, "estimated_ms": cost.total * 1000},
            "all_candidates": [],
            "pruned_candidates": [],
            "status": "manual",
        }]

        if rank == 0:
            logger.info(
                f"[auto] MANUAL plan override: pp={pp}  tp={tp}  dp={dp}  "
                f"(estimated {cost.total*1000:.1f} ms/step)\n"
                f"       {cost}",
                ranks=[0],
            )
    else:
        # Normal auto-plan mode.
        # By default: try all prefix world-sizes AND both dp_outside variants.
        # With --fixed-world-size: only evaluate exact world_size with both dp_outside.
        # ------------------------------------------------------------------
        if args.fixed_world_size:
            if rank == 0:
                logger.info(
                    "[auto] Phase 2: searching best (pp, tp, dp) with FIXED world_size="
                    f"{world_size} (subset search disabled) ...",
                    ranks=[0],
                )
        else:
            if rank == 0:
                logger.info(
                    "[auto] Phase 2: searching best (pp, tp, dp, world_size, dp_outside) ...",
                    ranks=[0],
                )

        if args.memory_gb is None and profile.min_free_memory_gb > 0:
            memory_budget_gb = profile.min_free_memory_gb
        else:
            memory_budget_gb = args.memory_gb

        # Build search space
        if args.fixed_world_size:
            # Only exact world_size, no subsets
            prefix_ws = [world_size]
            prefix_nodes = [node_gpus]
        else:
            # Try all prefix world sizes from node_gpus
            # e.g. node_gpus=[2,2,2] -> prefix_ws=[2,4,6], prefix_nodes=[[2],[2,2],[2,2,2]]
            prefix_ws = []
            prefix_nodes = []
            cumsum = 0
            prefix = []
            for g in node_gpus:
                cumsum += g
                prefix.append(g)
                prefix_ws.append(cumsum)
                prefix_nodes.append(prefix.copy())

        best_result = None
        best_cost = float('inf')
        all_results = []   # list of (world_size, dp_outside, PlanResult or None)

        for ws, sub_nodes in zip(prefix_ws, prefix_nodes):
            for dp_outside_flag in (True, False):
                try:
                    res = auto_plan(
                        cfg              = cfg,
                        world_size       = ws,
                        node_gpus        = sub_nodes,
                        profile          = profile,
                        num_microbatches = args.microbatches,
                        memory_budget_gb = memory_budget_gb,
                        dp_outside       = dp_outside_flag,
                    )
                    cost_ms = res.cost.total * 1000
                    all_results.append((ws, dp_outside_flag, res))
                    if cost_ms < best_cost:
                        best_cost = cost_ms
                        best_result = res
                except ValueError:
                    # No feasible plan for this (world_size, dp_outside) combo
                    all_results.append((ws, dp_outside_flag, None))

        result = best_result
        pp = result.pp
        tp = result.tp
        dp = result.dp
        best_ws = result.cost.total * 1000   # will recompute below
        training_dp_outside = args.dp_outside

        # Find best_ws from all_results matching best_result
        for ws, dpo, res in all_results:
            if res is best_result:
                best_ws = ws
                best_dpo = dpo
                break

        # Build all_evaluated for JSON export (all combinations tried)
        for ws, dpo, res in all_results:
            if res is None:
                all_evaluated.append({
                    "world_size": ws,
                    "dp_outside": dpo,
                    "status": "no_feasible_plan",
                })
            else:
                all_evaluated.append({
                    "world_size": ws,
                    "dp_outside": dpo,
                    "best_plan": {
                        "pp": res.pp, "tp": res.tp, "dp": res.dp,
                        "estimated_ms": res.cost.total * 1000,
                    },
                    "all_candidates": [
                        {
                            "pp": row["pp"], "tp": row["tp"], "dp": row["dp"],
                            "total_ms": row["cost"].total * 1000,
                            "compute_ms": row["cost"].T_compute * 1000,
                            "bubble_ms": row["cost"].T_bubble * 1000,
                            "tp_comm_ms": row["cost"].T_tp_comm * 1000,
                            "pp_comm_ms": row["cost"].T_pp_comm * 1000,
                            "dp_comm_ms": row["cost"].T_dp_comm * 1000,
                            "exec_oh_ms": row["cost"].T_execution * 1000,
                            "embed_ms": row["cost"].T_embedding * 1000,
                            "lm_head_ms": row["cost"].T_lm_head * 1000,
                        }
                        for row in res.scored_table
                    ],
                    "pruned_candidates": [
                        {"pp": row["pp"], "tp": row["tp"], "dp": row["dp"], "reason": row["reason"]}
                        for row in res.pruned_table
                    ],
                    "status": "feasible",
                })

        if rank == 0:
            # Build comprehensive comparison file with ALL candidates per combo
            comparison_lines = []
            comparison_lines.append("=" * 80)
            comparison_lines.append("DETAILED PLAN EVALUATION REPORT")
            comparison_lines.append("=" * 80)
            comparison_lines.append("")
            comparison_lines.append(f"Model: layers={args.layers}, hidden={args.hidden}, batch={args.batch}, microbatches={args.microbatches}")
            comparison_lines.append(f"Cluster: {node_gpus} (total {world_size} GPUs)")
            comparison_lines.append("")

            # ── Section 1: ALL candidate plans for each (world_size, dp_outside) ──
            comparison_lines.append("-" * 80)
            comparison_lines.append("SECTION 1: ALL CANDIDATE PLANS BY (world_size, dp_outside)")
            comparison_lines.append("-" * 80)
            comparison_lines.append("")

            for ws, dpo, res in all_results:
                comparison_lines.append(f"\n{'='*40}")
                comparison_lines.append(f"world_size={ws}  dp_outside={dpo}")
                comparison_lines.append(f"{'='*40}")

                if res is None:
                    comparison_lines.append("  Status: NO FEASIBLE PLAN (all candidates pruned)")
                    continue

                comparison_lines.append(f"  Best plan: pp={res.pp} tp={res.tp} dp={res.dp}  ({res.cost.total*1000:.1f} ms)")
                comparison_lines.append("")

                # All scored candidates with breakdown
                cand_header = f"    {'plan':>12}  {'total':>8}  {'compute':>8}  {'bubble':>7}  {'TP':>6}  {'PP':>6}  {'DP':>6}  {'exec':>5}  {'embed':>6}  {'lmhead':>6}"
                comparison_lines.append(cand_header)
                comparison_lines.append(f"    {'-'*96}")
                for row in res.scored_table:
                    p, t, d = row["pp"], row["tp"], row["dp"]
                    c = row["cost"]
                    best_mark = " *" if (p == res.pp and t == res.tp and d == res.dp) else "  "
                    comparison_lines.append(
                        f"    pp={p} tp={t} dp={d}{best_mark}  "
                        f"{c.total*1000:>8.1f}  {c.T_compute*1000:>8.1f}  {c.T_bubble*1000:>7.1f}  "
                        f"{c.T_tp_comm*1000:>6.1f}  {c.T_pp_comm*1000:>6.1f}  {c.T_dp_comm*1000:>6.1f}  "
                        f"{c.T_execution*1000:>5.1f}  {c.T_embedding*1000:>6.1f}  {c.T_lm_head*1000:>6.1f}"
                    )

                # Pruned candidates
                if res.pruned_table:
                    comparison_lines.append("")
                    comparison_lines.append("    Pruned:")
                    for row in res.pruned_table:
                        comparison_lines.append(
                            f"      pp={row['pp']} tp={row['tp']} dp={row['dp']}  → {row['reason']}"
                        )

            # ── Section 2: Summary (best plan per combo) ──
            comparison_lines.append("")
            comparison_lines.append("=" * 80)
            comparison_lines.append("SECTION 2: SUMMARY — Best plan per (world_size, dp_outside)")
            comparison_lines.append("=" * 80)
            comparison_lines.append("")

            header = f"{'world_size':>10}  {'dp_outside':>11}  {'best_pp':>7}  {'best_tp':>7}  {'best_dp':>7}  {'est_ms':>10}  {'status':>20}"
            comparison_lines.append(header)
            comparison_lines.append("-" * len(header))
            for ws, dpo, res in all_results:
                if res is None:
                    line = f"{ws:>10}  {str(dpo):>11}  {'--':>7}  {'--':>7}  {'--':>7}  {'--':>10}  {'no feasible plan':>20}"
                else:
                    marker = "  <<< BEST" if (res is best_result) else ""
                    line = (
                        f"{ws:>10}  {str(dpo):>11}  {res.pp:>7}  {res.tp:>7}  {res.dp:>7}  "
                        f"{res.cost.total*1000:>10.1f}  {'feasible':>20}{marker}"
                    )
                comparison_lines.append(line)
            comparison_lines.append("")
            comparison_lines.append(
                f"ABSOLUTE BEST: world_size={best_ws}  pp={pp}  tp={tp}  dp={dp}  "
                f"dp_outside={best_dpo}  (estimated {best_cost:.1f} ms/step)"
            )
            comparison_lines.append("")
            comparison_lines.append("=" * 80)

            # Print summary to console (not everything to avoid clutter)
            # Find index of SECTION 2 (use substring match since exact text is "SECTION 2: SUMMARY...")
            section2_idx = next(
                (i for i, s in enumerate(comparison_lines) if "SECTION 2" in s),
                len(comparison_lines) - 1
            )
            for line in comparison_lines[max(0, section2_idx - 1):]:
                if line.startswith("SECTION") or line.startswith("=") or line.startswith("-") or line.startswith("ABSOLUTE"):
                    logger.info(f"[auto] {line}", ranks=[0])
                elif not line.startswith("    ") and len(line) < 100:
                    logger.info(f"[auto] {line}", ranks=[0])

            # Save FULL report to file
            comparison_file = os.path.join(_here, "results", f"comparison_{args.model}_{world_size}gpu_{args.layers}L_{args.hidden}H.txt")
            os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
            with open(comparison_file, "w") as f:
                f.write("\n".join(comparison_lines))
            logger.info(
                f"[auto] Full comparison report saved to {comparison_file}",
                ranks=[0],
            )

            if not args.fixed_world_size and best_ws < world_size:
                n_optimal_nodes = len(prefix_nodes[prefix_ws.index(best_ws)])
                logger.warning(
                    f"[auto] WARNING: Best plan uses only {best_ws}/{world_size} GPUs. "
                    f"To train with this plan, relaunch with ONLY the first "
                    f"{n_optimal_nodes} node(s). "
                    f"Current run will train with all {world_size} GPUs.",
                    ranks=[0],
                )

                # Write relaunch flag for launch_nodes.sh to detect
                relaunch_file = os.path.join(_here, "RELAUNCH.txt")
                with open(relaunch_file, "w") as f:
                    f.write(f"{n_optimal_nodes}\n")
                logger.info(
                    f"[auto] Wrote relaunch flag to {relaunch_file} "
                    f"(optimal nodes = {n_optimal_nodes}). "
                    f"launch_nodes.sh will auto-relaunch.",
                    ranks=[0],
                )

                # Override to use the full world_size for training
                logger.info(
                    f"[auto] Re-selecting plan for ACTUAL world_size={world_size} "
                    f"(training will use all GPUs) ...",
                    ranks=[0],
                )
                result = auto_plan(
                    cfg              = cfg,
                    world_size       = world_size,
                    node_gpus        = node_gpus,
                    profile          = profile,
                    num_microbatches = args.microbatches,
                    memory_budget_gb = memory_budget_gb,
                    dp_outside       = best_dpo,
                )
                pp = result.pp
                tp = result.tp
                dp = result.dp
                training_dp_outside = best_dpo
                logger.info(
                    f"[auto] Training plan: pp={pp}  tp={tp}  dp={dp}  "
                    f"(estimated {result.cost.total*1000:.1f} ms/step)",
                    ranks=[0],
                )
            else:
                logger.info(
                    f"[auto] Best plan: pp={pp}  tp={tp}  dp={dp}  "
                    f"(estimated {result.cost.total*1000:.1f} ms/step)\n"
                    f"       {result.cost}",
                    ranks=[0],
                )
                # Print the full scored table so you can see all candidates
                logger.info("[auto] Full candidate table:", ranks=[0])
                result.print_table()

    # ------------------------------------------------------------------
    # Phase 3: Train with the auto-selected plan.
    #
    # HybridParallelPlugin wires up:
    #   - ShardFormer  → applies TP sharding to GPT2 attention + MLP
    #   - PipelineStage → PP scheduling (1F1B) across stages
    #   - DDP          → DP gradient sync (when dp > 1)
    #
    # dp_outside=True (default): ProcessGroupMesh shape = (dp, pp, tp)
    #   rank formula: rank = dp_rank*(pp*tp) + pp_rank*tp + tp_rank
    #   dp_rank = rank // (pp * tp)
    # dp_outside=False: ProcessGroupMesh shape = (pp, dp, tp)
    #   rank formula: rank = pp_rank*(dp*tp) + dp_rank*tp + tp_rank
    #   dp_rank = (rank // tp) % dp
    # ------------------------------------------------------------------
    if rank == 0:
        logger.info("[auto] Phase 3: building model and starting training ...", ranks=[0])

    if args.model == "gpt2":
        model = transformers.GPT2LMHeadModel(model_config)
    elif args.model in ("llama", "llama32", "smollm"):
        model = transformers.LlamaForCausalLM(model_config)
    elif args.model == "qwen25":
        model = transformers.Qwen2ForCausalLM(model_config)
    else:
        raise ValueError(f"Unknown --model: {args.model}")

    # 1F1B pipeline scheduler requires at least as many microbatches as stages.
    if pp > 1 and args.microbatches < pp:
        raise ValueError(
            f"Pipeline parallelism with pp={pp} requires num_microbatches >= pp, "
            f"but got microbatches={args.microbatches}. "
            f"Try --microbatches {pp} or larger."
        )

    plugin = HybridParallelPlugin(
        pp_size              = pp,
        tp_size              = tp,
        num_microbatches     = args.microbatches,
        enable_all_optimization = False,
        precision            = "fp32",
        dp_outside           = training_dp_outside,
    )
    booster   = Booster(plugin=plugin)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model, optimizer, *_ = booster.boost(model, optimizer)

    # Derive dp_rank correctly from the mesh layout.
    # dp_outside=True  → mesh (dp, pp, tp) → dp_rank = rank // (pp*tp)
    # dp_outside=False → mesh (pp, dp, tp) → dp_rank = (rank // tp) % dp
    if args.dp_outside:
        dp_rank = rank // (pp * tp)
    else:
        dp_rank = (rank // tp) % dp

    def make_batch(step: int):
        """Produce distinct data per DP replica, reproducible across runs.
        Pass the full batch — execute_pipeline splits it into microbatches internally.
        """
        g = torch.Generator()
        g.manual_seed(step * 1000 + dp_rank)
        ids  = torch.randint(0, model_config.vocab_size, (args.batch, args.seq), generator=g)
        mask = torch.ones(args.batch, args.seq, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": ids}

    def criterion(outputs, _inputs):
        return outputs.loss

    if rank == 0:
        logger.info(
            f"[auto] Training  pp={pp} tp={tp} dp={dp}  "
            f"layers={args.layers} hidden={args.hidden} "
            f"batch={args.batch} microbatches={args.microbatches} "
            f"steps={args.steps}",
            ranks=[0],
        )

    # Collect per-step wall times for the final comparison report.
    step_times_ms = []

    for step in range(args.steps):
        batch = make_batch(step)

        # Synchronise before timing so GPU is not still finishing previous work.
        torch.cuda.synchronize()
        t_step_start = time.perf_counter()

        if pp == 1:
            # No pipeline parallelism — standard forward/backward.
            # The model is still wrapped by ShardFormer for TP and DDP for DP.
            model.train()
            # Move batch to the same device as the model.
            device = next(model.parameters()).device
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch_gpu)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            outputs = {"loss": loss.item()}
            pp_rank = 0
        else:
            outputs = booster.execute_pipeline(
                iter([batch]),
                model,
                criterion=criterion,
                optimizer=optimizer,
                return_loss=True,
            )
            optimizer.step()
            optimizer.zero_grad()
            pp_rank = (rank // tp) % pp if not args.dp_outside else (rank % (pp * tp)) // tp

        # Synchronise after so the timer includes all GPU work.
        torch.cuda.synchronize()
        t_step_ms = (time.perf_counter() - t_step_start) * 1000
        step_times_ms.append(t_step_ms)

        if outputs.get("loss") is not None:
            logger.info(
                f"Step {step+1}/{args.steps}  loss={outputs['loss']:.4f}  "
                f"wall={t_step_ms:.1f}ms  "
                f"[rank={rank} pp={pp_rank} dp={dp_rank} tp={rank%tp}]",
                ranks=[rank],
            )

        dist.barrier()

    # ------------------------------------------------------------------
    # Validation report — compare profiled estimates vs actual training.
    #
    # Only rank 0 prints.  The step times are local to rank 0; a barrier()
    # after each step ensures ranks are synced, but we do not all-reduce
    # the timings (rank 0's time is representative for the pipeline master).
    # ------------------------------------------------------------------
    if rank == 0:
        # Compute cost breakdown from profiler values (same as Phase 2).
        from colossalai.auto_parallel.hybrid_planner.cost_model import estimate_step_time
        from colossalai.auto_parallel.hybrid_planner.topology import classify_comms
        topology = classify_comms(node_gpus, pp, tp, dp, dp_outside=args.dp_outside)
        estimated = estimate_step_time(cfg, pp, tp, dp, profile, topology, args.microbatches)

        avg_actual_ms = sum(step_times_ms) / len(step_times_ms)

        logger.info(
            f"\n[auto] ── Profiler validation report ──────────────────────────────\n"
            f"  Profiled T_block      : {profile.T_block*1000:.3f} ms  (isolated block fwd+bwd)\n"
            f"  Estimated step time   : {estimated.total*1000:.1f} ms  (cost model from profiled α/β/T_block)\n"
            f"  Actual avg step time  : {avg_actual_ms:.1f} ms  (wall clock on rank 0)\n"
            f"  Ratio actual/estimate : {avg_actual_ms / (estimated.total*1000):.2f}×\n"
            f"\n"
            f"  Cost model breakdown (estimated):\n"
            f"    compute = {estimated.T_compute*1000:.1f} ms  "
            f"bubble = {estimated.T_bubble*1000:.1f} ms  "
            f"TP = {estimated.T_tp_comm*1000:.1f} ms  "
            f"PP = {estimated.T_pp_comm*1000:.1f} ms  "
            f"DP = {estimated.T_dp_comm*1000:.1f} ms  "
            f"exec OH = {estimated.T_execution*1000:.1f} ms\n"
            f"    embed = {estimated.T_embedding*1000:.1f} ms  "
            f"LM head = {estimated.T_lm_head*1000:.1f} ms  "
            f"(vocab={cfg.vocab_size})\n"
            f"\n"
            f"  Interpretation:\n"
            f"    ratio < 1.5 → profiler estimates are representative\n"
            f"    ratio > 2.0 → framework/Python overhead significant (normal for tiny models)\n"
            f"────────────────────────────────────────────────────────────────────",
            ranks=[0],
        )

    # ------------------------------------------------------------------
    # JSON result export — rank 0 writes a machine-readable summary
    # for downstream benchmark aggregation.
    # ------------------------------------------------------------------
    if rank == 0:
        import json
        result_dict = {
            "plan": {
                "pp": pp,
                "tp": tp,
                "dp": dp,
                "world_size": world_size,
                "node_gpus": node_gpus,
            },
            "model": {
                "layers": args.layers,
                "hidden": args.hidden,
                "heads": args.heads,
                "seq": args.seq,
                "batch": args.batch,
                "microbatches": args.microbatches,
                "steps": args.steps,
                "dtype_bytes": cfg.dtype_bytes,
            },
            "profile": {
                "alpha_intra_us": profile.alpha_intra * 1e6,
                "beta_intra_ns_per_B": profile.beta_intra * 1e9,
                "alpha_cross_us": profile.alpha_cross * 1e6,
                "beta_cross_ns_per_B": profile.beta_cross * 1e9,
                "T_block_ms": profile.T_block * 1e3,
                "min_free_memory_gb": profile.min_free_memory_gb,
            },
            "estimated_step_time_ms": estimated.total * 1000,
            "estimated_breakdown_ms": {
                "compute": estimated.T_compute * 1000,
                "bubble": estimated.T_bubble * 1000,
                "tp_comm": estimated.T_tp_comm * 1000,
                "pp_comm": estimated.T_pp_comm * 1000,
                "dp_comm": estimated.T_dp_comm * 1000,
                "execution_overhead": estimated.T_execution * 1000,
                "embedding": estimated.T_embedding * 1000,
                "lm_head": estimated.T_lm_head * 1000,
            },
            "actual": {
                "avg_step_time_ms": avg_actual_ms,
                "step_times_ms": step_times_ms,
            },
            "scored_candidates": [
                {
                    "pp": row["pp"],
                    "tp": row["tp"],
                    "dp": row["dp"],
                    "total_ms": row["cost"].total * 1000,
                }
                for row in result.scored_table
            ],
            "pruned_candidates": [
                {
                    "pp": row["pp"],
                    "tp": row["tp"],
                    "dp": row["dp"],
                    "reason": row["reason"],
                }
                for row in result.pruned_table
            ],
            "dp_outside": args.dp_outside,
            "all_evaluated_plans": all_evaluated,
        }
        dp_suffix = "_no_dp_outside" if not args.dp_outside else ""
        out_path = os.path.join(
            _here, "results",
            f"{args.model}_{world_size}gpu_{args.layers}L_{args.hidden}H_"
            f"{args.batch}B_pp{pp}_tp{tp}_dp{dp}{dp_suffix}.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        logger.info(
            f"[auto] Results saved to {out_path}",
            ranks=[0],
        )

    if rank == 0:
        logger.info(
            f"[auto] Done. Auto 3D parallel training complete "
            f"(pp={pp} tp={tp} dp={dp}).",
            ranks=[0],
        )


if __name__ == "__main__":
    main()
