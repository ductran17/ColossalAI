"""
Auto 3D Parallel training — full pipeline: profile → plan → train.

Three phases run automatically on the cluster:
  Phase 1  profile_cluster()  ~2 s   measure α/β/T_block on real hardware
  Phase 2  auto_plan()        <1 ms  enumerate + score (pp,tp,dp) candidates
  Phase 3  HybridParallelPlugin       train with the chosen plan

No pp/tp flags needed — the planner picks them.

Usage:
  bash launch_3nodes.sh --auto                       # fully automatic
  bash launch_3nodes.sh --auto --layers 8 --batch 4  # specify model size

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
    p.add_argument("--layers",       type=int, default=8,   help="Transformer layers")
    p.add_argument("--hidden",       type=int, default=256, help="Hidden dimension")
    p.add_argument("--heads",        type=int, default=4,   help="Attention heads")
    p.add_argument("--seq",          type=int, default=64,  help="Sequence length")
    # Training
    p.add_argument("--batch",        type=int, default=4,   help="Global batch size (total sequences per step)")
    p.add_argument("--microbatches", type=int, default=4,   help="Microbatches for pipeline schedule")
    p.add_argument("--steps",        type=int, default=3,   help="Training steps")
    # Planner
    p.add_argument("--memory-gb",    type=float, default=None,
                   help="Per-GPU memory budget in GB for pruning (e.g. 24.0). "
                        "If omitted, memory pruning is skipped.")
    p.add_argument("--dp-outside",   action="store_true", default=True,
                   help="dp_outside flag for HybridParallelPlugin (default True)")
    # Manual plan override (for cost model validation / Priority 0)
    p.add_argument("--manual-pp",    type=int, default=None,
                   help="Force pipeline-parallel degree (bypass auto_plan). "
                        "If set, --manual-tp must also be set.")
    p.add_argument("--manual-tp",    type=int, default=None,
                   help="Force tensor-parallel degree (bypass auto_plan). "
                        "If set, --manual-pp must also be set.")
    # Profiler
    p.add_argument("--warmup",       type=int, default=3,   help="Profiler warmup iters")
    p.add_argument("--repeat",       type=int, default=10,  help="Profiler timed iters")
    return p.parse_args()


def main():
    args = parse_args()
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
        warmup=args.warmup,
        repeat=args.repeat,
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
            f"       min_free_mem={profile.min_free_memory_gb:.1f} GB  "
            f"(across all GPUs at profiling time)",
            ranks=[0],
        )

    # ------------------------------------------------------------------
    # Phase 2: Run the auto-planner.
    #
    # auto_plan() is pure Python (no distributed ops). Because
    # ClusterProfile is already all-reduced, every rank computes the
    # exact same plan independently — no broadcast needed.
    # ------------------------------------------------------------------
    cfg = ModelConfig(
        layers      = args.layers,
        hidden      = args.hidden,
        heads       = args.heads,
        seq         = args.seq,
        batch       = args.batch,
        dtype_bytes = 4,   # fp32 (plugin precision="fp32")
    )

    # ------------------------------------------------------------------
    # Phase 2: Plan selection (auto or manual override).
    #
    # Normal mode: auto_plan() searches and scores all candidates.
    # Manual mode (--manual-pp/--manual-tp): skip search, use forced plan.
    #   dp is derived from world_size / (pp * tp).
    # ------------------------------------------------------------------
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

        if rank == 0:
            logger.info(
                f"[auto] MANUAL plan override: pp={pp}  tp={tp}  dp={dp}  "
                f"(estimated {cost.total*1000:.1f} ms/step)\n"
                f"       {cost}",
                ranks=[0],
            )
    else:
        # Normal auto-plan mode.
        if rank == 0:
            logger.info("[auto] Phase 2: searching best (pp, tp, dp) ...", ranks=[0])

        # If the user did not pass --memory-gb, use the measured free memory
        # as the conservative default budget.  If the user passed an explicit
        # budget, respect it (they may want a head-room margin).
        if args.memory_gb is None and profile.min_free_memory_gb > 0:
            memory_budget_gb = profile.min_free_memory_gb
        else:
            memory_budget_gb = args.memory_gb

        if rank == 0 and memory_budget_gb is not None:
            logger.info(
                f"[auto] Memory budget for planning: {memory_budget_gb:.1f} GB",
                ranks=[0],
            )

        result = auto_plan(
            cfg              = cfg,
            world_size       = world_size,
            node_gpus        = node_gpus,
            profile          = profile,
            num_microbatches = args.microbatches,
            memory_budget_gb = memory_budget_gb,
            dp_outside       = args.dp_outside,
        )

        pp = result.pp
        tp = result.tp
        dp = result.dp

        if rank == 0:
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

    config = transformers.GPT2Config(
        n_positions = args.seq,
        n_layer     = args.layers,
        n_head      = args.heads,
        n_embd      = args.hidden,
        n_inner     = args.hidden * 4,
        vocab_size  = 1024,
        resid_pdrop = 0.0,
        attn_pdrop  = 0.0,
    )
    model = transformers.GPT2LMHeadModel(config)

    plugin = HybridParallelPlugin(
        pp_size              = pp,
        tp_size              = tp,
        num_microbatches     = args.microbatches,
        enable_all_optimization = False,
        precision            = "fp32",
        dp_outside           = args.dp_outside,
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
        ids  = torch.randint(0, config.vocab_size, (args.batch, args.seq), generator=g)
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
            f"DP = {estimated.T_dp_comm*1000:.1f} ms\n"
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
        }
        out_path = os.path.join(
            _here, "results",
            f"auto_parallel_{world_size}gpu_{args.layers}L_{args.hidden}H_"
            f"{args.batch}B_pp{pp}_tp{tp}_dp{dp}.json"
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
