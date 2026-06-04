"""
Search module for the hybrid auto-planner.

Enumerates every valid (pp, tp, dp) factorisation of world_size, prunes
infeasible or obviously bad candidates, scores the rest with the cost model,
and returns the best plan.

No torch.distributed dependency — pure Python, unit-testable on a laptop.
Call auto_plan() after profile_cluster() to get the best (pp, tp, dp) to
pass to HybridParallelPlugin.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .cost_model import CostBreakdown, ModelConfig, estimate_step_time, _param_bytes_per_layer
from .profiler import ClusterProfile
from .topology import TopologyInfo, classify_comms


# ---------------------------------------------------------------------------
# Memory helper
# ---------------------------------------------------------------------------

def _fits_in_memory(cfg: ModelConfig, pp: int, tp: int,
                    memory_budget_gb: float) -> bool:
    """
    Rough check: does the model shard for one GPU fit in memory?

    Each GPU holds:
      - parameters:    layers/pp layers, each with 1/tp of the params
      - gradients:     same size as parameters
      - optimizer state (Adam): 2× parameters in fp32 regardless of dtype
      - activations:   hard to estimate statically; we use a small constant
                       multiplier of the parameter size as a proxy.

    Adam optimizer states are always stored in fp32 (4 bytes/element) even
    when training in bf16.  So per-GPU memory ≈:
      params_bytes  = param_bytes_per_layer × (layers/pp) / tp
      grads_bytes   = params_bytes
      optim_bytes   = params_bytes × (4 / cfg.dtype_bytes) × 2   # fp32 Adam
      activ_bytes   = params_bytes × 0.5   # rough proxy

    Total ≈ params × (2 + 8/dtype_bytes + 0.5)
    """
    param_bytes = _param_bytes_per_layer(cfg, cfg.dtype_bytes)
    shard_param = param_bytes * (cfg.layers // pp) // tp
    shard_grad  = shard_param
    # Adam m and v are fp32 regardless of training dtype
    shard_optim = shard_param * (4 / cfg.dtype_bytes) * 2
    shard_activ = shard_param * 0.5
    total_bytes = shard_param + shard_grad + shard_optim + shard_activ
    return total_bytes <= memory_budget_gb * (1024 ** 3)


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------

def _all_candidates(world_size: int) -> List[Tuple[int, int, int]]:
    """
    Return every (pp, tp, dp) triple where pp × tp × dp == world_size.
    Ordered by pp ascending, then tp ascending.
    """
    result = []
    for pp in range(1, world_size + 1):
        if world_size % pp != 0:
            continue
        remainder = world_size // pp
        for tp in range(1, remainder + 1):
            if remainder % tp != 0:
                continue
            dp = remainder // tp
            result.append((pp, tp, dp))
    return result


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PlanResult:
    """
    Full result from auto_plan(), including the winning plan and the scored
    table so you can see why one plan beat the others.
    """
    pp:               int
    tp:               int
    dp:               int
    cost:             CostBreakdown
    topology:         TopologyInfo
    scored_table:     List[Dict]   # all candidates that were scored (not pruned)
    pruned_table:     List[Dict]   # all candidates that were pruned and why

    def __str__(self) -> str:
        lines = [
            f"Best plan: pp={self.pp}  tp={self.tp}  dp={self.dp}",
            f"Estimated step time: {self.cost.total*1000:.2f} ms",
            str(self.cost),
            "",
            f"Topology: tp_intra={self.topology.tp_intra_node}  "
            f"pp_intra={self.topology.pp_intra_node}  "
            f"dp_intra={self.topology.dp_intra_node}",
        ]
        return "\n".join(lines)

    def print_table(self) -> None:
        """Print all scored and pruned candidates."""
        header = f"{'plan':>20}  {'total ms':>9}  {'compute':>8}  {'bubble':>7}  "
        header += f"{'TP comm':>8}  {'PP comm':>8}  {'DP comm':>8}  {'overhead':>8}  tp_intra  pp_intra  dp_intra"
        print(header)
        print("-" * len(header))
        ms = lambda t: f"{t * 1000:.2f}"
        for row in self.scored_table:
            pp, tp, dp = row["pp"], row["tp"], row["dp"]
            bd = row["cost"]
            topo = row["topology"]
            best_marker = " *" if (pp == self.pp and tp == self.tp and dp == self.dp) else "  "
            print(
                f"  pp={pp} tp={tp} dp={dp}{best_marker}  "
                f"{ms(bd.total):>9}  {ms(bd.T_compute):>8}  {ms(bd.T_bubble):>7}  "
                f"{ms(bd.T_tp_comm):>8}  {ms(bd.T_pp_comm):>8}  {ms(bd.T_dp_comm):>8}  "
                f"{ms(bd.T_step_overhead):>8}  "
                f"{str(topo.tp_intra_node):>8}  {str(topo.pp_intra_node):>8}  "
                f"{str(topo.dp_intra_node):>8}"
            )
        if self.pruned_table:
            print()
            print("Pruned:")
            for row in self.pruned_table:
                pp, tp, dp = row["pp"], row["tp"], row["dp"]
                print(f"  pp={pp} tp={tp} dp={dp}  → {row['reason']}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def auto_plan(
    cfg:              ModelConfig,
    world_size:       int,
    node_gpus:        List[int],
    profile:          ClusterProfile,
    num_microbatches: int = 4,
    memory_budget_gb: Optional[float] = None,
    dp_outside:       bool = True,
) -> PlanResult:
    """
    Find the best (pp, tp, dp) parallelism plan for the given cluster and model.

    Args:
        cfg:              ModelConfig — layers, hidden, heads, seq, batch, dtype_bytes.
        world_size:       total number of GPUs (must equal sum(node_gpus)).
        node_gpus:        GPU counts per node in rank order, e.g. [2, 4, 2].
        profile:          ClusterProfile from profiler.profile_cluster().
        num_microbatches: microbatch count for the pipeline schedule.
        memory_budget_gb: if given, prune plans where the model shard does not
                          fit on a single GPU.  Use torch.cuda.get_device_properties
                          to get the real value (e.g. 24.0 for a 24 GB GPU).
        dp_outside:       must match the dp_outside flag of HybridParallelPlugin
                          (default True, which is also HybridParallelPlugin's default).

    Returns:
        PlanResult with .pp, .tp, .dp, .cost, .topology, and .print_table().

    Raises:
        ValueError: if no feasible candidate exists after pruning.
    """
    if sum(node_gpus) != world_size:
        raise ValueError(
            f"sum(node_gpus)={sum(node_gpus)} != world_size={world_size}"
        )

    min_gpus_per_node = min(node_gpus)
    candidates = _all_candidates(world_size)
    scored: List[Dict] = []
    pruned: List[Dict] = []

    for pp, tp, dp in candidates:
        # ── Pruning rule 1: TP cross-node ──────────────────────────────────
        # If tp > min GPUs per node, at least one TP group will span nodes.
        # TP AllReduce fires every layer — cross-node TP is ~100× slower.
        if tp > min_gpus_per_node:
            pruned.append({
                "pp": pp, "tp": tp, "dp": dp,
                "reason": f"tp={tp} > min_gpus_per_node={min_gpus_per_node} → cross-node TP",
            })
            continue

        # ── Pruning rule 2: layers must divide evenly across stages ─────────
        if cfg.layers % pp != 0:
            pruned.append({
                "pp": pp, "tp": tp, "dp": dp,
                "reason": f"layers={cfg.layers} not divisible by pp={pp}",
            })
            continue

        # ── Pruning rule 3: batch must divide into microbatches ─────────────
        if cfg.batch % num_microbatches != 0:
            pruned.append({
                "pp": pp, "tp": tp, "dp": dp,
                "reason": f"batch={cfg.batch} not divisible by num_microbatches={num_microbatches}",
            })
            continue

        # ── Pruning rule 4: memory budget ───────────────────────────────────
        if memory_budget_gb is not None and not _fits_in_memory(cfg, pp, tp, memory_budget_gb):
            pruned.append({
                "pp": pp, "tp": tp, "dp": dp,
                "reason": f"model shard exceeds memory_budget_gb={memory_budget_gb}",
            })
            continue

        # ── Score ───────────────────────────────────────────────────────────
        topology = classify_comms(node_gpus, pp, tp, dp, dp_outside=dp_outside)
        cost     = estimate_step_time(cfg, pp, tp, dp, profile, topology, num_microbatches)
        scored.append({
            "pp": pp, "tp": tp, "dp": dp,
            "cost": cost, "topology": topology,
        })

    if not scored:
        pruned_summary = "; ".join(
            f"pp={r['pp']} tp={r['tp']} dp={r['dp']} ({r['reason']})"
            for r in pruned
        )
        raise ValueError(
            f"No feasible parallelism plan found after pruning. "
            f"Pruned candidates: {pruned_summary}"
        )

    # Pick the candidate with the lowest estimated total step time
    best_row = min(scored, key=lambda r: r["cost"].total)

    return PlanResult(
        pp           = best_row["pp"],
        tp           = best_row["tp"],
        dp           = best_row["dp"],
        cost         = best_row["cost"],
        topology     = best_row["topology"],
        scored_table = scored,
        pruned_table = pruned,
    )
