"""
Topology classifier for the hybrid auto-planner.

Given:
  - node_gpus: list of GPU counts per node, e.g. [2, 4, 2]
  - (pp, tp, dp): the parallelism plan to evaluate

Determines whether each communication type (TP AllReduce, PP P2P, DP AllReduce)
is intra-node (fast PCIe) or cross-node (slow Ethernet).

No torch.distributed dependency — pure Python, unit-testable on a laptop.

Rank layout (HybridParallelPlugin ProcessGroupMesh order = PP, DP, TP):
  rank = pp_rank × (dp × tp) + dp_rank × tp + tp_rank

Example for pp=2, tp=2, dp=2 on cluster [2, 4, 2]:
  rank 0: pp=0 dp=0 tp=0  → node18
  rank 1: pp=0 dp=0 tp=1  → node18
  rank 2: pp=0 dp=1 tp=0  → node20
  rank 3: pp=0 dp=1 tp=1  → node20
  rank 4: pp=1 dp=0 tp=0  → node20
  rank 5: pp=1 dp=0 tp=1  → node20
  rank 6: pp=1 dp=1 tp=0  → node16
  rank 7: pp=1 dp=1 tp=1  → node16
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TopologyInfo:
    """
    Classification of each communication type for a given (pp, tp, dp) plan
    on a specific cluster topology.

    True  = intra-node (fast, PCIe / NVLink).
    False = cross-node (slow, Ethernet).

    Use these flags to select the right α/β when estimating communication cost.
    """
    tp_intra_node: bool   # True if ALL TP groups are contained within a single node
    pp_intra_node: bool   # True if ALL PP stage boundaries stay within a single node
    dp_intra_node: bool   # True if ALL DP groups are contained within a single node

    # Mirror fields for cost_model.py readability
    @property
    def tp_cross_node(self) -> bool:
        return not self.tp_intra_node

    @property
    def pp_cross_node(self) -> bool:
        return not self.pp_intra_node

    @property
    def dp_cross_node(self) -> bool:
        return not self.dp_intra_node


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_rank_to_node(node_gpus: List[int]) -> List[int]:
    """
    Build a list where rank_to_node[global_rank] = node_index.

    Example: node_gpus=[2, 4, 2] →
      rank_to_node = [0, 0, 1, 1, 1, 1, 2, 2]
    """
    rank_to_node: List[int] = []
    for node_idx, n_gpus in enumerate(node_gpus):
        rank_to_node.extend([node_idx] * n_gpus)
    return rank_to_node


def _global_rank(pp_rank: int, dp_rank: int, tp_rank: int,
                 dp: int, tp: int) -> int:
    """
    Convert (pp_rank, dp_rank, tp_rank) to global rank.

    ProcessGroupMesh layout used by HybridParallelPlugin:
      axes = (PP=0, DP=1, TP=2)
      rank = pp_rank × (dp × tp) + dp_rank × tp + tp_rank
    """
    return pp_rank * (dp * tp) + dp_rank * tp + tp_rank


def _all_same_node(ranks: List[int], rank_to_node: List[int]) -> bool:
    """Return True if every rank in the list maps to the same node."""
    nodes = {rank_to_node[r] for r in ranks}
    return len(nodes) == 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_comms(node_gpus: List[int], pp: int, tp: int, dp: int) -> TopologyInfo:
    """
    Classify TP, PP, and DP communications as intra-node or cross-node.

    Args:
        node_gpus: GPU counts per node in rank order, e.g. [2, 4, 2].
                   Total must equal pp * tp * dp.
        pp: pipeline-parallel degree.
        tp: tensor-parallel degree.
        dp: data-parallel degree.

    Returns:
        TopologyInfo with tp_intra_node, pp_intra_node, dp_intra_node booleans.
        A field is True only if EVERY group of that type is intra-node.

    Raises:
        ValueError: if sum(node_gpus) != pp * tp * dp.
    """
    world_size = sum(node_gpus)
    if world_size != pp * tp * dp:
        raise ValueError(
            f"sum(node_gpus)={world_size} != pp×tp×dp={pp}×{tp}×{dp}={pp*tp*dp}"
        )

    rank_to_node = _build_rank_to_node(node_gpus)

    # ------------------------------------------------------------------
    # TP groups: same (pp_rank, dp_rank), tp_rank varies over [0..tp)
    # Each group has `tp` members that communicate via AllReduce every layer.
    # ------------------------------------------------------------------
    tp_intra = True
    for pp_r in range(pp):
        for dp_r in range(dp):
            group = [_global_rank(pp_r, dp_r, tp_r, dp, tp)
                     for tp_r in range(tp)]
            if not _all_same_node(group, rank_to_node):
                tp_intra = False
                break
        if not tp_intra:
            break

    # ------------------------------------------------------------------
    # PP groups: same (dp_rank, tp_rank), adjacent pp_ranks communicate
    # via P2P send/recv at each microbatch boundary.
    # We check every adjacent (pp_r, pp_r+1) pair.
    # ------------------------------------------------------------------
    pp_intra = True
    for dp_r in range(dp):
        for tp_r in range(tp):
            for pp_r in range(pp - 1):
                src = _global_rank(pp_r,     dp_r, tp_r, dp, tp)
                dst = _global_rank(pp_r + 1, dp_r, tp_r, dp, tp)
                if rank_to_node[src] != rank_to_node[dst]:
                    pp_intra = False
                    break
            if not pp_intra:
                break
        if not pp_intra:
            break

    # ------------------------------------------------------------------
    # DP groups: same (pp_rank, tp_rank), dp_rank varies over [0..dp)
    # Members perform gradient AllReduce (mostly overlapped with backward).
    # ------------------------------------------------------------------
    dp_intra = True
    for pp_r in range(pp):
        for tp_r in range(tp):
            group = [_global_rank(pp_r, dp_r, tp_r, dp, tp)
                     for dp_r in range(dp)]
            if not _all_same_node(group, rank_to_node):
                dp_intra = False
                break
        if not dp_intra:
            break

    return TopologyInfo(
        tp_intra_node=tp_intra,
        pp_intra_node=pp_intra,
        dp_intra_node=dp_intra,
    )


def describe(node_gpus: List[int], pp: int, tp: int, dp: int) -> str:
    """
    Return a human-readable summary of the rank-to-node assignment and
    communication classification for the given plan.

    Useful for debugging and for the documentation file.
    """
    rank_to_node = _build_rank_to_node(node_gpus)
    topo = classify_comms(node_gpus, pp, tp, dp)
    world_size = sum(node_gpus)

    lines = []
    lines.append(f"node_gpus={node_gpus}  pp={pp}  tp={tp}  dp={dp}")
    lines.append("")

    # Rank assignment table
    lines.append(f"{'rank':>5}  {'pp_r':>5}  {'dp_r':>5}  {'tp_r':>5}  {'node':>5}")
    lines.append("-" * 35)
    for pp_r in range(pp):
        for dp_r in range(dp):
            for tp_r in range(tp):
                r = _global_rank(pp_r, dp_r, tp_r, dp, tp)
                lines.append(f"{r:>5}  {pp_r:>5}  {dp_r:>5}  {tp_r:>5}  {rank_to_node[r]:>5}")

    lines.append("")

    # TP groups
    lines.append("TP groups (same pp_r, dp_r — vary tp_r):")
    for pp_r in range(pp):
        for dp_r in range(dp):
            group = [_global_rank(pp_r, dp_r, tp_r, dp, tp) for tp_r in range(tp)]
            nodes = [rank_to_node[r] for r in group]
            same = "intra-node" if len(set(nodes)) == 1 else "CROSS-NODE"
            lines.append(f"  pp={pp_r} dp={dp_r}: ranks={group} nodes={nodes} → {same}")

    lines.append("")

    # PP pairs
    lines.append("PP stage boundaries (adjacent pp_ranks, same dp_r, tp_r):")
    for dp_r in range(dp):
        for tp_r in range(tp):
            for pp_r in range(pp - 1):
                src = _global_rank(pp_r,     dp_r, tp_r, dp, tp)
                dst = _global_rank(pp_r + 1, dp_r, tp_r, dp, tp)
                same = ("intra-node"
                        if rank_to_node[src] == rank_to_node[dst] else "CROSS-NODE")
                lines.append(f"  dp={dp_r} tp={tp_r}: rank {src}(node{rank_to_node[src]})"
                              f" → rank {dst}(node{rank_to_node[dst]}) {same}")

    lines.append("")

    # DP groups
    lines.append("DP groups (same pp_r, tp_r — vary dp_r):")
    for pp_r in range(pp):
        for tp_r in range(tp):
            group = [_global_rank(pp_r, dp_r, tp_r, dp, tp) for dp_r in range(dp)]
            nodes = [rank_to_node[r] for r in group]
            same = "intra-node" if len(set(nodes)) == 1 else "CROSS-NODE"
            lines.append(f"  pp={pp_r} tp={tp_r}: ranks={group} nodes={nodes} → {same}")

    lines.append("")
    lines.append(f"Summary:")
    lines.append(f"  tp_intra_node = {topo.tp_intra_node}")
    lines.append(f"  pp_intra_node = {topo.pp_intra_node}")
    lines.append(f"  dp_intra_node = {topo.dp_intra_node}")

    return "\n".join(lines)