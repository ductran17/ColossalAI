"""
Cost model for the hybrid auto-planner.

Estimates the wall-clock time of one full training step for a given
(pp, tp, dp) plan, cluster profile, and model configuration.

Four additive terms:
  T_total = T_compute + T_bubble + T_tp_comm + T_pp_comm + T_dp_comm

No FLOPs/GPU-spec assumptions — T_block is measured directly by profiler.py
and every other term is derived from that single measurement.

Design constraints:
  - Pure Python, no torch.distributed — unit-testable on a laptop.
  - Uses ClusterProfile and TopologyInfo from profiler.py / topology.py.
  - All sizes are in bytes; all times are in seconds.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from .profiler import ClusterProfile
from .topology import TopologyInfo


# ---------------------------------------------------------------------------
# Public inputs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    Describes the transformer model being trained.

    layers:      total number of transformer blocks (e.g. 12 for GPT2-small)
    hidden:      hidden dimension H (e.g. 768)
    heads:       number of attention heads (e.g. 12)
    seq:         sequence length (e.g. 1024)
    batch:       global batch size (number of sequences)
    dtype_bytes: bytes per element (2 for fp16/bf16, 4 for fp32). Default 2.
    """
    layers:      int
    hidden:      int
    heads:       int
    seq:         int
    batch:       int
    dtype_bytes: int = 2

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(
            layers      = d["layers"],
            hidden      = d["hidden"],
            heads       = d["heads"],
            seq         = d["seq"],
            batch       = d["batch"],
            dtype_bytes = d.get("dtype_bytes", 2),
        )


@dataclass
class CostBreakdown:
    """
    Detailed breakdown of the estimated step time.
    Useful for debugging and for explaining why a plan scores well or poorly.
    """
    T_compute:  float   # pure GPU compute (pp stages × T_block each)
    T_bubble:   float   # 1F1B pipeline bubble overhead
    T_tp_comm:  float   # TP AllReduce per layer
    T_pp_comm:  float   # PP P2P send/recv at stage boundaries
    T_dp_comm:  float   # DP gradient AllReduce (after overlap discount)

    @property
    def total(self) -> float:
        return self.T_compute + self.T_bubble + self.T_tp_comm + self.T_pp_comm + self.T_dp_comm

    def __str__(self) -> str:
        ms = lambda s: f"{s*1000:.3f} ms"
        pct = lambda t: f"{100*t/self.total:.1f}%"
        return (
            f"T_total    = {ms(self.total)}\n"
            f"  compute  = {ms(self.T_compute)}  ({pct(self.T_compute)})\n"
            f"  bubble   = {ms(self.T_bubble)}  ({pct(self.T_bubble)})\n"
            f"  TP comm  = {ms(self.T_tp_comm)}  ({pct(self.T_tp_comm)})\n"
            f"  PP comm  = {ms(self.T_pp_comm)}  ({pct(self.T_pp_comm)})\n"
            f"  DP comm  = {ms(self.T_dp_comm)}  ({pct(self.T_dp_comm)})"
        )


# ---------------------------------------------------------------------------
# Size helpers
# ---------------------------------------------------------------------------

def _activation_bytes(cfg: ModelConfig, batch: int, dtype_bytes: int) -> int:
    """
    Bytes in one activation tensor crossing a PP stage boundary or entering
    a TP AllReduce.

    Shape: (batch, seq, hidden)  →  batch × seq × hidden × dtype_bytes.

    We use the per-microbatch batch here; the caller divides global batch
    by num_microbatches before calling.
    """
    return batch * cfg.seq * cfg.hidden * dtype_bytes


def _param_bytes_per_layer(cfg: ModelConfig, dtype_bytes: int) -> int:
    """
    Parameter count for one full transformer block (attention + MLP),
    expressed in bytes.

    Standard GPT2-style block:
      Attention Q/K/V projections:  3 × H × H
      Attention output projection:  H × H
      MLP fc1:                      H × (4H)
      MLP fc2:                      (4H) × H
      Two LayerNorm (2 × 2H params, negligible but included)
    Total params per block ≈ 4H² + 8H² + 4H = 12H² + 4H ≈ 12H²
    """
    H = cfg.hidden
    params = (
        3 * H * H    # Q, K, V
        + H * H      # attention output
        + H * 4 * H  # MLP fc1
        + 4 * H * H  # MLP fc2
        + 4 * H      # two LayerNorm (gamma+beta each)
    )
    return params * dtype_bytes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_step_time(
    cfg:              ModelConfig,
    pp:               int,
    tp:               int,
    dp:               int,
    profile:          ClusterProfile,
    topology:         TopologyInfo,
    num_microbatches: int = 4,
) -> CostBreakdown:
    """
    Estimate wall-clock time of one training step.

    Args:
        cfg:              model configuration (layers, hidden, heads, seq, batch).
        pp:               pipeline-parallel degree.
        tp:               tensor-parallel degree.
        dp:               data-parallel degree.
        profile:          ClusterProfile from profiler.profile_cluster().
        topology:         TopologyInfo from topology.classify_comms().
        num_microbatches: number of microbatches per step (for pipeline scheduling).

    Returns:
        CostBreakdown with all five terms and a .total property.
    """
    if pp * tp * dp == 0:
        raise ValueError("pp, tp, dp must all be >= 1")
    if cfg.batch % num_microbatches != 0:
        raise ValueError(
            f"batch={cfg.batch} must be divisible by num_microbatches={num_microbatches}"
        )
    if cfg.layers % pp != 0:
        raise ValueError(
            f"layers={cfg.layers} must be divisible by pp={pp}"
        )

    dtype = cfg.dtype_bytes
    microbatch_size = cfg.batch // num_microbatches   # sequences per microbatch
    layers_per_stage = cfg.layers // pp

    # ------------------------------------------------------------------
    # Term 1: T_compute
    #
    # profile.T_block is the measured forward+backward time for one
    # transformer block on the slowest GPU in the cluster.
    #
    # With pipeline parallelism each rank processes `layers_per_stage`
    # blocks.  With tensor parallelism the block is split across tp GPUs —
    # the measured T_block already reflects the tp=1 baseline; when tp>1
    # each GPU does only 1/tp of the work per layer, so we divide by tp.
    #
    # With pipeline parallelism each rank processes all `num_microbatches`
    # microbatches sequentially; multiply by num_microbatches.
    # ------------------------------------------------------------------
    T_compute = layers_per_stage * (profile.T_block / tp) * num_microbatches

    # ------------------------------------------------------------------
    # Term 2: T_bubble
    #
    # 1F1B pipeline schedule leaves (pp-1) stages idle at the start and end
    # of each step.  Each bubble slot = time to process one stage = one
    # microbatch × layers_per_stage.
    #
    # Bubble fraction = (pp - 1) / num_microbatches
    # T_bubble = bubble_fraction × T_compute
    # ------------------------------------------------------------------
    if pp == 1:
        T_bubble = 0.0
    else:
        bubble_fraction = (pp - 1) / num_microbatches
        T_bubble = bubble_fraction * T_compute

    # ------------------------------------------------------------------
    # Term 3: T_tp_comm  (TP AllReduce per layer)
    #
    # Every transformer block has 2 AllReduce calls in the FORWARD pass when
    # using column+row tensor parallelism (Megatron-LM style):
    #   - one after the attention output projection (row-parallel c_proj)
    #   - one after the MLP second linear (row-parallel mlp.c_proj)
    #
    # NOTE: Backward pass also triggers 2 additional AllReduces for the
    # column-parallel layers (c_attn, mlp.c_fc) via
    # LinearWithAsyncCommunication.async_grad_allreduce. However, these are
    # launched with async_op=True and overlap with the weight-gradient matmul,
    # so their exposed latency on the critical path is effectively zero.
    # We therefore model only the forward-exposed 2 AllReduces.
    #
    # The tensor being all-reduced is the activation: (microbatch, seq, hidden).
    # We sum over all microbatches and all layers on this stage.
    #
    # Ring AllReduce cost for n participants, S bytes:
    #   T = 2 × (n-1)/n × (α + β × S)
    #
    # When tp == 1 there is no AllReduce (ClusterProfile.allreduce_time
    # returns 0.0 for n <= 1).
    # ------------------------------------------------------------------
    act_bytes = _activation_bytes(cfg, microbatch_size, dtype)
    allreduce_per_layer = 2 * profile.allreduce_time(
        act_bytes, tp, intra_node=topology.tp_intra_node
    )
    T_tp_comm = layers_per_stage * allreduce_per_layer * num_microbatches

    # ------------------------------------------------------------------
    # Term 4: T_pp_comm  (PP P2P send/recv at stage boundaries)
    #
    # At each microbatch boundary, one activation tensor is sent from one
    # pipeline stage to the next.  In a 1F1B schedule this is largely
    # pipelined with compute, so we count only the latency term that cannot
    # be hidden: α (one-way latency of the link).
    #
    # In practice the activation send and the next microbatch's compute
    # overlap well, but the latency α is always serialised.
    # We model:
    #   T_pp_comm = num_microbatches × p2p_time(act_bytes)
    #
    # This is conservative (overestimates by including β × act_bytes even
    # though bandwidth is mostly hidden).  The relative ordering of
    # candidates is preserved, which is all the planner needs.
    # ------------------------------------------------------------------
    if pp == 1:
        T_pp_comm = 0.0
    else:
        T_pp_comm = num_microbatches * profile.p2p_time(
            act_bytes, intra_node=topology.pp_intra_node
        )

    # ------------------------------------------------------------------
    # Term 5: T_dp_comm  (DP gradient AllReduce, partially overlapped)
    #
    # After the backward pass each rank all-reduces gradients across dp peers.
    # Modern frameworks (DDP, ZeRO) bucket and overlap this with the tail
    # of the backward pass.  Empirically 60-70 % of the cost is hidden.
    # We apply a 0.3 overlap factor (effective cost = 30 % of raw cost).
    #
    # Gradient size per GPU = one layer's parameters × layers_per_stage.
    # With tensor parallelism each GPU holds only 1/tp of each layer's params.
    # ------------------------------------------------------------------
    param_bytes = _param_bytes_per_layer(cfg, dtype)
    total_grad_bytes = param_bytes * layers_per_stage // tp
    overlap_factor = 0.3
    T_dp_comm = overlap_factor * profile.allreduce_time(
        total_grad_bytes, dp, intra_node=topology.dp_intra_node
    )

    return CostBreakdown(
        T_compute = T_compute,
        T_bubble  = T_bubble,
        T_tp_comm = T_tp_comm,
        T_pp_comm = T_pp_comm,
        T_dp_comm = T_dp_comm,
    )