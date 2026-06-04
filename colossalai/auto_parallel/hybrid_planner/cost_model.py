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
    vocab_size:  int = 50257   # default GPT-2 vocab; override for custom models

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(
            layers      = d["layers"],
            hidden      = d["hidden"],
            heads       = d["heads"],
            seq         = d["seq"],
            batch       = d["batch"],
            dtype_bytes = d.get("dtype_bytes", 2),
            vocab_size  = d.get("vocab_size", 50257),
        )


@dataclass
class CostBreakdown:
    """
    Detailed breakdown of the estimated step time.
    Useful for debugging and for explaining why a plan scores well or poorly.
    """
    T_compute:      float   # pure GPU compute (pp stages × T_block each)
    T_bubble:       float   # 1F1B pipeline bubble overhead
    T_tp_comm:      float   # TP AllReduce per layer
    T_pp_comm:      float   # PP P2P send/recv at stage boundaries
    T_dp_comm:      float   # DP gradient AllReduce (after overlap discount)
    T_step_overhead: float  # embedding + LM head + loss + optimizer (once per step)
    T_execution:    float   # framework + NCCL + dispatch overhead (formula-based)

    @property
    def total(self) -> float:
        return (
            self.T_compute + self.T_bubble + self.T_tp_comm
            + self.T_pp_comm + self.T_dp_comm + self.T_step_overhead
            + self.T_execution
        )

    def __str__(self) -> str:
        ms = lambda s: f"{s*1000:.3f} ms"
        pct = lambda t: f"{100*t/self.total:.1f}%"
        return (
            f"T_total    = {ms(self.total)}\n"
            f"  compute  = {ms(self.T_compute)}  ({pct(self.T_compute)})\n"
            f"  bubble   = {ms(self.T_bubble)}  ({pct(self.T_bubble)})\n"
            f"  TP comm  = {ms(self.T_tp_comm)}  ({pct(self.T_tp_comm)})\n"
            f"  PP comm  = {ms(self.T_pp_comm)}  ({pct(self.T_pp_comm)})\n"
            f"  DP comm  = {ms(self.T_dp_comm)}  ({pct(self.T_dp_comm)})\n"
            f"  step OH  = {ms(self.T_step_overhead)}  ({pct(self.T_step_overhead)})\n"
            f"  exec OH  = {ms(self.T_execution)}  ({pct(self.T_execution)})"
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
# Step-overhead helpers (embedding, LM head, loss, optimizer)
# ---------------------------------------------------------------------------

def _embedding_time(cfg: ModelConfig, profile: ClusterProfile) -> float:
    """
    Token embedding lookup time.
    Memory-bound: reads vocab_size × hidden parameters.
    Scaled proportionally to T_block.
    """
    bytes_read = cfg.vocab_size * cfg.hidden * cfg.dtype_bytes
    bytes_per_block = 12 * cfg.hidden ** 2 * cfg.dtype_bytes
    ratio = bytes_read / bytes_per_block
    return profile.T_block * ratio * 0.5   # 0.5: memory-bound vs compute-bound


def _lm_head_time(cfg: ModelConfig, profile: ClusterProfile) -> float:
    """
    LM head projection: linear layer (hidden → vocab_size).
    Forward + backward, scaled proportionally to T_block.
    """
    lm_head_flops = 2 * cfg.batch * cfg.seq * cfg.hidden * cfg.vocab_size
    block_flops = 12 * cfg.hidden ** 2 * cfg.batch * cfg.seq
    fwd_ratio = lm_head_flops / block_flops
    total_ratio = fwd_ratio * 3            # backward ≈ 2× forward for linear
    return profile.T_block * total_ratio


def _loss_time(cfg: ModelConfig, profile: ClusterProfile) -> float:
    """
    Cross-entropy loss: softmax over vocab + gather correct index.
    Forward + backward, scaled proportionally to T_block.
    """
    loss_flops = 3 * cfg.batch * cfg.seq * cfg.vocab_size
    block_flops = 12 * cfg.hidden ** 2 * cfg.batch * cfg.seq
    ratio = loss_flops / block_flops
    return profile.T_block * max(ratio, 0.005)


def _optimizer_time(cfg: ModelConfig, profile: ClusterProfile) -> float:
    """
    AdamW optimizer step.
    Memory-bandwidth bound: reads/writes param, grad, momentum, variance.
    Measured on L40: ~38 ms for 302M params → effective BW ~126 GB/s.
    Formula: 4 tensors × local_params × dtype / effective_bw
    """
    # effective_bw_adam is GPU-specific; default 126e9 from L40 measurement
    bw = getattr(profile, "effective_bw_adam", 126e9)
    # With PP, each GPU only optimizes its local layers
    # With TP, each GPU holds 1/tp of each layer
    local_params = (cfg.layers * 12 * cfg.hidden ** 2) / (1 if cfg.layers == 0 else 1)
    # Actually local params depend on pp and tp; caller should adjust
    total_params = cfg.layers * 12 * cfg.hidden ** 2
    adam_bytes = 4 * total_params * cfg.dtype_bytes  # param, grad, m, v
    return adam_bytes / bw


# ---------------------------------------------------------------------------
# Execution overhead formula (NEW — physically based, not empirical constants)
# ---------------------------------------------------------------------------

def _execution_overhead(
    cfg: ModelConfig,
    pp: int,
    tp: int,
    dp: int,
    profile: ClusterProfile,
    num_microbatches: int,
) -> float:
    """
    Formula-based execution overhead from ColossalAI framework, NCCL, and dispatch.

    Sources (all physically measurable, no fitted fudge factors):
      1. AdamW step:        memory-bandwidth bound (4 tensors × local_params)
      2. Gradient accum:    read+write grad buffer each microbatch
      3. NCCL launch:        ~100 µs CPU setup per collective
      4. PP transitions:     P2P boundary setup
      5. Python dispatch:    per-block ShardFormer / execute_pipeline overhead

    Coefficients are GPU-specific and measured via microbenchmarks
    (see debug_overhead_*.py scripts).  Defaults below are from L40 cluster.
    """
    layers_per_stage = cfg.layers // pp
    param_bytes = _param_bytes_per_layer(cfg, cfg.dtype_bytes)
    local_params = param_bytes * layers_per_stage // tp

    # ── 1. AdamW optimizer step ───────────────────────────────────────
    # Adam reads/writes: param, grad, momentum, variance = 4 tensors
    # local_param_bytes already includes dtype_bytes; no need to multiply again
    bw_adam = getattr(profile, "effective_bw_adam", 126e9)
    T_adam = (4 * local_params) / bw_adam

    # ── 2. Gradient accumulation traffic ──────────────────────────────
    # Each backward pass accumulates into grad buffer: read old + write new
    # Effective BW is ~50% of peak because it overlaps with compute
    bw_grad = getattr(profile, "effective_bw_grad_acc", 150e9)
    grad_acc_bytes = num_microbatches * layers_per_stage * param_bytes * 2
    T_grad_acc = grad_acc_bytes / bw_grad

    # ── 3. NCCL collective launch overhead ──────────────────────────
    # Each AllReduce requires ~100 µs of CPU enqueue + GPU kernel launch
    nccl_launch_us = getattr(profile, "nccl_launch_us", 100.0)
    nccl_launch_s = nccl_launch_us * 1e-6

    T_nccl = 0.0
    if tp > 1:
        # 2 AllReduces per layer (forward + backward column-parallel)
        n_tp_collectives = 2 * layers_per_stage * num_microbatches
        T_nccl += n_tp_collectives * nccl_launch_s
    if dp > 1:
        # 1 AllReduce per step (gradient sync after all microbatches)
        T_nccl += 1 * nccl_launch_s

    # ── 4. Pipeline stage transitions ──────────────────────────────
    # P2P send/recv setup at each microbatch boundary
    pp_transition_ms = getattr(profile, "pp_transition_ms", 0.5)
    T_pp_transition = 0.0
    if pp > 1:
        n_transitions = num_microbatches * (pp - 1)
        T_pp_transition = n_transitions * pp_transition_ms * 1e-3

    # ── 5. Python dispatch per block ────────────────────────────────
    # ColossalAI execute_pipeline + ShardFormer dispatch per transformer block
    # Base: ~0.15 ms/block.  TP adds ~0.20 ms/block for tensor manipulation.
    dispatch_base_ms = getattr(profile, "dispatch_base_ms", 0.15)
    dispatch_tp_ms = getattr(profile, "dispatch_tp_ms", 0.20)
    t_dispatch = (dispatch_base_ms + dispatch_tp_ms * max(0, tp - 1)) * 1e-3
    n_blocks_critical_path = num_microbatches * layers_per_stage
    T_dispatch = n_blocks_critical_path * t_dispatch

    return T_adam + T_grad_acc + T_nccl + T_pp_transition + T_dispatch


# ---------------------------------------------------------------------------
# DP overlap helper
# ---------------------------------------------------------------------------

def _dp_overlap_factor(
    T_compute: float,
    total_grad_bytes: int,
    dp: int,
    profile: ClusterProfile,
    topology: TopologyInfo,
) -> float:
    """
    Fraction of DP AllReduce time that is EXPOSED (not hidden by backward).

    Physics:
      - DDP launches async AllReduce during backward
      - Only AllReduce finishing BEFORE backward ends is hidden
      - On slow networks (Ethernet), most AllReduce is exposed
      - On fast networks (NVLink/PCIe), most is hidden

    Formula:
      overlap_factor = 1.0 - min(1, T_compute / T_allreduce_raw) * ddp_efficiency

    where ddp_efficiency = 0.7 for intra-node, 0.6 for cross-node.
    """
    if dp <= 1:
        return 0.0

    raw_allreduce = profile.allreduce_time(
        total_grad_bytes, dp, intra_node=topology.dp_intra_node
    )
    if raw_allreduce <= 0:
        return 0.0

    max_hidden_fraction = min(1.0, T_compute / raw_allreduce)

    if topology.dp_intra_node:
        ddp_efficiency = 0.7   # fast PCIe/NVLink
    else:
        ddp_efficiency = 0.3   # commodity Ethernet: ~30% of theoretical overlap achieved

    effective_hidden = max_hidden_fraction * ddp_efficiency
    overlap_factor = 1.0 - effective_hidden
    return max(0.2, min(0.9, overlap_factor))


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
        # Exact 1F1B bubble from Narayanan et al. (2021).
        # Accounts for both pipeline fill and drain phases.
        bubble_fraction = (pp - 1) / (num_microbatches + pp - 1)
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
    # of the backward pass.  The overlap factor is not constant: on fast
    # intra-node links most of the AllReduce is hidden, while on slow
    # cross-node Ethernet most of it is exposed.
    #
    # Gradient size per GPU = one layer's parameters × layers_per_stage.
    # With tensor parallelism each GPU holds only 1/tp of each layer's params.
    # ------------------------------------------------------------------
    param_bytes = _param_bytes_per_layer(cfg, dtype)
    total_grad_bytes = param_bytes * layers_per_stage // tp
    overlap_factor = _dp_overlap_factor(
        T_compute, total_grad_bytes, dp, profile, topology
    )
    T_dp_comm = overlap_factor * profile.allreduce_time(
        total_grad_bytes, dp, intra_node=topology.dp_intra_node
    )

    # ------------------------------------------------------------------
    # Term 6: T_step_overhead  (embedding + LM head + loss + optimizer)
    #
    # The profiler measures an isolated transformer block, but a real step
    # also includes token embedding, LM head projection, cross-entropy loss,
    # and the Adam optimizer update.  These run once per step (not per
    # microbatch) and are therefore added as a serial overhead.
    # ------------------------------------------------------------------
    T_step_overhead = (
        _embedding_time(cfg, profile)
        + _lm_head_time(cfg, profile)
        + _loss_time(cfg, profile)
        # NOTE: _optimizer_time is FLOP-scaled and gives ~1 ms (wrong).
        # The real Adam cost is captured in T_execution below.
    )

    # ------------------------------------------------------------------
    # Term 7: T_execution  (formula-based framework overhead)
    #
    # Five physically based terms: AdamW (memory BW), gradient accumulation,
    # NCCL launch, pipeline transitions, Python dispatch.
    # Coefficients are GPU-specific and measured via microbenchmarks.
    # ------------------------------------------------------------------
    T_execution = _execution_overhead(
        cfg, pp, tp, dp, profile, num_microbatches
    )

    return CostBreakdown(
        T_compute       = T_compute,
        T_bubble        = T_bubble,
        T_tp_comm       = T_tp_comm,
        T_pp_comm       = T_pp_comm,
        T_dp_comm       = T_dp_comm,
        T_step_overhead = T_step_overhead,
        T_execution     = T_execution,
    )