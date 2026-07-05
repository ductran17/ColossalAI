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

    Generic architecture coefficients (optional):
    intermediate_size:    MLP hidden dimension (defaults to 4*hidden for GPT-2)
    num_key_value_heads:  Number of KV heads for GQA/MQA (defaults to heads for MHA)
    mlp_gated:            True if using SwiGLU (3 projections), False for standard MLP (2)
    """
    layers:      int
    hidden:      int
    heads:       int
    seq:         int
    batch:       int
    dtype_bytes: int = 2
    vocab_size:  int = 50257   # default GPT-2 vocab; override for custom models

    # Generic architecture support
    intermediate_size:     Optional[int] = None
    num_key_value_heads:   Optional[int] = None
    mlp_gated:             bool = False

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        return cls(
            layers              = d["layers"],
            hidden              = d["hidden"],
            heads               = d["heads"],
            seq                 = d["seq"],
            batch               = d["batch"],
            dtype_bytes         = d.get("dtype_bytes", 2),
            vocab_size          = d.get("vocab_size", 50257),
            intermediate_size   = d.get("intermediate_size"),
            num_key_value_heads = d.get("num_key_value_heads"),
            mlp_gated           = d.get("mlp_gated", False),
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
    T_dp_comm:      float   # DP gradient AllReduce (fully exposed)
    T_execution:    float   # framework + NCCL + dispatch overhead
    T_embedding:    float   # token embedding lookup + gradient (vocab-bound)
    T_lm_head:      float   # LM head projection + cross-entropy (vocab-bound)

    @property
    def total(self) -> float:
        return (
            self.T_compute + self.T_bubble + self.T_tp_comm
            + self.T_pp_comm + self.T_dp_comm + self.T_execution
            + self.T_embedding + self.T_lm_head
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
            f"  exec OH  = {ms(self.T_execution)}  ({pct(self.T_execution)})\n"
            f"  embed    = {ms(self.T_embedding)}  ({pct(self.T_embedding)})\n"
            f"  LM head  = {ms(self.T_lm_head)}  ({pct(self.T_lm_head)})"
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

    Generic formula:
      P_attn = (2 + 2r) * H^2    where r = num_key_value_heads / heads
      P_mlp  = g * e * H^2       where e = intermediate_size / hidden, g = 2 or 3
      P_layer = P_attn + P_mlp

    If architecture coefficients are not provided, falls back to GPT-2 defaults:
      r = 1.0 (MHA), e = 4.0 (standard expansion), g = 2 (standard MLP)
      → P_layer = 12 * H^2

    Normalization parameters (LayerNorm/RMSNorm) are O(H) and contribute
    < 0.04 % of total layer parameters; they are omitted for simplicity.
    """
    H = cfg.hidden

    # Architecture coefficients
    r = 1.0  # default MHA
    if cfg.num_key_value_heads is not None and cfg.heads > 0:
        r = cfg.num_key_value_heads / cfg.heads

    e = 4.0  # default GPT-2 expansion ratio
    if cfg.intermediate_size is not None and cfg.hidden > 0:
        e = cfg.intermediate_size / cfg.hidden

    g = 3 if cfg.mlp_gated else 2

    # Attention projections: Q + KV + output
    # Q: H × H, K/V: H × (rH), output: H × H
    P_attn = H * H + 2 * H * (r * H) + H * H

    # MLP projections
    P_mlp = g * e * H * H

    params = P_attn + P_mlp
    return int(params) * dtype_bytes





# ---------------------------------------------------------------------------
# Embedding + LM head cost (vocab-size dependent)
# ---------------------------------------------------------------------------

def _embedding_lm_head_time(
    cfg: ModelConfig,
    pp: int,
    tp: int,
    num_microbatches: int,
) -> tuple[float, float]:
    """
    Estimate time for token embedding and LM head operations.

    These are the only components that scale with vocab_size, and they are
    the dominant source of error when vocab is large (e.g. 50257 or 151936).

    Returns (T_embedding, T_lm_head) in seconds.
    """
    batch_per_mb = cfg.batch // num_microbatches
    dtype = cfg.dtype_bytes
    H = cfg.hidden
    V = cfg.vocab_size

    # Memory bandwidth for large tensor ops (measured from AdamW microbenchmark)
    bw_mem = 126e9   # 126 GB/s on L40 / A30 class GPUs

    # Compute bandwidth for dense matmul (fp32 on L40)
    bw_compute = 30e12  # 30 TFLOPS conservative

    # ── 1. Embedding table operations ────────────────────────────────
    # Forward: gather (batch*seq) tokens from (V, H) table → negligible
    # Backward: scatter-add gradient into (V, H) table → memory-bound
    T_embed = V * H * dtype / bw_mem

    # ── 2. LM head projection (forward + backward) ───────────────────
    # Matmul: (B*S, H) × (H, V/tp)  →  (B*S, V/tp)
    # FLOPs = 2*B*S*H*(V/tp) for forward, same for backward dL/dW
    # Total = 4*B*S*H*V/tp
    lm_head_flops = 4 * batch_per_mb * cfg.seq * H * V // max(tp, 1)
    T_lm_head_matmul = lm_head_flops / bw_compute

    # ── 3. Cross-entropy loss (forward + backward) ───────────────────
    # Materializes logits (B*S*V), probs (B*S*V), grad_logits (B*S*V)
    # Memory traffic ≈ 8 * B * S * V * dtype for fwd+bwd
    loss_bytes = 8 * batch_per_mb * cfg.seq * V * dtype
    T_lm_head_loss = loss_bytes / bw_mem

    # ── 4. CUDA allocator overhead for large logits tensor ───────────
    # Empirically measured: allocating 103 MB (GPT-2) to 311 MB (Qwen)
    # logits tensor per microbatch costs ~1.9e-11 s/byte.
    # This captures fragmentation and context-switching not modeled above.
    k_alloc = 1.95e-11  # calibrated on GPT-2 Medium pp=2,tp=2
    logits_bytes = batch_per_mb * cfg.seq * V * dtype
    T_allocator = k_alloc * logits_bytes * num_microbatches

    T_lm_head_total = T_lm_head_matmul + T_lm_head_loss + T_allocator

    # ── Exposure factor ──────────────────────────────────────────────
    # With pp=1 these costs overlap with transformer compute.
    # With pp>1, embedding (stage 0) and LM head (last stage) are exposed
    # at pipeline boundaries and cannot hide behind T_block.
    if pp > 1:
        exposure = 1.0
    elif tp > 1:
        exposure = 0.5   # TP sync partially exposes LM head
    else:
        exposure = 0.06  # almost fully hidden for single-GPU

    return T_embed * exposure, T_lm_head_total * exposure


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
      2. NCCL launch:        ~100 µs CPU setup per collective
      3. PP transitions:     P2P boundary setup
      4. Python dispatch:    per-block ShardFormer / execute_pipeline overhead

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

    # ── 2. NCCL collective launch overhead ──────────────────────────
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

    # ── 3. Pipeline stage transitions ──────────────────────────────
    # P2P send/recv setup at each microbatch boundary
    pp_transition_ms = getattr(profile, "pp_transition_ms", 0.5)
    T_pp_transition = 0.0
    if pp > 1:
        n_transitions = num_microbatches * (pp - 1)
        T_pp_transition = n_transitions * pp_transition_ms * 1e-3

    # ── 4. Python dispatch per block ────────────────────────────────
    #
    # IMPORTANT: Bare Python loop overhead (measured in debug_overhead_1)
    # is ~2 ms per microbatch, but this is HIDDEN by ColossalAI's 1F1B
    # pipeline overlap when pp > 1 (confirmed by debug_overhead_5 showing
    # negative overhead for execute_pipeline()).
    #
    # Therefore, we only count the TP-specific ShardFormer tensor
    # manipulation overhead, measured by comparing execute_pipeline(tp=1)
    # vs execute_pipeline(tp=2) with the same pp and M.
    #
    # For pp = 1 (no pipeline overlap), base dispatch is exposed.
    # For pp > 1, only tp-specific ShardFormer overhead remains.
    dispatch_base_ms = getattr(profile, "dispatch_base_ms", 0.15)
    dispatch_tp_ms = getattr(profile, "dispatch_tp_ms", 0.05)  # measured from tp=1 vs tp=2 diff
    if pp == 1:
        # No pipeline overlap: base + tp overhead both exposed
        t_dispatch = (dispatch_base_ms + dispatch_tp_ms * max(0, tp - 1)) * 1e-3
    else:
        # Pipeline overlap hides base Python dispatch; only TP manipulation remains
        t_dispatch = (dispatch_tp_ms * max(0, tp - 1)) * 1e-3
    n_blocks_critical_path = num_microbatches * layers_per_stage
    T_dispatch = n_blocks_critical_path * t_dispatch

    return T_adam + T_nccl + T_pp_transition + T_dispatch


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
    Fraction of DP AllReduce time that is EXPOSED on the critical path.

    Conservative assumption: NO overlap between gradient AllReduce and backward
    compute.  This is physically accurate for pipeline parallelism (ColossalAI
    runs manual sync after all microbatches), and yields a safe upper-bound
    for pure DDP.  It avoids arbitrary fitted constants (e.g. 0.7, 0.3) that
    lack first-principles justification.

    Returns 1.0 (fully exposed) for any dp > 1, and 0.0 for dp == 1.
    """
    return 1.0 if dp > 1 else 0.0


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
    # profile.T_block_with_microbatches is measured with M microbatch
    # activations resident in memory, capturing allocator fragmentation
    # and L2 cache pollution.
    #
    # Conditional selection:
    #   pp == 1: isolated T_block (no pipeline, microbatches processed
    #            sequentially, no simultaneous memory pressure)
    #   pp > 1:  representative T_block (1F1B pipeline keeps M microbatches
    #            in-flight, creating real memory pressure)
    # ------------------------------------------------------------------
    if pp == 1:
        t_block_eff = profile.T_block
    else:
        t_block_eff = getattr(profile, "T_block_with_microbatches", profile.T_block)
        if t_block_eff <= 0:
            t_block_eff = profile.T_block   # fallback for old profiles

    T_compute = layers_per_stage * (t_block_eff / tp) * num_microbatches

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
    # Term 5: T_dp_comm  (DP gradient AllReduce, fully exposed)
    #
    # After the backward pass each rank all-reduces gradients across dp peers.
    # The cost model conservatively assumes the ENTIRE AllReduce time is
    # serial on the critical path (overlap_factor = 1.0).  This is physically
    # accurate for pipeline parallelism (manual sync after all microbatches)
    # and yields a safe upper-bound for pure DDP.
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
    # Term 6: T_execution  (formula-based framework overhead)
    #
    # Four physically based terms: AdamW (memory BW), NCCL launch,
    # pipeline transitions, Python dispatch.  Gradient accumulation overhead
    # is absorbed into the representative T_block^repr measurement.
    # Coefficients are GPU-specific and measured via microbenchmarks.
    # ------------------------------------------------------------------
    T_execution = _execution_overhead(
        cfg, pp, tp, dp, profile, num_microbatches
    )

    # ------------------------------------------------------------------
    # Term 7+8: T_embedding + T_lm_head  (vocab-size dependent)
    #
    # These were missing in the original 6-term model and are the dominant
    # source of error for large-vocab models (GPT-2 50257, Qwen 151936).
    # See _embedding_lm_head_time() for the physical derivation.
    # ------------------------------------------------------------------
    T_embedding, T_lm_head = _embedding_lm_head_time(
        cfg, pp, tp, num_microbatches
    )

    return CostBreakdown(
        T_compute   = T_compute,
        T_bubble    = T_bubble,
        T_tp_comm   = T_tp_comm,
        T_pp_comm   = T_pp_comm,
        T_dp_comm   = T_dp_comm,
        T_execution = T_execution,
        T_embedding = T_embedding,
        T_lm_head   = T_lm_head,
    )