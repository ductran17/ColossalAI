"""
Cluster profiler for the hybrid auto-planner.

Measures three things on the real cluster:
  1. α_intra, β_intra  — latency and inverse-bandwidth for intra-node P2P
  2. α_cross, β_cross  — same for cross-node P2P
  3. T_block           — actual GPU seconds for one transformer block forward+backward

All measurements are all-reduced so every rank ends up with identical values.
The result feeds directly into the cost model — no TFLOPS assumptions needed.

Design constraints:
  - dist.new_group() is a collective: ALL ranks must call it even if they don't
    participate in that particular pair's measurement.
  - We deliberately measure only ONE intra-node pair and ONE cross-node pair
    (not all C(8,2)=28) — this keeps profiling under ~3 seconds.
  - T_block is measured independently on every rank; we take the MAX across all
    ranks (slowest GPU determines the pipeline stage time).
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass
class ClusterProfile:
    """α/β/T_block values measured on the real cluster.

    All times are in seconds.
    α = latency (fixed overhead per message).
    β = inverse bandwidth (seconds per byte).
    T_block = forward+backward time for one transformer block on one GPU.
    min_free_memory_gb = minimum free GPU memory across all ranks (GB).
                         Used as the default memory budget for planning.
    """
    alpha_intra: float   # seconds  (e.g. 5e-6 for PCIe on same node)
    beta_intra:  float   # s/byte   (e.g. 1e-9 for ~1 GB/s PCIe)
    alpha_cross: float   # seconds  (e.g. 80e-6 for 100 GbE)
    beta_cross:  float   # s/byte   (e.g. 80e-9 for ~12.5 GB/s)
    T_block:     float   # seconds  (e.g. 1e-3 for one GPT2 block)
    min_free_memory_gb: float = 0.0   # GB, measured at profiling time

    def comm_time(self, nbytes: int, intra_node: bool) -> float:
        """Raw point-to-point send time for nbytes."""
        a = self.alpha_intra if intra_node else self.alpha_cross
        b = self.beta_intra  if intra_node else self.beta_cross
        return a + b * nbytes

    def allreduce_time(self, nbytes: int, n: int, intra_node: bool) -> float:
        """Ring-allreduce time: 2*(n-1)/n * (α + β*S).

        Args:
            nbytes: size of the tensor being all-reduced, in bytes.
            n: number of participants in the all-reduce group.
            intra_node: True if all participants are on the same node.
        """
        if n <= 1:
            return 0.0
        factor = 2.0 * (n - 1) / n
        return factor * self.comm_time(nbytes, intra_node)

    def p2p_time(self, nbytes: int, intra_node: bool) -> float:
        """One-directional P2P send time (pipeline stage boundary)."""
        return self.comm_time(nbytes, intra_node)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gather_node_layout(rank: int, world_size: int) -> List[List[int]]:
    """
    Returns a list of nodes, where each node is a list of global ranks that
    share the same physical machine.

    Strategy: each rank broadcasts its LOCAL_RANK and LOCAL_WORLD_SIZE.
    Node boundaries are detected where LOCAL_RANK resets to 0, which happens
    at the start of each new node in the global rank ordering.

    Example for a [2,4,2] cluster:
        [[0,1], [2,3,4,5], [6,7]]
    """
    local_rank       = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))

    # Pack into a tensor: [local_rank, local_world_size]
    info = torch.tensor([local_rank, local_world_size], dtype=torch.long,
                        device="cuda")
    gathered = [torch.zeros(2, dtype=torch.long, device="cuda")
                for _ in range(world_size)]
    dist.all_gather(gathered, info)

    # Reconstruct node → [global_ranks] mapping
    nodes: List[List[int]] = []
    current_node: List[int] = []
    for r, g in enumerate(gathered):
        lr  = g[0].item()
        lws = g[1].item()
        if lr == 0 and current_node:
            nodes.append(current_node)
            current_node = []
        current_node.append(r)
    if current_node:
        nodes.append(current_node)

    return nodes   # e.g. [[0,1], [2,3,4,5], [6,7]]


def _select_pairs(nodes: List[List[int]]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Choose one intra-node pair and one cross-node pair for profiling.

    Intra-node: first two ranks on the node with ≥2 GPUs.
    Cross-node:  rank 0 of node 0 and rank 0 of node 1.
    """
    # Intra-node: pick first node that has ≥2 ranks
    intra = None
    for node in nodes:
        if len(node) >= 2:
            intra = (node[0], node[1])
            break
    if intra is None:
        # Single-GPU nodes — treat intra == cross (won't be used with tp>1)
        intra = (nodes[0][0], nodes[1][0]) if len(nodes) > 1 else (0, 0)

    # Cross-node: first rank of node 0 → first rank of node 1
    if len(nodes) >= 2:
        cross = (nodes[0][0], nodes[1][0])
    else:
        cross = intra   # single-node cluster: cross = intra

    return intra, cross


def _measure_p2p(
    src: int,
    dst: int,
    pg: dist.ProcessGroup,
    rank: int,
    sizes_bytes: List[int],
    warmup: int = 10,
    repeat: int = 50,
) -> Tuple[float, float]:
    """
    Measure α and β for the src→dst link by sending tensors of different sizes.

    Improvements for stability:
      1. GPU-side timing via torch.cuda.Event (not CPU perf_counter).
      2. Median-of-repeats instead of mean (outlier-resistant).
      3. IQR-based clipping to remove OS jitter outliers.
      4. Larger warmup (10) and repeat (50) counts.
      5. torch.cuda.synchronize() before every iteration (not just batch).

    Only src and dst do real work; all other ranks skip the send/recv but
    the process group has already been created (collective) before this call.

    Returns (alpha_seconds, beta_seconds_per_byte).
    """
    per_size_medians = []

    for nbytes in sizes_bytes:
        # Round up to float32 element count
        n_elems = max(1, (nbytes + 3) // 4)
        tensor = torch.zeros(n_elems, dtype=torch.float32, device="cuda")

        # ── Warmup ──────────────────────────────────────────────────────
        for _ in range(warmup):
            if rank == src:
                dist.send(tensor, dst=dst, group=pg)
            elif rank == dst:
                dist.recv(tensor, src=src, group=pg)
            torch.cuda.synchronize()

        # ── Timed measurement (GPU events) ─────────────────────────────
        # Use cuda events for μS-accurate GPU-side timing.
        # Synchronize before EACH iteration to eliminate batching effects.
        iter_times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)

            start_evt.record()
            if rank == src:
                dist.send(tensor, dst=dst, group=pg)
            elif rank == dst:
                dist.recv(tensor, src=src, group=pg)
            end_evt.record()

            torch.cuda.synchronize()
            dt_ms = start_evt.elapsed_time(end_evt)  # GPU-side milliseconds
            iter_times.append(dt_ms / 1000.0)         # convert to seconds

        # ── Outlier rejection: IQR clip ────────────────────────────────
        # OS jitter / CPU scheduling can create extreme outliers.
        # Keep only points within 1.5× IQR of the median.
        t_arr = torch.tensor(iter_times, dtype=torch.float64)
        q1 = torch.quantile(t_arr, 0.25).item()
        q3 = torch.quantile(t_arr, 0.75).item()
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        clipped = t_arr[(t_arr >= lower) & (t_arr <= upper)]

        # If everything was clipped (rare), fall back to median of full set.
        median_time = clipped.median().item() if clipped.numel() > 0 else t_arr.median().item()
        per_size_medians.append(median_time)

    # ── Linear regression: T = α + β * S ─────────────────────────────
    # Least-squares on median-of-clipped-repeats for each size.
    s_arr = torch.tensor(sizes_bytes, dtype=torch.float64)
    t_arr = torch.tensor(per_size_medians, dtype=torch.float64)

    n = len(sizes_bytes)
    beta_num  = n * (s_arr * t_arr).sum() - s_arr.sum() * t_arr.sum()
    beta_den  = n * (s_arr * s_arr).sum() - s_arr.sum() ** 2
    beta  = (beta_num / beta_den).item() if beta_den.item() != 0 else 0.0
    alpha = (t_arr.mean() - beta * s_arr.mean()).item()

    # Clamp to physically plausible range
    alpha = max(alpha, 1e-8)
    beta  = max(beta,  1e-12)

    return alpha, beta


def _measure_T_block(
    batch: int,
    seq: int,
    hidden: int,
    heads: int,
    warmup: int = 10,
    repeat: int = 50,
) -> float:
    """
    Measure forward + backward time for ONE transformer block on this GPU.

    Improvements for stability:
      1. GPU-side timing via torch.cuda.Event.
      2. More warmup (10) and repeat (50) counts.
      3. IQR-based outlier clipping to remove thermal/clock jitter.
      4. Clear CUDA cache before measurement to reduce allocator noise.

    We build a minimal MLP + self-attention approximation inline so this
    module has no external dependencies and runs on any GPU in the cluster.

    Returns seconds (wall-clock GPU time, median of clipped repeats).
    """
    import math

    class _OneBlock(nn.Module):
        """Single transformer block: layer-norm + attention + MLP."""
        def __init__(self, h: int, a: int):
            super().__init__()
            self.ln1  = nn.LayerNorm(h)
            self.q    = nn.Linear(h, h, bias=False)
            self.k    = nn.Linear(h, h, bias=False)
            self.v    = nn.Linear(h, h, bias=False)
            self.out  = nn.Linear(h, h, bias=False)
            self.ln2  = nn.LayerNorm(h)
            self.fc1  = nn.Linear(h, 4 * h, bias=False)
            self.fc2  = nn.Linear(4 * h, h, bias=False)
            self.a    = a

        def forward(self, x):
            B, S, H = x.shape
            h = self.ln1(x)
            scale = math.sqrt(H // self.a)
            Q = self.q(h).reshape(B, S, self.a, -1).transpose(1, 2)
            K = self.k(h).reshape(B, S, self.a, -1).transpose(1, 2)
            V = self.v(h).reshape(B, S, self.a, -1).transpose(1, 2)
            att = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1) @ V
            att = att.transpose(1, 2).reshape(B, S, H)
            x = x + self.out(att)
            h = self.ln2(x)
            x = x + self.fc2(torch.relu(self.fc1(h)))
            return x

    device = torch.device("cuda")
    model  = _OneBlock(hidden, heads).to(device)
    opt    = torch.optim.SGD(model.parameters(), lr=1e-4)
    x      = torch.randn(batch, seq, hidden, device=device, requires_grad=True)

    # Clear allocator cache before measurement to reduce fragmentation noise.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()
        torch.cuda.synchronize()

    # Timed runs (individual events, not batched)
    iter_times_ms = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)

        start_evt.record()
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()
        end_evt.record()

        torch.cuda.synchronize()
        iter_times_ms.append(start_evt.elapsed_time(end_evt))

    # ── Outlier rejection: IQR clip ────────────────────────────────────
    # GPU clock fluctuations (thermal throttling, boost changes) create
    # occasional slow iterations.  Clip to 1.5× IQR around the median.
    t_arr = torch.tensor(iter_times_ms, dtype=torch.float64)
    q1 = torch.quantile(t_arr, 0.25).item()
    q3 = torch.quantile(t_arr, 0.75).item()
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    clipped = t_arr[(t_arr >= lower) & (t_arr <= upper)]

    median_ms = clipped.median().item() if clipped.numel() > 0 else t_arr.median().item()
    return median_ms / 1000.0   # seconds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_cluster(
    model_cfg: Optional[Dict] = None,
    warmup: int = 3,
    repeat: int = 10,
    sizes_bytes: Optional[List[int]] = None,
) -> ClusterProfile:
    """
    Measure α/β for intra-node and cross-node links, plus T_block.

    Must be called after dist.init_process_group().  All ranks call this
    together — it uses collective operations internally.

    Args:
        model_cfg: dict with keys 'batch', 'seq', 'hidden', 'heads'.
                   Used to make T_block representative of the real model.
                   Defaults to a small GPT2-like config.
        warmup:    warm-up iterations before timing.
        repeat:    timed iterations (median taken).
        sizes_bytes: tensor sizes used to fit the α/β line.
                     Defaults to a logarithmic sweep from 1 KB to 4 MB.

    Returns:
        ClusterProfile with all values synchronised across ranks.
    """
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    if model_cfg is None:
        model_cfg = {"batch": 2, "seq": 64, "hidden": 256, "heads": 4}

    if sizes_bytes is None:
        # Dense sweep with many small sizes for stable alpha (intercept) estimation.
        # Latency-dominated region (< 4 KB) anchors the y-intercept.
        # Bandwidth-dominated region (> 1 MB) anchors the slope.
        sizes_bytes = [
            256, 512, 1024, 2048, 4096,           # 5 pts < 4 KB (latency-dominated)
            8*1024, 16*1024, 32*1024, 64*1024,   # 4 pts 8–64 KB (transition)
            128*1024, 256*1024, 512*1024,        # 3 pts 128–512 KB
            1024*1024, 2*1024*1024, 4*1024*1024  # 3 pts 1–4 MB (bandwidth-dominated)
        ]

    # ------------------------------------------------------------------
    # Step 1: Gather node layout from LOCAL_RANK / LOCAL_WORLD_SIZE.
    # ------------------------------------------------------------------
    nodes = _gather_node_layout(rank, world_size)
    intra_pair, cross_pair = _select_pairs(nodes)

    # ------------------------------------------------------------------
    # Step 2: Create process groups (ALL ranks must call new_group).
    # ------------------------------------------------------------------
    # We need groups for exactly the two measured pairs.
    # Create both; non-members get a handle but never send/recv.
    intra_pg = dist.new_group(list(intra_pair))
    cross_pg = dist.new_group(list(cross_pair))

    # ------------------------------------------------------------------
    # Step 3: Measure intra-node α/β.
    # Only intra_pair[0] (src) and intra_pair[1] (dst) do real work.
    # ------------------------------------------------------------------
    if rank in intra_pair:
        src, dst = intra_pair
        alpha_intra, beta_intra = _measure_p2p(
            src, dst, intra_pg, rank, sizes_bytes, warmup, repeat
        )
    else:
        alpha_intra, beta_intra = 0.0, 0.0   # filled by all_reduce below

    # ------------------------------------------------------------------
    # Step 4: Measure cross-node α/β.
    # ------------------------------------------------------------------
    if intra_pair == cross_pair:
        # Single-node cluster: cross == intra
        alpha_cross, beta_cross = alpha_intra, beta_intra
    elif rank in cross_pair:
        src, dst = cross_pair
        alpha_cross, beta_cross = _measure_p2p(
            src, dst, cross_pg, rank, sizes_bytes, warmup, repeat
        )
    else:
        alpha_cross, beta_cross = 0.0, 0.0

    # ------------------------------------------------------------------
    # Step 5: Measure T_block on this GPU.
    # ------------------------------------------------------------------
    T_block_local = _measure_T_block(
        batch=model_cfg["batch"],
        seq=model_cfg["seq"],
        hidden=model_cfg["hidden"],
        heads=model_cfg["heads"],
        warmup=warmup,
        repeat=repeat,
    )

    # ------------------------------------------------------------------
    # Step 6: Measure free GPU memory on every rank.
    #
    # On shared clusters some GPUs may be partially in use by other jobs.
    # We take the MIN free memory across all GPUs as the conservative
    # budget for planning.
    # ------------------------------------------------------------------
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb_local = free_bytes / (1024 ** 3)
    free_tensor = torch.tensor([free_gb_local], dtype=torch.float32, device="cuda")
    dist.all_reduce(free_tensor, op=dist.ReduceOp.MIN)
    min_free_memory_gb = free_tensor[0].item()

    # ------------------------------------------------------------------
    # Step 7: All-reduce so every rank has the same profile values.
    #
    # α/β: non-measuring ranks have 0; MAX picks up the real measurement.
    # T_block: MAX across all ranks = slowest GPU sets the pace.
    # ------------------------------------------------------------------
    buf = torch.tensor(
        [alpha_intra, beta_intra, alpha_cross, beta_cross, T_block_local],
        dtype=torch.float64, device="cuda",
    )
    dist.all_reduce(buf, op=dist.ReduceOp.MAX)

    return ClusterProfile(
        alpha_intra = buf[0].item(),
        beta_intra  = buf[1].item(),
        alpha_cross = buf[2].item(),
        beta_cross  = buf[3].item(),
        T_block     = buf[4].item(),
        min_free_memory_gb = min_free_memory_gb,
    )
