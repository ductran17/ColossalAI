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
    """
    alpha_intra: float   # seconds  (e.g. 5e-6 for PCIe on same node)
    beta_intra:  float   # s/byte   (e.g. 1e-9 for ~1 GB/s PCIe)
    alpha_cross: float   # seconds  (e.g. 80e-6 for 100 GbE)
    beta_cross:  float   # s/byte   (e.g. 80e-9 for ~12.5 GB/s)
    T_block:     float   # seconds  (e.g. 1e-3 for one GPT2 block)

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

def _gather_node_layout(rank: int, world_size: int) -> List[int]:
    """
    Returns a list of length world_size where entry[r] is the LOCAL_RANK of
    global rank r.  From this we reconstruct which ranks share a node.

    Strategy: each rank broadcasts its LOCAL_RANK.  Ranks with the same
    'node_id' (consecutive block in global rank space with the same
    LOCAL_WORLD_SIZE) share a node.

    We actually collect (local_rank, local_world_size) per global rank and
    reconstruct node boundaries from the LOCAL_WORLD_SIZE resets to 0.
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
    warmup: int = 3,
    repeat: int = 10,
) -> Tuple[float, float]:
    """
    Measure α and β for the src→dst link by sending tensors of different sizes.

    Only src and dst do real work; all other ranks skip the send/recv but
    the process group has already been created (collective) before this call.

    Returns (alpha_seconds, beta_seconds_per_byte).
    """
    times = []

    for nbytes in sizes_bytes:
        # Round up to float32 element count
        n_elems = max(1, (nbytes + 3) // 4)
        tensor = torch.zeros(n_elems, dtype=torch.float32, device="cuda")

        # Warmup
        for _ in range(warmup):
            if rank == src:
                dist.send(tensor, dst=dst, group=pg)
            elif rank == dst:
                dist.recv(tensor, src=src, group=pg)
            torch.cuda.synchronize()

        # Timed measurement
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            if rank == src:
                dist.send(tensor, dst=dst, group=pg)
            elif rank == dst:
                dist.recv(tensor, src=src, group=pg)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0) / repeat)

    # Linear regression: T = α + β * S
    # Use least-squares fit over the measured (size, time) pairs.
    s_arr = torch.tensor(sizes_bytes, dtype=torch.float64)
    t_arr = torch.tensor(times,       dtype=torch.float64)

    # β = (n*ΣST - ΣS*ΣT) / (n*ΣS² - (ΣS)²)
    n = len(sizes_bytes)
    beta_num  = n * (s_arr * t_arr).sum() - s_arr.sum() * t_arr.sum()
    beta_den  = n * (s_arr * s_arr).sum() - s_arr.sum() ** 2
    beta  = (beta_num / beta_den).item()  if beta_den.item() != 0 else 0.0
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
    warmup: int = 3,
    repeat: int = 10,
) -> float:
    """
    Measure forward + backward time for ONE transformer block on this GPU.

    We build a minimal MLP + self-attention approximation inline so this
    module has no external dependencies and runs on any GPU in the cluster.

    Returns seconds (wall-clock GPU time, median over `repeat` runs).
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

    start_evt = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_evt   = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    # Warmup
    for _ in range(warmup):
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()
        torch.cuda.synchronize()

    # Timed runs
    for i in range(repeat):
        start_evt[i].record()
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()
        end_evt[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_evt, end_evt)]
    return sorted(times_ms)[len(times_ms) // 2] / 1000.0   # median, in seconds


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
        # Logarithmic sweep: 1 KB, 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB
        sizes_bytes = [1024 * (4 ** i) for i in range(7)]

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
    # Step 6: All-reduce so every rank has the same values.
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
    )
