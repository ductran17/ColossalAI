#!/usr/bin/env python3
"""
Raw alpha/beta profiler — records (size, time) pairs to CSV.
No regression here; notebook will fit alpha/beta later.

Usage:
    torchrun --nproc_per_node=2 profile_alpha_beta.py
    torchrun --nnodes=2 --nproc_per_node=2 --master_addr=10.10.10.18 profile_alpha_beta.py
"""

import os
import sys
import csv
import time
import argparse
from datetime import timedelta
import torch
import torch.distributed as dist
from pathlib import Path

def flog(msg):
    print(msg, flush=True)

def _get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--sizes", type=str,
        default="256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576")
    p.add_argument("--out", type=str, default="e1_alpha_beta_raw.csv")
    return p

def main():
    args = _get_parser().parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        # Select the CUDA device before creating the NCCL process group.  This
        # also makes NCCL barriers use the correct local device on every node.
        dist.init_process_group(
            "nccl",
            timeout=timedelta(minutes=3),
            device_id=torch.device("cuda", local_rank),
        )
    rank = dist.get_rank()
    ws = dist.get_world_size()

    sizes = [int(x.strip()) for x in args.sizes.split(",")]
    if any(size <= 0 for size in sizes):
        raise ValueError("All payload sizes must be positive")

    # Determine pairs from env
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    intra_pair = (0, 1) if local_ws >= 2 else (0, 0)
    cross_pair = (0, local_ws) if ws > local_ws else intra_pair

    flog(f"[Profiler] rank={rank} local_rank={local_rank} ws={ws} local_ws={local_ws}")
    if rank == 0:
        flog(f"[Profiler] intra={intra_pair} cross={cross_pair}")
        flog(f"[Profiler] {len(sizes)} sizes, warmup={args.warmup}, repeat={args.repeat}")

    records = []

    def phase_barrier(name):
        if rank == 0:
            flog(f"[Profiler] barrier: {name}")
        dist.barrier(device_ids=[local_rank])

    # ── INTRA measurement ──
    phase_barrier("before intra")
    if rank in intra_pair:
        src, dst = intra_pair
        peer = dst if rank == src else src
        for sz in sizes:
            if rank == 0:
                flog(f"[intra] size={sz} bytes")
            n = max(1, (sz + 3) // 4)
            t = torch.zeros(n, dtype=torch.float32, device="cuda")
            for _ in range(args.warmup):
                if rank == src: dist.send(t, dst=peer)
                else: dist.recv(t, src=peer)
            for it in range(args.repeat):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if rank == src: dist.send(t, dst=peer)
                else: dist.recv(t, src=peer)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                # Keep one observation per iteration.  The receiver performs
                # the matching operation but does not duplicate the sample.
                if rank == src:
                    records.append({"connection": "intra", "size_bytes": sz,
                                    "iteration": it, "time_s": t1 - t0})
    phase_barrier("after intra")

    # ── CROSS measurement ──
    phase_barrier("before inter")
    if cross_pair != intra_pair and rank in cross_pair:
        src, dst = cross_pair
        peer = dst if rank == src else src
        for sz in sizes:
            if rank == 0:
                flog(f"[cross] size={sz} bytes")
            n = max(1, (sz + 3) // 4)
            t = torch.zeros(n, dtype=torch.float32, device="cuda")
            for _ in range(args.warmup):
                if rank == src: dist.send(t, dst=peer)
                else: dist.recv(t, src=peer)
            for it in range(args.repeat):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if rank == src: dist.send(t, dst=peer)
                else: dist.recv(t, src=peer)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                if rank == src:
                    records.append({"connection": "inter", "size_bytes": sz,
                                    "iteration": it, "time_s": t1 - t0})

    phase_barrier("after inter")

    # Rank 0 collects via gather_object (avoids NCCL deadlock with file-per-rank fallback)
    all_records = [None] * ws
    dist.all_gather_object(all_records, records)

    if rank == 0:
        flat = [r for sub in all_records for r in sub]
        out = Path(args.out)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["connection", "size_bytes", "iteration", "time_s"])
            w.writeheader()
            w.writerows(flat)
        flog(f"[Profiler] Wrote {len(flat)} rows to {out}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
