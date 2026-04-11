"""
Test the cluster profiler on the real 3-node cluster.

Run via launch_3nodes.sh:
    bash launch_3nodes.sh --profile-test

Expected output (example values for 100 GbE Ethernet cluster):
    [rank 0] Node layout: [[0,1], [2,3,4,5], [6,7]]
    [rank 0] Intra-node pair: (0, 1)
    [rank 0] Cross-node pair: (0, 2)
    [rank 0] α_intra = 5.2 µs   β_intra = 1.1 ns/B   BW_intra = 909 MB/s
    [rank 0] α_cross = 82.3 µs  β_cross = 79.5 ns/B  BW_cross = 12.6 GB/s
    [rank 0] T_block  = 1.23 ms  (batch=2 seq=64 hidden=256)
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.distributed as dist
import colossalai
from colossalai.auto_parallel.hybrid_planner.profiler import (
    profile_cluster, _gather_node_layout, _select_pairs
)


def main():
    colossalai.launch_from_torch()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Show node layout so we can verify the pair selection is correct.
    nodes = _gather_node_layout(rank, world_size)
    intra_pair, cross_pair = _select_pairs(nodes)

    if rank == 0:
        print(f"\n[rank 0] World size : {world_size}")
        print(f"[rank 0] Node layout: {nodes}")
        print(f"[rank 0] Intra pair : {intra_pair}")
        print(f"[rank 0] Cross pair : {cross_pair}")
        print(f"[rank 0] Profiling ... (warmup=3, repeat=10, 7 sizes)\n")
        sys.stdout.flush()

    model_cfg = {"batch": 2, "seq": 64, "hidden": 256, "heads": 4}

    profile = profile_cluster(model_cfg=model_cfg, warmup=3, repeat=10)

    if rank == 0:
        bw_intra = 1.0 / profile.beta_intra / 1e9   # GB/s
        bw_cross = 1.0 / profile.beta_cross / 1e9   # GB/s

        print(f"[rank 0] ─── Cluster Profile ───────────────────────────────")
        print(f"[rank 0] α_intra  = {profile.alpha_intra*1e6:.2f} µs")
        print(f"[rank 0] β_intra  = {profile.beta_intra*1e9:.2f} ns/byte  "
              f"→  BW = {bw_intra:.2f} GB/s")
        print(f"[rank 0] α_cross  = {profile.alpha_cross*1e6:.2f} µs")
        print(f"[rank 0] β_cross  = {profile.beta_cross*1e9:.2f} ns/byte  "
              f"→  BW = {bw_cross:.2f} GB/s")
        print(f"[rank 0] T_block  = {profile.T_block*1e3:.3f} ms  "
              f"(batch={model_cfg['batch']} seq={model_cfg['seq']} "
              f"hidden={model_cfg['hidden']})")
        print(f"[rank 0] ─────────────────────────────────────────────────────")

        # Sanity checks
        ok = True
        if profile.alpha_intra > profile.alpha_cross * 2:
            print("[rank 0] WARNING: α_intra > α_cross — intra measurement may be wrong")
            ok = False
        if profile.beta_intra > profile.beta_cross:
            print("[rank 0] WARNING: β_intra > β_cross — intra link is slower than cross-node?")
            ok = False
        if profile.T_block <= 0:
            print("[rank 0] ERROR: T_block not measured correctly")
            ok = False
        if ok:
            print("[rank 0] Sanity checks PASSED")
        print()
        sys.stdout.flush()

    dist.barrier()


if __name__ == "__main__":
    main()
