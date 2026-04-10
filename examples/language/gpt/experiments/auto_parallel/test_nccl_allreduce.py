"""Minimal 8-rank NCCL all_reduce test — no ColossalAI, no profiler.
Run via launch_3nodes.sh:
    bash launch_3nodes.sh --nccl-test
Or directly with torchrun on each node.
"""
import os
import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# Each process must bind to its own GPU.
torch.cuda.set_device(local_rank)

print(f"[rank {rank}] init done, world_size={world_size}, local_rank={local_rank}, "
      f"device={torch.cuda.current_device()}", flush=True)

t = torch.ones(1, device="cuda") * rank
print(f"[rank {rank}] calling all_reduce...", flush=True)
dist.all_reduce(t)
expected = sum(range(world_size))
print(f"[rank {rank}] all_reduce done, result={t.item():.0f} (expected {expected})", flush=True)

dist.barrier()
if rank == 0:
    print("ALL RANKS PASSED", flush=True)
dist.destroy_process_group()
