"""
Profile one training step to see where time is actually spent.
Run with torchrun for distributed training.
"""
import torch, torch.distributed as dist, os, time
import json

# Need transformers + colossalai for actual training
from transformers import GPT2Config, GPT2LMHeadModel
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from torch.profiler import profile, record_function, ProfilerActivity

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Small config for fast profiling
    config = GPT2Config(
        n_positions=256,
        n_layer=24,
        n_head=16,
        n_embd=1024,
        n_inner=1024 * 4,
        vocab_size=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config)

    # Use pp=2, tp=1, dp=1 for 2-GPU local test (adjust as needed)
    pp, tp, dp = 2, 1, 1
    assert world_size == pp * tp * dp

    plugin = HybridParallelPlugin(
        tp_size=tp,
        pp_size=pp,
        num_microbatches=8,
        enable_all_optimization=True,
    )
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4))

    batch = {
        "input_ids": torch.randint(0, 1024, (16, 256)),
        "attention_mask": torch.ones(16, 256, dtype=torch.long),
        "labels": torch.randint(0, 1024, (16, 256)),
    }

    # Warmup
    for _ in range(3):
        output = booster.execute_pipeline(
            batch, model, lambda o, b: o.loss, optimizer, return_loss=True
        )

    # Profile one step
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        with record_function("execute_pipeline"):
            output = booster.execute_pipeline(
                batch, model, lambda o, b: o.loss, optimizer, return_loss=True
            )

    if rank == 0:
        # Print top 20 CUDA operations by time
        print("=" * 80)
        print(f"PyTorch Profiler: pp={pp}, tp={tp}, dp={dp}")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        # Also save to file for analysis
        prof.export_chrome_trace(f"/tmp/profiler_pp{pp}_tp{tp}.json")
        print(f"\nTrace saved to /tmp/profiler_pp{pp}_tp{tp}.json")
        print("Open in chrome://tracing to visualize")

    dist.destroy_process_group()
