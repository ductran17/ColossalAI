"""
Overhead 5: Measure PP stage manager dispatch overhead.
Compares actual execute_pipeline time vs bare PyTorch equivalent.
Run with torchrun matching pp*tp world_size.
"""
import torch, torch.distributed as dist, os, time
from transformers import GPT2Config, GPT2LMHeadModel
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # pp=2, tp=1, requires 2 GPUs
    pp, tp, dp = 2, 1, 1
    assert world_size == pp * tp * dp, f"world_size={world_size} != pp*tp*dp={pp*tp*dp}"

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

    plugin = HybridParallelPlugin(
        tp_size=tp,
        pp_size=pp,
        num_microbatches=8,
        enable_all_optimization=True,
    )
    booster = Booster(plugin=plugin)
    model, optimizer, _, _, _ = booster.boost(
        model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4)
    )

    # Create data iterator (execute_pipeline expects iterator)
    def make_batch():
        return {
            "input_ids": torch.randint(0, 1024, (16, 256)),
            "attention_mask": torch.ones(16, 256, dtype=torch.long),
            "labels": torch.randint(0, 1024, (16, 256)),
        }

    # Warmup
    for _ in range(3):
        data_iter = iter([make_batch()])
        output = booster.execute_pipeline(
            data_iter, model, lambda o, b: o.loss, optimizer, return_loss=True
        )

    # Measure execute_pipeline step time
    WARMUP = 5
    REPEAT = 20

    for _ in range(WARMUP):
        data_iter = iter([make_batch()])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = booster.execute_pipeline(
            data_iter, model, lambda o, b: o.loss, optimizer, return_loss=True
        )
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(REPEAT):
        data_iter = iter([make_batch()])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = booster.execute_pipeline(
            data_iter, model, lambda o, b: o.loss, optimizer, return_loss=True
        )
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    avg_time = sum(times_ms) / len(times_ms)

    # Bare PyTorch equivalent: 12 layers, 8 microbatches, forward+backward
    # We can't run this in the same process, so we use the framework measurement
    # from debug_overhead_1_framework.py:
    # T_block_bare = 1.94 ms per block per microbatch
    # Total bare = 12 layers * 8 microbatches * 1.94 ms = 186 ms
    # Plus Adam = 38 ms
    # Total bare equivalent ≈ 224 ms
    T_block_bare_ms = 1.94  # from debug_overhead_1_framework.py
    layers_per_stage = 24 // pp  # = 12
    bare_compute_ms = layers_per_stage * 8 * T_block_bare_ms  # 186 ms
    adam_ms = 37.77
    bare_total_ms = bare_compute_ms + adam_ms  # ~224 ms

    pp_dispatch_overhead_ms = avg_time - bare_total_ms

    if rank == 0:
        print("=" * 60)
        print("PP Stage Manager Dispatch Overhead")
        print("=" * 60)
        print(f"Actual execute_pipeline:     {avg_time:.2f} ms")
        print(f"Bare compute (12×8×1.94):    {bare_compute_ms:.2f} ms")
        print(f"Adam step:                   {adam_ms:.2f} ms")
        print(f"Bare total:                  {bare_total_ms:.2f} ms")
        print(f"PP dispatch overhead:        {pp_dispatch_overhead_ms:.2f} ms")
        print(f"Per-microbatch overhead:       {pp_dispatch_overhead_ms / 8:.2f} ms")

    dist.destroy_process_group()
