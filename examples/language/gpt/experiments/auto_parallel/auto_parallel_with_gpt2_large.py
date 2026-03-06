"""
Large model benchmark: ~2.7B parameter GPT-2-style model with synthetic data.
ColossalAI Auto-Parallel counterpart to the DeepSpeed ZeRO-3 benchmark.

Model: Custom GPT-2 (n_layer=32, n_embd=2560, n_head=32) — ~2.65B params
  fp16 params:         ~5.3 GB total

Data: Fully synthetic (random token tensors) — no download required.

Run example (4 GPUs):
  torchrun --nproc_per_node=4 auto_parallel_with_gpt2_large.py \
      --steps 100 --seq_len 512 --per_device_bs 2
"""

import argparse
import time
from functools import partial

import torch
import transformers
from gpt_modules import GPT2LMHeadModel, GPTLMLoss

from colossalai.auto_parallel.tensor_shard.initialize import autoparallelize
from colossalai.initialize import launch_from_torch
from colossalai.logging import disable_existing_loggers, get_dist_logger

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--steps",         type=int, default=100)
parser.add_argument("--seq_len",       type=int, default=512)
parser.add_argument("--per_device_bs", type=int, default=2)
parser.add_argument("--log_mem_every", type=int, default=10,
                    help="Print per-rank memory every N steps (0 = off)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model config — mirrors the DeepSpeed benchmark exactly
# ---------------------------------------------------------------------------
VOCAB_SIZE = 50257

MODEL_CFG = transformers.GPT2Config(
    vocab_size   = VOCAB_SIZE,
    n_positions  = 1024,
    n_embd       = 2560,
    n_layer      = 32,
    n_head       = 32,
    n_inner      = 10240,
    resid_pdrop  = 0.0,
    attn_pdrop   = 0.0,
    embd_pdrop   = 0.0,
)

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def mem_stats_str(prefix=""):
    ma    = torch.cuda.memory_allocated()    / 1024**3
    max_ma= torch.cuda.max_memory_allocated() / 1024**3
    ca    = torch.cuda.memory_reserved()     / 1024**3
    max_ca= torch.cuda.max_memory_reserved() / 1024**3
    tag   = f" [{prefix}]" if prefix else ""
    return (
        f"{tag} "
        f"MA {ma:.3f} GB | Max_MA {max_ma:.3f} GB | "
        f"CA {ca:.3f} GB | Max_CA {max_ca:.3f} GB"
    )


def print_mem_all_ranks(label=""):
    world_size = torch.distributed.get_world_size()
    local_rank = torch.distributed.get_rank()
    torch.distributed.barrier()
    for r in range(world_size):
        if local_rank == r:
            print(
                f"[Rank {r} | GPU {r}]" + mem_stats_str(label),
                flush=True,
            )
        torch.distributed.barrier()


def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12) / 8


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    disable_existing_loggers()
    launch_from_torch()
    logger = get_dist_logger()

    rank       = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    total_params = sum(
        MODEL_CFG.n_layer * (
            2 * MODEL_CFG.n_embd +
            MODEL_CFG.n_embd * 3 * MODEL_CFG.n_embd +
            MODEL_CFG.n_embd * MODEL_CFG.n_embd +
            2 * MODEL_CFG.n_embd +
            MODEL_CFG.n_embd * MODEL_CFG.n_inner +
            MODEL_CFG.n_inner * MODEL_CFG.n_embd
        ) for _ in [0]
    ) + VOCAB_SIZE * MODEL_CFG.n_embd + 1024 * MODEL_CFG.n_embd

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  Framework:    ColossalAI Auto-Parallel")
        print(f"  Model:        GPT-2 style, ~{total_params/1e9:.2f}B parameters")
        print(f"  Seq len:      {args.seq_len}")
        print(f"  Per-device BS:{args.per_device_bs}")
        print(f"  Steps:        {args.steps}")
        print(f"  GPUs:         {world_size}")
        print(f"  Est. fp16 model size: {total_params*2/1e9:.1f} GB total")
        print(f"{'='*60}\n")

    # Build model in fp16 on CUDA
    model = GPT2LMHeadModel(config=MODEL_CFG).half().cuda()
    global_numel = sum(p.numel() for p in model.parameters())

    # Meta sample for auto-parallel tracing
    meta_input_sample = {
        "input_ids":      torch.zeros((args.per_device_bs, args.seq_len), dtype=torch.int64).to("meta"),
        "attention_mask": torch.zeros((args.per_device_bs, args.seq_len), dtype=torch.int64).to("meta"),
    }

    logger.info("Running autoparallelize...", ranks=[0])
    gm, solution = autoparallelize(model, meta_input_sample, return_solution=True)

    if rank == 0:
        for node_strategy in solution:
            print(node_strategy)

    criterion = GPTLMLoss()
    optimizer = torch.optim.Adam(gm.parameters(), lr=1e-4)

    logger.info("Model initialized." + mem_stats_str("post-init"), ranks=[0])
    print_mem_all_ranks("post-init")

    get_tflops_func = partial(get_tflops, global_numel, args.per_device_bs * world_size, args.seq_len)

    torch.cuda.synchronize()
    gm.train()

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    step_times   = []
    tokens_total = 0

    for step in range(args.steps):
        input_ids = torch.randint(
            0, VOCAB_SIZE,
            (args.per_device_bs, args.seq_len),
            device=torch.cuda.current_device(),
        )
        attention_mask = torch.ones_like(input_ids)

        optimizer.zero_grad()
        t0 = time.time()

        outputs = gm(input_ids, attention_mask)
        loss    = criterion(outputs, input_ids)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - t0

        step_times.append(elapsed)
        tokens_total += args.per_device_bs * args.seq_len * world_size

        if args.log_mem_every > 0 and (step + 1) % args.log_mem_every == 0:
            if rank == 0:
                tflops = get_tflops_func(elapsed)
                print(
                    f"\n--- Step {step+1}/{args.steps} | "
                    f"loss: {loss.item():.4f} | "
                    f"step_time: {elapsed*1000:.1f} ms | "
                    f"TFLOPS: {tflops:.3f} ---"
                )
            print_mem_all_ranks(f"step {step+1}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    torch.cuda.synchronize()

    if rank == 0:
        print("\n=== Final Memory (all ranks) ===")
    print_mem_all_ranks("final")

    warmup  = min(10, args.steps // 5)
    stable  = step_times[warmup:]
    avg_ms  = sum(stable) / len(stable) * 1000

    if rank == 0:
        global_bs          = args.per_device_bs * world_size
        throughput_samples = global_bs / (avg_ms / 1000)
        throughput_tokens  = global_bs * args.seq_len / (avg_ms / 1000)

        # Final memory snapshot on rank 0
        ma    = torch.cuda.memory_allocated()    / 1024**3
        max_ma= torch.cuda.max_memory_allocated() / 1024**3
        ca    = torch.cuda.memory_reserved()     / 1024**3
        max_ca= torch.cuda.max_memory_reserved() / 1024**3

        print("\n=== Benchmark Results ===")
        print(f"Framework:       ColossalAI Auto-Parallel")
        print(f"Model:           GPT-2 style ~2.65B params")
        print(f"GPUs:            {world_size}")
        print(f"Global BS:       {global_bs} (per-device: {args.per_device_bs})")
        print(f"Seq len:         {args.seq_len}")
        print(f"Warmup steps:    {warmup}")
        print(f"Measured steps:  {len(stable)}")
        print(f"Avg step time:   {avg_ms:.1f} ms  (steps {warmup+1}-{args.steps})")
        print(f"Throughput:      {throughput_samples:.2f} samples/sec")
        print(f"Throughput:      {throughput_tokens/1000:.2f} K tokens/sec")
        print(f"Rank-0 memory:   MA {ma:.3f} GB | Max_MA {max_ma:.3f} GB | "
              f"CA {ca:.3f} GB | Max_CA {max_ca:.3f} GB")
        print(f"(per-rank peak memory shown above)")


if __name__ == "__main__":
    main()
