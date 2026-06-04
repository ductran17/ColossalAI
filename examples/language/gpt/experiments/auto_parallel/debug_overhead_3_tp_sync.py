"""
Overhead 3: Measure real AllReduce time vs analytical model.
Run with torchrun on 2 GPUs (intra-node).
"""
import torch, torch.distributed as dist, os, time

def measure_allreduce(tensor_size_bytes, n, warmup=10, repeat=50):
    """Returns (raw_time_ms, model_time_ms)"""
    rank = dist.get_rank()
    tensor = torch.randn(tensor_size_bytes // 4, device="cuda")  # fp32 = 4 bytes

    # Analytical model (use your measured alpha/beta)
    alpha = 104e-6   # seconds (your alpha_intra)
    beta = 0.043e-9  # s/byte (your beta_intra)
    model_time = 2 * (n - 1) / n * (alpha + beta * tensor_size_bytes)

    for _ in range(warmup):
        dist.all_reduce(tensor)
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    raw_time = sum(times) / len(times)
    return raw_time * 1000, model_time * 1000

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    n = dist.get_world_size()

    # For H=1024, B=2, S=256, dtype=fp32:
    # activation_bytes = 2 * 256 * 1024 * 4 = 2,097,152 bytes ≈ 2 MB
    act_bytes = 2 * 256 * 1024 * 4

    raw, model = measure_allreduce(act_bytes, n)
    if dist.get_rank() == 0:
        print(f"[AllReduce {act_bytes/1024/1024:.2f} MB, n={n}]")
        print(f"  Raw measured: {raw:.2f} ms")
        print(f"  Model predicted: {model:.2f} ms")
        print(f"  Sync overhead: {raw - model:.2f} ms ({(raw/model - 1)*100:.1f}%)")

        # For a plan with tp=2, layers=12, M=8:
        # 2 AllReduces/layer × 12 layers × 8 microbatches = 192 collectives
        total_tp_sync = (raw - model) * 2 * 12 * 8
        print(f"  Total TP sync overhead for tp=2,pp=2: {total_tp_sync:.2f} ms")

    dist.destroy_process_group()
