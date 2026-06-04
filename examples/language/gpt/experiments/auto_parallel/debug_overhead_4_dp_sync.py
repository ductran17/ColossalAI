"""
Overhead 4: Measure DP AllReduce with real gradient sizes.
Run with torchrun on 2+ GPUs for cross-node measurement.
"""
import torch, torch.distributed as dist, os, time

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    if dist.get_rank() == 0:
        print("=" * 60)
        print("DP AllReduce Overhead Measurement")
        print("=" * 60)

    H = 1024
    dtype_bytes = 4
    param_bytes_per_layer = (3*H*H + H*H + H*4*H + 4*H*H + 4*H) * dtype_bytes

    for pp in [1, 2, 4]:
        layers_per_stage = 24 // pp
        total_grad_bytes = param_bytes_per_layer * layers_per_stage

        grad = torch.randn(total_grad_bytes // 4, device="cuda")
        warmup = 10
        repeat = 50

        for _ in range(warmup):
            dist.all_reduce(grad)
            torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(grad)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        T_raw_ms = sum(times) / len(times) * 1000

        alpha = 77e-6
        beta = 0.36e-9
        n = dist.get_world_size()
        T_model_ms = 2 * (n - 1) / n * (alpha + beta * total_grad_bytes) * 1000

        if dist.get_rank() == 0:
            print(f"[DP AllReduce pp={pp}] {total_grad_bytes/1024/1024:.2f} MB, n={n}")
            print(f"  Raw measured: {T_raw_ms:.2f} ms")
            print(f"  Model predicted: {T_model_ms:.2f} ms")
            print(f"  Exposed overhead: {T_raw_ms - T_model_ms:.2f} ms")
            print(f"  Implied overlap factor: {T_model_ms / T_raw_ms:.2f}")
            print()

    dist.destroy_process_group()
