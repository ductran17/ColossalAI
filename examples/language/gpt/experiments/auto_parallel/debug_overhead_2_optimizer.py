"""
Overhead 2: Measure Adam optimizer step time in isolation.
Run on 1 GPU.
"""
import torch, time

H = 1024
layers = 24
total_params = layers * 12 * H * H  # ~302M params

# Create a dummy model with same param count
model = torch.nn.Sequential(
    torch.nn.Linear(total_params, 1, bias=False)
)
device = torch.device("cuda")
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Fill with gradients
for p in model.parameters():
    p.grad = torch.randn_like(p)

WARMUP = 10
REPEAT = 50

for _ in range(WARMUP):
    opt.step()
    opt.zero_grad()
    for p in model.parameters():
        p.grad = torch.randn_like(p)

times = []
for _ in range(REPEAT):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    opt.step()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    for p in model.parameters():
        p.grad = torch.randn_like(p)

T_optimizer_ms = sum(times) / len(times) * 1000
print(f"[Adam step] {total_params/1e6:.0f}M params: {T_optimizer_ms:.2f} ms")
print(f"[Cost model estimate] ~1.0 ms (from T_block scaling)")
print(f"[Gap] {T_optimizer_ms - 1.0:.2f} ms")
