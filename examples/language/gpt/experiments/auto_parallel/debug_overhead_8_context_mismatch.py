"""
Test Claude's hypothesis: Is T_grad_acc actually compensating for T_compute underestimate?

Measures T_block in two contexts:
1. ISOLATION: single block, single microbatch (current profiler)
2. DIST_CONTEXT: same block running while full distributed training is active

If T_block_distributed / T_block_isolated ≈ 1.4-1.6, then T_grad_acc (40-65% of exec OH)
is actually compensating for T_compute being too low.
"""
import torch, torch.nn as nn, time, math
import torch.distributed as dist
import os

H = 1024
B = 2
S = 256

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(H)
        self.q = nn.Linear(H, H, bias=False)
        self.k = nn.Linear(H, H, bias=False)
        self.v = nn.Linear(H, H, bias=False)
        self.out = nn.Linear(H, H, bias=False)
        self.ln2 = nn.LayerNorm(H)
        self.fc1 = nn.Linear(H, 4*H, bias=False)
        self.fc2 = nn.Linear(4*H, H, bias=False)
    def forward(self, x):
        B, S, H = x.shape
        h = self.ln1(x)
        Q = self.q(h).reshape(B, S, 16, -1).transpose(1,2)
        K = self.k(h).reshape(B, S, 16, -1).transpose(1,2)
        V = self.v(h).reshape(B, S, 16, -1).transpose(1,2)
        att = torch.softmax(Q @ K.transpose(-2,-1) / math.sqrt(H//16), dim=-1) @ V
        att = att.transpose(1,2).reshape(B, S, H)
        x = x + self.out(att)
        h = self.ln2(x)
        x = x + self.fc2(torch.relu(self.fc1(h)))
        return x

device = torch.device("cuda")

print("=" * 80)
print("CLAUDE HYPOTHESIS TEST: T_compute context mismatch")
print("=" * 80)

# ── 1. Measure T_block in ISOLATION (same as profiler) ──────────────
block = Block().to(device)
opt = torch.optim.SGD(block.parameters(), lr=1e-4)
x = torch.randn(B, S, H, device=device, requires_grad=True)

# Warmup
for _ in range(10):
    loss = block(x).sum()
    loss.backward()
    opt.zero_grad()
    torch.cuda.synchronize()

times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = block(x).sum()
    loss.backward()
    opt.zero_grad()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

T_block_isolated = sum(times) / len(times)
print(f"\n1. T_block ISOLATED (single block, 1 microbatch):")
print(f"   {T_block_isolated*1000:.2f} ms")

# ── 2. Measure T_block with MEMORY PRESSURE (simulate pipeline) ──────
# Keep M=8 microbatches' worth of activations in memory
del block, opt, x
block2 = Block().to(device)
opt2 = torch.optim.SGD(block2.parameters(), lr=1e-4)

# Pre-allocate memory pressure: store activations for 8 microbatches
activation_buffer = []
for _ in range(8):
    x_temp = torch.randn(B, S, H, device=device, requires_grad=True)
    out = block2(x_temp)
    activation_buffer.append(out)  # keep refs to simulate memory pressure

x2 = torch.randn(B, S, H, device=device, requires_grad=True)

# Warmup
for _ in range(10):
    loss = block2(x2).sum()
    loss.backward()
    opt2.zero_grad()
    torch.cuda.synchronize()

times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = block2(x2).sum()
    loss.backward()
    opt2.zero_grad()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

T_block_with_pressure = sum(times) / len(times)
print(f"\n2. T_block WITH MEMORY PRESSURE (8 microbatch activations in GPU):")
print(f"   {T_block_with_pressure*1000:.2f} ms")

# Clean up
del activation_buffer, block2, opt2, x2
torch.cuda.empty_cache()

# ── 3. Measure T_block with L2 CACHE POLLUTION (simulate grad buffers) ─
block3 = Block().to(device)
opt3 = torch.optim.SGD(block3.parameters(), lr=1e-4)

# Pre-fill L2 cache with gradient-sized buffers
grad_size = sum(p.numel() for p in block3.parameters()) * 4  # fp32
fake_grads = [torch.randn(grad_size // 4, device=device) for _ in range(10)]

x3 = torch.randn(B, S, H, device=device, requires_grad=True)

for _ in range(10):
    loss = block3(x3).sum()
    loss.backward()
    opt3.zero_grad()
    torch.cuda.synchronize()

times = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = block3(x3).sum()
    loss.backward()
    opt3.zero_grad()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

T_block_with_cache_pollution = sum(times) / len(times)
print(f"\n3. T_block WITH L2 CACHE POLLUTION (grad buffers in cache):")
print(f"   {T_block_with_cache_pollution*1000:.2f} ms")

del fake_grads, block3, opt3, x3
torch.cuda.empty_cache()

# ── 4. Summary ───────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("RATIOS (distributed / isolated)")
print(f"{'='*80}")
ratio_pressure = T_block_with_pressure / T_block_isolated
ratio_cache = T_block_with_cache_pollution / T_block_isolated

print(f"  Memory pressure:     {ratio_pressure:.2f}x")
print(f"  Cache pollution:     {ratio_cache:.2f}x")
print(f"\n  If ratio ≈ 1.0:  T_compute formula is correct, T_grad_acc is real overhead")
print(f"  If ratio ≈ 1.3+: T_compute is underestimated, T_grad_acc compensates for it")

# ── 5. What T_grad_acc currently adds ──────────────────────────────
M = 8
layers = 24
pp = 2
layers_per_stage = layers // pp
param_bytes_per_layer = (3*H*H + H*H + H*4*H + 4*H*H + 4*H) * 4  # fp32
bw_grad = 150e9

T_grad_acc_formula = (M * layers_per_stage * param_bytes_per_layer * 2) / bw_grad
print(f"\n  Current T_grad_acc formula gives: {T_grad_acc_formula*1000:.1f} ms")
print(f"  T_compute for this config:        {layers_per_stage * (T_block_isolated/1) * M * 1000:.1f} ms")
print(f"  T_grad_acc / T_compute ratio:     {T_grad_acc_formula / (layers_per_stage * T_block_isolated * M):.2f}")

print(f"\n{'='*80}")
print("HYPOTHESIS CHECK:")
print(f"{'='*80}")
if ratio_pressure > 1.2 or ratio_cache > 1.2:
    print("  ✓ CLAUDE IS RIGHT: T_compute is underestimated due to context mismatch.")
    print(f"    Fix: T_compute_corrected = T_compute × {max(ratio_pressure, ratio_cache):.2f}")
    print("    Then T_grad_acc can be removed (it was compensating).")
else:
    print("  ✗ Claude hypothesis NOT confirmed: T_block in context ≈ isolated.")
    print("    T_grad_acc is a real overhead, not double-counting.")
    print("    But the formula may still overestimate by 2-3x.")
EOF