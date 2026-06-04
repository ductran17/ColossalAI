"""
Overhead 1: Measure per-microbatch framework overhead.
Run on 1 GPU (no distributed needed).
"""
import torch, torch.nn as nn, time, math

H = 1024
B = 2   # per-microbatch size
S = 256
M = 8   # num microbatches
WARMUP = 10
REPEAT = 50

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
model = Block().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Baseline: one block, one microbatch
x = torch.randn(B, S, H, device=device, requires_grad=True)
for _ in range(WARMUP):
    model(x).sum().backward()
    opt.zero_grad()

times = []
for _ in range(REPEAT):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = model(x).sum()
    loss.backward()
    opt.zero_grad()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

T_block_bare_ms = sum(times) / len(times) * 1000
print(f"[Bare PyTorch] 1 block 1 microbatch: {T_block_bare_ms:.2f} ms")

# Simulated framework overhead: run M microbatches sequentially
# with explicit Python loop and zero_grad between each (mimics PipelineStageManager)
model2 = Block().to(device)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
for _ in range(WARMUP):
    for _ in range(M):
        loss = model2(x).sum()
        loss.backward()
    opt2.zero_grad()

times = []
for _ in range(REPEAT):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(M):
        loss = model2(x).sum()
        loss.backward()
    opt2.zero_grad()
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

T_with_loop_ms = sum(times) / len(times) * 1000
per_microbatch_overhead_ms = (T_with_loop_ms - T_block_bare_ms * M) / M

print(f"[With Python loop] {M} microbatches: {T_with_loop_ms:.2f} ms")
print(f"[Framework overhead] per microbatch: {per_microbatch_overhead_ms:.2f} ms")
print(f"[Total framework overhead for M={M}]: {per_microbatch_overhead_ms * M:.2f} ms")

# Scaled for full model (24 layers, pp stages)
for pp in [1, 2, 4]:
    layers_per_stage = 24 // pp
    total_framework_ms = per_microbatch_overhead_ms * M * layers_per_stage * pp
    print(f"  [pp={pp}] layers_per_stage={layers_per_stage}, total framework overhead: {total_framework_ms:.2f} ms")
