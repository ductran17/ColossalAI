"""
Measure whether gradient accumulation adds EXPOSED overhead.
Compare backward pass with/without gradient accumulation to see if
grad_acc overlaps with compute or adds serial time.
"""
import torch, torch.nn as nn, time, math

H = 1024
layers = 24
B = 2   # per-microbatch size
S = 256
WARMUP = 5
REPEAT = 20

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

class Model(nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

device = torch.device("cuda")

# ── 1. Measure WITHOUT grad accumulation (no .backward()) ─────────────
# This is pure forward pass time
def measure_forward_only(model, x, warmup, repeat):
    for _ in range(warmup):
        y = model(x).sum()
        torch.cuda.synchronize()
    
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = model(x).sum()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

# ── 2. Measure backward WITHOUT grad accumulation (single microbatch) ──
# grad.zero_grad() before backward → no accumulation
def measure_backward_no_accum(model, x, opt, warmup, repeat):
    for _ in range(warmup):
        opt.zero_grad()
        loss = model(x).sum()
        loss.backward()
        opt.zero_grad()
        torch.cuda.synchronize()
    
    times = []
    for _ in range(repeat):
        opt.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(x).sum()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

# ── 3. Measure backward WITH grad accumulation (multiple microbatches) ──
def measure_backward_with_accum(model, x, opt, M, warmup, repeat):
    for _ in range(warmup):
        opt.zero_grad()
        for _ in range(M):
            loss = model(x).sum()
            loss.backward()
        opt.zero_grad()
        torch.cuda.synchronize()
    
    times = []
    for _ in range(repeat):
        opt.zero_grad()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(M):
            loss = model(x).sum()
            loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)

print("=" * 70)
print("GRADIENT ACCUMULATION OVERHEAD TEST")
print("=" * 70)
print(f"Model: {layers} layers, H={H}, batch={B}, seq={S}")
print()

# Test with different layer counts to see scaling
for test_layers in [1, 6, 12, 24]:
    model = Model(test_layers).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    x = torch.randn(B, S, H, device=device, requires_grad=True)
    
    print(f"--- {test_layers} layers ---")
    
    # Forward only
    t_fwd = measure_forward_only(model, x, WARMUP, REPEAT)
    print(f"  Forward only:              {t_fwd*1000:.2f} ms")
    
    # Backward no accumulation
    t_bwd_no_accum = measure_backward_no_accum(model, x, opt, WARMUP, REPEAT)
    print(f"  Backward (no grad accum):  {t_bwd_no_accum*1000:.2f} ms")
    
    # Backward with accumulation M=8
    M = 8
    t_bwd_accum = measure_backward_with_accum(model, x, opt, M, WARMUP, REPEAT)
    print(f"  Backward (M={M}, with grad accum): {t_bwd_accum*1000:.2f} ms")
    
    # Calculate exposed overhead
    expected_if_no_overlap = t_bwd_no_accum * M
    actual = t_bwd_accum
    exposed_overhead = actual - expected_if_no_overlap
    
    print(f"  Expected (M × single):     {expected_if_no_overlap*1000:.2f} ms")
    print(f"  Actual (M with accum):     {actual*1000:.2f} ms")
    print(f"  Exposed overhead:          {exposed_overhead*1000:.2f} ms")
    
    if abs(exposed_overhead) < t_bwd_no_accum * 0.1:
        print(f"  → Grad acc OVERLAPS with backward (exposed < 10% of single)")
    elif exposed_overhead > 0:
        print(f"  → Grad acc ADDS exposed overhead!")
    else:
        print(f"  → Grad acc is HIDDEN (negative overhead = better overlap)")
    print()
    
    del model, opt
    torch.cuda.empty_cache()

print("=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("If 'Exposed overhead' ≈ 0 for all layer counts:")
print("  → grad_acc is fully overlapped with backward compute")
print("  → REMOVE T_grad_acc from cost model (double counting)")
print()
print("If 'Exposed overhead' > 0 significantly:")
print("  → grad_acc adds serial time beyond backward compute")
print("  → Keep T_grad_acc but fix formula to max(0, raw - T_compute)")
