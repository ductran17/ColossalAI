"""
Debug script: compare Qwen2 block timing between:
  A) Standalone _TransformerBlock (profiler)
  B) Real Qwen2ForCausalLM WITHOUT ShardFormer (raw transformers)
  C) Real Qwen2ForCausalLM WITH ColossalAI ShardFormer (tp=2)

This identifies whether the slowdown comes from:
  - ColossalAI ShardFormer overhead (TP wrapping, dispatch)
  - RoPE/RMSNorm/GQA implementation differences vs our profiler
  - Memory layout / allocator differences

Run on a single GPU (no distributed needed for A and B):
  python debug_qwen_block.py

For C (ShardFormer) you need 2 GPUs or mock distributed:
  torchrun --nproc_per_node=2 debug_qwen_block.py
"""
import torch
import torch.nn as nn
import time
import math
import os
import sys

# ── Configuration (match your training setup) ──
BATCH = 2
SEQ = 256
HIDDEN = 896
HEADS = 14
N_KV = 2
INTERMEDIATE = 4864
LAYERS = 1
VOCAB = 151936
WARMUP = 10
REPEAT = 50


def iqr_clip(times_ms):
    t = torch.tensor(times_ms, dtype=torch.float64)
    q1, q3 = torch.quantile(t, 0.25).item(), torch.quantile(t, 0.75).item()
    iqr = q3 - q1
    clipped = t[(t >= q1 - 1.5*iqr) & (t <= q3 + 1.5*iqr)]
    return clipped.median().item() if clipped.numel() > 0 else t.median().item()


def time_fwd_bwd(name, model, x, extra_args=None):
    """Time forward+backward for a model that returns scalar loss."""
    device = torch.device("cuda")
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    extra_args = extra_args or {}
    
    # Warmup
    for _ in range(WARMUP):
        out = model(x, **extra_args)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        opt.zero_grad()
        torch.cuda.synchronize()
    
    # Timed
    times = []
    for _ in range(REPEAT):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        out = model(x, **extra_args)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        opt.zero_grad()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    
    median = iqr_clip(times)
    print(f"  {name:50s}: {median:7.3f} ms")
    return median


# ── A) Standalone profiler block ──
class ProfilerBlock(nn.Module):
    """Exact copy from profiler.py _TransformerBlock."""
    def __init__(self):
        super().__init__()
        h, a = HIDDEN, HEADS
        self.ln1 = nn.LayerNorm(h)
        self.q = nn.Linear(h, h, bias=False)
        kv_dim = h * N_KV // a
        self.k = nn.Linear(h, kv_dim, bias=False)
        self.v = nn.Linear(h, kv_dim, bias=False)
        self.out = nn.Linear(h, h, bias=False)
        self.ln2 = nn.LayerNorm(h)
        self.gate_proj = nn.Linear(h, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(h, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, h, bias=False)
        self.a = a
        self.n_kv = N_KV

    def forward(self, x):
        B, S, H = x.shape
        h = self.ln1(x)
        scale = math.sqrt(H // self.a)
        Q = self.q(h).reshape(B, S, self.a, -1).transpose(1, 2)
        K = self.k(h).reshape(B, S, self.n_kv, -1).transpose(1, 2)
        V = self.v(h).reshape(B, S, self.n_kv, -1).transpose(1, 2)
        if self.n_kv < self.a:
            r = self.a // self.n_kv
            K = K.repeat_interleave(r, dim=1)
            V = V.repeat_interleave(r, dim=1)
        att = torch.softmax(Q @ K.transpose(-2, -1) / scale, dim=-1) @ V
        att = att.transpose(1, 2).reshape(B, S, H)
        x = x + self.out(att)
        h = self.ln2(x)
        x = x + self.down_proj(torch.nn.functional.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


# ── B) Raw transformers Qwen2 block ──
def make_raw_qwen_block():
    import transformers
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=N_KV,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=32768,
        hidden_act="silu",
        use_cache=False,
    )
    model = transformers.Qwen2ForCausalLM(cfg)
    block = model.model.layers[0]
    return block.cuda()


# ── C) ColossalAI ShardFormer Qwen2 block ──
def make_shardformer_qwen_block():
    import transformers
    import colossalai
    from colossalai.shardformer import ShardFormer
    from colossalai.shardformer.policies.qwen2 import Qwen2ForCausalLMPolicy
    
    # Initialize distributed for ShardFormer
    if not dist.is_initialized():
        colossalai.launch_from_torch()
    
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=HEADS,
        num_key_value_heads=N_KV,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=32768,
        hidden_act="silu",
        use_cache=False,
    )
    model = transformers.Qwen2ForCausalLM(cfg).cuda()
    
    # Shard with tp=2
    shardformer = ShardFormer(
        tp_size=2,
        dp_size=1,
        pp_size=1,
    )
    policy = Qwen2ForCausalLMPolicy()
    sharded_model, _ = shardformer.optimize(model, policy)
    
    return sharded_model.model.layers[0]


if __name__ == "__main__":
    print("=" * 70)
    print("Qwen2 Block Timing Comparison")
    print(f"Config: batch={BATCH}, seq={SEQ}, hidden={HIDDEN}, heads={HEADS}, n_kv={N_KV}")
    print("=" * 70)
    
    device = torch.device("cuda")
    x = torch.randn(BATCH, SEQ, HIDDEN, device=device, requires_grad=True)
    pos_ids = torch.arange(SEQ, device=device).unsqueeze(0).expand(BATCH, -1)
    
    # Precompute RoPE cos/sin for Qwen
    import transformers
    cfg = transformers.Qwen2Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=1,
        num_attention_heads=HEADS, num_key_value_heads=N_KV,
        intermediate_size=INTERMEDIATE, max_position_embeddings=32768,
        hidden_act="silu", use_cache=False,
    )
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rope = Qwen2RotaryEmbedding(cfg, device=device)
    cos, sin = rope(x, pos_ids)
    pos_emb = (cos, sin)
    
    # A) Profiler standalone
    print("\n[A] Profiler _TransformerBlock (standalone):")
    blk_a = ProfilerBlock().cuda()
    t_profiler = time_fwd_bwd("ProfilerBlock fwd+bwd", blk_a, x)
    
    # B) Raw transformers
    print("\n[B] Raw transformers Qwen2DecoderLayer:")
    blk_b = make_raw_qwen_block()
    t_raw = time_fwd_bwd("Raw Qwen2 block fwd+bwd", blk_b, x, extra_args={"position_embeddings": pos_emb})
    
    # C) ColossalAI ShardFormer
    print("\n[C] ColossalAI ShardFormer Qwen2DecoderLayer (tp=2):")
    try:
        blk_c = make_shardformer_qwen_block()
        t_shard = time_fwd_bwd("ShardFormer Qwen2 block fwd+bwd", blk_c, x, extra_args={"position_embeddings": pos_emb})
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  (ShardFormer requires torchrun with >=2 GPUs)")
        t_shard = None
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  [A] Profiler standalone : {t_profiler:.3f} ms")
    print(f"  [B] Raw transformers    : {t_raw:.3f} ms  (vs A: {t_raw/t_profiler:.2f}x)")
    if t_shard:
        print(f"  [C] ShardFormer (tp=2)  : {t_shard:.3f} ms  (vs A: {t_shard/t_profiler:.2f}x, vs B: {t_shard/t_raw:.2f}x)")
    
    print("\nInterpretation:")
    if t_raw / t_profiler > 1.5:
        print("  Raw transformers is >1.5x slower than profiler.")
        print("  → Root cause: RoPE/RMSNorm/GQA implementation differs from profiler.")
    else:
        print("  Raw transformers ~matches profiler.")
        print("  → Slowness is NOT from RoPE/RMSNorm/GQA.")
    
    if t_shard and t_shard / t_raw > 1.3:
        print("  ShardFormer is >1.3x slower than raw.")
        print("  → Root cause: ColossalAI TP wrapping / dispatch overhead.")
    elif t_shard:
        print("  ShardFormer ~matches raw.")
        print("  → ColossalAI overhead is negligible.")
    print("=" * 70)

