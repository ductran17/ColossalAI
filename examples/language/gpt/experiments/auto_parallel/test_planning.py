import sys
sys.path.insert(0, '.')
import torch
import transformers
from gpt_modules import GPT2Block
from colossalai.auto_parallel.pipeline_shard import build_pipeline_plan

config = transformers.GPT2Config(n_positions=64, n_layer=4, n_head=4, n_embd=128, resid_pdrop=0.0, attn_pdrop=0.0)
layers = [GPT2Block(config, layer_idx=i) for i in range(4)]
meta_args = {'hidden_states': torch.empty(2, 64, 128, device='meta')}

# Pure Python, no distributed needed — runs on 1 process
plan = build_pipeline_plan(
    layers=layers, meta_args=meta_args,
    num_devices=2, num_microbatches=2,
    num_hosts=1, devices_per_host=2,
    uniform_tp_degree=2,
    mesh_alpha=[1e-5, 1e-5], mesh_beta=[1e-11, 1e-11],
    cache_path='/tmp/auto3d_test_cache',
)
print(f'pp={plan.pp_size}, tp={plan.tp_size}, dp={plan.dp_size}')
print(f'stage ranges: {plan.stage_layer_ranges}')
print(f'estimated cost: {plan.estimated_cost:.4f}')
