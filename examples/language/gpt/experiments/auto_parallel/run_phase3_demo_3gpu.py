"""
Phase 3 end-to-end demo: variable-size pipeline stages on 3 GPUs.

Topology: pp=2, tp=[1,2], dp=1 — 3 GPUs total.
  • Stage 0: 1 GPU (tp=1) — first half of layers
  • Stage 1: 2 GPUs (tp=2) — second half of layers

This demonstrates that alpa_dp can plan pipelines where different stages
own different numbers of devices, which Phase 1/2 cannot do.

Run with:
    torchrun --nproc_per_node=3 run_phase3_demo_3gpu.py

Expected output (on rank 0 / last stage):
    [Phase 3] Plan: pp=2, var_stages=True
      Stage 0: layers[0:2], tp=1, dp=1, ranks=[0]
      Stage 1: layers[2:4], tp=2, dp=1, ranks=[1, 2]
    Step 1/3  loss=...
    Step 2/3  loss=...
    Step 3/3  loss=...
    Done. Phase 3 variable-size stage demo completed successfully.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", ".."))
for p in [_root, _here]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
import transformers
from gpt_modules import GPT2Block

import colossalai
from colossalai.auto_parallel.pipeline_shard import (
    CrossMeshP2PCommunication,
    PipelinePlan,
    VariableStagePipelineManager,
    autoparallelize_with_pp,
)
from colossalai.logging import disable_existing_loggers, get_dist_logger

# ------------------------------------------------------------------ #
# Hyper-parameters                                                     #
# ------------------------------------------------------------------ #
NUM_LAYERS = 4
BATCH_SIZE = 2
SEQ_LEN = 64
HIDDEN = 256
HEADS = 4
NUM_STEPS = 3
CACHE_PATH = "/tmp/auto3d_phase3_demo"


def main():
    disable_existing_loggers()
    colossalai.launch_from_torch()
    logger = get_dist_logger()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    assert world_size == 3, (
        f"This demo requires exactly 3 GPUs (got {world_size}). "
        "Run with: torchrun --nproc_per_node=3 run_phase3_demo_3gpu.py"
    )

    # Build transformer layers.
    config = transformers.GPT2Config(
        n_positions=SEQ_LEN,
        n_layer=NUM_LAYERS,
        n_head=HEADS,
        n_embd=HIDDEN,
        n_inner=HIDDEN * 4,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    layers = [GPT2Block(config, layer_idx=i) for i in range(NUM_LAYERS)]
    meta_args = {
        "hidden_states": torch.empty(BATCH_SIZE, SEQ_LEN, HIDDEN, device="meta"),
    }

    if rank == 0:
        logger.info(
            f"[Phase 3] Auto-planning variable-size stages: "
            f"{world_size} GPUs, {NUM_LAYERS} layers, "
            f"batch={BATCH_SIZE}, seq={SEQ_LEN}, hidden={HIDDEN}",
            ranks=[0],
        )

    # ------------------------------------------------------------------ #
    # Auto-plan with variable_stage_sizes=True.                           #
    # The planner will find the optimal split with different device counts #
    # per stage (e.g. 1 GPU for stage 0, 2 GPUs for stage 1).            #
    # ------------------------------------------------------------------ #
    stage_module, stage_manager, plan = autoparallelize_with_pp(
        layers=layers,
        meta_args=meta_args,
        num_microbatches=BATCH_SIZE,
        variable_stage_sizes=True,
        cache_path=CACHE_PATH,
    )

    if rank == 0:
        logger.info(
            f"[Phase 3] Plan: pp={plan.pp_size}, var_stages={plan.variable_stage_sizes}, "
            f"estimated_cost={plan.estimated_cost:.4f}s",
            ranks=[0],
        )
        for i, (start, end) in enumerate(plan.stage_layer_ranges):
            tp_s = plan.tp_per_stage[i]
            dp_s = plan.dp_per_stage[i]
            logger.info(
                f"  Stage {i}: layers[{start}:{end}], tp={tp_s}, dp={dp_s}, "
                f"ranks={plan.rank_ranges[i]}",
                ranks=[0],
            )

    # ------------------------------------------------------------------ #
    # Training loop: 1F1B with CrossMeshP2PCommunication.                 #
    # ------------------------------------------------------------------ #
    stage_module = stage_module.cuda()
    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)
    p2p = CrossMeshP2PCommunication(plan, stage_manager)
    loss_fn = nn.MSELoss()

    act_shape = (BATCH_SIZE, SEQ_LEN, HIDDEN)
    act_dtype = torch.float32

    is_first = stage_manager.is_first_stage()
    is_last = stage_manager.is_last_stage()

    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        # ---------- FORWARD ----------
        if is_first:
            x = torch.randn(*act_shape, dtype=act_dtype, device="cuda",
                            requires_grad=True)
            out = stage_module(x)
            p2p.send_forward(out)
            saved_input = x
            saved_output = out

        elif is_last:
            recv, _ = p2p.recv_forward(act_shape, act_dtype)
            recv = recv.requires_grad_(True)
            out = stage_module(recv)
            target = torch.zeros_like(out)
            loss = loss_fn(out, target)
            saved_input = recv
            saved_output = out

        else:
            recv, _ = p2p.recv_forward(act_shape, act_dtype)
            recv = recv.requires_grad_(True)
            out = stage_module(recv)
            p2p.send_forward(out)
            saved_input = recv
            saved_output = out

        # ---------- BACKWARD ----------
        if is_last:
            loss.backward()
            p2p.send_backward(saved_input.grad)
            logger.info(
                f"Step {step+1}/{NUM_STEPS}  loss={loss.item():.4f}",
                ranks=[rank],
            )

        elif is_first:
            grad, _ = p2p.recv_backward(act_shape, act_dtype)
            saved_output.backward(grad)
            optimizer.step()

        else:
            grad, _ = p2p.recv_backward(act_shape, act_dtype)
            saved_output.backward(grad)
            p2p.send_backward(saved_input.grad)
            optimizer.step()

        torch.distributed.barrier()

    if rank == 0:
        logger.info(
            "Done. Phase 3 variable-size stage demo completed successfully.",
            ranks=[0],
        )


if __name__ == "__main__":
    main()
