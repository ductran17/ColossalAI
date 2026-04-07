# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
End-to-end test for auto 3D parallel training.

Phase 1 — Uniform TP (default):
    All pipeline stages share one TP degree. No boundary resharding needed.

    torchrun --nproc_per_node=2 run_3d_auto_parallel.py            # auto-search
    torchrun --nproc_per_node=2 run_3d_auto_parallel.py --tp 2     # force TP=2

Phase 2 — Heterogeneous TP (--hetero flag):
    Different stages may use different TP degrees. BoundaryReshardingModules
    handle activation resharding at TP-mismatched stage boundaries.
    Requires 4+ GPUs to observe actual TP transitions.

    torchrun --nproc_per_node=4 run_3d_auto_parallel.py --hetero

Flags:
    --tp <int>     Fix TP degree (default: auto-search). Phase 1 only.
    --hetero       Enable Phase 2 heterogeneous TP search.
    --layers <int> Number of GPT2Block layers (default: 4).
    --batch <int>  Batch size (default: 2).
    --seq <int>    Sequence length (default: 64).
    --steps <int>  Number of training steps (default: 3).
    --hidden <int> Hidden dimension size (default: 256).
    --heads <int>  Number of attention heads (default: 4).
    --cache <str>  Path prefix for compute cost cache (default: /tmp/auto3d_cache).
"""

import argparse
import os
import sys

# Add ColossalAI root (5 levels up from this file's directory) to sys.path so
# that `import colossalai` works regardless of how the script is launched.
_here = os.path.dirname(os.path.abspath(__file__))
_colossalai_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", ".."))
if _colossalai_root not in sys.path:
    sys.path.insert(0, _colossalai_root)
# Also allow importing gpt_modules from same directory.
if _here not in sys.path:
    sys.path.insert(0, _here)

import torch
import torch.nn as nn
import transformers
from gpt_modules import GPT2Block

import colossalai
from colossalai.auto_parallel.pipeline_shard import PipelinePlan, autoparallelize_with_pp, build_pipeline_plan
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.pipeline.p2p import PipelineP2PCommunication


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=None, help="Fix TP degree (Phase 1 only).")
    p.add_argument("--hetero", action="store_true", help="Phase 2: heterogeneous TP search.")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--cache", type=str, default="/tmp/auto3d_gpt_cache")
    return p.parse_args()


def main():
    args = parse_args()
    disable_existing_loggers()
    colossalai.launch_from_torch()
    logger = get_dist_logger()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # ------------------------------------------------------------------ #
    # Build the transformer layers (GPT2Block handles attention + MLP).   #
    # ------------------------------------------------------------------ #
    config = transformers.GPT2Config(
        n_positions=args.seq,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.hidden,
        n_inner=args.hidden * 4,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    # Create all layers on CPU first; initialize_model will move them to GPU.
    layers = [GPT2Block(config, layer_idx=i) for i in range(args.layers)]

    # meta_args: shape description of a single layer's input.
    # hidden_states shape: (batch, seq_len, hidden_dim)
    meta_args = {
        "hidden_states": torch.empty(args.batch, args.seq, args.hidden, device="meta"),
    }

    if rank == 0:
        mode_str = "hetero-TP (Phase 2)" if args.hetero else "uniform-TP (Phase 1)"
        logger.info(
            f"Auto 3D parallel [{mode_str}]: {world_size} GPUs, {args.layers} layers, "
            f"batch={args.batch}, seq={args.seq}, hidden={args.hidden}",
            ranks=[0],
        )

    # ------------------------------------------------------------------ #
    # Run the auto planner + sharding.                                    #
    # autoparallelize_with_pp() profiles α/β, calls alpa_dp, then        #
    # shards each stage's layers with the optimal TP+DP strategy.         #
    # In Phase 2 (--hetero), boundary modules are also created.           #
    # ------------------------------------------------------------------ #
    stage_module, stage_manager, plan = autoparallelize_with_pp(
        layers=layers,
        meta_args=meta_args,
        num_microbatches=args.batch,
        uniform_tp_degree=args.tp,
        heterogeneous_tp=args.hetero,
        cache_path=args.cache,
    )

    if rank == 0:
        logger.info(
            f"Plan: pp={plan.pp_size}, tp={plan.tp_size}, dp={plan.dp_size}, "
            f"hetero={plan.heterogeneous_tp}, "
            f"estimated_cost={plan.estimated_cost:.4f}s",
            ranks=[0],
        )
        for i, (start, end) in enumerate(plan.stage_layer_ranges):
            tp_s = plan.tp_per_stage[i] if plan.tp_per_stage else plan.tp_size
            logger.info(f"  Stage {i}: layers[{start}:{end}], tp={tp_s}", ranks=[0])

    # ------------------------------------------------------------------ #
    # Training loop.                                                       #
    # For PP=1: no pipeline, just forward + backward normally.            #
    # For PP>1: manual 1F1B micro-step using PipelineP2PCommunication.    #
    # ------------------------------------------------------------------ #
    stage_module = stage_module.cuda()
    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)
    p2p = PipelineP2PCommunication(stage_manager, overlap_p2p=False)

    loss_fn = nn.MSELoss()

    for step in range(args.steps):
        optimizer.zero_grad()

        if plan.pp_size == 1:
            # No pipeline: standard forward + backward on a single stage.
            x = torch.randn(args.batch, args.seq, args.hidden, device="cuda")
            out = stage_module(x)
            target = torch.zeros_like(out)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            if rank == 0:
                logger.info(f"Step {step+1}/{args.steps}  loss={loss.item():.4f}", ranks=[0])

        else:
            # Pipeline: simplified single-microbatch 1F1B.
            current_stage = stage_manager.stage
            is_first = stage_manager.is_first_stage()
            is_last = stage_manager.is_last_stage()

            # Boundary modules for Phase 2 (None in Phase 1).
            # send_mod: applied to this stage's output BEFORE P2P send.
            # recv_mod: applied to received activation AFTER P2P recv.
            send_mod = plan.send_boundary_modules.get(current_stage)
            recv_mod = plan.recv_boundary_modules.get(current_stage)

            # ---------- FORWARD PASS ----------
            if is_first:
                x = torch.randn(args.batch, args.seq, args.hidden, device="cuda", requires_grad=True)
                out = stage_module(x)
                # Phase 2: AllGather shards before sending so receiver gets full tensor.
                if send_mod is not None:
                    out = send_mod(out)
                p2p.send_forward(out)
                saved_input = x
                saved_output = out

            elif is_last:
                recv, _ = p2p.recv_forward()
                recv = recv.requires_grad_(True)
                # Phase 2: Split full tensor to this rank's TP shard after recv.
                act = recv_mod(recv) if recv_mod is not None else recv
                out = stage_module(act)
                target = torch.zeros_like(out)
                loss = loss_fn(out, target)
                saved_input = recv   # grad of recv (before split) goes back via P2P
                saved_output = out

            else:
                recv, _ = p2p.recv_forward()
                recv = recv.requires_grad_(True)
                act = recv_mod(recv) if recv_mod is not None else recv
                out = stage_module(act)
                if send_mod is not None:
                    out = send_mod(out)
                p2p.send_forward(out)
                saved_input = recv
                saved_output = out

            # ---------- BACKWARD PASS ----------
            if is_last:
                loss.backward()
                # BoundarySplit.backward ran automatically → recv.grad is zero-padded full tensor.
                p2p.send_backward(saved_input.grad)
                if rank == torch.distributed.get_world_size() - 1:
                    logger.info(
                        f"Step {step+1}/{args.steps}  loss={loss.item():.4f}",
                        ranks=[rank],
                    )

            elif is_first:
                grad, _ = p2p.recv_backward()
                # BoundaryAllGather.backward runs automatically → extracts this rank's grad shard.
                saved_output.backward(grad)
                optimizer.step()

            else:
                grad, _ = p2p.recv_backward()
                saved_output.backward(grad)
                p2p.send_backward(saved_input.grad)

            if not is_last:
                optimizer.step()

        torch.distributed.barrier()

    if rank == 0:
        logger.info("Done. 3D auto parallel test completed successfully.", ranks=[0])


if __name__ == "__main__":
    main()
