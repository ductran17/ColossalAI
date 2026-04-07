"""
Heterogeneous TP demo — 4 GPUs, Phase 2 code path verification.

WHY HETERO TP CANNOT BE AUTO-FOUND ON 4 GPUs:
    alpa_dp tracks a device budget. For pp=2 on 4 GPUs each stage consumes exactly
    2 devices. The only 2-device submesh is (1,2) → both stages forced to tp=2.

    Even with a manually crafted plan (tp=[1,2]), there is a TOPOLOGY BUG:
      Stage 0 (tp=1, dp=2): ranks [0,1] are dp REPLICAS → process DIFFERENT data.
      Stage 1 (tp=2, dp=1): ranks [2,3] are TP PARTNERS → must process SAME data.
      P2P: rank 0 → rank 2 (different data), rank 1 → rank 3 (different data).
      Result: TP partners rank 2 and rank 3 hold DIFFERENT activations → AllReduce wrong.

    For correct hetero TP you need 8+ GPUs where:
      Stage 0 (tp=4, 4dev): all 4 ranks process SAME data (TP group) → same output.
      Stage 1 (tp=2, 4dev): splits into 2×TP-pairs × 2×DP-replicas.
      Each TP-pair gets the same output from its paired tp ranks in stage 0 ✓.

WHAT THIS DEMO DOES (4 GPU, valid configuration):
    Uses pp=2, tp=[2,2] (uniform, hetero_tp=False) with dp=1.
    This is the only correct topology for 4 GPUs.
    Exercises the full Phase 2 infrastructure:
      - Phase 2 build_pipeline_plan path
      - Per-stage DeviceMesh with tp_per_stage
      - Stage boundary module creation (no-ops since tp is uniform)
      - Pipeline P2P communication with ColossalAI Megatron-style TP

NOTE ON BoundaryAllGather/BoundarySplit:
    ColossalAI uses Megatron-style TP: stage INPUT and OUTPUT are always FULL
    tensors (AllReduce is applied at end of each TP block). Boundary resharding
    primitives are no-ops for this TP style.
    They would be active for non-Megatron TP where stage output IS sharded
    (column-parallel without AllReduce at end), which requires a custom model.

RUN:
    torchrun --nproc_per_node=4 run_hetero_demo_4gpu.py
    torchrun --nproc_per_node=4 run_hetero_demo_4gpu.py --layers 8 --steps 5

FOR ACTUAL AUTO HETERO TP:
    torchrun --nproc_per_node=8 run_3d_auto_parallel.py --hetero
    → auto-selects pp=2, tp=[4,2] (Stage 0: (1,4) submesh; Stage 1: (2,2) submesh)
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_colossalai_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", ".."))
if _colossalai_root not in sys.path:
    sys.path.insert(0, _colossalai_root)
if _here not in sys.path:
    sys.path.insert(0, _here)

import torch
import torch.nn as nn
import transformers
from gpt_modules import GPT2Block

import colossalai
from colossalai.auto_parallel.pipeline_shard import PipelinePlan, autoparallelize_with_pp
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.pipeline.p2p import PipelineP2PCommunication


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--batch",  type=int, default=2)
    p.add_argument("--seq",    type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--heads",  type=int, default=4)
    p.add_argument("--steps",  type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    disable_existing_loggers()
    colossalai.launch_from_torch()
    logger = get_dist_logger()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    assert world_size == 4, f"This demo requires exactly 4 GPUs, got {world_size}."
    assert args.layers >= 2 and args.layers % 2 == 0, "--layers must be even and >= 2."

    # ------------------------------------------------------------------ #
    # Model                                                                #
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
    layers = [GPT2Block(config, layer_idx=i) for i in range(args.layers)]
    meta_args = {"hidden_states": torch.empty(args.batch, args.seq, args.hidden, device="meta")}

    half = args.layers // 2

    # ------------------------------------------------------------------ #
    # Valid plan for 4 GPUs: pp=2, tp=[2,2], dp=1                         #
    #   pp * tp * dp = 2 * 2 * 1 = 4 = world_size ✓                       #
    #   Stage 0 (tp=2): ranks [0,1] → TP partners, process same data      #
    #   Stage 1 (tp=2): ranks [2,3] → TP partners, process same data      #
    #   P2P: rank 0 → rank 2 (tp rank 0 to tp rank 0) ✓                   #
    #        rank 1 → rank 3 (tp rank 1 to tp rank 1) ✓                   #
    #                                                                      #
    # This is uniform TP (hetero_tp=False) because tp degrees are equal.  #
    # hetero_tp=True requires 8+ GPUs: pp=2, tp=[4,2] where both stages   #
    # have 4 devices but different TP (auto-found by --hetero on 8 GPUs). #
    # ------------------------------------------------------------------ #
    plan = PipelinePlan(
        stage_layer_ranges=[(0, half), (half, args.layers)],
        submesh_per_stage=[(1, 2), (1, 2)],   # both tp=2
        tp_per_stage=[2, 2],
        pp_size=2,
        tp_size=2,
        dp_size=1,
        estimated_cost=0.0,
        heterogeneous_tp=False,  # uniform tp on 4 GPUs
    )

    if rank == 0:
        logger.info(
            f"[Phase 2 Demo] world={world_size}, layers={args.layers}, "
            f"batch={args.batch}, seq={args.seq}, hidden={args.hidden}",
            ranks=[0],
        )
        logger.info(
            f"Plan: pp=2, tp=[2,2], dp=1  (uniform — only valid topology for 4 GPUs)",
            ranks=[0],
        )
        logger.info(
            f"  Stage 0: layers[0:{half}]  → tp=2, dp=1  (ranks 0,1 tensor-parallel)",
            ranks=[0],
        )
        logger.info(
            f"  Stage 1: layers[{half}:{args.layers}] → tp=2, dp=1  (ranks 2,3 tensor-parallel)",
            ranks=[0],
        )
        logger.info(
            "  Note: hetero_tp=True (different TP) requires 8 GPUs where "
            "(1,4) and (2,2) submeshes both use 4 devices with different TP.",
            ranks=[0],
        )

    # ------------------------------------------------------------------ #
    # Shard + create boundary modules                                      #
    # Skip planning (plan= supplied) → goes to Step 4..7 directly.        #
    # ------------------------------------------------------------------ #
    stage_module, stage_manager, plan = autoparallelize_with_pp(
        layers=layers,
        meta_args=meta_args,
        num_microbatches=args.batch,
        heterogeneous_tp=False,   # uniform TP — only correct topology for 4 GPUs
        plan=plan,
    )

    current_stage = stage_manager.stage
    is_first = stage_manager.is_first_stage()
    is_last  = stage_manager.is_last_stage()

    if rank == 0:
        logger.info(
            f"Phase 2 infrastructure active: pp={plan.pp_size}, "
            f"tp_per_stage={plan.tp_per_stage}, hetero={plan.heterogeneous_tp}",
            ranks=[0],
        )

    # ------------------------------------------------------------------ #
    # Training loop — simplified 1F1B (one microbatch)                    #
    # No boundary resharding: ColossalAI Megatron-style TP always has     #
    # full tensors [B,S,H] at stage boundaries (AllReduce inside block).  #
    # BoundaryAllGather/Split would activate for non-Megatron sharded TP. #
    # ------------------------------------------------------------------ #
    stage_module = stage_module.cuda()
    optimizer = torch.optim.Adam(stage_module.parameters(), lr=1e-4)
    p2p = PipelineP2PCommunication(stage_manager, overlap_p2p=False)
    loss_fn = nn.MSELoss()

    # Boundary modules are None since tp_per_stage=[2,2] is uniform.
    send_mod = plan.send_boundary_modules.get(current_stage)
    recv_mod = plan.recv_boundary_modules.get(current_stage)

    for step in range(args.steps):
        optimizer.zero_grad()

        # ---------- FORWARD ----------
        if is_first:
            x = torch.randn(args.batch, args.seq, args.hidden, device="cuda", requires_grad=True)
            out = stage_module(x)
            # send_mod is None (uniform TP, no AllGather needed at boundary)
            if send_mod is not None:
                out = send_mod(out)
            p2p.send_forward(out)
            saved_input  = x
            saved_output = out

        else:  # is_last
            recv, _ = p2p.recv_forward()
            recv = recv.requires_grad_(True)
            # recv_mod is None for Megatron TP: stage expects full [B,S,H] input.
            # (BoundarySplit would break here — GPT2Block LayerNorm needs full hidden.)
            out = stage_module(recv)
            target = torch.zeros_like(out)
            loss   = loss_fn(out, target)
            saved_input  = recv
            saved_output = out

        # ---------- BACKWARD ----------
        if is_last:
            loss.backward()
            p2p.send_backward(saved_input.grad)
            logger.info(
                f"Step {step+1}/{args.steps}  loss={loss.item():.4f}  "
                f"[rank {rank}, stage {current_stage}]",
                ranks=[rank],
            )
        else:  # is_first
            grad, _ = p2p.recv_backward()
            saved_output.backward(grad)
            optimizer.step()

        if not is_last:
            optimizer.step()

        torch.distributed.barrier()

    if rank == 0:
        logger.info("Done. Phase 2 pipeline demo completed successfully.", ranks=[0])
        logger.info(
            "This ran pp=2, tp=[2,2] on 4 GPUs (uniform TP, the only valid 4-GPU topology).\n"
            "For actual hetero TP auto-selection run on 8 GPUs:\n"
            "  torchrun --nproc_per_node=8 run_3d_auto_parallel.py --hetero\n"
            "  → auto-selects pp=2, tp=[4,2]: Stage 0 uses (1,4) submesh, Stage 1 uses (2,2) submesh.",
            ranks=[0],
        )


if __name__ == "__main__":
    main()
