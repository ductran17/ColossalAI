"""
GPT2 3D parallel training test with ColossalAI HybridParallelPlugin.
Tests tensor parallel (TP) + pipeline parallel (PP) + data parallel (DP)
simultaneously — true 3D parallelism.

ColossalAI rank layout with pp=2, dp=2, tp=2 (default, 8 GPUs):
  Mesh axes: (PP=0, DP=1, TP=2)  ← ProcessGroupMesh(pp, dp, tp) order

  rank | pp | dp | tp | node
  -----|----|----|----|-----------
    0  |  0 |  0 |  0 | node18 GPU0
    1  |  0 |  0 |  1 | node18 GPU1   ← TP pair (intra-node, fast)
    2  |  0 |  1 |  0 | node20 GPU0
    3  |  0 |  1 |  1 | node20 GPU1   ← TP pair (intra-node, fast)
    4  |  1 |  0 |  0 | node20 GPU2
    5  |  1 |  0 |  1 | node20 GPU3   ← TP pair (intra-node, fast)
    6  |  1 |  1 |  0 | node16 GPU0
    7  |  1 |  1 |  1 | node16 GPU1   ← TP pair (intra-node, fast)

  PP communication (cross-node, stage boundary):
    stage 0 → stage 1:  {0,1} → {4,5}  (node18 → node20)
                         {2,3} → {6,7}  (node20 → node16)

  DP communication (cross-node, gradient sync):
    dp_group within stage 0:  {0,1} ↔ {2,3}  (node18 ↔ node20)
    dp_group within stage 1:  {4,5} ↔ {6,7}  (node20 ↔ node16)

  TP communication (intra-node, fast NVLink/PCIe):
    tp_group stage 0 dp 0:  {0,1}  all within node18
    tp_group stage 0 dp 1:  {2,3}  all within node20
    tp_group stage 1 dp 0:  {4,5}  all within node20
    tp_group stage 1 dp 1:  {6,7}  all within node16

Run:
  bash launch_3nodes.sh --hybrid                         # pp=2 tp=2 dp=2
  bash launch_3nodes.sh --hybrid --pp 4 --tp 2           # dp=1, no DP
  bash launch_3nodes.sh --hybrid --pp 2 --tp 1           # dp=4, no TP
"""

import argparse
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.normpath(os.path.join(_here, "..", "..", "..", "..", ".."))
for p in (_root, _here):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import transformers

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import HybridParallelPlugin
from colossalai.logging import disable_existing_loggers, get_dist_logger


def parse_args():
    p = argparse.ArgumentParser()
    # Default: pp=2, tp=2  →  dp = 8/(2*2) = 2  →  true 3D parallel
    p.add_argument("--pp",          type=int, default=2,  help="Pipeline parallel degree")
    p.add_argument("--tp",          type=int, default=2,  help="Tensor parallel degree")
    p.add_argument("--layers",      type=int, default=4,  help="Number of GPT2 transformer layers")
    p.add_argument("--hidden",      type=int, default=256)
    p.add_argument("--heads",       type=int, default=4)
    p.add_argument("--seq",         type=int, default=64)
    p.add_argument("--batch",       type=int, default=2,  help="Microbatch size (per DP replica)")
    p.add_argument("--microbatches",type=int, default=2,  help="Number of PP microbatches")
    p.add_argument("--steps",       type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    disable_existing_loggers()
    colossalai.launch_from_torch()
    logger = get_dist_logger()

    rank       = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # ------------------------------------------------------------------ #
    # Compute and verify DP degree.                                        #
    # ------------------------------------------------------------------ #
    if args.pp * args.tp > world_size or world_size % (args.pp * args.tp) != 0:
        raise ValueError(
            f"pp={args.pp} × tp={args.tp} = {args.pp*args.tp} does not divide "
            f"world_size={world_size}. Adjust --pp / --tp."
        )
    dp = world_size // (args.pp * args.tp)

    if rank == 0:
        logger.info(
            f"3D Parallel — pp={args.pp}  tp={args.tp}  dp={dp}  "
            f"(world={world_size} GPUs, layers={args.layers}, "
            f"hidden={args.hidden}, seq={args.seq}, "
            f"microbatch={args.batch}×{args.microbatches}={args.batch*args.microbatches} tokens/step/dp-replica)",
            ranks=[0],
        )
        if dp == 1:
            logger.info(
                "  Note: dp=1, data parallelism is OFF. "
                "Use --pp 2 --tp 2 for true 3D with 8 GPUs.",
                ranks=[0],
            )

    # ------------------------------------------------------------------ #
    # Build model on CPU. Booster.boost() shards and moves it to GPU.     #
    # ------------------------------------------------------------------ #
    config = transformers.GPT2Config(
        n_positions=args.seq,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.hidden,
        n_inner=args.hidden * 4,
        vocab_size=1024,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = transformers.GPT2LMHeadModel(config)

    # ------------------------------------------------------------------ #
    # HybridParallelPlugin — axes are (PP, DP, TP) in rank order.         #
    # ShardFormer applies GPT2 TP sharding automatically.                  #
    # DDP wraps the model for DP gradient synchronisation when dp > 1.    #
    # ------------------------------------------------------------------ #
    plugin = HybridParallelPlugin(
        pp_size=args.pp,
        tp_size=args.tp,
        num_microbatches=args.microbatches,
        enable_all_optimization=False,
        precision="fp32",
    )
    booster  = Booster(plugin=plugin)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model, optimizer, *_ = booster.boost(model, optimizer)

    # ------------------------------------------------------------------ #
    # Batch generation.                                                    #
    # Each DP replica must receive DIFFERENT data — that is the whole      #
    # point of data parallelism.  We seed per (step, dp_rank) so runs     #
    # are reproducible and clearly show the DP dimension is active.        #
    #                                                                      #
    # In production you'd use a DistributedSampler; here we derive the    #
    # DP rank from the global rank using the (PP, DP, TP) mesh layout.    #
    # ------------------------------------------------------------------ #
    # Mesh: rank = pp_rank*(dp*tp) + dp_rank*tp + tp_rank
    # → dp_rank = (rank // tp) % dp
    tp = args.tp
    dp_rank = (rank // tp) % dp

    def make_batch(step: int):
        """Generate reproducible but DP-rank-distinct token batches."""
        g = torch.Generator()
        g.manual_seed(step * 1000 + dp_rank)   # different seed per DP replica
        ids = torch.randint(0, config.vocab_size, (args.batch, args.seq), generator=g)
        # attention_mask is required by ColossalAI's GPT2 pipeline forward;
        # all-ones = no padding (every token is attended to).
        mask = torch.ones(args.batch, args.seq, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask, "labels": ids}

    # criterion(model_output, input_batch) → loss
    # GPT2LMHeadModel computes CE loss internally when labels are present.
    def criterion(outputs, _inputs):
        return outputs.loss

    # ------------------------------------------------------------------ #
    # Training loop.                                                       #
    # execute_pipeline():                                                   #
    #   • splits the batch into num_microbatches                           #
    #   • runs the 1F1B pipeline schedule across PP stages                 #
    #   • DDP gradient sync happens automatically inside optimizer.step()  #
    # ------------------------------------------------------------------ #
    for step in range(args.steps):
        batch = make_batch(step)

        outputs = booster.execute_pipeline(
            iter([batch]),
            model,
            criterion=criterion,
            optimizer=optimizer,
            return_loss=True,
        )

        optimizer.step()
        optimizer.zero_grad()

        # Loss is only available on the last PP stage; only rank 0 of that
        # stage logs it (ColossalAI sets loss=None on non-last-stage ranks).
        if outputs.get("loss") is not None:
            logger.info(
                f"Step {step+1}/{args.steps}  loss={outputs['loss']:.4f}  "
                f"[rank={rank} pp={rank//(dp*tp)} dp={dp_rank} tp={rank%tp}]",
                ranks=[rank],
            )

        torch.distributed.barrier()

    if rank == 0:
        logger.info(
            f"Done. 3D parallel (pp={args.pp} tp={args.tp} dp={dp}) "
            "test completed successfully.",
            ranks=[0],
        )


if __name__ == "__main__":
    main()