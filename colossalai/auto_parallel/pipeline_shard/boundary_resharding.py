# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Phase 2 — Heterogeneous TP boundary resharding.

When adjacent pipeline stages use different TP degrees, activations must be
resharded at stage boundaries so each rank receives its correct shard.

Two ops are injected:

  before_send (end of stage s, sender side):
      BoundaryAllGather — when sender_tp > 1, gather all shards so every
      sender rank holds the full activation tensor before P2P send.

  after_recv (start of stage s+1, receiver side):
      BoundarySplit — when receiver_tp > 1, split the received full tensor
      so each rank keeps only its local TP shard.

Autograd support:
  BoundaryAllGather backward = chunk (take this rank's gradient slice)
  BoundarySplit backward     = zero-pad (gradient only came from this shard)

These are symmetric — AllGather.forward + Split.backward cancel out, and
Split.forward + AllGather.backward cancel out, giving correct gradient flow
through the pipeline P2P communication.

Usage in training loop (pp > 1, heterogeneous_tp = True):

  # ---- forward, sender stage s ----
  out = stage_module(x)
  if send_module is not None:           # BoundaryReshardingModule("before_send")
      out = send_module(out)
  p2p.send_forward(out)
  saved_output = out

  # ---- backward, sender stage s ----
  grad, _ = p2p.recv_backward()
  saved_output.backward(grad)           # BoundaryAllGather.backward runs automatically

  # ---- forward, receiver stage s+1 ----
  recv, _ = p2p.recv_forward()
  recv = recv.requires_grad_(True)
  if recv_module is not None:           # BoundaryReshardingModule("after_recv")
      act = recv_module(recv)
  else:
      act = recv
  out = stage_module(act)

  # ---- backward, receiver stage s+1 ----
  loss.backward()
  p2p.send_backward(recv.grad)          # BoundarySplit.backward ran automatically
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn


class BoundaryAllGather(torch.autograd.Function):
    """All-gather along the hidden (last) dimension across a TP process group.

    Forward:  shard  [B, S, H/T]  →  full  [B, S, H]   (all_gather + cat)
    Backward: grad_full [B, S, H]  →  grad_shard [B, S, H/T]  (chunk[rank])

    Placed at the END of a stage before P2P send when sender_tp > 1.
    After this op every sender rank holds the full activation tensor, so
    the subsequent 1-to-1 P2P send delivers the full tensor to each
    corresponding receiver rank.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_group: dist.ProcessGroup) -> torch.Tensor:
        ctx.tp_group = tp_group
        ctx.world_size = dist.get_world_size(tp_group)
        ctx.rank = dist.get_rank(tp_group)
        # Each rank contributes its shard; gather into a list then cat.
        shards = [torch.empty_like(x) for _ in range(ctx.world_size)]
        dist.all_gather(shards, x.contiguous(), group=tp_group)
        return torch.cat(shards, dim=-1)

    @staticmethod
    def backward(ctx, grad_full: torch.Tensor):
        # Take this rank's slice of the incoming gradient.
        return grad_full.chunk(ctx.world_size, dim=-1)[ctx.rank].contiguous(), None


class BoundarySplit(torch.autograd.Function):
    """Split along the hidden (last) dimension — no communication, pure tensor op.

    Forward:  full  [B, S, H]    →  shard  [B, S, H/T]  (chunk[tp_rank])
    Backward: grad_shard [B, S, H/T]  →  grad_full [B, S, H]  (zero-pad)

    Placed at the START of a stage after P2P recv when receiver_tp > 1.
    Each receiver rank takes only its TP shard of the received full tensor.

    The zero-pad backward ensures the full-size gradient tensor is sent back
    through the pipeline P2P, allowing BoundaryAllGather.backward on the
    sender side to correctly extract its rank's gradient slice.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
        ctx.tp_size = tp_size
        ctx.tp_rank = tp_rank
        ctx.full_hidden = x.shape[-1]
        return x.chunk(tp_size, dim=-1)[tp_rank].contiguous()

    @staticmethod
    def backward(ctx, grad_shard: torch.Tensor):
        # Reconstruct full-size gradient tensor with zeros outside this shard.
        shape = list(grad_shard.shape)
        shape[-1] = ctx.full_hidden
        grad_full = grad_shard.new_zeros(shape)
        shard_size = ctx.full_hidden // ctx.tp_size
        grad_full[..., ctx.tp_rank * shard_size : (ctx.tp_rank + 1) * shard_size] = grad_shard
        return grad_full, None, None


class BoundaryReshardingModule(nn.Module):
    """Resharding op inserted at a pipeline stage boundary where TP degrees differ.

    Two modes:
      "before_send": placed at the output of the sender stage.
          When sender_tp > 1: applies BoundaryAllGather so every sender rank
          holds the full tensor before the P2P send.
          When sender_tp == 1: no-op (tensor is already full).

      "after_recv":  placed at the input of the receiver stage.
          When receiver_tp > 1: applies BoundarySplit so each receiver rank
          takes only its local TP shard after P2P recv.
          When receiver_tp == 1: no-op (receiver expects full tensor).

    A module is only created when sender_tp != receiver_tp. When they are
    equal (uniform TP, Phase 1 behaviour) no resharding is needed and these
    modules are not inserted.

    Args:
        mode: "before_send" or "after_recv".
        sender_tp: TP degree of the stage that sends activations.
        receiver_tp: TP degree of the stage that receives activations.
        tp_group: TP process group of the sender stage.
                  Required only when mode=="before_send" and sender_tp > 1.
        tp_rank: This rank's position in the receiver stage's TP group.
                 Required only when mode=="after_recv" and receiver_tp > 1.
    """

    def __init__(
        self,
        mode: str,
        sender_tp: int,
        receiver_tp: int,
        tp_group: Optional[dist.ProcessGroup] = None,
        tp_rank: int = 0,
    ) -> None:
        super().__init__()
        assert mode in ("before_send", "after_recv"), f"Unknown mode: {mode!r}"
        self.mode = mode
        self.sender_tp = sender_tp
        self.receiver_tp = receiver_tp
        self.tp_group = tp_group
        self.tp_rank = tp_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sender_tp == self.receiver_tp:
            return x  # uniform TP — no resharding needed

        if self.mode == "before_send":
            # Gather shards so every sender rank has the full tensor.
            if self.sender_tp > 1:
                assert self.tp_group is not None, (
                    "BoundaryReshardingModule('before_send') requires tp_group "
                    "when sender_tp > 1."
                )
                return BoundaryAllGather.apply(x, self.tp_group)
            # sender_tp == 1 → already full, nothing to do.
            return x

        else:  # "after_recv"
            # Take this rank's shard from the received full tensor.
            if self.receiver_tp > 1:
                return BoundarySplit.apply(x, self.receiver_tp, self.tp_rank)
            # receiver_tp == 1 → expects full tensor, nothing to do.
            return x

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, sender_tp={self.sender_tp}, "
            f"receiver_tp={self.receiver_tp}, tp_rank={self.tp_rank}"
        )
