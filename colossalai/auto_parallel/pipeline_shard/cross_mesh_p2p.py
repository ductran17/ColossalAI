# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

"""Cross-mesh P2P communication for Phase 3 variable-size pipeline stages.

In Phase 3, adjacent pipeline stages may own different numbers of devices
(e.g. stage 0: tp=1/1 GPU, stage 1: tp=2/2 GPUs). Standard PipelineP2PCommunication
assumes a uniform ProcessGroupMesh layout and cannot handle this.

CrossMeshP2PCommunication uses dist.broadcast collectives instead of
point-to-point sends, exploiting the fact that Megatron-style TP produces
FULL (non-sharded) tensors at stage boundaries — every TP rank in a stage
holds an identical copy of the activation tensor.

Layout convention (matching DeviceMesh in orchestrator.py):
    stage_ranks.reshape(tp_s, dp_s)
    → element [tp_rank][dp_rank] = global rank

For the forward direction (stage s → stage s+1), per dp-replica r:
  * sender_rep  = stage_s_ranks[0 * dp_s + r]   (tp_rank=0, dp_rank=r)
  * recv_ranks  = [stage_{s+1}_ranks[i * dp_{s+1} + r] for i in range(tp_{s+1})]
  * group       = all sender TP ranks for dp_r  +  recv_ranks
  * collective  = dist.broadcast(tensor, src=sender_rep, group=group)
    - ALL sender TP ranks (which hold the same full tensor) participate.
    - ALL receiver TP ranks receive the same full tensor — exactly what
      Megatron-style TP requires as input for the next stage.

Backward direction (stage s+1 → stage s): mirror of the above with
  * backward_src = stage_{s+1}_ranks[0 * dp_{s+1} + r]

Usage in training loop::

    p2p = CrossMeshP2PCommunication(plan, stage_manager)

    # Forward pass on sender stage:
    output = stage_module(input)
    p2p.send_forward(output)

    # Forward pass on receiver stage:
    recv_input, _ = p2p.recv_forward(recv_shape=(B, S, H), recv_dtype=torch.float16)
    output = stage_module(recv_input)

    # Backward: receiver sends grad to sender
    p2p.send_backward(grad_output)
    recv_grad, _ = p2p.recv_backward(recv_shape=(B, S, H), recv_dtype=torch.float16)
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from colossalai.auto_parallel.pipeline_shard.orchestrator import (
    PipelinePlan,
    VariableStagePipelineManager,
)

__all__ = ["CrossMeshP2PCommunication"]


class CrossMeshP2PCommunication:
    """Broadcast-based pipeline P2P for variable-size stages (Phase 3).

    At construction time, ALL ranks in the world build the same set of
    process groups (one per dp-replica per adjacent stage pair) to satisfy
    PyTorch's requirement that dist.new_group() is called identically on
    every rank.

    Args:
        plan: The Phase 3 PipelinePlan (must have variable_stage_sizes=True).
        stage_manager: VariableStagePipelineManager for this rank.
    """

    def __init__(
        self,
        plan: PipelinePlan,
        stage_manager: VariableStagePipelineManager,
    ) -> None:
        assert plan.variable_stage_sizes, (
            "CrossMeshP2PCommunication requires a Phase 3 plan "
            "(variable_stage_sizes=True)."
        )
        self._plan = plan
        self._sm = stage_manager
        self._rank = dist.get_rank()
        self._stage = stage_manager.stage
        pp = plan.pp_size
        dp = plan.dp_per_stage[0]  # uniform dp across all stages

        # ------------------------------------------------------------------
        # Build one process group per (adjacent stage pair, dp replica).
        # groups[s][r] covers stage pair (s, s+1) and dp replica r.
        # ALL ranks call dist.new_group() for every group.
        # ------------------------------------------------------------------
        # groups[s][r]    — dist.ProcessGroup for pair (s, s+1), dp replica r
        # fwd_srcs[s][r]  — global rank that is root for forward broadcast
        # bwd_srcs[s][r]  — global rank that is root for backward broadcast
        self._groups: List[List[dist.ProcessGroup]] = []
        self._fwd_srcs: List[List[int]] = []
        self._bwd_srcs: List[List[int]] = []

        for s in range(pp - 1):
            rr_s = plan.rank_ranges[s]
            rr_s1 = plan.rank_ranges[s + 1]
            tp_s = plan.tp_per_stage[s]
            tp_s1 = plan.tp_per_stage[s + 1]
            dp_s = plan.dp_per_stage[s]
            dp_s1 = plan.dp_per_stage[s + 1]

            pair_groups: List[dist.ProcessGroup] = []
            pair_fwd_srcs: List[int] = []
            pair_bwd_srcs: List[int] = []

            for r in range(dp):
                # All TP ranks for dp_replica r in stage s
                sender_tp_ranks = [rr_s[i * dp_s + r] for i in range(tp_s)]
                # All TP ranks for dp_replica r in stage s+1
                recv_tp_ranks = [rr_s1[i * dp_s1 + r] for i in range(tp_s1)]

                group_ranks = sender_tp_ranks + recv_tp_ranks
                group = dist.new_group(group_ranks)
                pair_groups.append(group)

                # Forward root: tp_rank=0 of stage s for dp_replica r
                pair_fwd_srcs.append(rr_s[0 * dp_s + r])   # = rr_s[r]
                # Backward root: tp_rank=0 of stage s+1 for dp_replica r
                pair_bwd_srcs.append(rr_s1[0 * dp_s1 + r]) # = rr_s1[r]

            self._groups.append(pair_groups)
            self._fwd_srcs.append(pair_fwd_srcs)
            self._bwd_srcs.append(pair_bwd_srcs)

        # ------------------------------------------------------------------
        # Precompute this rank's dp_rank within its stage for fast lookup.
        # Layout: stage_ranks.reshape(tp_s, dp_s) → rank at [tp_rank][dp_rank]
        # local_idx = i * dp_s + r  →  dp_rank = local_idx % dp_s
        # ------------------------------------------------------------------
        rr = plan.rank_ranges[self._stage]
        dp_s = plan.dp_per_stage[self._stage]
        local_idx = rr.index(self._rank)
        self._dp_rank: int = local_idx % dp_s

    # ------------------------------------------------------------------ #
    # Forward direction: stage s  →  stage s+1                            #
    # ------------------------------------------------------------------ #

    def send_forward(self, tensor: torch.Tensor) -> List:
        """Send activation to the next stage.

        All TP ranks in the sender stage call this. The tp_rank=0 rank is
        root; all other TP ranks participate but are overwritten with the
        same data (harmless since they hold identical activations).

        Args:
            tensor: The activation tensor to send (full tensor, not sharded).

        Returns:
            Empty list (no async handles; broadcast is synchronous).
        """
        s = self._stage
        assert not self._sm.is_last_stage(), (
            "send_forward called on the last pipeline stage."
        )
        r = self._dp_rank
        dist.broadcast(tensor, src=self._fwd_srcs[s][r], group=self._groups[s][r])
        return []

    def recv_forward(
        self,
        recv_shape: Tuple[int, ...],
        recv_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, List]:
        """Receive activation from the previous stage.

        All TP ranks in the receiver stage call this and receive the same
        full activation tensor via broadcast.

        Args:
            recv_shape: Shape of the tensor to receive, e.g. (B, S, H).
            recv_dtype: Data type of the tensor to receive.

        Returns:
            (recv_tensor, [])  — the received tensor and an empty handle list.
        """
        s = self._stage
        assert not self._sm.is_first_stage(), (
            "recv_forward called on the first pipeline stage."
        )
        r = self._dp_rank
        recv_tensor = torch.empty(recv_shape, dtype=recv_dtype, device="cuda")
        dist.broadcast(recv_tensor, src=self._fwd_srcs[s - 1][r],
                       group=self._groups[s - 1][r])
        return recv_tensor, []

    # ------------------------------------------------------------------ #
    # Backward direction: stage s+1  →  stage s                           #
    # ------------------------------------------------------------------ #

    def send_backward(self, grad: torch.Tensor) -> List:
        """Send gradient to the previous stage.

        Args:
            grad: The gradient tensor (full, not sharded).

        Returns:
            Empty list.
        """
        s = self._stage  # this rank is in stage s+1
        assert not self._sm.is_first_stage(), (
            "send_backward called on the first pipeline stage."
        )
        r = self._dp_rank
        dist.broadcast(grad, src=self._bwd_srcs[s - 1][r],
                       group=self._groups[s - 1][r])
        return []

    def recv_backward(
        self,
        recv_shape: Tuple[int, ...],
        recv_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, List]:
        """Receive gradient from the next stage.

        Args:
            recv_shape: Shape of the gradient tensor to receive.
            recv_dtype: Data type.

        Returns:
            (recv_grad, [])
        """
        s = self._stage
        assert not self._sm.is_last_stage(), (
            "recv_backward called on the last pipeline stage."
        )
        r = self._dp_rank
        recv_grad = torch.empty(recv_shape, dtype=recv_dtype, device="cuda")
        dist.broadcast(recv_grad, src=self._bwd_srcs[s][r],
                       group=self._groups[s][r])
        return recv_grad, []
