"""
Unit tests for Phase 2 boundary resharding components.

Tests cover:
  - BoundaryAllGather: forward shape, backward (gradient) correctness
  - BoundarySplit: forward shape, backward (zero-pad) correctness
  - BoundaryReshardingModule: mode dispatch, uniform-TP no-op
  - get_boundary_cost_table: shape, zero diagonal, AllGather cost formula
  - build_pipeline_plan with heterogeneous_tp: returns PipelinePlan with
    tp_per_stage populated

These tests run without any distributed process groups (single-process).
Distributed ops in BoundaryAllGather are exercised via mock process groups
or by setting tp_group to None and testing BoundarySplit only.
"""
import sys
import os

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

import numpy as np
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from colossalai.auto_parallel.pipeline_shard.boundary_resharding import (
    BoundaryAllGather,
    BoundarySplit,
    BoundaryReshardingModule,
)
from colossalai.auto_parallel.pipeline_shard.compute_cost import get_boundary_cost_table


# ============================================================
# BoundarySplit — no distributed needed
# ============================================================

class TestBoundarySplit:
    def test_forward_shape(self):
        """Split along hidden dim: [B, S, H] → [B, S, H/T]"""
        x = torch.randn(2, 8, 16)
        out = BoundarySplit.apply(x, 4, 0)   # tp_size=4, tp_rank=0
        assert out.shape == (2, 8, 4)        # H/4 = 4

    def test_forward_values(self):
        """Each rank's shard is the correct slice of hidden."""
        x = torch.arange(32, dtype=torch.float32).reshape(1, 1, 32)
        for rank in range(4):
            shard = BoundarySplit.apply(x, 4, rank)
            expected = x[:, :, rank * 8 : (rank + 1) * 8]
            assert torch.equal(shard, expected), f"Mismatch at rank {rank}"

    def test_backward_zeroped(self):
        """BoundarySplit.backward zero-pads gradient to full hidden size."""
        x = torch.randn(2, 4, 16, requires_grad=True)
        shard = BoundarySplit.apply(x, 4, 2)  # take chunk 2 of 4
        loss = shard.sum()
        loss.backward()

        # Gradient should be 1 in chunk[2] and 0 elsewhere.
        grad = x.grad
        assert grad is not None
        assert grad.shape == (2, 4, 16)
        assert torch.all(grad[:, :, :8] == 0.0)   # chunks 0 and 1
        assert torch.all(grad[:, :, 8:12] == 1.0)  # chunk 2
        assert torch.all(grad[:, :, 12:] == 0.0)  # chunk 3

    def test_tp_rank_1(self):
        """Rank 1 of 2 takes the second half."""
        x = torch.ones(1, 1, 8)
        x[:, :, 4:] = 2.0
        shard = BoundarySplit.apply(x, 2, 1)
        assert shard.shape == (1, 1, 4)
        assert torch.all(shard == 2.0)


# ============================================================
# BoundaryAllGather — mock the dist calls
# ============================================================

class TestBoundaryAllGather:
    def _make_fake_group(self, world_size: int, rank: int):
        """Create a mock process group object."""
        pg = MagicMock()
        return pg, world_size, rank

    def test_backward_chunks(self):
        """BoundaryAllGather.backward returns this rank's chunk of grad_full."""
        # Simulate: forward gathered 4 shards of size [1,1,4] → full [1,1,16]
        # Backward: grad_full [1,1,16] → grad_shard [1,1,4] for rank 2
        pg, world_size, rank = self._make_fake_group(world_size=4, rank=2)

        # Manually set up the ctx that backward would receive
        ctx = MagicMock()
        ctx.world_size = 4
        ctx.rank = 2

        grad_full = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
        # backward returns (grad_input, None) — unpack first element
        result = BoundaryAllGather.backward(ctx, grad_full)
        grad_shard = result[0]

        # Rank 2 should get chunk index 2 (elements 8..11)
        expected = grad_full[:, :, 8:12]
        assert torch.equal(grad_shard, expected)

    def test_backward_all_ranks(self):
        """Each rank gets the correct slice."""
        ctx = MagicMock()
        ctx.world_size = 4
        grad_full = torch.randn(2, 8, 16)
        for r in range(4):
            ctx.rank = r
            result = BoundaryAllGather.backward(ctx, grad_full)
            g = result[0]  # first element of (grad_input, None) tuple
            assert g.shape == (2, 8, 4)
            assert torch.equal(g, grad_full[:, :, r * 4 : (r + 1) * 4])


# ============================================================
# BoundaryReshardingModule
# ============================================================

class TestBoundaryReshardingModule:
    def test_uniform_tp_noop(self):
        """No resharding when sender_tp == receiver_tp."""
        x = torch.randn(2, 4, 8)
        mod = BoundaryReshardingModule("before_send", sender_tp=2, receiver_tp=2)
        out = mod(x)
        assert out is x  # exact same object, no copy

    def test_before_send_tp1_sender_noop(self):
        """sender_tp=1 → full tensor already, no AllGather needed."""
        x = torch.randn(2, 4, 8)
        mod = BoundaryReshardingModule("before_send", sender_tp=1, receiver_tp=4)
        out = mod(x)
        assert out is x

    def test_after_recv_tp1_receiver_noop(self):
        """receiver_tp=1 → full tensor expected, no Split needed."""
        x = torch.randn(2, 4, 8)
        mod = BoundaryReshardingModule("after_recv", sender_tp=4, receiver_tp=1)
        out = mod(x)
        assert out is x

    def test_after_recv_split_shape(self):
        """after_recv with receiver_tp=2 splits the hidden dim."""
        x = torch.randn(2, 4, 16)
        mod = BoundaryReshardingModule(
            "after_recv", sender_tp=1, receiver_tp=2, tp_rank=0
        )
        out = mod(x)
        assert out.shape == (2, 4, 8)

    def test_after_recv_split_correct_rank(self):
        """Each receiver rank gets the correct shard."""
        x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
        for rank in range(4):
            mod = BoundaryReshardingModule(
                "after_recv", sender_tp=1, receiver_tp=4, tp_rank=rank
            )
            out = mod(x)
            assert torch.equal(out, x[:, :, rank * 4 : (rank + 1) * 4])

    def test_before_send_requires_group_when_sender_tp_gt1(self):
        """before_send with sender_tp>1 must raise if tp_group is None."""
        mod = BoundaryReshardingModule(
            "before_send", sender_tp=4, receiver_tp=1, tp_group=None
        )
        x = torch.randn(2, 4, 8)
        with pytest.raises(AssertionError):
            mod(x)

    def test_extra_repr(self):
        mod = BoundaryReshardingModule("after_recv", sender_tp=2, receiver_tp=4, tp_rank=1)
        r = mod.extra_repr()
        assert "after_recv" in r
        assert "tp_rank=1" in r


# ============================================================
# get_boundary_cost_table
# ============================================================

class TestGetBoundaryCostTable:
    def _submeshes(self):
        return [(1, 1), (1, 2), (1, 4)]

    def test_shape(self):
        table = get_boundary_cost_table(self._submeshes(), 1024, [1e-5, 1e-5], [1e-11, 1e-11])
        assert table.shape == (3, 3)

    def test_uniform_tp_zero_cost(self):
        """Same submesh → boundary cost = 0."""
        table = get_boundary_cost_table(self._submeshes(), 1024, [1e-5, 1e-5], [1e-11, 1e-11])
        for i in range(3):
            assert table[i, i] == 0.0, f"Diagonal [{i},{i}] should be 0"

    def test_tp1_sender_zero_cost(self):
        """tp1=1 sender → no AllGather → cost = 0 regardless of receiver."""
        table = get_boundary_cost_table(self._submeshes(), 1024, [1e-5, 1e-5], [1e-11, 1e-11])
        # submesh index 0 = (1,1) → tp=1 (sender)
        assert table[0, 1] == 0.0  # tp=1 → tp=2: no AllGather
        assert table[0, 2] == 0.0  # tp=1 → tp=4: no AllGather

    def test_allgather_cost_formula(self):
        """tp_send=2 → AllGather communicates (1 - 1/2) of activation_bytes."""
        alpha = 1e-5
        beta = 1e-11
        activation_bytes = 1024
        # mesh_alpha and mesh_beta are per-axis: [axis0, axis1]
        # get_boundary_cost_table uses axis-1 (intra-node/TP axis) values.
        mesh_alpha = [alpha, alpha]   # both axes have the same latency
        mesh_beta  = [beta,  beta]    # both axes have the same bandwidth
        table = get_boundary_cost_table([(1, 1), (1, 2)], activation_bytes, mesh_alpha, mesh_beta)
        # submesh 1 = (1,2) → tp=2 as sender; submesh 0 = (1,1) → tp=1 as receiver
        expected = alpha + beta * activation_bytes * (1.0 - 1.0 / 2)
        assert abs(float(table[1, 0]) - expected) < 1e-12, f"Got {table[1,0]}, expected {expected}"

    def test_allgather_cost_tp4(self):
        """tp_send=4 → AllGather communicates 3/4 of activation_bytes."""
        alpha = 1e-5
        beta = 1e-11
        activation_bytes = 4096
        mesh_alpha = [alpha, alpha]
        mesh_beta  = [beta,  beta]
        table = get_boundary_cost_table([(1, 1), (1, 4)], activation_bytes, mesh_alpha, mesh_beta)
        expected = alpha + beta * activation_bytes * (1.0 - 1.0 / 4)
        assert abs(float(table[1, 0]) - expected) < 1e-12

    def test_dtype_float32(self):
        table = get_boundary_cost_table(self._submeshes(), 512, [1e-5], [1e-11])
        assert table.dtype == np.float32


# ============================================================
# build_pipeline_plan — heterogeneous_tp=True
# ============================================================

class TestBuildPipelinePlanHetero:
    """Test that build_pipeline_plan with heterogeneous_tp=True runs and
    returns a PipelinePlan with tp_per_stage populated."""

    def _make_layers(self, n=4):
        return [nn.Linear(16, 16) for _ in range(n)]

    def _meta_args(self):
        return {"hidden_states": torch.empty(1, 8, 16, device="meta")}

    def test_returns_plan_with_tp_per_stage(self):
        """hetero mode returns a PipelinePlan with tp_per_stage of length pp_size."""
        from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
        import numpy as np

        layers = self._make_layers(4)
        meta_args = self._meta_args()

        # Mock get_compute_cost and alpa_dp to avoid real ILP calls.
        fake_cost = np.full((4, 5, 2, 1), np.inf, dtype=np.float32)
        fake_cost[0, 2, 0, 0] = 10.0  # stage covering layers 0:2 on submesh 0
        fake_cost[2, 4, 1, 0] = 12.0  # stage covering layers 2:4 on submesh 1

        fake_solution = [
            ((0, 2), 0, 0),  # stage 0: layers[0:2], submesh[0]=(1,1)
            ((2, 4), 1, 0),  # stage 1: layers[2:4], submesh[1]=(1,2)
        ]

        # Use devices_per_host=2 so get_submesh_choices(1,2) returns exactly 2
        # submeshes: [(1,1), (1,2)] — matching fake_cost's M=2.
        with patch(
            "colossalai.auto_parallel.pipeline_shard.orchestrator.get_compute_cost",
            return_value=fake_cost,
        ), patch(
            "colossalai.auto_parallel.pipeline_shard.orchestrator.alpa_dp",
            return_value=(22.0, fake_solution),
        ), patch(
            "colossalai.auto_parallel.pipeline_shard.orchestrator.get_boundary_cost_table",
            return_value=np.zeros((2, 2), dtype=np.float32),
        ):
            plan = build_pipeline_plan(
                layers=layers,
                meta_args=meta_args,
                num_devices=4,
                num_microbatches=2,
                num_hosts=1,
                devices_per_host=2,   # → submeshes [(1,1),(1,2)], M=2 matches fake_cost
                heterogeneous_tp=True,
            )

        assert plan.pp_size == 2
        assert len(plan.tp_per_stage) == 2
        assert plan.tp_per_stage[0] == 1  # submesh (1,1) → tp=1
        assert plan.tp_per_stage[1] == 2  # submesh (1,2) → tp=2

    def test_hetero_detected_when_tp_differs(self):
        """heterogeneous_tp field is True when adjacent stages have different TP."""
        from colossalai.auto_parallel.pipeline_shard.orchestrator import _parse_solution

        submeshes = [(1, 1), (1, 2)]
        solution = [((0, 2), 0, 0), ((2, 4), 1, 0)]  # tp=1 then tp=2

        plan = _parse_solution(solution, submeshes, num_devices=4, cost=10.0,
                               heterogeneous_tp=True)
        assert plan.heterogeneous_tp is True
        assert plan.tp_per_stage == [1, 2]

    def test_uniform_hetero_not_flagged(self):
        """heterogeneous_tp=False when all stages have same TP even in hetero mode."""
        from colossalai.auto_parallel.pipeline_shard.orchestrator import _parse_solution

        submeshes = [(1, 2), (1, 2)]
        solution = [((0, 2), 0, 0), ((2, 4), 1, 0)]  # both tp=2

        plan = _parse_solution(solution, submeshes, num_devices=4, cost=10.0,
                               heterogeneous_tp=True)
        assert plan.heterogeneous_tp is False
        assert plan.tp_per_stage == [2, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
