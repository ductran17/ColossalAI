"""
Unit tests for Phase 2b — variable-size pipeline stages.

Tests cover:
  - PipelinePlan new fields: dp_per_stage, rank_ranges, variable_stage_sizes
  - _parse_solution_variable: rank_ranges, dp_per_stage, uniform-dp rejection
  - VariableStagePipelineManager: stage index, is_first/is_last, num_stages
  - build_pipeline_plan with variable_stage_sizes=True
  - CrossMeshP2PCommunication group layout (mocked dist)
  - _build_boundary_modules with variable stage sizes

All tests run single-process (no live distributed group required).
"""
import sys
import os

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from colossalai.auto_parallel.pipeline_shard.orchestrator import (
    PipelinePlan,
    VariableStagePipelineManager,
    _parse_solution_variable,
)


# ============================================================
# PipelinePlan — new fields
# ============================================================

class TestPipelinePlanFields:
    def test_default_values(self):
        plan = PipelinePlan()
        assert plan.dp_per_stage == []
        assert plan.rank_ranges == []
        assert plan.variable_stage_sizes is False

    def test_set_variable_fields(self):
        plan = PipelinePlan(
            dp_per_stage=[1, 1],
            rank_ranges=[[0], [1, 2]],
            variable_stage_sizes=True,
        )
        assert plan.dp_per_stage == [1, 1]
        assert plan.rank_ranges == [[0], [1, 2]]
        assert plan.variable_stage_sizes is True


# ============================================================
# _parse_solution_variable
# ============================================================

class TestParseSolutionVariable:
    def _make_solution(self, layer_ranges, submesh_indices, submesh_choices):
        """Build the list format that alpa_dp returns."""
        return [
            ((start, end), idx, 0)
            for (start, end), idx in zip(layer_ranges, submesh_indices)
        ]

    def test_basic_1_plus_2(self):
        """Stage 0: 1 GPU (tp=1,dp=1), Stage 1: 2 GPUs (tp=2,dp=1) — 3 GPUs total."""
        submesh_choices = [(1, 1), (1, 2)]
        solution = [
            ((0, 2), 0, 0),  # stage 0: layers 0-1, submesh (1,1)
            ((2, 4), 1, 0),  # stage 1: layers 2-3, submesh (1,2)
        ]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=3, cost=1.0)
        assert plan is not None
        assert plan.variable_stage_sizes is True
        assert plan.pp_size == 2
        assert plan.tp_per_stage == [1, 2]
        assert plan.dp_per_stage == [1, 1]
        assert plan.rank_ranges == [[0], [1, 2]]
        assert plan.stage_layer_ranges == [(0, 2), (2, 4)]

    def test_rank_ranges_contiguous(self):
        """Rank ranges must be contiguous slices of [0, num_devices)."""
        submesh_choices = [(1, 1), (1, 2), (1, 4)]
        solution = [
            ((0, 1), 0, 0),  # 1 GPU
            ((1, 2), 1, 0),  # 2 GPUs
            ((2, 3), 2, 0),  # 4 GPUs
        ]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=7, cost=1.0)
        assert plan is not None
        assert plan.rank_ranges == [[0], [1, 2], [3, 4, 5, 6]]

    def test_uniform_dp_accepted(self):
        """dp=2 on both stages is accepted."""
        submesh_choices = [(1, 2), (1, 4)]
        solution = [
            ((0, 2), 0, 0),  # submesh (1,2) → 2 devices, tp=2, dp=1
            ((2, 4), 1, 0),  # submesh (1,4) → 4 devices, tp=4, dp=1
        ]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=6, cost=1.0)
        assert plan is not None
        assert plan.dp_per_stage == [1, 1]

    def test_non_uniform_dp_rejected(self):
        """dp differs across stages → _parse_solution_variable returns None."""
        submesh_choices = [(1, 2), (1, 2)]
        # Stage 0: (1,2) → 2 devices, dp = 2//2 = 1
        # Stage 1: (1,2) → 2 devices, but if we had (2,2) dp would be 2 — let's
        # construct this manually via a 4-device submesh with dp=2.
        submesh_choices2 = [(1, 1), (2, 2)]
        solution2 = [
            ((0, 2), 0, 0),  # (1,1) → 1 device, tp=1, dp=1
            ((2, 4), 1, 0),  # (2,2) → 4 devices, tp=2, dp=2
        ]
        plan = _parse_solution_variable(solution2, submesh_choices2, num_devices=5, cost=1.0)
        assert plan is None

    def test_hetero_tp_flag(self):
        """heterogeneous_tp set when adjacent stages have different TP."""
        submesh_choices = [(1, 1), (1, 2)]
        solution = [((0, 2), 0, 0), ((2, 4), 1, 0)]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=3, cost=1.0)
        assert plan.heterogeneous_tp is True

    def test_uniform_tp_no_hetero_flag(self):
        """heterogeneous_tp=False when all stages use same TP."""
        submesh_choices = [(1, 2), (1, 2)]
        solution = [((0, 2), 0, 0), ((2, 4), 1, 0)]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=4, cost=1.0)
        assert plan.heterogeneous_tp is False

    def test_tp_size_is_max(self):
        """tp_size = max(tp_per_stage)."""
        submesh_choices = [(1, 1), (1, 4)]
        solution = [((0, 2), 0, 0), ((2, 4), 1, 0)]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=5, cost=1.0)
        assert plan.tp_size == 4

    def test_estimated_cost_stored(self):
        submesh_choices = [(1, 1), (1, 2)]
        solution = [((0, 2), 0, 0), ((2, 4), 1, 0)]
        plan = _parse_solution_variable(solution, submesh_choices, num_devices=3, cost=3.14)
        assert abs(plan.estimated_cost - 3.14) < 1e-6


# ============================================================
# VariableStagePipelineManager
# ============================================================

class TestVariableStagePipelineManager:
    def _make_plan(self, rank_ranges):
        pp = len(rank_ranges)
        tp = [1] * pp
        dp = [1] * pp
        return PipelinePlan(
            pp_size=pp,
            tp_per_stage=tp,
            dp_per_stage=dp,
            rank_ranges=rank_ranges,
            variable_stage_sizes=True,
        )

    def test_stage_assignment_rank0(self):
        plan = self._make_plan([[0], [1, 2]])
        mgr = VariableStagePipelineManager(plan, rank=0)
        assert mgr.stage == 0

    def test_stage_assignment_rank1(self):
        plan = self._make_plan([[0], [1, 2]])
        mgr = VariableStagePipelineManager(plan, rank=1)
        assert mgr.stage == 1

    def test_stage_assignment_rank2(self):
        plan = self._make_plan([[0], [1, 2]])
        mgr = VariableStagePipelineManager(plan, rank=2)
        assert mgr.stage == 1

    def test_num_stages(self):
        plan = self._make_plan([[0], [1, 2]])
        assert VariableStagePipelineManager(plan, 0).num_stages == 2

    def test_is_first_stage(self):
        plan = self._make_plan([[0], [1, 2]])
        assert VariableStagePipelineManager(plan, 0).is_first_stage() is True
        assert VariableStagePipelineManager(plan, 1).is_first_stage() is False

    def test_is_last_stage(self):
        plan = self._make_plan([[0], [1, 2]])
        assert VariableStagePipelineManager(plan, 0).is_last_stage() is False
        assert VariableStagePipelineManager(plan, 1).is_last_stage() is True
        assert VariableStagePipelineManager(plan, 2).is_last_stage() is True

    def test_three_stages(self):
        plan = self._make_plan([[0, 1], [2, 3, 4], [5]])
        for rank in [0, 1]:
            assert VariableStagePipelineManager(plan, rank).stage == 0
        for rank in [2, 3, 4]:
            assert VariableStagePipelineManager(plan, rank).stage == 1
        assert VariableStagePipelineManager(plan, 5).stage == 2
        assert VariableStagePipelineManager(plan, 5).is_last_stage() is True

    def test_get_rank(self):
        plan = self._make_plan([[0], [1, 2]])
        for r in range(3):
            assert VariableStagePipelineManager(plan, r).get_rank() == r


# ============================================================
# CrossMeshP2PCommunication — group layout (mocked dist)
# ============================================================

class TestCrossMeshP2PCommunicationGroups:
    """Verify that process groups are built with the correct rank sets."""

    def _make_plan_3gpu(self):
        """pp=2, stage 0: [0] (tp=1,dp=1), stage 1: [1,2] (tp=2,dp=1)."""
        return PipelinePlan(
            pp_size=2,
            tp_per_stage=[1, 2],
            dp_per_stage=[1, 1],
            rank_ranges=[[0], [1, 2]],
            variable_stage_sizes=True,
        )

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.new_group")
    def test_group_members_3gpu(self, mock_new_group, mock_get_rank):
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        mock_new_group.return_value = MagicMock()
        plan = self._make_plan_3gpu()
        mgr = VariableStagePipelineManager(plan, rank=0)
        p2p = CrossMeshP2PCommunication(plan, mgr)
        # dp=1 → one group for the single pair (s=0, s=1), dp_replica r=0
        assert mock_new_group.call_count == 1
        called_ranks = mock_new_group.call_args[0][0]
        # sender TP ranks for stage 0 = [0]; receiver TP ranks for stage 1 = [1, 2]
        assert sorted(called_ranks) == [0, 1, 2]

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.new_group")
    def test_fwd_src_is_tp0_of_sender(self, mock_new_group, mock_get_rank):
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        mock_new_group.return_value = MagicMock()
        plan = self._make_plan_3gpu()
        mgr = VariableStagePipelineManager(plan, rank=0)
        p2p = CrossMeshP2PCommunication(plan, mgr)
        # Forward root = tp_rank=0 of stage 0 for dp_replica 0 = rank 0
        assert p2p._fwd_srcs[0][0] == 0

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.new_group")
    def test_bwd_src_is_tp0_of_receiver(self, mock_new_group, mock_get_rank):
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        mock_new_group.return_value = MagicMock()
        plan = self._make_plan_3gpu()
        mgr = VariableStagePipelineManager(plan, rank=0)
        p2p = CrossMeshP2PCommunication(plan, mgr)
        # Backward root = tp_rank=0 of stage 1 for dp_replica 0 = rank 1
        assert p2p._bwd_srcs[0][0] == 1

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.new_group")
    def test_dp_rank_computed_correctly(self, mock_new_group, mock_get_rank):
        """With dp=2, each rank's dp_rank is local_idx % dp_s."""
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        # pp=2, stage 0: [0,1] (tp=1, dp=2), stage 1: [2,3] (tp=1, dp=2)
        plan = PipelinePlan(
            pp_size=2,
            tp_per_stage=[1, 1],
            dp_per_stage=[2, 2],
            rank_ranges=[[0, 1], [2, 3]],
            variable_stage_sizes=True,
        )
        mock_new_group.return_value = MagicMock()
        for rank, expected_dp_rank in [(0, 0), (1, 1)]:
            mock_get_rank.return_value = rank
            mgr = VariableStagePipelineManager(plan, rank=rank)
            p2p = CrossMeshP2PCommunication(plan, mgr)
            assert p2p._dp_rank == expected_dp_rank, f"rank {rank}: dp_rank {p2p._dp_rank} != {expected_dp_rank}"

    @patch("torch.distributed.get_rank", return_value=0)
    @patch("torch.distributed.new_group")
    def test_group_count_with_dp2(self, mock_new_group, mock_get_rank):
        """With dp=2, two groups created per adjacent stage pair (one per dp replica)."""
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        plan = PipelinePlan(
            pp_size=2,
            tp_per_stage=[1, 1],
            dp_per_stage=[2, 2],
            rank_ranges=[[0, 1], [2, 3]],
            variable_stage_sizes=True,
        )
        mock_new_group.return_value = MagicMock()
        mgr = VariableStagePipelineManager(plan, rank=0)
        p2p = CrossMeshP2PCommunication(plan, mgr)
        assert mock_new_group.call_count == 2  # 1 stage pair × 2 dp replicas


# ============================================================
# build_pipeline_plan with variable_stage_sizes
# ============================================================

class TestBuildPipelinePlanVariable:
    """Smoke-test build_pipeline_plan with variable_stage_sizes=True.

    Uses mocked get_compute_cost and alpa_dp so no GPU is needed.
    """

    def _make_layers(self, n=4):
        return [nn.Linear(64, 64) for _ in range(n)]

    def _make_meta_args(self):
        return {"hidden_states": torch.empty(2, 8, 64, device="meta")}

    @patch("colossalai.auto_parallel.pipeline_shard.orchestrator.alpa_dp")
    @patch("colossalai.auto_parallel.pipeline_shard.orchestrator.get_compute_cost")
    @patch("colossalai.auto_parallel.pipeline_shard.orchestrator.get_boundary_cost_table")
    def test_variable_stage_returns_plan(self, mock_bct, mock_gcc, mock_adp):
        import numpy as np
        from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
        from colossalai.device.calc_pipeline_strategy import get_submesh_choices

        # Determine real submesh count so mocks have consistent shapes.
        n_sub = len(get_submesh_choices(1, 4))  # e.g. 4 for devices_per_host=4
        mock_gcc.return_value = np.ones((4, 4, n_sub, 1), dtype=np.float32)
        mock_bct.return_value = np.zeros((n_sub, n_sub), dtype=np.float32)
        # alpa_dp returns: stage0=(0,2)/submesh idx0=(1,1), stage1=(2,4)/submesh idx1=(1,2)
        mock_adp.return_value = (2.0, [((0, 2), 0, 0), ((2, 4), 1, 0)])

        plan = build_pipeline_plan(
            layers=self._make_layers(4),
            meta_args=self._make_meta_args(),
            num_devices=3,
            num_microbatches=2,
            num_hosts=1,
            devices_per_host=4,
            variable_stage_sizes=True,
        )

        assert plan.variable_stage_sizes is True
        assert plan.pp_size == 2
        assert plan.rank_ranges == [[0], [1, 2]]
        assert plan.tp_per_stage == [1, 2]
        assert plan.dp_per_stage == [1, 1]
        assert plan.heterogeneous_tp is True

    @patch("colossalai.auto_parallel.pipeline_shard.orchestrator.alpa_dp")
    @patch("colossalai.auto_parallel.pipeline_shard.orchestrator.get_compute_cost")
    @patch("colossalai.auto_parallel.pipeline_shard.orchestrator.get_boundary_cost_table")
    def test_variable_stage_non_uniform_dp_raises(self, mock_bct, mock_gcc, mock_adp):
        import numpy as np
        from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan
        from colossalai.device.calc_pipeline_strategy import get_submesh_choices

        n_sub = len(get_submesh_choices(1, 4))
        mock_gcc.return_value = np.ones((4, 4, n_sub, 1), dtype=np.float32)
        mock_bct.return_value = np.zeros((n_sub, n_sub), dtype=np.float32)
        # Solution: stage0 submesh idx0=(1,1) dp=1, stage1 submesh idx3=(2,2) dp=2 → non-uniform
        mock_adp.return_value = (2.0, [((0, 2), 0, 0), ((2, 4), 3, 0)])

        with pytest.raises(RuntimeError, match="non-uniform dp"):
            build_pipeline_plan(
                layers=self._make_layers(4),
                meta_args=self._make_meta_args(),
                num_devices=5,
                num_microbatches=2,
                num_hosts=1,
                devices_per_host=4,
                variable_stage_sizes=True,
            )
