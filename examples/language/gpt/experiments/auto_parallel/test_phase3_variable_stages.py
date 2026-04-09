"""
Unit tests for Phase 3 variable-size pipeline stage components.

Tests cover (all single-process, no distributed needed):
  - _parse_solution_variable: rank_ranges, dp_per_stage, uniform-dp rejection
  - VariableStagePipelineManager: stage index, is_first/is_last, num_stages
  - PipelinePlan phase3 fields: variable_stage_sizes, rank_ranges, dp_per_stage
  - CrossMeshP2PCommunication group layout: correct group membership per dp replica

Run with:
    python test_phase3_variable_stages.py
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

# ---------------------------------------------------------------------------
# Import warm-up: on Python 3.13 with CUDA arch >= 89, importing
# colossalai.auto_parallel.pipeline_shard triggers a torch.compile error
# in fp8.py. The import chain partially populates sys.modules on each
# failed attempt; after 2 failures the third attempt succeeds.
# ---------------------------------------------------------------------------
for _attempt in range(3):
    try:
        import colossalai.auto_parallel.pipeline_shard.orchestrator as _orch_mod
        break
    except RuntimeError:
        pass


# ============================================================
# Helpers
# ============================================================

def _make_solution(stage_specs):
    """Build a fake alpa_dp solution list.

    Args:
        stage_specs: list of ((start, end), submesh_idx) pairs.

    Returns:
        solution: list of ((start, end), submesh_idx, 0)
    """
    return [((s, e), m, 0) for (s, e), m in stage_specs]


# Submesh choices used in tests:
# Index 0: (1, 1) → 1 device, tp=1, dp=1
# Index 1: (1, 2) → 2 devices, tp=2, dp=1
# Index 2: (2, 2) → 4 devices, tp=2, dp=2
SUBMESHES = [(1, 1), (1, 2), (2, 2)]


def _make_var_plan(rank_ranges, tp_per_stage, dp_per_stage, stage_layer_ranges=None):
    """Convenience: create a Phase 3 PipelinePlan."""
    from colossalai.auto_parallel.pipeline_shard.orchestrator import PipelinePlan
    pp = len(rank_ranges)
    tp_size = max(tp_per_stage)
    dp_size = dp_per_stage[0]
    if stage_layer_ranges is None:
        stage_layer_ranges = [(i * 2, (i + 1) * 2) for i in range(pp)]
    submesh_per_stage = [(dp_per_stage[s], tp_per_stage[s]) for s in range(pp)]
    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        submesh_per_stage=submesh_per_stage,
        tp_per_stage=tp_per_stage,
        dp_per_stage=dp_per_stage,
        rank_ranges=rank_ranges,
        pp_size=pp,
        tp_size=tp_size,
        dp_size=dp_size,
        variable_stage_sizes=True,
        estimated_cost=1.0,
    )


# ============================================================
# _parse_solution_variable
# ============================================================

class TestParseSolutionVariable(unittest.TestCase):
    def setUp(self):
        from colossalai.auto_parallel.pipeline_shard.orchestrator import _parse_solution_variable
        self.parse = _parse_solution_variable

    def test_uniform_dp_3gpu(self):
        """pp=2, stage0=(1,1)=1dev, stage1=(1,2)=2dev → total 3 GPUs, dp=[1,1]."""
        solution = _make_solution([((0, 2), 0), ((2, 4), 1)])
        plan = self.parse(solution, SUBMESHES, num_devices=3, cost=1.0)
        self.assertIsNotNone(plan)
        self.assertTrue(plan.variable_stage_sizes)
        self.assertEqual(plan.pp_size, 2)
        self.assertEqual(plan.tp_per_stage, [1, 2])
        self.assertEqual(plan.dp_per_stage, [1, 1])

    def test_rank_ranges_3gpu(self):
        """Stage 0 owns ranks [0], stage 1 owns ranks [1, 2]."""
        solution = _make_solution([((0, 2), 0), ((2, 4), 1)])
        plan = self.parse(solution, SUBMESHES, num_devices=3, cost=1.0)
        self.assertEqual(plan.rank_ranges, [[0], [1, 2]])

    def test_rank_ranges_4gpu_equal(self):
        """pp=2, (1,2)+(1,2): stage0=[0,1], stage1=[2,3]."""
        solution = _make_solution([((0, 2), 1), ((2, 4), 1)])
        plan = self.parse(solution, SUBMESHES, num_devices=4, cost=2.0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.rank_ranges, [[0, 1], [2, 3]])

    def test_dp2_per_stage(self):
        """pp=2, (2,2)+(2,2): device_count=4 each, tp=2, dp=4//2=2."""
        solution = _make_solution([((0, 2), 2), ((2, 4), 2)])
        plan = self.parse(solution, SUBMESHES, num_devices=8, cost=3.0)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.dp_per_stage, [2, 2])
        self.assertEqual(plan.rank_ranges, [[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_non_uniform_dp_returns_none(self):
        """stage0=(1,1) dp=1, stage1=(2,2) dp=2 → non-uniform → None."""
        solution = _make_solution([((0, 2), 0), ((2, 4), 2)])
        plan = self.parse(solution, SUBMESHES, num_devices=5, cost=1.0)
        self.assertIsNone(plan)

    def test_device_count_mismatch_raises(self):
        """Stage device counts summing to 3 but num_devices=5 → AssertionError."""
        solution = _make_solution([((0, 2), 0), ((2, 4), 1)])
        with self.assertRaises(AssertionError):
            self.parse(solution, SUBMESHES, num_devices=5, cost=1.0)

    def test_tp_size_is_max(self):
        """tp_size field = max over all stages."""
        solution = _make_solution([((0, 2), 0), ((2, 4), 1)])
        plan = self.parse(solution, SUBMESHES, num_devices=3, cost=1.0)
        self.assertEqual(plan.tp_size, 2)

    def test_stage_layer_ranges(self):
        """Layer ranges are preserved from the solution."""
        solution = _make_solution([((0, 3), 0), ((3, 6), 1)])
        plan = self.parse(solution, SUBMESHES, num_devices=3, cost=1.0)
        self.assertEqual(plan.stage_layer_ranges, [(0, 3), (3, 6)])

    def test_hetero_tp_detected(self):
        """heterogeneous_tp=True when adjacent stages differ in tp."""
        solution = _make_solution([((0, 2), 0), ((2, 4), 1)])  # tp=[1,2]
        plan = self.parse(solution, SUBMESHES, num_devices=3, cost=1.0)
        self.assertTrue(plan.heterogeneous_tp)

    def test_uniform_tp_not_hetero(self):
        """heterogeneous_tp=False when all stages share tp."""
        solution = _make_solution([((0, 2), 1), ((2, 4), 1)])  # tp=[2,2]
        plan = self.parse(solution, SUBMESHES, num_devices=4, cost=1.0)
        self.assertIsNotNone(plan)
        self.assertFalse(plan.heterogeneous_tp)

    def test_single_stage(self):
        """pp=1 is valid: dp trivially uniform."""
        solution = _make_solution([((0, 4), 1)])
        plan = self.parse(solution, SUBMESHES, num_devices=2, cost=0.5)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.pp_size, 1)
        self.assertEqual(plan.rank_ranges, [[0, 1]])
        self.assertEqual(plan.dp_per_stage, [1])


# ============================================================
# VariableStagePipelineManager
# ============================================================

class TestVariableStagePipelineManager(unittest.TestCase):
    def setUp(self):
        from colossalai.auto_parallel.pipeline_shard.orchestrator import VariableStagePipelineManager
        self.Manager = VariableStagePipelineManager
        # pp=2: stage0=[rank0], stage1=[rank1, rank2]
        self.plan = _make_var_plan([[0], [1, 2]], [1, 2], [1, 1])

    def test_stage_rank0(self):
        mgr = self.Manager(self.plan, rank=0)
        self.assertEqual(mgr.stage, 0)

    def test_stage_rank1(self):
        mgr = self.Manager(self.plan, rank=1)
        self.assertEqual(mgr.stage, 1)

    def test_stage_rank2(self):
        """Both rank1 and rank2 belong to stage 1."""
        mgr = self.Manager(self.plan, rank=2)
        self.assertEqual(mgr.stage, 1)

    def test_num_stages(self):
        mgr = self.Manager(self.plan, rank=0)
        self.assertEqual(mgr.num_stages, 2)

    def test_is_first_stage(self):
        self.assertTrue(self.Manager(self.plan, rank=0).is_first_stage())
        self.assertFalse(self.Manager(self.plan, rank=1).is_first_stage())

    def test_is_last_stage(self):
        self.assertFalse(self.Manager(self.plan, rank=0).is_last_stage())
        self.assertTrue(self.Manager(self.plan, rank=1).is_last_stage())

    def test_get_rank(self):
        self.assertEqual(self.Manager(self.plan, rank=2).get_rank(), 2)

    def test_3_stages(self):
        plan = _make_var_plan([[0], [1, 2], [3, 4, 5, 6]], [1, 2, 4], [1, 1, 1],
                              stage_layer_ranges=[(0, 2), (2, 4), (4, 6)])
        mgr_first = self.Manager(plan, rank=0)
        mgr_mid = self.Manager(plan, rank=2)
        mgr_last = self.Manager(plan, rank=5)
        self.assertEqual(mgr_first.stage, 0)
        self.assertEqual(mgr_mid.stage, 1)
        self.assertEqual(mgr_last.stage, 2)
        self.assertTrue(mgr_first.is_first_stage())
        self.assertFalse(mgr_first.is_last_stage())
        self.assertFalse(mgr_mid.is_first_stage())
        self.assertFalse(mgr_mid.is_last_stage())
        self.assertFalse(mgr_last.is_first_stage())
        self.assertTrue(mgr_last.is_last_stage())

    def test_uniform_size_stages(self):
        """Also works when all stages have the same size (pp=2, tp=2 each, 4 GPUs)."""
        plan = _make_var_plan([[0, 1], [2, 3]], [2, 2], [1, 1])
        Manager = self.Manager
        for r in [0, 1]:
            self.assertEqual(Manager(plan, rank=r).stage, 0)
        for r in [2, 3]:
            self.assertEqual(Manager(plan, rank=r).stage, 1)


# ============================================================
# CrossMeshP2PCommunication — group layout (mocked dist)
# ============================================================

class TestCrossMeshP2PCommunicationGroups(unittest.TestCase):
    """Verify that CrossMeshP2PCommunication creates the right process groups."""

    def _make_p2p(self, plan, rank, mock_new_group):
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        from colossalai.auto_parallel.pipeline_shard.orchestrator import VariableStagePipelineManager
        mock_new_group.return_value = MagicMock()
        stage_manager = VariableStagePipelineManager(plan, rank)
        with patch("torch.distributed.get_rank", return_value=rank):
            p2p = CrossMeshP2PCommunication(plan, stage_manager)
        return p2p, mock_new_group.call_args_list

    @patch("torch.distributed.new_group")
    def test_3gpu_group_members(self, mock_new_group):
        """pp=2, tp=[1,2], dp=1.
        Stage 0: rank [0], Stage 1: ranks [1,2]
        Expected 1 group covering all 3 ranks.
        """
        plan = _make_var_plan([[0], [1, 2]], [1, 2], [1, 1])
        p2p, calls = self._make_p2p(plan, rank=0, mock_new_group=mock_new_group)
        self.assertEqual(len(calls), 1)
        group_ranks = calls[0][0][0]
        self.assertEqual(sorted(group_ranks), [0, 1, 2])

    @patch("torch.distributed.new_group")
    def test_3gpu_fwd_src_is_stage0_tp0(self, mock_new_group):
        """fwd_src for dp_r=0 = stage0[tp0,dp0] = rank_ranges[0][0] = 0."""
        plan = _make_var_plan([[0], [1, 2]], [1, 2], [1, 1])
        p2p, _ = self._make_p2p(plan, rank=0, mock_new_group=mock_new_group)
        self.assertEqual(p2p._fwd_srcs[0][0], 0)

    @patch("torch.distributed.new_group")
    def test_3gpu_bwd_src_is_stage1_tp0(self, mock_new_group):
        """bwd_src for dp_r=0 = stage1[tp0,dp0] = rank_ranges[1][0] = 1."""
        plan = _make_var_plan([[0], [1, 2]], [1, 2], [1, 1])
        p2p, _ = self._make_p2p(plan, rank=0, mock_new_group=mock_new_group)
        self.assertEqual(p2p._bwd_srcs[0][0], 1)

    @patch("torch.distributed.new_group")
    def test_4gpu_dp2_creates_2_groups(self, mock_new_group):
        """pp=2, tp=[1,1], dp=2.
        Groups: dp_r=0 → [0,2], dp_r=1 → [1,3]
        """
        plan = _make_var_plan([[0, 1], [2, 3]], [1, 1], [2, 2])
        _, calls = self._make_p2p(plan, rank=0, mock_new_group=mock_new_group)
        self.assertEqual(len(calls), 2)
        all_groups = sorted([sorted(c[0][0]) for c in calls])
        self.assertEqual(all_groups, [[0, 2], [1, 3]])

    @patch("torch.distributed.new_group")
    def test_fwd_src_per_dp_replica(self, mock_new_group):
        """fwd_srcs[s][r] = rank_ranges[s][r] (tp_rank=0 for each dp replica)."""
        plan = _make_var_plan([[0, 1], [2, 3]], [1, 1], [2, 2])
        mock_new_group.return_value = MagicMock()
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        from colossalai.auto_parallel.pipeline_shard.orchestrator import VariableStagePipelineManager
        with patch("torch.distributed.get_rank", return_value=0):
            p2p = CrossMeshP2PCommunication(plan, VariableStagePipelineManager(plan, rank=0))
        # dp_r=0: rank_ranges[0][0*2+0] = rank_ranges[0][0] = 0
        self.assertEqual(p2p._fwd_srcs[0][0], 0)
        # dp_r=1: rank_ranges[0][0*2+1] = rank_ranges[0][1] = 1
        self.assertEqual(p2p._fwd_srcs[0][1], 1)

    @patch("torch.distributed.new_group")
    def test_bwd_src_per_dp_replica(self, mock_new_group):
        """bwd_srcs[s][r] = rank_ranges[s+1][r]."""
        plan = _make_var_plan([[0, 1], [2, 3]], [1, 1], [2, 2])
        mock_new_group.return_value = MagicMock()
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        from colossalai.auto_parallel.pipeline_shard.orchestrator import VariableStagePipelineManager
        with patch("torch.distributed.get_rank", return_value=0):
            p2p = CrossMeshP2PCommunication(plan, VariableStagePipelineManager(plan, rank=0))
        self.assertEqual(p2p._bwd_srcs[0][0], 2)
        self.assertEqual(p2p._bwd_srcs[0][1], 3)

    @patch("torch.distributed.new_group")
    def test_dp_rank_computed_correctly(self, mock_new_group):
        """dp_rank = local_idx % dp_s."""
        plan = _make_var_plan([[0, 1], [2, 3]], [1, 1], [2, 2])
        mock_new_group.return_value = MagicMock()
        from colossalai.auto_parallel.pipeline_shard.cross_mesh_p2p import CrossMeshP2PCommunication
        from colossalai.auto_parallel.pipeline_shard.orchestrator import VariableStagePipelineManager
        # rank=0: local_idx=0, dp_s=2, dp_rank=0
        with patch("torch.distributed.get_rank", return_value=0):
            p2p0 = CrossMeshP2PCommunication(plan, VariableStagePipelineManager(plan, rank=0))
        self.assertEqual(p2p0._dp_rank, 0)
        # rank=1: local_idx=1, dp_s=2, dp_rank=1
        with patch("torch.distributed.get_rank", return_value=1):
            p2p1 = CrossMeshP2PCommunication(plan, VariableStagePipelineManager(plan, rank=1))
        self.assertEqual(p2p1._dp_rank, 1)

    @patch("torch.distributed.new_group")
    def test_3stage_pipeline_group_count(self, mock_new_group):
        """pp=3, dp=1 → 2 stage pairs × 1 dp replica = 2 groups."""
        plan = _make_var_plan([[0], [1, 2], [3, 4, 5, 6]], [1, 2, 4], [1, 1, 1],
                              stage_layer_ranges=[(0, 2), (2, 4), (4, 6)])
        _, _ = self._make_p2p(plan, rank=0, mock_new_group=mock_new_group)
        self.assertEqual(mock_new_group.call_count, 2)

    @patch("torch.distributed.new_group")
    def test_rank_not_in_group_still_creates_group(self, mock_new_group):
        """All ranks (including those not in adjacent stages) call new_group."""
        # rank=5 is in stage 2, but still creates the group for pair(0,1).
        plan = _make_var_plan([[0], [1, 2], [3, 4, 5, 6]], [1, 2, 4], [1, 1, 1],
                              stage_layer_ranges=[(0, 2), (2, 4), (4, 6)])
        _, calls_from_rank5 = self._make_p2p(plan, rank=5, mock_new_group=mock_new_group)
        # rank 5 still creates 2 groups (for pair 0→1 and pair 1→2)
        self.assertEqual(mock_new_group.call_count, 2)


# ============================================================
# PipelinePlan phase3 fields
# ============================================================

class TestPipelinePlanPhase3Fields(unittest.TestCase):
    def setUp(self):
        from colossalai.auto_parallel.pipeline_shard.orchestrator import PipelinePlan
        self.PipelinePlan = PipelinePlan

    def test_defaults(self):
        plan = self.PipelinePlan()
        self.assertFalse(plan.variable_stage_sizes)
        self.assertEqual(plan.dp_per_stage, [])
        self.assertEqual(plan.rank_ranges, [])

    def test_phase3_fields_set(self):
        plan = _make_var_plan([[0], [1, 2]], [1, 2], [1, 1])
        self.assertTrue(plan.variable_stage_sizes)
        self.assertEqual(plan.rank_ranges, [[0], [1, 2]])
        self.assertEqual(plan.dp_per_stage, [1, 1])
        self.assertEqual(plan.tp_per_stage, [1, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
