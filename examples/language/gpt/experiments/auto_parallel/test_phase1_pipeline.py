# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team
"""
Unit tests for Phase 1 auto 3D pipeline parallelism (uniform TP).

These tests are designed to run WITHOUT a GPU / distributed setup.
They verify:
  1. alpa_dp shape assertion accepts (K, K+1, M, C) arrays.
  2. solve_solution() returns (solution, objective) tuple.
  3. get_compute_cost() calls _estimate_stage_cost correctly and
     produces a (K, K+1, M, 1) array.
  4. build_pipeline_plan() produces a valid PipelinePlan on a tiny toy model.

Run with:
    python test_phase1_pipeline.py
or:
    pytest test_phase1_pipeline.py -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Test 1: alpa_dp assertion accepts the corrected shape
# ---------------------------------------------------------------------------
class TestAlpaDpShape(unittest.TestCase):
    def setUp(self):
        from colossalai.device.calc_pipeline_strategy import alpa_dp

        self.alpa_dp = alpa_dp

    def _make_cost(self, num_layers, num_submeshes, fill=1.0):
        """Create a valid cost array of shape (K, K+1, M, 1)."""
        cost = np.full((num_layers, num_layers + 1, num_submeshes, 1), fill, dtype=np.float32)
        # Fill inf for invalid (k >= i) entries so the DP is well-formed.
        for k in range(num_layers):
            for i in range(k + 1):
                cost[k, i, :, :] = np.inf
        return cost

    def test_correct_shape_is_accepted(self):
        """alpa_dp should not raise for shape (K, K+1, M, 1)."""
        num_layers = 4
        submesh_choices = [(1, 1), (1, 2), (2, 2)]
        cost = self._make_cost(num_layers, len(submesh_choices), fill=0.5)
        # Should not raise AssertionError.
        try:
            self.alpa_dp(
                num_layers=num_layers,
                num_devices=4,
                num_microbatches=4,
                submesh_choices=submesh_choices,
                num_autosharding_configs=1,
                compute_cost=cost,
            )
        except AssertionError as e:
            self.fail(f"alpa_dp raised AssertionError unexpectedly: {e}")

    def test_old_shape_is_rejected(self):
        """alpa_dp should raise for the old incorrect shape (K, K, M, 1)."""
        num_layers = 4
        submesh_choices = [(1, 1), (1, 2)]
        # Old wrong shape: second dim = num_layers, not num_layers+1
        bad_cost = np.ones((num_layers, num_layers, len(submesh_choices), 1), dtype=np.float32)
        with self.assertRaises(AssertionError):
            self.alpa_dp(
                num_layers=num_layers,
                num_devices=4,
                num_microbatches=4,
                submesh_choices=submesh_choices,
                num_autosharding_configs=1,
                compute_cost=bad_cost,
            )

    def test_returns_valid_plan_for_feasible_cost(self):
        """alpa_dp should return a non-inf cost and a non-None solution when feasible."""
        num_layers = 4
        num_devices = 4
        submesh_choices = [(1, 2), (2, 2)]
        cost = self._make_cost(num_layers, len(submesh_choices), fill=1.0)
        total_cost, solution = self.alpa_dp(
            num_layers=num_layers,
            num_devices=num_devices,
            num_microbatches=2,
            submesh_choices=submesh_choices,
            num_autosharding_configs=1,
            compute_cost=cost,
        )
        self.assertFalse(np.isinf(total_cost), "Expected a finite total cost for a feasible problem.")
        self.assertIsNotNone(solution, "Expected a non-None solution.")
        # Each element of solution is ((start, end), submesh_idx, config_idx).
        total_layers_covered = sum(end - start for (start, end), _, _ in solution)
        self.assertEqual(total_layers_covered, num_layers, "Solution should cover all layers exactly once.")


# ---------------------------------------------------------------------------
# Test 2: solve_solution returns (solution, objective) tuple
# ---------------------------------------------------------------------------
class TestSolveSolutionReturnType(unittest.TestCase):
    def test_returns_tuple(self):
        """solve_solution must return a 2-tuple (solution, objective)."""
        from colossalai.auto_parallel.tensor_shard.initialize import solve_solution

        # Build a minimal mock that satisfies solve_solution's interface.
        mock_gm = MagicMock()
        mock_gm.graph = MagicMock()
        mock_strategy = MagicMock()
        mock_strategy.leaf_strategies = []

        mock_cost_graph = MagicMock()
        mock_solver = MagicMock()
        # Solver.call_solver_serialized_args() returns (s_val, e_val, objective, status).
        mock_solver.call_solver_serialized_args.return_value = ([0, 1, 0], [], 42.5, "Optimal")

        with patch("colossalai.auto_parallel.tensor_shard.initialize.CostGraph", return_value=mock_cost_graph):
            with patch("colossalai.auto_parallel.tensor_shard.initialize.Solver", return_value=mock_solver):
                result = solve_solution(mock_gm, mock_strategy)

        self.assertIsInstance(result, tuple, "solve_solution must return a tuple.")
        self.assertEqual(len(result), 2, "solve_solution must return exactly 2 elements.")
        solution, objective = result
        self.assertIsInstance(solution, list, "First element (solution) must be a list.")
        self.assertIsInstance(objective, float, "Second element (objective) must be a float.")
        self.assertAlmostEqual(objective, 42.5)


# ---------------------------------------------------------------------------
# Test 3: get_compute_cost produces the right shape and fills correctly
# ---------------------------------------------------------------------------
class TestGetComputeCost(unittest.TestCase):
    def _make_layers(self, n):
        """Create n identical linear layers for testing."""
        return [nn.Linear(8, 8) for _ in range(n)]

    def test_output_shape(self):
        """get_compute_cost must return shape (K, K+1, M, 1)."""
        from colossalai.auto_parallel.pipeline_shard.compute_cost import get_compute_cost

        num_layers = 3
        submesh_choices = [(1, 1), (1, 2)]
        layers = self._make_layers(num_layers)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        # Patch _estimate_stage_cost so we don't actually run ILP.
        with patch(
            "colossalai.auto_parallel.pipeline_shard.compute_cost._estimate_stage_cost",
            return_value=1.0,
        ):
            cost = get_compute_cost(
                layers=layers,
                meta_args=meta_args,
                submesh_choices=submesh_choices,
                mesh_alpha=[1e-5, 1e-5],
                mesh_beta=[1e-11, 1e-11],
            )

        expected_shape = (num_layers, num_layers + 1, len(submesh_choices), 1)
        self.assertEqual(cost.shape, expected_shape, f"Expected shape {expected_shape}, got {cost.shape}.")

    def test_homogeneous_broadcast(self):
        """cost[k, k+s, m] should equal cost[k', k'+s, m] for all valid k, k' (homogeneous)."""
        from colossalai.auto_parallel.pipeline_shard.compute_cost import get_compute_cost

        num_layers = 4
        submesh_choices = [(1, 2)]
        layers = self._make_layers(num_layers)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        # Assign distinct cost per stage_size so we can verify broadcasting.
        side_effects = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        call_count = [0]

        def fake_estimate(num_stage_layers, **kwargs):
            call_count[0] += 1
            return side_effects[num_stage_layers]

        with patch(
            "colossalai.auto_parallel.pipeline_shard.compute_cost._estimate_stage_cost",
            side_effect=fake_estimate,
        ):
            cost = get_compute_cost(
                layers=layers,
                meta_args=meta_args,
                submesh_choices=submesh_choices,
                mesh_alpha=[1e-5, 1e-5],
                mesh_beta=[1e-11, 1e-11],
            )

        # Exactly num_layers calls (one per stage_size) — NOT K*(K+1)/2.
        self.assertEqual(call_count[0], num_layers, "Should call _estimate_stage_cost exactly K times.")

        # All entries cost[k, k+1, 0, 0] must equal side_effects[1] = 0.1.
        for k in range(num_layers):
            self.assertAlmostEqual(
                cost[k, k + 1, 0, 0],
                0.1,
                places=5,
                msg=f"cost[{k}, {k+1}, 0, 0] should be 0.1 (stage_size=1)",
            )

        # cost[0, 3, 0, 0] must equal side_effects[3] = 0.3.
        self.assertAlmostEqual(cost[0, 3, 0, 0], 0.3, places=5)

    def test_invalid_entries_are_inf(self):
        """cost[k, i, m, 0] must be inf for i <= k (no stage can go backwards)."""
        from colossalai.auto_parallel.pipeline_shard.compute_cost import get_compute_cost

        num_layers = 3
        submesh_choices = [(1, 1)]
        layers = self._make_layers(num_layers)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        with patch(
            "colossalai.auto_parallel.pipeline_shard.compute_cost._estimate_stage_cost",
            return_value=1.0,
        ):
            cost = get_compute_cost(
                layers=layers,
                meta_args=meta_args,
                submesh_choices=submesh_choices,
                mesh_alpha=[1e-5, 1e-5],
                mesh_beta=[1e-11, 1e-11],
            )

        # Diagonal and below (i <= k) were never filled → must remain inf.
        for k in range(num_layers):
            for i in range(k + 1):
                self.assertTrue(
                    np.isinf(cost[k, i, 0, 0]),
                    f"cost[{k}, {i}, 0, 0] should be inf (i <= k).",
                )

    def test_caching(self):
        """get_compute_cost should save and reload cache without re-running ILP."""
        import os
        import tempfile

        from colossalai.auto_parallel.pipeline_shard.compute_cost import get_compute_cost

        num_layers = 2
        submesh_choices = [(1, 1)]
        layers = self._make_layers(num_layers)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = os.path.join(tmpdir, "cost_tp1.pkl")
            call_count = [0]

            def counting_estimate(*args, **kwargs):
                call_count[0] += 1
                return 0.5

            with patch(
                "colossalai.auto_parallel.pipeline_shard.compute_cost._estimate_stage_cost",
                side_effect=counting_estimate,
            ):
                # First call: should compute and save.
                cost1 = get_compute_cost(
                    layers=layers,
                    meta_args=meta_args,
                    submesh_choices=submesh_choices,
                    mesh_alpha=[1e-5, 1e-5],
                    mesh_beta=[1e-11, 1e-11],
                    cache_path=cache_file,
                )
                calls_after_first = call_count[0]
                self.assertGreater(calls_after_first, 0)

                # Second call: should load from cache, not call _estimate_stage_cost again.
                cost2 = get_compute_cost(
                    layers=layers,
                    meta_args=meta_args,
                    submesh_choices=submesh_choices,
                    mesh_alpha=[1e-5, 1e-5],
                    mesh_beta=[1e-11, 1e-11],
                    cache_path=cache_file,
                )
                self.assertEqual(
                    call_count[0],
                    calls_after_first,
                    "Second call should not invoke _estimate_stage_cost (cache hit).",
                )
                np.testing.assert_array_equal(cost1, cost2)


# ---------------------------------------------------------------------------
# Test 4: build_pipeline_plan produces a valid PipelinePlan on toy model
# ---------------------------------------------------------------------------
class TestBuildPipelinePlan(unittest.TestCase):
    def _make_layers(self, n, hidden=8):
        return [nn.Linear(hidden, hidden) for _ in range(n)]

    def test_plan_covers_all_layers(self):
        """All layers must be assigned to exactly one stage."""
        from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan

        num_layers = 4
        num_devices = 4
        layers = self._make_layers(num_layers)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        # Mock both get_compute_cost and alpa_dp so the test is independent of
        # submesh enumeration details and DP correctness (tested separately above).
        # The mock solution: 2 stages, layers[0:2] and layers[2:4], submesh index 0.
        mock_solution = [((0, 2), 0, 0), ((2, 4), 0, 0)]
        # Shape must match what alpa_dp would receive: (K, K+1, M, 1).
        # With uniform_tp_degree=2, get_submesh_choices(1,4) filtered gives 2 submeshes.
        fake_cost = np.ones((num_layers, num_layers + 1, 2, 1), dtype=np.float32)

        with patch("colossalai.auto_parallel.pipeline_shard.orchestrator.get_compute_cost", return_value=fake_cost):
            with patch(
                "colossalai.auto_parallel.pipeline_shard.orchestrator.alpa_dp", return_value=(4.0, mock_solution)
            ):
                plan = build_pipeline_plan(
                    layers=layers,
                    meta_args=meta_args,
                    num_devices=num_devices,
                    num_microbatches=4,
                    num_hosts=1,
                    devices_per_host=4,
                    uniform_tp_degree=2,
                    mesh_alpha=[1e-5, 1e-5],
                    mesh_beta=[1e-11, 1e-11],
                )

        # Check all layers are covered once with no gaps.
        sorted_ranges = sorted(plan.stage_layer_ranges)
        prev_end = 0
        for start, end in sorted_ranges:
            self.assertEqual(start, prev_end, f"Gap between layers: expected start={prev_end}, got {start}.")
            prev_end = end
        self.assertEqual(prev_end, num_layers, f"Stages should cover all {num_layers} layers.")

    def test_plan_dimensions_are_consistent(self):
        """pp_size * tp_size * dp_size must equal num_devices."""
        from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan

        num_layers = 4
        num_devices = 4
        layers = self._make_layers(num_layers)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        # 2 stages × submesh(1,2) = 2 devices each = 4 total.
        # filtered_submeshes[0] = (1,2): n_rows=1, n_cols=2 → tp_size=2, dp_size=4/(2*1*2)=1
        mock_solution = [((0, 2), 0, 0), ((2, 4), 0, 0)]
        fake_cost = np.ones((num_layers, num_layers + 1, 2, 1), dtype=np.float32)

        with patch("colossalai.auto_parallel.pipeline_shard.orchestrator.get_compute_cost", return_value=fake_cost):
            with patch(
                "colossalai.auto_parallel.pipeline_shard.orchestrator.alpa_dp", return_value=(4.0, mock_solution)
            ):
                plan = build_pipeline_plan(
                    layers=layers,
                    meta_args=meta_args,
                    num_devices=num_devices,
                    num_microbatches=4,
                    num_hosts=1,
                    devices_per_host=4,
                    uniform_tp_degree=2,
                    mesh_alpha=[1e-5, 1e-5],
                    mesh_beta=[1e-11, 1e-11],
                )

        # pp=2, tp=2, dp=1 → product=4 == num_devices
        product = plan.pp_size * plan.tp_size * plan.dp_size
        self.assertEqual(
            product,
            num_devices,
            f"pp={plan.pp_size} * tp={plan.tp_size} * dp={plan.dp_size} = {product} != {num_devices}",
        )

    def test_invalid_tp_degree_raises(self):
        """build_pipeline_plan should raise ValueError for a TP degree not in submesh_choices."""
        from colossalai.auto_parallel.pipeline_shard.orchestrator import build_pipeline_plan

        layers = self._make_layers(4)
        meta_args = {"hidden_states": torch.empty(1, 8, device="meta")}

        with self.assertRaises(ValueError):
            build_pipeline_plan(
                layers=layers,
                meta_args=meta_args,
                num_devices=4,
                num_microbatches=4,
                num_hosts=1,
                devices_per_host=4,
                uniform_tp_degree=99,  # impossible TP degree
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
