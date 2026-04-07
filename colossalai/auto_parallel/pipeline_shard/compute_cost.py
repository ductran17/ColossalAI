# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from colossalai._analyzer.fx.codegen import ActivationCheckpointCodeGen
from colossalai._analyzer.fx.graph_module import ColoGraphModule
from colossalai._analyzer.fx.passes import shape_prop_pass
from colossalai._analyzer.fx.tracer.tracer import ColoTracer
from colossalai.auto_parallel.tensor_shard.initialize import build_strategy_constructor, solve_solution
from colossalai.device.device_mesh import DeviceMesh


class _StageModule(nn.Module):
    """Wraps a contiguous slice of layers as one pipeline stage.

    Assumptions:
      - Each layer accepts (hidden_states, attention_mask=None).
      - Each layer returns a tensor (hidden_states). If the layer returns a tuple,
        only the first element is forwarded to the next layer.

    This class is used both for ILP cost estimation (tracing with ColoTracer) and
    at runtime as the nn.Module handed to initialize_model().
    """

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        for layer in self.layers:
            out = layer(hidden_states, attention_mask)
            # Unpack tuple outputs (e.g. HuggingFace-style (hidden_states, present_key_value,...))
            # Use concrete check so FX tracing sees a scalar bool, not a Proxy comparison.
            if isinstance(out, (tuple, list)):
                hidden_states = out[0]
            else:
                hidden_states = out
        return hidden_states


def _estimate_stage_cost(
    num_stage_layers: int,
    representative_layers: List[nn.Module],
    meta_args: Dict[str, torch.Tensor],
    device_mesh: DeviceMesh,
    memory_budget: float = -1.0,
    solver_preference: str = "standard",
    dataloader_option: str = "replicated",
    shard_option: str = "standard",
) -> float:
    """Return the ILP objective for a stage with num_stage_layers on device_mesh.

    Uses representative_layers[:num_stage_layers] for tracing. All layers are assumed
    identical (homogeneous), so this is a valid proxy for any stage of that size.

    Returns np.inf if tracing or the ILP solver fails (e.g. memory infeasible).
    """
    stage_module = _StageModule(representative_layers[:num_stage_layers])

    try:
        tracer = ColoTracer(trace_act_ckpt=True, bias_addition_split=True)
        graph = tracer.trace(root=stage_module, meta_args=meta_args)
        graph.set_codegen(ActivationCheckpointCodeGen())
        gm = ColoGraphModule(stage_module, graph, stage_module.__class__.__name__)

        shape_prop_pass(gm, *meta_args.values())
        gm.recompile()

        strategies_constructor = build_strategy_constructor(
            graph,
            device_mesh,
            solver_preference=solver_preference,
            dataloader_option=dataloader_option,
            shard_option=shard_option,
        )
        _, objective = solve_solution(gm, strategies_constructor, memory_budget)
        return float(objective)
    except Exception:
        # ILP infeasible or tracing failed for this (stage_size, submesh) combination.
        return float(np.inf)


def get_compute_cost(
    layers: List[nn.Module],
    meta_args: Dict[str, torch.Tensor],
    submesh_choices: List[Tuple[int, int]],
    mesh_alpha: List[float],
    mesh_beta: List[float],
    memory_budget: float = -1.0,
    solver_preference: str = "standard",
    dataloader_option: str = "replicated",
    shard_option: str = "standard",
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """Build the compute cost table required by alpa_dp().

    Assumes homogeneous layers (all identical, e.g. transformer blocks). The cost
    of a stage with s layers on submesh m depends only on s and m, not on which
    specific layers are in the stage. This reduces ILP calls from O(K^2 * M) to O(K * M).

    This function is pure Python — it runs the ILP solver with a dummy DeviceMesh
    (init_process_group=False) and requires no active distributed process groups.
    It can be called before launching distributed training, or on any single process.

    Args:
        layers: List of identical repeating layer modules (e.g. transformer blocks).
                All layers must accept (hidden_states, attention_mask=None).
        meta_args: Input tensor shapes for one layer, e.g.
                   {'hidden_states': torch.empty(1, 512, 768, device='meta')}.
        submesh_choices: Output of get_submesh_choices(), optionally filtered by TP degree.
        mesh_alpha: Alpha (latency) per mesh axis, e.g. [1e-5, 1e-4].
        mesh_beta: Beta (inverse bandwidth) per mesh axis, e.g. [1e-11, 1e-10].
        memory_budget: Per-device memory limit in bytes. -1.0 = unlimited.
        cache_path: If provided, load cost table from this file if it exists,
                    otherwise compute and save to this file.

    Returns:
        cost: np.ndarray of shape (num_layers, num_layers+1, num_submeshes, 1).
              cost[k, i, m, 0] = estimated time for a stage covering layers[k:i] on submesh m.
              np.inf means the combination is infeasible.
    """
    num_layers = len(layers)
    num_submeshes = len(submesh_choices)

    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            cost = pickle.load(fh)
        expected_shape = (num_layers, num_layers + 1, num_submeshes, 1)
        assert cost.shape == expected_shape, (
            f"Cached cost table shape {cost.shape} does not match expected {expected_shape}. "
            "Delete the cache file and rerun."
        )
        return cost

    cost = np.full((num_layers, num_layers + 1, num_submeshes, 1), np.inf, dtype=np.float32)

    for m, submesh in enumerate(submesh_choices):
        n_rows, n_cols = int(submesh[0]), int(submesh[1])
        n_devices = n_rows * n_cols

        # Dummy DeviceMesh: no real process groups, only mesh geometry and α/β for cost estimation.
        fake_ranks = torch.arange(n_devices)
        device_mesh = DeviceMesh(
            physical_mesh_id=fake_ranks,
            logical_mesh_id=fake_ranks.reshape(n_rows, n_cols),
            mesh_alpha=mesh_alpha,
            mesh_beta=mesh_beta,
            init_process_group=False,
        )

        # For each stage size, estimate cost once and broadcast across all (k, k+s) pairs.
        for stage_size in range(1, num_layers + 1):
            stage_cost = _estimate_stage_cost(
                num_stage_layers=stage_size,
                representative_layers=layers,
                meta_args=meta_args,
                device_mesh=device_mesh,
                memory_budget=memory_budget,
                solver_preference=solver_preference,
                dataloader_option=dataloader_option,
                shard_option=shard_option,
            )
            # Fill cost[k, k+stage_size, m, 0] for every valid starting layer k.
            for k in range(num_layers - stage_size + 1):
                cost[k, k + stage_size, m, 0] = stage_cost

    if cache_path is not None:
        parent_dir = os.path.dirname(os.path.abspath(cache_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(cache_path, "wb") as fh:
            pickle.dump(cost, fh)

    return cost


def get_boundary_cost_table(
    submesh_choices: List[Tuple[int, int]],
    activation_bytes: int,
    mesh_alpha: List[float],
    mesh_beta: List[float],
) -> np.ndarray:
    """Compute the (M, M) table of boundary resharding costs between submeshes.

    boundary_cost_table[m1, m2] = estimated time to reshard activations from
    the output format of submesh m1 (TP=cols1) to the input format of submesh
    m2 (TP=cols2) at a pipeline stage boundary.

    When cols1 == cols2: cost = 0 (Phase 1, no resharding needed).
    When cols1 > 1 (sender must AllGather before P2P send):
        cost = alpha + beta * activation_bytes * (1 - 1/cols1)
        (AllGather communicates all-but-one shard across the TP group)
    When cols1 == 1 and cols2 > 1 (receiver Splits after P2P recv):
        cost = 0 (pure tensor op, no communication)

    Args:
        submesh_choices: List of (rows, cols) submesh shapes.
        activation_bytes: Size in bytes of the activation tensor at the boundary
                          (e.g. batch * seq_len * hidden * element_size).
        mesh_alpha: Latency constants per mesh axis [alpha_axis0, alpha_axis1].
        mesh_beta: Inverse-bandwidth constants per mesh axis.

    Returns:
        cost_table: np.ndarray of shape (M, M), dtype float32.
    """
    M = len(submesh_choices)
    cost_table = np.zeros((M, M), dtype=np.float32)

    # Use intra-node (axis 1 / cols axis) alpha and beta for TP AllGather.
    # Fall back to axis 0 if only one axis provided.
    alpha = mesh_alpha[1] if len(mesh_alpha) > 1 else mesh_alpha[0]
    beta = mesh_beta[1] if len(mesh_beta) > 1 else mesh_beta[0]

    for m1, s1 in enumerate(submesh_choices):
        tp1 = int(s1[1])  # cols = TP degree of sender
        for m2, s2 in enumerate(submesh_choices):
            tp2 = int(s2[1])  # cols = TP degree of receiver
            if tp1 == tp2:
                cost_table[m1, m2] = 0.0
            elif tp1 > 1:
                # Sender must AllGather: communicates (tp1-1)/tp1 of the tensor.
                bytes_moved = activation_bytes * (1.0 - 1.0 / tp1)
                cost_table[m1, m2] = float(alpha + beta * bytes_moved)
            else:
                # Sender tp1=1 (full tensor already), receiver Splits locally.
                cost_table[m1, m2] = 0.0

    return cost_table
