# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.auto_parallel.pipeline_shard.compute_cost import _StageModule, get_compute_cost
from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
from colossalai.cluster import ProcessGroupMesh
from colossalai.device.alpha_beta_profiler import AlphaBetaProfiler
from colossalai.device.calc_pipeline_strategy import alpa_dp, get_submesh_choices
from colossalai.device.device_mesh import DeviceMesh
from colossalai.pipeline.stage_manager import PipelineStageManager


@dataclass
class PipelinePlan:
    """Optimal 3D parallel plan produced by build_pipeline_plan().

    Attributes:
        stage_layer_ranges: [(start, end), ...] — layer index range per stage (end exclusive).
        submesh_per_stage: (rows, cols) submesh shape per stage (all identical in uniform-TP mode).
        pp_size: Number of pipeline stages.
        tp_size: Tensor-parallel degree (uniform across all stages).
        dp_size: Data-parallel degree.
        estimated_cost: alpa_dp total estimated time (pipeline makespan in seconds).
    """

    stage_layer_ranges: List[Tuple[int, int]] = field(default_factory=list)
    submesh_per_stage: List[Tuple[int, int]] = field(default_factory=list)
    pp_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    estimated_cost: float = float("inf")


def build_pipeline_plan(
    layers: List[nn.Module],
    meta_args: Dict[str, torch.Tensor],
    num_devices: int,
    num_microbatches: int,
    num_hosts: int = 1,
    devices_per_host: int = -1,
    uniform_tp_degree: Optional[int] = None,
    mesh_alpha: Optional[List[float]] = None,
    mesh_beta: Optional[List[float]] = None,
    memory_budget: float = -1.0,
    solver_preference: str = "standard",
    dataloader_option: str = "replicated",
    shard_option: str = "standard",
    cache_path: Optional[str] = None,
) -> PipelinePlan:
    """Find the optimal PP + TP + DP plan for a homogeneous-layer model.

    This is a pure-Python function: it runs ILP solvers but requires no live
    distributed process groups. Call it before torch.distributed.init_process_group()
    if desired, or on a single driver process to produce a plan to broadcast.

    Args:
        layers: Identical repeating layer modules (e.g. transformer blocks).
        meta_args: Input tensor shapes for one layer.
        num_devices: Total GPU count (== world_size).
        num_microbatches: Pipeline microbatch count B. Larger B → smaller bubble fraction.
        num_hosts: Number of machines. Used to determine submesh choices.
        devices_per_host: GPUs per machine. Inferred from num_devices / num_hosts if -1.
        uniform_tp_degree: Fix TP degree for all stages. None = search all valid TP degrees.
        mesh_alpha: Alpha (latency) per mesh axis. Defaults to [1e-5, 1e-5] if None.
        mesh_beta: Beta (inv-bandwidth) per mesh axis. Defaults to [1e-11, 1e-11] if None.
        memory_budget: Per-device memory cap in bytes. -1.0 = unlimited.
        cache_path: Path prefix for caching per-TP cost tables (e.g. '/tmp/my_model').
                    Files saved as '<cache_path>_tp<N>.pkl'.

    Returns:
        PipelinePlan with stage assignments, submesh per stage, pp/tp/dp sizes, and cost.

    Raises:
        ValueError: If no valid submesh matches the requested uniform_tp_degree.
        RuntimeError: If alpa_dp finds no feasible plan (try more devices or fewer layers).
    """
    num_layers = len(layers)

    if devices_per_host == -1:
        assert num_devices % num_hosts == 0, (
            f"num_devices ({num_devices}) must be divisible by num_hosts ({num_hosts})"
        )
        devices_per_host = num_devices // num_hosts

    # get_submesh_choices requires power-of-2 devices_per_host.
    assert devices_per_host & (devices_per_host - 1) == 0, (
        f"devices_per_host must be a power of 2, got {devices_per_host}. "
        "Adjust num_hosts so that num_devices / num_hosts is a power of 2."
    )

    if mesh_alpha is None:
        mesh_alpha = [1e-5, 1e-5]
    if mesh_beta is None:
        mesh_beta = [1e-11, 1e-11]

    all_submeshes = get_submesh_choices(num_hosts, devices_per_host)

    # Determine which TP degrees to search over.
    if uniform_tp_degree is not None:
        tp_candidates = [uniform_tp_degree]
        filtered = [s for s in all_submeshes if int(s[1]) == uniform_tp_degree]
        if not filtered:
            raise ValueError(
                f"No submesh with TP degree {uniform_tp_degree} in {all_submeshes}. "
                "Check num_hosts / devices_per_host."
            )
    else:
        # Search every TP degree independently; keep the globally cheapest plan.
        tp_candidates = sorted({int(s[1]) for s in all_submeshes})

    best_plan = None
    best_cost = float("inf")

    for tp in tp_candidates:
        filtered_submeshes = [s for s in all_submeshes if int(s[1]) == tp]
        if not filtered_submeshes:
            continue

        cost_table = get_compute_cost(
            layers=layers,
            meta_args=meta_args,
            submesh_choices=filtered_submeshes,
            mesh_alpha=mesh_alpha,
            mesh_beta=mesh_beta,
            memory_budget=memory_budget,
            solver_preference=solver_preference,
            dataloader_option=dataloader_option,
            shard_option=shard_option,
            cache_path=f"{cache_path}_tp{tp}.pkl" if cache_path else None,
        )

        cost, solution = alpa_dp(
            num_layers=num_layers,
            num_devices=num_devices,
            num_microbatches=num_microbatches,
            submesh_choices=filtered_submeshes,
            num_autosharding_configs=1,
            compute_cost=cost_table,
        )

        if cost < best_cost:
            best_cost = cost
            best_plan = (solution, filtered_submeshes, tp)

    if best_plan is None or best_plan[0] is None:
        raise RuntimeError(
            "alpa_dp found no feasible pipeline plan. "
            "Try increasing num_devices, reducing num_layers, or relaxing memory_budget."
        )

    solution, chosen_submeshes, chosen_tp = best_plan

    # Parse alpa_dp solution into PipelinePlan fields.
    # solution is a list of ((start_layer, end_layer), submesh_idx, config_idx) per stage.
    stage_layer_ranges = []
    submesh_per_stage = []
    for (start, end), submesh_idx, _ in solution:
        stage_layer_ranges.append((int(start), int(end)))
        submesh_per_stage.append(chosen_submeshes[submesh_idx])

    pp_size = len(stage_layer_ranges)
    # In uniform-TP mode, all stages have the same submesh (rows × cols).
    first_submesh = submesh_per_stage[0]
    n_rows, n_cols = int(first_submesh[0]), int(first_submesh[1])
    # tp_size = number of devices along the second (column) axis of the submesh.
    tp_size = n_cols
    # dp_size fills the remaining devices after accounting for pp and tp.
    dp_size = num_devices // (pp_size * n_rows * n_cols)
    if dp_size < 1:
        dp_size = 1

    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        submesh_per_stage=submesh_per_stage,
        pp_size=pp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        estimated_cost=float(best_cost),
    )


def autoparallelize_with_pp(
    layers: List[nn.Module],
    meta_args: Dict[str, torch.Tensor],
    num_microbatches: int,
    uniform_tp_degree: Optional[int] = None,
    memory_budget: float = -1.0,
    solver_preference: str = "standard",
    dataloader_option: str = "replicated",
    shard_option: str = "standard",
    cache_path: Optional[str] = None,
    plan: Optional[PipelinePlan] = None,
) -> Tuple[nn.Module, PipelineStageManager, PipelinePlan]:
    """Auto 3D parallelism: find and apply a PP + TP + DP plan.

    Must be called after torch.distributed.init_process_group(). Each rank returns
    a ModuleWrapper for its own pipeline stage, sharded with the optimal TP+DP strategy.

    Process groups created:
      - PP groups: managed by PipelineStageManager (P2P send/recv between stages).
      - TP groups: managed by the per-stage DeviceMesh (col axis of the 3D mesh).
      - DP groups: managed by the per-stage DeviceMesh (row axis of the 3D mesh).

    All ranks create DeviceMeshes for ALL stages so that dist.new_group() calls are
    consistent across the process group (PyTorch requirement).

    Args:
        layers: Identical repeating layer modules (e.g. transformer blocks).
        meta_args: Input tensor shapes for one layer.
        num_microbatches: Pipeline microbatch count.
        uniform_tp_degree: Fix TP degree. None = auto-search.
        memory_budget: Per-device memory cap in bytes. -1.0 = unlimited.
        cache_path: Path prefix for caching cost tables (speeds up repeated runs).
        plan: Pre-computed PipelinePlan. Skip planning and go straight to sharding.

    Returns:
        (stage_module, stage_manager, plan)
        - stage_module: ModuleWrapper for this rank's stage (TP+DP sharded).
        - stage_manager: PipelineStageManager for P2P pipeline communication.
        - plan: The PipelinePlan used (inspect for stage assignments / costs).

    Example::

        colossalai.launch_from_torch(config={})
        layers = list(model.transformer.h)   # GPT-2 transformer blocks
        meta_args = {'hidden_states': torch.empty(1, 512, 768, device='meta')}
        stage_mod, stage_mgr, plan = autoparallelize_with_pp(
            layers, meta_args, num_microbatches=4
        )
        # Use stage_mod + stage_mgr in a 1F1B training loop.
    """
    assert dist.is_initialized(), (
        "torch.distributed must be initialized before calling autoparallelize_with_pp(). "
        "Call colossalai.launch_from_torch() first."
    )

    world_size = dist.get_world_size()
    physical_devices = list(range(world_size))

    # ------------------------------------------------------------------ #
    # Step 1: Profile cluster communication costs (α, β).                 #
    # ------------------------------------------------------------------ #
    ab_profiler = AlphaBetaProfiler(physical_devices)
    mesh_alpha, mesh_beta = ab_profiler.extract_alpha_beta_for_device_mesh()

    # ------------------------------------------------------------------ #
    # Step 2: Infer cluster topology (num_hosts, devices_per_host).        #
    # ------------------------------------------------------------------ #
    # Detect topology from profiled α/β: intra-node pairs have lower β
    # (NVLink) than cross-node pairs (InfiniBand). Fall back to single-node
    # if the topology cannot be determined (e.g. homogeneous α/β).
    devices_per_host = _infer_devices_per_host(ab_profiler.alpha_beta_dict, world_size)
    num_hosts = world_size // devices_per_host

    # ------------------------------------------------------------------ #
    # Step 3: Plan (or use a supplied pre-computed plan).                  #
    # ------------------------------------------------------------------ #
    if plan is None:
        plan = build_pipeline_plan(
            layers=layers,
            meta_args=meta_args,
            num_devices=world_size,
            num_microbatches=num_microbatches,
            num_hosts=num_hosts,
            devices_per_host=devices_per_host,
            uniform_tp_degree=uniform_tp_degree,
            mesh_alpha=list(mesh_alpha),
            mesh_beta=list(mesh_beta),
            memory_budget=memory_budget,
            solver_preference=solver_preference,
            dataloader_option=dataloader_option,
            shard_option=shard_option,
            cache_path=cache_path,
        )

    pp_size = plan.pp_size
    tp_size = plan.tp_size
    dp_size = plan.dp_size

    assert pp_size * tp_size * dp_size == world_size, (
        f"Plan (pp={pp_size}, tp={tp_size}, dp={dp_size}) does not multiply to "
        f"world_size={world_size}. This is a bug in build_pipeline_plan()."
    )

    # ------------------------------------------------------------------ #
    # Step 4: Create 3D ProcessGroupMesh.                                  #
    # Layout: axis 0 = PP, axis 1 = TP, axis 2 = DP                       #
    # rank r → (r//(tp*dp), (r%(tp*dp))//dp, r%dp)                        #
    # ------------------------------------------------------------------ #
    pg_mesh = ProcessGroupMesh(pp_size, tp_size, dp_size)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)
    current_stage = stage_manager.stage

    # ------------------------------------------------------------------ #
    # Step 5: Create DeviceMeshes for ALL stages on ALL ranks.             #
    # PyTorch requires every rank to call dist.new_group() for every group #
    # that is created anywhere in the job (even for groups this rank is    #
    # not a member of). Creating all meshes satisfies this invariant.      #
    # ------------------------------------------------------------------ #
    devices_per_stage = tp_size * dp_size
    all_stage_meshes = []
    for s in range(pp_size):
        # Ranks assigned to stage s, ordered as a (tp_size × dp_size) grid.
        stage_ranks = list(range(s * devices_per_stage, (s + 1) * devices_per_stage))
        stage_ranks_t = torch.tensor(stage_ranks)
        mesh = DeviceMesh(
            physical_mesh_id=stage_ranks_t,
            logical_mesh_id=stage_ranks_t.reshape(tp_size, dp_size),
            mesh_alpha=list(mesh_alpha),
            mesh_beta=list(mesh_beta),
            init_process_group=True,  # all ranks participate → dist.new_group is consistent
        )
        all_stage_meshes.append(mesh)

    stage_device_mesh = all_stage_meshes[current_stage]

    # ------------------------------------------------------------------ #
    # Step 6: Shard this rank's stage with TP+DP auto-parallelism.         #
    # ------------------------------------------------------------------ #
    stage_start, stage_end = plan.stage_layer_ranges[current_stage]
    stage_layers = layers[stage_start:stage_end]
    stage_module = _StageModule(stage_layers)

    stage_wrapped = initialize_model(
        model=stage_module,
        meta_args=meta_args,
        device_mesh=stage_device_mesh,
        memory_budget=memory_budget,
        solver_preference=solver_preference,
        dataloader_option=dataloader_option,
        shard_option=shard_option,
    )

    return stage_wrapped, stage_manager, plan


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _infer_devices_per_host(
    alpha_beta_dict: dict,
    world_size: int,
) -> int:
    """Heuristically infer the number of GPUs per host from α/β profiles.

    NVLink-connected pairs (intra-node) have significantly lower β than
    InfiniBand-connected pairs (cross-node). If all betas are similar
    (homogeneous cluster or single-node), fall back to world_size (single host).

    Returns a power-of-2 value in [1, world_size].
    """
    if world_size == 1 or not alpha_beta_dict:
        return world_size

    betas = [v[1] for v in alpha_beta_dict.values() if v[1] > 0]
    if not betas:
        return world_size

    betas_sorted = sorted(betas)
    median_beta = betas_sorted[len(betas_sorted) // 2]
    # Cross-node betas are typically 10× higher than intra-node betas.
    has_cross_node = any(b > median_beta * 3.0 for b in betas_sorted)

    if not has_cross_node:
        # Single-node or homogeneous network.
        return world_size

    # Count intra-node neighbours: ranks with beta < 2× median.
    intra_count = sum(1 for b in betas_sorted if b < median_beta * 2.0)
    # Each rank has (devs_per_host - 1) intra-node neighbours.
    # Total intra-node pairs = world_size * (devs_per_host - 1) / 2.
    # intra_count (total intra-node beta values) ≈ world_size * (devs_per_host - 1)
    # (because alpha_beta_dict is symmetric: (i,j) and (j,i) are both present)
    # => devs_per_host ≈ intra_count / world_size + 1
    estimated = intra_count // world_size + 1

    # Round to nearest power of 2 that divides world_size.
    candidate = 1
    while candidate * 2 <= estimated and (world_size % (candidate * 2)) == 0:
        candidate *= 2

    return max(1, candidate)
