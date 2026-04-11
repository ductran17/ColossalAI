# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team

from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from colossalai.auto_parallel.pipeline_shard.boundary_resharding import BoundaryReshardingModule
from colossalai.auto_parallel.pipeline_shard.compute_cost import (
    _StageModule,
    get_boundary_cost_table,
    get_compute_cost,
)
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
        submesh_per_stage: (rows, cols) submesh shape per stage.
        tp_per_stage: TP degree per stage. All equal in uniform-TP (Phase 1) mode.
        pp_size: Number of pipeline stages.
        tp_size: Maximum TP degree across all stages.
        dp_size: Data-parallel degree (for the stage with maximum TP).
        estimated_cost: alpa_dp total estimated time (pipeline makespan).
        heterogeneous_tp: True when adjacent stages have different TP degrees (Phase 2).
        send_boundary_modules: {stage_idx: BoundaryReshardingModule} applied before P2P send.
                                Populated by autoparallelize_with_pp() when heterogeneous_tp=True.
        recv_boundary_modules: {stage_idx: BoundaryReshardingModule} applied after P2P recv.
                                Populated by autoparallelize_with_pp() when heterogeneous_tp=True.
    """

    stage_layer_ranges: List[Tuple[int, int]] = field(default_factory=list)
    submesh_per_stage: List[Tuple[int, int]] = field(default_factory=list)
    tp_per_stage: List[int] = field(default_factory=list)
    dp_per_stage: List[int] = field(default_factory=list)
    rank_ranges: List[List[int]] = field(default_factory=list)
    pp_size: int = 1
    tp_size: int = 1
    dp_size: int = 1
    estimated_cost: float = float("inf")
    heterogeneous_tp: bool = False
    variable_stage_sizes: bool = False
    send_boundary_modules: dict = field(default_factory=dict)
    recv_boundary_modules: dict = field(default_factory=dict)


def build_pipeline_plan(
    layers: List[nn.Module],
    meta_args: Dict[str, torch.Tensor],
    num_devices: int,
    num_microbatches: int,
    num_hosts: int = 1,
    devices_per_host: int = -1,
    uniform_tp_degree: Optional[int] = None,
    heterogeneous_tp: bool = False,
    variable_stage_sizes: bool = False,
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

    Phase 1 (heterogeneous_tp=False, default):
        Searches per TP degree with all stages constrained to the same TP.
        No boundary resharding needed at stage transitions.

    Phase 2 (heterogeneous_tp=True):
        Searches all submeshes simultaneously. Different stages may have
        different TP degrees. Adds boundary cost penalties to the cost table
        so the DP accounts for AllGather overhead at TP-mismatched boundaries.
        BoundaryReshardingModules are created by autoparallelize_with_pp().

    Args:
        layers: Identical repeating layer modules (e.g. transformer blocks).
        meta_args: Input tensor shapes for one layer.
        num_devices: Total GPU count (== world_size).
        num_microbatches: Pipeline microbatch count B. Larger B → smaller bubble fraction.
        num_hosts: Number of machines. Used to determine submesh choices.
        devices_per_host: GPUs per machine. Inferred from num_devices / num_hosts if -1.
        uniform_tp_degree: Fix TP degree for all stages. None = search all valid TP degrees.
        heterogeneous_tp: Phase 2 mode — allow different TP per stage.
        mesh_alpha: Alpha (latency) per mesh axis. Defaults to [1e-5, 1e-5] if None.
        mesh_beta: Beta (inv-bandwidth) per mesh axis. Defaults to [1e-11, 1e-11] if None.
        memory_budget: Per-device memory cap in bytes. -1.0 = unlimited.
        cache_path: Path prefix for caching per-TP cost tables (e.g. '/tmp/my_model').
                    Files saved as '<cache_path>_tp<N>.pkl' (Phase 1) or
                    '<cache_path>_hetero.pkl' (Phase 2).

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

    # ------------------------------------------------------------------
    # Phase 2b: variable-size stages — search all submeshes simultaneously,
    # allow different device counts per stage.
    # ------------------------------------------------------------------
    if variable_stage_sizes and uniform_tp_degree is None:
        cost_table = get_compute_cost(
            layers=layers,
            meta_args=meta_args,
            submesh_choices=all_submeshes,
            mesh_alpha=mesh_alpha,
            mesh_beta=mesh_beta,
            memory_budget=memory_budget,
            solver_preference=solver_preference,
            dataloader_option=dataloader_option,
            shard_option=shard_option,
            cache_path=f"{cache_path}_varstage.pkl" if cache_path else None,
        )

        activation_bytes = _estimate_activation_bytes(meta_args)
        boundary_table = get_boundary_cost_table(
            all_submeshes, activation_bytes, mesh_alpha, mesh_beta
        )
        min_incoming = boundary_table.min(axis=0)
        for m in range(len(all_submeshes)):
            cost_table[:, :, m, 0] += min_incoming[m]

        cost, solution = alpa_dp(
            num_layers=num_layers,
            num_devices=num_devices,
            num_microbatches=num_microbatches,
            submesh_choices=all_submeshes,
            num_autosharding_configs=1,
            compute_cost=cost_table,
        )

        if solution is None:
            raise RuntimeError(
                "alpa_dp found no feasible variable-stage plan. "
                "Try increasing num_devices or reducing num_layers."
            )

        plan = _parse_solution_variable(
            solution, all_submeshes, num_devices, float(cost)
        )
        if plan is None:
            raise RuntimeError(
                "alpa_dp solution has non-uniform dp across stages. "
                "Phase 2b (variable_stage_sizes) requires uniform dp across all stages. "
                "Try more devices or adjust the model."
            )
        return plan

    # ------------------------------------------------------------------
    # Phase 2: heterogeneous TP — search all submeshes simultaneously.
    # ------------------------------------------------------------------
    if heterogeneous_tp and uniform_tp_degree is None:
        cost_table = get_compute_cost(
            layers=layers,
            meta_args=meta_args,
            submesh_choices=all_submeshes,
            mesh_alpha=mesh_alpha,
            mesh_beta=mesh_beta,
            memory_budget=memory_budget,
            solver_preference=solver_preference,
            dataloader_option=dataloader_option,
            shard_option=shard_option,
            cache_path=f"{cache_path}_hetero.pkl" if cache_path else None,
        )

        # Add boundary cost penalty: for each (k,i,m), add the minimum
        # boundary cost that could arise when transitioning to submesh m.
        # This guides alpa_dp away from plans with expensive TP transitions.
        activation_bytes = _estimate_activation_bytes(meta_args)
        boundary_table = get_boundary_cost_table(
            all_submeshes, activation_bytes, mesh_alpha, mesh_beta
        )
        # For each submesh m, the cheapest incoming boundary cost is
        # min over all m' of boundary_table[m', m].
        min_incoming = boundary_table.min(axis=0)  # shape (M,)
        for m in range(len(all_submeshes)):
            cost_table[:, :, m, 0] += min_incoming[m]

        cost, solution = alpa_dp(
            num_layers=num_layers,
            num_devices=num_devices,
            num_microbatches=num_microbatches,
            submesh_choices=all_submeshes,
            num_autosharding_configs=1,
            compute_cost=cost_table,
        )

        if solution is None:
            raise RuntimeError(
                "alpa_dp found no feasible heterogeneous pipeline plan. "
                "Try increasing num_devices or reducing num_layers."
            )

        return _parse_solution(solution, all_submeshes, num_devices, float(cost),
                               heterogeneous_tp=True)

    # ------------------------------------------------------------------
    # Phase 1: uniform TP — search per TP degree, keep cheapest plan.
    # ------------------------------------------------------------------
    if uniform_tp_degree is not None:
        tp_candidates = [uniform_tp_degree]
        filtered = [s for s in all_submeshes if int(s[1]) == uniform_tp_degree]
        if not filtered:
            raise ValueError(
                f"No submesh with TP degree {uniform_tp_degree} in {all_submeshes}. "
                "Check num_hosts / devices_per_host."
            )
    else:
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

    solution, chosen_submeshes, _ = best_plan
    return _parse_solution(solution, chosen_submeshes, num_devices, float(best_cost),
                           heterogeneous_tp=False)


def _estimate_activation_bytes(meta_args: Dict[str, torch.Tensor]) -> int:
    """Estimate activation tensor size in bytes from meta_args.

    Uses the first tensor in meta_args (typically hidden_states).
    Assumes float16 (2 bytes per element) as the typical training dtype.
    """
    for t in meta_args.values():
        n_elements = 1
        for d in t.shape:
            n_elements *= d
        return n_elements * 2  # float16 = 2 bytes
    return 1  # fallback


def _parse_solution(
    solution: list,
    submesh_choices: List[Tuple[int, int]],
    num_devices: int,
    cost: float,
    heterogeneous_tp: bool,
) -> "PipelinePlan":
    """Parse an alpa_dp solution list into a PipelinePlan."""
    stage_layer_ranges = []
    submesh_per_stage = []
    tp_per_stage = []

    for (start, end), submesh_idx, _ in solution:
        stage_layer_ranges.append((int(start), int(end)))
        submesh = submesh_choices[submesh_idx]
        submesh_per_stage.append(submesh)
        tp_per_stage.append(int(submesh[1]))  # cols = TP degree

    pp_size = len(stage_layer_ranges)

    # tp_size = max TP degree (for ProcessGroupMesh when uniform; per-stage when hetero).
    tp_size = max(tp_per_stage) if tp_per_stage else 1

    # dp_size from first stage submesh (rows * dp within stage).
    first_submesh = submesh_per_stage[0]
    n_rows, n_cols = int(first_submesh[0]), int(first_submesh[1])
    dp_size = num_devices // (pp_size * n_rows * n_cols)
    if dp_size < 1:
        dp_size = 1

    # Detect if any adjacent stages have different TP degrees.
    actual_hetero = heterogeneous_tp and any(
        tp_per_stage[i] != tp_per_stage[i + 1] for i in range(len(tp_per_stage) - 1)
    )

    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        submesh_per_stage=submesh_per_stage,
        tp_per_stage=tp_per_stage,
        pp_size=pp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        estimated_cost=cost,
        heterogeneous_tp=actual_hetero,
    )


def _parse_solution_variable(
    solution: list,
    submesh_choices: List[Tuple[int, int]],
    num_devices: int,
    cost: float,
) -> Optional["PipelinePlan"]:
    """Parse an alpa_dp solution with variable per-stage device counts (Phase 2b).

    Returns None if the solution has non-uniform dp across stages — uniform dp
    is required so that CrossMeshP2PCommunication can align dp replicas across
    stage boundaries.
    """
    stage_layer_ranges = []
    submesh_per_stage = []
    tp_per_stage = []
    device_counts = []

    for (start, end), submesh_idx, _ in solution:
        stage_layer_ranges.append((int(start), int(end)))
        submesh = submesh_choices[submesh_idx]
        submesh_per_stage.append(submesh)
        tp_per_stage.append(int(submesh[1]))
        device_counts.append(int(prod(int(x) for x in submesh)))

    pp_size = len(stage_layer_ranges)
    dp_per_stage = [d // t for d, t in zip(device_counts, tp_per_stage)]

    if len(set(dp_per_stage)) > 1:
        return None  # non-uniform dp — not supported

    # Assign contiguous rank ranges: stage s owns ranks [offset, offset + device_counts[s])
    rank_ranges: List[List[int]] = []
    offset = 0
    for n in device_counts:
        rank_ranges.append(list(range(offset, offset + n)))
        offset += n

    assert offset == num_devices, (
        f"Stage device counts sum to {offset}, expected {num_devices}."
    )

    tp_size = max(tp_per_stage) if tp_per_stage else 1
    dp_size = dp_per_stage[0]
    actual_hetero = any(
        tp_per_stage[i] != tp_per_stage[i + 1] for i in range(pp_size - 1)
    )

    return PipelinePlan(
        stage_layer_ranges=stage_layer_ranges,
        submesh_per_stage=submesh_per_stage,
        tp_per_stage=tp_per_stage,
        dp_per_stage=dp_per_stage,
        rank_ranges=rank_ranges,
        pp_size=pp_size,
        tp_size=tp_size,
        dp_size=dp_size,
        estimated_cost=cost,
        heterogeneous_tp=actual_hetero,
        variable_stage_sizes=True,
    )


class VariableStagePipelineManager:
    """Pipeline stage manager for variable-size stages (Phase 2b).

    Replaces PipelineStageManager when different pipeline stages own different
    numbers of devices. Stage membership is determined directly from
    plan.rank_ranges — no ProcessGroupMesh required.

    Provides the same interface as PipelineStageManager used by the training
    loop: .stage, .num_stages, .is_first_stage(), .is_last_stage().
    """

    def __init__(self, plan: "PipelinePlan", rank: int) -> None:
        self._plan = plan
        self._rank = rank
        self._pp_size = plan.pp_size
        self._stage: int = next(
            s for s, rr in enumerate(plan.rank_ranges) if rank in rr
        )

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def num_stages(self) -> int:
        return self._pp_size

    def is_first_stage(self, ignore_chunk: bool = False) -> bool:
        return self._stage == 0

    def is_last_stage(self, ignore_chunk: bool = False) -> bool:
        return self._stage == self._pp_size - 1

    def get_rank(self) -> int:
        return self._rank


def autoparallelize_with_pp(
    layers: List[nn.Module],
    meta_args: Dict[str, torch.Tensor],
    num_microbatches: int,
    uniform_tp_degree: Optional[int] = None,
    heterogeneous_tp: bool = False,
    variable_stage_sizes: bool = False,
    memory_budget: float = -1.0,
    solver_preference: str = "standard",
    dataloader_option: str = "replicated",
    shard_option: str = "standard",
    cache_path: Optional[str] = None,
    plan: Optional[PipelinePlan] = None,
    skip_profile: bool = False,
) -> Tuple[nn.Module, PipelineStageManager, PipelinePlan]:
    """Auto 3D parallelism: find and apply a PP + TP + DP plan.

    Must be called after torch.distributed.init_process_group(). Each rank returns
    a ModuleWrapper for its own pipeline stage, sharded with the optimal TP+DP strategy.

    Phase 1 (heterogeneous_tp=False, default):
        All stages share one TP degree. Stage boundaries require no resharding.

    Phase 2 (heterogeneous_tp=True):
        Stages may use different TP degrees. BoundaryReshardingModules are
        created for each stage boundary where TP degrees differ and stored in
        plan.send_boundary_modules and plan.recv_boundary_modules.
        The training loop must apply these before/after P2P send/recv.

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
        heterogeneous_tp: Phase 2 mode — allow different TP per stage.
        memory_budget: Per-device memory cap in bytes. -1.0 = unlimited.
        cache_path: Path prefix for caching cost tables (speeds up repeated runs).
        plan: Pre-computed PipelinePlan. Skip planning and go straight to sharding.

    Returns:
        (stage_module, stage_manager, plan)
        - stage_module: ModuleWrapper for this rank's stage (TP+DP sharded).
        - stage_manager: PipelineStageManager for P2P pipeline communication.
        - plan: The PipelinePlan used. In Phase 2, plan.send_boundary_modules
                and plan.recv_boundary_modules contain the resharding ops.
    """
    assert dist.is_initialized(), (
        "torch.distributed must be initialized before calling autoparallelize_with_pp(). "
        "Call colossalai.launch_from_torch() first."
    )

    world_size = dist.get_world_size()
    physical_devices = list(range(world_size))

    # ------------------------------------------------------------------ #
    # Step 1: Profile cluster communication costs (α, β).                 #
    # AlphaBetaProfiler.extract_alpha_beta_for_device_mesh() requires a   #
    # power-of-2 world size. For non-power-of-2 sizes fall back to the    #
    # measured median α/β from the profiler's raw dict.                   #
    #                                                                      #
    # Use warmup=1, repeat=3 instead of defaults (warmup=5, repeat=25)    #
    # to keep profiling under ~15s even on slow Ethernet links.            #
    # ------------------------------------------------------------------ #
    def _is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    if skip_profile:
        # Build a synthetic α/β dict using typical Ethernet values so the
        # planner still distinguishes intra- vs inter-node links.
        # All pairs get the same value; topology inference will fall back to
        # treating the cluster as one flat mesh (conservative but correct).
        _fake_alpha = 8e-5   # 80 µs — typical Ethernet latency
        _fake_beta  = 8e-11  # 1 / (12.5 GB/s) — typical 100 GbE
        _fake_ab = {(i, j): (_fake_alpha, _fake_beta)
                    for i in physical_devices for j in physical_devices if i != j}
        ab_profiler = AlphaBetaProfiler(physical_devices, alpha_beta_dict=_fake_ab)
    else:
        ab_profiler = AlphaBetaProfiler(physical_devices, warmup=1, repeat=3)

    # ------------------------------------------------------------------
    # Synchronise α/β measurements across all ranks.
    #
    # AlphaBetaProfiler measures from each rank's local perspective, so
    # alpha_beta_dict differs between ranks.  Alpa's design collects all
    # measurements centrally before planning; we approximate that here by
    # all-reducing the pair tensors with min() (conservative: take the
    # faster of the two endpoints' measurements for each pair).
    #
    # This makes the α/β dict — and therefore every downstream quantity
    # (topology inference, mesh_alpha/beta, build_pipeline_plan) —
    # identical on all ranks, so the plan is deterministic everywhere.
    # ------------------------------------------------------------------
    _n = world_size
    _pairs = [(i, j) for i in range(_n) for j in range(_n) if i != j]
    _local_ab = ab_profiler.alpha_beta_dict
    # Pack into two flat CUDA tensors [num_pairs] — one for alpha, one for beta.
    _alpha_t = torch.tensor(
        [_local_ab.get((i, j), (1e-5, 1e-11))[0] for (i, j) in _pairs],
        dtype=torch.float64, device="cuda",
    )
    _beta_t = torch.tensor(
        [_local_ab.get((i, j), (1e-5, 1e-11))[1] for (i, j) in _pairs],
        dtype=torch.float64, device="cuda",
    )
    # all_reduce with MIN: take the most optimistic (fastest) measurement
    # across all ranks for each pair — matches Alpa's "best observed" policy.
    dist.all_reduce(_alpha_t, op=dist.ReduceOp.MIN)
    dist.all_reduce(_beta_t,  op=dist.ReduceOp.MIN)
    _synced_ab = {
        (i, j): (_alpha_t[k].item(), _beta_t[k].item())
        for k, (i, j) in enumerate(_pairs)
    }
    ab_profiler.alpha_beta_dict = _synced_ab

    if _is_power_of_two(world_size):
        mesh_alpha, mesh_beta = ab_profiler.extract_alpha_beta_for_device_mesh()
    else:
        ab_vals = list(ab_profiler.alpha_beta_dict.values())
        if ab_vals:
            alphas = sorted(v[0] for v in ab_vals if v[0] > 0)
            betas  = sorted(v[1] for v in ab_vals if v[1] > 0)
            median_alpha = alphas[len(alphas) // 2] if alphas else 1e-5
            median_beta  = betas[len(betas) // 2]   if betas  else 1e-11
        else:
            median_alpha, median_beta = 1e-5, 1e-11
        mesh_alpha = [median_alpha, median_alpha]
        mesh_beta  = [median_beta,  median_beta]

    # ------------------------------------------------------------------ #
    # Step 2: Infer cluster topology (num_hosts, devices_per_host).        #
    # Now that alpha_beta_dict is synchronised, all ranks derive the same  #
    # topology and build_pipeline_plan produces a deterministic plan.      #
    # ------------------------------------------------------------------ #
    devices_per_host = _infer_devices_per_host(ab_profiler.alpha_beta_dict, world_size)
    # get_submesh_choices requires power-of-2 devices_per_host. Round up if needed.
    if not _is_power_of_two(devices_per_host):
        p = 1
        while p < devices_per_host:
            p *= 2
        devices_per_host = p
    num_hosts = max(1, world_size // devices_per_host)

    # ------------------------------------------------------------------ #
    # Step 3: Plan (or use a supplied pre-computed plan).                  #
    # build_pipeline_plan is pure-Python (no distributed). Because the    #
    # α/β dict is now synchronised across all ranks, every rank derives   #
    # the same topology and the same plan — no broadcast needed.           #
    # ------------------------------------------------------------------ #
    rank = dist.get_rank()
    if plan is None:
        plan = build_pipeline_plan(
            layers=layers,
            meta_args=meta_args,
            num_devices=world_size,
            num_microbatches=num_microbatches,
            num_hosts=num_hosts,
            devices_per_host=devices_per_host,
            uniform_tp_degree=uniform_tp_degree,
            heterogeneous_tp=heterogeneous_tp,
            variable_stage_sizes=variable_stage_sizes,
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

    # tp_per_stage may differ in Phase 2/2b; fall back to uniform tp_size for Phase 1.
    tp_per_stage = plan.tp_per_stage if plan.tp_per_stage else [tp_size] * pp_size

    # ------------------------------------------------------------------
    # Phase 2b: variable-size stages — skip ProcessGroupMesh.
    # Each stage may own a different number of devices. Rank-to-stage
    # membership is determined by plan.rank_ranges.
    # ------------------------------------------------------------------
    if plan.variable_stage_sizes:
        current_stage = next(
            s for s, rr in enumerate(plan.rank_ranges) if rank in rr
        )

        # Create per-stage DeviceMeshes using rank_ranges (ALL ranks call for ALL stages).
        all_stage_meshes = []
        for s in range(pp_size):
            tp_s = tp_per_stage[s]
            dp_s = plan.dp_per_stage[s]
            stage_ranks = plan.rank_ranges[s]
            stage_ranks_t = torch.tensor(stage_ranks)
            mesh = DeviceMesh(
                physical_mesh_id=stage_ranks_t,
                logical_mesh_id=stage_ranks_t.reshape(tp_s, dp_s),
                mesh_alpha=list(mesh_alpha),
                mesh_beta=list(mesh_beta),
                init_process_group=True,
            )
            all_stage_meshes.append(mesh)

        stage_device_mesh = all_stage_meshes[current_stage]
        stage_manager = VariableStagePipelineManager(plan, rank)

        # Shard this rank's stage.
        stage_start, stage_end = plan.stage_layer_ranges[current_stage]
        stage_layers = layers[stage_start:stage_end]
        stage_module_raw = _StageModule(stage_layers)
        stage_wrapped = initialize_model(
            model=stage_module_raw,
            meta_args=meta_args,
            device_mesh=stage_device_mesh,
            memory_budget=memory_budget,
            solver_preference=solver_preference,
            dataloader_option=dataloader_option,
            shard_option=shard_option,
        )

        # Build boundary resharding modules for TP-mismatched boundaries.
        if plan.heterogeneous_tp:
            _build_boundary_modules(
                plan, pp_size, tp_per_stage, all_stage_meshes, current_stage,
                rank=rank,
            )

        return stage_wrapped, stage_manager, plan

    # ------------------------------------------------------------------
    # Step 4: Create ProcessGroupMesh for PP axis + TP/DP axes.
    # In Phase 2 (heterogeneous TP), tp_size is the maximum TP degree so
    # the mesh still has a consistent shape. Ranks with lower per-stage TP
    # create the groups but only those with matching TP degrees participate
    # in the TP collectives (handled by per-stage DeviceMesh in Step 5).
    # ------------------------------------------------------------------
    assert pp_size * tp_size * dp_size == world_size, (
        f"Plan (pp={pp_size}, tp={tp_size}, dp={dp_size}) does not multiply to "
        f"world_size={world_size}. This is a bug in build_pipeline_plan()."
    )

    pg_mesh = ProcessGroupMesh(pp_size, tp_size, dp_size)
    stage_manager = PipelineStageManager(pg_mesh, pipeline_axis=0)
    current_stage = stage_manager.stage

    # ------------------------------------------------------------------
    # Step 5: Create DeviceMeshes for ALL stages on ALL ranks.
    # Each stage's mesh shape = (tp_s, dp_s) where tp_s is per-stage TP.
    # All ranks must call dist.new_group() for every group created anywhere.
    # ------------------------------------------------------------------
    devices_per_stage = tp_size * dp_size
    all_stage_meshes = []
    for s in range(pp_size):
        tp_s = tp_per_stage[s]
        dp_s = devices_per_stage // tp_s  # dp within this stage
        stage_ranks = list(range(s * devices_per_stage, (s + 1) * devices_per_stage))
        stage_ranks_t = torch.tensor(stage_ranks)
        mesh = DeviceMesh(
            physical_mesh_id=stage_ranks_t,
            logical_mesh_id=stage_ranks_t.reshape(tp_s, dp_s),
            mesh_alpha=list(mesh_alpha),
            mesh_beta=list(mesh_beta),
            init_process_group=True,
        )
        all_stage_meshes.append(mesh)

    stage_device_mesh = all_stage_meshes[current_stage]

    # ------------------------------------------------------------------
    # Step 6: Shard this rank's stage with TP+DP auto-parallelism.
    # ------------------------------------------------------------------
    stage_start, stage_end = plan.stage_layer_ranges[current_stage]
    stage_layers = layers[stage_start:stage_end]
    stage_module_raw = _StageModule(stage_layers)

    stage_wrapped = initialize_model(
        model=stage_module_raw,
        meta_args=meta_args,
        device_mesh=stage_device_mesh,
        memory_budget=memory_budget,
        solver_preference=solver_preference,
        dataloader_option=dataloader_option,
        shard_option=shard_option,
    )

    # ------------------------------------------------------------------
    # Step 7 (Phase 2 only): Create BoundaryReshardingModules for each
    # stage boundary where adjacent TP degrees differ.
    # ------------------------------------------------------------------
    if plan.heterogeneous_tp:
        _build_boundary_modules(plan, pp_size, tp_per_stage, all_stage_meshes, current_stage)

    return stage_wrapped, stage_manager, plan


def _build_boundary_modules(
    plan: "PipelinePlan",
    pp_size: int,
    tp_per_stage: List[int],
    all_stage_meshes: list,
    current_stage: int,
    rank: Optional[int] = None,
) -> None:
    """Populate plan.send_boundary_modules and plan.recv_boundary_modules.

    For each boundary between stage s and stage s+1 where tp_per_stage[s] != tp_per_stage[s+1]:

      send_boundary_modules[s]:
          BoundaryReshardingModule("before_send") on stage s ranks.
          AllGather when sender_tp > 1 so every rank has the full tensor.

      recv_boundary_modules[s+1]:
          BoundaryReshardingModule("after_recv") on stage s+1 ranks.
          Split when receiver_tp > 1 so each rank takes its TP shard.

    All modules are created on the current rank's stage only.
    Modules for other stages are None (not needed by this rank).
    """
    if rank is None:
        rank = dist.get_rank()

    for s in range(pp_size - 1):
        tp_send = tp_per_stage[s]
        tp_recv = tp_per_stage[s + 1]
        if tp_send == tp_recv:
            continue  # no resharding needed

        # ---- sender-side module (stage s) ----
        if current_stage == s:
            tp_group = all_stage_meshes[s].get_process_group(axis=0)
            plan.send_boundary_modules[s] = BoundaryReshardingModule(
                mode="before_send",
                sender_tp=tp_send,
                receiver_tp=tp_recv,
                tp_group=tp_group,
            )

        # ---- receiver-side module (stage s+1) ----
        if current_stage == s + 1:
            # Determine this rank's TP position within stage s+1.
            # Layout: stage_ranks.reshape(tp_recv, dp_recv) → [tp_rank][dp_rank]
            if plan.variable_stage_sizes:
                # Variable stage sizes: use rank_ranges for stage start offset.
                stage_start = plan.rank_ranges[s + 1][0]
                dp_recv = plan.dp_per_stage[s + 1]
            else:
                # Uniform stage sizes: compute stage start from fixed device budget.
                devices_per_stage = all_stage_meshes[s + 1].num_devices
                stage_start = (s + 1) * devices_per_stage
                dp_recv = devices_per_stage // tp_recv
            local_index = rank - stage_start
            tp_rank = local_index // dp_recv

            plan.recv_boundary_modules[s + 1] = BoundaryReshardingModule(
                mode="after_recv",
                sender_tp=tp_send,
                receiver_tp=tp_recv,
                tp_rank=tp_rank,
            )


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
