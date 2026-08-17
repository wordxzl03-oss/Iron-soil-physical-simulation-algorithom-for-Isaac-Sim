"""Shared earthmoving physics core for interactive and headless 390F V2."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from ..bulk_exchange import (
    AirborneParcelModel,
    BucketRetentionResult,
    DumpAdvanceResult,
    DumpReleaseResult,
    DumpTarget,
    TerrainDumpOperator,
)
from ..bulk_interaction import (
    BulkInteractionResult,
    BulkMaterialInteractionModel,
    BucketIntakeResult,
    FailureZone,
    MobileLayerSolver,
    OptimizedMobileLayerSolver,
    ToolTerrainIntersectionModel,
    TrackSoilModel,
    TrackSoilConfig,
    TrackSoilResult,
    WarpDepositionStep,
    WarpMobileStep,
    WarpTrackSoilStep,
)
from ..bulk_interaction.large_avalanche import (
    LargeAvalancheTransitionConfig,
    LargeAvalancheTransitionController,
    RestingToMobileTransitionResult,
    TerrainSettledDiagnostic,
)
from ..bulk_state import (
    BucketInternalFillModel,
    BulkStateManager,
    ConservativeTransfer,
    MaterialScenario,
    PayloadState,
    TerrainState,
    TerrainVolumeIntegrator,
    Reservoir,
)
from ..config import SolverConfig, SweepConfig
from ..interaction import ContinuousSweepBuilder
from ..performance import PerformanceProfiler
from ..soil_force import MobileMomentumBudget, SoilForceModel, SoilForceResult
from ..solvers import (
    CompactTileFrontier,
    EventDrivenMinimumSlopeAdapter,
    MinimumSlopeAdapter,
    SparseTileFrontierMinimumSlopeAdapter,
)
from ..terrain import TerrainGrid
from ..tools import ToolDescriptor, ToolState
from .interactive_control import ResetLevel, SoilForceMode
from .bulk_state_authority import (
    BulkStateAuthorityError,
    DeviceBulkState,
    DeviceTransferSnapshot,
)
from .gpu_airborne_bridge import DeviceAirborneAdvanceResult
from .gpu_bulk_operator_chain import GpuBulkOperatorChain
from .gpu_failure_bridge import DeviceFailureZoneResult
from .gpu_intake_bridge import DeviceIntakeResult
from .gpu_runtime_metadata import GpuRuntimeMetadata, GpuRuntimeScalarSnapshot
from .gpu_large_avalanche import DeviceLargeAvalancheResult
from .physics_diagnostics import PhysicsDiagnostics


DIG_PHASES = frozenset({"penetrate", "coordinated_cut", "curl_filling"})


@dataclass(frozen=True)
class PhysicsCoreStepResult:
    state: TerrainState | None
    applied_force_terrain_n: np.ndarray
    computed_force_terrain_n: np.ndarray
    quasi_static_force_terrain_n: np.ndarray
    momentum_force_terrain_n: np.ndarray
    interaction: BulkInteractionResult | "GpuBulkInteractionResult" | None
    force_result: SoilForceResult | None
    dump_release: DumpReleaseResult | "GpuDumpReleaseResult" | None
    dump_advance: DumpAdvanceResult | "GpuDumpAdvanceResult" | None
    mass_balance_error_m3: float
    timings_ms: dict[str, float]
    terrain_settled: bool
    static_relaxation_pending: bool
    static_relaxation_iterations: int
    static_relaxation_active_tiles: int
    avalanche_transition: RestingToMobileTransitionResult | DeviceLargeAvalancheResult | None
    terrain_settled_diagnostic: TerrainSettledDiagnostic | "GpuTerrainSettledDiagnostic"
    physics_diagnostics: PhysicsDiagnostics
    runtime_backend: str = "HOST_REFERENCE"
    scalar_state: GpuRuntimeScalarSnapshot | None = None
    device_transfer: DeviceTransferSnapshot | None = None


@dataclass(frozen=True)
class GpuBulkInteractionResult:
    """Production GPU dig result without a reconstructed host TerrainState."""

    failure_zone: FailureZone
    failure_bridge: DeviceFailureZoneResult
    intake_result: BucketIntakeResult
    intake_bridge: DeviceIntakeResult
    mobile_result: WarpMobileStep
    activated_volume_m3: float
    activation_mode: str
    activation_tool_impulse_on_mobile_terrain_ns: np.ndarray
    timings_ms: dict[str, float]


@dataclass(frozen=True)
class GpuDumpReleaseResult:
    target: DumpTarget
    retention: BucketRetentionResult
    released_volume_m3: float
    created_parcel_count: int
    exported_volume_m3: float


@dataclass(frozen=True)
class GpuDumpAdvanceResult:
    airborne: DeviceAirborneAdvanceResult
    mobile: WarpMobileStep
    deposition: WarpDepositionStep


@dataclass(frozen=True)
class GpuTerrainSettledDiagnostic:
    """Scalar-only settled diagnostic for a device-authoritative step."""

    settled: bool
    current_mobile_active: bool
    mobile_volume_m3: float
    maximum_mobile_speed_m_s: float
    airborne_volume_m3: float
    reason: str
    large_avalanche_status: str


class EarthmovingPhysicsCore:
    """The single physics implementation used by both runtime frontends."""

    def __init__(
        self,
        *,
        grid: TerrainGrid,
        descriptor: ToolDescriptor,
        material: MaterialScenario,
        initial_heightmap_m: np.ndarray,
        runtime_backend: str = "HOST_REFERENCE",
        gpu_device: str = "cuda:0",
        solver_backend: str = "OPTIMIZED",
        slope_backend: str | None = None,
        tile_size: int = 64,
        large_avalanche_iteration_threshold: int = 1_000,
        numerical_safety_max_iterations: int = 1_000_000,
        minislope_round_budget_per_step: int = 1,
        minislope_tolerance_m: float = 1.0e-8,
        residual_projection_max_rounds: int = 4096,
        large_avalanche_config: LargeAvalancheTransitionConfig | None = None,
        mass_tolerance_m3: float = 1.0e-8,
        track_soil_config: TrackSoilConfig | None = None,
    ) -> None:
        runtime = str(runtime_backend).upper()
        if runtime not in {"HOST_REFERENCE", "GPU_RUNTIME"}:
            raise ValueError(
                "[V2PhysicsCore] runtime_backend must be HOST_REFERENCE/GPU_RUNTIME"
            )
        backend = str(solver_backend).upper()
        if backend not in {"REFERENCE", "OPTIMIZED"}:
            raise ValueError("[V2PhysicsCore] backend must be REFERENCE/OPTIMIZED")
        self.grid = grid
        self.descriptor = descriptor
        self.material = material
        self.backend = backend
        self.runtime_backend = runtime
        selected_slope_backend = (
            str(slope_backend).upper()
            if slope_backend is not None
            else (
                "FULL_DOMAIN_REFERENCE"
                if backend == "REFERENCE"
                else "CPU_OPTIMIZED_COMPACT_TILE_FRONTIER"
            )
        )
        if selected_slope_backend not in {
            "FULL_DOMAIN_REFERENCE",
            "CPU_OPTIMIZED_REACHABLE_BBOX_BASELINE",
            "CPU_OPTIMIZED_COMPACT_TILE_FRONTIER",
        }:
            raise ValueError(
                f"[V2PhysicsCore] unsupported slope backend={selected_slope_backend}"
            )
        self.slope_backend = selected_slope_backend
        if (
            not isinstance(minislope_round_budget_per_step, int)
            or minislope_round_budget_per_step < 1
        ):
            raise ValueError(
                "[V2PhysicsCore] minislope_round_budget_per_step must be >= 1"
            )
        self.minislope_round_budget_per_step = minislope_round_budget_per_step
        if not isinstance(residual_projection_max_rounds, int) or residual_projection_max_rounds < 1:
            raise ValueError("[V3PhysicsCore] residual_projection_max_rounds must be >= 1")
        self.residual_projection_max_rounds = residual_projection_max_rounds
        if (
            not np.isfinite(minislope_tolerance_m)
            or minislope_tolerance_m <= 0.0
            or minislope_tolerance_m > 0.1 * min(grid.dx, grid.dy)
        ):
            raise ValueError(
                "[V2PhysicsCore] minislope_tolerance_m must be positive and "
                "no larger than 10% of grid spacing"
            )
        self.minislope_tolerance_m = float(minislope_tolerance_m)
        self.integrator = TerrainVolumeIntegrator.from_grid(grid)
        self._initial_height = np.array(
            grid.validate_heightmap(initial_heightmap_m), dtype=np.float64, copy=True
        )
        self._initial_height.setflags(write=False)
        payload = PayloadState(
            0.0,
            descriptor.effective_capacity_m3,
            material.assumed_bulk_density_kg_m3,
            np.zeros(3),
        )
        self._initial_state = TerrainState(
            self._initial_height,
            np.zeros(grid.shape),
            np.zeros(grid.shape + (2,)),
            payload,
            (),
            material,
            0.0,
            0.0,
            0,
        )
        resolved_large_avalanche_config = (
            large_avalanche_config or LargeAvalancheTransitionConfig()
        )
        self.manager: BulkStateManager | None = None
        self.device_state: DeviceBulkState | None = None
        self.gpu_chain: GpuBulkOperatorChain | None = None
        self.gpu_metadata: GpuRuntimeMetadata | None = None
        if runtime == "HOST_REFERENCE":
            self.manager = BulkStateManager(
                self._initial_state,
                self.integrator,
                boundary_condition="closed",
                absolute_tolerance_m3=mass_tolerance_m3,
                relative_tolerance=1.0e-10,
            )
        else:
            self.device_state = DeviceBulkState(
                grid,
                self._initial_height,
                device=gpu_device,
                tile_size=tile_size,
            )
            self.gpu_chain = GpuBulkOperatorChain(
                self.device_state,
                material,
                grid,
                self.integrator,
                resolved_large_avalanche_config,
                descriptor,
            )
            self.gpu_metadata = GpuRuntimeMetadata(self.device_state, payload)
        mobile_solver = (
            MobileLayerSolver()
            if backend == "REFERENCE"
            else OptimizedMobileLayerSolver(tile_size=tile_size)
        )
        self.mobile_solver = mobile_solver
        self.interaction_model = BulkMaterialInteractionModel(mobile_layer=mobile_solver)
        self.dump_operator = TerrainDumpOperator(
            mobile=mobile_solver,
            airborne=AirborneParcelModel(descriptor=descriptor, material=material),
        )
        self.intersection_model = ToolTerrainIntersectionModel()
        self.track_soil_model = TrackSoilModel(track_soil_config, tile_size=tile_size)
        self.avalanche_controller = LargeAvalancheTransitionController(
            resolved_large_avalanche_config
        )
        if runtime == "HOST_REFERENCE":
            self.track_soil_model.initialize(self._initial_height)
        self.sweep_builder = ContinuousSweepBuilder(SweepConfig())
        self.soil_force_model = SoilForceModel()
        slope_config = SolverConfig(
            critical_angle_deg=material.stop_angle_deg,
            max_iterations=1000,
            large_avalanche_iteration_threshold=(
                large_avalanche_iteration_threshold
            ),
            numerical_safety_max_iterations=numerical_safety_max_iterations,
            tolerance=self.minislope_tolerance_m,
            boundary_condition="closed",
            conservation_tolerance_m3=mass_tolerance_m3,
        )
        if selected_slope_backend == "FULL_DOMAIN_REFERENCE":
            self.slope_solver = MinimumSlopeAdapter(grid, slope_config)
        elif selected_slope_backend == "CPU_OPTIMIZED_REACHABLE_BBOX_BASELINE":
            self.slope_solver = EventDrivenMinimumSlopeAdapter(
                grid, slope_config, tile_size=tile_size
            )
            self.slope_solver.set_reference_height(self._initial_height)
        else:
            self.slope_solver = SparseTileFrontierMinimumSlopeAdapter(
                grid, slope_config, tile_size=tile_size
            )
            self.slope_solver.set_reference_height(self._initial_height)
        self.profiler = PerformanceProfiler()
        self.previous_tool_state: ToolState | None = None
        self._dump_released_phase_token: tuple[int, str] | None = None
        self._dump_deposition_seed_mask = np.zeros(self.grid.shape, dtype=bool)
        self._host_mobile_export_cumulative_m3 = np.zeros(
            self.grid.shape, dtype=np.float64
        )
        self._terrain_settled = True
        self._static_relaxation_pending = False
        self._static_relaxation_iterations = 0
        self._static_relaxation_active_tiles = 0
        self._residual_solver_iterations_total = 0
        self._physical_simulation_time_s = 0.0
        self._dynamic_flow_time_s = 0.0
        self._last_physics_diagnostics: PhysicsDiagnostics | None = None
        self._last_avalanche_transition: RestingToMobileTransitionResult | None = None
        self._last_gpu_scalar_snapshot: GpuRuntimeScalarSnapshot | None = None
        self._last_gpu_avalanche_transition: DeviceLargeAvalancheResult | None = None
        self._gpu_frontier_controller = (
            CompactTileFrontier(grid.shape, tile_size)
            if runtime == "GPU_RUNTIME"
            else None
        )
        self._gpu_frontier_active_tiles = np.empty(0, dtype=np.int32)
        self._gpu_frontier_iterations = 0
        if runtime == "HOST_REFERENCE":
            self._last_settled_diagnostic = self.avalanche_controller.settled_diagnostic(
                self._initial_state.H_resting_m,
                self._initial_state.mobile_height_m,
                self._initial_state.mobile_momentum_m2_s,
                material,
                grid,
                self.integrator,
            )
        else:
            self._last_settled_diagnostic = GpuTerrainSettledDiagnostic(
                settled=True,
                current_mobile_active=False,
                mobile_volume_m3=0.0,
                maximum_mobile_speed_m_s=0.0,
                airborne_volume_m3=0.0,
                reason="INITIAL_DEVICE_STATE_QUIET",
                large_avalanche_status="DEVICE_PREPROCESSOR_READY",
            )
        self._reset_generation = 0

    @property
    def state(self) -> TerrainState:
        if self.manager is None:
            raise BulkStateAuthorityError(
                "GPU_RUNTIME_HAS_NO_HOST_TERRAINSTATE: use device_state, "
                "scalar_state(), or compact publication/query APIs"
            )
        return self.manager.current_state

    def scalar_state(self) -> GpuRuntimeScalarSnapshot | None:
        """Return scalar GPU reservoirs without reconstructing terrain fields."""

        if self.runtime_backend == "HOST_REFERENCE":
            return None
        assert self.device_state is not None and self.gpu_metadata is not None
        return self.gpu_metadata.snapshot(self.device_state)

    @property
    def payload(self) -> PayloadState:
        if self.runtime_backend == "GPU_RUNTIME":
            assert self.gpu_metadata is not None
            return self.gpu_metadata.payload
        return self.state.payload

    @property
    def airborne_parcels(self):
        if self.runtime_backend == "GPU_RUNTIME":
            assert self.gpu_metadata is not None
            return self.gpu_metadata.airborne
        return self.state.airborne_parcels

    def reservoir_observation(self) -> dict[str, float | np.ndarray]:
        """Backend-neutral material observation using reductions on device."""

        if self.runtime_backend == "GPU_RUNTIME":
            # A production GPU step already creates the authoritative scalar
            # reservoir snapshot.  Reuse it here so launcher/HUD observation
            # does not repeat the full-field device reductions after every
            # step (especially while the event-driven core is idle).
            snapshot = self._last_gpu_scalar_snapshot
            if snapshot is None:
                snapshot = self.scalar_state()
            assert snapshot is not None
            ledger = snapshot.material_ledger
            return {
                "resting_volume_m3": ledger.resting_m3,
                "mobile_volume_m3": ledger.mobile_m3,
                "payload_volume_m3": ledger.payload_m3,
                "airborne_volume_m3": ledger.airborne_m3,
                "outflow_volume_m3": ledger.outflow_m3,
                "mass_balance_error_m3": ledger.absolute_volume_error_m3,
            }
        state = self.state
        ledger = self.manager.ledger_snapshot()  # type: ignore[union-attr]
        return {
            "resting_volume_m3": self.integrator.integrate(state.H_resting_m),
            "mobile_volume_m3": self.integrator.integrate(state.mobile_height_m),
            "payload_volume_m3": state.payload.volume_m3,
            "airborne_volume_m3": state.airborne_volume_m3,
            "outflow_volume_m3": state.outflow_volume_m3,
            "mass_balance_error_m3": ledger.balance.absolute_volume_error_m3,
        }

    def surface_at_grid_coordinates(self, row_col: np.ndarray) -> np.ndarray:
        """Query current Resting+Mobile surface without exposing its authority."""

        queries = np.asarray(row_col, dtype=np.float64).reshape(-1, 2)
        if self.runtime_backend == "GPU_RUNTIME":
            assert self.device_state is not None
            return self.device_state.sample_surface_bilinear(
                queries, source="launcher_observation"
            )
        surface = self.state.H_resting_m + self.state.mobile_height_m
        rows = np.clip(np.rint(queries[:, 0]).astype(int), 0, self.grid.ny - 1)
        cols = np.clip(np.rint(queries[:, 1]).astype(int), 0, self.grid.nx - 1)
        return np.asarray(surface[rows, cols], dtype=np.float64)

    def surface_at_terrain_points(self, points_terrain_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_terrain_m, dtype=np.float64).reshape(-1, 3)
        queries = np.asarray(
            [self.grid.terrain_to_grid(point) for point in points],
            dtype=np.float64,
        )
        return self.surface_at_grid_coordinates(queries)

    def consume_dirty_surface_tiles(self) -> dict[int, np.ndarray]:
        """Publish only dirty device tiles; host frontend keeps no full field."""

        if self.runtime_backend != "GPU_RUNTIME":
            raise BulkStateAuthorityError(
                "[V2PhysicsCore] dirty device tiles require GPU_RUNTIME"
            )
        assert self.device_state is not None
        return self.device_state.download_dirty_tiles()

    @property
    def reset_generation(self) -> int:
        return self._reset_generation

    @property
    def terrain_settled(self) -> bool:
        """Whether airborne, Mobile and residual static work are all quiet."""

        return self._terrain_settled

    @property
    def physics_diagnostics(self) -> PhysicsDiagnostics | None:
        """Latest observational V3 monitor sample; never used for control."""

        return self._last_physics_diagnostics

    def initialize_tool(self, tool_state: ToolState) -> None:
        self.previous_tool_state = tool_state

    def step(
        self,
        tool_state: ToolState,
        *,
        phase: str,
        cycle: int,
        dt_s: float,
        soil_force_mode: SoilForceMode,
        phase_ending: bool = False,
    ) -> PhysicsCoreStepResult:
        """Advance the selected production authority through one shared Core."""

        if self.runtime_backend == "GPU_RUNTIME":
            return self._step_gpu(
                tool_state,
                phase=phase,
                cycle=cycle,
                dt_s=dt_s,
                soil_force_mode=soil_force_mode,
                phase_ending=phase_ending,
            )
        return self._step_host(
            tool_state,
            phase=phase,
            cycle=cycle,
            dt_s=dt_s,
            soil_force_mode=soil_force_mode,
            phase_ending=phase_ending,
        )

    def _step_host(
        self,
        tool_state: ToolState,
        *,
        phase: str,
        cycle: int,
        dt_s: float,
        soil_force_mode: SoilForceMode,
        phase_ending: bool = False,
    ) -> PhysicsCoreStepResult:
        assert self.manager is not None
        step_start = perf_counter()
        interaction = None
        force_result = None
        release = None
        dump_advance = None
        applied = np.zeros(3)
        computed = np.zeros(3)
        quasi = np.zeros(3)
        momentum = np.zeros(3)
        timings: dict[str, float] = {}
        avalanche_transition = None
        if self.previous_tool_state is None:
            self.previous_tool_state = tool_state

        if phase in DIG_PHASES:
            before = self.manager.current_state
            coupled_tool = BucketInternalFillModel.apply_secondary_separation(
                tool_state, before.payload, self.descriptor
            )
            with self.profiler.measure("bucket_terrain_intersection"):
                sweep = self.sweep_builder.build(
                    self.previous_tool_state, tool_state, self.grid, self.descriptor
                )
                intersection = self.intersection_model.compute(
                    before.H_resting_m + before.mobile_height_m,
                    sweep,
                    coupled_tool,
                    self.grid,
                    self.integrator,
                )
            with self.profiler.measure("soil_solver"):
                interaction = self.interaction_model.advance(
                    self.manager,
                    intersection,
                    coupled_tool,
                    self.descriptor,
                    self.grid,
                    self.integrator,
                    dt_s,
                    action_index=max(0, (cycle - 1) * 12),
                )
            timings.update(interaction.timings_ms)
            with self.profiler.measure("soil_force"):
                force_result = self.soil_force_model.compute(
                    interaction.failure_zone,
                    intersection,
                    before.material,
                    self.descriptor,
                    coupled_tool,
                    MobileMomentumBudget.from_mobile_result(
                        interaction.mobile_result,
                        interaction.activation_tool_impulse_on_mobile_terrain_ns,
                        dt_s,
                    ),
                )
            computed = np.asarray(force_result.force_terrain_n)
            quasi = np.asarray(force_result.quasi_static_force_terrain_n)
            momentum = np.asarray(force_result.active_momentum_force_terrain_n)
            mode = SoilForceMode(soil_force_mode)
            if mode is SoilForceMode.QUASI_STATIC_ONLY:
                applied = quasi
            elif mode is SoilForceMode.FULL_SOIL_FORCE:
                applied = computed
        elif phase == "dump_spill":
            token = (int(cycle), phase)
            if self._dump_released_phase_token != token:
                # A MiniSlope event is seeded by changes caused by this dump,
                # not by every resting/mobile/track update since simulator
                # initialization.  Capture the authoritative pre-release
                # surface before any parcel can land in this step.  Dynamic
                # frontier propagation remains unrestricted once seeded.
                if isinstance(self.slope_solver, EventDrivenMinimumSlopeAdapter):
                    self.slope_solver.set_reference_height(
                        self.manager.current_state.H_resting_m
                    )
                self._dump_deposition_seed_mask.fill(False)
                with self.profiler.measure("retention_spill"):
                    release = self.dump_operator.release(
                        self.manager, tool_state, self.descriptor, target=DumpTarget.TERRAIN
                    )
                # A zero-volume retention evaluation is not a completed dump.
                # The bucket can enter the release gate while its orientation
                # is still changing, so latching that first dry evaluation
                # permanently strands a real payload.  Retry until material is
                # actually released (or the payload is already empty).
                if (
                    release.released_volume_m3 > 1.0e-12
                    or release.state.payload.volume_m3 <= 1.0e-12
                ):
                    self._dump_released_phase_token = token

        if phase not in DIG_PHASES:
            # Deposition is integrated every physics step.  ``phase_ending``
            # is intentionally not a trigger for a convergence-to-equilibrium
            # call: task progression may wait for ``terrain_settled``, while
            # Isaac continues to receive one bounded terrain update per step.
            with self.profiler.measure("airborne_deposition"):
                dump_advance = self.dump_operator.advance_airborne(
                    self.manager,
                    self.grid,
                    self.integrator,
                    dt_s,
                )
            if (
                dump_advance is not None
                and dump_advance.airborne.landed_volume_m3 > 0.0
            ):
                # The global Mobile reservoir also contains excavation and
                # track material.  Deposition.eligible_mask therefore cannot
                # identify dump provenance.  Ballistic landing points do:
                # each parcel is inserted at the rounded authoritative grid
                # vertex, which becomes an instability seed (not a final ROI).
                for point_world in dump_advance.airborne.landing_points_world_m:
                    point_terrain = self.grid.world_to_terrain(point_world)
                    row_f, column_f = self.grid.terrain_to_grid(point_terrain)
                    row = int(np.clip(round(row_f), 0, self.grid.ny - 1))
                    column = int(np.clip(round(column_f), 0, self.grid.nx - 1))
                    self._dump_deposition_seed_mask[row, column] = True
            if (
                dump_advance is not None
                and dump_advance.deposition is not None
                and dump_advance.deposition.deposited_volume_m3 > 0.0
                and np.any(self._dump_deposition_seed_mask)
            ):
                # Landing points retain dump provenance.  The exact cells that
                # became Resting are also seeds so incremental deposition can
                # trigger local residual stabilization without waiting for the
                # complete payload to land.
                self._dump_deposition_seed_mask |= (
                    dump_advance.deposition.deposited_height_m
                    > max(self.slope_solver.config.tolerance, 1.0e-12)
                )
        last_export = getattr(
            self.mobile_solver, "last_conservative_export_m3", None
        )
        if last_export is not None:
            exported = np.asarray(last_export, dtype=np.float64)
            if exported.shape == self.grid.shape:
                self._host_mobile_export_cumulative_m3 += exported
        with self.profiler.measure("large_avalanche_transition"):
            avalanche_transition = self._advance_large_avalanche_transition(dt_s)
        timings.update(
            {
                f"large_avalanche_{name}": float(value)
                for name, value in self.avalanche_controller.last_profile_ms.items()
            }
        )
        self._last_avalanche_transition = avalanche_transition
        self._advance_static_relaxation_budget()
        self.previous_tool_state = tool_state
        ledger = self.manager.ledger_snapshot()
        timings["physics_core_total"] = (perf_counter() - step_start) * 1_000.0
        self._physical_simulation_time_s += float(dt_s)
        observed_mobile = (
            interaction.mobile_result if interaction is not None
            else dump_advance.mobile if dump_advance is not None else None
        )
        moving_volume = 0.0 if observed_mobile is None else observed_mobile.moving_mobile_volume_m3
        if moving_volume > 0.0:
            self._dynamic_flow_time_s += float(dt_s)
        diagnostics = self._make_host_physics_diagnostics(
            interaction, observed_mobile, force_result,
            ledger.balance.absolute_volume_error_m3, moving_volume,
            dt_s, timings["physics_core_total"],
        )
        self._last_physics_diagnostics = diagnostics
        return PhysicsCoreStepResult(
            state=self.manager.current_state,
            applied_force_terrain_n=applied,
            computed_force_terrain_n=computed,
            quasi_static_force_terrain_n=quasi,
            momentum_force_terrain_n=momentum,
            interaction=interaction,
            force_result=force_result,
            dump_release=release,
            dump_advance=dump_advance,
            mass_balance_error_m3=ledger.balance.absolute_volume_error_m3,
            timings_ms=timings,
            terrain_settled=self._terrain_settled,
            static_relaxation_pending=self._static_relaxation_pending,
            static_relaxation_iterations=self._static_relaxation_iterations,
            static_relaxation_active_tiles=self._static_relaxation_active_tiles,
            avalanche_transition=avalanche_transition,
            terrain_settled_diagnostic=self._last_settled_diagnostic,
            physics_diagnostics=diagnostics,
        )

    def _step_gpu(
        self,
        tool_state: ToolState,
        *,
        phase: str,
        cycle: int,
        dt_s: float,
        soil_force_mode: SoilForceMode,
        phase_ending: bool = False,
    ) -> PhysicsCoreStepResult:
        """Production device-authoritative path; never constructs TerrainState."""

        del phase_ending
        state = self.device_state
        chain = self.gpu_chain
        metadata = self.gpu_metadata
        if state is None or chain is None or metadata is None:
            raise RuntimeError("[V2PhysicsCore] GPU runtime was not initialized")
        state.begin_physics_step()
        step_start = perf_counter()
        timings: dict[str, float] = {}
        interaction: GpuBulkInteractionResult | None = None
        force_result: SoilForceResult | None = None
        release: GpuDumpReleaseResult | None = None
        dump_advance: GpuDumpAdvanceResult | None = None
        avalanche_transition: DeviceLargeAvalancheResult | None = None
        applied = np.zeros(3, dtype=np.float64)
        computed = np.zeros(3, dtype=np.float64)
        quasi = np.zeros(3, dtype=np.float64)
        momentum = np.zeros(3, dtype=np.float64)
        if self.previous_tool_state is None:
            self.previous_tool_state = tool_state

        cached_snapshot = self._last_gpu_scalar_snapshot
        cached_avalanche = self._last_gpu_avalanche_transition
        if (
            phase not in DIG_PHASES
            and phase != "dump_spill"
            and self._terrain_settled
            and not metadata.airborne
            and self._gpu_frontier_active_tiles.size == 0
            and cached_snapshot is not None
            and cached_avalanche is not None
            and self._last_physics_diagnostics is not None
            and cached_snapshot.material_ledger.mobile_m3
            <= self.avalanche_controller.config.settled_mobile_volume_m3
        ):
            state.advance_time(dt_s)
            metadata.advance_time(dt_s)
            self.previous_tool_state = tool_state
            self._physical_simulation_time_s += float(dt_s)
            snapshot = replace(cached_snapshot, timestamp_s=metadata.timestamp_s)
            self._last_gpu_scalar_snapshot = snapshot
            wall_ms = (perf_counter() - step_start) * 1_000.0
            timings["event_driven_idle"] = wall_ms
            timings["physics_core_total"] = wall_ms
            diagnostics = replace(
                self._last_physics_diagnostics,
                soil_force_terrain_n=np.zeros(3, dtype=np.float64),
                soil_torque_terrain_nm=np.zeros(3, dtype=np.float64),
                terrain_state="SETTLED",
                not_settled_reason="",
                physical_simulation_time_s=self._physical_simulation_time_s,
                dynamic_flow_time_s=self._dynamic_flow_time_s,
                rtf=float(dt_s / max(wall_ms * 1.0e-3, 1.0e-12)),
            )
            self._last_physics_diagnostics = diagnostics
            state.assert_normal_step_transfer_budget()
            return PhysicsCoreStepResult(
                state=None,
                applied_force_terrain_n=np.zeros(3, dtype=np.float64),
                computed_force_terrain_n=np.zeros(3, dtype=np.float64),
                quasi_static_force_terrain_n=np.zeros(3, dtype=np.float64),
                momentum_force_terrain_n=np.zeros(3, dtype=np.float64),
                interaction=None,
                force_result=None,
                dump_release=None,
                dump_advance=None,
                mass_balance_error_m3=snapshot.material_ledger.absolute_volume_error_m3,
                timings_ms=timings,
                terrain_settled=True,
                static_relaxation_pending=False,
                static_relaxation_iterations=self._gpu_frontier_iterations,
                static_relaxation_active_tiles=0,
                avalanche_transition=cached_avalanche,
                terrain_settled_diagnostic=self._last_settled_diagnostic,
                runtime_backend="GPU_RUNTIME",
                scalar_state=snapshot,
                device_transfer=state.transfer_snapshot(),
                physics_diagnostics=diagnostics,
            )

        if phase in DIG_PHASES:
            coupled_tool = BucketInternalFillModel.apply_secondary_separation(
                tool_state, metadata.payload, self.descriptor
            )
            start = perf_counter()
            sweep = self.sweep_builder.build(
                self.previous_tool_state, tool_state, self.grid, self.descriptor
            )
            failure = chain.apply_failure_zone(sweep, coupled_tool)
            timings.update(failure.timings_ms)
            timings["bucket_terrain_intersection"] = (
                perf_counter() - start
            ) * 1_000.0
            try:
                start = perf_counter()
                mobile = chain.step_mobile(dt_s)
                timings["gpu_mobile_layer"] = (perf_counter() - start) * 1_000.0
            finally:
                chain.clear_failure_tool_forcing(failure)
            start = perf_counter()
            intake = chain.apply_bucket_intake(
                metadata.payload,
                coupled_tool,
                self.descriptor,
                dt_s,
            )
            metadata.commit_mobile_to_payload(
                metadata.payload,
                intake.intake.payload,
                intake.transaction.volume_m3,
            )
            timings.update(intake.timings_ms)
            timings["gpu_bucket_intake_total"] = (
                perf_counter() - start
            ) * 1_000.0
            budget = MobileMomentumBudget(
                momentum_before_terrain_kg_m_s=(
                    mobile.momentum_before_terrain_kg_m_s
                    - failure.activation_tool_impulse_on_mobile_terrain_ns
                ),
                momentum_after_terrain_kg_m_s=(
                    mobile.momentum_after_terrain_kg_m_s
                ),
                gravity_pressure_impulse_terrain_ns=(
                    mobile.gravity_pressure_impulse_terrain_ns
                ),
                basal_friction_impulse_terrain_ns=(
                    mobile.basal_friction_impulse_terrain_ns
                ),
                numerical_dissipative_impulse_terrain_ns=(
                    mobile.numerical_dissipative_impulse_terrain_ns
                ),
                tool_impulse_on_mobile_terrain_ns=(
                    mobile.tool_impulse_on_mobile_terrain_ns
                    + failure.activation_tool_impulse_on_mobile_terrain_ns
                ),
                integration_window_s=dt_s,
            )
            start = perf_counter()
            force_result = self.soil_force_model.compute(
                failure.failure_zone,
                failure.intersection,
                self.material,
                self.descriptor,
                coupled_tool,
                budget,
            )
            timings["soil_force"] = (perf_counter() - start) * 1_000.0
            computed = np.asarray(force_result.force_terrain_n)
            quasi = np.asarray(force_result.quasi_static_force_terrain_n)
            momentum = np.asarray(force_result.active_momentum_force_terrain_n)
            mode = SoilForceMode(soil_force_mode)
            if mode is SoilForceMode.QUASI_STATIC_ONLY:
                applied = quasi
            elif mode is SoilForceMode.FULL_SOIL_FORCE:
                applied = computed
            interaction = GpuBulkInteractionResult(
                failure_zone=failure.failure_zone,
                failure_bridge=failure,
                intake_result=intake.intake,
                intake_bridge=intake,
                mobile_result=mobile,
                activated_volume_m3=failure.activated_volume_m3,
                activation_mode=failure.activation_mode,
                activation_tool_impulse_on_mobile_terrain_ns=(
                    failure.activation_tool_impulse_on_mobile_terrain_ns
                ),
                timings_ms=dict(timings),
            )
            metadata.advance_time(dt_s)
        else:
            if phase == "dump_spill":
                token = (int(cycle), phase)
                if self._dump_released_phase_token != token:
                    release = self._gpu_release_payload(tool_state, cycle=cycle)
                    if (
                        release.released_volume_m3 > 1.0e-12
                        or metadata.payload.volume_m3 <= 1.0e-12
                    ):
                        self._dump_released_phase_token = token
            start = perf_counter()
            airborne = chain.advance_airborne(metadata.airborne, dt_s)
            timings["airborne"] = (perf_counter() - start) * 1_000.0
            timings.update(airborne.timings_ms)
            start = perf_counter()
            mobile = chain.step_mobile(dt_s)
            timings["mobile_transport"] = (perf_counter() - start) * 1_000.0
            start = perf_counter()
            deposition = chain.step_deposition(
                dt_s,
                settle_subcell_tail=(
                    mobile.volume_after_m3
                    <= self.grid.dx * self.grid.dy * min(self.grid.dx, self.grid.dy)
                ),
            )
            timings["deposition"] = (perf_counter() - start) * 1_000.0
            # Landing points are provenance, not proof of instability.  The
            # cohesive device yield pass below supplies residual frontier seeds
            # only when the deposited free surface is physically unstable.
            start = perf_counter()
            metadata.commit_airborne_advance(
                airborne.remaining_parcels,
                airborne.landed_volume_m3,
                deposition.deposited_volume_m3,
                dt_s,
            )
            timings["payload_airborne_ledger"] = (
                perf_counter() - start
            ) * 1_000.0
            dump_advance = GpuDumpAdvanceResult(
                airborne=airborne,
                mobile=mobile,
                deposition=deposition,
            )
            timings["gpu_airborne_mobile_deposition"] = (
                timings["airborne"]
                + timings["mobile_transport"]
                + timings["deposition"]
                + timings["payload_airborne_ledger"]
            )

        # Physical yield ownership is resolved before any numerical residual
        # projection.  The previous order let a pending frontier flatten cells
        # before the current cohesive Y_start mask had a chance to re-enter the
        # Mobile path.
        start = perf_counter()
        pre_avalanche_snapshot = metadata.snapshot(state)
        timings["mass_ledger_pre"] = (perf_counter() - start) * 1_000.0
        maximum_speed = (
            interaction.mobile_result.maximum_speed_m_s
            if interaction is not None
            else dump_advance.mobile.maximum_speed_m_s
            if dump_advance is not None
            else 0.0
        )
        dynamic_quiet_for_residual = bool(
            pre_avalanche_snapshot.airborne_volume_m3 <= 1.0e-10
            and pre_avalanche_snapshot.material_ledger.mobile_m3
            <= self.avalanche_controller.config.settled_mobile_volume_m3
            and maximum_speed <= self.avalanche_controller.config.settled_speed_m_s
        )
        start = perf_counter()
        avalanche_transition = chain.advance_large_avalanche(
            dt_s, release_settled_latches=True
        )
        timings["large_avalanche"] = (perf_counter() - start) * 1_000.0
        if avalanche_transition.residual_seed_tile_ids.size:
            # These are compact candidate tiles only.  They are never allowed
            # to own the state while the current cohesive Y_start detector is
            # non-empty.
            self._gpu_frontier_active_tiles = np.union1d(
                self._gpu_frontier_active_tiles,
                avalanche_transition.residual_seed_tile_ids,
            ).astype(np.int32, copy=False)
        # Y_start describes stress on Resting material; it does not by itself
        # prove that a *new* tranche is eligible.  A tranche which mobilized,
        # returned locally and left H_free unchanged remains latched to prevent
        # a zero-transport R->M->R loop.  Only positive currently mobilizable
        # volume may own the physical-flow path and exclude residual/final
        # closure.
        physical_yield_present = bool(
            avalanche_transition.unstable_cell_count > 0
            and avalanche_transition.largest_connected_mobilizable_volume_m3
            > self.avalanche_controller.config.dry_tolerance_m
            * self.grid.dx
            * self.grid.dy
        )
        residual_advanced_this_step = False
        residual_start = perf_counter()
        if (
            dynamic_quiet_for_residual
            and self._gpu_frontier_active_tiles.size
            and not physical_yield_present
        ):
            if self.material.cohesion_proxy_pa > 0.0:
                # In a cohesive material, an old fixed-angle frontier is not
                # itself a physical instability.  Once the V3 cohesive
                # Y_start mask is empty, discard candidate seeds without
                # flattening a cohesive-stable steep face.
                self._gpu_frontier_active_tiles = np.empty(0, dtype=np.int32)
                state.runtime.arrays["frontier_reached"].zero_()
            else:
                self._advance_gpu_residual_frontier()
                residual_advanced_this_step = True
        timings["residual_projection"] = (
            perf_counter() - residual_start
        ) * 1_000.0
        timings["gpu_large_avalanche"] = (
            timings["large_avalanche"] + timings["residual_projection"]
        )

        start = perf_counter()
        snapshot = metadata.snapshot(state)
        timings["mass_ledger_post"] = (perf_counter() - start) * 1_000.0
        active = bool(
            snapshot.material_ledger.mobile_m3
            >= self.avalanche_controller.config.mobile_activity_volume_m3
            and maximum_speed
            >= self.avalanche_controller.config.mobile_activity_speed_m_s
        )
        settled = bool(
            snapshot.airborne_volume_m3 <= 1.0e-10
            and snapshot.material_ledger.mobile_m3
            <= self.avalanche_controller.config.settled_mobile_volume_m3
            and maximum_speed <= self.avalanche_controller.config.settled_speed_m_s
        )
        if physical_yield_present or avalanche_transition.classification in {
            chain.large_avalanche.PERSISTING_LARGE_UNSTABLE_REGION,
            chain.large_avalanche.LARGE_AVALANCHE_MOBILE_PATH,
        }:
            settled = False
        if self._gpu_frontier_active_tiles.size:
            settled = False
        self._terrain_settled = settled
        self._static_relaxation_pending = bool(
            self._gpu_frontier_active_tiles.size
        )
        if settled:
            settled_reason = "DEVICE_DYNAMIC_RESERVOIRS_QUIET"
        elif snapshot.airborne_volume_m3 > 1.0e-10:
            settled_reason = "DEVICE_AIRBORNE_ACTIVITY"
        elif active:
            settled_reason = "DEVICE_DYNAMIC_MOBILE_FLOW"
        elif physical_yield_present:
            settled_reason = "DEVICE_PHYSICAL_YIELD_ACTIVITY"
        elif self._gpu_frontier_active_tiles.size:
            settled_reason = (
                "DEVICE_RESIDUAL_PROJECTION_ACTIVE"
                if residual_advanced_this_step
                else "DEVICE_RESIDUAL_PROJECTION_PENDING"
            )
        else:
            settled_reason = "DEVICE_MOBILE_ARREST_DEPOSITION"
        self._last_settled_diagnostic = GpuTerrainSettledDiagnostic(
            settled=settled,
            current_mobile_active=active,
            mobile_volume_m3=snapshot.material_ledger.mobile_m3,
            maximum_mobile_speed_m_s=float(maximum_speed),
            airborne_volume_m3=snapshot.airborne_volume_m3,
            reason=settled_reason,
            large_avalanche_status=avalanche_transition.classification,
        )
        self.previous_tool_state = tool_state
        timings["physics_core_total"] = (perf_counter() - step_start) * 1_000.0
        state.assert_normal_step_transfer_budget()
        transfer = state.transfer_snapshot()
        self._physical_simulation_time_s += float(dt_s)
        moving_volume = (
            snapshot.material_ledger.mobile_m3 if maximum_speed > 0.01 else 0.0
        )
        if moving_volume > 0.0:
            self._dynamic_flow_time_s += float(dt_s)
        diagnostics = self._make_gpu_physics_diagnostics(
            interaction, force_result, snapshot, moving_volume, maximum_speed,
            avalanche_transition, dt_s, timings["physics_core_total"],
        )
        self._last_physics_diagnostics = diagnostics
        self._last_gpu_scalar_snapshot = snapshot
        self._last_gpu_avalanche_transition = avalanche_transition
        return PhysicsCoreStepResult(
            state=None,
            applied_force_terrain_n=applied,
            computed_force_terrain_n=computed,
            quasi_static_force_terrain_n=quasi,
            momentum_force_terrain_n=momentum,
            interaction=interaction,
            force_result=force_result,
            dump_release=release,
            dump_advance=dump_advance,
            mass_balance_error_m3=(
                snapshot.material_ledger.absolute_volume_error_m3
            ),
            timings_ms=timings,
            terrain_settled=settled,
            static_relaxation_pending=self._static_relaxation_pending,
            static_relaxation_iterations=self._gpu_frontier_iterations,
            static_relaxation_active_tiles=int(
                self._gpu_frontier_active_tiles.size
            ),
            avalanche_transition=avalanche_transition,
            terrain_settled_diagnostic=self._last_settled_diagnostic,
            runtime_backend="GPU_RUNTIME",
            scalar_state=snapshot,
            device_transfer=transfer,
            physics_diagnostics=diagnostics,
        )

    def _gpu_release_payload(
        self, tool_state: ToolState, *, cycle: int
    ) -> GpuDumpReleaseResult:
        """Payload→Airborne transaction using existing retention/parcel physics."""

        metadata = self.gpu_metadata
        if metadata is None:
            raise RuntimeError("[V2PhysicsCore] GPU metadata is unavailable")
        payload = metadata.payload
        retention = self.dump_operator.retention_model.evaluate(
            payload,
            self.descriptor,
            tool_state,
            material_stop_angle_deg=self.material.stop_angle_deg,
        )
        released = retention.spill_volume_m3
        parcels = self.dump_operator.airborne_model.create_from_bucket_release(
            released,
            self.material,
            tool_state,
            self.descriptor,
            timestamp_s=(
                0.0 if self.device_state is None else self.device_state.timestamp_device_s
            ),
            source="bucket_spill_or_dump",
            id_prefix=(
                f"a{max(0, (cycle - 1) * 12):04d}_"
                f"t{int(round((0.0 if self.device_state is None else self.device_state.timestamp_device_s) * 1000)):010d}"
            ),
            free_surface_normal_bucket_frame=(
                retention.free_surface_normal_bucket_frame
            ),
            payload_center_of_mass_bucket_frame_m=(
                payload.center_of_mass_bucket_frame_m
            ),
        )
        metadata.commit_payload_to_airborne(
            retention.payload, parcels, released
        )
        return GpuDumpReleaseResult(
            target=DumpTarget.TERRAIN,
            retention=retention,
            released_volume_m3=float(released),
            created_parcel_count=len(parcels),
            exported_volume_m3=0.0,
        )

    def _seed_gpu_residual_frontier(self, flat_indices: np.ndarray) -> None:
        state = self.device_state
        frontier = self._gpu_frontier_controller
        indices = np.unique(np.asarray(flat_indices, dtype=np.int32))
        if state is None or frontier is None or indices.size == 0:
            return
        state.apply_host_indices(
            "frontier_reached",
            indices,
            1,
            reason="airborne_landing_residual_minislope_seed",
        )
        rows, cols = np.divmod(indices, self.grid.nx)
        triggered = frontier.tiles_around_cells(rows, cols)
        self._gpu_frontier_active_tiles = np.union1d(
            self._gpu_frontier_active_tiles, triggered
        ).astype(np.int32, copy=False)

    def _advance_gpu_residual_frontier(self) -> None:
        """Finish a bounded numerical projection without advancing physics time."""

        chain = self.gpu_chain
        frontier = self._gpu_frontier_controller
        if chain is None or frontier is None:
            return
        from slope_model import _neighbor_pairs

        critical = float(np.tan(np.deg2rad(self.material.stop_angle_deg)))
        neighbors = _neighbor_pairs(self.grid.dx, self.grid.dy)
        tolerance = float(self.slope_solver.config.tolerance)
        rounds_this_projection = 0
        for _ in range(self.residual_projection_max_rounds):
            if self._gpu_frontier_active_tiles.size == 0:
                return
            self._gpu_frontier_iterations += 1
            self._residual_solver_iterations_total += 1
            rounds_this_projection += 1
            if (
                self._gpu_frontier_iterations
                > self.slope_solver.config.numerical_safety_max_iterations
            ):
                raise RuntimeError(
                    "NUMERICAL_NONCONVERGENCE: GPU residual MiniSlope exceeded "
                    "the numerical safety limit"
                )
            phase_tiles = self._gpu_frontier_active_tiles
            triggered_tiles: list[np.ndarray] = []
            for direction_index, (di, dj, distance) in enumerate(neighbors):
                for phase in (0, 1):
                    batch = frontier.edge_batch(
                        phase_tiles,
                        direction_index=direction_index,
                        di=di,
                        dj=dj,
                        phase=phase,
                    )
                    result = chain.transfer_frontier_phase(
                        batch,
                        critical_difference_m=critical * distance,
                    )
                    if result.triggered_tile_ids.size:
                        triggered_tiles.append(result.triggered_tile_ids)
                        phase_tiles = np.union1d(
                            phase_tiles, triggered_tiles[-1]
                        ).astype(np.int32, copy=False)
            # Match the accepted CPU compact-frontier semantics exactly: all
            # colored transfers complete first, then final height is scanned
            # in every direction. Intermediate-phase owners are not carried
            # into the next round after a later phase stabilizes them.
            unstable_owners: list[np.ndarray] = []
            for direction_index, (di, dj, distance) in enumerate(neighbors):
                for phase in (0, 1):
                    batch = frontier.edge_batch(
                        phase_tiles,
                        direction_index=direction_index,
                        di=di,
                        dj=dj,
                        phase=phase,
                    )
                    result = chain.scan_frontier_phase(
                        batch,
                        critical_difference_m=critical * distance,
                        tolerance_m=tolerance,
                    )
                    if result.unstable_owner_tile_ids.size:
                        unstable_owners.append(result.unstable_owner_tile_ids)
            candidates = unstable_owners
            self._gpu_frontier_active_tiles = (
                np.unique(np.concatenate(candidates)).astype(np.int32)
                if candidates
                else np.empty(0, dtype=np.int32)
            )
            if self._gpu_frontier_active_tiles.size == 0:
                assert self.device_state is not None
                self.device_state.runtime.arrays["frontier_reached"].zero_()
                return
        if self._gpu_frontier_active_tiles.size:
            raise RuntimeError(
                "NUMERICAL_NONCONVERGENCE: V3 GPU residual projection exceeded "
                f"{rounds_this_projection} bounded numerical rounds"
            )

    def reset(self, level: ResetLevel) -> None:
        selected = ResetLevel(level)
        if self.runtime_backend == "GPU_RUNTIME":
            self._reset_gpu(selected)
            return
        assert self.manager is not None
        if selected is ResetLevel.ALL:
            self.manager.reset()
        else:
            current = self.manager.current_state
            payload = current.payload
            resting = current.H_resting_m
            mobile = current.mobile_height_m
            momentum = current.mobile_momentum_m2_s
            parcels = current.airborne_parcels
            if selected is ResetLevel.PAYLOAD:
                payload = PayloadState(
                    0.0,
                    current.payload.capacity_m3,
                    current.material.assumed_bulk_density_kg_m3,
                    np.zeros(3),
                )
            elif selected is ResetLevel.TERRAIN:
                resting = self._initial_height
            elif selected is ResetLevel.ROBOT:
                # Robot ownership is in the Isaac frontend; physics reservoirs
                # intentionally remain untouched.
                self.previous_tool_state = None
                self._reset_generation += 1
                return
            rebuilt = TerrainState(
                resting,
                mobile,
                momentum,
                payload,
                parcels,
                current.material,
                current.outflow_volume_m3,
                current.timestamp_s,
                current.action_index,
            )
            self.manager = BulkStateManager(
                rebuilt,
                self.integrator,
                boundary_condition="closed",
                absolute_tolerance_m3=1.0e-8,
                relative_tolerance=1.0e-10,
            )
        self.previous_tool_state = None
        self._dump_released_phase_token = None
        self._dump_deposition_seed_mask.fill(False)
        self._host_mobile_export_cumulative_m3.fill(0.0)
        self._terrain_settled = True
        self._static_relaxation_pending = False
        self._static_relaxation_iterations = 0
        self._static_relaxation_active_tiles = 0
        self._last_avalanche_transition = None
        self._gpu_frontier_active_tiles = np.empty(0, dtype=np.int32)
        self._gpu_frontier_iterations = 0
        self._residual_solver_iterations_total = 0
        self._physical_simulation_time_s = 0.0
        self._dynamic_flow_time_s = 0.0
        self._last_physics_diagnostics = None
        self._last_gpu_scalar_snapshot = None
        self._last_gpu_avalanche_transition = None
        self.avalanche_controller.reset()
        self.slope_solver.reset()
        if isinstance(self.slope_solver, EventDrivenMinimumSlopeAdapter):
            self.slope_solver.set_reference_height(self.manager.current_state.H_resting_m)
        if hasattr(self.mobile_solver, "reset"):
            self.mobile_solver.reset()
        self.track_soil_model.initialize(self.manager.current_state.H_resting_m)
        self._reset_generation += 1

    def _reset_gpu(self, selected: ResetLevel) -> None:
        state = self.device_state
        metadata = self.gpu_metadata
        if state is None or metadata is None:
            raise RuntimeError("[V2PhysicsCore] GPU runtime was not initialized")
        if selected is ResetLevel.ROBOT:
            self.previous_tool_state = None
            self._reset_generation += 1
            return
        if selected is ResetLevel.ALL:
            state.reset()
            payload = self._initial_state.payload
            self.gpu_metadata = GpuRuntimeMetadata(state, payload)
        elif selected is ResetLevel.TERRAIN:
            # Reset is an explicit synchronization boundary; no host terrain
            # shadow is created and all non-terrain reservoirs are preserved.
            state.runtime.wp.copy(
                state.runtime.arrays["resting"],
                state.runtime.arrays["initial_resting"],
            )
            state.runtime.synchronize()
            state.mark_dirty_tiles(range(state.tile_count))
        elif selected is ResetLevel.PAYLOAD:
            empty = PayloadState(
                0.0,
                metadata.payload.capacity_m3,
                self.material.assumed_bulk_density_kg_m3,
                np.zeros(3),
            )
            # Rebuild the scalar ledger at an explicit reset boundary while
            # retaining spatial device authority.
            metadata.reset_payload(state, empty)
        self.previous_tool_state = None
        self._dump_released_phase_token = None
        self._host_mobile_export_cumulative_m3.fill(0.0)
        self._terrain_settled = True
        self._static_relaxation_pending = False
        self._static_relaxation_iterations = 0
        self._static_relaxation_active_tiles = 0
        self._last_avalanche_transition = None
        self._gpu_frontier_active_tiles = np.empty(0, dtype=np.int32)
        self._gpu_frontier_iterations = 0
        self._residual_solver_iterations_total = 0
        self._physical_simulation_time_s = 0.0
        self._dynamic_flow_time_s = 0.0
        self._last_physics_diagnostics = None
        self._last_gpu_scalar_snapshot = None
        self._last_gpu_avalanche_transition = None
        state.runtime.arrays["frontier_reached"].zero_()
        self.avalanche_controller.reset()
        assert self.gpu_chain is not None
        self.gpu_chain.large_avalanche.reset()
        self._reset_generation += 1

    def apply_track_soil(
        self,
        *,
        left_footprint_mask: np.ndarray,
        right_footprint_mask: np.ndarray,
        left_track_velocity_xy_m_s: np.ndarray,
        right_track_velocity_xy_m_s: np.ndarray,
        base_velocity_xy_m_s: np.ndarray,
        dt_s: float,
    ) -> TrackSoilResult | WarpTrackSoilStep:
        """Conservatively commit one reduced-order Track--Soil update."""

        if self.runtime_backend == "GPU_RUNTIME":
            if self.gpu_chain is None or self.device_state is None:
                raise RuntimeError("[V2PhysicsCore] GPU runtime was not initialized")
            result = self.gpu_chain.apply_track_soil(
                left_footprint_mask=left_footprint_mask,
                right_footprint_mask=right_footprint_mask,
                left_track_velocity_xy_m_s=left_track_velocity_xy_m_s,
                right_track_velocity_xy_m_s=right_track_velocity_xy_m_s,
                base_velocity_xy_m_s=base_velocity_xy_m_s,
                dt_s=dt_s,
            )
            if result.resting_to_mobile_volume_m3 > 0.0:
                self._terrain_settled = False
                self._last_gpu_scalar_snapshot = None
                self._last_gpu_avalanche_transition = None
            self.device_state.assert_normal_step_transfer_budget()
            return result

        assert self.manager is not None
        before = self.manager.current_state
        result = self.track_soil_model.apply(
            before.H_resting_m,
            before.mobile_height_m,
            before.mobile_momentum_m2_s,
            left_footprint_mask=left_footprint_mask,
            right_footprint_mask=right_footprint_mask,
            left_track_velocity_xy_m_s=left_track_velocity_xy_m_s,
            right_track_velocity_xy_m_s=right_track_velocity_xy_m_s,
            base_velocity_xy_m_s=base_velocity_xy_m_s,
            grid=self.grid,
            integrator=self.integrator,
            dt_s=dt_s,
        )
        next_state = TerrainState(
            result.H_resting_m,
            result.mobile_height_m,
            result.mobile_momentum_m2_s,
            before.payload,
            before.airborne_parcels,
            before.material,
            before.outflow_volume_m3,
            before.timestamp_s,
            before.action_index,
        )
        transfers = ()
        if result.resting_to_mobile_volume_m3 > 0.0:
            transfers = (
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.MOBILE,
                    result.resting_to_mobile_volume_m3,
                    "conservative_track_rut_to_mobile_shoulder",
                ),
            )
        self.manager.commit_transfers(next_state, transfers)
        return result

    def performance_report(self) -> dict[str, object]:
        report = self.profiler.report()
        report["backend"] = self.backend
        report["runtime_backend"] = self.runtime_backend
        report["slope_backend"] = self.slope_backend
        report["incremental_static_relaxation"] = {
            "round_budget_per_step": self.minislope_round_budget_per_step,
            "tolerance_m": self.minislope_tolerance_m,
            "pending": self._static_relaxation_pending,
            "iterations": self._static_relaxation_iterations,
            "active_tiles": self._static_relaxation_active_tiles,
            "terrain_settled": self._terrain_settled,
            "blocking_phase_end_solve": False,
        }
        report["large_avalanche_mobile_path"] = {
            "classification": (
                None
                if self._last_avalanche_transition is None
                else self._last_avalanche_transition.diagnostics.classification
            ),
            "parameter_basis": self.avalanche_controller.config.parameter_basis,
            "sensitivity_case": self.avalanche_controller.config.sensitivity_case,
            "terrain_settled_reason": self._last_settled_diagnostic.reason,
            "host_profile_ms": self.avalanche_controller.last_profile_ms,
        }
        if self.runtime_backend == "GPU_RUNTIME":
            assert self.gpu_chain is not None and self.device_state is not None
            report["gpu_runtime"] = self.gpu_chain.diagnostics()
            report["gpu_runtime"]["large_avalanche_status"] = (
                self._last_settled_diagnostic.large_avalanche_status
            )
            report["large_avalanche_mobile_path"]["classification"] = (
                self._last_settled_diagnostic.large_avalanche_status
            )
            report["large_avalanche_mobile_path"]["host_profile_ms"] = 0.0
            return report
        if isinstance(self.mobile_solver, OptimizedMobileLayerSolver):
            report["active_domain"] = (
                None
                if self.mobile_solver.last_active_snapshot is None
                else {
                    "active_ratio": self.mobile_solver.last_active_snapshot.active_ratio,
                    "active_tile_count": self.mobile_solver.last_active_snapshot.active_tile_count,
                    "tile_size": self.mobile_solver.last_active_snapshot.tile_size,
                }
            )
        return report

    def _make_host_physics_diagnostics(
        self,
        interaction: BulkInteractionResult | None,
        mobile_result,
        force_result: SoilForceResult | None,
        mass_error: float,
        moving_volume: float,
        dt_s: float,
        wall_ms: float,
    ) -> PhysicsDiagnostics:
        state = self.state
        failure_volume = 0.0 if interaction is None else interaction.failure_zone.active_volume_m3
        yielded_area = 0.0 if mobile_result is None else mobile_result.yielded_area_m2
        velocity_p95 = 0.0 if mobile_result is None else mobile_result.mobile_velocity_p95_m_s
        force = np.zeros(3) if force_result is None else force_result.force_terrain_n
        torque = np.zeros(3) if force_result is None else force_result.torque_about_tool_origin_terrain_nm
        terrain_state = (
            "SETTLED" if self._terrain_settled else
            "QUASI_STATIC_RESIDUAL_PROJECTION" if self._static_relaxation_pending else
            "DYNAMIC_MOBILE_FLOW"
        )
        return PhysicsDiagnostics(
            material_profile=state.material.name, backend=self.runtime_backend,
            resting_m3=self.integrator.integrate(state.H_resting_m),
            mobile_m3=self.integrator.integrate(state.mobile_height_m),
            payload_m3=state.payload.volume_m3, airborne_m3=state.airborne_volume_m3,
            soil_force_terrain_n=force, soil_torque_terrain_nm=torque,
            yielded_area_m2=yielded_area, failure_active_volume_m3=failure_volume,
            mobile_moving_volume_m3=moving_volume, mobile_velocity_p95_m_s=velocity_p95,
            terrain_state=terrain_state,
            not_settled_reason="" if self._terrain_settled else self._last_settled_diagnostic.reason,
            mass_balance_error_m3=abs(float(mass_error)),
            physical_simulation_time_s=self._physical_simulation_time_s,
            dynamic_flow_time_s=self._dynamic_flow_time_s,
            residual_solver_iterations=self._residual_solver_iterations_total,
            rtf=float(dt_s / max(wall_ms * 1.0e-3, 1.0e-12)),
        )

    def _make_gpu_physics_diagnostics(
        self,
        interaction: GpuBulkInteractionResult | None,
        force_result: SoilForceResult | None,
        snapshot: GpuRuntimeScalarSnapshot,
        moving_volume: float,
        maximum_speed: float,
        avalanche: DeviceLargeAvalancheResult,
        dt_s: float,
        wall_ms: float,
    ) -> PhysicsDiagnostics:
        ledger = snapshot.material_ledger
        failure_volume = 0.0 if interaction is None else interaction.failure_zone.active_volume_m3
        force = np.zeros(3) if force_result is None else force_result.force_terrain_n
        torque = np.zeros(3) if force_result is None else force_result.torque_about_tool_origin_terrain_nm
        reason = (
            "" if self._terrain_settled else self._last_settled_diagnostic.reason
        )
        # State names describe the solver which actually owns this step.  A
        # compact frontier may be pending while Mobile or physical yield still
        # owns the terrain; pending work alone is not a residual projection.
        terrain_state = (
            "SETTLED"
            if self._terrain_settled
            else "QUASI_STATIC_RESIDUAL_PROJECTION"
            if reason == "DEVICE_RESIDUAL_PROJECTION_ACTIVE"
            else "RESIDUAL_PROJECTION_PENDING"
            if reason == "DEVICE_RESIDUAL_PROJECTION_PENDING"
            else "DYNAMIC_MOBILE_FLOW"
        )
        return PhysicsDiagnostics(
            material_profile=self.material.name, backend="GPU_RUNTIME/DEVICE",
            resting_m3=ledger.resting_m3, mobile_m3=ledger.mobile_m3,
            payload_m3=ledger.payload_m3, airborne_m3=ledger.airborne_m3,
            soil_force_terrain_n=force, soil_torque_terrain_nm=torque,
            yielded_area_m2=float(avalanche.largest_connected_area_m2),
            failure_active_volume_m3=failure_volume,
            mobile_moving_volume_m3=moving_volume,
            # Device p95 reduction is intentionally not a full-field D2H;
            # maximum speed is the conservative compact scalar proxy.
            mobile_velocity_p95_m_s=float(maximum_speed), terrain_state=terrain_state,
            not_settled_reason=reason,
            mass_balance_error_m3=abs(float(ledger.absolute_volume_error_m3)),
            physical_simulation_time_s=self._physical_simulation_time_s,
            dynamic_flow_time_s=self._dynamic_flow_time_s,
            residual_solver_iterations=self._residual_solver_iterations_total,
            rtf=float(dt_s / max(wall_ms * 1.0e-3, 1.0e-12)),
        )

    def _advance_static_relaxation_budget(self) -> None:
        """Perform a bounded numerical projection, never physical-time flow."""

        state = self.manager.current_state
        settled = self.avalanche_controller.settled_diagnostic(
            state.H_resting_m,
            state.mobile_height_m,
            state.mobile_momentum_m2_s,
            state.material,
            self.grid,
            self.integrator,
        )
        self._last_settled_diagnostic = settled
        dynamic_quiet = (
            state.airborne_volume_m3 <= 1.0e-10
            and not settled.current_mobile_active
            and settled.mobile_volume_m3
            <= self.avalanche_controller.config.settled_mobile_volume_m3
        )
        solver = self.slope_solver
        if not dynamic_quiet:
            self._static_relaxation_pending = bool(
                np.any(self._dump_deposition_seed_mask)
                or (
                    isinstance(solver, SparseTileFrontierMinimumSlopeAdapter)
                    and solver.incremental_active
                )
            )
            self._terrain_settled = False
            return
        if settled.settled:
            # A deposition seed is not itself a physical instability.  In
            # particular, cohesive steep cuts may be above a geometric repose
            # angle while remaining below yield.  Discard observational seeds
            # without changing the free surface.
            self._dump_deposition_seed_mask.fill(False)
            self._static_relaxation_pending = False
            self._static_relaxation_active_tiles = 0
            self._update_terrain_settled_status()
            return
        if not settled.ready_for_final_minislope:
            # A connected large static failure is still accumulating physical
            # persistence and must remain owned by the Resting->Mobile path.
            # MiniSlope must not erase it before the transition criterion is
            # evaluated over finite simulation time.
            self._static_relaxation_pending = True
            self._terrain_settled = False
            return

        if isinstance(solver, SparseTileFrontierMinimumSlopeAdapter):
            if not solver.incremental_active and np.any(
                self._dump_deposition_seed_mask
            ):
                solver.set_reference_height(state.H_resting_m)
                solver.mark_changed(self._dump_deposition_seed_mask)
                solver.begin_incremental(
                    state.H_resting_m, self._dump_deposition_seed_mask
                )
                self._dump_deposition_seed_mask.fill(False)
                self._static_relaxation_iterations = 0
            if solver.incremental_active:
                with self.profiler.measure("minislope_incremental"):
                    start_iteration = self._static_relaxation_iterations
                    progress = solver.advance_incremental(
                        round_budget=self.residual_projection_max_rounds
                    )
                self._commit_incremental_resting(progress.heightmap_m)
                self._static_relaxation_iterations = progress.iteration_count
                self._residual_solver_iterations_total += max(
                    0, progress.iteration_count - start_iteration
                )
                self._static_relaxation_active_tiles = progress.active_tile_count
                self._static_relaxation_pending = not progress.complete
                if not progress.complete:
                    raise RuntimeError(
                        "NUMERICAL_NONCONVERGENCE: V3 bounded residual projection "
                        f"exceeded {self.residual_projection_max_rounds} rounds; "
                        "large physical residuals must remain on the Mobile/yield path"
                    )
            else:
                self._static_relaxation_pending = False
                self._static_relaxation_active_tiles = 0
        else:
            # Monolithic reference/bounding-box solvers remain available for
            # offline CPU_REFERENCE acceptance only.  They are deliberately
            # never invoked from the interactive step path.
            self._static_relaxation_pending = bool(
                np.any(self._dump_deposition_seed_mask)
            )
            self._static_relaxation_active_tiles = 0
        self._update_terrain_settled_status()

    def _advance_large_avalanche_transition(
        self, dt_s: float
    ) -> RestingToMobileTransitionResult:
        before = self.manager.current_state
        result = self.avalanche_controller.observe_and_maybe_mobilize(
            before.H_resting_m,
            before.mobile_height_m,
            before.mobile_momentum_m2_s,
            before.material,
            self.grid,
            self.integrator,
            dt_s,
            release_settled_latches=not (
                isinstance(
                    self.slope_solver,
                    SparseTileFrontierMinimumSlopeAdapter,
                )
                and self.slope_solver.incremental_active
            ),
            conservative_export_cumulative_m3=(
                self._host_mobile_export_cumulative_m3
            ),
        )
        if (
            result.diagnostics.classification
            == self.avalanche_controller.LOCAL_STATIC_INSTABILITY
        ):
            self._dump_deposition_seed_mask |= (
                result.diagnostics.largest_connected_mask
            )
        if not result.transitioned:
            return result
        next_state = TerrainState(
            result.H_resting_m,
            result.mobile_height_m,
            result.mobile_momentum_m2_s,
            before.payload,
            before.airborne_parcels,
            before.material,
            before.outflow_volume_m3,
            before.timestamp_s,
            before.action_index,
        )
        self.manager.commit_transfers(
            next_state,
            (
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.MOBILE,
                    result.transferred_volume_m3,
                    "physical_large_avalanche_resting_to_mobile",
                ),
            ),
        )
        return result

    def _commit_incremental_resting(self, resting: np.ndarray) -> None:
        before = self.manager.current_state
        before_volume = self.integrator.integrate(before.H_resting_m)
        after_volume = self.integrator.integrate(resting)
        tolerance = max(1.0e-11, 1.0e-10 * max(before_volume, 1.0))
        if abs(before_volume - after_volume) > tolerance:
            raise RuntimeError(
                "[V2PhysicsCore] incremental MiniSlope violated closed-domain "
                f"volume conservation: before={before_volume}, after={after_volume}"
            )
        next_state = TerrainState(
            resting,
            before.mobile_height_m,
            before.mobile_momentum_m2_s,
            before.payload,
            before.airborne_parcels,
            before.material,
            before.outflow_volume_m3,
            before.timestamp_s,
            before.action_index,
        )
        self.manager.commit_transfers(next_state, ())

    def _update_terrain_settled_status(self) -> None:
        state = self.manager.current_state
        physical = self.avalanche_controller.settled_diagnostic(
            state.H_resting_m,
            state.mobile_height_m,
            state.mobile_momentum_m2_s,
            state.material,
            self.grid,
            self.integrator,
        )
        self._last_settled_diagnostic = physical
        self._terrain_settled = bool(
            state.airborne_volume_m3 <= 1.0e-10
            and physical.settled
            and physical.mobile_volume_m3
            <= self.avalanche_controller.config.settled_mobile_volume_m3
            and not self._static_relaxation_pending
            and not np.any(self._dump_deposition_seed_mask)
        )
