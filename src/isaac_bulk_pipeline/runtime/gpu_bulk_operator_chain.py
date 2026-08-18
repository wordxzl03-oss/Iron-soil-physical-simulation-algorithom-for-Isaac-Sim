"""Shared-device execution chain for the accepted Warp bulk operators.

This is intentionally a state-owner adapter, not a second physics model.  It
binds the already accepted Mobile, TrackSoil and CompactActiveEdge kernels to
one :class:`DeviceBulkState` so a later runtime phase cannot accidentally keep
three divergent GPU terrain copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from ..bulk_interaction import (
    WarpProductionMobileV2Solver,
    WarpMobileStep,
    WarpDepositionOperator,
    WarpDepositionStep,
    WarpTrackSoilOperator,
    WarpTrackSoilStep,
)
from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..solvers import CompactActiveEdgeBatch, WarpCompactActiveEdgeOperator, WarpFrontierPhaseResult
from ..terrain import TerrainGrid
from ..interaction import SweepResult
from ..tools import ToolState
from .bulk_state_authority import DeviceBulkState
from .gpu_failure_bridge import DeviceFailureZoneBridge, DeviceFailureZoneResult
from .gpu_intake_bridge import DeviceBucketIntakeBridge, DeviceIntakeResult
from .gpu_airborne_bridge import DeviceAirborneAdvanceResult, DeviceAirborneBridge
from ..bulk_state import PayloadState
from ..tools import ToolDescriptor
from ..bulk_state import MaterialParcel
from ..bulk_interaction.large_avalanche import LargeAvalancheTransitionConfig
from ..experimental.mobile_v2_reference import MobileV2Config
from .gpu_large_avalanche import (
    DeviceLargeAvalancheBridge,
    DeviceLargeAvalancheResult,
)


@dataclass(frozen=True)
class GpuBulkOperatorStep:
    module: str
    wall_time_ms: float
    transfer: dict[str, int | float]
    dirty_tile_count: int


class GpuBulkOperatorChain:
    """One-device bridge for the three existing GPU terrain operators."""

    backend_identity = "GPU_RUNTIME_SHARED_DEVICE_OPERATOR_CHAIN"

    def __init__(
        self,
        state: DeviceBulkState,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        large_avalanche_config: LargeAvalancheTransitionConfig | None = None,
        descriptor: ToolDescriptor | None = None,
    ) -> None:
        if state.grid is not grid:
            raise ValueError("[GpuBulkChain] state/grid identity mismatch")
        self.state = state
        self.material = material
        self.grid = grid
        self.integrator = integrator
        self.mobile = WarpProductionMobileV2Solver(
            runtime=state.runtime,
            config=MobileV2Config(
                dx_m=grid.dx,
                dy_m=grid.dy,
                start_angle_deg=material.start_angle_deg,
                stop_angle_deg=material.stop_angle_deg,
                basal_friction_coefficient=material.mobile_friction_coefficient,
            ),
        )
        self.mobile.bind_device_state(state, material, grid, integrator)
        self.track = WarpTrackSoilOperator(grid.shape, runtime=state.runtime)
        self.track.bind_device_state(state)
        self.deposition = WarpDepositionOperator(runtime=state.runtime)
        self.deposition.bind_device_state(state)
        self.frontier = WarpCompactActiveEdgeOperator(
            grid.shape, runtime=state.runtime, tile_size=state.tile_size
        )
        self.frontier.bind_device_state(state)
        self.failure_zone = DeviceFailureZoneBridge(
            state, material, grid, integrator, descriptor=descriptor
        )
        self.intake = DeviceBucketIntakeBridge(state, grid, integrator)
        from ..bulk_exchange.airborne import AirborneParcelModel

        self.airborne = DeviceAirborneBridge(
            state,
            grid,
            integrator,
            parcel_model=AirborneParcelModel(descriptor=descriptor, material=material),
        )
        self.large_avalanche = DeviceLargeAvalancheBridge(
            state,
            material,
            grid,
            integrator,
            large_avalanche_config or LargeAvalancheTransitionConfig(),
        )
        self._records: list[GpuBulkOperatorStep] = []
        # Acceptance-only state boundary observer.  Production leaves this
        # unset, so no full-field device transfer is added to the normal path.
        # It exists to make operator causality directly auditable rather than
        # reconstructing intermediate fields from an end-of-step terrain.
        self.audit_state_observer: Callable[[str], None] | None = None
        self.audit_mobile_substep_observer: (
            Callable[[int, float, np.ndarray], None] | None
        ) = None
        self.audit_tool_contact_observer: Callable[[object], None] | None = None
        self.airborne.audit_state_observer = self._audit_boundary
        self.failure_zone.audit_state_observer = self._audit_boundary
        self.mobile.audit_substep_observer = self._audit_mobile_substep
        self._audit_readback_total_ms = 0.0

    def _audit_boundary(self, label: str) -> None:
        if self.audit_state_observer is not None:
            start = perf_counter()
            self.audit_state_observer(label)
            self._audit_readback_total_ms += (perf_counter() - start) * 1_000.0

    def _audit_mobile_substep(
        self, substep_index: int, dt_s: float, diagnostic: np.ndarray
    ) -> None:
        if self.audit_mobile_substep_observer is not None:
            start = perf_counter()
            self.audit_mobile_substep_observer(substep_index, dt_s, diagnostic)
            self._audit_readback_total_ms += (perf_counter() - start) * 1_000.0

    @property
    def audit_readback_total_ms(self) -> float:
        """Cumulative acceptance-observer wall time; no observer means zero."""

        return float(self._audit_readback_total_ms)

    def _record(self, module: str, start: float, dirty_tile_count: int) -> None:
        self._records.append(
            GpuBulkOperatorStep(
                module=module,
                wall_time_ms=(perf_counter() - start) * 1_000.0,
                transfer=self.state.transfer_snapshot().to_dict(),
                dirty_tile_count=int(dirty_tile_count),
            )
        )

    def step_mobile(
        self,
        dt_s: float,
        *,
        tool_mobile_contact=None,
    ) -> WarpMobileStep:
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        if self.audit_tool_contact_observer is not None:
            audit_start = perf_counter()
            self.audit_tool_contact_observer(tool_mobile_contact)
            self._audit_readback_total_ms += (
                perf_counter() - audit_start
            ) * 1_000.0
        result = self.mobile.step_resident(
            dt_s,
            tool_mobile_contact=tool_mobile_contact,
            tool_mobile_friction_coefficient=self.material.tool_friction_coefficient,
        )
        self.state.advance_time(dt_s)
        dirty = self.state.collect_surface_dirty_tiles()
        self._audit_boundary(
            "AFTER_MOBILE_FUSED_FACE_TOPOGRAPHY_TOOL_FRICTION"
        )
        self._record("Mobile", start, dirty.size)
        return result

    def apply_failure_zone(
        self, sweep: SweepResult, tool_state: ToolState
    ) -> DeviceFailureZoneResult:
        """Commit real CAD FailureZone activation through a compact bridge."""

        start = perf_counter()
        result = self.failure_zone.execute(sweep, tool_state)
        self._record("FailureZoneActivation", start, 0)
        return result

    def clear_failure_tool_forcing(self, result: DeviceFailureZoneResult) -> None:
        self.failure_zone.clear_tool_forcing(result.tool_mobile_contact)

    def apply_bucket_intake(
        self,
        payload: PayloadState,
        tool_state: ToolState,
        descriptor: ToolDescriptor,
        dt_s: float,
    ) -> DeviceIntakeResult:
        start = perf_counter()
        result = self.intake.execute(payload, tool_state, descriptor, dt_s)
        self._audit_boundary("AFTER_BUCKET_INTAKE")
        self._record("BucketIntake", start, 0)
        return result

    def advance_airborne(
        self, parcels: tuple[MaterialParcel, ...], dt_s: float
    ) -> DeviceAirborneAdvanceResult:
        start = perf_counter()
        result = self.airborne.advance(parcels, dt_s)
        self._audit_boundary("AFTER_AIRBORNE_TO_MOBILE")
        self._record("AirborneLanding", start, 0)
        return result

    def apply_track_soil(
        self,
        *,
        left_footprint_mask: np.ndarray,
        right_footprint_mask: np.ndarray,
        left_track_velocity_xy_m_s: np.ndarray,
        right_track_velocity_xy_m_s: np.ndarray,
        base_velocity_xy_m_s: np.ndarray,
        dt_s: float,
    ) -> WarpTrackSoilStep:
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        result = self.track.apply_compact_resident(
            left_footprint_indices=np.flatnonzero(left_footprint_mask),
            right_footprint_indices=np.flatnonzero(right_footprint_mask),
            left_track_velocity_xy_m_s=left_track_velocity_xy_m_s,
            right_track_velocity_xy_m_s=right_track_velocity_xy_m_s,
            base_velocity_xy_m_s=base_velocity_xy_m_s,
            dt_s=dt_s,
        )
        dirty = self.state.collect_surface_dirty_tiles()
        self._record("TrackSoil", start, dirty.size)
        return result

    def step_deposition(self, dt_s: float) -> WarpDepositionStep:
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        result = self.deposition.step_resident(self.material, dt_s)
        dirty = self.state.collect_surface_dirty_tiles()
        self._audit_boundary("AFTER_DEPOSITION")
        self._record("Deposition", start, dirty.size)
        return result

    def advance_large_avalanche(
        self, dt_s: float, *, release_settled_latches: bool = True
    ) -> DeviceLargeAvalancheResult:
        start = perf_counter()
        result = self.large_avalanche.observe_and_maybe_mobilize(
            dt_s, release_settled_latches=release_settled_latches
        )
        self._audit_boundary("AFTER_LARGE_AVALANCHE")
        self._record("LargeAvalancheDeviceTransition", start, 0)
        return result

    def relax_frontier_phase(
        self,
        batch: CompactActiveEdgeBatch,
        *,
        critical_difference_m: float,
        tolerance_m: float,
    ) -> WarpFrontierPhaseResult:
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        result = self.frontier.run_phase(
            batch,
            critical_difference_m=critical_difference_m,
            tolerance_m=tolerance_m,
        )
        dirty = self.state.collect_surface_dirty_tiles()
        self._record("MiniSlopeCompactFrontier", start, dirty.size)
        return result

    def transfer_frontier_phase(
        self,
        batch: CompactActiveEdgeBatch,
        *,
        critical_difference_m: float,
    ) -> WarpFrontierPhaseResult:
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        result = self.frontier.relax_phase(
            batch, critical_difference_m=critical_difference_m
        )
        dirty = self.state.collect_surface_dirty_tiles()
        self._record("MiniSlopeCompactFrontier", start, dirty.size)
        return result

    def scan_frontier_phase(
        self,
        batch: CompactActiveEdgeBatch,
        *,
        critical_difference_m: float,
        tolerance_m: float,
    ) -> WarpFrontierPhaseResult:
        start = perf_counter()
        result = self.frontier.scan_phase(
            batch,
            critical_difference_m=critical_difference_m,
            tolerance_m=tolerance_m,
        )
        self._record("MiniSlopeCompactFrontier", start, 0)
        return result

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend_identity": self.backend_identity,
            "state_authority": self.state.authority.value,
            "state": self.state.diagnostics(),
            "operators": {
                "mobile": self.mobile.diagnostics(),
                "track_soil": self.track.diagnostics(),
                "deposition": self.deposition.diagnostics(),
                "compact_frontier": self.frontier.diagnostics(),
                "failure_zone": self.failure_zone.diagnostics(),
                "bucket_intake": {"backend": self.intake.backend_identity},
                "airborne": {"backend": self.airborne.backend_identity},
                "large_avalanche": self.large_avalanche.diagnostics(),
            },
            "records": [
                {
                    "module": item.module,
                    "wall_time_ms": item.wall_time_ms,
                    "transfer": item.transfer,
                    "dirty_tile_count": item.dirty_tile_count,
                }
                for item in self._records
            ],
        }
