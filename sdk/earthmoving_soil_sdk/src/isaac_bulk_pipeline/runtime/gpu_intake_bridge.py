"""Compact DeviceBulkState bridge for the accepted bucket-mouth intake model."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..bulk_interaction import BucketIntakeModel, BucketIntakeResult
from ..bulk_state import PayloadState, TerrainVolumeIntegrator
from ..terrain import TerrainGrid
from ..tools import ToolDescriptor, ToolState
from .bulk_state_authority import DeviceBulkState


@dataclass(frozen=True)
class DevicePayloadTransaction:
    """Logical Mobile→Payload transaction coupled to the device scatter."""

    volume_m3: float
    payload_before_m3: float
    payload_after_m3: float
    source: str = "bucket_mouth_relative_flux"

    def __post_init__(self) -> None:
        if not np.isclose(self.payload_after_m3 - self.payload_before_m3, self.volume_m3, atol=1e-11):
            raise RuntimeError("[GpuIntakeBridge] payload transaction is not conservative")


@dataclass(frozen=True)
class DeviceIntakeResult:
    patch_bbox_yx: tuple[int, int, int, int]
    patch_cell_count: int
    patch_fraction_of_terrain: float
    intake: BucketIntakeResult
    transaction: DevicePayloadTransaction
    timings_ms: dict[str, float]


class DeviceBucketIntakeBridge:
    """Apply real bucket-mouth flux using only its geometry-derived patch."""

    backend_identity = "GPU_RUNTIME_COMPACT_HOST_BUCKET_INTAKE_BRIDGE"

    def __init__(
        self,
        state: DeviceBulkState,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        intake_model: BucketIntakeModel | None = None,
    ) -> None:
        if state.grid is not grid:
            raise ValueError("[GpuIntakeBridge] state/grid identity mismatch")
        self.state = state
        self.grid = grid
        self.integrator = integrator
        self.intake_model = intake_model or BucketIntakeModel()

    def execute(
        self,
        payload: PayloadState,
        tool_state: ToolState,
        descriptor: ToolDescriptor,
        dt_s: float,
    ) -> DeviceIntakeResult:
        bbox = self._mouth_bbox(tool_state, descriptor)
        start = perf_counter()
        patch = self.state.download_region(bbox, source="bucket_intake")
        query_ms = (perf_counter() - start) * 1_000.0
        row0, row1, col0, col1 = bbox
        local = TerrainGrid(
            col1 - col0, row1 - row0, self.grid.dx, self.grid.dy,
            self.grid.origin_x + col0 * self.grid.dx,
            self.grid.origin_y + row0 * self.grid.dy,
            "/World/Terrain/CompactIntakePatch",
            terrain_to_world_matrix=self.grid.terrain_to_world_matrix,
        )
        local_integrator = TerrainVolumeIntegrator.from_grid(local)
        start = perf_counter()
        intake = self.intake_model.apply(
            patch.H_mobile_m,
            patch.mobile_momentum_m2_s,
            payload,
            tool_state,
            descriptor,
            local,
            local_integrator,
            dt_s,
            terrain_surface_m=patch.H_resting_m,
        )
        intake_ms = (perf_counter() - start) * 1_000.0
        start = perf_counter()
        self.state.capture_surface_for_dirty_tracking()
        self.state.apply_host_patch("mobile", bbox, intake.mobile_height_m, reason="bucket_intake")
        self.state.apply_host_patch("momentum_x", bbox, intake.mobile_momentum_m2_s[..., 0], reason="bucket_intake")
        self.state.apply_host_patch("momentum_y", bbox, intake.mobile_momentum_m2_s[..., 1], reason="bucket_intake")
        self.state.collect_surface_dirty_tiles()
        scatter_ms = (perf_counter() - start) * 1_000.0
        transaction = DevicePayloadTransaction(
            volume_m3=float(intake.bucket_inflow_volume_m3),
            payload_before_m3=float(payload.volume_m3),
            payload_after_m3=float(intake.payload.volume_m3),
        )
        return DeviceIntakeResult(
            patch_bbox_yx=bbox,
            patch_cell_count=(row1 - row0) * (col1 - col0),
            patch_fraction_of_terrain=float((row1 - row0) * (col1 - col0) / self.state.size),
            intake=intake,
            transaction=transaction,
            timings_ms={
                "terrain_patch_query": query_ms,
                "bucket_intake": intake_ms,
                "intake_scatter": scatter_ms,
            },
        )

    def _mouth_bbox(self, tool_state: ToolState, descriptor: ToolDescriptor) -> tuple[int, int, int, int]:
        geometry = descriptor.bucket_geometry
        if geometry is None:
            raise ValueError("[GpuIntakeBridge] authoritative bucket geometry is required")
        lip = geometry.transform_points(tool_state.pose_terrain, geometry.lip_local)
        x_min, y_min = np.minimum(lip[0, :2], lip[-1, :2])
        x_max, y_max = np.maximum(lip[0, :2], lip[-1, :2])
        # Exactly mirrors BucketIntakeModel's control-volume bounds; the
        # one-cell extension is geometric discretization, not a tuning halo.
        col0 = max(0, int(np.floor((x_min - self.grid.origin_x) / self.grid.dx)) - 1)
        col1 = min(self.grid.nx, int(np.ceil((x_max - self.grid.origin_x) / self.grid.dx)) + 2)
        row0 = max(0, int(np.floor((y_min - self.grid.origin_y) / self.grid.dy)) - 1)
        row1 = min(self.grid.ny, int(np.ceil((y_max - self.grid.origin_y) / self.grid.dy)) + 2)
        if row1 <= row0 or col1 <= col0:
            raise RuntimeError("[GpuIntakeBridge] empty mouth geometry patch")
        return row0, row1, col0, col1
