"""CPU parcel integration with batched device terrain landing queries."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from ..bulk_exchange.airborne import AirborneLandingRecord, AirborneParcelModel
from ..bulk_state import MaterialParcel, TerrainVolumeIntegrator
from ..terrain import TerrainGrid
from .bulk_state_authority import DeviceBulkState


@dataclass(frozen=True)
class DeviceAirborneAdvanceResult:
    remaining_parcels: tuple[MaterialParcel, ...]
    landed_volume_m3: float
    landing_points_world_m: np.ndarray
    parcel_count_before: int
    parcel_count_after: int
    landing_flat_indices: np.ndarray
    timings_ms: dict[str, float]
    landing_records: tuple[AirborneLandingRecord, ...] = ()


class DeviceAirborneBridge:
    """Keep sparse ballistic parcels on host without a host terrain field."""

    backend_identity = "GPU_RUNTIME_CPU_PARCELS_BATCHED_DEVICE_LANDING_QUERY"

    def __init__(
        self,
        state: DeviceBulkState,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        parcel_model: AirborneParcelModel | None = None,
    ) -> None:
        if state.grid is not grid:
            raise ValueError("[GpuAirborneBridge] state/grid identity mismatch")
        self.state = state
        self.grid = grid
        self.integrator = integrator
        self.parcel_model = parcel_model or AirborneParcelModel()
        # Acceptance-only boundary hook. Production leaves this unset.
        self.audit_state_observer: Callable[[str], None] | None = None

    def _audit_boundary(self, label: str) -> None:
        if self.audit_state_observer is not None:
            self.audit_state_observer(label)

    def advance(
        self, parcels: tuple[MaterialParcel, ...], dt_s: float
    ) -> DeviceAirborneAdvanceResult:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[GpuAirborneBridge] dt_s must be finite/positive")
        gravity = np.asarray([0.0, 0.0, -self.parcel_model.config.gravity_m_s2])
        candidates: list[tuple[MaterialParcel, np.ndarray, np.ndarray, float, float]] = []
        remaining: list[MaterialParcel] = []
        resolved_parcels = self.parcel_model.resolve_missing_footprints(parcels)
        for parcel in resolved_parcels:
            position = (
                np.asarray(parcel.position_world_m)
                + np.asarray(parcel.velocity_world_m_s) * dt
                + 0.5 * gravity * dt * dt
            )
            velocity = np.asarray(parcel.velocity_world_m_s) + gravity * dt
            terrain_point = self.grid.world_to_terrain(position)
            row, col = self.grid.terrain_to_grid(terrain_point)
            if 0.0 <= row <= self.grid.ny - 1 and 0.0 <= col <= self.grid.nx - 1:
                candidates.append((parcel, position, velocity, float(row), float(col)))
            else:
                remaining.append(self._advanced_parcel(parcel, position, velocity, dt))
        query_start = perf_counter()
        query_rc = np.asarray([[row, col] for _, _, _, row, col in candidates], dtype=np.float64).reshape(-1, 2)
        heights = self.state.sample_surface_bilinear(query_rc, source="airborne_landing")
        query_ms = (perf_counter() - query_start) * 1_000.0
        indices: list[int] = []
        mobile_add: list[float] = []
        momentum_x_add: list[float] = []
        momentum_y_add: list[float] = []
        landed = 0.0
        points: list[np.ndarray] = []
        landing_records: list[AirborneLandingRecord] = []
        for (parcel, position, velocity, row_f, col_f), surface in zip(candidates, heights):
            terrain_point = self.grid.world_to_terrain(position)
            if terrain_point[2] <= float(surface) + self.parcel_model.config.terrain_clearance_m:
                record = self.parcel_model.rasterize_landing(
                    parcel, position, velocity, self.grid, self.integrator
                )
                delta_h = (
                    record.delivered_volumes_m3
                    / record.recipient_control_areas_m2
                )
                indices.extend(record.recipient_flat_indices.tolist())
                mobile_add.extend(delta_h.tolist())
                momentum_x_add.extend(
                    (
                        delta_h
                        * velocity[0]
                        * self.parcel_model.config.landing_velocity_scale
                    ).tolist()
                )
                momentum_y_add.extend(
                    (
                        delta_h
                        * velocity[1]
                        * self.parcel_model.config.landing_velocity_scale
                    ).tolist()
                )
                landed += parcel.volume_m3
                landing_records.append(record)
                points.append(
                    self.grid.terrain_to_world(
                        np.asarray([terrain_point[0], terrain_point[1], float(surface)])
                    )
                )
            else:
                remaining.append(self._advanced_parcel(parcel, position, velocity, dt))
        scatter_start = perf_counter()
        flat = np.asarray(indices, dtype=np.int32)
        if flat.size:
            self._audit_boundary("AIRBORNE_COLLISION_DETECTED")
            self.state.capture_surface_for_dirty_tracking()
            self.state.add_host_indices("mobile", flat, np.asarray(mobile_add), reason="ballistic_parcel_landing")
            self._audit_boundary("POST_AIRBORNE_TO_MOBILE_WRITE")
            self.state.add_host_indices("momentum_x", flat, np.asarray(momentum_x_add), reason="ballistic_parcel_landing")
            self.state.add_host_indices("momentum_y", flat, np.asarray(momentum_y_add), reason="ballistic_parcel_landing")
            self._audit_boundary("POST_MOBILE_MOMENTUM_WRITE")
            self.state.collect_surface_dirty_tiles()
        scatter_ms = (perf_counter() - scatter_start) * 1_000.0
        return DeviceAirborneAdvanceResult(
            remaining_parcels=tuple(remaining),
            landed_volume_m3=float(landed),
            landing_points_world_m=np.asarray(points, dtype=np.float64).reshape(-1, 3),
            parcel_count_before=len(parcels),
            parcel_count_after=len(remaining),
            landing_flat_indices=flat,
            timings_ms={"terrain_landing_query": query_ms, "landing_scatter": scatter_ms},
            landing_records=tuple(landing_records),
        )

    @staticmethod
    def _advanced_parcel(
        parcel: MaterialParcel, position: np.ndarray, velocity: np.ndarray, dt_s: float
    ) -> MaterialParcel:
        return MaterialParcel(
            parcel_id=parcel.parcel_id,
            volume_m3=parcel.volume_m3,
            assumed_bulk_density_kg_m3=parcel.assumed_bulk_density_kg_m3,
            position_world_m=position,
            velocity_world_m_s=velocity,
            timestamp_s=parcel.timestamp_s + dt_s,
            source=parcel.source,
            footprint=parcel.footprint,
        )
