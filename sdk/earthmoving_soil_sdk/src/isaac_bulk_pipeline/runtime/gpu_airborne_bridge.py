"""CPU parcel integration with batched device terrain landing queries."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..bulk_exchange.airborne import AirborneParcelModel
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

    def advance(
        self, parcels: tuple[MaterialParcel, ...], dt_s: float
    ) -> DeviceAirborneAdvanceResult:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[GpuAirborneBridge] dt_s must be finite/positive")
        gravity = np.asarray([0.0, 0.0, -self.parcel_model.config.gravity_m_s2])
        candidates: list[tuple[MaterialParcel, np.ndarray, np.ndarray, float, float]] = []
        remaining: list[MaterialParcel] = []
        for parcel in parcels:
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
        for (parcel, position, velocity, row_f, col_f), surface in zip(candidates, heights):
            terrain_point = self.grid.world_to_terrain(position)
            if terrain_point[2] <= float(surface) + self.parcel_model.config.terrain_clearance_m:
                row = int(np.clip(round(row_f), 0, self.grid.ny - 1))
                col = int(np.clip(round(col_f), 0, self.grid.nx - 1))
                weight = float(self.integrator.vertex_weights_m2[row, col])
                if weight <= 0.0:
                    raise RuntimeError("[GpuAirborneBridge] landing control area is zero")
                indices.append(row * self.grid.nx + col)
                mobile_add.append(parcel.volume_m3 / weight)
                momentum_x_add.append(
                    parcel.volume_m3 * velocity[0] * self.parcel_model.config.landing_velocity_scale / weight
                )
                momentum_y_add.append(
                    parcel.volume_m3 * velocity[1] * self.parcel_model.config.landing_velocity_scale / weight
                )
                landed += parcel.volume_m3
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
            self.state.capture_surface_for_dirty_tracking()
            self.state.add_host_indices("mobile", flat, np.asarray(mobile_add), reason="ballistic_parcel_landing")
            self.state.add_host_indices("momentum_x", flat, np.asarray(momentum_x_add), reason="ballistic_parcel_landing")
            self.state.add_host_indices("momentum_y", flat, np.asarray(momentum_y_add), reason="ballistic_parcel_landing")
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
        )
