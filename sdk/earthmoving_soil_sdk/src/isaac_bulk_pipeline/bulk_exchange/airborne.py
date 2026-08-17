"""Coarse authoritative airborne parcels with ballistic terrain landing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import MaterialParcel, MaterialScenario, TerrainVolumeIntegrator
from ..terrain.terrain_grid import TerrainGrid
from ..tools import ToolDescriptor, ToolState


def _ro(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AirborneParcelConfig:
    target_volume_m3: float = 0.04
    minimum_count: int = 1
    maximum_count: int = 100
    gravity_m_s2: float = 9.81
    terrain_clearance_m: float = 0.01
    landing_velocity_scale: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.target_volume_m3,
                self.gravity_m_s2,
                self.terrain_clearance_m,
                self.landing_velocity_scale,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[Airborne] numeric configuration must be finite/positive")
        if not isinstance(self.minimum_count, int) or not isinstance(self.maximum_count, int):
            raise TypeError("[Airborne] parcel counts must be integers")
        if self.minimum_count < 1 or self.maximum_count < self.minimum_count:
            raise ValueError("[Airborne] invalid parcel count bounds")


@dataclass(frozen=True)
class AirborneAdvanceResult:
    remaining_parcels: tuple[MaterialParcel, ...]
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    landed_volume_m3: float
    landing_points_world_m: np.ndarray
    parcel_count_before: int
    parcel_count_after: int

    def __post_init__(self) -> None:
        height = np.asarray(self.mobile_height_m, dtype=np.float64)
        momentum = np.asarray(self.mobile_momentum_m2_s, dtype=np.float64)
        points = np.asarray(self.landing_points_world_m, dtype=np.float64)
        if height.ndim != 2 or momentum.shape != height.shape + (2,):
            raise ValueError("[Airborne] output mobile state shapes invalid")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("[Airborne] landing points must have shape (N,3)")
        if not np.all(np.isfinite(height)) or np.any(height < 0.0) or not np.all(np.isfinite(momentum)) or not np.all(np.isfinite(points)):
            raise ValueError("[Airborne] output contains invalid values")
        if not np.isfinite(self.landed_volume_m3) or self.landed_volume_m3 < 0.0:
            raise ValueError("[Airborne] landed volume invalid")
        object.__setattr__(self, "remaining_parcels", tuple(self.remaining_parcels))
        object.__setattr__(self, "mobile_height_m", _ro(height))
        object.__setattr__(self, "mobile_momentum_m2_s", _ro(momentum))
        object.__setattr__(self, "landing_points_world_m", _ro(points))


class AirborneParcelModel:
    def __init__(self, config: AirborneParcelConfig | None = None) -> None:
        self.config = config or AirborneParcelConfig()

    def create_from_bucket_release(
        self,
        volume_m3: float,
        material: MaterialScenario,
        tool_state: ToolState,
        descriptor: ToolDescriptor,
        *,
        timestamp_s: float,
        source: str,
        id_prefix: str,
        free_surface_normal_bucket_frame: np.ndarray | None = None,
        payload_center_of_mass_bucket_frame_m: np.ndarray | None = None,
    ) -> tuple[MaterialParcel, ...]:
        volume = float(volume_m3)
        if not np.isfinite(volume) or volume < 0.0:
            raise ValueError("[Airborne] release volume must be finite/non-negative")
        if volume == 0.0:
            return ()
        count = int(np.ceil(volume / self.config.target_volume_m3))
        count = min(max(count, self.config.minimum_count), self.config.maximum_count)
        base = volume / count
        geometry = descriptor.bucket_geometry
        if geometry is None:
            raise ValueError("[Airborne] authoritative bucket geometry is required")
        local_edge = np.asarray(geometry.lip_local, dtype=np.float64)
        if free_surface_normal_bucket_frame is not None:
            normal = np.asarray(free_surface_normal_bucket_frame, dtype=np.float64)
            if normal.shape != (3,) or not np.all(np.isfinite(normal)):
                raise ValueError("[Airborne] free-surface normal must be finite shape (3,)")
            top = np.asarray(geometry.top_edge_local, dtype=np.float64)
            # Spill releases from the lower of the two lateral mouth edges in
            # free-surface elevation.  This uses actual mouth geometry rather
            # than assuming the cutting edge is always the release edge.
            if float(np.mean(top @ normal)) < float(np.mean(local_edge @ normal)):
                local_edge = top
        left = local_edge[0]
        right = local_edge[-1]
        fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
        local_positions = left[None, :] + fractions[:, None] * (right - left)[None, :]
        homogeneous = np.column_stack([local_positions, np.ones(count)])
        world_positions = (tool_state.pose_world @ homogeneous.T).T[:, :3]
        rotation = tool_state.pose_world[:3, :3]
        rotation = self._closest_rotation(rotation)
        terrain_rotation = self._closest_rotation(tool_state.pose_terrain[:3, :3])
        world_from_terrain_rotation = rotation @ terrain_rotation.T
        relative_release_world = np.zeros(3, dtype=np.float64)
        if payload_center_of_mass_bucket_frame_m is not None:
            center_local = np.asarray(
                payload_center_of_mass_bucket_frame_m, dtype=np.float64
            )
            if center_local.shape != (3,) or not np.all(np.isfinite(center_local)):
                raise ValueError("[Airborne] payload COM must be finite shape (3,)")
            center_world = (
                tool_state.pose_world @ np.r_[center_local, 1.0]
            )[:3]
            release_center_world = np.mean(world_positions, axis=0)
            potential_drop_m = max(
                0.0, float(center_world[2] - release_center_world[2])
            )
            if potential_drop_m > 0.0:
                outward_world = rotation @ geometry.mouth_normal_local
                outward_world /= np.linalg.norm(outward_world)
                relative_release_world = outward_world * np.sqrt(
                    2.0 * self.config.gravity_m_s2 * potential_drop_m
                )
        parcels: list[MaterialParcel] = []
        for index, position in enumerate(world_positions):
            radius_terrain = (
                terrain_rotation @ local_positions[index]
            )
            point_velocity_terrain = (
                np.asarray(tool_state.linear_velocity, dtype=np.float64)
                + np.cross(
                    np.asarray(tool_state.angular_velocity, dtype=np.float64),
                    radius_terrain,
                )
            )
            # Released material initially shares the rigid mouth-point
            # velocity.  There is deliberately no fitted ejection impulse.
            velocity = (
                world_from_terrain_rotation @ point_velocity_terrain
                + relative_release_world
            )
            parcel_volume = base if index < count - 1 else volume - base * (count - 1)
            parcels.append(
                MaterialParcel(
                    parcel_id=f"{id_prefix}_{index:03d}",
                    volume_m3=parcel_volume,
                    assumed_bulk_density_kg_m3=material.assumed_bulk_density_kg_m3,
                    position_world_m=position,
                    velocity_world_m_s=velocity,
                    timestamp_s=timestamp_s,
                    source=source,
                )
            )
        if not np.isclose(sum(item.volume_m3 for item in parcels), volume, atol=1e-12):
            raise RuntimeError("[Airborne] parcel split lost release volume")
        return tuple(parcels)

    def advance(
        self,
        parcels: tuple[MaterialParcel, ...],
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
    ) -> AirborneAdvanceResult:
        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        mobile = np.asarray(mobile_height_m, dtype=np.float64).copy()
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64).copy()
        if mobile.shape != grid.shape or momentum.shape != grid.shape + (2,):
            raise ValueError("[Airborne] mobile/grid shape mismatch")
        if integrator.shape != grid.shape:
            raise ValueError("[Airborne] integrator/grid shape mismatch")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[Airborne] dt_s must be finite/positive")
        weights = integrator.vertex_weights_m2
        gravity = np.asarray([0.0, 0.0, -self.config.gravity_m_s2])
        remaining: list[MaterialParcel] = []
        landed = 0.0
        points: list[np.ndarray] = []
        for parcel in parcels:
            position = np.asarray(parcel.position_world_m) + np.asarray(parcel.velocity_world_m_s) * dt + 0.5 * gravity * dt * dt
            velocity = np.asarray(parcel.velocity_world_m_s) + gravity * dt
            terrain_point = grid.world_to_terrain(position)
            row_f, col_f = grid.terrain_to_grid(terrain_point)
            inside_xy = 0.0 <= row_f <= grid.ny - 1 and 0.0 <= col_f <= grid.nx - 1
            surface_height = self._bilinear(resting + mobile, row_f, col_f) if inside_xy else None
            if inside_xy and terrain_point[2] <= float(surface_height) + self.config.terrain_clearance_m:
                row = int(np.clip(round(row_f), 0, grid.ny - 1))
                col = int(np.clip(round(col_f), 0, grid.nx - 1))
                if weights[row, col] <= 0.0:
                    raise RuntimeError("[Airborne] landing control area is zero")
                mobile[row, col] += parcel.volume_m3 / weights[row, col]
                momentum[row, col] += (
                    parcel.volume_m3
                    * velocity[:2]
                    * self.config.landing_velocity_scale
                    / weights[row, col]
                )
                landed += parcel.volume_m3
                landing_terrain = np.asarray([terrain_point[0], terrain_point[1], float(surface_height)])
                points.append(grid.terrain_to_world(landing_terrain))
            else:
                remaining.append(
                    MaterialParcel(
                        parcel_id=parcel.parcel_id,
                        volume_m3=parcel.volume_m3,
                        assumed_bulk_density_kg_m3=parcel.assumed_bulk_density_kg_m3,
                        position_world_m=position,
                        velocity_world_m_s=velocity,
                        timestamp_s=parcel.timestamp_s + dt,
                        source=parcel.source,
                    )
                )
        return AirborneAdvanceResult(
            remaining_parcels=tuple(remaining),
            mobile_height_m=mobile,
            mobile_momentum_m2_s=momentum,
            landed_volume_m3=landed,
            landing_points_world_m=np.asarray(points, dtype=np.float64).reshape(-1, 3),
            parcel_count_before=len(parcels),
            parcel_count_after=len(remaining),
        )

    @staticmethod
    def _bilinear(field: np.ndarray, row: float, column: float) -> float:
        r0 = int(np.floor(row))
        c0 = int(np.floor(column))
        r1 = min(r0 + 1, field.shape[0] - 1)
        c1 = min(c0 + 1, field.shape[1] - 1)
        fr = row - r0
        fc = column - c0
        return float(
            (1 - fr) * (1 - fc) * field[r0, c0]
            + (1 - fr) * fc * field[r0, c1]
            + fr * (1 - fc) * field[r1, c0]
            + fr * fc * field[r1, c1]
        )

    @staticmethod
    def _closest_rotation(linear: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return rotation
