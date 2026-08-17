"""Coarse authoritative airborne parcels with ballistic terrain landing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import (
    AirborneParcelFootprint,
    MaterialParcel,
    MaterialScenario,
    TerrainVolumeIntegrator,
)
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
    landing_records: tuple["AirborneLandingRecord", ...] = ()

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
        object.__setattr__(self, "landing_records", tuple(self.landing_records))


@dataclass(frozen=True)
class AirborneLandingRecord:
    """Auditable conservative parcel-to-control-volume rasterization."""

    parcel_id: str
    represented_volume_m3: float
    impact_position_world_m: np.ndarray
    impact_velocity_world_m_s: np.ndarray
    source: str
    footprint: AirborneParcelFootprint
    recipient_flat_indices: np.ndarray
    recipient_control_areas_m2: np.ndarray
    delivered_volumes_m3: np.ndarray
    overlap_areas_m2: np.ndarray
    terrain_normal_world: np.ndarray
    normal_impact_impulse_on_terrain_ns: np.ndarray
    normal_kinetic_energy_dissipated_j: float

    def __post_init__(self) -> None:
        indices = np.asarray(self.recipient_flat_indices, dtype=np.int32).reshape(-1)
        control = np.asarray(self.recipient_control_areas_m2, dtype=np.float64).reshape(-1)
        delivered = np.asarray(self.delivered_volumes_m3, dtype=np.float64).reshape(-1)
        overlap = np.asarray(self.overlap_areas_m2, dtype=np.float64).reshape(-1)
        if not indices.size or not (
            indices.shape == control.shape == delivered.shape == overlap.shape
        ):
            raise ValueError("[Airborne] landing recipient arrays are invalid")
        if np.any(control <= 0.0) or np.any(delivered < 0.0) or np.any(overlap < 0.0):
            raise ValueError("[Airborne] landing recipient values are invalid")
        if not np.isclose(
            np.sum(delivered), self.represented_volume_m3, atol=1.0e-12, rtol=1.0e-12
        ):
            raise ValueError("[Airborne] landing rasterization lost parcel volume")
        object.__setattr__(self, "impact_position_world_m", _ro(self.impact_position_world_m))
        object.__setattr__(self, "impact_velocity_world_m_s", _ro(self.impact_velocity_world_m_s))
        indices = np.ascontiguousarray(indices.copy()); indices.setflags(write=False)
        object.__setattr__(self, "recipient_flat_indices", indices)
        object.__setattr__(self, "recipient_control_areas_m2", _ro(control))
        object.__setattr__(self, "delivered_volumes_m3", _ro(delivered))
        object.__setattr__(self, "overlap_areas_m2", _ro(overlap))
        normal = np.asarray(self.terrain_normal_world, dtype=np.float64)
        impulse = np.asarray(
            self.normal_impact_impulse_on_terrain_ns, dtype=np.float64
        )
        if normal.shape != (3,) or impulse.shape != (3,):
            raise ValueError("[Airborne] impact normal/impulse must have shape (3,)")
        if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(impulse)):
            raise ValueError("[Airborne] impact normal/impulse must be finite")
        if not np.isfinite(self.normal_kinetic_energy_dissipated_j) or self.normal_kinetic_energy_dissipated_j < 0.0:
            raise ValueError("[Airborne] invalid dissipated normal kinetic energy")
        object.__setattr__(self, "terrain_normal_world", _ro(normal))
        object.__setattr__(self, "normal_impact_impulse_on_terrain_ns", _ro(impulse))

    @property
    def recipient_cell_count(self) -> int:
        return int(self.recipient_flat_indices.size)

    @property
    def footprint_area_m2(self) -> float:
        return float(np.sum(self.overlap_areas_m2))


class AirborneParcelModel:
    def __init__(
        self,
        config: AirborneParcelConfig | None = None,
        *,
        descriptor: ToolDescriptor | None = None,
        material: MaterialScenario | None = None,
    ) -> None:
        self.config = config or AirborneParcelConfig()
        self.descriptor = descriptor
        self.material = material

    @staticmethod
    def footprint_from_geometry(
        volume_m3: float,
        descriptor: ToolDescriptor | None,
        lateral_axis_world_m: np.ndarray,
        material: MaterialScenario | None = None,
    ) -> AirborneParcelFootprint:
        """Derive compact landing support from volume and bucket geometry.

        With authoritative geometry, mean interior depth is capacity/mouth
        area.  The parcel's finite support area is V/depth, while its aspect
        ratio follows the actual bucket mouth.  This is explicitly an
        engineering closure, not a calibrated granular dispersion law.
        """

        volume = float(volume_m3)
        geometry = None if descriptor is None else descriptor.bucket_geometry
        if geometry is None:
            area = volume ** (2.0 / 3.0)
            aspect = 1.0
        else:
            mean_depth_m = descriptor.effective_capacity_m3 / geometry.mouth_area_m2
            area = volume / mean_depth_m
            lateral_m = geometry.cutting_edge_length_m
            longitudinal_m = geometry.mouth_area_m2 / lateral_m
            aspect = lateral_m / longitudinal_m
        profile = "UNIFORM_RECTANGLE"
        lateral_extent = np.sqrt(area * aspect)
        longitudinal_extent = np.sqrt(area / aspect)
        if material is not None:
            # The aggregate parcel does not retain a resolved internal fill
            # field after release. Use the unique zero-parameter compact cone
            # whose volume and boundary slope equal this parcel volume and the
            # already-frozen material stop angle. The state still enters
            # Mobile; this is an impact source profile, not forced settling.
            stop_tangent = float(np.tan(np.deg2rad(material.stop_angle_deg)))
            equivalent_radius = np.cbrt(
                3.0 * volume / (np.pi * stop_tangent)
            )
            lateral_extent = 2.0 * equivalent_radius * np.sqrt(aspect)
            longitudinal_extent = 2.0 * equivalent_radius / np.sqrt(aspect)
            profile = "ELLIPTIC_CONE_STOP_ANGLE_CLOSURE"
        return AirborneParcelFootprint(
            lateral_axis_world_m=np.asarray(lateral_axis_world_m, dtype=np.float64),
            lateral_extent_m=float(lateral_extent),
            longitudinal_extent_m=float(longitudinal_extent),
            provenance="CONSERVATION_BASED_ENGINEERING_CLOSURE",
            mass_profile=profile,
        )

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
        lateral_axis_world = rotation @ (right - left)
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
                    footprint=self.footprint_from_geometry(
                        parcel_volume, descriptor, lateral_axis_world, material
                    ),
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
        landing_records: list[AirborneLandingRecord] = []
        resolved = self.resolve_missing_footprints(parcels)
        for parcel in resolved:
            position = np.asarray(parcel.position_world_m) + np.asarray(parcel.velocity_world_m_s) * dt + 0.5 * gravity * dt * dt
            velocity = np.asarray(parcel.velocity_world_m_s) + gravity * dt
            terrain_point = grid.world_to_terrain(position)
            row_f, col_f = grid.terrain_to_grid(terrain_point)
            inside_xy = 0.0 <= row_f <= grid.ny - 1 and 0.0 <= col_f <= grid.nx - 1
            surface_height = self._bilinear(resting + mobile, row_f, col_f) if inside_xy else None
            if inside_xy and terrain_point[2] <= float(surface_height) + self.config.terrain_clearance_m:
                record = self.rasterize_landing(
                    parcel, position, velocity, grid, integrator
                )
                rows, cols = np.divmod(record.recipient_flat_indices, grid.nx)
                delta_h = record.delivered_volumes_m3 / record.recipient_control_areas_m2
                mobile[rows, cols] += delta_h
                momentum[rows, cols] += (
                    delta_h[:, None]
                    * velocity[:2][None, :]
                    * self.config.landing_velocity_scale
                )
                landed += parcel.volume_m3
                landing_records.append(record)
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
                        footprint=parcel.footprint,
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
            landing_records=tuple(landing_records),
        )

    def resolve_missing_footprints(
        self, parcels: tuple[MaterialParcel, ...]
    ) -> tuple[MaterialParcel, ...]:
        """Backfill legacy checkpoint parcels without changing their dynamics."""

        if not parcels or all(parcel.footprint is not None for parcel in parcels):
            return tuple(parcels)
        positions = np.asarray([parcel.position_world_m for parcel in parcels])
        if len(positions) >= 2:
            _, _, vt = np.linalg.svd(positions - np.mean(positions, axis=0), full_matrices=False)
            lateral = vt[0]
        else:
            velocity = np.asarray(parcels[0].velocity_world_m_s)
            lateral = np.asarray([-velocity[1], velocity[0], 0.0])
            if np.linalg.norm(lateral) <= 1.0e-12:
                lateral = np.asarray([1.0, 0.0, 0.0])
        return tuple(
            parcel
            if parcel.footprint is not None
            else MaterialParcel(
                parcel.parcel_id,
                parcel.volume_m3,
                parcel.assumed_bulk_density_kg_m3,
                parcel.position_world_m,
                parcel.velocity_world_m_s,
                parcel.timestamp_s,
                parcel.source,
                self.footprint_from_geometry(
                    parcel.volume_m3, self.descriptor, lateral, self.material
                ),
            )
            for parcel in parcels
        )

    @staticmethod
    def rasterize_landing(
        parcel: MaterialParcel,
        impact_position_world_m: np.ndarray,
        impact_velocity_world_m_s: np.ndarray,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> AirborneLandingRecord:
        footprint = parcel.footprint
        if footprint is None:
            raise RuntimeError("[Airborne] parcel footprint must be resolved before landing")
        center_world = np.asarray(impact_position_world_m, dtype=np.float64)
        center = grid.world_to_terrain(center_world)[:2]
        axis_endpoint = grid.world_to_terrain(
            center_world + footprint.lateral_axis_world_m
        )[:2]
        lateral = axis_endpoint - center
        norm = float(np.linalg.norm(lateral))
        if norm <= 1.0e-12:
            lateral = np.asarray([1.0, 0.0])
        else:
            lateral /= norm
        longitudinal = np.asarray([-lateral[1], lateral[0]])
        half_lateral = 0.5 * footprint.lateral_extent_m * lateral
        half_longitudinal = 0.5 * footprint.longitudinal_extent_m * longitudinal
        polygon = np.asarray(
            [
                center - half_lateral - half_longitudinal,
                center + half_lateral - half_longitudinal,
                center + half_lateral + half_longitudinal,
                center - half_lateral + half_longitudinal,
            ],
            dtype=np.float64,
        )
        x_min, y_min = np.min(polygon, axis=0)
        x_max, y_max = np.max(polygon, axis=0)
        c0 = max(0, int(np.floor((x_min - grid.origin_x) / grid.dx - 0.5)))
        c1 = min(grid.nx - 1, int(np.ceil((x_max - grid.origin_x) / grid.dx + 0.5)))
        r0 = max(0, int(np.floor((y_min - grid.origin_y) / grid.dy - 0.5)))
        r1 = min(grid.ny - 1, int(np.ceil((y_max - grid.origin_y) / grid.dy + 0.5)))
        indices: list[int] = []
        overlaps: list[float] = []
        profile_weights: list[float] = []
        for row in range(r0, r1 + 1):
            y = grid.origin_y + row * grid.dy
            lower_y = max(grid.origin_y, y - 0.5 * grid.dy)
            upper_y = min(grid.origin_y + (grid.ny - 1) * grid.dy, y + 0.5 * grid.dy)
            for col in range(c0, c1 + 1):
                if grid.valid_mask is not None and not grid.valid_mask[row, col]:
                    continue
                x = grid.origin_x + col * grid.dx
                lower_x = max(grid.origin_x, x - 0.5 * grid.dx)
                upper_x = min(grid.origin_x + (grid.nx - 1) * grid.dx, x + 0.5 * grid.dx)
                clipped = AirborneParcelModel._clip_rectangle(
                    polygon, lower_x, upper_x, lower_y, upper_y
                )
                area = AirborneParcelModel._polygon_area(clipped)
                if area > 1.0e-15:
                    profile_density = 1.0
                    if footprint.mass_profile == "ELLIPTIC_CONE_STOP_ANGLE_CLOSURE":
                        centroid = AirborneParcelModel._polygon_centroid(clipped)
                        offset = centroid - center
                        u = float(np.dot(offset, lateral)) / (0.5 * footprint.lateral_extent_m)
                        v = float(np.dot(offset, longitudinal)) / (0.5 * footprint.longitudinal_extent_m)
                        profile_density = max(0.0, 1.0 - np.hypot(u, v))
                    weighted_area = area * profile_density
                    if weighted_area > 1.0e-15:
                        indices.append(row * grid.nx + col)
                        overlaps.append(area)
                        profile_weights.append(weighted_area)
        if not indices:
            raise RuntimeError("[Airborne] finite footprint has no terrain recipients")
        flat = np.asarray(indices, dtype=np.int32)
        overlap = np.asarray(overlaps, dtype=np.float64)
        mass_weight = np.asarray(profile_weights, dtype=np.float64)
        delivered = parcel.volume_m3 * mass_weight / np.sum(mass_weight)
        control = integrator.vertex_weights_m2.reshape(-1)[flat]
        terrain_normal_world = np.asarray(
            grid.terrain_to_world_matrix[:3, 2], dtype=np.float64
        )
        terrain_normal_world /= np.linalg.norm(terrain_normal_world)
        normal_velocity = min(
            0.0,
            float(np.dot(impact_velocity_world_m_s, terrain_normal_world)),
        )
        represented_mass = (
            parcel.volume_m3 * parcel.assumed_bulk_density_kg_m3
        )
        return AirborneLandingRecord(
            parcel_id=parcel.parcel_id,
            represented_volume_m3=parcel.volume_m3,
            impact_position_world_m=center_world,
            impact_velocity_world_m_s=impact_velocity_world_m_s,
            source=parcel.source,
            footprint=footprint,
            recipient_flat_indices=flat,
            recipient_control_areas_m2=control,
            delivered_volumes_m3=delivered,
            overlap_areas_m2=overlap,
            terrain_normal_world=terrain_normal_world,
            normal_impact_impulse_on_terrain_ns=(
                represented_mass * normal_velocity * terrain_normal_world
            ),
            normal_kinetic_energy_dissipated_j=(
                0.5 * represented_mass * normal_velocity * normal_velocity
            ),
        )

    @staticmethod
    def _clip_rectangle(
        polygon: np.ndarray,
        lower_x: float,
        upper_x: float,
        lower_y: float,
        upper_y: float,
    ) -> np.ndarray:
        result = np.asarray(polygon, dtype=np.float64)
        for axis, bound, keep_greater in (
            (0, lower_x, True), (0, upper_x, False),
            (1, lower_y, True), (1, upper_y, False),
        ):
            if not len(result):
                break
            output: list[np.ndarray] = []
            previous = result[-1]
            previous_inside = previous[axis] >= bound if keep_greater else previous[axis] <= bound
            for current in result:
                current_inside = current[axis] >= bound if keep_greater else current[axis] <= bound
                if current_inside != previous_inside:
                    delta = current - previous
                    fraction = 0.0 if abs(delta[axis]) <= 1.0e-15 else (bound - previous[axis]) / delta[axis]
                    output.append(previous + fraction * delta)
                if current_inside:
                    output.append(current)
                previous = current
                previous_inside = current_inside
            result = np.asarray(output, dtype=np.float64).reshape(-1, 2)
        return result

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        if len(polygon) < 3:
            return 0.0
        return float(
            0.5 * abs(
                np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
                - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
            )
        )

    @staticmethod
    def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
        """Area centroid of a clipped convex polygon."""

        if len(polygon) < 3:
            return np.mean(polygon, axis=0)
        x = polygon[:, 0]
        y = polygon[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        twice_area = float(np.sum(cross))
        if abs(twice_area) <= 1.0e-18:
            return np.mean(polygon, axis=0)
        return np.asarray(
            [
                np.sum((x + np.roll(x, -1)) * cross) / (3.0 * twice_area),
                np.sum((y + np.roll(y, -1)) * cross) / (3.0 * twice_area),
            ],
            dtype=np.float64,
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
