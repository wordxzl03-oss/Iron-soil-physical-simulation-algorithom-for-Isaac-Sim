"""Conservative 2.5-D flux through an authoritative bucket mouth segment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import (
    BucketFillPhase,
    BucketInternalFillModel,
    PayloadState,
    TerrainVolumeIntegrator,
)
from ..terrain.terrain_grid import TerrainGrid
from ..tools import GeometrySource, ToolDescriptor, ToolState


_EPS = 1.0e-12


def _ro(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BucketIntakeConfig:
    dry_tolerance_m: float = 1.0e-10
    minimum_inflow_speed_m_s: float = 0.0
    require_non_legacy_geometry: bool = True

    def __post_init__(self) -> None:
        values = np.asarray([self.dry_tolerance_m, self.minimum_inflow_speed_m_s])
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("[BucketIntake] tolerances must be finite/non-negative")


@dataclass(frozen=True)
class BucketIntakeResult:
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    payload: PayloadState
    admitted_volume_field_m3: np.ndarray
    bucket_inflow_volume_m3: float
    mouth_intersection_area_m2: float
    mouth_overlap_length_m: float
    mean_positive_relative_speed_m_s: float
    capacity_limited: bool
    candidate_flux_volume_m3: float
    intake_centroid_terrain_m: np.ndarray
    geometry_source: str
    legacy_sampling_band_used: bool = False

    def __post_init__(self) -> None:
        height = np.asarray(self.mobile_height_m, dtype=np.float64)
        momentum = np.asarray(self.mobile_momentum_m2_s, dtype=np.float64)
        admitted = np.asarray(self.admitted_volume_field_m3, dtype=np.float64)
        centroid = np.asarray(self.intake_centroid_terrain_m, dtype=np.float64)
        if height.ndim != 2 or momentum.shape != height.shape + (2,) or admitted.shape != height.shape:
            raise ValueError("[BucketIntake] output field shapes invalid")
        if not np.all(np.isfinite(height)) or np.any(height < -_EPS):
            raise ValueError("[BucketIntake] mobile height invalid")
        if not np.all(np.isfinite(momentum)) or not np.all(np.isfinite(admitted)) or np.any(admitted < -_EPS):
            raise ValueError("[BucketIntake] momentum/admitted field invalid")
        if centroid.shape != (3,) or not np.all(np.isfinite(centroid)):
            raise ValueError("[BucketIntake] intake centroid must be finite shape (3,)")
        for name in (
            "bucket_inflow_volume_m3",
            "mouth_intersection_area_m2",
            "mouth_overlap_length_m",
            "mean_positive_relative_speed_m_s",
            "candidate_flux_volume_m3",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"[BucketIntake] {name} invalid")
        if self.payload.volume_m3 > self.payload.capacity_m3 + _EPS:
            raise ValueError("[BucketIntake] output payload exceeds capacity")
        if not isinstance(self.geometry_source, str) or not self.geometry_source:
            raise ValueError("[BucketIntake] geometry_source must be non-empty")
        object.__setattr__(self, "mobile_height_m", _ro(np.maximum(height, 0.0)))
        object.__setattr__(self, "mobile_momentum_m2_s", _ro(momentum))
        object.__setattr__(self, "admitted_volume_field_m3", _ro(np.maximum(admitted, 0.0)))
        object.__setattr__(self, "intake_centroid_terrain_m", _ro(centroid))


class BucketIntakeModel:
    """Integrate available Mobile Layer flux across the projected mouth lip.

    The Mobile Layer has horizontal velocity and column thickness only.  The
    implemented control surface is consequently the projected lip segment,
    with the mobile-column/mouth-aperture vertical overlap supplying effective
    thickness.  There is no capture coefficient or sampling band.
    """

    def __init__(self, config: BucketIntakeConfig | None = None) -> None:
        self.config = config or BucketIntakeConfig()
        self.internal_fill_model = BucketInternalFillModel()

    def apply(
        self,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        payload: PayloadState,
        tool_state: ToolState,
        descriptor: ToolDescriptor,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        terrain_surface_m: np.ndarray | None = None,
    ) -> BucketIntakeResult:
        height = np.asarray(mobile_height_m, dtype=np.float64).copy()
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64).copy()
        if height.shape != grid.shape or momentum.shape != grid.shape + (2,):
            raise ValueError("[BucketIntake] state/grid shape mismatch")
        if not np.all(np.isfinite(height)) or np.any(height < 0.0) or not np.all(np.isfinite(momentum)):
            raise ValueError("[BucketIntake] input state invalid")
        if integrator.shape != grid.shape:
            raise ValueError("[BucketIntake] integrator/grid shape mismatch")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[BucketIntake] dt_s must be finite/positive")
        geometry = descriptor.bucket_geometry
        if geometry is None:
            raise ValueError("[BucketIntake] authoritative BucketGeometryDescriptor is required")
        if self.config.require_non_legacy_geometry and geometry.legacy_fallback_used:
            raise ValueError(
                "[BucketIntake] LEGACY_FALLBACK geometry is explicitly rejected; "
                "load markers, an explicit profile, or validated mesh geometry"
            )
        expected_capacity = geometry.effective_capacity_m3
        if not np.isclose(expected_capacity, payload.capacity_m3, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                "[BucketIntake] payload capacity must equal the same interior's "
                f"effective capacity; payload={payload.capacity_m3}, geometry={expected_capacity}"
            )
        surface = (
            np.zeros(grid.shape, dtype=np.float64)
            if terrain_surface_m is None
            else np.asarray(grid.validate_heightmap(terrain_surface_m), dtype=np.float64)
        )
        pose = np.asarray(tool_state.pose_terrain, dtype=np.float64)
        lip = geometry.transform_points(pose, geometry.lip_local)
        top_edge = geometry.transform_points(pose, geometry.top_edge_local)
        lip_start_xy = lip[0, :2]
        lip_end_xy = lip[-1, :2]
        projected_lip_length = float(np.linalg.norm(lip_end_xy - lip_start_xy))
        mouth_centroid = geometry.transform_points(
            pose, geometry.mouth_centroid_local[None, :]
        )[0]
        if projected_lip_length <= _EPS:
            return self._zero_result(height, momentum, payload, mouth_centroid, geometry.geometry_source.value)

        linear = pose[:3, :3]
        mouth_normal = np.linalg.inv(linear).T @ geometry.mouth_normal_local
        mouth_normal /= np.linalg.norm(mouth_normal)
        normal_xy = mouth_normal[:2]
        if np.linalg.norm(normal_xy) <= _EPS:
            return self._zero_result(height, momentum, payload, mouth_centroid, geometry.geometry_source.value)

        material_velocity = np.divide(
            momentum,
            height[..., None],
            out=np.zeros_like(momentum),
            where=height[..., None] > self.config.dry_tolerance_m,
        )
        weights = np.asarray(integrator.vertex_weights_m2, dtype=np.float64)
        candidate = np.zeros(grid.shape, dtype=np.float64)
        representative = np.zeros(grid.shape + (3,), dtype=np.float64)
        aperture_area = 0.0
        overlap_length = 0.0
        positive_speed_weight = 0.0
        positive_speed_denominator = 0.0

        x_min, y_min = np.minimum(lip_start_xy, lip_end_xy)
        x_max, y_max = np.maximum(lip_start_xy, lip_end_xy)
        column_low = max(0, int(np.floor((x_min - grid.origin_x) / grid.dx)) - 1)
        column_high = min(grid.nx - 1, int(np.ceil((x_max - grid.origin_x) / grid.dx)) + 1)
        row_low = max(0, int(np.floor((y_min - grid.origin_y) / grid.dy)) - 1)
        row_high = min(grid.ny - 1, int(np.ceil((y_max - grid.origin_y) / grid.dy)) + 1)
        for row in range(row_low, row_high + 1):
            y0, y1 = self._control_bounds(row, grid.ny, grid.origin_y, grid.dy)
            for column in range(column_low, column_high + 1):
                if weights[row, column] <= _EPS:
                    continue
                if grid.valid_mask is not None and not grid.valid_mask[row, column]:
                    continue
                x0, x1 = self._control_bounds(column, grid.nx, grid.origin_x, grid.dx)
                # Vertex-centred control rectangles share their boundaries.  A
                # lip exactly collinear with an internal boundary would
                # otherwise be integrated twice.  Give that boundary to the
                # control volume on its positive-coordinate side.
                delta_xy = lip_end_xy - lip_start_xy
                if (
                    abs(delta_xy[0]) <= _EPS
                    and column < grid.nx - 1
                    and abs(lip_start_xy[0] - x1) <= _EPS
                ):
                    continue
                if (
                    abs(delta_xy[1]) <= _EPS
                    and row < grid.ny - 1
                    and abs(lip_start_xy[1] - y1) <= _EPS
                ):
                    continue
                clipped = self._clip_segment_to_rectangle(
                    lip_start_xy,
                    lip_end_xy,
                    x0,
                    x1,
                    y0,
                    y1,
                )
                if clipped is None:
                    continue
                length, midpoint_xy, midpoint_parameter = clipped
                if length <= _EPS:
                    continue
                cut_z = (1.0 - midpoint_parameter) * lip[0, 2] + midpoint_parameter * lip[-1, 2]
                top_z = (
                    (1.0 - midpoint_parameter) * top_edge[0, 2]
                    + midpoint_parameter * top_edge[-1, 2]
                )
                aperture_low, aperture_high = sorted((float(cut_z), float(top_z)))
                material_low = float(surface[row, column])
                material_high = material_low + float(height[row, column])
                overlap_height = max(
                    0.0,
                    min(material_high, aperture_high) - max(material_low, aperture_low),
                )
                if overlap_height <= self.config.dry_tolerance_m:
                    continue
                point = np.asarray(
                    [
                        midpoint_xy[0],
                        midpoint_xy[1],
                        0.5 * (max(material_low, aperture_low) + min(material_high, aperture_high)),
                    ]
                )
                radius = point - pose[:3, 3]
                mouth_velocity = tool_state.linear_velocity + np.cross(tool_state.angular_velocity, radius)
                relative_speed = float(
                    np.dot(material_velocity[row, column] - mouth_velocity[:2], normal_xy)
                )
                positive_speed = (
                    relative_speed
                    if relative_speed > self.config.minimum_inflow_speed_m_s
                    else 0.0
                )
                local_area = overlap_height * length
                aperture_area += local_area
                overlap_length += length
                if positive_speed <= 0.0:
                    continue
                requested = overlap_height * positive_speed * length * dt
                candidate[row, column] = min(requested, height[row, column] * weights[row, column])
                representative[row, column] = point
                positive_speed_weight += positive_speed * local_area
                positive_speed_denominator += local_area

        candidate_total = float(np.sum(candidate, dtype=np.float64))
        admitted_total = min(candidate_total, payload.remaining_capacity_m3)
        capacity_limited = candidate_total > payload.remaining_capacity_m3 + _EPS
        scale = admitted_total / candidate_total if candidate_total > 0.0 else 0.0
        admitted = candidate * scale
        removal_height = np.divide(admitted, weights, out=np.zeros_like(admitted), where=weights > _EPS)
        old_height = height.copy()
        height = np.maximum(height - removal_height, 0.0)
        fraction_remaining = np.divide(
            height,
            old_height,
            out=np.zeros_like(height),
            where=old_height > self.config.dry_tolerance_m,
        )
        momentum *= fraction_remaining[..., None]
        momentum[height <= self.config.dry_tolerance_m] = 0.0

        if admitted_total > 0.0:
            centroid_terrain = np.sum(representative * admitted[..., None], axis=(0, 1)) / admitted_total
            tool_from_terrain = np.linalg.inv(pose)
            centroid_h = tool_from_terrain @ np.r_[centroid_terrain, 1.0]
            admitted_centroid_local = centroid_h[:3] / centroid_h[3]
            old_volume = payload.volume_m3
            new_volume = old_volume + admitted_total
            center = (
                payload.center_of_mass_bucket_frame_m * old_volume
                + admitted_centroid_local * admitted_total
            ) / new_volume
        else:
            centroid_terrain = mouth_centroid
            new_volume = payload.volume_m3
            center = payload.center_of_mass_bucket_frame_m
        # The geometric fill is authoritative for COM/inertia.  The admitted
        # flux centroid remains telemetry; it is not allowed to leave the
        # payload COM inconsistent with its occupied interior.
        if payload.free_surface is not None:
            surface_normal = payload.free_surface.normal_bucket_frame
        else:
            rotation = self._closest_rotation(pose[:3, :3])
            local_up = rotation.T @ np.asarray([0.0, 0.0, 1.0])
            surface_normal = np.asarray([0.0, local_up[1], local_up[2]])
        next_payload = self.internal_fill_model.solve_payload(
            payload,
            descriptor,
            surface_normal,
            volume_m3=new_volume,
            phase=(
                BucketFillPhase.RELAXING
                if admitted_total > 0.0
                else BucketFillPhase.STATIC
            ),
        )
        return BucketIntakeResult(
            mobile_height_m=height,
            mobile_momentum_m2_s=momentum,
            payload=next_payload,
            admitted_volume_field_m3=admitted,
            bucket_inflow_volume_m3=admitted_total,
            mouth_intersection_area_m2=float(aperture_area),
            mouth_overlap_length_m=float(overlap_length),
            mean_positive_relative_speed_m_s=(
                positive_speed_weight / positive_speed_denominator
                if positive_speed_denominator > 0.0
                else 0.0
            ),
            capacity_limited=capacity_limited,
            candidate_flux_volume_m3=candidate_total,
            intake_centroid_terrain_m=np.asarray(centroid_terrain),
            geometry_source=geometry.geometry_source.value,
            legacy_sampling_band_used=False,
        )

    @staticmethod
    def _control_bounds(index: int, count: int, origin: float, spacing: float) -> tuple[float, float]:
        lower = origin if index == 0 else origin + (index - 0.5) * spacing
        upper = origin + (count - 1) * spacing if index == count - 1 else origin + (index + 0.5) * spacing
        return float(lower), float(upper)

    @staticmethod
    def _clip_segment_to_rectangle(
        start: np.ndarray,
        end: np.ndarray,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> tuple[float, np.ndarray, float] | None:
        delta = np.asarray(end) - np.asarray(start)
        t_low, t_high = 0.0, 1.0
        for axis, lower, upper in ((0, x_min, x_max), (1, y_min, y_max)):
            if abs(delta[axis]) <= _EPS:
                if start[axis] < lower - _EPS or start[axis] > upper + _EPS:
                    return None
                continue
            first = (lower - start[axis]) / delta[axis]
            second = (upper - start[axis]) / delta[axis]
            enter, leave = min(first, second), max(first, second)
            t_low = max(t_low, enter)
            t_high = min(t_high, leave)
            if t_high <= t_low + _EPS:
                return None
        projected_length = float(np.linalg.norm(delta))
        length = (t_high - t_low) * projected_length
        parameter = 0.5 * (t_low + t_high)
        midpoint = np.asarray(start) + parameter * delta
        return float(length), midpoint, float(parameter)

    @staticmethod
    def _zero_result(
        height: np.ndarray,
        momentum: np.ndarray,
        payload: PayloadState,
        centroid: np.ndarray,
        geometry_source: str,
    ) -> BucketIntakeResult:
        return BucketIntakeResult(
            mobile_height_m=height,
            mobile_momentum_m2_s=momentum,
            payload=payload,
            admitted_volume_field_m3=np.zeros_like(height),
            bucket_inflow_volume_m3=0.0,
            mouth_intersection_area_m2=0.0,
            mouth_overlap_length_m=0.0,
            mean_positive_relative_speed_m_s=0.0,
            capacity_limited=False,
            candidate_flux_volume_m3=0.0,
            intake_centroid_terrain_m=np.asarray(centroid),
            geometry_source=geometry_source,
            legacy_sampling_band_used=False,
        )

    @staticmethod
    def _closest_rotation(linear: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(np.asarray(linear, dtype=np.float64))
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return rotation
