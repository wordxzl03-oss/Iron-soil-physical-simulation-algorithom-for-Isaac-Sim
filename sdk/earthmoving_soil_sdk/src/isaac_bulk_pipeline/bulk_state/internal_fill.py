"""Geometry-resolved bucket fill, free-surface and payload inertia solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..tools import ToolDescriptor, ToolState
from .models import (
    BucketFillPhase,
    BucketInternalFillState,
    PayloadFreeSurface,
    PayloadState,
)


_EPS = 1.0e-12


class BucketInternalFillModel:
    """Solve an extruded interior occupied by material below a planar surface.

    The free surface is ``n dot p = lambda`` and material occupies
    ``n dot p <= lambda``.  ``lambda`` is solved by monotone bisection so the
    clipped cross-section times the measured interior width equals the given
    payload volume.  No lip constraint is imposed by this volume solve.
    """

    def __init__(self, *, volume_tolerance_m3: float = 1.0e-10, max_iterations: int = 80) -> None:
        if not np.isfinite(volume_tolerance_m3) or volume_tolerance_m3 <= 0.0:
            raise ValueError("[BucketInternalFill] volume tolerance must be positive")
        if not isinstance(max_iterations, int) or max_iterations < 10:
            raise ValueError("[BucketInternalFill] max_iterations must be >= 10")
        self.volume_tolerance_m3 = float(volume_tolerance_m3)
        self.max_iterations = max_iterations

    def solve_payload(
        self,
        payload: PayloadState,
        descriptor: ToolDescriptor,
        free_surface_normal_bucket_frame: np.ndarray,
        *,
        volume_m3: float | None = None,
        phase: BucketFillPhase = BucketFillPhase.STATIC,
    ) -> PayloadState:
        geometry = descriptor.bucket_geometry
        if geometry is None:
            raise ValueError("[BucketInternalFill] authoritative bucket geometry is required")
        volume = payload.volume_m3 if volume_m3 is None else float(volume_m3)
        if not np.isfinite(volume) or volume < 0.0:
            raise ValueError("[BucketInternalFill] volume must be finite/non-negative")
        if volume > payload.capacity_m3 + self.volume_tolerance_m3:
            raise ValueError("[BucketInternalFill] volume exceeds payload capacity")
        if not np.isclose(
            payload.capacity_m3, geometry.effective_capacity_m3, atol=1e-9, rtol=0.0
        ):
            raise ValueError("[BucketInternalFill] geometry/payload capacities disagree")

        normal3 = np.asarray(free_surface_normal_bucket_frame, dtype=np.float64)
        if normal3.shape != (3,) or not np.all(np.isfinite(normal3)):
            raise ValueError("[BucketInternalFill] surface normal must be finite shape (3,)")
        normal_yz = normal3[1:3]
        norm_yz = float(np.linalg.norm(normal_yz))
        if norm_yz <= _EPS:
            raise ValueError("[BucketInternalFill] surface normal has no profile-plane component")
        normal_yz = normal_yz / norm_yz
        normal3 = np.asarray([0.0, normal_yz[0], normal_yz[1]], dtype=np.float64)

        profile = np.asarray(geometry.interior_profile_local[:, 1:3], dtype=np.float64)
        width = float(geometry.interior_width_m)
        target_area = volume / width
        projections = profile @ normal_yz
        lower = float(np.min(projections))
        upper = float(np.max(projections))
        full_area = abs(self._polygon_properties(profile)[0])
        if full_area <= _EPS:
            raise ValueError("[BucketInternalFill] interior profile has zero area")
        if target_area > full_area + self.volume_tolerance_m3 / width:
            raise ValueError("[BucketInternalFill] requested volume exceeds geometric interior")

        if target_area <= self.volume_tolerance_m3 / width:
            offset = lower
            clipped = np.empty((0, 2), dtype=np.float64)
        elif full_area - target_area <= self.volume_tolerance_m3 / width:
            offset = upper
            clipped = np.array(profile, copy=True)
        else:
            clipped = np.empty((0, 2), dtype=np.float64)
            for _ in range(self.max_iterations):
                offset = 0.5 * (lower + upper)
                clipped = self.clip_half_plane(profile, normal_yz, offset)
                area = abs(self._polygon_properties(clipped)[0]) if len(clipped) >= 3 else 0.0
                error_m3 = (area - target_area) * width
                if abs(error_m3) <= self.volume_tolerance_m3:
                    break
                if area < target_area:
                    lower = offset
                else:
                    upper = offset
            offset = 0.5 * (lower + upper)
            clipped = self.clip_half_plane(profile, normal_yz, offset)

        polygon3 = (
            np.column_stack((np.zeros(len(clipped)), clipped))
            if len(clipped)
            else np.empty((0, 3), dtype=np.float64)
        )
        if len(clipped) >= 3:
            area, centroid_yz, central_y2, central_z2, central_yz = self._polygon_properties(clipped)
            area = abs(area)
            com = np.asarray([0.0, centroid_yz[0], centroid_yz[1]], dtype=np.float64)
            second = np.asarray(
                [
                    [area * width**3 / 12.0, 0.0, 0.0],
                    [0.0, width * central_y2, width * central_yz],
                    [0.0, width * central_yz, width * central_z2],
                ],
                dtype=np.float64,
            )
        else:
            com = np.zeros(3, dtype=np.float64)
            second = np.zeros((3, 3), dtype=np.float64)
        density = payload.assumed_bulk_density_kg_m3
        inertia = density * (np.trace(second) * np.eye(3) - second)
        geometric_capacity = float(geometry.geometric_capacity_m3)
        rated_capacity = float(
            geometry.geometric_capacity_m3
            if geometry.rated_capacity_m3 is None
            else geometry.rated_capacity_m3
        )
        secondary = self.secondary_separation_direction(
            geometry.separation_plane_direction_local,
            geometry.cutting_edge_local,
            geometry.top_edge_local,
            volume / geometric_capacity,
        )
        surface = PayloadFreeSurface(normal_bucket_frame=normal3, offset_m=float(offset))
        state = BucketInternalFillState(
            volume_m3=volume,
            mass_kg=volume * density,
            geometric_fill_ratio=volume / geometric_capacity,
            rated_fill_ratio=volume / rated_capacity,
            free_surface=surface,
            occupied_polygon_bucket_frame_m=polygon3,
            center_of_mass_bucket_frame_m=com,
            second_moment_volume_m5=second,
            inertia_tensor_kg_m2=inertia,
            phase=phase,
            secondary_separation_direction_bucket_frame=secondary,
            geometric_capacity_m3=geometric_capacity,
            rated_capacity_m3=rated_capacity,
        )
        return PayloadState(
            volume_m3=volume,
            capacity_m3=payload.capacity_m3,
            assumed_bulk_density_kg_m3=density,
            center_of_mass_bucket_frame_m=com,
            free_surface=surface,
            internal_fill=state,
        )

    @staticmethod
    def secondary_separation_direction(
        primary_direction_local: np.ndarray,
        cutting_edge_local: np.ndarray,
        top_edge_local: np.ndarray,
        geometric_fill_ratio: float,
    ) -> np.ndarray:
        """Continuously move the separation plate from bucket bottom to mouth face."""

        primary = np.array(primary_direction_local, dtype=np.float64, copy=True)
        primary /= np.linalg.norm(primary)
        lip_center = np.mean(np.asarray(cutting_edge_local, dtype=np.float64), axis=0)
        top_center = np.mean(np.asarray(top_edge_local, dtype=np.float64), axis=0)
        mouth_face = np.array(top_center - lip_center, dtype=np.float64, copy=True)
        mouth_face /= np.linalg.norm(mouth_face)
        alpha = float(np.clip(geometric_fill_ratio, 0.0, 1.0))
        direction = (1.0 - alpha) * primary + alpha * mouth_face
        if np.linalg.norm(direction) <= _EPS:
            direction = mouth_face
        return direction / np.linalg.norm(direction)

    @staticmethod
    def apply_secondary_separation(
        tool_state: ToolState,
        payload: PayloadState,
        descriptor: ToolDescriptor,
    ) -> ToolState:
        if payload.internal_fill is None:
            return tool_state
        geometry = descriptor.bucket_geometry
        if geometry is None:
            return tool_state
        local = payload.internal_fill.secondary_separation_direction_bucket_frame
        terrain = geometry.transform_direction(tool_state.pose_terrain, local)
        diagnostics = dict(tool_state.diagnostics)
        diagnostics.update(
            {
                "bucket_geometric_fill_ratio": payload.internal_fill.geometric_fill_ratio,
                "fill_dependent_separation_active": True,
                "internal_fill_phase": payload.internal_fill.phase.value,
            }
        )
        return replace(
            tool_state,
            separation_plane_direction_terrain=terrain,
            separation_plane_source="BUCKET_INTERNAL_FILL_SECONDARY_PLATE",
            tool_plus_z_separation_fallback_used=False,
            diagnostics=diagnostics,
        )

    @staticmethod
    def clip_half_plane(polygon: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
        """Keep profile points satisfying ``dot(normal, point) <= offset``."""

        values = np.asarray(polygon, dtype=np.float64)
        if len(values) == 0:
            return np.empty((0, 2), dtype=np.float64)
        output: list[np.ndarray] = []
        previous = values[-1]
        previous_value = float(np.dot(normal, previous) - offset)
        previous_inside = previous_value <= _EPS
        for current in values:
            current_value = float(np.dot(normal, current) - offset)
            current_inside = current_value <= _EPS
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                fraction = previous_value / denominator if abs(denominator) > _EPS else 0.0
                output.append(previous + fraction * (current - previous))
            if current_inside:
                output.append(np.asarray(current, dtype=np.float64))
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        return np.asarray(output, dtype=np.float64).reshape(-1, 2)

    @staticmethod
    def _polygon_properties(polygon: np.ndarray) -> tuple[float, np.ndarray, float, float, float]:
        """Return area, centroid and centred y²/z²/yz area integrals."""

        points = np.asarray(polygon, dtype=np.float64)
        if len(points) < 3:
            return 0.0, np.zeros(2), 0.0, 0.0, 0.0
        y = points[:, 0]
        z = points[:, 1]
        yn = np.roll(y, -1)
        zn = np.roll(z, -1)
        cross = y * zn - yn * z
        signed_area = 0.5 * float(np.sum(cross))
        if abs(signed_area) <= _EPS:
            return 0.0, np.zeros(2), 0.0, 0.0, 0.0
        if signed_area < 0.0:
            return BucketInternalFillModel._polygon_properties(points[::-1])
        cy = float(np.sum((y + yn) * cross) / (6.0 * signed_area))
        cz = float(np.sum((z + zn) * cross) / (6.0 * signed_area))
        raw_y2 = float(np.sum((y * y + y * yn + yn * yn) * cross) / 12.0)
        raw_z2 = float(np.sum((z * z + z * zn + zn * zn) * cross) / 12.0)
        raw_yz = float(
            np.sum((2.0 * y * z + y * zn + yn * z + 2.0 * yn * zn) * cross) / 24.0
        )
        return (
            signed_area,
            np.asarray([cy, cz]),
            max(0.0, raw_y2 - signed_area * cy * cy),
            max(0.0, raw_z2 - signed_area * cz * cz),
            raw_yz - signed_area * cy * cz,
        )
