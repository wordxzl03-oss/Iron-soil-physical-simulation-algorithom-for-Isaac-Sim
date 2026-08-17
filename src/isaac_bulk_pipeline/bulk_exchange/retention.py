"""Bucket-interior free-surface retention and geometry-derived spill."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import (
    BucketFillPhase,
    BucketInternalFillModel,
    PayloadState,
)
from ..tools import ToolDescriptor, ToolState


def _ro(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BucketRetentionConfig:
    gravity_m_s2: float = 9.81
    orientation_samples: int = 81
    use_effective_gravity: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0.0:
            raise ValueError("[BucketRetention] gravity must be finite/positive")
        if not isinstance(self.orientation_samples, int) or self.orientation_samples < 3:
            raise ValueError("[BucketRetention] orientation_samples must be >= 3")


@dataclass(frozen=True)
class BucketRetentionResult:
    payload: PayloadState
    retainable_volume_m3: float
    spill_volume_m3: float
    free_surface_normal_bucket_frame: np.ndarray
    effective_gravity_bucket_frame_m_s2: np.ndarray
    retained_cross_section_fraction: float
    free_surface_offset_m: float

    def __post_init__(self) -> None:
        for name in (
            "retainable_volume_m3",
            "spill_volume_m3",
            "retained_cross_section_fraction",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"[BucketRetention] {name} invalid")
        if self.retained_cross_section_fraction > 1.0 + 1e-9:
            raise ValueError("[BucketRetention] fraction exceeds one")
        normal = np.asarray(self.free_surface_normal_bucket_frame, dtype=np.float64)
        gravity = np.asarray(self.effective_gravity_bucket_frame_m_s2, dtype=np.float64)
        if normal.shape != (3,) or gravity.shape != (3,) or not np.all(np.isfinite(np.r_[normal, gravity])):
            raise ValueError("[BucketRetention] normal/gravity vectors invalid")
        if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6):
            raise ValueError("[BucketRetention] free-surface normal must be unit")
        object.__setattr__(self, "free_surface_normal_bucket_frame", _ro(normal))
        object.__setattr__(self, "effective_gravity_bucket_frame_m_s2", _ro(gravity))


class BucketRetentionSpillModel:
    """Resolve stable retention against the complete bucket-mouth boundary.

    A non-spilling payload gets the exact volume-solved free-surface offset.
    Only the limiting surface used to compute spill capacity touches the
    lowest mouth boundary under the selected effective-gravity orientation.
    """

    def __init__(self, config: BucketRetentionConfig | None = None) -> None:
        self.config = config or BucketRetentionConfig()
        self.internal_fill_model = BucketInternalFillModel()

    def evaluate(
        self,
        payload: PayloadState,
        descriptor: ToolDescriptor,
        tool_state: ToolState,
        *,
        material_stop_angle_deg: float,
        bucket_linear_acceleration_terrain_m_s2: np.ndarray | None = None,
    ) -> BucketRetentionResult:
        geometry = descriptor.bucket_geometry
        if geometry is None:
            raise ValueError("[BucketRetention] authoritative bucket geometry is required")
        if not np.isclose(
            geometry.effective_capacity_m3, payload.capacity_m3, atol=1e-9, rtol=0.0
        ):
            raise ValueError("[BucketRetention] geometry/payload capacities disagree")
        stop_angle = float(material_stop_angle_deg)
        if not np.isfinite(stop_angle) or not 0.0 <= stop_angle < 90.0:
            raise ValueError("[BucketRetention] stop angle must be in [0,90)")
        acceleration = (
            np.zeros(3, dtype=np.float64)
            if bucket_linear_acceleration_terrain_m_s2 is None
            else np.asarray(bucket_linear_acceleration_terrain_m_s2, dtype=np.float64)
        )
        if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("[BucketRetention] bucket acceleration must be finite shape (3,)")
        gravity_terrain = np.asarray([0.0, 0.0, -self.config.gravity_m_s2])
        effective_terrain = gravity_terrain - acceleration if self.config.use_effective_gravity else gravity_terrain
        rotation = self._closest_rotation(tool_state.pose_terrain[:3, :3])
        effective_local = rotation.T @ effective_terrain
        yz_up = -effective_local[1:]
        if np.linalg.norm(yz_up) <= 1e-12:
            yz_up = np.asarray([0.0, 1.0])
        yz_up /= np.linalg.norm(yz_up)
        base_angle = float(np.arctan2(yz_up[1], yz_up[0]))
        offsets = np.linspace(
            -np.deg2rad(stop_angle),
            np.deg2rad(stop_angle),
            self.config.orientation_samples,
        )
        profile = np.asarray(geometry.interior_profile_local[:, 1:3], dtype=np.float64)
        full_area = abs(self._polygon_area(profile))
        if full_area <= 1e-12:
            raise ValueError("[BucketRetention] interior profile has zero area")
        mouth_yz = np.asarray(geometry.mouth_polygon_local[:, 1:3], dtype=np.float64)
        best_volume = -1.0
        best_normal_yz = yz_up
        best_limit_offset = 0.0
        for offset in offsets:
            angle = base_angle + float(offset)
            normal = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)
            limiting_offset = float(np.min(mouth_yz @ normal))
            clipped = self.internal_fill_model.clip_half_plane(
                profile, normal, limiting_offset
            )
            area = abs(self._polygon_area(clipped)) if len(clipped) >= 3 else 0.0
            volume = min(area * geometry.interior_width_m, payload.capacity_m3)
            if volume > best_volume:
                best_volume = volume
                best_normal_yz = normal
                best_limit_offset = limiting_offset
        retainable = float(max(0.0, best_volume))
        fraction = float(
            np.clip(
                retainable / geometry.geometric_capacity_m3,
                0.0,
                1.0,
            )
        )
        retained_volume = min(payload.volume_m3, retainable)
        spill = payload.volume_m3 - retained_volume
        normal3 = np.asarray([0.0, best_normal_yz[0], best_normal_yz[1]], dtype=np.float64)
        normal3 /= np.linalg.norm(normal3)
        normal_changed = (
            payload.free_surface is None
            or not np.allclose(
                payload.free_surface.normal_bucket_frame, normal3, atol=1e-6
            )
        )
        phase = (
            BucketFillPhase.SPILLING
            if spill > 1e-12
            else BucketFillPhase.RELAXING
            if normal_changed
            else BucketFillPhase.STATIC
        )
        next_payload = self.internal_fill_model.solve_payload(
            payload,
            descriptor,
            normal3,
            volume_m3=retained_volume,
            phase=phase,
        )
        return BucketRetentionResult(
            payload=next_payload,
            retainable_volume_m3=retainable,
            spill_volume_m3=spill,
            free_surface_normal_bucket_frame=normal3,
            effective_gravity_bucket_frame_m_s2=effective_local,
            retained_cross_section_fraction=fraction,
            free_surface_offset_m=next_payload.free_surface.offset_m,
        )

    @staticmethod
    def _polygon_area(polygon: np.ndarray) -> float:
        if len(polygon) < 3:
            return 0.0
        return float(
            0.5
            * np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )

    @staticmethod
    def _closest_rotation(linear: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return rotation
