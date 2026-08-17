"""Non-authoritative debug-visualization configuration and immutable frame data."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DebugVisualizationConfig:
    show_failure_zone: bool = False
    show_mobile_layer: bool = False
    show_bucket_mouth: bool = False
    show_payload_com: bool = False
    show_soil_force_vector: bool = False
    show_contact_points: bool = False
    show_planned_path: bool = False
    show_attack_candidates: bool = False
    show_slope_map: bool = False


@dataclass(frozen=True)
class DebugVisualizationFrame:
    failure_zone_points_world_m: np.ndarray
    mobile_layer_points_world_m: np.ndarray
    bucket_mouth_world_m: np.ndarray
    payload_com_world_m: np.ndarray | None
    soil_force_origin_world_m: np.ndarray | None
    soil_force_world_n: np.ndarray | None
    contact_points_world_m: np.ndarray
    planned_path_world_m: np.ndarray
    attack_candidate_poses_xy_yaw: np.ndarray

    def __post_init__(self) -> None:
        for name in ("failure_zone_points_world_m", "mobile_layer_points_world_m", "bucket_mouth_world_m", "contact_points_world_m", "planned_path_world_m", "attack_candidate_poses_xy_yaw"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim != 2 or value.shape[1] != 3 or not np.all(np.isfinite(value)):
                raise ValueError(f"[DebugVisualization] {name} must be finite (N,3)")
            copy = np.ascontiguousarray(value.copy()); copy.setflags(write=False); object.__setattr__(self, name, copy)
        for name in ("payload_com_world_m", "soil_force_origin_world_m", "soil_force_world_n"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.shape != (3,) or not np.all(np.isfinite(array)):
                    raise ValueError(f"[DebugVisualization] {name} must be finite (3,)")
                copy = np.ascontiguousarray(array.copy()); copy.setflags(write=False); object.__setattr__(self, name, copy)
