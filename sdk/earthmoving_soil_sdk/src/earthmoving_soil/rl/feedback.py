"""Read-only RL feedback and slower-policy aggregation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import numpy as np


@dataclass(frozen=True)
class RLSoilFeedback:
    bucket_force_world: np.ndarray | None
    bucket_torque_world: np.ndarray | None
    bucket_force_norm_n: float | None
    bucket_force_peak_since_last_rl_step_n: float | None
    bucket_force_mean_since_last_rl_step_n: np.ndarray | None
    bucket_impulse_since_last_rl_step_ns: np.ndarray | None
    bucket_torque_peak_since_last_rl_step_nm: float | None
    bucket_torque_mean_since_last_rl_step_nm: np.ndarray | None
    left_track_force_world: np.ndarray | None
    left_track_torque_world: np.ndarray | None
    right_track_force_world: np.ndarray | None
    right_track_torque_world: np.ndarray | None
    penetration_depth_m: float | None
    contact_area_m2: float | None
    yielded_area_m2: float | None
    failure_active_volume_m3: float | None
    payload_volume_m3: float
    payload_mass_kg: float
    captured_volume_delta_m3: float
    captured_mass_delta_kg: float
    displaced_volume_delta_m3: float | None
    displaced_mass_delta_kg: float | None
    spill_volume_delta_m3: float | None
    spill_mass_delta_kg: float | None
    deposited_volume_delta_m3: float | None
    deposited_mass_delta_kg: float | None
    mobile_volume_m3: float
    mobile_mass_kg: float
    moving_mobile_volume_m3: float | None
    mobile_speed_summary_m_s: dict[str, float] | None
    local_heightmap_patch_m: np.ndarray | None
    terrain_delta_patch_m: np.ndarray | None
    physical_yield_area_m2: float | None
    terrain_state: str
    mass_error_m3: float
    soil_power_w: float | None
    soil_work_delta_j: float | None
    physics_core_version: str
    package_version: str
    material_profile: str
    config_hash: str
    availability: dict[str, str]


class RLFeedbackAccumulator:
    """Aggregate physical outputs between RL decisions without feeding physics."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._duration_s = 0.0
        self._force_time = np.zeros(3)
        self._torque_time = np.zeros(3)
        self._impulse = np.zeros(3)
        self._peak_force = 0.0
        self._peak_torque = 0.0
        self._work_j = 0.0
        self._captured_mass_kg = 0.0
        self._spill_mass_kg = 0.0
        self._terrain_abs_change_m3_proxy = 0.0

    def add(
        self,
        feedback: RLSoilFeedback,
        dt_s: float,
        *,
        tool_linear_velocity_world: np.ndarray | None = None,
        tool_angular_velocity_world: np.ndarray | None = None,
    ) -> None:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        self._duration_s += dt
        if feedback.bucket_force_world is not None:
            force = np.asarray(feedback.bucket_force_world, dtype=np.float64)
            torque = np.asarray(feedback.bucket_torque_world, dtype=np.float64)
            self._force_time += force * dt
            self._torque_time += torque * dt
            self._impulse += force * dt
            self._peak_force = max(self._peak_force, float(np.linalg.norm(force)))
            self._peak_torque = max(self._peak_torque, float(np.linalg.norm(torque)))
            if tool_linear_velocity_world is not None and tool_angular_velocity_world is not None:
                velocity = np.asarray(tool_linear_velocity_world, dtype=np.float64)
                angular = np.asarray(tool_angular_velocity_world, dtype=np.float64)
                self._work_j += float(np.dot(force, velocity) + np.dot(torque, angular)) * dt
        self._captured_mass_kg += feedback.captured_mass_delta_kg
        self._spill_mass_kg += feedback.spill_mass_delta_kg or 0.0
        if feedback.terrain_delta_patch_m is not None:
            self._terrain_abs_change_m3_proxy += float(np.sum(np.abs(feedback.terrain_delta_patch_m)))

    def emit(self, latest: RLSoilFeedback, *, reset: bool = True) -> RLSoilFeedback:
        duration = max(self._duration_s, 1.0e-15)
        result = replace(
            latest,
            bucket_force_peak_since_last_rl_step_n=self._peak_force,
            bucket_force_mean_since_last_rl_step_n=self._force_time / duration,
            bucket_impulse_since_last_rl_step_ns=self._impulse.copy(),
            bucket_torque_peak_since_last_rl_step_nm=self._peak_torque,
            bucket_torque_mean_since_last_rl_step_nm=self._torque_time / duration,
            soil_work_delta_j=self._work_j,
            captured_mass_delta_kg=self._captured_mass_kg,
            spill_mass_delta_kg=self._spill_mass_kg,
        )
        if reset:
            self.reset()
        return result

