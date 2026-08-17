"""Stable, lightweight and strictly observational physics monitor schema."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicsDiagnostics:
    material_profile: str
    backend: str
    resting_m3: float
    mobile_m3: float
    payload_m3: float
    airborne_m3: float
    soil_force_terrain_n: np.ndarray
    soil_torque_terrain_nm: np.ndarray
    yielded_area_m2: float
    failure_active_volume_m3: float
    mobile_moving_volume_m3: float
    mobile_velocity_p95_m_s: float
    terrain_state: str
    not_settled_reason: str
    mass_balance_error_m3: float
    physical_simulation_time_s: float
    dynamic_flow_time_s: float
    residual_solver_iterations: int
    rtf: float

    def __post_init__(self) -> None:
        for name in ("soil_force_terrain_n", "soil_torque_terrain_nm"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"[PhysicsDiagnostics] {name} must be finite shape (3,)")
            frozen = np.ascontiguousarray(value.copy())
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        if not self.material_profile or not self.backend or not self.terrain_state:
            raise ValueError("[PhysicsDiagnostics] identity/state labels must be non-empty")
        numeric = np.asarray([
            self.resting_m3, self.mobile_m3, self.payload_m3, self.airborne_m3,
            self.yielded_area_m2, self.failure_active_volume_m3,
            self.mobile_moving_volume_m3, self.mobile_velocity_p95_m_s,
            self.mass_balance_error_m3, self.physical_simulation_time_s,
            self.dynamic_flow_time_s, self.rtf,
        ])
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            raise ValueError("[PhysicsDiagnostics] scalars must be finite/non-negative")
        if self.residual_solver_iterations < 0:
            raise ValueError("[PhysicsDiagnostics] residual iterations must be non-negative")

