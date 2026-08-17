"""Vehicle stability monitor for planner hard constraints and episode logs."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SafetyMonitorConfig:
    maximum_roll_deg: float = 18.0
    maximum_pitch_deg: float = 25.0
    minimum_wheel_contacts: int = 3
    com_support_margin_m: float = 0.15


@dataclass(frozen=True)
class VehicleStabilityObservation:
    roll_rad: float
    pitch_rad: float
    wheel_contact_states: tuple[bool, ...]
    com_projection_vehicle_xy_m: np.ndarray
    support_polygon_vehicle_xy_m: np.ndarray


@dataclass(frozen=True)
class VehicleSafetyStatus:
    safe: bool
    roll_limit_exceeded: bool
    pitch_limit_exceeded: bool
    wheel_contact_loss: bool
    com_outside_support: bool
    tip_risk_proxy: float
    support_margin_m: float


class VehicleSafetyMonitor:
    def __init__(self, config: SafetyMonitorConfig | None = None) -> None:
        self.config = config or SafetyMonitorConfig()

    def evaluate(self, observation: VehicleStabilityObservation) -> VehicleSafetyStatus:
        roll = float(observation.roll_rad); pitch = float(observation.pitch_rad)
        polygon = np.asarray(observation.support_polygon_vehicle_xy_m, dtype=float)
        com = np.asarray(observation.com_projection_vehicle_xy_m, dtype=float)
        if not np.all(np.isfinite([roll, pitch])) or com.shape != (2,) or polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.all(np.isfinite(polygon)):
            raise ValueError("[Safety] observation geometry is invalid")
        area = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))
        orientation = 1.0 if area >= 0.0 else -1.0
        signed = []
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = second - first
            signed.append(orientation * (edge[0] * (com[1] - first[1]) - edge[1] * (com[0] - first[0])) / max(np.linalg.norm(edge), 1e-12))
        margin = float(min(signed))
        roll_ratio = abs(roll) / np.deg2rad(self.config.maximum_roll_deg)
        pitch_ratio = abs(pitch) / np.deg2rad(self.config.maximum_pitch_deg)
        contact_loss = sum(bool(value) for value in observation.wheel_contact_states) < self.config.minimum_wheel_contacts
        outside = margin < self.config.com_support_margin_m
        risk = float(max(roll_ratio, pitch_ratio, max(0.0, 1.0 - margin / max(self.config.com_support_margin_m, 1e-9)), 1.0 if contact_loss else 0.0))
        roll_bad = roll_ratio > 1.0; pitch_bad = pitch_ratio > 1.0
        return VehicleSafetyStatus(not (roll_bad or pitch_bad or contact_loss or outside), roll_bad, pitch_bad, contact_loss, outside, risk, margin)
