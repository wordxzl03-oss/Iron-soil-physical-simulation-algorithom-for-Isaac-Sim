"""Curvature smoothing and bounded nonlinear shooting tracker."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..vehicle import VehicleCommand
from .lattice import ArticulatedState


def smooth_path(states: tuple[ArticulatedState, ...], iterations: int = 2) -> np.ndarray:
    if not states:
        return np.empty((0, 4), dtype=np.float64)
    points = np.asarray([[s.x_m, s.y_m, s.yaw_rad, s.articulation_rad] for s in states], dtype=np.float64)
    for _ in range(max(0, int(iterations))):
        if len(points) < 3:
            break
        smoothed = [points[0]]
        for i in range(len(points) - 1):
            q = 0.75 * points[i] + 0.25 * points[i + 1]
            r = 0.25 * points[i] + 0.75 * points[i + 1]
            smoothed.extend((q, r))
        smoothed.append(points[-1]); points = np.asarray(smoothed)
    if len(points) > 1:
        delta = np.diff(points[:, :2], axis=0)
        yaw = np.arctan2(delta[:, 1], delta[:, 0])
        points[:-1, 2] = yaw; points[-1, 2] = yaw[-1]
    return points


@dataclass(frozen=True)
class PathTrackerConfig:
    wheelbase_m: float = 3.2
    horizon_steps: int = 8
    dt_s: float = 0.15
    target_speed_m_s: float = 1.2
    maximum_articulation_deg: float = 38.0
    maximum_articulation_rate_deg_s: float = 22.0
    position_weight: float = 2.0
    heading_weight: float = 0.8
    articulation_weight: float = 0.2
    effort_weight: float = 0.05


class NonlinearPathTracker:
    """Small bounded NMPC-style shooting baseline with explicit constraints."""

    def __init__(self, config: PathTrackerConfig | None = None) -> None:
        self.config = config or PathTrackerConfig()

    def command(self, state: ArticulatedState, speed_m_s: float, path_xy_yaw_gamma: np.ndarray) -> VehicleCommand:
        path = np.asarray(path_xy_yaw_gamma, dtype=float)
        if path.ndim != 2 or path.shape[1] != 4 or len(path) == 0:
            return VehicleCommand(brake=1.0)
        nearest = int(np.argmin(np.sum((path[:, :2] - np.array([state.x_m, state.y_m])) ** 2, axis=1)))
        reference = path[nearest : nearest + self.config.horizon_steps]
        best = None
        for normalized_rate in np.linspace(-1.0, 1.0, 9):
            cost = self._rollout_cost(state, speed_m_s, normalized_rate, reference)
            if best is None or cost < best[0]:
                best = (cost, float(normalized_rate))
        speed_error = self.config.target_speed_m_s - float(speed_m_s)
        throttle = float(np.clip(0.55 * speed_error, -0.6, 0.6))
        return VehicleCommand(throttle=throttle, steering=best[1]).normalized()

    def _rollout_cost(self, initial: ArticulatedState, speed: float, normalized_rate: float, reference: np.ndarray) -> float:
        cfg = self.config; state = initial; total = 0.0
        max_gamma = math.radians(cfg.maximum_articulation_deg)
        rate = normalized_rate * math.radians(cfg.maximum_articulation_rate_deg_s)
        for step in range(cfg.horizon_steps):
            gamma = float(np.clip(state.articulation_rad + rate * cfg.dt_s, -max_gamma, max_gamma))
            velocity = float(np.clip(speed + 0.8 * (cfg.target_speed_m_s - speed) * cfg.dt_s, -cfg.target_speed_m_s, cfg.target_speed_m_s))
            yaw = state.yaw_rad + velocity * math.sin(gamma) / cfg.wheelbase_m * cfg.dt_s
            state = ArticulatedState(state.x_m + velocity * math.cos(yaw) * cfg.dt_s, state.y_m + velocity * math.sin(yaw) * cfg.dt_s, yaw, gamma, 1)
            target = reference[min(step, len(reference) - 1)]
            position_error = np.sum((np.array([state.x_m, state.y_m]) - target[:2]) ** 2)
            heading_error = ((state.yaw_rad - target[2] + math.pi) % (2 * math.pi) - math.pi) ** 2
            total += cfg.position_weight * position_error + cfg.heading_weight * heading_error + cfg.articulation_weight * (gamma - target[3]) ** 2 + cfg.effort_weight * normalized_rate**2
        return float(total)
