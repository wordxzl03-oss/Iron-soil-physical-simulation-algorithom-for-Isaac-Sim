"""Hybrid state-lattice search using an articulated-loader kinematic model."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .terrain_view import PlannerTerrainView


def _wrap(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class ArticulatedState:
    x_m: float
    y_m: float
    yaw_rad: float
    articulation_rad: float
    direction: int = 1


@dataclass(frozen=True)
class ArticulatedLatticeConfig:
    wheelbase_m: float = 3.2
    primitive_length_m: float = 0.8
    xy_resolution_m: float = 0.5
    yaw_bins: int = 32
    articulation_bins: int = 15
    maximum_articulation_deg: float = 38.0
    articulation_step_deg: float = 8.0
    vehicle_length_m: float = 6.5
    vehicle_width_m: float = 2.5
    goal_tolerance_m: float = 0.8
    reverse_penalty: float = 1.6
    steering_weight: float = 0.4
    slope_weight: float = 2.0
    max_expansions: int = 25_000

    def __post_init__(self) -> None:
        numeric = np.asarray([self.wheelbase_m, self.primitive_length_m, self.xy_resolution_m, self.maximum_articulation_deg, self.articulation_step_deg, self.vehicle_length_m, self.vehicle_width_m, self.goal_tolerance_m])
        if np.any(~np.isfinite(numeric)) or np.any(numeric <= 0.0):
            raise ValueError("[Lattice] dimensions must be finite/positive")
        if self.yaw_bins < 8 or self.articulation_bins < 3 or self.max_expansions < 1:
            raise ValueError("[Lattice] bin/expansion configuration invalid")


@dataclass(frozen=True)
class LatticePath:
    states: tuple[ArticulatedState, ...]
    total_cost: float
    expansions: int
    reached_goal: bool


class StateLatticePlanner:
    def __init__(self, config: ArticulatedLatticeConfig | None = None) -> None:
        self.config = config or ArticulatedLatticeConfig()

    def plan(self, view: PlannerTerrainView, start: ArticulatedState, goal_xy_yaw: np.ndarray) -> LatticePath:
        goal = np.asarray(goal_xy_yaw, dtype=float)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("[Lattice] goal must be finite [x,y,yaw]")
        start_key = self._key(start)
        records: dict[tuple[int, int, int, int, int], ArticulatedState] = {start_key: start}
        parent: dict[tuple[int, int, int, int, int], tuple[int, int, int, int, int] | None] = {start_key: None}
        cost = {start_key: 0.0}
        queue: list[tuple[float, int, tuple[int, int, int, int, int]]] = []
        serial = 0
        heapq.heappush(queue, (self._heuristic(start, goal), serial, start_key))
        best_key = start_key; best_h = self._heuristic(start, goal); expansions = 0
        while queue and expansions < self.config.max_expansions:
            _, _, key = heapq.heappop(queue)
            current = records[key]
            current_cost = cost[key]
            expansions += 1
            h = self._heuristic(current, goal)
            if h < best_h:
                best_h, best_key = h, key
            if np.hypot(current.x_m - goal[0], current.y_m - goal[1]) <= self.config.goal_tolerance_m and abs(_wrap(current.yaw_rad - goal[2])) <= 2 * math.pi / self.config.yaw_bins:
                best_key = key
                return LatticePath(self._reconstruct(best_key, parent, records), current_cost, expansions, True)
            for successor, primitive_cost in self._successors(current, view):
                successor_key = self._key(successor)
                tentative = current_cost + primitive_cost
                if tentative + 1e-12 >= cost.get(successor_key, float("inf")):
                    continue
                cost[successor_key] = tentative; parent[successor_key] = key; records[successor_key] = successor
                serial += 1
                heapq.heappush(queue, (tentative + self._heuristic(successor, goal), serial, successor_key))
        return LatticePath(self._reconstruct(best_key, parent, records), cost[best_key], expansions, False)

    def _successors(self, state: ArticulatedState, view: PlannerTerrainView):
        cfg = self.config
        gamma_max = math.radians(cfg.maximum_articulation_deg)
        gamma_step = math.radians(cfg.articulation_step_deg)
        for direction in (1, -1):
            for change in (-gamma_step, 0.0, gamma_step):
                gamma = float(np.clip(state.articulation_rad + change, -gamma_max, gamma_max))
                distance = direction * cfg.primitive_length_m
                yaw_rate_per_m = math.sin(gamma) / cfg.wheelbase_m
                yaw_mid = state.yaw_rad + 0.5 * distance * yaw_rate_per_m
                successor = ArticulatedState(
                    state.x_m + distance * math.cos(yaw_mid),
                    state.y_m + distance * math.sin(yaw_mid),
                    _wrap(state.yaw_rad + distance * yaw_rate_per_m),
                    gamma,
                    direction,
                )
                valid, terrain_cost, slope = self._footprint_cost(successor, view)
                if not valid:
                    continue
                primitive = abs(distance) * terrain_cost
                primitive += cfg.steering_weight * abs(change)
                primitive += cfg.slope_weight * abs(math.tan(slope))
                if direction < 0:
                    primitive *= cfg.reverse_penalty
                if direction != state.direction:
                    primitive += 0.5
                yield successor, float(primitive)

    def _footprint_cost(self, state: ArticulatedState, view: PlannerTerrainView) -> tuple[bool, float, float]:
        half_l = 0.5 * self.config.vehicle_length_m; half_w = 0.5 * self.config.vehicle_width_m
        c, s = math.cos(state.yaw_rad), math.sin(state.yaw_rad)
        values = []
        for longitudinal, lateral in ((0, 0), (half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)):
            x = state.x_m + c * longitudinal - s * lateral
            y = state.y_m + s * longitudinal + c * lateral
            ok, cost, slope = view.sample(x, y)
            if not ok:
                return False, float("inf"), float("inf")
            values.append((cost, slope))
        return True, float(np.mean([item[0] for item in values])), float(max(item[1] for item in values))

    def _key(self, state: ArticulatedState) -> tuple[int, int, int, int, int]:
        cfg = self.config; gamma_max = math.radians(cfg.maximum_articulation_deg)
        return (
            int(round(state.x_m / cfg.xy_resolution_m)),
            int(round(state.y_m / cfg.xy_resolution_m)),
            int(round((_wrap(state.yaw_rad) + math.pi) / (2 * math.pi) * cfg.yaw_bins)) % cfg.yaw_bins,
            int(round((np.clip(state.articulation_rad, -gamma_max, gamma_max) + gamma_max) / (2 * gamma_max) * (cfg.articulation_bins - 1))),
            int(np.sign(state.direction) or 1),
        )

    @staticmethod
    def _heuristic(state: ArticulatedState, goal: np.ndarray) -> float:
        return float(np.hypot(state.x_m - goal[0], state.y_m - goal[1]) + 0.4 * abs(_wrap(state.yaw_rad - goal[2])))

    @staticmethod
    def _reconstruct(key, parent, records) -> tuple[ArticulatedState, ...]:
        result = []
        while key is not None:
            result.append(records[key]); key = parent[key]
        return tuple(reversed(result))
