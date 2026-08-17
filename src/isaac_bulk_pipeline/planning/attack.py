"""Terrain-front candidates and an explainable reduced-order evaluator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .terrain_view import PlannerTerrainView


@dataclass(frozen=True)
class AttackCandidate:
    candidate_id: str
    attack_pose_xy_yaw: np.ndarray
    pre_dig_pose_xy_yaw: np.ndarray
    penetration_direction_xy: np.ndarray
    local_height_profile_m: np.ndarray
    estimated_accessibility: float
    local_slope_rad: float


class AttackCandidateGenerator:
    def __init__(self, *, pile_height_threshold_m: float = 0.25, spacing_m: float = 1.0, pre_dig_distance_m: float = 4.0, footprint_clearance_m: float = 1.2) -> None:
        if min(pile_height_threshold_m, spacing_m, pre_dig_distance_m, footprint_clearance_m) <= 0.0:
            raise ValueError("[Attack] generator distances must be positive")
        self.threshold = pile_height_threshold_m
        self.spacing_m = spacing_m
        self.pre_dig_distance_m = pre_dig_distance_m
        self.footprint_clearance_m = footprint_clearance_m

    def generate(self, view: PlannerTerrainView, H_resting_m: np.ndarray) -> tuple[AttackCandidate, ...]:
        height = np.asarray(view.grid.validate_heightmap(H_resting_m), dtype=float)
        floor = float(np.percentile(height, 10.0))
        pile = height >= floor + self.threshold
        padded = np.pad(pile, 1, constant_values=False)
        eroded = padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        boundary = pile & ~eroded
        rows, cols = np.nonzero(boundary)
        if rows.size == 0:
            return ()
        step = max(1, int(round(self.spacing_m / min(view.grid.dx, view.grid.dy))))
        order = np.lexsort((cols, rows))
        candidates: list[AttackCandidate] = []
        gy, gx = np.gradient(height, view.grid.dy, view.grid.dx)
        for ordinal, index in enumerate(order[::step]):
            row, col = int(rows[index]), int(cols[index])
            inward = np.array([gx[row, col], gy[row, col]], dtype=float)
            norm = float(np.linalg.norm(inward))
            if norm < 1e-8:
                centre = np.array([np.mean(cols), np.mean(rows)])
                inward = np.array([centre[0] - col, centre[1] - row], dtype=float)
                norm = float(np.linalg.norm(inward))
            if norm < 1e-8:
                continue
            inward /= norm
            xyz = view.grid.grid_to_terrain(row, col, height[row, col])
            attack_xy = xyz[:2]
            pre_xy = attack_xy - self.pre_dig_distance_m * inward
            ok, cost, _ = view.sample(float(pre_xy[0]), float(pre_xy[1]))
            if not ok:
                continue
            yaw = math.atan2(inward[1], inward[0])
            distances = np.linspace(0.0, 2.0, 9)
            profile = []
            for distance in distances:
                sample_xy = attack_xy + distance * inward
                rc = view.grid.terrain_to_grid(np.r_[sample_xy, 0.0])
                rr = int(np.clip(round(rc[0]), 0, view.grid.ny - 1)); cc = int(np.clip(round(rc[1]), 0, view.grid.nx - 1))
                profile.append(height[rr, cc])
            accessibility = float(np.clip(1.0 / cost, 0.0, 1.0))
            candidates.append(AttackCandidate(f"attack-{ordinal:04d}", np.array([*attack_xy, yaw]), np.array([*pre_xy, yaw]), inward, np.asarray(profile), accessibility, float(view.slope_rad[row, col])))
        return tuple(candidates)


@dataclass(frozen=True)
class AttackObjectiveWeights:
    delivered_mass: float = 1.0
    time: float = 0.15
    energy: float = 2e-6
    travel: float = 0.08
    steering: float = 0.12
    slope_risk: float = 1.0
    stability_risk: float = 2.0


@dataclass(frozen=True)
class AttackEvaluation:
    candidate: AttackCandidate
    estimated_payload_volume_m3: float
    estimated_payload_mass_kg: float
    estimated_time_s: float
    estimated_energy_j: float
    travel_cost: float
    steering_cost: float
    slope_risk: float
    stability_risk: float
    score: float
    pareto_values: dict[str, float]


class ClassicalAttackEvaluator:
    """Deterministic local failure-wedge rollout estimate, never a world model."""

    def __init__(self, *, bucket_width_m: float, bucket_capacity_m3: float, assumed_density_kg_m3: float, weights: AttackObjectiveWeights | None = None) -> None:
        if min(bucket_width_m, bucket_capacity_m3, assumed_density_kg_m3) <= 0.0:
            raise ValueError("[Attack] evaluator geometry/density must be positive")
        self.width = bucket_width_m; self.capacity = bucket_capacity_m3; self.density = assumed_density_kg_m3
        self.weights = weights or AttackObjectiveWeights()

    def evaluate(self, candidate: AttackCandidate, vehicle_pose_xy_yaw: np.ndarray) -> AttackEvaluation:
        pose = np.asarray(vehicle_pose_xy_yaw, dtype=float)
        if pose.shape != (3,):
            raise ValueError("[Attack] vehicle pose must be [x,y,yaw]")
        relief = np.maximum(candidate.local_height_profile_m - np.min(candidate.local_height_profile_m), 0.0)
        depth = float(np.percentile(relief, 75.0))
        penetration = 1.5
        wedge_volume = 0.5 * self.width * penetration * depth
        payload = min(self.capacity, wedge_volume)
        travel = float(np.linalg.norm(pose[:2] - candidate.pre_dig_pose_xy_yaw[:2]))
        heading = abs((candidate.pre_dig_pose_xy_yaw[2] - pose[2] + math.pi) % (2 * math.pi) - math.pi)
        slope_risk = float(np.tan(candidate.local_slope_rad) ** 2)
        stability = float(slope_risk * (1.0 + payload / self.capacity))
        time = 12.0 + travel / 1.5 + 2.5 * heading
        force_proxy = self.density * 9.81 * self.width * max(depth, 0.05) ** 2
        energy = force_proxy * penetration + travel * 14_000.0
        mass = payload * self.density
        w = self.weights
        score = w.delivered_mass * mass - w.time * time - w.energy * energy - w.travel * travel - w.steering * heading - w.slope_risk * slope_risk - w.stability_risk * stability
        pareto = {"delivered_mass_kg": mass, "time_s": time, "energy_j": energy, "travel_m": travel, "steering_rad": heading, "slope_risk": slope_risk, "stability_risk": stability}
        return AttackEvaluation(candidate, payload, mass, time, energy, travel, heading, slope_risk, stability, float(score), pareto)
