"""Reduced-order collision/resistance-aware wheel-loader trajectory planner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from animate_loader_3d import LoaderTrajectory


@dataclass(frozen=True)
class MaterialParameters:
    bulk_density_kg_m3: float = 1950.0
    cohesion_pa: float = 7000.0
    internal_friction_deg: float = 42.0
    bucket_friction: float = 0.45
    velocity_drag: float = 1.2


@dataclass(frozen=True)
class MachineLimits:
    max_resistance_n: float = 240_000.0
    ground_clearance_m: float = 0.12
    max_cutting_edge_lift_m: float = 1.45


@dataclass(frozen=True)
class PlannedDig:
    path: np.ndarray
    pitch_deg: np.ndarray
    cut_fraction: np.ndarray
    cutting_edge_z: np.ndarray
    cut_depth: np.ndarray
    resistance_n: np.ndarray
    speed_m_s: np.ndarray
    force_limited: np.ndarray


def excavation_resistance(
    depth: float,
    width: float,
    speed: float,
    accumulated_volume: float,
    material: MaterialParameters,
) -> float:
    """FEE-inspired reduced-order horizontal resistance estimate."""
    if depth <= 0:
        return 0.0
    gravity = 9.81
    phi = np.deg2rad(material.internal_friction_deg)
    passive_factor = np.tan(np.pi / 4 + phi / 2) ** 2
    cohesive = material.cohesion_pa * width * depth
    failure_wedge = (
        0.5
        * material.bulk_density_kg_m3
        * gravity
        * width
        * depth**2
        * passive_factor
    )
    payload_friction = (
        material.bucket_friction
        * material.bulk_density_kg_m3
        * gravity
        * accumulated_volume
    )
    dynamic = (
        material.velocity_drag
        * material.bulk_density_kg_m3
        * width
        * depth
        * speed**2
    )
    return float(cohesive + failure_wedge + payload_friction + dynamic)


def _sample_height(
    height: np.ndarray,
    plot_xy: np.ndarray,
    spacing: tuple[float, float],
) -> float:
    dx, dy = spacing
    # plot coordinates are (soil y, soil x)
    i = int(np.clip(round(plot_xy[1] / dx), 0, height.shape[0] - 1))
    j = int(np.clip(round(plot_xy[0] / dy), 0, height.shape[1] - 1))
    return float(height[i, j])


def plan_resistance_aware_dig(
    height: np.ndarray,
    spacing: tuple[float, float],
    entry_plot: np.ndarray,
    forward_plot: np.ndarray,
    trajectory: LoaderTrajectory,
    material: MaterialParameters,
    limits: MachineLimits,
    count: int = 61,
) -> PlannedDig:
    """Plan approach, force-limited rising cut, curl and lift.

    The cutting edge stays close to ground at the pile toe. During penetration,
    it rises when the predicted resistance would exceed the traction/hydraulic
    limit. Curl begins during the latter half of penetration.
    """
    path = np.zeros((count, 3), dtype=float)
    pitch = np.zeros(count)
    cut_fraction = np.zeros(count)
    edge_z = np.full(count, limits.ground_clearance_m)
    cut_depth = np.zeros(count)
    resistance = np.zeros(count)
    speed = np.zeros(count)
    force_limited = np.zeros(count, dtype=bool)
    bucket_half = trajectory.travel_length * 0.44
    dt = 0.1 / trajectory.speed_scale
    accumulated_volume = 0.0
    previous_edge_xy = None

    for index, t in enumerate(np.linspace(0, 1, count)):
        if t < 0.20:  # clear-ground approach
            p = t / 0.20
            cutting_xy = (
                entry_plot
                - trajectory.approach_distance * (1 - p) * forward_plot
            )
        elif t < 0.45:  # bucket stays level and cuts straight in
            p = (t - 0.20) / 0.25
            cutting_xy = (
                entry_plot
                + trajectory.travel_length * 0.48 * p * forward_plot
            )
            cut_fraction[index] = 0.48 * p
            terrain = _sample_height(height, cutting_xy, spacing)
            candidate_z = limits.ground_clearance_m
            local_speed = (
                0.0
                if previous_edge_xy is None
                else float(np.linalg.norm(cutting_xy - previous_edge_xy) / dt)
            )
            local_speed = min(local_speed, 1.25)
            desired_depth = min(
                trajectory.max_depth,
                max(0.0, terrain - candidate_z),
            )
            predicted = excavation_resistance(
                desired_depth,
                trajectory.bucket_width,
                local_speed,
                accumulated_volume,
                material,
            )
            if predicted > limits.max_resistance_n:
                force_limited[index] = True
                low, high = 0.0, desired_depth
                for _ in range(24):
                    middle = 0.5 * (low + high)
                    trial = excavation_resistance(
                        middle,
                        trajectory.bucket_width,
                        local_speed,
                        accumulated_volume,
                        material,
                    )
                    if trial <= limits.max_resistance_n:
                        low = middle
                    else:
                        high = middle
                desired_depth = low
                candidate_z = max(candidate_z, terrain - desired_depth)
                predicted = excavation_resistance(
                    desired_depth,
                    trajectory.bucket_width,
                    local_speed,
                    accumulated_volume,
                    material,
                )
            edge_z[index] = candidate_z
            cut_depth[index] = desired_depth
            resistance[index] = predicted
            speed[index] = local_speed
            ds = trajectory.travel_length / max(
                1, round(0.50 * (count - 1))
            )
            accumulated_volume += (
                desired_depth * trajectory.bucket_width * ds * 0.72
            )
            previous_edge_xy = cutting_xy
        elif t < 0.70:  # continue forward while lifting and curling
            p = (t - 0.45) / 0.25
            cutting_xy = (
                entry_plot
                + trajectory.travel_length * (0.48 + 0.52 * p) * forward_plot
            )
            cut_fraction[index] = 0.48 + 0.52 * p
            terrain = _sample_height(height, cutting_xy, spacing)
            candidate_z = (
                limits.ground_clearance_m
                + limits.max_cutting_edge_lift_m * p**1.25
            )
            local_speed = 0.0 if previous_edge_xy is None else min(
                float(np.linalg.norm(cutting_xy - previous_edge_xy) / dt),
                1.25,
            )
            desired_depth = min(
                trajectory.max_depth, max(0.0, terrain - candidate_z)
            )
            predicted = excavation_resistance(
                desired_depth, trajectory.bucket_width, local_speed,
                accumulated_volume, material
            )
            if predicted > limits.max_resistance_n:
                force_limited[index] = True
                low, high = 0.0, desired_depth
                for _ in range(24):
                    middle = 0.5 * (low + high)
                    if excavation_resistance(
                        middle, trajectory.bucket_width, local_speed,
                        accumulated_volume, material
                    ) <= limits.max_resistance_n:
                        low = middle
                    else:
                        high = middle
                desired_depth = low
                candidate_z = max(candidate_z, terrain - desired_depth)
                predicted = excavation_resistance(
                    desired_depth, trajectory.bucket_width, local_speed,
                    accumulated_volume, material
                )
            edge_z[index] = candidate_z
            cut_depth[index] = desired_depth
            resistance[index] = predicted
            speed[index] = local_speed
            ds = trajectory.travel_length / max(
                1, round(0.50 * (count - 1))
            )
            accumulated_volume += (
                desired_depth * trajectory.bucket_width * ds * 0.72
            )
            pitch[index] = trajectory.curl_angle_deg * p
            previous_edge_xy = cutting_xy
        elif t < 0.80:  # finish curl and establish carry height
            p = (t - 0.70) / 0.10
            cutting_xy = (
                entry_plot + trajectory.travel_length * forward_plot
            )
            edge_z[index] = (
                limits.ground_clearance_m
                + limits.max_cutting_edge_lift_m
                + 0.80 * trajectory.lift_height * p
            )
            pitch[index] = trajectory.curl_angle_deg
            cut_fraction[index] = 1.0
        else:  # keep the loaded bucket raised and reverse out
            p = (t - 0.80) / 0.20
            cutting_xy = (
                entry_plot
                + trajectory.travel_length * forward_plot
                - trajectory.approach_distance * 1.55 * p * forward_plot
            )
            edge_z[index] = (
                limits.ground_clearance_m
                + limits.max_cutting_edge_lift_m
                + 0.80 * trajectory.lift_height
            )
            pitch[index] = trajectory.curl_angle_deg
            cut_fraction[index] = 1.0

        # Approximate bucket origin from the cutting edge. This preserves a
        # non-penetrating edge while the bucket curls.
        pitch_rad = np.deg2rad(pitch[index])
        center_xy = cutting_xy - bucket_half * np.cos(pitch_rad) * forward_plot
        center_z = edge_z[index] - bucket_half * np.sin(pitch_rad)
        path[index] = [center_xy[0], center_xy[1], max(center_z, 0.05)]

    return PlannedDig(
        path, pitch, cut_fraction, edge_z, cut_depth,
        resistance, speed, force_limited
    )


def apply_planned_cut(
    height: np.ndarray,
    spacing: tuple[float, float],
    entry_xy: tuple[float, float],
    heading_deg: float,
    trajectory: LoaderTrajectory,
    plan: PlannedDig,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Remove the variable-depth swept volume described by ``plan``."""
    dx, dy = spacing
    x = np.arange(height.shape[0]) * dx
    y = np.arange(height.shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    heading = np.deg2rad(heading_deg)
    rel_x, rel_y = xx - entry_xy[0], yy - entry_xy[1]
    forward = rel_x * np.sin(heading) + rel_y * np.cos(heading)
    transverse = np.abs(rel_x * np.cos(heading) - rel_y * np.sin(heading))
    penetration_mask = plan.cut_fraction > 0
    progress = plan.cut_fraction[penetration_mask]
    depths = plan.cut_depth[penetration_mask]
    order = np.argsort(progress)
    requested = np.interp(
        np.clip(forward / trajectory.travel_length, 0, 1),
        progress[order],
        depths[order],
        left=0,
        right=depths[order][-1] if len(order) else 0,
    )
    side_taper = np.clip(
        1 - (2 * transverse / trajectory.bucket_width) ** 6, 0, 1
    )
    inside = (
        (forward >= 0)
        & (forward <= trajectory.travel_length)
        & (transverse <= trajectory.bucket_width / 2)
    )
    removed = np.minimum(height, requested * side_taper) * inside
    result = height - removed
    return result, float(removed.sum() * dx * dy), removed
