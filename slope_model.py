"""Critical-slope cellular automaton for a 2.5-D soil height map.

Reproduces the soil-slump mechanism described in Section III-B.1 of
"Large Scale Robotic Material Handling: Learning, Planning, and Control".
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RelaxationStats:
    iterations: int
    converged: bool
    max_slope: float
    volume_before: float
    volume_after: float


@dataclass(frozen=True)
class AvalancheStats:
    """Diagnostics for one hysteretic avalanche event."""

    iterations: int
    triggered: bool
    converged: bool
    moved_volume: float
    max_slope_before: float
    max_slope_after: float
    active_cell_count: int
    volume_before: float
    volume_after: float


@dataclass(frozen=True)
class LocalFailureStats:
    """Diagnostics for collision-triggered, spatially bounded slope failure."""

    triggered: bool
    converged: bool
    iterations: int
    collision_energy_j: float
    energy_used_j: float
    moved_volume_m3: float
    active_area_m2: float
    max_propagation_distance_m: float
    max_mobilized_depth_m: float
    min_factor_of_safety: float
    volume_before_m3: float
    volume_after_m3: float


def gaussian_pile(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    peak_height: float = 2.0,
    sigma_x: float = 1.15,
    sigma_y: float = 0.9,
) -> np.ndarray:
    """Create a smooth initial pile similar to the paper's sampled profiles."""
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    y = (np.arange(ny) - (ny - 1) / 2) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return peak_height * np.exp(
        -0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2)
    )


def fractal_perlin_noise(
    nx: int,
    ny: int,
    seed: int | None = None,
    *,
    base_cells: int = 3,
    octaves: int = 5,
    persistence: float = 0.52,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Generate repeatable, dependency-free 2-D fractal gradient noise."""
    if nx < 2 or ny < 2:
        raise ValueError("noise dimensions must be at least 2")
    if base_cells < 1 or octaves < 1:
        raise ValueError("base_cells and octaves must be positive")
    rng = np.random.default_rng(seed)
    result = np.zeros((nx, ny), dtype=float)
    weight_sum = 0.0
    amplitude = 1.0
    frequency = float(base_cells)
    sample_x = np.linspace(0.0, 1.0, nx)
    sample_y = np.linspace(0.0, 1.0, ny)

    for _ in range(octaves):
        cells = max(1, int(round(frequency)))
        angles = rng.uniform(0.0, 2.0 * np.pi, (cells + 1, cells + 1))
        gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        gx = np.minimum((sample_x * cells).astype(int), cells - 1)
        gy = np.minimum((sample_y * cells).astype(int), cells - 1)
        tx = sample_x * cells - gx
        ty = sample_y * cells - gy
        tx[-1], ty[-1] = 1.0, 1.0
        ix, iy = np.meshgrid(gx, gy, indexing="ij")
        ux, uy = np.meshgrid(tx, ty, indexing="ij")
        fade_x = ux**3 * (ux * (ux * 6.0 - 15.0) + 10.0)
        fade_y = uy**3 * (uy * (uy * 6.0 - 15.0) + 10.0)

        def dot(di: int, dj: int) -> np.ndarray:
            grad = gradients[ix + di, iy + dj]
            return grad[..., 0] * (ux - di) + grad[..., 1] * (uy - dj)

        n00, n10 = dot(0, 0), dot(1, 0)
        n01, n11 = dot(0, 1), dot(1, 1)
        nx0 = n00 + fade_x * (n10 - n00)
        nx1 = n01 + fade_x * (n11 - n01)
        result += amplitude * (nx0 + fade_y * (nx1 - nx0))
        weight_sum += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    result /= max(weight_sum, 1e-12)
    maximum = float(np.max(np.abs(result)))
    return result / maximum if maximum > 0.0 else result


def random_pile(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    seed: int | None = None,
    peak_height: float = 2.0,
    spatial_scale: float = 1.0,
) -> np.ndarray:
    """Create an asymmetric finite pile using a warped fractal-noise terrain."""
    rng = np.random.default_rng(seed)
    if spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")
    x = (np.arange(nx) - (nx - 1) / 2) * dx / spatial_scale
    y = (np.arange(ny) - (ny - 1) / 2) * dy / spatial_scale
    xx, yy = np.meshgrid(x, y, indexing="ij")
    noise_seed = int(rng.integers(0, 2**31 - 1))
    macro = fractal_perlin_noise(
        nx, ny, noise_seed, base_cells=2, octaves=5
    )
    warp_x = fractal_perlin_noise(
        nx, ny, noise_seed + 1, base_cells=2, octaves=3
    )
    warp_y = fractal_perlin_noise(
        nx, ny, noise_seed + 2, base_cells=2, octaves=3
    )

    # A rotated, domain-warped footprint makes ridges, off-centre summits and
    # non-elliptical toes while retaining a compact stockpile boundary.
    warped_x = xx + rng.uniform(0.22, 0.55) * warp_x
    warped_y = yy + rng.uniform(0.22, 0.55) * warp_y
    envelope = np.zeros_like(xx)
    mound_count = int(rng.integers(1, 5))
    for mound_index in range(mound_count):
        theta = rng.uniform(0.0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        if mound_index == 0:
            cx, cy = rng.uniform(-0.25, 0.25, size=2)
            mound_scale = 1.0
        else:
            cx, cy = rng.uniform(-1.25, 1.25, size=2)
            mound_scale = rng.uniform(0.48, 0.88)
        xr = cos_t * (warped_x - cx) + sin_t * (warped_y - cy)
        yr = -sin_t * (warped_x - cx) + cos_t * (warped_y - cy)
        radius = np.sqrt(
            (xr / rng.uniform(1.75, 2.85)) ** 2
            + (yr / rng.uniform(1.45, 2.75)) ** 2
        )
        mound = (
            mound_scale
            * np.clip(1.0 - radius, 0.0, 1.0)
            ** rng.uniform(0.68, 1.45)
        )
        # Maximum preserves separate summits and saddle lines after slope
        # stabilization better than summing several Gaussian bells.
        envelope = np.maximum(envelope, mound)
    roughness = np.clip(1.0 + rng.uniform(0.48, 0.82) * macro, 0.22, None)
    ridge = np.clip(1.0 - 0.32 * np.abs(warp_x - warp_y), 0.55, 1.0)
    height = envelope * roughness * ridge

    # Add one broad shoulder; unlike the old many-Gaussian construction, the
    # fractal field controls most of the shape variation at several scales.
    shoulder_x, shoulder_y = rng.uniform(-1.2, 1.2, size=2)
    shoulder = np.exp(
        -0.5 * (
            ((xx - shoulder_x) / rng.uniform(0.75, 1.35)) ** 2
            + ((yy - shoulder_y) / rng.uniform(0.65, 1.25)) ** 2
        )
    )
    height = np.clip(height + rng.uniform(0.08, 0.24) * shoulder * envelope, 0.0, None)
    if height.max() > 0:
        height *= peak_height / height.max()
    return height


def scoop_ellipsoid(
    height: np.ndarray,
    center_xy: tuple[float, float],
    grid_spacing: tuple[float, float],
    radii_xy: tuple[float, float] = (0.75, 0.55),
    max_depth: float = 0.65,
) -> tuple[np.ndarray, float]:
    """Remove an ellipsoidal clamshell footprint and return removed volume."""
    dx, dy = grid_spacing
    x = np.arange(height.shape[0]) * dx
    y = np.arange(height.shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    rx, ry = radii_xy
    r2 = ((xx - center_xy[0]) / rx) ** 2 + ((yy - center_xy[1]) / ry) ** 2
    requested_depth = max_depth * np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    removed_height = np.minimum(height, requested_depth)
    result = height - removed_height
    return result, float(removed_height.sum() * dx * dy)


def scoop_loader_bucket(
    height: np.ndarray,
    entry_xy: tuple[float, float],
    grid_spacing: tuple[float, float],
    travel_length: float = 0.92,
    bucket_width: float = 1.12,
    max_depth: float = 0.42,
    heading_deg: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Remove a wheel-loader bucket's swept, ramp-shaped cutting volume.

    ``entry_xy`` is the centre of the cutting edge at the start. A zero-degree
    heading travels in positive y; positive heading rotates toward positive x.
    """
    dx, dy = grid_spacing
    x = np.arange(height.shape[0]) * dx
    y = np.arange(height.shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    heading = np.deg2rad(heading_deg)
    rel_x = xx - entry_xy[0]
    rel_y = yy - entry_xy[1]
    travel = rel_x * np.sin(heading) + rel_y * np.cos(heading)
    transverse = np.abs(rel_x * np.cos(heading) - rel_y * np.sin(heading))
    inside = (transverse <= bucket_width / 2) & (
        (travel >= 0) & (travel <= travel_length)
    )
    side_taper = np.clip(1.0 - (2.0 * transverse / bucket_width) ** 6, 0, 1)
    longitudinal = np.clip(travel / travel_length, 0, 1)
    requested_depth = max_depth * (0.20 + 0.80 * longitudinal) * side_taper
    removed_height = np.minimum(height, requested_depth) * inside
    result = height - removed_height
    return result, float(removed_height.sum() * dx * dy), removed_height


def max_neighbor_slope(
    height: np.ndarray, grid_spacing: tuple[float, float]
) -> float:
    """Maximum absolute slope between any pair of 8-connected cells."""
    dx, dy = grid_spacing
    maximum = 0.0
    for di, dj, distance in _neighbor_pairs(dx, dy):
        a, b = _paired_views(height, di, dj)
        maximum = max(maximum, float(np.max(np.abs(a - b) / distance)))
    return maximum


def relax_critical_slope(
    height: np.ndarray,
    grid_spacing: tuple[float, float],
    critical_angle_deg: float = 34.0,
    max_iterations: int = 20_000,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, RelaxationStats]:
    """Relax all 8-neighbour slopes to ``tan(critical_angle_deg)``.

    For a cell pair with height difference ``d`` and spacing ``L``, the
    excess height is ``d - s_crit*L``. Moving half that excess from the high
    cell to the low cell makes the pair exactly critical. Four non-overlapping
    checkerboard phases are used for each neighbour direction, avoiding
    order-dependent double updates within a phase.
    """
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("height must be a 2-D array with both dimensions >= 2")
    if np.any(height < 0):
        raise ValueError("soil heights must be non-negative")
    if not 0.0 < critical_angle_deg < 90.0:
        raise ValueError("critical_angle_deg must be between 0 and 90")

    h = np.asarray(height, dtype=float).copy()
    dx, dy = grid_spacing
    if dx <= 0 or dy <= 0:
        raise ValueError("grid spacing must be positive")
    cell_area = dx * dy
    volume_before = float(h.sum() * cell_area)
    critical_slope = float(np.tan(np.deg2rad(critical_angle_deg)))
    converged = False

    # Each undirected edge appears exactly once.  The red/black masks depend
    # only on array shape, so build them once instead of allocating two full
    # index grids for every direction on every relaxation iteration.  This is
    # especially important for long, high-resolution sequential excavations.
    neighbors = _neighbor_pairs(dx, dy)
    neighbor_phases: list[tuple[int, int, float, tuple[np.ndarray, np.ndarray]]] = []
    for di, dj, distance in neighbors:
        a, _ = _paired_views(h, di, dj)
        if di:
            phase_coordinate = np.arange(a.shape[0], dtype=np.int8)[:, None]
        else:
            phase_coordinate = np.arange(a.shape[1], dtype=np.int8)[None, :]
        phase_zero = (phase_coordinate & 1) == 0
        neighbor_phases.append((di, dj, distance, (phase_zero, ~phase_zero)))
    for iteration in range(1, max_iterations + 1):
        max_excess = 0.0
        for di, dj, distance, phase_masks in neighbor_phases:
            # Red-black style phases reduce directional bias.
            for active in phase_masks:
                a, b = _paired_views(h, di, dj)
                difference = a - b
                excess = np.maximum(np.abs(difference) - critical_slope * distance, 0)
                transfer = 0.5 * excess * np.sign(difference)
                transfer *= active
                a -= transfer
                b += transfer
                max_excess = max(max_excess, float(excess.max(initial=0.0)))
        if max_excess <= tolerance:
            converged = True
            break

    volume_after = float(h.sum() * cell_area)
    stats = RelaxationStats(
        iterations=iteration,
        converged=converged,
        max_slope=max_neighbor_slope(h, grid_spacing),
        volume_before=volume_before,
        volume_after=volume_after,
    )
    return h, stats


def relax_hysteretic_avalanche(
    height: np.ndarray,
    grid_spacing: tuple[float, float],
    start_angle_deg: float = 60.0,
    stop_angle_deg: float = 38.0,
    *,
    friction_angle_deg: float = 38.0,
    cohesion_pa: float = 2500.0,
    bulk_density: float = 1800.0,
    disturbance: np.ndarray | float = 0.0,
    failure_mask: np.ndarray | None = None,
    damage: np.ndarray | None = None,
    damage_rate: float = 0.35,
    stochasticity: float = 0.03,
    seed: int | None = None,
    max_iterations: int = 4000,
    tolerance: float = 1e-7,
    record_history: bool = False,
) -> tuple[np.ndarray, AvalancheStats, np.ndarray, list[np.ndarray]]:
    """Collapse a metastable steep face using start/stop-angle hysteresis.

    Material remains unchanged below ``start_angle_deg`` even when its slope is
    above the final repose angle.  Failure begins when either the start angle is
    exceeded or a local Mohr-Coulomb factor of safety falls below one.  The
    connected moving region then relaxes toward ``stop_angle_deg``.

    ``disturbance`` is a dimensionless [0, 1] field representing bucket impact,
    vibration or rain. ``failure_mask`` can restrict propagation to a known
    exposed face. Repeated calls can feed the returned ``damage`` back into this
    function, allowing a face to weaken before a sudden event.
    Heights and all transfers are mass conserving.  The optional history is
    suitable for animation.
    """
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("height must be a 2-D array with both dimensions >= 2")
    if np.any(height < 0):
        raise ValueError("soil heights must be non-negative")
    if not 0 < stop_angle_deg < start_angle_deg < 90:
        raise ValueError("require 0 < stop_angle_deg < start_angle_deg < 90")
    if not 0 < friction_angle_deg < 90:
        raise ValueError("friction_angle_deg must be between 0 and 90")
    if cohesion_pa < 0 or bulk_density <= 0:
        raise ValueError("cohesion must be non-negative and density positive")

    dx, dy = grid_spacing
    if dx <= 0 or dy <= 0:
        raise ValueError("grid spacing must be positive")
    h = np.asarray(height, dtype=float).copy()
    disturbance_field = np.broadcast_to(
        np.asarray(disturbance, dtype=float), h.shape
    ).copy()
    if np.any((disturbance_field < 0) | (disturbance_field > 1)):
        raise ValueError("disturbance must lie in [0, 1]")
    if failure_mask is None:
        permitted = np.ones_like(h, dtype=bool)
    else:
        permitted = np.asarray(failure_mask, dtype=bool)
        if permitted.shape != h.shape:
            raise ValueError("failure_mask must have the same shape as height")
    if damage is None:
        damage_field = np.zeros_like(h)
    else:
        damage_field = np.asarray(damage, dtype=float).copy()
        if damage_field.shape != h.shape:
            raise ValueError("damage must have the same shape as height")
    damage_field = np.clip(
        damage_field + damage_rate * disturbance_field, 0.0, 1.0
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, stochasticity, h.shape)
    active = np.zeros_like(h, dtype=bool)
    cell_area = dx * dy
    volume_before = float(h.sum() * cell_area)
    max_before = max_neighbor_slope(h, grid_spacing)
    start_slope = float(np.tan(np.deg2rad(start_angle_deg)))
    stop_slope = float(np.tan(np.deg2rad(stop_angle_deg)))
    tan_phi = float(np.tan(np.deg2rad(friction_angle_deg)))
    gravity = 9.81

    # Seed failure using geometric instability and a depth-dependent
    # Mohr-Coulomb factor of safety. Disturbance and accumulated damage reduce
    # the effective cohesion without changing the conserved height field.
    for di, dj, distance in _neighbor_pairs(dx, dy):
        a, b = _paired_views(h, di, dj)
        da, db = _paired_views(damage_field, di, dj)
        ua, ub = _paired_views(disturbance_field, di, dj)
        na, nb = _paired_views(noise, di, dj)
        aa, ab = _paired_views(active, di, dj)
        pa, pb = _paired_views(permitted, di, dj)
        difference = np.abs(a - b)
        slope = difference / distance
        theta = np.arctan(slope)
        depth = np.maximum(0.5 * (a + b), min(dx, dy))
        normal = bulk_density * gravity * depth * np.cos(theta) ** 2
        shear = bulk_density * gravity * depth * np.sin(theta) * np.cos(theta)
        weakening = np.clip(0.75 * 0.5 * (da + db) + 0.55 * 0.5 * (ua + ub), 0, 0.95)
        effective_cohesion = cohesion_pa * (1.0 - weakening)
        factor_of_safety = (effective_cohesion + normal * tan_phi) / np.maximum(
            shear, 1e-12
        )
        random_margin = 1.0 + 0.5 * (na + nb)
        seed_edge = ((slope > start_slope * random_margin) | (
            factor_of_safety < random_margin
        )) & pa & pb
        aa |= seed_edge
        ab |= seed_edge

    triggered = bool(active.any())
    history = [h.copy()] if record_history else []
    moved_volume = 0.0
    converged = not triggered
    iteration = 0

    # Once seeded, failure propagates through adjacent super-repose edges.
    for iteration in range(1, max_iterations + 1):
        max_transfer = 0.0
        for di, dj, distance in _neighbor_pairs(dx, dy):
            for parity in (0, 1):
                a, b = _paired_views(h, di, dj)
                aa, ab = _paired_views(active, di, dj)
                pa, pb = _paired_views(permitted, di, dj)
                ii, jj = np.indices(a.shape)
                phase_coordinate = ii if di else jj
                phase = (phase_coordinate & 1) == parity
                difference = a - b
                excess = np.maximum(np.abs(difference) - stop_slope * distance, 0.0)
                mobile = phase & excess.astype(bool) & (aa | ab) & pa & pb
                transfer = 0.5 * excess * np.sign(difference) * mobile
                a -= transfer
                b += transfer
                aa |= mobile
                ab |= mobile
                moved_volume += float(np.abs(transfer).sum() * cell_area)
                max_transfer = max(
                    max_transfer, float(np.abs(transfer).max(initial=0.0))
                )
        if record_history and (
            iteration <= 20 or iteration % 5 == 0 or max_transfer <= tolerance
        ):
            history.append(h.copy())
        if max_transfer <= tolerance:
            converged = True
            break

    # Flow disturbs the failed patch, but settling partially heals it.
    damage_field[active] = np.maximum(0.15, 0.55 * damage_field[active])
    volume_after = float(h.sum() * cell_area)
    stats = AvalancheStats(
        iterations=iteration if triggered else 0,
        triggered=triggered,
        converged=converged,
        moved_volume=moved_volume,
        max_slope_before=max_before,
        max_slope_after=max_neighbor_slope(h, grid_spacing),
        active_cell_count=int(active.sum()),
        volume_before=volume_before,
        volume_after=volume_after,
    )
    return h, stats, damage_field, history


def relax_localized_failure_wedge(
    height: np.ndarray,
    grid_spacing: tuple[float, float],
    collision_xy: tuple[float, float],
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    collision_force_n: float = 190_000.0,
    collision_displacement_m: float = 0.45,
    energy_efficiency: float = 0.22,
    bucket_width_m: float = 2.7,
    disturbance_length_m: float = 1.8,
    max_propagation_radius_m: float = 5.5,
    active_layer_depth_m: float = 0.65,
    wedge_half_angle_deg: float = 58.0,
    stop_angle_deg: float = 38.0,
    friction_angle_deg: float = 40.0,
    cohesion_pa: float = 12_000.0,
    bulk_density: float = 1900.0,
    max_iterations: int = 2500,
    tolerance: float = 1e-7,
    record_history: bool = False,
) -> tuple[
    np.ndarray,
    LocalFailureStats,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
]:
    """Apply a collision-triggered local slip wedge to a height map.

    Pipeline: collision energy -> exponentially decaying disturbance ->
    Mohr-Coulomb factor of safety -> connected upslope wedge -> energy/radius
    constraints -> active-layer-only relaxation. Cells outside the selected
    wedge are bitwise unchanged.

    Returns ``(height, stats, disturbance, factor_of_safety, failure_mask,
    history)``.
    """
    if height.ndim != 2 or min(height.shape) < 2:
        raise ValueError("height must be a 2-D array")
    if np.any(height < 0):
        raise ValueError("soil heights must be non-negative")
    dx, dy = grid_spacing
    if dx <= 0 or dy <= 0:
        raise ValueError("grid spacing must be positive")
    if collision_force_n < 0 or collision_displacement_m < 0:
        raise ValueError("collision force and displacement must be non-negative")
    if not 0 < energy_efficiency <= 1:
        raise ValueError("energy_efficiency must lie in (0, 1]")
    if min(
        bucket_width_m, disturbance_length_m, max_propagation_radius_m,
        active_layer_depth_m,
    ) <= 0:
        raise ValueError("length and active-layer parameters must be positive")
    if not 0 < stop_angle_deg < friction_angle_deg < 90:
        raise ValueError("require 0 < stop angle < friction angle < 90")

    h = np.asarray(height, dtype=float).copy()
    cell_area = dx * dy
    x = origin_xy[0] + np.arange(h.shape[0]) * dx
    y = origin_xy[1] + np.arange(h.shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    delta_x = xx - collision_xy[0]
    delta_y = yy - collision_xy[1]
    radius = np.hypot(delta_x, delta_y)
    disturbance = np.exp(-radius / disturbance_length_m)
    disturbance *= radius <= max_propagation_radius_m

    grad_x, grad_y = np.gradient(h, dx, dy, edge_order=1)
    slope = np.hypot(grad_x, grad_y)
    theta = np.arctan(slope)
    representative_depth = np.maximum(
        np.minimum(h, active_layer_depth_m), 0.5 * min(dx, dy)
    )
    normal = bulk_density * 9.81 * representative_depth * np.cos(theta) ** 2
    shear = bulk_density * 9.81 * representative_depth * np.sin(theta) * np.cos(theta)
    factor_of_safety = (
        cohesion_pa + normal * np.tan(np.deg2rad(friction_angle_deg))
    ) / np.maximum(shear, 1e-9)

    ci = int(np.clip(round((collision_xy[0] - origin_xy[0]) / dx), 0, h.shape[0] - 1))
    cj = int(np.clip(round((collision_xy[1] - origin_xy[1]) / dy), 0, h.shape[1] - 1))
    uphill = np.array([grad_x[ci, cj], grad_y[ci, cj]], dtype=float)
    uphill_norm = float(np.linalg.norm(uphill))
    if uphill_norm < 1e-8:
        # Search a bucket-width neighbourhood if the exact impact cell is flat.
        nearby = radius <= bucket_width_m
        weighted = np.array([
            float(np.sum(grad_x[nearby] * disturbance[nearby])),
            float(np.sum(grad_y[nearby] * disturbance[nearby])),
        ])
        uphill = weighted
        uphill_norm = float(np.linalg.norm(uphill))
    if uphill_norm < 1e-8:
        uphill = np.array([0.0, 1.0])
    else:
        uphill /= uphill_norm
    lateral = np.array([-uphill[1], uphill[0]])
    along = delta_x * uphill[0] + delta_y * uphill[1]
    across = np.abs(delta_x * lateral[0] + delta_y * lateral[1])
    cone_width = (
        0.5 * bucket_width_m
        + np.maximum(along, 0.0) * np.tan(np.deg2rad(wedge_half_angle_deg))
    )
    geometric_wedge = (
        (radius <= max_propagation_radius_m)
        & (along >= -0.55 * bucket_width_m)
        & (across <= cone_width)
    )
    stop_slope = float(np.tan(np.deg2rad(stop_angle_deg)))
    strength_demand = 1.0 + 1.8 * disturbance
    eligible = (
        geometric_wedge
        & (slope >= 0.70 * stop_slope)
        & (factor_of_safety <= strength_demand)
        & (h > 0)
    )

    # Select only the connected candidate component reached from the impact.
    candidate_indices = np.argwhere(eligible)
    failure_mask = np.zeros_like(h, dtype=bool)
    if len(candidate_indices):
        distances = (
            (candidate_indices[:, 0] - ci) ** 2
            + (candidate_indices[:, 1] - cj) ** 2
        )
        seed = tuple(candidate_indices[int(np.argmin(distances))])
        queue: deque[tuple[int, int]] = deque([seed])
        failure_mask[seed] = True
        while queue:
            i, j = queue.popleft()
            for ni in range(max(0, i - 1), min(h.shape[0], i + 2)):
                for nj in range(max(0, j - 1), min(h.shape[1], j + 2)):
                    if eligible[ni, nj] and not failure_mask[ni, nj]:
                        failure_mask[ni, nj] = True
                        queue.append((ni, nj))

    collision_energy = (
        collision_force_n * collision_displacement_m * energy_efficiency
    )
    energy_remaining = collision_energy
    volume_before = float(h.sum() * cell_area)
    moved_from = np.zeros_like(h)
    moved_volume = 0.0
    history = [h.copy()] if record_history else []
    converged = not failure_mask.any()
    iteration = 0

    for iteration in range(1, max_iterations + 1):
        maximum_transfer = 0.0
        for di, dj, distance in _neighbor_pairs(dx, dy):
            for parity in (0, 1):
                a, b = _paired_views(h, di, dj)
                fa, fb = _paired_views(failure_mask, di, dj)
                ma, mb = _paired_views(moved_from, di, dj)
                ii, jj = np.indices(a.shape)
                phase_coordinate = ii if di else jj
                phase = (phase_coordinate & 1) == parity
                difference = a - b
                excess = np.maximum(
                    np.abs(difference) - stop_slope * distance, 0.0
                )
                high_a = difference > 0
                donor_active = np.where(high_a, fa, fb)
                receiver_allowed = np.where(high_a, fb, fa)
                capacity = np.where(
                    high_a,
                    active_layer_depth_m - ma,
                    active_layer_depth_m - mb,
                )
                requested = np.minimum(0.5 * excess, np.maximum(capacity, 0.0))
                mobile = phase & donor_active & receiver_allowed & (requested > tolerance)
                if not np.any(mobile) or energy_remaining <= 0:
                    continue

                # Friction/cohesion work per moved volume provides a finite,
                # physically interpretable propagation budget.
                energy_per_volume = (
                    cohesion_pa / active_layer_depth_m
                    + bulk_density * 9.81 * distance
                    * np.tan(np.deg2rad(friction_angle_deg))
                )
                possible_volume = float((requested * mobile).sum() * cell_area)
                scale = min(
                    1.0,
                    energy_remaining
                    / max(possible_volume * energy_per_volume, 1e-12),
                )
                transfer_height = requested * mobile * scale
                signed = transfer_height * np.where(high_a, 1.0, -1.0)
                a -= signed
                b += signed
                ma += transfer_height * high_a
                mb += transfer_height * (~high_a)
                transferred_volume = float(transfer_height.sum() * cell_area)
                moved_volume += transferred_volume
                energy_remaining -= transferred_volume * energy_per_volume
                maximum_transfer = max(
                    maximum_transfer,
                    float(transfer_height.max(initial=0.0)),
                )
        if record_history and (
            iteration <= 16 or iteration % 5 == 0
            or maximum_transfer <= tolerance or energy_remaining <= 0
        ):
            history.append(h.copy())
        if maximum_transfer <= tolerance or energy_remaining <= 0:
            converged = maximum_transfer <= tolerance
            break

    active_indices = np.argwhere(failure_mask)
    propagation = (
        float(radius[failure_mask].max(initial=0.0))
        if failure_mask.any() else 0.0
    )
    stats = LocalFailureStats(
        triggered=bool(failure_mask.any()),
        converged=converged,
        iterations=iteration if failure_mask.any() else 0,
        collision_energy_j=collision_energy,
        energy_used_j=collision_energy - max(energy_remaining, 0.0),
        moved_volume_m3=moved_volume,
        active_area_m2=float(len(active_indices) * cell_area),
        max_propagation_distance_m=propagation,
        max_mobilized_depth_m=float(moved_from.max(initial=0.0)),
        min_factor_of_safety=float(
            factor_of_safety[failure_mask].min(initial=np.inf)
            if failure_mask.any() else np.inf
        ),
        volume_before_m3=volume_before,
        volume_after_m3=float(h.sum() * cell_area),
    )
    return h, stats, disturbance, factor_of_safety, failure_mask, history


def _neighbor_pairs(dx: float, dy: float) -> list[tuple[int, int, float]]:
    return [
        (1, 0, dx),
        (0, 1, dy),
        (1, 1, float(np.hypot(dx, dy))),
        (1, -1, float(np.hypot(dx, dy))),
    ]


def _paired_views(
    h: np.ndarray, di: int, dj: int
) -> tuple[np.ndarray, np.ndarray]:
    if (di, dj) == (1, 0):
        return h[:-1, :], h[1:, :]
    if (di, dj) == (0, 1):
        return h[:, :-1], h[:, 1:]
    if (di, dj) == (1, 1):
        return h[:-1, :-1], h[1:, 1:]
    if (di, dj) == (1, -1):
        return h[:-1, 1:], h[1:, :-1]
    raise ValueError("unsupported neighbour offset")


def run_demo(output: Path, angle: float, no_show: bool) -> None:
    import matplotlib.pyplot as plt

    dx = dy = 0.15  # paper: 6 x 6 m pile grid at 0.15 m resolution
    shape = (41, 41)
    initial = gaussian_pile(*shape, dx, dy)
    scoop_center = (shape[0] * dx / 2 - 0.45, shape[1] * dy / 2)
    scooped, removed_volume = scoop_ellipsoid(
        initial, scoop_center, (dx, dy)
    )
    relaxed, stats = relax_critical_slope(
        scooped, (dx, dy), critical_angle_deg=angle
    )

    extent = (0, shape[1] * dy, 0, shape[0] * dx)
    vmax = float(initial.max())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (initial, scooped, relaxed),
        ("Initial pile", f"After scoop ({removed_volume:.3f} m^3)", "After slump"),
    ):
        image = axis.imshow(
            data, origin="lower", extent=extent, vmin=0, vmax=vmax, cmap="terrain"
        )
        axis.set(title=title, xlabel="y [m]", ylabel="x [m]")
        fig.colorbar(image, ax=axis, label="height [m]", shrink=0.82)
    fig.suptitle(
        f"Critical angle={angle:g}°, iterations={stats.iterations}, "
        f"max slope={stats.max_slope:.4f}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"saved: {output}")
    print(f"removed volume: {removed_volume:.6f} m^3")
    print(stats)
    if not no_show:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle", type=float, default=34.0, help="angle of repose [deg]")
    parser.add_argument("--output", type=Path, default=Path("slope_demo.png"))
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    run_demo(args.output, args.angle, args.no_show)


if __name__ == "__main__":
    main()
