"""Explicit-boundary causal diagnostics for V3 flow/arrest ownership.

This module is deliberately excluded from the production hot path.  It turns
an explicit host checkpoint into mutually exclusive A--E ownership masks and
reports the physical yield, Mobile and numerical-residual concepts separately.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..bulk_interaction.large_avalanche import (
    LargeAvalancheTransitionConfig,
    _largest_connected_region,
)
from ..bulk_interaction.yield_criterion import evaluate_cohesive_yield
from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain import TerrainGrid


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "minimum": None, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(data.size),
        "minimum": float(np.min(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "maximum": float(np.max(data)),
    }


def _mask_stats(
    mask: np.ndarray,
    resting: np.ndarray,
    mobile: np.ndarray,
    weights: np.ndarray,
    mobilization_depth_m: float,
    connectivity: int,
) -> dict[str, float | int]:
    source = np.asarray(mask, dtype=bool)
    largest, count, largest_count = _largest_connected_region(source, connectivity)
    return {
        "cell_count": int(np.count_nonzero(source)),
        "area_m2": float(np.sum(weights[source], dtype=np.float64)),
        "connected_component_count": int(count),
        "largest_component_cell_count": int(largest_count),
        "largest_component_area_m2": float(np.sum(weights[largest], dtype=np.float64)),
        "available_resting_layer_volume_m3": float(
            np.sum(np.minimum(resting[source], mobilization_depth_m) * weights[source], dtype=np.float64)
        ),
        "mobile_volume_m3": float(np.sum(mobile[source] * weights[source], dtype=np.float64)),
    }


def _mobile_fluxes(
    mobile: np.ndarray,
    velocity: np.ndarray,
    weights: np.ndarray,
    grid: TerrainGrid,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the production donor-limited edge transport for one step."""

    rows, cols = mobile.shape
    size = rows * cols
    flat_h = mobile.ravel()
    flat_v = velocity.reshape(-1, 2)
    cell_volume = flat_h * weights.ravel()
    donor_parts: list[np.ndarray] = []
    receiver_parts: list[np.ndarray] = []
    amount_parts: list[np.ndarray] = []

    left = np.arange(rows * (cols - 1), dtype=np.int64)
    left = (left // (cols - 1)) * cols + left % (cols - 1)
    right = left + 1
    face_vx = 0.5 * (flat_v[left, 0] + flat_v[right, 0])
    donor_parts.append(np.where(face_vx >= 0.0, left, right))
    receiver_parts.append(np.where(face_vx >= 0.0, right, left))
    amount_parts.append(flat_h[donor_parts[-1]] * np.abs(face_vx) * grid.dy * dt_s)

    down = np.arange((rows - 1) * cols, dtype=np.int64)
    up = down + cols
    face_vy = 0.5 * (flat_v[down, 1] + flat_v[up, 1])
    donor_parts.append(np.where(face_vy >= 0.0, down, up))
    receiver_parts.append(np.where(face_vy >= 0.0, up, down))
    amount_parts.append(flat_h[donor_parts[-1]] * np.abs(face_vy) * grid.dx * dt_s)

    donors = np.concatenate(donor_parts)
    receivers = np.concatenate(receiver_parts)
    amounts = np.concatenate(amount_parts)
    requested = np.zeros(size, dtype=np.float64)
    np.add.at(requested, donors, amounts)
    factors = np.ones(size, dtype=np.float64)
    limited = requested > cell_volume
    factors[limited] = np.divide(
        cell_volume[limited], requested[limited],
        out=np.zeros_like(cell_volume[limited]), where=requested[limited] > 0.0,
    )
    amounts *= factors[donors]
    outgoing = np.zeros(size, dtype=np.float64)
    incoming = np.zeros(size, dtype=np.float64)
    np.add.at(outgoing, donors, amounts)
    np.add.at(incoming, receivers, amounts)
    return outgoing.reshape(mobile.shape) / dt_s, incoming.reshape(mobile.shape) / dt_s


def _residual_owner_mask(
    free: np.ndarray,
    grid: TerrainGrid,
    critical_tangent: float,
    tolerance_m: float,
) -> np.ndarray:
    result = np.zeros(free.shape, dtype=bool)
    # Four unique undirected directions reproduce the 8-neighbour stencil.
    for di, dj, distance in (
        (0, 1, grid.dx), (1, 0, grid.dy),
        (1, 1, float(np.hypot(grid.dx, grid.dy))),
        (1, -1, float(np.hypot(grid.dx, grid.dy))),
    ):
        if dj >= 0:
            first = free[: free.shape[0] - di or None, : free.shape[1] - dj or None]
            second = free[di:, dj:]
            first_rows = slice(0, free.shape[0] - di or None)
            first_cols = slice(0, free.shape[1] - dj or None)
            second_rows = slice(di, None)
            second_cols = slice(dj, None)
        else:
            width = -dj
            first = free[: free.shape[0] - di or None, width:]
            second = free[di:, : free.shape[1] - width]
            first_rows = slice(0, free.shape[0] - di or None)
            first_cols = slice(width, None)
            second_rows = slice(di, None)
            second_cols = slice(0, free.shape[1] - width)
        violation = np.abs(first - second) > critical_tangent * distance + tolerance_m
        high_first = violation & (first > second)
        high_second = violation & ~high_first
        result[first_rows, first_cols] |= high_first
        result[second_rows, second_cols] |= high_second
    return result


def classify_flow_arrest_state(
    H_resting_m: np.ndarray,
    h_mobile_m: np.ndarray,
    mobile_momentum_m2_s: np.ndarray,
    material: MaterialScenario,
    grid: TerrainGrid,
    integrator: TerrainVolumeIntegrator,
    config: LargeAvalancheTransitionConfig,
    *,
    residual_active_tile_ids: np.ndarray = np.empty(0, dtype=np.int32),
    tile_size: int = 64,
    residual_tolerance_m: float = 0.002,
    dt_s: float = 1.0 / 60.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Return JSON-ready diagnostics and exact masks at an acceptance boundary."""

    resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
    mobile = np.asarray(h_mobile_m, dtype=np.float64)
    momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64)
    if mobile.shape != grid.shape or momentum.shape != grid.shape + (2,):
        raise ValueError("[FlowArrestAudit] field shape mismatch")
    weights = np.asarray(integrator.vertex_weights_m2, dtype=np.float64)
    dry = config.dry_tolerance_m
    mobile_mask = mobile > dry
    velocity = np.divide(
        momentum, mobile[..., None], out=np.zeros_like(momentum),
        where=mobile[..., None] > dry,
    )
    speed = np.linalg.norm(velocity, axis=-1)
    moving = mobile_mask & (speed > config.mobile_activity_speed_m_s)
    failure_depth = np.minimum(resting, config.mobilization_depth_m)
    resting_yield = evaluate_cohesive_yield(
        resting, mobile, material, grid, integrator,
        layer_depth_m=failure_depth,
    )
    mobile_yield = evaluate_cohesive_yield(
        resting, mobile, material, grid, integrator,
        layer_depth_m=mobile, moving_mask=moving,
    )
    physical_start = resting_yield.start_mask
    physical_stop = mobile_mask & (mobile_yield.yield_stop_margin_pa > 0.0)
    outgoing, incoming = _mobile_fluxes(mobile, velocity, weights, grid, dt_s)
    mobile_dynamic = mobile_mask & (physical_stop | moving | (outgoing > 0.0))

    largest, _, largest_count = _largest_connected_region(physical_start, config.connectivity)
    largest_area = float(np.sum(weights[largest], dtype=np.float64))
    hysteresis = np.maximum(
        resting_yield.tau_resist_start_pa - resting_yield.tau_resist_stop_pa,
        1.0e-12,
    )
    severity = np.clip(resting_yield.yield_stop_margin_pa / hysteresis, 0.0, 1.0)
    mobilizable = float(np.sum(
        np.minimum(resting[largest], config.mobilization_depth_m * severity[largest])
        * weights[largest], dtype=np.float64,
    ))
    mean_excess = float(np.mean(
        np.maximum(np.rad2deg(resting_yield.slope_rad[largest]) - material.start_angle_deg, 0.0)
    )) if largest_count else 0.0
    candidate_pass = bool(
        largest_count >= config.minimum_connected_cells
        and largest_area >= config.minimum_connected_area_m2
        and mobilizable >= config.minimum_mobilizable_volume_m3
        and mean_excess >= config.minimum_mean_excess_start_deg
    )
    large_candidate = largest & candidate_pass

    free = resting + mobile
    residual_violation = _residual_owner_mask(
        free, grid, float(np.tan(np.deg2rad(material.stop_angle_deg))),
        residual_tolerance_m,
    )
    gy, gx = np.gradient(free, grid.dy, grid.dx, edge_order=1)
    legacy_geometric = np.arctan(np.hypot(gx, gy)) > np.deg2rad(material.stop_angle_deg)
    residual_pending = np.zeros(grid.shape, dtype=bool)
    tiles_x = (grid.nx + tile_size - 1) // tile_size
    for tile in np.asarray(residual_active_tile_ids, dtype=np.int64):
        tile_y, tile_x = divmod(int(tile), tiles_x)
        r0, c0 = tile_y * tile_size, tile_x * tile_size
        residual_pending[r0:min(r0 + tile_size, grid.ny), c0:min(c0 + tile_size, grid.nx)] = True

    # The reported GPU unstable mask is exactly physical_start.  The ordering
    # below makes A--E mutually exclusive while preserving that definition.
    reported = physical_start.copy()
    category_a = reported & physical_start
    remaining = reported & ~category_a
    meaningful_mobile = mobile_dynamic & (
        mobile * weights >= config.mobile_activity_volume_m3
    )
    category_b = remaining & meaningful_mobile
    remaining &= ~category_b
    category_c = remaining & mobile_mask & ~physical_stop & ~moving & (outgoing <= 0.0)
    remaining &= ~category_c
    category_d = remaining & residual_violation
    category_e = remaining & ~category_d

    masks = {
        "physical_yield_start_mask": physical_start,
        "physical_yield_stop_mask": physical_stop,
        "large_avalanche_candidate_mask": large_candidate,
        "mobile_dynamic_mask": mobile_dynamic,
        "residual_violation_mask": residual_violation,
        "legacy_or_geometric_instability_mask": legacy_geometric,
        "residual_pending_mask": residual_pending,
        "A_PHYSICALLY_YIELDED_RESTING": category_a,
        "B_PHYSICALLY_FLOWING_MOBILE": category_b,
        "C_ARRESTABLE_MOBILE": category_c,
        "D_RESIDUAL_ONLY": category_d,
        "E_STALE_OR_INCONSISTENT_MASK": category_e,
    }
    stats = {
        name: _mask_stats(
            mask, resting, mobile, weights, config.mobilization_depth_m,
            config.connectivity,
        )
        for name, mask in masks.items()
    }
    mobile_values = mobile[mobile_mask]
    momentum_norm = np.linalg.norm(momentum, axis=-1)
    moving_mobile_volume = float(np.sum(mobile[moving] * weights[moving], dtype=np.float64))
    kinetic_proxy = float(
        0.5 * material.assumed_bulk_density_kg_m3
        * np.sum(mobile * speed * speed * weights, dtype=np.float64)
    )
    integrated_momentum = material.assumed_bulk_density_kg_m3 * np.asarray([
        np.sum(momentum[..., 0] * weights, dtype=np.float64),
        np.sum(momentum[..., 1] * weights, dtype=np.float64),
    ])
    total_mobile = float(np.sum(mobile * weights, dtype=np.float64))
    if not np.any(mobile_mask):
        tail_classification = "NO_MOBILE"
    elif (
        moving_mobile_volume >= config.mobile_activity_volume_m3
        and np.max(speed, initial=0.0) >= config.mobile_activity_speed_m_s
    ):
        tail_classification = "PHYSICAL_FLOW"
    else:
        tail_classification = "NUMERICAL_MOBILE_TAIL"
    report: dict[str, Any] = {
        "detector_producing_reported_unstable": "COHESIVE_Y_START_ON_H_FREE",
        "reported_unstable_cell_count": int(np.count_nonzero(reported)),
        "mask_statistics": stats,
        "classification_A_to_E": {
            name: stats[name]
            for name in (
                "A_PHYSICALLY_YIELDED_RESTING",
                "B_PHYSICALLY_FLOWING_MOBILE",
                "C_ARRESTABLE_MOBILE",
                "D_RESIDUAL_ONLY",
                "E_STALE_OR_INCONSISTENT_MASK",
            )
        },
        "large_candidate": {
            "pass": candidate_pass,
            "largest_cell_count": int(largest_count),
            "largest_area_m2": largest_area,
            "mobilizable_volume_m3": mobilizable,
            "mean_geometric_excess_start_deg": mean_excess,
        },
        "mobile_tail": {
            "classification": tail_classification,
            "cell_count": int(np.count_nonzero(mobile_mask)),
            "component_count": stats["mobile_dynamic_mask"]["connected_component_count"],
            "largest_component_area_m2": stats["mobile_dynamic_mask"]["largest_component_area_m2"],
            "total_volume_m3": total_mobile,
            "moving_volume_m3": moving_mobile_volume,
            "total_kinetic_energy_proxy_j": kinetic_proxy,
            "integrated_momentum_norm_kg_m_s": float(np.linalg.norm(integrated_momentum)),
            "h_mobile_m": _distribution(mobile_values),
            "speed_m_s": _distribution(speed[mobile_mask]),
            "momentum_m2_s": _distribution(momentum_norm[mobile_mask]),
            "Y_start_pa": _distribution(resting_yield.yield_start_margin_pa[mobile_mask]),
            "Y_stop_pa": _distribution(mobile_yield.yield_stop_margin_pa[mobile_mask]),
            "outgoing_flux_m3_s": _distribution(outgoing[mobile_mask]),
            "incoming_flux_m3_s": _distribution(incoming[mobile_mask]),
        },
        "concept_conflation": {
            "pre_fix": "LOCAL_STATIC_INSTABILITY_PHYSICAL_Y_START_WAS_ALSO_USED_AS_RESIDUAL_FRONTIER_SEED",
            "post_fix": "PHYSICAL_YIELD_OWNS_BEFORE_RESIDUAL_AND_COHESIVE_STABLE_GEOMETRIC_SEEDS_DO_NOT_FLATTEN_H_FREE",
        },
    }
    return report, masks

