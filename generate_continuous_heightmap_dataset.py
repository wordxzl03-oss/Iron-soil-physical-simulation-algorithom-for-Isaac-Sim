"""Generate video-inspired, sequential Minslope excavation height maps.

The default preset places a nominally 25 m stockpile inside a larger 35 m
carrier grid at 0.05 m resolution.  The extra ground ring lets every side of
the pile taper naturally to zero instead of being clipped by the height-map
boundary.  Each sequence records the stable height field before and after six
consecutive wheel-loader scoops.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slope_model import (
    RelaxationStats,
    fractal_perlin_noise,
    max_neighbor_slope,
    relax_critical_slope,
    scoop_loader_bucket,
)


SCHEMA_VERSION = "minslope-continuous-heightmap-v3"
RELAXATION_TOLERANCE_M = 3e-5
RELAXATION_MAX_ITERATIONS = 30_000
ACTIVE_HEIGHT_THRESHOLD_M = 0.02
ACTION_FIELDS = [
    "entry_x_m",
    "entry_y_m",
    "travel_length_m",
    "bucket_width_m",
    "max_depth_m",
    "heading_deg",
]


def _footprint_closure_info(
    height: np.ndarray,
    spacing_m: float,
    threshold_m: float = ACTIVE_HEIGHT_THRESHOLD_M,
) -> dict[str, Any]:
    """Measure whether the pile footprint is completely inside the grid."""
    edge_values = np.concatenate(
        (height[0, :], height[-1, :], height[:, 0], height[:, -1])
    )
    active = height > threshold_m
    indices = np.argwhere(active)
    if not len(indices):
        return {
            "closed_footprint": False,
            "boundary_max_height_m": float(np.max(edge_values, initial=0.0)),
            "active_boundary_cells": 0,
            "minimum_active_clearance_to_boundary_m": 0.0,
        }
    low = indices.min(axis=0)
    high = indices.max(axis=0)
    boundary_cells = int(
        active[0, :].sum()
        + active[-1, :].sum()
        + active[:, 0].sum()
        + active[:, -1].sum()
    )
    clearance_cells = min(
        int(low[0]),
        int(low[1]),
        int(height.shape[0] - 1 - high[0]),
        int(height.shape[1] - 1 - high[1]),
    )
    boundary_max = float(np.max(edge_values, initial=0.0))
    return {
        "closed_footprint": bool(boundary_cells == 0 and boundary_max <= 1e-10),
        "boundary_max_height_m": boundary_max,
        "active_boundary_cells": boundary_cells,
        "minimum_active_clearance_to_boundary_m": float(
            clearance_cells * spacing_m
        ),
    }


def _front_contact_y(
    height: np.ndarray,
    x_m: float,
    spacing_m: float,
    threshold_m: float,
) -> float:
    i = int(np.clip(round(x_m / spacing_m), 0, height.shape[0] - 1))
    active = np.flatnonzero(height[i] > threshold_m)
    if len(active):
        return float(active[0] * spacing_m)
    peak_j = int(np.argmax(height[i]))
    return float(peak_j * spacing_m)


def _video_inspired_pile(
    grid_size: int,
    workspace_size_m: float,
    pile_span_m: float,
    spacing_m: float,
    seed: int,
    target_peak_m: float,
    repose_angle_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a tall, curved stockpile ridge with an irregular footprint.

    The construction uses the maximum of several slope-bounded cones along a
    curved ridge. Max/min combinations preserve the common Lipschitz slope
    bound, so the non-rectangular footprint is stable before excavation.
    """
    rng = np.random.default_rng(seed)
    grid = np.arange(grid_size, dtype=np.float64) * spacing_m
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    critical_slope = float(np.tan(np.deg2rad(repose_angle_deg)))
    base_slope = 0.90 * critical_slope
    height = np.zeros((grid_size, grid_size), dtype=np.float64)

    ridge_count = int(rng.integers(7, 11))
    pile_center_x_m = 0.50 * workspace_size_m
    pile_center_y_m = 0.52 * workspace_size_m
    ridge_position = np.linspace(-1.0, 1.0, ridge_count)
    ridge_x = pile_center_x_m + 0.30 * pile_span_m * ridge_position
    phase = float(rng.uniform(-np.pi, np.pi))
    ridge_y = (
        pile_center_y_m
        + 0.028 * pile_span_m
        * np.sin(np.linspace(0.0, 2.3 * np.pi, ridge_count) + phase)
        + rng.normal(0.0, 0.010 * pile_span_m, ridge_count)
    )
    # Taper the end summits so the long ridge terminates in rounded toes rather
    # than continuing through the left and right boundaries.
    end_taper = 0.34 + 0.66 * np.cos(0.5 * np.pi * ridge_position) ** 0.72
    ridge_peaks = (
        target_peak_m * end_taper * rng.uniform(0.86, 0.99, ridge_count)
    )
    ridge_peaks[int(np.argmin(np.abs(ridge_position)))] = target_peak_m
    components: list[dict[str, float]] = []

    for cx, cy, peak in zip(ridge_x, ridge_y, ridge_peaks):
        cone = peak - base_slope * np.hypot(xx - cx, yy - cy)
        height = np.maximum(height, cone)
        components.append(
            {"center_x_m": float(cx), "center_y_m": float(cy), "peak_m": float(peak)}
        )

    # Lower shoulders make the exposed face and side toes asymmetric instead
    # of producing one clean oval or rectangular terrain patch.
    for _ in range(int(rng.integers(4, 8))):
        cx = float(pile_center_x_m + rng.uniform(-0.42, 0.42) * pile_span_m)
        cy = float(pile_center_y_m + rng.uniform(-0.15, 0.15) * pile_span_m)
        peak = float(target_peak_m * rng.uniform(0.26, 0.50))
        shoulder_slope = float(base_slope * rng.uniform(0.90, 1.0))
        height = np.maximum(
            height,
            peak - shoulder_slope * np.hypot(xx - cx, yy - cy),
        )
    height = np.maximum(height, 0.0)

    # A slope-bounded cap reserves an explicit flat-ground ring around the
    # carrier grid.  Taking the minimum of two slope-bounded surfaces keeps the
    # result stable while guaranteeing zero height on all four outer edges.
    ground_ring_m = max(2.0 * spacing_m, 0.04 * pile_span_m)
    distance_to_boundary_m = np.minimum.reduce(
        (xx, yy, workspace_size_m - xx, workspace_size_m - yy)
    )
    closure_surface = 0.94 * critical_slope * np.maximum(
        distance_to_boundary_m - ground_ring_m, 0.0
    )
    height = np.minimum(height, closure_surface)

    # Add metre-scale undulation while retaining a slope reserve. If a random
    # realization is too steep, reduce only the roughness contribution.
    roughness = fractal_perlin_noise(
        grid_size,
        grid_size,
        seed=seed + 1009,
        base_cells=5,
        octaves=4,
        persistence=0.48,
    )
    taper = np.clip(height / max(0.20 * target_peak_m, 1e-9), 0.0, 1.0)
    roughness_amplitude_m = 0.024 * target_peak_m
    candidate = height
    for _ in range(8):
        candidate = np.maximum(height + roughness_amplitude_m * roughness * taper, 0.0)
        candidate *= target_peak_m / max(float(candidate.max()), 1e-12)
        if max_neighbor_slope(candidate, (spacing_m, spacing_m)) <= 0.97 * critical_slope:
            break
        roughness_amplitude_m *= 0.5
    height = candidate

    # Shallow, slope-bounded face scars reflect earlier loading marks visible
    # in the reference video without making them part of the six saved actions.
    scar_count = int(rng.integers(2, 5))
    scar_records: list[dict[str, float]] = []
    for scar_index in range(scar_count):
        fraction = (scar_index + 1) / (scar_count + 1)
        cx = float(
            pile_center_x_m
            + pile_span_m * (-0.25 + 0.50 * fraction)
            + rng.normal(0.0, 0.020 * pile_span_m)
        )
        contact_y = _front_contact_y(
            height, cx, spacing_m, threshold_m=0.20 * target_peak_m
        )
        cy = float(contact_y + rng.uniform(0.02, 0.06) * pile_span_m)
        ci = int(np.clip(round(cx / spacing_m), 0, grid_size - 1))
        cj = int(np.clip(round(cy / spacing_m), 0, grid_size - 1))
        local_height = float(height[ci, cj])
        depth = float(rng.uniform(0.035, 0.075) * target_peak_m)
        scar_surface = (
            max(local_height - depth, 0.0)
            + 0.88 * critical_slope * np.hypot(xx - cx, yy - cy)
        )
        height = np.minimum(height, np.maximum(scar_surface, 0.0))
        scar_records.append(
            {"center_x_m": cx, "center_y_m": cy, "depth_m": depth}
        )

    height = np.minimum(height, closure_surface)
    final_height_scale = target_peak_m / max(float(height.max()), 1e-12)
    height *= final_height_scale
    roughness_amplitude_m *= final_height_scale
    for scar in scar_records:
        scar["depth_m"] *= final_height_scale
    max_slope = max_neighbor_slope(height, (spacing_m, spacing_m))
    active = height > ACTIVE_HEIGHT_THRESHOLD_M
    indices = np.argwhere(active)
    if not len(indices):
        raise RuntimeError("procedural stockpile is empty")
    low = indices.min(axis=0)
    high = indices.max(axis=0)
    bbox_cells = int(np.prod(high - low + 1))
    active_fraction_of_bbox = float(active.sum() / bbox_cells)
    if max_slope > critical_slope + 1e-8:
        raise RuntimeError("initial procedural pile exceeds the repose slope")
    closure_info = _footprint_closure_info(height, spacing_m)
    if not closure_info["closed_footprint"]:
        raise RuntimeError("initial procedural pile is open at the grid boundary")
    return height, {
        "nominal_pile_span_m": pile_span_m,
        "carrier_workspace_size_m": workspace_size_m,
        "reserved_ground_ring_m": ground_ring_m,
        "ridge_components": components,
        "face_scars": scar_records,
        "roughness_amplitude_m": roughness_amplitude_m,
        "final_height_scale": final_height_scale,
        "initial_max_neighbor_slope": max_slope,
        "initial_peak_height_m": float(height.max()),
        "active_area_m2": float(active.sum() * spacing_m**2),
        "active_fraction_of_bounding_box": active_fraction_of_bbox,
        "active_bbox_xy_m": [
            float(low[0] * spacing_m),
            float(low[1] * spacing_m),
            float(high[0] * spacing_m),
            float(high[1] * spacing_m),
        ],
        **closure_info,
    }


def _select_scoop(
    rng: np.random.Generator,
    height: np.ndarray,
    spacing_m: float,
    workspace_size_m: float,
    pile_span_m: float,
    scoop_index: int,
) -> tuple[dict[str, float], np.ndarray, float, np.ndarray]:
    """Choose a realistic front-face cut near a prescribed lateral offset."""
    linear_scale = pile_span_m / 25.0
    # Use the mass-weighted ridge centre instead of the single highest summit;
    # a real elongated stockpile may have its maximum close to one end.
    column_weight = np.sum(height, axis=1)
    center_x_m = float(
        np.dot(np.arange(height.shape[0]) * spacing_m, column_weight)
        / max(float(column_weight.sum()), 1e-12)
    )
    offset_pattern = np.asarray([0.0, -2.6, 2.6, -1.3, 1.3, 0.4])
    desired_x_m = center_x_m + offset_pattern[scoop_index % len(offset_pattern)] * linear_scale
    desired_x_m = float(np.clip(desired_x_m, 0.14 * workspace_size_m, 0.86 * workspace_size_m))
    toe_threshold_m = max(0.04, 0.18 * linear_scale)

    best: tuple[dict[str, float], np.ndarray, float, np.ndarray] | None = None
    for _ in range(8):
        entry_x_m = float(
            np.clip(
                desired_x_m + rng.normal(0.0, 0.18 * linear_scale),
                0.08 * workspace_size_m,
                0.92 * workspace_size_m,
            )
        )
        front_y_m = _front_contact_y(
            height,
            entry_x_m,
            spacing_m,
            threshold_m=toe_threshold_m,
        )
        action = {
            "entry_x_m": entry_x_m,
            "entry_y_m": max(0.0, front_y_m - rng.uniform(0.10, 0.28) * linear_scale),
            "travel_length_m": float(rng.uniform(2.6, 3.3) * linear_scale),
            "bucket_width_m": float(rng.uniform(2.9, 3.4) * linear_scale),
            "max_depth_m": float(rng.uniform(0.85, 1.20) * linear_scale),
            "heading_deg": float(rng.uniform(-7.0, 7.0)),
        }
        post_cut, removed_volume_m3, removed = scoop_loader_bucket(
            height,
            (action["entry_x_m"], action["entry_y_m"]),
            (spacing_m, spacing_m),
            travel_length=action["travel_length_m"],
            bucket_width=action["bucket_width_m"],
            max_depth=action["max_depth_m"],
            heading_deg=action["heading_deg"],
        )
        if best is None or removed_volume_m3 > best[2]:
            best = action, post_cut, removed_volume_m3, removed
    assert best is not None
    if best[2] <= 0.0:
        raise RuntimeError(f"scoop {scoop_index + 1} removed no material")
    return best


def _relax_local_cut(
    post_cut: np.ndarray,
    removed: np.ndarray,
    spacing_m: float,
    repose_angle_deg: float,
    pile_span_m: float,
    scoop_index: int = 0,
) -> tuple[np.ndarray, RelaxationStats, dict[str, Any]]:
    """Apply Minslope relaxation to an adaptively padded crop.

    Later scoops can connect to earlier disturbed regions. The crop therefore
    expands until relaxation no longer reaches an interior crop boundary.
    """
    indices = np.argwhere(removed > 0.0)
    if not len(indices):
        raise ValueError("removed height map is empty")
    linear_scale = pile_span_m / 25.0
    pad_m = (4.5 + 1.1 * scoop_index) * linear_scale
    boundary_tolerance_m = max(3e-4 * linear_scale, 8.0 * RELAXATION_TOLERANCE_M)
    relaxation_attempts = 0
    stats = None
    relaxed_crop = post_cut
    crop = post_cut
    low = np.zeros(2, dtype=int)
    high = np.asarray(post_cut.shape, dtype=int)
    maximum_interior_border_change_m = float("inf")

    for relaxation_attempts in range(1, 6):
        pad_cells = int(np.ceil(pad_m / spacing_m))
        low = np.maximum(indices.min(axis=0) - pad_cells, 0)
        high = np.minimum(indices.max(axis=0) + pad_cells + 1, post_cut.shape)
        slices = (
            slice(int(low[0]), int(high[0])),
            slice(int(low[1]), int(high[1])),
        )
        crop = post_cut[slices]
        relaxed_crop, stats = relax_critical_slope(
            crop,
            (spacing_m, spacing_m),
            critical_angle_deg=repose_angle_deg,
            max_iterations=RELAXATION_MAX_ITERATIONS,
            tolerance=RELAXATION_TOLERANCE_M,
        )
        crop_change = relaxed_crop - crop
        interior_borders: list[np.ndarray] = []
        if low[0] > 0:
            interior_borders.append(crop_change[:2].ravel())
        if high[0] < post_cut.shape[0]:
            interior_borders.append(crop_change[-2:].ravel())
        if low[1] > 0:
            interior_borders.append(crop_change[:, :2].ravel())
        if high[1] < post_cut.shape[1]:
            interior_borders.append(crop_change[:, -2:].ravel())
        maximum_interior_border_change_m = (
            float(
                np.max(
                    np.abs(np.concatenate(interior_borders)),
                    initial=0.0,
                )
            )
            if interior_borders
            else 0.0
        )
        if maximum_interior_border_change_m <= boundary_tolerance_m:
            break
        pad_m *= 1.55

    assert stats is not None
    slices = (slice(int(low[0]), int(high[0])), slice(int(low[1]), int(high[1])))
    final = post_cut.copy()
    final[slices] = relaxed_crop
    return final, stats, {
        "relaxation_crop_ij": [
            int(low[0]), int(low[1]), int(high[0]), int(high[1])
        ],
        "relaxation_crop_shape": list(crop.shape),
        "relaxation_crop_attempts": relaxation_attempts,
        "max_abs_relaxation_change_at_interior_crop_border_m": (
            maximum_interior_border_change_m
        ),
    }


def _save_heightmap_preview(
    output: Path,
    states: np.ndarray,
    workspace_size_m: float,
) -> None:
    vmax = float(states[0].max())
    total_delta = states[-1] - states[0]
    delta_limit = float(max(abs(total_delta.min()), abs(total_delta.max()), 1e-9))
    extent = (0.0, workspace_size_m, 0.0, workspace_size_m)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), constrained_layout=True)
    for state_index, axis in enumerate(axes.flat[:7]):
        image = axis.imshow(
            states[state_index],
            origin="lower",
            extent=extent,
            cmap="terrain",
            vmin=0.0,
            vmax=vmax,
        )
        axis.set(
            title=("H0: initial" if state_index == 0 else f"H{state_index}: after scoop {state_index}"),
            xlabel="y [m]",
            ylabel="x [m]",
        )
        fig.colorbar(image, ax=axis, label="height [m]", shrink=0.76)
    delta_axis = axes.flat[7]
    image = delta_axis.imshow(
        total_delta,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-delta_limit,
        vmax=delta_limit,
    )
    delta_axis.set(
        title="H6 - H0",
        xlabel="y [m]",
        ylabel="x [m]",
    )
    fig.colorbar(image, ax=delta_axis, label="delta [m]", shrink=0.76)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _save_surface_preview(
    output: Path,
    states: np.ndarray,
    spacing_m: float,
    workspace_size_m: float,
) -> None:
    stride = max(1, int(round(0.25 / spacing_m)))
    grid = np.arange(states.shape[-1], dtype=float) * spacing_m
    xx, yy = np.meshgrid(grid[::stride], grid[::stride], indexing="ij")
    fig = plt.figure(figsize=(13.2, 5.7), constrained_layout=True)
    for panel, state_index in enumerate((0, states.shape[0] - 1), start=1):
        axis = fig.add_subplot(1, 2, panel, projection="3d")
        surface = states[state_index, ::stride, ::stride]
        axis.plot_surface(
            yy,
            xx,
            surface,
            cmap="terrain",
            vmin=0.0,
            vmax=float(states[0].max()),
            linewidth=0,
            antialiased=False,
        )
        axis.set(
            title=("Initial irregular ridge H0" if state_index == 0 else "After six scoops H6"),
            xlim=(0, workspace_size_m),
            ylim=(0, workspace_size_m),
            zlim=(0, float(states[0].max()) * 1.05),
            xlabel="y [m]",
            ylabel="x [m]",
            zlabel="height [m]",
        )
        axis.view_init(elev=22, azim=-120)
        axis.set_box_aspect((1.0, 1.0, 0.48))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _resample_square_states(
    values: np.ndarray,
    source_grid_m: np.ndarray,
    target_grid_m: np.ndarray,
) -> np.ndarray:
    """Bilinearly resample one or more square height fields.

    Piecewise-linear interpolation preserves the zero-valued outer boundary
    and does not introduce slopes steeper than those already present on the
    solver grid.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-2:] != (len(source_grid_m), len(source_grid_m)):
        raise ValueError("source grid does not match height-field shape")
    if np.array_equal(source_grid_m, target_grid_m):
        return array.copy()
    leading_shape = array.shape[:-2]
    fields = array.reshape((-1, len(source_grid_m), len(source_grid_m)))
    interpolated = np.empty(
        (len(fields), len(target_grid_m), len(target_grid_m)), dtype=np.float64
    )
    intermediate = np.empty(
        (len(source_grid_m), len(target_grid_m)), dtype=np.float64
    )
    for field_index, field in enumerate(fields):
        for source_i in range(len(source_grid_m)):
            intermediate[source_i] = np.interp(
                target_grid_m, source_grid_m, field[source_i]
            )
        for target_j in range(len(target_grid_m)):
            interpolated[field_index, :, target_j] = np.interp(
                target_grid_m,
                source_grid_m,
                intermediate[:, target_j],
            )
    return interpolated.reshape(
        (*leading_shape, len(target_grid_m), len(target_grid_m))
    )


def _stabilize_resampled_states(
    states: np.ndarray,
    spacing_m: float,
    repose_angle_deg: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Remove small diagonal slope violations introduced by interpolation."""
    slope_limit = float(np.tan(np.deg2rad(repose_angle_deg)))
    slope_tolerance = RELAXATION_TOLERANCE_M / spacing_m + 1e-8
    stable_states: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for state_index, state in enumerate(np.asarray(states, dtype=np.float64)):
        slope_before = max_neighbor_slope(state, (spacing_m, spacing_m))
        if slope_before <= slope_limit + slope_tolerance:
            stable = state
            stats = None
        else:
            stable, stats = relax_critical_slope(
                state,
                (spacing_m, spacing_m),
                critical_angle_deg=repose_angle_deg,
                max_iterations=2_000,
                tolerance=RELAXATION_TOLERANCE_M,
            )
            if not stats.converged:
                raise RuntimeError(
                    f"output state H{state_index} did not stabilize after resampling"
                )
        slope_after = max_neighbor_slope(stable, (spacing_m, spacing_m))
        if slope_after > slope_limit + slope_tolerance:
            raise RuntimeError(
                f"output state H{state_index} exceeds repose slope after stabilization"
            )
        stable_states.append(stable)
        records.append(
            {
                "state": f"H{state_index}",
                "max_slope_before": slope_before,
                "max_slope_after": slope_after,
                "relaxation_iterations": 0 if stats is None else stats.iterations,
                "volume_conservation_error_m3": 0.0
                if stats is None
                else abs(stats.volume_after - stats.volume_before),
            }
        )
    return np.stack(stable_states), records


def generate_continuous_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.sequence_count < 1 or args.scoops < 1:
        raise ValueError("sequence count and scoop count must be positive")
    if args.scoops != 6:
        raise ValueError("this video-reference dataset requires exactly six scoops")
    if args.workspace_size <= 0 or args.grid_spacing <= 0:
        raise ValueError("workspace size and grid spacing must be positive")
    solver_spacing_m = float(getattr(args, "solver_spacing", args.grid_spacing))
    if solver_spacing_m < args.grid_spacing:
        raise ValueError("solver spacing must be greater than or equal to output spacing")
    spacing_ratio = solver_spacing_m / args.grid_spacing
    if not np.isclose(spacing_ratio, round(spacing_ratio), atol=1e-10):
        raise ValueError("solver spacing must be an integer multiple of output spacing")
    if args.pile_span <= 0 or args.pile_span >= args.workspace_size:
        raise ValueError("pile span must be positive and smaller than the workspace")
    if args.workspace_size - args.pile_span < 4.0 * args.grid_spacing:
        raise ValueError("workspace must leave a ground margin around the pile")
    if not 0 < args.repose_angle_min <= args.repose_angle_max < 90:
        raise ValueError("invalid repose-angle range")
    if not 0 < args.peak_height_min <= args.peak_height_max:
        raise ValueError("invalid peak-height range")
    if args.bulk_density <= 0:
        raise ValueError("bulk density must be positive")

    output_dir: Path = args.output_dir
    sequence_dir = output_dir / "sequences"
    sequence_dir.mkdir(parents=True, exist_ok=True)
    solver_grid_size = int(round(args.workspace_size / solver_spacing_m)) + 1
    actual_workspace_size_m = (solver_grid_size - 1) * solver_spacing_m
    output_grid_size = int(round(actual_workspace_size_m / args.grid_spacing)) + 1
    solver_grid = np.arange(solver_grid_size, dtype=np.float64) * solver_spacing_m
    grid = np.arange(output_grid_size, dtype=np.float64) * args.grid_spacing
    if not np.isclose(grid[-1], actual_workspace_size_m, atol=1e-10):
        raise ValueError("workspace size must align with both grid spacings")
    cell_area_m2 = solver_spacing_m**2
    rng = np.random.default_rng(args.seed)

    dataset_states: list[np.ndarray] = []
    dataset_post_cut: list[np.ndarray] = []
    dataset_removed: list[np.ndarray] = []
    dataset_deltas: list[np.ndarray] = []
    dataset_masks: list[np.ndarray] = []
    dataset_actions: list[np.ndarray] = []
    sequence_seeds: list[int] = []
    repose_angles: list[float] = []
    records: list[dict[str, Any]] = []

    for sequence_index in range(args.sequence_count):
        sequence_seed = int(rng.integers(0, 2**31 - 1))
        sequence_rng = np.random.default_rng(sequence_seed)
        target_peak_m = float(
            sequence_rng.uniform(args.peak_height_min, args.peak_height_max)
        )
        repose_angle_deg = float(
            sequence_rng.uniform(args.repose_angle_min, args.repose_angle_max)
        )
        initial, pile_info = _video_inspired_pile(
            solver_grid_size,
            actual_workspace_size_m,
            args.pile_span,
            solver_spacing_m,
            sequence_seed,
            target_peak_m,
            repose_angle_deg,
        )
        states = [initial]
        post_cut_states: list[np.ndarray] = []
        removed_maps: list[np.ndarray] = []
        actions: list[list[float]] = []
        scoop_records: list[dict[str, Any]] = []

        for scoop_index in range(args.scoops):
            before = states[-1]
            action, post_cut, removed_volume_m3, removed = _select_scoop(
                sequence_rng,
                before,
                solver_spacing_m,
                actual_workspace_size_m,
                args.pile_span,
                scoop_index,
            )
            final, relax_stats, local_info = _relax_local_cut(
                post_cut,
                removed,
                solver_spacing_m,
                repose_angle_deg,
                args.pile_span,
                scoop_index,
            )
            volume_before_m3 = float(before.sum() * cell_area_m2)
            volume_after_m3 = float(final.sum() * cell_area_m2)
            conservation_error_m3 = abs(
                volume_after_m3 - (volume_before_m3 - removed_volume_m3)
            )
            max_slope = max_neighbor_slope(
                final, (solver_spacing_m, solver_spacing_m)
            )
            slope_tolerance = RELAXATION_TOLERANCE_M / solver_spacing_m + 1e-8
            slope_limit = float(np.tan(np.deg2rad(repose_angle_deg)))
            closure_info = _footprint_closure_info(final, solver_spacing_m)
            valid = bool(
                relax_stats.converged
                and conservation_error_m3 <= 1e-8
                and max_slope <= slope_limit + slope_tolerance
                and removed_volume_m3 > 0.0
                and np.all(np.isfinite(final))
                and float(final.min()) >= -1e-10
                and closure_info["closed_footprint"]
            )
            if not valid:
                raise RuntimeError(
                    f"sequence {sequence_index}, scoop {scoop_index + 1} failed "
                    "validation: "
                    f"converged={relax_stats.converged}, "
                    f"iterations={relax_stats.iterations}, "
                    f"conservation_error_m3={conservation_error_m3:.3e}, "
                    f"max_slope={max_slope:.6f}, slope_limit={slope_limit:.6f}, "
                    f"closed={closure_info['closed_footprint']}"
                )
            states.append(final)
            post_cut_states.append(post_cut)
            removed_maps.append(removed)
            actions.append([action[field] for field in ACTION_FIELDS])
            scoop_records.append(
                {
                    "scoop_number": scoop_index + 1,
                    "action": action,
                    "removed_volume_m3": removed_volume_m3,
                    "removed_mass_kg": removed_volume_m3 * args.bulk_density,
                    "volume_before_m3": volume_before_m3,
                    "volume_after_m3": volume_after_m3,
                    "conservation_error_m3": conservation_error_m3,
                    "max_neighbor_slope": max_slope,
                    **closure_info,
                    "relaxation": asdict(relax_stats),
                    **local_info,
                    "validation_passed": valid,
                }
            )
            print(
                f"[sequence {sequence_index + 1:02d}/{args.sequence_count:02d}, "
                f"scoop {scoop_index + 1}/6] removed={removed_volume_m3:.3f} m^3 "
                f"relax={relax_stats.iterations} iterations",
                flush=True,
            )

        resampled_states = _resample_square_states(
            np.stack(states), solver_grid, grid
        )
        resampled_states, output_stabilization = _stabilize_resampled_states(
            resampled_states, args.grid_spacing, repose_angle_deg
        )
        state_array = resampled_states.astype(np.float32)
        post_cut_array = _resample_square_states(
            np.stack(post_cut_states), solver_grid, grid
        ).astype(np.float32)
        removed_array = _resample_square_states(
            np.stack(removed_maps), solver_grid, grid
        ).astype(np.float32)
        delta_array = np.diff(state_array, axis=0).astype(np.float32)
        mask_array = state_array > ACTIVE_HEIGHT_THRESHOLD_M
        all_states_closed = all(
            _footprint_closure_info(state, args.grid_spacing)["closed_footprint"]
            for state in state_array
        )
        if not all_states_closed:
            raise RuntimeError(
                f"sequence {sequence_index} contains an open-footprint state"
            )
        output_max_slopes = [
            max_neighbor_slope(state, (args.grid_spacing, args.grid_spacing))
            for state in state_array
        ]
        output_slope_tolerance = (
            RELAXATION_TOLERANCE_M / args.grid_spacing + 1e-8
        )
        if max(output_max_slopes) > slope_limit + output_slope_tolerance:
            raise RuntimeError(
                f"sequence {sequence_index} interpolation exceeded repose slope"
            )
        action_array = np.asarray(actions, dtype=np.float32)
        sequence_name = f"sequence_{sequence_index:03d}"
        np.savez_compressed(
            sequence_dir / f"{sequence_name}.npz",
            heightmaps_m=state_array,
            post_cut_heightmaps_m=post_cut_array,
            removed_heightmaps_m=removed_array,
            delta_heightmaps_m=delta_array,
            active_masks=mask_array,
            actions=action_array,
            action_fields=np.asarray(ACTION_FIELDS),
            state_labels=np.asarray([f"H{i}" for i in range(args.scoops + 1)]),
            grid_x_m=grid.astype(np.float32),
            grid_y_m=grid.astype(np.float32),
            grid_spacing_m=np.float32(args.grid_spacing),
            solver_grid_spacing_m=np.float32(solver_spacing_m),
            pile_span_m=np.float32(args.pile_span),
            origin_xy_m=np.array([0.0, 0.0], dtype=np.float32),
            repose_angle_deg=np.float32(repose_angle_deg),
            bulk_density_kg_m3=np.float32(args.bulk_density),
        )
        _save_heightmap_preview(
            output_dir / f"{sequence_name}_heightmaps.png",
            state_array,
            actual_workspace_size_m,
        )
        _save_surface_preview(
            output_dir / f"{sequence_name}_surface.png",
            state_array,
            args.grid_spacing,
            actual_workspace_size_m,
        )
        if sequence_index == 0:
            np.savetxt(
                output_dir / "sequence_000_H0_initial_m.csv",
                state_array[0],
                delimiter=",",
                fmt="%.6f",
            )
            np.savetxt(
                output_dir / "sequence_000_H6_final_m.csv",
                state_array[-1],
                delimiter=",",
                fmt="%.6f",
            )

        state_volumes = [float(state.sum() * cell_area_m2) for state in states]
        output_state_volumes = [
            float(state.sum(dtype=np.float64) * args.grid_spacing**2)
            for state in state_array
        ]
        records.append(
            {
                "sequence_id": sequence_index,
                "file": f"sequences/{sequence_name}.npz",
                "seed": sequence_seed,
                "target_peak_height_m": target_peak_m,
                "repose_angle_deg": repose_angle_deg,
                "pile": pile_info,
                "state_volumes_m3": state_volumes,
                "output_grid_state_volumes_m3": output_state_volumes,
                "output_grid_max_neighbor_slopes": output_max_slopes,
                "output_grid_stabilization": output_stabilization,
                "total_removed_volume_m3": state_volumes[0] - state_volumes[-1],
                "all_states_closed": all_states_closed,
                "scoops": scoop_records,
                "validation_passed": all(
                    item["validation_passed"] for item in scoop_records
                ),
            }
        )
        dataset_states.append(state_array)
        dataset_post_cut.append(post_cut_array)
        dataset_removed.append(removed_array)
        dataset_deltas.append(delta_array)
        dataset_masks.append(mask_array)
        dataset_actions.append(action_array)
        sequence_seeds.append(sequence_seed)
        repose_angles.append(repose_angle_deg)

    np.savez_compressed(
        output_dir / "continuous_heightmap_dataset.npz",
        heightmaps_m=np.stack(dataset_states),
        post_cut_heightmaps_m=np.stack(dataset_post_cut),
        removed_heightmaps_m=np.stack(dataset_removed),
        delta_heightmaps_m=np.stack(dataset_deltas),
        active_masks=np.stack(dataset_masks),
        actions=np.stack(dataset_actions),
        action_fields=np.asarray(ACTION_FIELDS),
        state_labels=np.asarray([f"H{i}" for i in range(args.scoops + 1)]),
        sequence_seeds=np.asarray(sequence_seeds, dtype=np.int64),
        repose_angles_deg=np.asarray(repose_angles, dtype=np.float32),
        grid_x_m=grid.astype(np.float32),
        grid_y_m=grid.astype(np.float32),
        grid_spacing_m=np.float32(args.grid_spacing),
        solver_grid_spacing_m=np.float32(solver_spacing_m),
        pile_span_m=np.float32(args.pile_span),
        origin_xy_m=np.array([0.0, 0.0], dtype=np.float32),
        workspace_size_m=np.float32(actual_workspace_size_m),
        bulk_density_kg_m3=np.float32(args.bulk_density),
    )

    reference_name = (
        args.reference_video.name if args.reference_video is not None else None
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "description": "Closed-footprint six-step continuous Minslope excavation height maps",
        "reference_video": reference_name,
        "reference_video_observations": [
            "Tall stockpile ridge extending along one side of an indoor storage area.",
            "Irregular front toe, multiple summits/shoulders, and visible old bucket scars.",
            "Large wheel loader approaches the exposed face approximately normally.",
            "Bucket dimensions and pile scale are approximate because the video has no camera calibration.",
        ],
        "sequence_count": args.sequence_count,
        "scoops_per_sequence": args.scoops,
        "state_count_per_sequence": args.scoops + 1,
        "master_seed": args.seed,
        "shape": [output_grid_size, output_grid_size],
        "solver_shape": [solver_grid_size, solver_grid_size],
        "dtype": "float32",
        "workspace_size_m": actual_workspace_size_m,
        "nominal_pile_span_m": args.pile_span,
        "grid_spacing_m": [args.grid_spacing, args.grid_spacing],
        "solver_grid_spacing_m": [solver_spacing_m, solver_spacing_m],
        "resampling": "Bilinear interpolation from the Minslope solver grid to the output grid; closed boundary and repose slope are revalidated after interpolation.",
        "height_unit": "m",
        "origin_xy_m": [0.0, 0.0],
        "axis_convention": "heightmap[i,j] is z at x=grid_x[i], y=grid_y[j]",
        "state_definition": "H0 is initial; H1..H6 are stable states after consecutive scoops.",
        "active_mask_definition": "True where pile height exceeds 0.02 m; all masks are closed inside a zero-height ground ring.",
        "closed_footprint_required": True,
        "action_fields": ACTION_FIELDS,
        "bulk_density_kg_m3": args.bulk_density,
        "bulk_density_note": "Used only for volume-to-mass conversion, not Minslope dynamics.",
        "all_sequences_valid": all(
            item["validation_passed"] and item["all_states_closed"]
            for item in records
        ),
        "sequences": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        f"""# Closed continuous {args.pile_span:g} m Minslope height-map dataset

The aggregate `continuous_heightmap_dataset.npz` stores several independent
stockpiles. Each stockpile is a continuous seven-state sequence `H0...H6`,
where `H0` is the initial pile and `H1...H6` are the stable states after six
consecutive scoops.

The nominal pile scale is {args.pile_span:g} m and the carrier grid is
{actual_workspace_size_m:g} m square at {args.grid_spacing:g} m spacing. The
larger carrier is intentional: every pile side slopes down to a zero-height
ground ring, so neither end of the stockpile is clipped open.

Minslope relaxation is solved at {solver_spacing_m:g} m spacing and sampled
onto the {args.grid_spacing:g} m output grid with bilinear interpolation.
Closed boundaries and the repose-angle limit are checked again on the output
grid.

Main array shapes:

- `heightmaps_m`: `(sequence, 7, x, y)`;
- `post_cut_heightmaps_m`: `(sequence, 6, x, y)`;
- `removed_heightmaps_m`: `(sequence, 6, x, y)`;
- `delta_heightmaps_m`: `(sequence, 6, x, y)`;
- `active_masks`: `(sequence, 7, x, y)`;
- `actions`: `(sequence, 6, action_field)`.

The array domain is rectangular because a height map is a regular grid, but
the soil footprint is closed and irregular. Use `active_masks` when
triangulating or importing the pile so cells outside the footprint are
omitted. `manifest.json` records the boundary-height and clearance checks for
every state.

Coordinate convention: `heightmap[i,j]` is z at `x=grid_x_m[i]` and
`y=grid_y_m[j]`. Units are metres. See `manifest.json` for every action,
volume, convergence result, and the qualitative observations taken from the
reference site video.

This is a reduced-order height-field approximation. The video provides shape
and scale cues but no camera calibration or measured soil parameters, so the
dataset is suitable for interface and model testing rather than ground truth.
""",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-count", type=int, default=3)
    parser.add_argument("--scoops", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("continuous_heightmap_25m_closed_dataset"),
    )
    parser.add_argument("--workspace-size", type=float, default=35.0)
    parser.add_argument("--pile-span", type=float, default=25.0)
    parser.add_argument("--grid-spacing", type=float, default=0.05)
    parser.add_argument("--solver-spacing", type=float, default=0.10)
    parser.add_argument("--peak-height-min", type=float, default=8.0)
    parser.add_argument("--peak-height-max", type=float, default=9.2)
    parser.add_argument("--repose-angle-min", type=float, default=34.0)
    parser.add_argument("--repose-angle-max", type=float, default=38.0)
    parser.add_argument("--bulk-density", type=float, default=1800.0)
    parser.add_argument("--reference-video", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_continuous_dataset(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "shape": manifest["shape"],
                "sequence_count": manifest["sequence_count"],
                "states_per_sequence": manifest["state_count_per_sequence"],
                "all_sequences_valid": manifest["all_sequences_valid"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
