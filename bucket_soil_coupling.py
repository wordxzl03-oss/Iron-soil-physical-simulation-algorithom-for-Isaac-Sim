"""Deprecated offline prototype for bucket/height-field deformation.

Do not use ``HeightfieldSoil`` as an Isaac six-scoop runtime entrypoint.  The
authoritative demo is ``isaac_loader/interactive_dig_demo.py`` and runs through
``src/isaac_bulk_pipeline``.  This module remains only for legacy regression
tests and offline pile-generation examples.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slope_model import fractal_perlin_noise, relax_critical_slope


LEGACY_ISAAC_RUNTIME_DEPRECATED = True


@dataclass(frozen=True)
class BucketGeometry:
    """Local geometry matching ``isaac_loader/build_loader_usd.py``."""

    rear_x_m: float = -1.10
    cutting_edge_x_m: float = 1.30
    half_width_m: float = 1.60
    floor_z_m: float = 0.0

    def floor_corners_local(self) -> np.ndarray:
        """Return rear-left, rear-right, edge-right, edge-left corners."""
        return np.asarray(
            [
                [self.rear_x_m, self.half_width_m, self.floor_z_m],
                [self.rear_x_m, -self.half_width_m, self.floor_z_m],
                [self.cutting_edge_x_m, -self.half_width_m, self.floor_z_m],
                [self.cutting_edge_x_m, self.half_width_m, self.floor_z_m],
            ],
            dtype=np.float64,
        )


@dataclass
class BucketFrame:
    """Actual world-space bucket floor geometry for one simulation frame."""

    floor_corners_world_m: np.ndarray
    joint_positions_rad: dict[str, float] = field(default_factory=dict)
    time_s: float = 0.0

    def __post_init__(self) -> None:
        corners = np.asarray(self.floor_corners_world_m, dtype=np.float64)
        if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
            raise ValueError("floor_corners_world_m must be finite with shape (4, 3)")
        self.floor_corners_world_m = corners

    @property
    def rear_left(self) -> np.ndarray:
        return self.floor_corners_world_m[0]

    @property
    def rear_right(self) -> np.ndarray:
        return self.floor_corners_world_m[1]

    @property
    def edge_right(self) -> np.ndarray:
        return self.floor_corners_world_m[2]

    @property
    def edge_left(self) -> np.ndarray:
        return self.floor_corners_world_m[3]


@dataclass
class DigResult:
    removed_volume_m3: float
    changed_cell_count: int
    changed_bbox_ij: tuple[int, int, int, int] | None
    minimum_cutting_edge_z_m: float


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a conventional column-vector 4x4 transform to 3-D points."""
    matrix = np.asarray(transform, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    homogeneous = np.column_stack((values, np.ones(len(values))))
    return (matrix @ homogeneous.T).T[:, :3]


def frame_from_transform(
    transform: np.ndarray,
    geometry: BucketGeometry = BucketGeometry(),
    *,
    joint_positions_rad: Mapping[str, float] | None = None,
    time_s: float = 0.0,
) -> BucketFrame:
    return BucketFrame(
        transform_points(transform, geometry.floor_corners_local()),
        dict(joint_positions_rad or {}),
        time_s,
    )


def generate_realistic_closed_pile(
    *,
    workspace_size_m: float = 35.0,
    spacing_m: float = 0.05,
    seed: int = 20260806,
    peak_height_m: float = 8.3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Create a closed, irregular stockpile ridge with scars and shoulders."""
    if workspace_size_m <= 0 or spacing_m <= 0 or peak_height_m <= 0:
        raise ValueError("workspace, spacing, and peak height must be positive")
    size = int(round(workspace_size_m / spacing_m)) + 1
    actual_workspace = (size - 1) * spacing_m
    coordinates = np.arange(size, dtype=np.float64) * spacing_m
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    rng = np.random.default_rng(seed)

    # A chain of compact anisotropic lobes creates a long, multi-summit ridge.
    # End lobes are deliberately lower and smaller so both ends close naturally.
    ridge_fraction = np.asarray([0.17, 0.28, 0.40, 0.51, 0.62, 0.73, 0.83])
    peak_fraction = np.asarray([0.42, 0.72, 0.96, 0.78, 1.00, 0.70, 0.38])
    phase = float(rng.uniform(-np.pi, np.pi))
    centers_x = actual_workspace * ridge_fraction + rng.normal(
        0.0, 0.38, len(ridge_fraction)
    )
    centers_y = (
        0.56 * actual_workspace
        + 0.75 * np.sin(np.linspace(0.0, 2.2 * np.pi, len(ridge_fraction)) + phase)
        + rng.normal(0.0, 0.28, len(ridge_fraction))
    )
    heights = peak_height_m * peak_fraction * rng.uniform(0.94, 1.04, len(ridge_fraction))
    heights[4] = peak_height_m
    radii_x = rng.uniform(5.4, 7.3, len(ridge_fraction))
    radii_y = rng.uniform(8.2, 10.8, len(ridge_fraction))
    radii_x[[0, -1]] *= 0.82
    radii_y[[0, -1]] *= 0.78

    ridge_power = 5.0
    ridge_accumulator = np.zeros_like(xx)
    ridge_records: list[dict[str, float]] = []
    for cx, cy, peak, radius_x, radius_y in zip(
        centers_x, centers_y, heights, radii_x, radii_y
    ):
        # Generalized elliptical compact support. Exponents below two retain
        # angular, granular-looking shoulders instead of Gaussian domes.
        rho = (
            (np.abs(xx - cx) / radius_x) ** 1.65
            + (np.abs(yy - cy) / radius_y) ** 1.65
        ) ** (1.0 / 1.65)
        component = peak * np.maximum(1.0 - rho, 0.0) ** 1.08
        ridge_accumulator += component**ridge_power
        ridge_records.append(
            {
                "center_x_m": float(cx),
                "center_y_m": float(cy),
                "peak_m": float(peak),
                "radius_x_m": float(radius_x),
                "radius_y_m": float(radius_y),
            }
        )

    # A lower, wide common base removes the repeated-lobe appearance while the
    # high-order blend preserves distinct irregular summits.
    foundation_rho = (
        (np.abs(xx - 0.50 * actual_workspace) / 15.2) ** 1.55
        + (np.abs(yy - 0.56 * actual_workspace) / 10.8) ** 1.55
    ) ** (1.0 / 1.55)
    foundation = 4.0 * np.maximum(1.0 - foundation_rho, 0.0) ** 1.10
    ridge_accumulator += foundation**ridge_power
    pile = ridge_accumulator ** (1.0 / ridge_power)

    # Asymmetric toe lobes and old dumped material break the clean ridge outline.
    for side in (-1.0, 1.0):
        for _ in range(3):
            cx = float(rng.uniform(0.25, 0.76) * actual_workspace)
            cy = float(
                0.56 * actual_workspace
                + side * rng.uniform(5.0, 8.2)
                + rng.normal(0.0, 0.35)
            )
            peak = float(rng.uniform(1.1, 2.8))
            radius_x = float(rng.uniform(2.0, 4.5))
            radius_y = float(rng.uniform(2.2, 4.8))
            rho = np.hypot((xx - cx) / radius_x, (yy - cy) / radius_y)
            lobe = peak * np.maximum(1.0 - rho, 0.0) ** 1.15
            pile = np.maximum(pile, lobe)

    # Correlated surface texture is strongest on the face and vanishes at toes.
    noise = fractal_perlin_noise(
        size,
        size,
        seed=seed + 7919,
        base_cells=7,
        octaves=5,
        persistence=0.52,
    )
    toe_taper = np.clip(pile / 1.4, 0.0, 1.0)
    pile = np.maximum(pile + (0.18 + 0.018 * pile) * noise * toe_taper, 0.0)

    # Several nonuniform old bucket scars and slide channels on the exposed face.
    scar_records: list[dict[str, float]] = []
    for scar_x in (10.3, 14.8, 20.9, 25.4):
        scar_x += float(rng.normal(0.0, 0.45))
        column = pile[int(np.clip(round(scar_x / spacing_m), 0, size - 1))]
        contact = np.flatnonzero(column > 0.35)
        if not len(contact):
            continue
        scar_y = float(contact[0] * spacing_m + rng.uniform(1.0, 2.4))
        depth = float(rng.uniform(0.45, 1.15))
        length = float(rng.uniform(2.0, 3.8))
        width = float(rng.uniform(1.0, 1.8))
        depression = depth * np.exp(
            -((xx - scar_x) / width) ** 4 - ((yy - scar_y) / length) ** 4
        )
        pile = np.maximum(pile - depression * np.clip(pile / 0.8, 0.0, 1.0), 0.0)
        scar_records.append(
            {
                "center_x_m": scar_x,
                "center_y_m": scar_y,
                "depth_m": depth,
                "width_m": 2.0 * width,
                "length_m": 2.0 * length,
            }
        )

    # Explicit zero ground ring guarantees a complete footprint on every side.
    boundary_distance = np.minimum.reduce(
        (xx, yy, actual_workspace - xx, actual_workspace - yy)
    )
    ground_ring_m = 0.8
    edge_taper = np.clip((boundary_distance - ground_ring_m) / 1.2, 0.0, 1.0)
    edge_taper = edge_taper * edge_taper * (3.0 - 2.0 * edge_taper)
    pile *= edge_taper
    pile[pile < 0.015] = 0.0
    pile *= peak_height_m / max(float(pile.max()), 1e-12)
    pile[[0, -1], :] = 0.0
    pile[:, [0, -1]] = 0.0

    active = pile > 0.02
    indices = np.argwhere(active)
    low = indices.min(axis=0)
    high = indices.max(axis=0)
    edge_max = float(
        np.max(
            np.concatenate((pile[0], pile[-1], pile[:, 0], pile[:, -1])),
            initial=0.0,
        )
    )
    clearance_cells = min(
        int(low[0]), int(low[1]), size - 1 - int(high[0]), size - 1 - int(high[1])
    )
    info: dict[str, object] = {
        "seed": seed,
        "workspace_size_m": actual_workspace,
        "spacing_m": spacing_m,
        "shape": [size, size],
        "peak_height_m": float(pile.max()),
        "ridge_components": ridge_records,
        "old_bucket_scars": scar_records,
        "active_bbox_xy_m": [
            float(low[0] * spacing_m),
            float(low[1] * spacing_m),
            float(high[0] * spacing_m),
            float(high[1] * spacing_m),
        ],
        "minimum_clearance_to_boundary_m": float(clearance_cells * spacing_m),
        "boundary_max_height_m": edge_max,
        "closed_footprint": bool(edge_max == 0.0 and not np.any(active[[0, -1], :]) and not np.any(active[:, [0, -1]])),
    }
    return pile, info


class HeightfieldSoil:
    """Mutable height field cut by the measured sweep of a bucket floor."""

    def __init__(
        self,
        height_m: np.ndarray,
        spacing_m: float,
        *,
        origin_xy_m: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        height = np.asarray(height_m, dtype=np.float64)
        if height.ndim != 2 or min(height.shape) < 2:
            raise ValueError("height_m must be a 2-D array")
        if spacing_m <= 0 or np.any(height < 0) or not np.all(np.isfinite(height)):
            raise ValueError("invalid height field or spacing")
        self.height_m = height.copy()
        self.spacing_m = float(spacing_m)
        self.origin_xy_m = np.asarray(origin_xy_m, dtype=np.float64)
        self.previous_frame: BucketFrame | None = None
        self.event_log: list[dict[str, object]] = []
        self._changed_mask = np.zeros_like(height, dtype=bool)

    @property
    def cell_area_m2(self) -> float:
        return self.spacing_m**2

    @property
    def volume_m3(self) -> float:
        return float(self.height_m.sum() * self.cell_area_m2)

    def _rasterize_triangle(self, vertices: np.ndarray) -> tuple[float, np.ndarray]:
        xy = vertices[:, :2]
        determinant = (
            (xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0])
            + (xy[2, 0] - xy[1, 0]) * (xy[0, 1] - xy[2, 1])
        )
        changed = np.zeros_like(self.height_m, dtype=bool)
        if abs(float(determinant)) < 1e-10:
            return 0.0, changed

        minimum = np.floor((xy.min(axis=0) - self.origin_xy_m) / self.spacing_m).astype(int)
        maximum = np.ceil((xy.max(axis=0) - self.origin_xy_m) / self.spacing_m).astype(int)
        minimum = np.maximum(minimum, 0)
        maximum = np.minimum(maximum, np.asarray(self.height_m.shape) - 1)
        if np.any(maximum < minimum):
            return 0.0, changed

        ii = np.arange(minimum[0], maximum[0] + 1)
        jj = np.arange(minimum[1], maximum[1] + 1)
        gx, gy = np.meshgrid(
            self.origin_xy_m[0] + ii * self.spacing_m,
            self.origin_xy_m[1] + jj * self.spacing_m,
            indexing="ij",
        )
        w0 = (
            (xy[1, 1] - xy[2, 1]) * (gx - xy[2, 0])
            + (xy[2, 0] - xy[1, 0]) * (gy - xy[2, 1])
        ) / determinant
        w1 = (
            (xy[2, 1] - xy[0, 1]) * (gx - xy[2, 0])
            + (xy[0, 0] - xy[2, 0]) * (gy - xy[2, 1])
        ) / determinant
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8)
        target_z = np.maximum(
            w0 * vertices[0, 2] + w1 * vertices[1, 2] + w2 * vertices[2, 2],
            0.0,
        )
        view = self.height_m[np.ix_(ii, jj)]
        cut = inside & (view > target_z)
        if not np.any(cut):
            return 0.0, changed
        removed_height = np.where(cut, view - target_z, 0.0)
        view[cut] = target_z[cut]
        self.height_m[np.ix_(ii, jj)] = view
        local_changed = changed[np.ix_(ii, jj)]
        local_changed[cut] = True
        changed[np.ix_(ii, jj)] = local_changed
        return float(removed_height.sum() * self.cell_area_m2), changed

    def apply_bucket_frame(
        self,
        frame: BucketFrame,
        *,
        cutting_enabled: bool = True,
    ) -> DigResult:
        """Cut soil with current floor plus the swept cutting edge."""
        before_volume = self.volume_m3
        changed = np.zeros_like(self.height_m, dtype=bool)
        triangles = [
            frame.floor_corners_world_m[[0, 1, 2]],
            frame.floor_corners_world_m[[0, 2, 3]],
        ]
        if cutting_enabled and self.previous_frame is not None:
            previous = self.previous_frame
            # The edge curtain is the volume crossed by the real 3.2 m cutting
            # edge between physics frames. Curl/lift therefore changes the cut.
            curtain = np.asarray(
                [previous.edge_left, previous.edge_right, frame.edge_right, frame.edge_left]
            )
            triangles.extend((curtain[[0, 1, 2]], curtain[[0, 2, 3]]))

        if cutting_enabled:
            for triangle in triangles:
                _, triangle_changed = self._rasterize_triangle(triangle)
                changed |= triangle_changed

        self.previous_frame = frame
        self._changed_mask |= changed
        indices = np.argwhere(changed)
        bbox = None
        if len(indices):
            low = indices.min(axis=0)
            high = indices.max(axis=0)
            bbox = (int(low[0]), int(low[1]), int(high[0]), int(high[1]))
        result = DigResult(
            removed_volume_m3=max(0.0, before_volume - self.volume_m3),
            changed_cell_count=int(changed.sum()),
            changed_bbox_ij=bbox,
            minimum_cutting_edge_z_m=float(
                min(frame.edge_left[2], frame.edge_right[2])
            ),
        )
        if result.removed_volume_m3 > 0.0:
            self.event_log.append(
                {
                    "time_s": frame.time_s,
                    "joint_positions_rad": frame.joint_positions_rad,
                    "removed_volume_m3": result.removed_volume_m3,
                    "changed_cell_count": result.changed_cell_count,
                    "cutting_edge_center_xyz_m": (
                        0.5 * (frame.edge_left + frame.edge_right)
                    ).tolist(),
                    "cutting_edge_left_xyz_m": frame.edge_left.tolist(),
                    "cutting_edge_right_xyz_m": frame.edge_right.tolist(),
                }
            )
        return result

    def settle_changed_region(
        self,
        *,
        repose_angle_deg: float = 38.0,
        padding_m: float = 2.0,
    ) -> dict[str, object]:
        indices = np.argwhere(self._changed_mask)
        if not len(indices):
            return {"iterations": 0, "converged": True, "changed": False}
        pad = int(np.ceil(padding_m / self.spacing_m))
        low = np.maximum(indices.min(axis=0) - pad, 0)
        high = np.minimum(indices.max(axis=0) + pad + 1, self.height_m.shape)
        slices = (slice(int(low[0]), int(high[0])), slice(int(low[1]), int(high[1])))
        relaxed, stats = relax_critical_slope(
            self.height_m[slices],
            (self.spacing_m, self.spacing_m),
            critical_angle_deg=repose_angle_deg,
            max_iterations=3_000,
            tolerance=1e-4,
        )
        self.height_m[slices] = relaxed
        self._changed_mask.fill(False)
        return {
            "iterations": stats.iterations,
            "converged": stats.converged,
            "changed": True,
            "crop_ij": [int(low[0]), int(low[1]), int(high[0]), int(high[1])],
            "volume_error_m3": abs(stats.volume_after - stats.volume_before),
        }


def heightfield_mesh(height_m: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return static-topology mesh points and triangle indices."""
    height = np.asarray(height_m, dtype=np.float64)
    nx, ny = height.shape
    xx, yy = np.meshgrid(
        np.arange(nx) * spacing_m,
        np.arange(ny) * spacing_m,
        indexing="ij",
    )
    points = np.column_stack((xx.ravel(), yy.ravel(), height.ravel())).astype(np.float32)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = i * ny + j
    b = (i + 1) * ny + j
    c = (i + 1) * ny + (j + 1)
    d = i * ny + (j + 1)
    faces = np.stack((a, b, c, a, c, d), axis=-1).reshape(-1, 3).astype(np.int32)
    return points, faces


def _rotation_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_y(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def synthetic_bucket_transform(
    *,
    base_x_m: float,
    base_y_m: float,
    lift_deg: float,
    curl_deg: float,
    heading_deg: float = 90.0,
) -> np.ndarray:
    """Approximate loader kinematics used only by the offline dry run."""
    heading = np.deg2rad(heading_deg)
    rotation = _rotation_z(heading) @ _rotation_y(-np.deg2rad(curl_deg))
    forward = np.asarray([np.cos(heading), np.sin(heading), 0.0])
    center = (
        np.asarray([base_x_m, base_y_m, 0.0])
        + 4.15 * forward
        + np.asarray([0.0, 0.0, 0.42 + 0.038 * (lift_deg + 8.0)])
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    return transform


def _front_contact_y(height: np.ndarray, x_m: float, spacing_m: float) -> float:
    index = int(np.clip(round(x_m / spacing_m), 0, height.shape[0] - 1))
    active = np.flatnonzero(height[index] > 0.25)
    return float((active[0] if len(active) else np.argmax(height[index])) * spacing_m)


def run_offline_joint_coupled_demo(
    initial_height_m: np.ndarray,
    spacing_m: float,
    *,
    scoop_count: int = 6,
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    """Dry-run the same bucket sweep contract used by the Isaac adapter."""
    soil = HeightfieldSoil(initial_height_m, spacing_m)
    states = [soil.height_m.copy()]
    scoop_records: list[dict[str, object]] = []
    offsets = [0.0, -3.0, 3.0, -1.5, 1.5, 0.6]
    weights = initial_height_m.sum(axis=1)
    center_x = float(
        np.dot(np.arange(len(weights)) * spacing_m, weights)
        / max(float(weights.sum()), 1e-12)
    )
    time_s = 0.0

    for scoop_index in range(scoop_count):
        entry_x = center_x + offsets[scoop_index % len(offsets)]
        contact_y = _front_contact_y(soil.height_m, entry_x, spacing_m)
        base_start_y = contact_y - 5.65
        event_start = len(soil.event_log)
        volume_before = soil.volume_m3
        soil.previous_frame = None

        phases = [
            (np.linspace(base_start_y - 0.8, base_start_y, 8), -8.0, -5.0, False),
            (np.linspace(base_start_y, base_start_y + 3.2, 28), -8.0, -5.0, True),
            (np.full(22, base_start_y + 3.2), np.linspace(-8.0, 18.0, 22), np.linspace(-5.0, 44.0, 22), True),
        ]
        for base_values, lift_values, curl_values, enabled in phases:
            count = len(base_values)
            lifts = np.full(count, lift_values) if np.isscalar(lift_values) else lift_values
            curls = np.full(count, curl_values) if np.isscalar(curl_values) else curl_values
            for base_y, lift, curl in zip(base_values, lifts, curls):
                joints = {
                    "lift_joint": float(np.deg2rad(lift)),
                    "bucket_joint": float(np.deg2rad(curl)),
                }
                frame = frame_from_transform(
                    synthetic_bucket_transform(
                        base_x_m=entry_x,
                        base_y_m=float(base_y),
                        lift_deg=float(lift),
                        curl_deg=float(curl),
                    ),
                    joint_positions_rad=joints,
                    time_s=time_s,
                )
                soil.apply_bucket_frame(frame, cutting_enabled=enabled)
                time_s += 1.0 / 30.0

        settling = soil.settle_changed_region(repose_angle_deg=39.0, padding_m=2.2)
        removed = volume_before - soil.volume_m3
        if removed <= 0.0:
            raise RuntimeError(f"offline scoop {scoop_index + 1} removed no material")
        states.append(soil.height_m.copy())
        scoop_records.append(
            {
                "scoop_number": scoop_index + 1,
                "entry_x_m": entry_x,
                "front_contact_y_m": contact_y,
                "removed_volume_m3": removed,
                "joint_event_start": event_start,
                "joint_event_end": len(soil.event_log),
                "settling": settling,
            }
        )
    return np.stack(states), scoop_records, soil.event_log


def _save_previews(output_dir: Path, states: np.ndarray, spacing_m: float) -> None:
    extent = [0.0, (states.shape[1] - 1) * spacing_m, 0.0, (states.shape[2] - 1) * spacing_m]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.5), constrained_layout=True)
    vmax = float(states[0].max())
    for index, axis in enumerate(axes.flat[:7]):
        image = axis.imshow(states[index].T, origin="lower", extent=extent, cmap="terrain", vmin=0, vmax=vmax)
        axis.set(title=f"H{index}", xlabel="x [m]", ylabel="y [m]")
        fig.colorbar(image, ax=axis, shrink=0.75)
    delta = states[-1] - states[0]
    image = axes.flat[-1].imshow(delta.T, origin="lower", extent=extent, cmap="RdBu_r")
    axes.flat[-1].set(title="H6 - H0", xlabel="x [m]", ylabel="y [m]")
    fig.colorbar(image, ax=axes.flat[-1], shrink=0.75)
    fig.savefig(output_dir / "joint_coupled_heightmaps.png", dpi=150)
    plt.close(fig)

    stride = max(1, int(round(0.25 / spacing_m)))
    coords = np.arange(states.shape[1]) * spacing_m
    xx, yy = np.meshgrid(coords[::stride], coords[::stride], indexing="ij")
    fig = plt.figure(figsize=(13, 5.8), constrained_layout=True)
    for panel, index in enumerate((0, 6), 1):
        axis = fig.add_subplot(1, 2, panel, projection="3d")
        axis.plot_surface(xx, yy, states[index, ::stride, ::stride], cmap="terrain", linewidth=0, antialiased=False)
        axis.set(title=f"H{index}: {'initial' if index == 0 else 'after six bucket sweeps'}", xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        axis.view_init(elev=24, azim=-125)
        axis.set_box_aspect((1.0, 1.0, 0.42))
    fig.savefig(output_dir / "realistic_closed_pile_surface.png", dpi=150)
    plt.close(fig)


def generate_demo(output_dir: Path, *, seed: int = 20260806) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    initial, pile_info = generate_realistic_closed_pile(seed=seed)
    spacing_m = float(pile_info["spacing_m"])
    states, scoops, events = run_offline_joint_coupled_demo(initial, spacing_m)
    points, faces = heightfield_mesh(states[0], spacing_m)
    np.savez_compressed(
        output_dir / "interactive_pile_demo.npz",
        heightmaps_m=states.astype(np.float32),
        grid_spacing_m=np.float32(spacing_m),
        origin_xy_m=np.asarray([0.0, 0.0], dtype=np.float32),
        mesh_face_vertex_indices=faces,
        mesh_initial_points_m=points,
        active_masks=states > 0.02,
    )
    manifest: dict[str, object] = {
        "description": "Closed irregular pile driven by the swept geometry of a 3.2 m loader bucket",
        "pile": pile_info,
        "state_shape": list(states.shape),
        "bucket_geometry": BucketGeometry().__dict__,
        "scoops": scoops,
        "joint_coupled_event_count": len(events),
        "joint_coupled_events": events,
        "boundary_max_height_m": float(
            np.max(np.concatenate((states[:, 0, :].ravel(), states[:, -1, :].ravel(), states[:, :, 0].ravel(), states[:, :, -1].ravel())), initial=0.0)
        ),
        "all_states_closed": bool(
            np.all(states[:, 0, :] == 0.0)
            and np.all(states[:, -1, :] == 0.0)
            and np.all(states[:, :, 0] == 0.0)
            and np.all(states[:, :, -1] == 0.0)
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _save_previews(output_dir, states, spacing_m)
    return manifest


def main() -> None:
    print(
        "DEPRECATED: bucket_soil_coupling.py is an offline legacy prototype; "
        "use isaac_loader/interactive_dig_demo.py for Isaac Sim.",
        file=sys.stderr,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("isaac_interactive_pile_demo"))
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    manifest = generate_demo(args.output_dir, seed=args.seed)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "state_shape": manifest["state_shape"], "all_states_closed": manifest["all_states_closed"], "events": manifest["joint_coupled_event_count"]}, indent=2))


if __name__ == "__main__":
    main()
