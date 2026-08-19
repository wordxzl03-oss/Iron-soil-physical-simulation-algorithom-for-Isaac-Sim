"""Geometry and conservative accounting helpers for the bucket-local 3D prototype.

This module is deliberately Isaac-independent so its geometry and audit logic can
be regression-tested with ordinary Python. It does not implement granular
dynamics; the Isaac/PhysX runtime owns the particle motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BucketCavityGeometry:
    """Extruded bucket cavity derived from the audited 390F descriptor."""

    half_width_m: float
    profile_yz_m: np.ndarray

    @classmethod
    def from_descriptor(cls, descriptor_path: str | Path) -> "BucketCavityGeometry":
        data = json.loads(Path(descriptor_path).read_text(encoding="utf-8"))
        profile = np.asarray(data["interior_profile_local"], dtype=np.float64)
        if profile.ndim != 2 or profile.shape[1] != 3 or profile.shape[0] < 3:
            raise ValueError("interior_profile_local must contain >=3 xyz points")
        width = float(data["nominal_width_m"])
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("nominal_width_m must be positive")
        yz = profile[:, 1:3]
        keep = np.ones(len(yz), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(yz, axis=0), axis=1) > 1.0e-12
        yz = yz[keep]
        if len(yz) < 3:
            raise ValueError("bucket profile degenerates after duplicate removal")
        return cls(half_width_m=0.5 * width, profile_yz_m=yz)

    def contains_local(self, points_local_m: np.ndarray, *, wall_margin_m: float = 0.0) -> np.ndarray:
        """Return mask for particles inside the extruded CAD cavity."""

        points = np.asarray(points_local_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_local_m must have shape (N, 3)")
        half_width = self.half_width_m - float(wall_margin_m)
        if half_width <= 0.0:
            raise ValueError("wall_margin_m removes the entire cavity width")
        in_width = np.abs(points[:, 0]) <= half_width + 1.0e-12
        in_profile = _points_in_polygon(points[:, 1:3], self.profile_yz_m)
        return in_width & in_profile


def _points_in_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    """Vectorized even-odd rule including polygon-boundary points."""

    points = np.asarray(points_xy, dtype=np.float64)
    polygon = np.asarray(polygon_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("polygon_xy must have shape (M>=3, 2)")

    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    on_boundary = np.zeros(len(points), dtype=bool)
    eps = 1.0e-10

    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        dx = xj - xi
        dy = yj - yi

        cross = (x - xi) * dy - (y - yi) * dx
        dot = (x - xi) * (x - xj) + (y - yi) * (y - yj)
        on_boundary |= (np.abs(cross) <= eps * max(1.0, abs(dx) + abs(dy))) & (dot <= eps)

        intersects = ((yi > y) != (yj > y))
        x_intersection = (xj - xi) * (y - yi) / (yj - yi + np.finfo(np.float64).eps) + xi
        inside ^= intersects & (x < x_intersection)
        j = i
    return inside | on_boundary


def transform_points(matrix_world_from_local: np.ndarray, points_local_m: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix_world_from_local, dtype=np.float64)
    points = np.asarray(points_local_m, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("matrix_world_from_local must have shape (4, 4)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_local_m must have shape (N, 3)")
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (matrix @ homogeneous.T).T[:, :3]


def inverse_transform_points(matrix_world_from_local: np.ndarray, points_world_m: np.ndarray) -> np.ndarray:
    return transform_points(np.linalg.inv(np.asarray(matrix_world_from_local, dtype=np.float64)), points_world_m)


def sample_rectangular_lattice(
    minimum_m: Iterable[float],
    maximum_m: Iterable[float],
    spacing_m: float,
) -> np.ndarray:
    """Sample cell-centred pseudo-particles in an axis-aligned local box."""

    minimum = np.asarray(tuple(minimum_m), dtype=np.float64)
    maximum = np.asarray(tuple(maximum_m), dtype=np.float64)
    spacing = float(spacing_m)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("minimum_m and maximum_m must each contain three values")
    if spacing <= 0.0:
        raise ValueError("spacing_m must be positive")
    if np.any(maximum <= minimum):
        raise ValueError("maximum_m must be strictly greater than minimum_m")

    axes = []
    for lo, hi in zip(minimum, maximum):
        count = int(np.floor((hi - lo) / spacing))
        if count < 1:
            raise ValueError("seed box is thinner than one particle spacing")
        axes.append(lo + spacing * (0.5 + np.arange(count, dtype=np.float64)))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


@dataclass(frozen=True)
class ParticleVolumeAccounting:
    """Bulk-volume accounting for equal-volume pseudo-particles."""

    represented_bulk_volume_per_particle_m3: float

    def volume_from_mask(self, mask: np.ndarray) -> float:
        values = np.asarray(mask, dtype=bool)
        return float(np.count_nonzero(values) * self.represented_bulk_volume_per_particle_m3)

    def volume_from_count(self, count: int) -> float:
        if int(count) < 0:
            raise ValueError("count must be non-negative")
        return float(int(count) * self.represented_bulk_volume_per_particle_m3)
