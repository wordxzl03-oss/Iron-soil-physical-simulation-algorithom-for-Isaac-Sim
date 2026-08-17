"""Deterministic planar-slope fixtures for 0/10/20 degree contact tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain.terrain_grid import TerrainGrid


@dataclass(frozen=True)
class SlopePatchSpec:
    slope_deg: float
    uphill_direction_xy: tuple[float, float] = (1.0, 0.0)
    minimum_height_m: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope_deg) or not 0.0 <= self.slope_deg < 45.0:
            raise ValueError("[SlopePatch] slope_deg must be finite and in [0,45)")
        direction = np.asarray(self.uphill_direction_xy, dtype=np.float64)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)):
            raise ValueError("[SlopePatch] uphill_direction_xy must contain two finite values")
        if float(np.linalg.norm(direction)) <= 1e-12:
            raise ValueError("[SlopePatch] uphill_direction_xy must be nonzero")
        if not np.isfinite(self.minimum_height_m) or self.minimum_height_m < 0.0:
            raise ValueError("[SlopePatch] minimum_height_m must be non-negative")


def build_planar_slope_heightmap(grid: TerrainGrid, spec: SlopePatchSpec) -> np.ndarray:
    """Return a non-negative vertex height field with the requested plane slope."""

    direction = np.asarray(spec.uphill_direction_xy, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    x = grid.origin_x + np.arange(grid.nx, dtype=np.float64) * grid.dx
    y = grid.origin_y + np.arange(grid.ny, dtype=np.float64) * grid.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    projected = xx * direction[0] + yy * direction[1]
    projected -= float(np.min(projected))
    height = spec.minimum_height_m + np.tan(np.deg2rad(spec.slope_deg)) * projected
    return np.ascontiguousarray(height, dtype=np.float64)


def estimate_slope_deg(heightmap: np.ndarray, grid: TerrainGrid) -> np.ndarray:
    """Return local steepest slope angle in degrees for validation/telemetry."""

    height = grid.validate_heightmap(heightmap)
    derivative_y, derivative_x = np.gradient(height, grid.dy, grid.dx)
    return np.rad2deg(np.arctan(np.hypot(derivative_x, derivative_y)))
