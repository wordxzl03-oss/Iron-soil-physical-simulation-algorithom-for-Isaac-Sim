"""Planner-only cost maps derived from the authoritative height field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain import TerrainGrid


@dataclass(frozen=True)
class TerrainCostConfig:
    maximum_slope_deg: float = 25.0
    maximum_roughness_m: float = 0.20
    slope_weight: float = 2.0
    roughness_weight: float = 1.0


@dataclass(frozen=True)
class PlannerTerrainView:
    grid: TerrainGrid
    slope_rad: np.ndarray
    roughness_m: np.ndarray
    traversable: np.ndarray
    cost: np.ndarray

    @classmethod
    def derive(cls, grid: TerrainGrid, H_resting_m: np.ndarray, config: TerrainCostConfig | None = None) -> "PlannerTerrainView":
        cfg = config or TerrainCostConfig()
        height = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        gy, gx = np.gradient(height, grid.dy, grid.dx)
        slope = np.arctan(np.hypot(gx, gy))
        padded = np.pad(height, 1, mode="edge")
        neighbours = np.stack(
            [padded[0:-2, 0:-2], padded[0:-2, 1:-1], padded[0:-2, 2:], padded[1:-1, 0:-2], padded[1:-1, 1:-1], padded[1:-1, 2:], padded[2:, 0:-2], padded[2:, 1:-1], padded[2:, 2:]],
            axis=0,
        )
        roughness = np.std(neighbours, axis=0)
        traversable = (slope <= np.deg2rad(cfg.maximum_slope_deg)) & (roughness <= cfg.maximum_roughness_m)
        if grid.valid_mask is not None:
            traversable &= grid.valid_mask
        cost = 1.0 + cfg.slope_weight * slope / max(np.deg2rad(cfg.maximum_slope_deg), 1e-9) + cfg.roughness_weight * roughness / max(cfg.maximum_roughness_m, 1e-9)
        cost = np.where(traversable, cost, np.inf)
        for value in (slope, roughness, traversable, cost):
            value.setflags(write=False)
        return cls(grid, slope, roughness, traversable, cost)

    def sample(self, x_m: float, y_m: float) -> tuple[bool, float, float]:
        rc = self.grid.terrain_to_grid(np.array([x_m, y_m, 0.0]))
        row, col = int(round(rc[0])), int(round(rc[1]))
        if not self.grid.is_inside(row, col):
            return False, float("inf"), float("inf")
        return bool(self.traversable[row, col]), float(self.cost[row, col]), float(self.slope_rad[row, col])
