"""Rasterization and support observations for real left/right track bodies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..terrain import TerrainGrid


@dataclass(frozen=True)
class TrackFootprintConfig:
    """Measured 390F track envelope plus reduced-order belt settings."""

    length_m: float = 6.17
    width_m: float = 0.80
    nominal_belt_speed_m_s: float = 1.20
    contact_gap_m: float = 0.08
    provenance: str = "CAD_TRACK_BOUNDS_AND_UNCALIBRATED_BELT_SPEED"

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.length_m, self.width_m, self.nominal_belt_speed_m_s, self.contact_gap_m],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[TrackFootprint] dimensions/speed/gap must be finite/positive")


@dataclass(frozen=True)
class TrackFootprint:
    mask: np.ndarray
    active_bbox_grid: tuple[int, int, int, int]
    center_terrain_m: np.ndarray
    forward_terrain_xy: np.ndarray

    def __post_init__(self) -> None:
        mask = np.array(self.mask, dtype=bool, copy=True, order="C")
        center = np.asarray(self.center_terrain_m, dtype=np.float64)
        forward = np.asarray(self.forward_terrain_xy, dtype=np.float64)
        if mask.ndim != 2 or center.shape != (3,) or forward.shape != (2,):
            raise ValueError("[TrackFootprint] invalid result shape")
        mask.setflags(write=False)
        center = np.ascontiguousarray(center.copy()); center.setflags(write=False)
        forward = np.ascontiguousarray(forward.copy()); forward.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "center_terrain_m", center)
        object.__setattr__(self, "forward_terrain_xy", forward)


class TrackFootprintRasterizer:
    """Project an oriented world-space track rectangle onto ``H[y,x]``."""

    def __init__(self, grid: TerrainGrid, config: TrackFootprintConfig | None = None) -> None:
        self.grid = grid
        self.config = config or TrackFootprintConfig()

    def rasterize(
        self,
        center_world_m: np.ndarray,
        forward_world: np.ndarray,
    ) -> TrackFootprint:
        center_world = np.asarray(center_world_m, dtype=np.float64)
        forward_world = np.asarray(forward_world, dtype=np.float64)
        if center_world.shape != (3,) or forward_world.shape != (3,):
            raise ValueError("[TrackFootprint] center/forward must have shape (3,)")
        if not np.all(np.isfinite(center_world)) or not np.all(np.isfinite(forward_world)):
            raise ValueError("[TrackFootprint] center/forward must be finite")
        center = self.grid.world_to_terrain(center_world)
        linear_world_from_terrain = self.grid.terrain_to_world_matrix[:3, :3]
        direction_terrain = np.linalg.solve(linear_world_from_terrain, forward_world)
        forward = direction_terrain[:2]
        norm = float(np.linalg.norm(forward))
        if norm <= 1.0e-12:
            raise ValueError("[TrackFootprint] projected forward direction is degenerate")
        forward /= norm
        lateral = np.asarray([-forward[1], forward[0]])
        half_length = 0.5 * self.config.length_m
        half_width = 0.5 * self.config.width_m
        extent = np.abs(forward) * half_length + np.abs(lateral) * half_width

        col_start = max(0, int(np.floor((center[0] - extent[0] - self.grid.origin_x) / self.grid.dx)))
        col_stop = min(self.grid.nx, int(np.ceil((center[0] + extent[0] - self.grid.origin_x) / self.grid.dx)) + 1)
        row_start = max(0, int(np.floor((center[1] - extent[1] - self.grid.origin_y) / self.grid.dy)))
        row_stop = min(self.grid.ny, int(np.ceil((center[1] + extent[1] - self.grid.origin_y) / self.grid.dy)) + 1)
        mask = np.zeros(self.grid.shape, dtype=bool)
        if row_start < row_stop and col_start < col_stop:
            rows = self.grid.origin_y + np.arange(row_start, row_stop) * self.grid.dy
            cols = self.grid.origin_x + np.arange(col_start, col_stop) * self.grid.dx
            xx, yy = np.meshgrid(cols, rows, indexing="xy")
            dx = xx - center[0]
            dy = yy - center[1]
            along = dx * forward[0] + dy * forward[1]
            across = dx * lateral[0] + dy * lateral[1]
            local = (np.abs(along) <= half_length) & (np.abs(across) <= half_width)
            if self.grid.valid_mask is not None:
                local &= self.grid.valid_mask[row_start:row_stop, col_start:col_stop]
            mask[row_start:row_stop, col_start:col_stop] = local
        rows_active, cols_active = np.nonzero(mask)
        bbox = (
            (0, 0, 0, 0)
            if rows_active.size == 0
            else (
                int(rows_active.min()), int(cols_active.min()),
                int(rows_active.max()) + 1, int(cols_active.max()) + 1,
            )
        )
        return TrackFootprint(mask, bbox, center, forward)

