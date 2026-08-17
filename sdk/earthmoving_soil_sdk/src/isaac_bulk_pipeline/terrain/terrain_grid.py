"""Terrain-grid geometry and coordinate transforms.

All public distances are metres. Height maps use ``H[y, x]``: array axis 0 is
the row along terrain-local +Y and axis 1 is the column along terrain-local +X.
Transforms use the conventional NumPy column-vector form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _point3(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(
            f"[TerrainGrid] {name} must have shape (3,) with finite metre values; "
            f"received shape={point.shape}, value={point!r}"
        )
    return point


@dataclass
class TerrainGrid:
    """Map an authoritative ``H[y, x]`` array to terrain and world coordinates.

    ``terrain_to_world_matrix`` is named differently from the required
    :meth:`terrain_to_world` method to avoid a field/method name collision. It
    represents ``T_world_from_terrain`` and uses column-vector multiplication.
    """

    nx: int
    ny: int
    dx: float
    dy: float
    origin_x: float
    origin_y: float
    terrain_prim_path: str
    terrain_to_world_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    valid_mask: np.ndarray | None = None
    _world_to_terrain_matrix: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise TypeError("[TerrainGrid] nx and ny must be integers")
        if self.nx < 2 or self.ny < 2:
            raise ValueError(
                f"[TerrainGrid] nx and ny must be >= 2; nx={self.nx}, ny={self.ny}"
            )
        for name, value in (
            ("dx", self.dx),
            ("dy", self.dy),
            ("origin_x", self.origin_x),
            ("origin_y", self.origin_y),
        ):
            if not np.isfinite(value):
                raise ValueError(f"[TerrainGrid] {name} must be finite; value={value}")
        if self.dx <= 0.0 or self.dy <= 0.0:
            raise ValueError(
                f"[TerrainGrid] cell spacing must be positive metres; "
                f"dx={self.dx}, dy={self.dy}"
            )
        if not self.terrain_prim_path.startswith("/"):
            raise ValueError(
                "[TerrainGrid] terrain_prim_path must be an absolute USD Prim path; "
                f"value={self.terrain_prim_path!r}"
            )

        transform = np.asarray(self.terrain_to_world_matrix, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(
                "[TerrainGrid] terrain_to_world_matrix must be a finite (4, 4) "
                f"array; shape={transform.shape}"
            )
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
            raise ValueError(
                "[TerrainGrid] terrain_to_world_matrix must be affine using the "
                f"column-vector convention; last_row={transform[3].tolist()}"
            )
        try:
            inverse = np.linalg.inv(transform)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "[TerrainGrid] terrain_to_world_matrix is singular"
            ) from exc
        self.terrain_to_world_matrix = np.ascontiguousarray(transform.copy())
        self._world_to_terrain_matrix = np.ascontiguousarray(inverse)

        if self.valid_mask is not None:
            mask = np.asarray(self.valid_mask, dtype=bool)
            if mask.shape != self.shape:
                raise ValueError(
                    "[TerrainGrid] valid_mask shape must equal (ny, nx); "
                    f"expected={self.shape}, received={mask.shape}"
                )
            self.valid_mask = np.ascontiguousarray(mask.copy())

    @property
    def shape(self) -> tuple[int, int]:
        """Expected authoritative height-map shape as ``(ny, nx)``."""

        return self.ny, self.nx

    @property
    def cell_area(self) -> float:
        """Horizontal area represented by one height sample, in m²."""

        return float(self.dx * self.dy)

    def world_to_terrain(self, point_world: Sequence[float] | np.ndarray) -> np.ndarray:
        """Transform one XYZ point from world space to terrain-local metres."""

        return self._transform_point(
            self._world_to_terrain_matrix,
            _point3(point_world, name="point_world"),
        )

    def terrain_to_world(self, point_terrain: Sequence[float] | np.ndarray) -> np.ndarray:
        """Transform one XYZ point from terrain-local space to world metres."""

        return self._transform_point(
            self.terrain_to_world_matrix,
            _point3(point_terrain, name="point_terrain"),
        )

    def terrain_to_grid(
        self, point_terrain: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        """Return continuous ``(row_y, column_x)`` indices for a terrain point."""

        point = _point3(point_terrain, name="point_terrain")
        row = (point[1] - self.origin_y) / self.dy
        column = (point[0] - self.origin_x) / self.dx
        return np.asarray([row, column], dtype=np.float64)

    def grid_to_terrain(self, row: float, column: float, height: float) -> np.ndarray:
        """Map a possibly fractional ``(row_y, column_x, height)`` to XYZ metres."""

        values = np.asarray([row, column, height], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "[TerrainGrid] row, column and height must be finite; "
                f"values={values.tolist()}"
            )
        return np.asarray(
            [
                self.origin_x + column * self.dx,
                self.origin_y + row * self.dy,
                height,
            ],
            dtype=np.float64,
        )

    def is_inside(self, row: int, column: int) -> bool:
        """Return whether an integer cell index lies inside the valid terrain."""

        if not isinstance(row, (int, np.integer)) or not isinstance(
            column, (int, np.integer)
        ):
            raise TypeError("[TerrainGrid] row and column must be integer indices")
        inside = 0 <= row < self.ny and 0 <= column < self.nx
        if inside and self.valid_mask is not None:
            inside = bool(self.valid_mask[row, column])
        return bool(inside)

    def compute_volume(self, heightmap: np.ndarray) -> float:
        """Compute geometric terrain volume in m³ over valid cells."""

        height = self.validate_heightmap(heightmap)
        values = height if self.valid_mask is None else height[self.valid_mask]
        return float(values.sum(dtype=np.float64) * self.cell_area)

    def validate_heightmap(self, heightmap: np.ndarray) -> np.ndarray:
        """Validate ``H[y, x]`` shape, units and numeric domain without copying."""

        height = np.asarray(heightmap)
        if height.shape != self.shape:
            raise ValueError(
                "[TerrainGrid] heightmap must use H[y,x] with shape (ny,nx); "
                f"expected={self.shape}, received={height.shape}, "
                f"prim_path={self.terrain_prim_path}"
            )
        if not np.issubdtype(height.dtype, np.number):
            raise TypeError(
                f"[TerrainGrid] heightmap dtype must be numeric; dtype={height.dtype}"
            )
        if not np.all(np.isfinite(height)):
            raise ValueError(
                f"[TerrainGrid] heightmap contains NaN or Inf; shape={height.shape}"
            )
        if np.any(height < 0.0):
            minimum = float(np.min(height))
            raise ValueError(
                "[TerrainGrid] heightmap contains negative metre heights; "
                f"minimum={minimum}, shape={height.shape}"
            )
        return height

    @staticmethod
    def _transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
        homogeneous = np.concatenate((point, np.ones(1, dtype=np.float64)))
        transformed = matrix @ homogeneous
        if abs(float(transformed[3])) < 1e-12:
            raise ValueError("[TerrainGrid] point transform produced zero homogeneous w")
        return np.asarray(transformed[:3] / transformed[3], dtype=np.float64)
