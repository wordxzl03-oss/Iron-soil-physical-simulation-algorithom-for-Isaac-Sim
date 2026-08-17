"""Dirty-region triangle-mesh data backend for deforming terrain contact."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..performance import GridWindow
from ..terrain import TerrainGrid
from ..visualization.dynamic_mesh_adapter import build_mesh_arrays


@dataclass(frozen=True)
class ContactChunkSnapshot:
    key: tuple[int, int]
    window: GridWindow
    points_m: np.ndarray
    face_counts: np.ndarray
    face_indices: np.ndarray
    revision: int


@dataclass(frozen=True)
class ChunkedContactUpdate:
    elapsed_ms: float
    updated_keys: tuple[tuple[int, int], ...]
    updated_vertex_count: int
    revision: int


class ChunkedContactMeshBackend:
    """Maintain exact 0.05 m contact chunks and update only dirty vertices."""

    def __init__(self, chunk_size_cells: int = 64) -> None:
        if not isinstance(chunk_size_cells, int) or chunk_size_cells < 8:
            raise ValueError("[ChunkedContact] chunk_size_cells must be >= 8")
        self.chunk_size_cells = chunk_size_cells
        self._grid: TerrainGrid | None = None
        self._chunks: dict[tuple[int, int], dict[str, object]] = {}
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def grid(self) -> TerrainGrid:
        if self._grid is None:
            raise RuntimeError("[ChunkedContact] initialize must be called first")
        return self._grid

    def initialize(self, grid: TerrainGrid, heightmap: np.ndarray) -> None:
        if self._chunks:
            raise RuntimeError("[ChunkedContact] already initialized")
        height = grid.validate_heightmap(heightmap)
        self._grid = grid
        for key, window in self._windows(grid.shape).items():
            local = self._local_grid(grid, window)
            points, counts, indices = build_mesh_arrays(height[window.slices], local)
            self._chunks[key] = {
                "window": window,
                "points": points,
                "counts": counts,
                "indices": indices,
                "revision": 0,
            }

    def update(
        self,
        heightmap: np.ndarray,
        *,
        changed_mask: np.ndarray | None = None,
        affected_bbox: tuple[int, int, int, int] | None = None,
    ) -> ChunkedContactUpdate:
        if self._grid is None:
            raise RuntimeError("[ChunkedContact] initialize must be called first")
        height = self._grid.validate_heightmap(heightmap)
        dirty = self.dirty_keys(changed_mask=changed_mask, affected_bbox=affected_bbox)
        start = perf_counter()
        vertices = 0
        if dirty:
            self._revision += 1
        for key in dirty:
            record = self._chunks[key]
            window = record["window"]
            points = record["points"]
            points[:, 2] = height[window.slices].ravel().astype(np.float32, copy=False)
            record["revision"] = self._revision
            vertices += int(points.shape[0])
        return ChunkedContactUpdate(
            elapsed_ms=(perf_counter() - start) * 1_000.0,
            updated_keys=dirty,
            updated_vertex_count=vertices,
            revision=self._revision,
        )

    def update_tile_samples(
        self,
        samples_by_key: dict[tuple[int, int], np.ndarray],
    ) -> ChunkedContactUpdate:
        """Update selected fixed-topology chunks from compact device samples.

        This is the contact equivalent of a dirty-tile publication.  It avoids
        constructing a temporary full 701×701 surface merely to touch a few
        PhysX chunks.
        """

        if self._grid is None:
            raise RuntimeError("[ChunkedContact] initialize must be called first")
        selected = tuple(sorted(samples_by_key))
        start = perf_counter()
        if selected:
            self._revision += 1
        vertices = 0
        for key in selected:
            if key not in self._chunks:
                raise ValueError("[ChunkedContact] unknown tile key")
            record = self._chunks[key]
            window = record["window"]
            sample = np.asarray(samples_by_key[key], dtype=np.float64)
            if sample.shape != window.shape:
                raise ValueError(
                    "[ChunkedContact] device tile sample shape does not match contact chunk"
                )
            points = record["points"]
            points[:, 2] = sample.ravel().astype(np.float32, copy=False)
            record["revision"] = self._revision
            vertices += int(points.shape[0])
        return ChunkedContactUpdate(
            elapsed_ms=(perf_counter() - start) * 1_000.0,
            updated_keys=selected,
            updated_vertex_count=vertices,
            revision=self._revision,
        )

    def dirty_keys(
        self,
        *,
        changed_mask: np.ndarray | None = None,
        affected_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        if self._grid is None:
            raise RuntimeError("[ChunkedContact] initialize must be called first")
        if changed_mask is None and affected_bbox is None:
            return tuple(sorted(self._chunks))
        if changed_mask is not None:
            changed = np.asarray(changed_mask, dtype=bool)
            if changed.shape != self._grid.shape:
                raise ValueError("[ChunkedContact] changed mask shape mismatch")
            rows, cols = np.nonzero(changed)
            if rows.size == 0:
                return ()
            event = GridWindow(
                int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1
            )
        else:
            assert affected_bbox is not None
            event = GridWindow(*affected_bbox)
        return tuple(
            key
            for key, record in sorted(self._chunks.items())
            if self._intersects(record["window"], event)
        )

    def snapshot(self, key: tuple[int, int]) -> ContactChunkSnapshot:
        record = self._chunks[key]
        return ContactChunkSnapshot(
            key=key,
            window=record["window"],
            points_m=np.array(record["points"], copy=True),
            face_counts=np.array(record["counts"], copy=True),
            face_indices=np.array(record["indices"], copy=True),
            revision=int(record["revision"]),
        )

    def snapshots(self) -> tuple[ContactChunkSnapshot, ...]:
        return tuple(self.snapshot(key) for key in sorted(self._chunks))

    def _windows(self, shape: tuple[int, int]) -> dict[tuple[int, int], GridWindow]:
        ny, nx = shape
        result = {}
        for tile_row, row_start in enumerate(range(0, ny - 1, self.chunk_size_cells)):
            row_stop = min(ny, row_start + self.chunk_size_cells + 1)
            for tile_col, col_start in enumerate(range(0, nx - 1, self.chunk_size_cells)):
                col_stop = min(nx, col_start + self.chunk_size_cells + 1)
                result[(tile_row, tile_col)] = GridWindow(
                    row_start, col_start, row_stop, col_stop
                )
        return result

    @staticmethod
    def _local_grid(grid: TerrainGrid, window: GridWindow) -> TerrainGrid:
        return TerrainGrid(
            window.shape[1],
            window.shape[0],
            grid.dx,
            grid.dy,
            grid.origin_x + window.col_start * grid.dx,
            grid.origin_y + window.row_start * grid.dy,
            "/Local/ContactChunk",
        )

    @staticmethod
    def _intersects(first: GridWindow, second: GridWindow) -> bool:
        return not (
            first.row_stop <= second.row_start
            or second.row_stop <= first.row_start
            or first.col_stop <= second.col_start
            or second.col_stop <= first.col_start
        )
