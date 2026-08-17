"""Dirty-chunk visual terrain adapter for the V2 interactive runtime."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from ..config import MeshConfig
from ..performance import GridWindow
from ..terrain import TerrainGrid
from .dynamic_mesh_adapter import DynamicMeshAdapter, MeshUpdateMetrics


@dataclass(frozen=True)
class ChunkedMeshUpdateMetrics:
    elapsed_ms: float
    dirty_chunk_count: int
    total_chunk_count: int
    updated_vertex_count: int
    chunk_metrics: tuple[MeshUpdateMetrics, ...]


class ChunkedDynamicMeshAdapter:
    """Represent a full heightmap as independently updateable visual chunks."""

    def __init__(self, chunk_size_cells: int = 64) -> None:
        if not isinstance(chunk_size_cells, int) or chunk_size_cells < 8:
            raise ValueError("[ChunkedMesh] chunk_size_cells must be >= 8")
        self.chunk_size_cells = chunk_size_cells
        self._chunks: dict[tuple[int, int], tuple[GridWindow, DynamicMeshAdapter]] = {}
        self._grid: TerrainGrid | None = None

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def initialize(
        self,
        stage: Any,
        grid: TerrainGrid,
        config: MeshConfig,
        heightmap: np.ndarray,
        *,
        root_prim_path: str = "/World/Terrain/VisualChunks",
    ) -> None:
        if self._chunks:
            raise RuntimeError("[ChunkedMesh] already initialized")
        height = grid.validate_heightmap(heightmap)
        self._grid = grid
        for key, window in self._chunk_windows(grid.shape).items():
            local_grid = self._local_grid(
                grid, window, f"{root_prim_path}/chunk_{key[0]:02d}_{key[1]:02d}"
            )
            adapter = DynamicMeshAdapter()
            adapter.initialize(stage, local_grid, config, height[window.slices])
            self._chunks[key] = (window, adapter)

    def update(
        self,
        heightmap: np.ndarray,
        *,
        changed_mask: np.ndarray | None = None,
        affected_bbox: tuple[int, int, int, int] | None = None,
    ) -> ChunkedMeshUpdateMetrics:
        if self._grid is None:
            raise RuntimeError("[ChunkedMesh] initialize must be called first")
        height = self._grid.validate_heightmap(heightmap)
        dirty = self.dirty_chunk_indices(changed_mask=changed_mask, affected_bbox=affected_bbox)
        start = perf_counter()
        metrics = tuple(
            self._chunks[key][1].update(height[self._chunks[key][0].slices])
            for key in dirty
        )
        return ChunkedMeshUpdateMetrics(
            elapsed_ms=(perf_counter() - start) * 1_000.0,
            dirty_chunk_count=len(dirty),
            total_chunk_count=len(self._chunks),
            updated_vertex_count=sum(item.vertex_count for item in metrics),
            chunk_metrics=metrics,
        )

    def reset(self, heightmap: np.ndarray) -> ChunkedMeshUpdateMetrics:
        return self.update(heightmap)

    def update_tile_samples(
        self,
        samples_by_tile_id: dict[int, np.ndarray],
        *,
        tile_size_cells: int,
    ) -> ChunkedMeshUpdateMetrics:
        """Publish device-originated dirty tiles without a full host heightmap.

        ``DeviceBulkState`` tiles contain ``tile_size_cells`` terrain cells and
        the final shared vertex row/column.  That matches this adapter's chunk
        topology exactly when both tile sizes agree.
        """

        if self._grid is None:
            raise RuntimeError("[ChunkedMesh] initialize must be called first")
        if int(tile_size_cells) != self.chunk_size_cells:
            raise ValueError("[ChunkedMesh] device/visual tile size mismatch")
        tiles_x = (self._grid.shape[1] - 1 + tile_size_cells - 1) // tile_size_cells
        start = perf_counter()
        metrics: list[MeshUpdateMetrics] = []
        for tile_id, sample in sorted(samples_by_tile_id.items()):
            key = divmod(int(tile_id), tiles_x)
            record = self._chunks.get(key)
            if record is None:
                raise ValueError("[ChunkedMesh] device tile ID outside visual chunks")
            window, adapter = record
            local = np.asarray(sample, dtype=np.float64)
            if local.shape != window.shape:
                raise ValueError(
                    "[ChunkedMesh] device tile sample shape does not match visual chunk"
                )
            metrics.append(adapter.update(local))
        return ChunkedMeshUpdateMetrics(
            elapsed_ms=(perf_counter() - start) * 1_000.0,
            dirty_chunk_count=len(metrics),
            total_chunk_count=len(self._chunks),
            updated_vertex_count=sum(item.vertex_count for item in metrics),
            chunk_metrics=tuple(metrics),
        )

    def dirty_chunk_indices(
        self,
        *,
        changed_mask: np.ndarray | None = None,
        affected_bbox: tuple[int, int, int, int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        if self._grid is None:
            raise RuntimeError("[ChunkedMesh] initialize must be called first")
        if changed_mask is None and affected_bbox is None:
            return tuple(sorted(self._chunks))
        if changed_mask is not None:
            changed = np.asarray(changed_mask, dtype=bool)
            if changed.shape != self._grid.shape:
                raise ValueError("[ChunkedMesh] changed_mask shape mismatch")
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
            for key, (window, _) in sorted(self._chunks.items())
            if self._intersects(window, event)
        )

    def _chunk_windows(self, shape: tuple[int, int]) -> dict[tuple[int, int], GridWindow]:
        ny, nx = shape
        result = {}
        tile_row = 0
        for row_start in range(0, ny - 1, self.chunk_size_cells):
            row_stop = min(ny, row_start + self.chunk_size_cells + 1)
            tile_col = 0
            for col_start in range(0, nx - 1, self.chunk_size_cells):
                col_stop = min(nx, col_start + self.chunk_size_cells + 1)
                result[(tile_row, tile_col)] = GridWindow(
                    row_start, col_start, row_stop, col_stop
                )
                tile_col += 1
            tile_row += 1
        return result

    @staticmethod
    def _local_grid(grid: TerrainGrid, window: GridWindow, prim_path: str) -> TerrainGrid:
        return TerrainGrid(
            nx=window.shape[1],
            ny=window.shape[0],
            dx=grid.dx,
            dy=grid.dy,
            origin_x=grid.origin_x + window.col_start * grid.dx,
            origin_y=grid.origin_y + window.row_start * grid.dy,
            terrain_prim_path=prim_path,
            terrain_to_world_matrix=grid.terrain_to_world_matrix,
        )

    @staticmethod
    def _intersects(first: GridWindow, second: GridWindow) -> bool:
        return not (
            first.row_stop <= second.row_start
            or second.row_stop <= first.row_start
            or first.col_stop <= second.col_start
            or second.col_stop <= first.col_start
        )
