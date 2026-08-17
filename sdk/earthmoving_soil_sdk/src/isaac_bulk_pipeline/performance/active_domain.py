"""Dynamic tile and bounding-window management for a full-resolution terrain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Iterable

import numpy as np


class ActiveReason(IntFlag):
    """Why a tile participates in the current high-cost update."""

    STATIC = 0
    BUCKET = 1 << 0
    FAILURE = 1 << 1
    MOBILE = 1 << 2
    TRACK_LEFT = 1 << 3
    TRACK_RIGHT = 1 << 4
    SPILL = 1 << 5
    DEPOSITION = 1 << 6
    SLOPE = 1 << 7
    DIRTY_VISUAL = 1 << 8
    DIRTY_CONTACT = 1 << 9


@dataclass(frozen=True, order=True)
class GridWindow:
    """Half-open canonical ``H[y,x]`` window."""

    row_start: int
    col_start: int
    row_stop: int
    col_stop: int

    def __post_init__(self) -> None:
        values = (self.row_start, self.col_start, self.row_stop, self.col_stop)
        if not all(isinstance(value, (int, np.integer)) for value in values):
            raise TypeError("[ActiveDomain] window bounds must be integers")
        if self.row_start < 0 or self.col_start < 0:
            raise ValueError("[ActiveDomain] window starts must be non-negative")
        if self.row_stop < self.row_start or self.col_stop < self.col_start:
            raise ValueError("[ActiveDomain] window stops precede starts")

    @property
    def shape(self) -> tuple[int, int]:
        return self.row_stop - self.row_start, self.col_stop - self.col_start

    @property
    def cell_count(self) -> int:
        rows, columns = self.shape
        return rows * columns

    @property
    def empty(self) -> bool:
        return self.row_stop == self.row_start or self.col_stop == self.col_start

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.row_start, self.row_stop), slice(self.col_start, self.col_stop)

    def expanded(self, cells: int, shape: tuple[int, int]) -> "GridWindow":
        halo = int(cells)
        if halo < 0:
            raise ValueError("[ActiveDomain] expansion cells must be non-negative")
        ny, nx = shape
        return GridWindow(
            max(0, self.row_start - halo),
            max(0, self.col_start - halo),
            min(ny, self.row_stop + halo),
            min(nx, self.col_stop + halo),
        )

    def union(self, other: "GridWindow") -> "GridWindow":
        if self.empty:
            return other
        if other.empty:
            return self
        return GridWindow(
            min(self.row_start, other.row_start),
            min(self.col_start, other.col_start),
            max(self.row_stop, other.row_stop),
            max(self.col_stop, other.col_stop),
        )


@dataclass(frozen=True)
class ActiveDomainSnapshot:
    window: GridWindow
    active_tile_count: int
    tile_count: int
    active_cell_count: int
    full_cell_count: int
    active_ratio: float
    tile_size: int
    tile_flags: np.ndarray

    def __post_init__(self) -> None:
        flags = np.array(self.tile_flags, dtype=np.uint16, copy=True, order="C")
        flags.setflags(write=False)
        object.__setattr__(self, "tile_flags", flags)


class ActiveDomainManager:
    """Track sparse activity without replacing the authoritative full arrays.

    The manager keeps full-resolution state ownership separate from expensive
    update ownership. Callers mark compact masks/windows as events occur. Tile
    flags support dirty visual/contact updates, while ``combined_window`` gives
    numerical kernels a dynamically sized crop with an explicit stencil halo.
    """

    def __init__(self, shape: tuple[int, int], tile_size: int = 32) -> None:
        if len(shape) != 2 or min(shape) < 2:
            raise ValueError("[ActiveDomain] shape must be a 2-D grid >= 2")
        if not isinstance(tile_size, int) or tile_size < 4:
            raise ValueError("[ActiveDomain] tile_size must be an integer >= 4")
        self.shape = int(shape[0]), int(shape[1])
        self.tile_size = tile_size
        self.tile_shape = (
            (self.shape[0] + tile_size - 1) // tile_size,
            (self.shape[1] + tile_size - 1) // tile_size,
        )
        self._tile_flags = np.zeros(self.tile_shape, dtype=np.uint16)
        self._windows: dict[ActiveReason, GridWindow] = {}

    @staticmethod
    def window_from_mask(mask: np.ndarray) -> GridWindow:
        active = np.asarray(mask, dtype=bool)
        rows, columns = np.nonzero(active)
        if rows.size == 0:
            return GridWindow(0, 0, 0, 0)
        return GridWindow(
            int(rows.min()), int(columns.min()), int(rows.max()) + 1, int(columns.max()) + 1
        )

    def mark_mask(
        self, mask: np.ndarray, reason: ActiveReason, *, halo_cells: int = 0
    ) -> GridWindow:
        if np.asarray(mask).shape != self.shape:
            raise ValueError("[ActiveDomain] mask shape mismatch")
        return self.mark_window(
            self.window_from_mask(mask).expanded(halo_cells, self.shape), reason
        )

    def mark_points(
        self,
        rows_columns: np.ndarray,
        reason: ActiveReason,
        *,
        halo_cells: int = 0,
    ) -> GridWindow:
        points = np.asarray(rows_columns)
        if points.size == 0:
            return GridWindow(0, 0, 0, 0)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("[ActiveDomain] rows_columns must be finite shape (N,2)")
        rows = np.clip(points[:, 0].astype(np.int64), 0, self.shape[0] - 1)
        cols = np.clip(points[:, 1].astype(np.int64), 0, self.shape[1] - 1)
        window = GridWindow(
            int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1
        ).expanded(halo_cells, self.shape)
        return self.mark_window(window, reason)

    def mark_window(self, window: GridWindow, reason: ActiveReason) -> GridWindow:
        if reason is ActiveReason.STATIC or int(reason) == 0:
            raise ValueError("[ActiveDomain] STATIC cannot be marked active")
        clipped = GridWindow(
            min(window.row_start, self.shape[0]),
            min(window.col_start, self.shape[1]),
            min(window.row_stop, self.shape[0]),
            min(window.col_stop, self.shape[1]),
        )
        if clipped.empty:
            return clipped
        current = self._windows.get(reason)
        self._windows[reason] = clipped if current is None else current.union(clipped)
        tile_rows, tile_cols = self._tile_slices(clipped)
        self._tile_flags[tile_rows, tile_cols] |= np.uint16(int(reason))
        return clipped

    def clear(self, reasons: ActiveReason | None = None) -> None:
        if reasons is None:
            self._windows.clear()
            self._tile_flags.fill(0)
            return
        mask = int(reasons)
        for reason in tuple(self._windows):
            if int(reason) & mask:
                del self._windows[reason]
        self._rebuild_tiles()

    def windows_for(self, reasons: ActiveReason) -> tuple[GridWindow, ...]:
        mask = int(reasons)
        return tuple(
            window for reason, window in self._windows.items() if int(reason) & mask
        )

    def combined_window(
        self, reasons: ActiveReason | None = None, *, halo_cells: int = 0
    ) -> GridWindow:
        windows: Iterable[GridWindow]
        if reasons is None:
            windows = self._windows.values()
        else:
            windows = self.windows_for(reasons)
        result = GridWindow(0, 0, 0, 0)
        for window in windows:
            result = result.union(window)
        return result.expanded(halo_cells, self.shape)

    def snapshot(self, reasons: ActiveReason | None = None) -> ActiveDomainSnapshot:
        window = self.combined_window(reasons)
        flags = self._tile_flags if reasons is None else self._tile_flags & int(reasons)
        active_tiles = int(np.count_nonzero(flags))
        return ActiveDomainSnapshot(
            window=window,
            active_tile_count=active_tiles,
            tile_count=int(flags.size),
            active_cell_count=window.cell_count,
            full_cell_count=int(np.prod(self.shape)),
            active_ratio=window.cell_count / float(np.prod(self.shape)),
            tile_size=self.tile_size,
            tile_flags=flags,
        )

    def _tile_slices(self, window: GridWindow) -> tuple[slice, slice]:
        return (
            slice(window.row_start // self.tile_size, (window.row_stop - 1) // self.tile_size + 1),
            slice(window.col_start // self.tile_size, (window.col_stop - 1) // self.tile_size + 1),
        )

    def _rebuild_tiles(self) -> None:
        self._tile_flags.fill(0)
        for reason, window in self._windows.items():
            rows, cols = self._tile_slices(window)
            self._tile_flags[rows, cols] |= np.uint16(int(reason))
