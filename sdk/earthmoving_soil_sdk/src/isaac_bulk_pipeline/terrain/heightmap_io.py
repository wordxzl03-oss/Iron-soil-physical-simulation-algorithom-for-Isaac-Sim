"""Validated height-map loading and saving with explicit axis conversion."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from .terrain_grid import TerrainGrid

AxisOrder = Literal["yx", "xy"]


def validate_heightmap(
    heightmap: np.ndarray,
    *,
    expected_shape: tuple[int, int] | None = None,
    require_nonnegative: bool = True,
    output_dtype: np.dtype | type = np.float64,
    copy: bool = True,
) -> np.ndarray:
    """Return a validated contiguous ``H[y, x]`` array in metres.

    This function never guesses orientation. Legacy ``H[x, y]`` inputs must be
    converted explicitly by :meth:`HeightmapIO.load` with
    ``source_axis_order="xy"``.
    """

    height = np.asarray(heightmap)
    if height.ndim != 2:
        raise ValueError(
            f"[HeightmapIO] heightmap must be 2-D H[y,x]; shape={height.shape}"
        )
    if expected_shape is not None and height.shape != expected_shape:
        raise ValueError(
            "[HeightmapIO] heightmap shape mismatch under H[y,x] convention; "
            f"expected={expected_shape}, received={height.shape}"
        )
    if not np.issubdtype(height.dtype, np.number):
        raise TypeError(
            f"[HeightmapIO] heightmap dtype must be numeric; dtype={height.dtype}"
        )
    if not np.all(np.isfinite(height)):
        bad_count = int(np.size(height) - np.count_nonzero(np.isfinite(height)))
        raise ValueError(
            f"[HeightmapIO] heightmap contains {bad_count} NaN/Inf values; "
            f"shape={height.shape}"
        )
    if require_nonnegative and np.any(height < 0.0):
        raise ValueError(
            "[HeightmapIO] heightmap contains negative metre heights; "
            f"minimum={float(np.min(height))}"
        )
    converted = np.asarray(height, dtype=output_dtype)
    if copy:
        return np.array(converted, dtype=output_dtype, order="C", copy=True)
    # NumPy 2 rejects ``np.array(non_contiguous_view, copy=False, order='C')``.
    # A chunk slice may legitimately need one contiguity copy even when callers
    # do not request an unconditional semantic copy.
    return np.ascontiguousarray(converted, dtype=output_dtype)


class HeightmapIO:
    """Read and write authoritative height maps in metres as ``H[y, x]``."""

    @staticmethod
    def load(
        path: str | Path,
        *,
        grid: TerrainGrid | None = None,
        key: str | None = None,
        source_axis_order: AxisOrder = "yx",
        output_dtype: np.dtype | type = np.float64,
    ) -> np.ndarray:
        """Load ``.npy``, ``.npz`` or ``.csv`` and explicitly normalize axes.

        ``source_axis_order="xy"`` is the only supported compatibility path for
        legacy MiniSlope arrays whose axis 0 is X. The returned array is always
        an independent contiguous ``H[y, x]`` copy.
        """

        source = Path(path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"[HeightmapIO] file not found: {source}")
        suffix = source.suffix.lower()
        if suffix == ".npy":
            raw = np.load(source, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                selected_key = key
                if selected_key is None:
                    if len(archive.files) != 1:
                        raise ValueError(
                            "[HeightmapIO] NPZ contains multiple arrays; an explicit "
                            f"key is required. path={source}, keys={archive.files}"
                        )
                    selected_key = archive.files[0]
                if selected_key not in archive.files:
                    raise KeyError(
                        f"[HeightmapIO] key={selected_key!r} not found in {source}; "
                        f"keys={archive.files}"
                    )
                raw = np.array(archive[selected_key], copy=True)
        elif suffix == ".csv":
            raw = np.loadtxt(source, delimiter=",")
        else:
            raise ValueError(
                f"[HeightmapIO] unsupported extension={suffix!r}; path={source}"
            )

        if source_axis_order == "xy":
            raw = np.asarray(raw).T
        elif source_axis_order != "yx":
            raise ValueError(
                "[HeightmapIO] source_axis_order must be 'yx' or explicit legacy "
                f"'xy'; value={source_axis_order!r}"
            )
        expected_shape = None if grid is None else grid.shape
        return validate_heightmap(
            raw,
            expected_shape=expected_shape,
            output_dtype=output_dtype,
            copy=True,
        )

    @staticmethod
    def save(path: str | Path, heightmap: np.ndarray, *, key: str = "heightmap_m") -> Path:
        """Save one validated ``H[y, x]`` array without changing orientation."""

        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        height = validate_heightmap(heightmap, copy=True)
        suffix = destination.suffix.lower()
        if suffix == ".npy":
            np.save(destination, height)
        elif suffix == ".npz":
            np.savez_compressed(destination, **{key: height})
        elif suffix == ".csv":
            np.savetxt(destination, height, delimiter=",")
        else:
            raise ValueError(
                f"[HeightmapIO] unsupported extension={suffix!r}; path={destination}"
            )
        return destination
