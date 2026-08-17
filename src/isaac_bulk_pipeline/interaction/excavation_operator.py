"""Robot- and solver-independent 2.5-D terrain column clipping."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..config import ExcavationConfig
from ..terrain.terrain_grid import TerrainGrid
from .continuous_sweep import SweepResult


@dataclass(frozen=True)
class ExcavationResult:
    """Authoritative height map and geometric volume after one tool sweep."""

    heightmap_excavated: np.ndarray
    removed_volume_m3: float
    affected_bbox_grid: tuple[int, int, int, int]
    affected_mask: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        height = np.asarray(self.heightmap_excavated, dtype=np.float64)
        mask = np.asarray(self.affected_mask, dtype=bool)
        if height.ndim != 2 or mask.shape != height.shape:
            raise ValueError(
                "[ExcavationOperator] result height/mask shape mismatch; "
                f"height={height.shape}, mask={mask.shape}"
            )
        if not np.all(np.isfinite(height)) or np.any(height < 0.0):
            raise ValueError(
                "[ExcavationOperator] result contains invalid terrain heights"
            )
        if not np.isfinite(self.removed_volume_m3) or self.removed_volume_m3 < 0.0:
            raise ValueError(
                "[ExcavationOperator] removed_volume_m3 must be non-negative"
            )
        height = np.ascontiguousarray(height.copy())
        mask = np.ascontiguousarray(mask.copy())
        height.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "heightmap_excavated", height)
        object.__setattr__(self, "affected_mask", mask)
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class ExcavationOperator:
    """Apply a :class:`SweepResult` without knowing robot, USD or solver details."""

    def __init__(self, config: ExcavationConfig | None = None) -> None:
        self._config = config or ExcavationConfig()

    @property
    def config(self) -> ExcavationConfig:
        return self._config

    def apply(
        self,
        heightmap: np.ndarray,
        sweep: SweepResult,
        grid: TerrainGrid,
    ) -> ExcavationResult:
        """Clip affected columns and report removed geometric volume in m³."""

        old = np.asarray(grid.validate_heightmap(heightmap), dtype=np.float64)
        if sweep.affected_mask.shape != grid.shape or sweep.cut_surface.shape != grid.shape:
            raise ValueError(
                "[ExcavationOperator] sweep shape must equal terrain H[y,x]; "
                f"grid={grid.shape}, mask={sweep.affected_mask.shape}, "
                f"cut_surface={sweep.cut_surface.shape}, "
                f"prim_path={grid.terrain_prim_path}"
            )
        candidate = sweep.affected_mask & np.isfinite(sweep.cut_surface)
        if grid.valid_mask is not None:
            candidate &= grid.valid_mask
        target = np.maximum(
            sweep.cut_surface,
            self._config.minimum_terrain_height_m,
        )
        depth = np.where(candidate, old - target, 0.0)
        changed = candidate & (depth >= self._config.minimum_cut_depth_m) & (depth > 0.0)
        result = np.array(old, dtype=np.float64, copy=True, order="C")
        result[changed] = np.minimum(old[changed], target[changed])
        removed_height = old - result
        removed_volume = float(removed_height.sum(dtype=np.float64) * grid.cell_area)
        bbox = self._bbox(changed)
        return ExcavationResult(
            heightmap_excavated=result,
            removed_volume_m3=removed_volume,
            affected_bbox_grid=bbox,
            affected_mask=changed,
            diagnostics={
                "mode": self._config.mode,
                "candidate_cell_count": int(candidate.sum()),
                "changed_cell_count": int(changed.sum()),
                "minimum_cut_depth_m": self._config.minimum_cut_depth_m,
                "minimum_terrain_height_m": self._config.minimum_terrain_height_m,
                "maximum_removed_depth_m": float(
                    removed_height.max(initial=0.0)
                ),
                "sweep_bbox_grid": sweep.affected_bbox_grid,
            },
        )

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        indices = np.argwhere(mask)
        if not len(indices):
            return (0, 0, 0, 0)
        low = indices.min(axis=0)
        high = indices.max(axis=0) + 1
        return int(low[0]), int(low[1]), int(high[0]), int(high[1])
