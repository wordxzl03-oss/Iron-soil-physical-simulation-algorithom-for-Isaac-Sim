"""Topology-consistent integration for authoritative vertex height fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SurfaceTopology(str, Enum):
    """Supported interpolation/topology semantics for a vertex field."""

    TRIANGLE_A_C = "triangle_a_c"
    BILINEAR_TRAPEZOID = "bilinear_trapezoid"


@dataclass(frozen=True)
class TerrainVolumeIntegrator:
    """Integrate ``H[y,x]`` over the actual vertex-field footprint.

    ``TRIANGLE_A_C`` exactly matches ``DynamicMeshAdapter``: each quad with
    vertices ``a=top-left, b=top-right, c=bottom-right, d=bottom-left`` is split
    into triangles ``(a,b,c)`` and ``(a,c,d)``. Its exact contribution is::

        dx*dy/6 * (2*a + b + 2*c + d)

    ``BILINEAR_TRAPEZOID`` is an explicit alternative whose contribution is the
    four-corner trapezoidal average. The two modes are intentionally not treated
    as equivalent for non-planar quads.
    """

    nx: int
    ny: int
    dx_m: float
    dy_m: float
    topology: SurfaceTopology = SurfaceTopology.TRIANGLE_A_C
    valid_cell_mask: np.ndarray | None = field(default=None, compare=False)
    _vertex_weights_m2: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.nx, int) or not isinstance(self.ny, int):
            raise TypeError("[TerrainVolumeIntegrator] nx and ny must be integers")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("[TerrainVolumeIntegrator] nx and ny must be >= 2")
        dx = float(self.dx_m)
        dy = float(self.dy_m)
        if not np.all(np.isfinite([dx, dy])) or dx <= 0.0 or dy <= 0.0:
            raise ValueError(
                "[TerrainVolumeIntegrator] dx_m and dy_m must be finite and positive"
            )
        try:
            topology = SurfaceTopology(self.topology)
        except ValueError as exc:
            raise ValueError(
                f"[TerrainVolumeIntegrator] unsupported topology={self.topology!r}"
            ) from exc
        object.__setattr__(self, "dx_m", dx)
        object.__setattr__(self, "dy_m", dy)
        object.__setattr__(self, "topology", topology)

        if self.valid_cell_mask is None:
            mask = np.ones((self.ny - 1, self.nx - 1), dtype=bool)
        else:
            mask = np.array(self.valid_cell_mask, dtype=bool, copy=True)
            if mask.shape != (self.ny - 1, self.nx - 1):
                raise ValueError(
                    "[TerrainVolumeIntegrator] valid_cell_mask is cell-centered and "
                    f"must have shape {(self.ny - 1, self.nx - 1)}; "
                    f"received={mask.shape}"
                )
        mask.setflags(write=False)
        object.__setattr__(self, "valid_cell_mask", mask)

        weights = np.zeros((self.ny, self.nx), dtype=np.float64)
        active = mask.astype(np.float64, copy=False)
        if topology is SurfaceTopology.TRIANGLE_A_C:
            factor = active * (dx * dy / 6.0)
            weights[:-1, :-1] += 2.0 * factor
            weights[:-1, 1:] += factor
            weights[1:, 1:] += 2.0 * factor
            weights[1:, :-1] += factor
        else:
            factor = active * (dx * dy / 4.0)
            weights[:-1, :-1] += factor
            weights[:-1, 1:] += factor
            weights[1:, 1:] += factor
            weights[1:, :-1] += factor
        weights.setflags(write=False)
        object.__setattr__(self, "_vertex_weights_m2", weights)

    @classmethod
    def from_grid(
        cls,
        grid: Any,
        *,
        topology: SurfaceTopology = SurfaceTopology.TRIANGLE_A_C,
        valid_cell_mask: np.ndarray | None = None,
    ) -> "TerrainVolumeIntegrator":
        """Construct from a TerrainGrid-like object without changing that class."""

        return cls(
            nx=int(grid.nx),
            ny=int(grid.ny),
            dx_m=float(grid.dx),
            dy_m=float(grid.dy),
            topology=topology,
            valid_cell_mask=valid_cell_mask,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def vertex_weights_m2(self) -> np.ndarray:
        """Return an independent read-only copy of exact vertex area weights."""

        result = np.array(self._vertex_weights_m2, copy=True)
        result.setflags(write=False)
        return result

    @property
    def domain_area_m2(self) -> float:
        return float(np.sum(self._vertex_weights_m2, dtype=np.float64))

    def integrate(
        self,
        height_m: np.ndarray,
        *,
        require_nonnegative: bool = True,
    ) -> float:
        """Return volume in m³ for a finite vertex field."""

        field = self.validate_vertex_field(
            height_m,
            name="height_m",
            require_nonnegative=require_nonnegative,
        )
        volume = float(
            np.sum(field * self._vertex_weights_m2, dtype=np.float64)
        )
        if not np.isfinite(volume):
            raise ValueError("[TerrainVolumeIntegrator] non-finite integrated volume")
        return volume

    def integrate_delta(self, before_m: np.ndarray, after_m: np.ndarray) -> float:
        """Integrate signed ``after-before`` volume without clipping."""

        before = self.validate_vertex_field(
            before_m,
            name="before_m",
            require_nonnegative=False,
        )
        after = self.validate_vertex_field(
            after_m,
            name="after_m",
            require_nonnegative=False,
        )
        delta = after - before
        return float(np.sum(delta * self._vertex_weights_m2, dtype=np.float64))

    def cell_contributions_m3(self, height_m: np.ndarray) -> np.ndarray:
        """Return one exact contribution per quad, including masked-out zeros."""

        height = self.validate_vertex_field(
            height_m,
            name="height_m",
            require_nonnegative=False,
        )
        a = height[:-1, :-1]
        b = height[:-1, 1:]
        c = height[1:, 1:]
        d = height[1:, :-1]
        if self.topology is SurfaceTopology.TRIANGLE_A_C:
            result = self.dx_m * self.dy_m / 6.0 * (
                2.0 * a + b + 2.0 * c + d
            )
        else:
            result = self.dx_m * self.dy_m / 4.0 * (a + b + c + d)
        return np.ascontiguousarray(
            np.where(self.valid_cell_mask, result, 0.0),
            dtype=np.float64,
        )

    def validate_vertex_field(
        self,
        value: np.ndarray,
        *,
        name: str,
        require_nonnegative: bool,
    ) -> np.ndarray:
        result = np.asarray(value)
        if result.shape != self.shape or not np.issubdtype(result.dtype, np.number):
            raise ValueError(
                f"[TerrainVolumeIntegrator] {name} must be numeric H[y,x] with "
                f"shape={self.shape}; received shape={result.shape}, dtype={result.dtype}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(
                f"[TerrainVolumeIntegrator] {name} contains NaN or Inf"
            )
        if require_nonnegative and np.any(result < 0.0):
            raise ValueError(
                f"[TerrainVolumeIntegrator] {name} contains negative thickness/height"
            )
        return result
