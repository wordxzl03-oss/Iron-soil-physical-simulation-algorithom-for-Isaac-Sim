"""Abstract interface shared by height-field relaxation solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..config import SolverConfig
from ..terrain.terrain_grid import TerrainGrid


@dataclass(frozen=True)
class RelaxationResult:
    """Stable terrain, true intermediate copies and geometric volume balance."""

    heightmap_stable: np.ndarray
    heightmap_sequence: tuple[np.ndarray, ...]
    iteration_count: int
    volume_before_m3: float
    volume_after_m3: float
    boundary_outflow_m3: float
    converged: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        stable = np.asarray(self.heightmap_stable, dtype=np.float64)
        if stable.ndim != 2 or not np.all(np.isfinite(stable)) or np.any(stable < 0.0):
            raise ValueError(
                "[TerrainRelaxationSolver] stable heightmap must be finite, non-negative 2-D"
            )
        stable = np.ascontiguousarray(stable.copy())
        stable.setflags(write=False)
        sequence: list[np.ndarray] = []
        for index, item in enumerate(self.heightmap_sequence):
            height = np.asarray(item)
            if height.shape != stable.shape or not np.all(np.isfinite(height)):
                raise ValueError(
                    "[TerrainRelaxationSolver] sequence shape/value error; "
                    f"index={index}, expected={stable.shape}, received={height.shape}"
                )
            if height.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
                height = np.asarray(height, dtype=np.float64)
            copy = np.ascontiguousarray(height.copy())
            copy.setflags(write=False)
            sequence.append(copy)
        if not sequence:
            raise ValueError("[TerrainRelaxationSolver] sequence must not be empty")
        for name in (
            "volume_before_m3",
            "volume_after_m3",
            "boundary_outflow_m3",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"[TerrainRelaxationSolver] {name} must be finite and non-negative"
                )
        if self.iteration_count < 0:
            raise ValueError("[TerrainRelaxationSolver] iteration_count must be >= 0")
        object.__setattr__(self, "heightmap_stable", stable)
        object.__setattr__(self, "heightmap_sequence", tuple(sequence))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class TerrainRelaxationSolver(ABC):
    """Solver boundary that prevents runtime code from depending on MiniSlope."""

    @abstractmethod
    def initialize(self, config: SolverConfig, grid: TerrainGrid | None = None) -> None:
        """Initialize validated configuration and terrain geometry."""

    @abstractmethod
    def step(self, heightmap: np.ndarray) -> RelaxationResult:
        """Run one solver relaxation iteration."""

    @abstractmethod
    def solve(self, heightmap: np.ndarray) -> RelaxationResult:
        """Run to convergence or the configured iteration limit."""

    @abstractmethod
    def solve_sequence(self, heightmap: np.ndarray) -> RelaxationResult:
        """Run while recording true intermediate array copies."""

    @abstractmethod
    def reset(self) -> None:
        """Clear solver temporal state and diagnostics."""
