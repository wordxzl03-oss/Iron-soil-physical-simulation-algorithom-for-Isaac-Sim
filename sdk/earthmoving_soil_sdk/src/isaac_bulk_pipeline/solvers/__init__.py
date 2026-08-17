"""Replaceable terrain-relaxation solver interfaces and adapters."""

from .base_solver import RelaxationResult, TerrainRelaxationSolver
from .event_driven_slope import (
    EventDrivenMinimumSlopeAdapter,
    NumericalNonconvergenceError,
)
from .minimum_slope_adapter import MinimumSlopeAdapter
from .sparse_tile_slope import (
    CompactActiveEdgeBatch,
    CompactTileFrontier,
    IncrementalSlopeProgress,
    SparseTileFrontierMinimumSlopeAdapter,
)
from .warp_compact_frontier import (
    WarpCompactActiveEdgeOperator,
    WarpFrontierPhaseResult,
)

__all__ = [
    "EventDrivenMinimumSlopeAdapter",
    "NumericalNonconvergenceError",
    "MinimumSlopeAdapter",
    "CompactActiveEdgeBatch",
    "CompactTileFrontier",
    "IncrementalSlopeProgress",
    "SparseTileFrontierMinimumSlopeAdapter",
    "WarpCompactActiveEdgeOperator",
    "WarpFrontierPhaseResult",
    "RelaxationResult",
    "TerrainRelaxationSolver",
]
