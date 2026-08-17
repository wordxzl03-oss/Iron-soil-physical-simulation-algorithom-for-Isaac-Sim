"""Adapter for the repository's legacy ``H[x,y]`` minimum-slope solver."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..config import SolverConfig
from ..terrain.terrain_grid import TerrainGrid
from .base_solver import RelaxationResult, TerrainRelaxationSolver


class MinimumSlopeAdapter(TerrainRelaxationSolver):
    """Expose ``slope_model.relax_critical_slope`` using canonical ``H[y,x]``.

    The legacy function uses axis 0 as X and axis 1 as Y. Transposition is
    performed only at this adapter boundary and never leaks into runtime state.
    Closed boundaries use the legacy conservative domain directly. Open
    boundaries add a one-cell fixed-height ghost ring for every iteration and
    count material lost from the interior as ``boundary_outflow_m3``.
    """

    def __init__(
        self,
        grid: TerrainGrid | None = None,
        config: SolverConfig | None = None,
    ) -> None:
        self._grid = grid
        self._config = config
        self._initialized = False
        self._last_result: RelaxationResult | None = None
        if grid is not None and config is not None:
            self.initialize(config, grid)

    @property
    def config(self) -> SolverConfig:
        if self._config is None or not self._initialized:
            raise RuntimeError("[MinimumSlopeAdapter] solver is not initialized")
        return self._config

    @property
    def grid(self) -> TerrainGrid:
        if self._grid is None or not self._initialized:
            raise RuntimeError("[MinimumSlopeAdapter] solver is not initialized")
        return self._grid

    @property
    def last_result(self) -> RelaxationResult | None:
        return self._last_result

    def initialize(self, config: SolverConfig, grid: TerrainGrid | None = None) -> None:
        """Bind settings and grid, rejecting unsupported masked domains."""

        if grid is not None:
            self._grid = grid
        if self._grid is None:
            raise ValueError("[MinimumSlopeAdapter] initialize requires TerrainGrid")
        if self._grid.valid_mask is not None and not np.all(self._grid.valid_mask):
            raise ValueError(
                "[MinimumSlopeAdapter] legacy solver does not support holes in valid_mask; "
                f"shape={self._grid.shape}, prim_path={self._grid.terrain_prim_path}"
            )
        self._config = config
        self._validate_sequence_budget()
        self._initialized = True
        self._last_result = None

    def step(self, heightmap: np.ndarray) -> RelaxationResult:
        """Run exactly one legacy relaxation pass and return its balance."""

        initial = self._validated_copy(heightmap)
        stable, stats = self._legacy_call(initial, max_iterations=1)
        result = self._make_result(
            initial,
            stable,
            sequence=self._endpoint_sequence(initial, stable),
            iteration_count=1,
            converged=bool(stats.converged),
            legacy_stats=stats,
        )
        self._last_result = result
        return result

    def solve(self, heightmap: np.ndarray) -> RelaxationResult:
        """Solve efficiently; closed domains use one legacy solver call."""

        initial = self._validated_copy(heightmap)
        if self.config.boundary_condition == "closed":
            # This class is the CPU full-domain reference.  Local/event-driven
            # execution belongs to EventDrivenMinimumSlopeAdapter, whose
            # frontier expands from event seeds without a precomputed ROI.
            stable, stats = self._legacy_call(
                initial, max_iterations=self.config.max_iterations
            )
            sequence = self._endpoint_sequence(initial, stable)
            iterations = int(stats.iterations)
            converged = bool(stats.converged)
        else:
            stable, sequence, iterations, converged, stats = self._iterative_solve(
                initial, record_sequence=False
            )
        result = self._make_result(
            initial,
            stable,
            sequence=sequence,
            iteration_count=iterations,
            converged=converged,
            legacy_stats=stats,
        )
        self._last_result = result
        return result

    def solve_sequence(self, heightmap: np.ndarray) -> RelaxationResult:
        """Solve with an explicitly bounded optional intermediate sequence.

        When ``sequence_enabled`` is false this method deliberately retains
        only independent pre/post states and uses the efficient solver path.
        Debug sequences are capped by both frame count and a byte budget.
        """

        if not self.config.sequence_enabled:
            return self.solve(heightmap)

        initial = self._validated_copy(heightmap)
        stable, sequence, iterations, converged, stats = self._iterative_solve(
            initial, record_sequence=True
        )
        result = self._make_result(
            initial,
            stable,
            sequence=sequence,
            iteration_count=iterations,
            converged=converged,
            legacy_stats=stats,
        )
        self._last_result = result
        return result

    def reset(self) -> None:
        """Clear the previous result; this solver has no hidden material state."""

        self._last_result = None

    def _validated_copy(self, heightmap: np.ndarray) -> np.ndarray:
        return np.array(
            self.grid.validate_heightmap(heightmap),
            dtype=np.float64,
            copy=True,
            order="C",
        )

    def _legacy_call(self, height_yx: np.ndarray, *, max_iterations: int):
        try:
            from slope_model import relax_critical_slope
        except ImportError as exc:
            raise RuntimeError(
                "[MinimumSlopeAdapter] slope_model.relax_critical_slope is unavailable"
            ) from exc

        if self.config.boundary_condition == "open":
            source_yx = np.pad(
                height_yx,
                1,
                mode="constant",
                constant_values=self.config.boundary_height_m,
            )
        else:
            source_yx = height_yx
        # Explicit canonical H[y,x] -> legacy H[x,y] conversion.
        legacy_xy = np.ascontiguousarray(source_yx.T)
        relaxed_xy, stats = relax_critical_slope(
            legacy_xy,
            (self.grid.dx, self.grid.dy),
            critical_angle_deg=self.config.critical_angle_deg,
            max_iterations=max_iterations,
            tolerance=self.config.tolerance,
        )
        relaxed_yx = np.ascontiguousarray(relaxed_xy.T)
        if self.config.boundary_condition == "open":
            relaxed_yx = np.ascontiguousarray(relaxed_yx[1:-1, 1:-1])
        return relaxed_yx, stats

    def _iterative_solve(
        self,
        initial: np.ndarray,
        *,
        record_sequence: bool,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...], int, bool, Any]:
        current = np.array(initial, copy=True)
        sequence: list[np.ndarray] = [self._sequence_copy(initial)]
        converged = False
        stats: Any = None
        iteration = 0
        for iteration in range(1, self.config.max_iterations + 1):
            current, stats = self._legacy_call(current, max_iterations=1)
            if (
                record_sequence
                and iteration % self.config.sequence_stride == 0
                and len(sequence) < self.config.sequence_max_frames - 1
            ):
                sequence.append(self._sequence_copy(current))
            if stats.converged:
                converged = True
                break
        final = self._sequence_copy(current)
        if len(sequence) == 1 or not np.array_equal(sequence[-1], final):
            if len(sequence) < self.config.sequence_max_frames:
                sequence.append(final)
            else:
                sequence[-1] = final
        return current, tuple(sequence), iteration, converged, stats

    def _endpoint_sequence(
        self, initial: np.ndarray, stable: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._sequence_copy(initial), self._sequence_copy(stable)

    def _sequence_copy(self, heightmap: np.ndarray) -> np.ndarray:
        return np.array(
            heightmap,
            dtype=np.dtype(self.config.sequence_dtype),
            copy=True,
            order="C",
        )

    def _validate_sequence_budget(self) -> None:
        assert self._config is not None
        assert self._grid is not None
        dtype = np.dtype(self._config.sequence_dtype)
        frame_count = (
            self._config.sequence_max_frames
            if self._config.sequence_enabled
            else 2
        )
        required_bytes = int(np.prod(self._grid.shape)) * dtype.itemsize * frame_count
        limit_bytes = int(self._config.sequence_memory_limit_mb * 1024 * 1024)
        if required_bytes > limit_bytes:
            raise ValueError(
                "[MinimumSlopeAdapter] configured relaxation sequence exceeds "
                "memory budget; "
                f"shape={self._grid.shape}, dtype={dtype.name}, frames={frame_count}, "
                f"required_mb={required_bytes / 1024**2:.3f}, "
                f"limit_mb={self._config.sequence_memory_limit_mb:.3f}"
            )

    def _make_result(
        self,
        initial: np.ndarray,
        stable: np.ndarray,
        *,
        sequence: tuple[np.ndarray, ...],
        iteration_count: int,
        converged: bool,
        legacy_stats: Any,
        additional_diagnostics: Mapping[str, Any] | None = None,
    ) -> RelaxationResult:
        before = self.grid.compute_volume(initial)
        after = self.grid.compute_volume(stable)
        if self.config.boundary_condition == "open":
            if after > before + self.config.conservation_tolerance_m3:
                raise RuntimeError(
                    "[MinimumSlopeAdapter] open boundary created terrain volume; "
                    f"before={before}, after={after}, shape={initial.shape}"
                )
            outflow = max(before - after, 0.0)
        else:
            outflow = 0.0
        balance_error = before - after - outflow
        if (
            self.config.boundary_condition == "closed"
            and abs(balance_error) > self.config.conservation_tolerance_m3
        ):
            raise RuntimeError(
                "[MinimumSlopeAdapter] closed-boundary volume conservation failed; "
                f"shape={initial.shape}, before={before}, after={after}, "
                f"error={balance_error}, tolerance="
                f"{self.config.conservation_tolerance_m3}"
            )
        diagnostics = {
            "adapter": "MinimumSlopeAdapter",
            "legacy_entrypoint": "slope_model.relax_critical_slope",
            "canonical_axis_order": "yx",
            "legacy_axis_order": "xy",
            "axis_conversion": "transpose_at_adapter_boundary",
            "input_was_copied": True,
            "grid_spacing_m": [self.grid.dx, self.grid.dy],
            "boundary_condition": self.config.boundary_condition,
            "critical_angle_deg": self.config.critical_angle_deg,
            "maximum_iterations": self.config.max_iterations,
            "sequence_enabled": self.config.sequence_enabled,
            "sequence_stride": self.config.sequence_stride,
            "sequence_max_frames": self.config.sequence_max_frames,
            "sequence_dtype": self.config.sequence_dtype,
            "sequence_memory_limit_mb": self.config.sequence_memory_limit_mb,
            "sequence_frame_count": len(sequence),
            "sequence_memory_mb": float(
                sum(item.nbytes for item in sequence) / 1024**2
            ),
            "volume_balance_error_m3": balance_error,
            "legacy_max_slope": None
            if legacy_stats is None
            else float(legacy_stats.max_slope),
            "model_fidelity": (
                "reduced-order geometric minimum-slope relaxation; "
                "not validated iron-ore physical truth"
            ),
        }
        if additional_diagnostics is not None:
            diagnostics.update(additional_diagnostics)
        return RelaxationResult(
            heightmap_stable=stable,
            heightmap_sequence=sequence,
            iteration_count=iteration_count,
            volume_before_m3=before,
            volume_after_m3=after,
            boundary_outflow_m3=outflow,
            converged=converged,
            diagnostics=diagnostics,
        )
