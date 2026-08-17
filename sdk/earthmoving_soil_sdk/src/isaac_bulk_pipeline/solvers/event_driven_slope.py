"""Dynamically expanding, batched-frontier MiniSlope backend."""

from __future__ import annotations

from math import radians, tan
from time import perf_counter
from typing import Any

import numpy as np

from ..performance import ActiveDomainManager, ActiveReason, GridWindow
from .base_solver import RelaxationResult
from .minimum_slope_adapter import MinimumSlopeAdapter


class EventDrivenMinimumSlopeAdapter(MinimumSlopeAdapter):
    """Relax only instability reachable from the current terrain event.

    Changed cells are seeds, never a final region of interest.  The solver
    performs the reference MiniSlope direction/phase schedule on a batched
    cell/tile frontier.  Every conservative transfer marks both endpoints as
    reached, so the next stencil view expands across tile boundaries for as
    long as the avalanche creates a reachable unstable edge.

    ``reached`` is deliberately monotone for one solve.  A stable neighbour
    edge is therefore checked again after later transfers alter either of its
    endpoints.  There is no runout estimate, fixed safety margin, maximum
    active-domain size, or serial Python cell queue.
    """

    backend_name = "CPU_OPTIMIZED_REACHABLE_BBOX_BASELINE"
    LOCALIZED_AVALANCHE = "LOCALIZED_AVALANCHE"
    LARGE_SUSTAINED_AVALANCHE = "LARGE_SUSTAINED_AVALANCHE"
    NO_EVENT = "NO_EVENT"

    def __init__(self, *args, tile_size: int = 32, **kwargs) -> None:
        self.tile_size = int(tile_size)
        if self.tile_size < 4:
            raise ValueError("[EventDrivenMiniSlope] tile_size must be >= 4")
        self._reference_height: np.ndarray | None = None
        self._pending_changed: np.ndarray | None = None
        self._domain: ActiveDomainManager | None = None
        self.last_active_window = GridWindow(0, 0, 0, 0)
        self.last_active_ratio = 0.0
        self.last_frontier_cell_count = 0
        self.last_reached_cell_count = 0
        self.last_solve_ms = 0.0
        self.last_classification = self.NO_EVENT
        self.last_diagnostics: dict[str, Any] = {}
        super().__init__(*args, **kwargs)

    def initialize(self, config, grid=None) -> None:
        super().initialize(config, grid)
        if self.config.large_avalanche_iteration_threshold < 1:
            raise ValueError(
                "[EventDrivenMiniSlope] large avalanche threshold must be >= 1"
            )
        if (
            self.config.numerical_safety_max_iterations
            < self.config.large_avalanche_iteration_threshold
        ):
            raise ValueError(
                "[EventDrivenMiniSlope] numerical safety limit must be >= "
                "large avalanche threshold"
            )
        self._domain = ActiveDomainManager(self.grid.shape, self.tile_size)
        self._reference_height = None
        self._pending_changed = None
        self._clear_diagnostics()

    def set_reference_height(self, heightmap: np.ndarray) -> None:
        self._reference_height = self._validated_copy(heightmap)
        self._pending_changed = np.zeros(self.grid.shape, dtype=bool)

    def mark_changed(self, changed_mask: np.ndarray) -> None:
        changed = np.asarray(changed_mask, dtype=bool)
        if changed.shape != self.grid.shape:
            raise ValueError("[EventDrivenMiniSlope] changed mask shape mismatch")
        if self._pending_changed is None:
            self._pending_changed = np.array(changed, copy=True)
        else:
            self._pending_changed |= changed

    def solve(self, heightmap: np.ndarray) -> RelaxationResult:
        start = perf_counter()
        initial = self._validated_copy(heightmap)
        if self._reference_height is None:
            self.set_reference_height(initial)
        assert self._reference_height is not None
        changed = np.abs(initial - self._reference_height) > max(
            self.config.tolerance, 1.0e-12
        )
        if self._pending_changed is not None:
            changed |= self._pending_changed

        if not np.any(changed):
            stats = self._unchanged_stats(initial)
            diagnostics = self._event_diagnostics(
                changed,
                changed,
                iterations=0,
                tile_history=(),
                classification=self.NO_EVENT,
            )
            result = self._make_result(
                initial,
                initial,
                sequence=self._endpoint_sequence(initial, initial),
                iteration_count=0,
                converged=True,
                legacy_stats=stats,
                additional_diagnostics=diagnostics,
            )
            self._record_completion(initial, changed, diagnostics, start)
            self._last_result = result
            return result

        stable, reached, iterations, stats, tile_history = self._relax_dynamic_frontier(
            initial, changed
        )
        reached_ratio = float(np.count_nonzero(reached)) / float(reached.size)
        classification = (
            self.LARGE_SUSTAINED_AVALANCHE
            if reached_ratio >= 0.5
            or iterations > self.config.large_avalanche_iteration_threshold
            else self.LOCALIZED_AVALANCHE
        )
        diagnostics = self._event_diagnostics(
            changed,
            reached,
            iterations=iterations,
            tile_history=tile_history,
            classification=classification,
        )
        result = self._make_result(
            initial,
            stable,
            sequence=self._endpoint_sequence(initial, stable),
            iteration_count=iterations,
            converged=True,
            legacy_stats=stats,
            additional_diagnostics=diagnostics,
        )
        self._record_completion(stable, reached, diagnostics, start)
        self._last_result = result
        return result

    def solve_sequence(self, heightmap: np.ndarray) -> RelaxationResult:
        # The high-performance contract retains endpoints.  A future backend
        # may expose sampled frontier rounds without changing solver physics.
        return self.solve(heightmap)

    def step(self, heightmap: np.ndarray) -> RelaxationResult:
        """Complete the seeded event; never fall back to a full-grid step."""

        return self.solve(heightmap)

    def reset(self) -> None:
        super().reset()
        self._reference_height = None
        self._pending_changed = None
        if self._domain is not None:
            self._domain.clear()
        self._clear_diagnostics()

    def _relax_dynamic_frontier(
        self, initial_yx: np.ndarray, seeds_yx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, Any, tuple[int, ...]]:
        """Run reference-ordered vector phases on a growing reachable set."""

        from slope_model import RelaxationStats, _neighbor_pairs, _paired_views
        from slope_model import max_neighbor_slope

        is_open = self.config.boundary_condition == "open"
        if is_open:
            source_yx = np.pad(
                initial_yx,
                1,
                mode="constant",
                constant_values=self.config.boundary_height_m,
            )
            source_seeds_yx = np.pad(seeds_yx, 1, mode="constant")
        else:
            source_yx = initial_yx
            source_seeds_yx = seeds_yx

        # Match the legacy H[x,y] storage and direction schedule exactly.
        h_xy = np.ascontiguousarray(source_yx.T)
        reached_xy = np.ascontiguousarray(source_seeds_yx.T)
        seed_indices = np.nonzero(reached_xy)
        reached_bounds = [
            int(seed_indices[0].min()),
            int(seed_indices[0].max()),
            int(seed_indices[1].min()),
            int(seed_indices[1].max()),
        ]
        active_tiles_xy = np.zeros(
            (
                (h_xy.shape[0] + self.tile_size - 1) // self.tile_size,
                (h_xy.shape[1] + self.tile_size - 1) // self.tile_size,
            ),
            dtype=bool,
        )
        active_tiles_xy[
            seed_indices[0] // self.tile_size,
            seed_indices[1] // self.tile_size,
        ] = True
        critical_slope = tan(radians(self.config.critical_angle_deg))
        neighbors = _neighbor_pairs(self.grid.dx, self.grid.dy)
        tile_history: list[int] = [int(np.count_nonzero(active_tiles_xy))]
        iterations = 0

        # The avalanche threshold is diagnostic only.  The separate numerical
        # safety limit raises an explicit failure instead of returning a
        # truncated physical state.
        while True:
            iterations += 1
            if iterations > self.config.numerical_safety_max_iterations:
                raise NumericalNonconvergenceError(
                    iterations=iterations - 1,
                    safety_limit=self.config.numerical_safety_max_iterations,
                    active_cell_count=int(np.count_nonzero(reached_xy)),
                    active_tile_count=int(np.count_nonzero(active_tiles_xy)),
                )
            maximum_reachable_excess = 0.0
            for di, dj, distance in neighbors:
                for phase in (0, 1):
                    window = self._stencil_window_xy(reached_xy.shape, reached_bounds)
                    h_crop = h_xy[window]
                    reached_crop = reached_xy[window]
                    a, b = _paired_views(h_crop, di, dj)
                    reached_a, reached_b = _paired_views(reached_crop, di, dj)
                    difference = a - b
                    excess = np.maximum(
                        np.abs(difference) - critical_slope * distance, 0.0
                    )
                    reachable_edge = reached_a | reached_b
                    if np.any(reachable_edge):
                        maximum_reachable_excess = max(
                            maximum_reachable_excess,
                            float(excess[reachable_edge].max(initial=0.0)),
                        )
                    phase_mask = self._global_phase_mask(
                        a.shape, di, window, phase
                    )
                    transfer = 0.5 * excess * np.sign(difference)
                    transfer *= phase_mask
                    transfer *= reachable_edge
                    transferred = transfer != 0.0
                    if np.any(transferred):
                        a -= transfer
                        b += transfer
                        reached_a |= transferred
                        reached_b |= transferred
                        self._expand_reached_bounds_and_tiles(
                            transferred,
                            di,
                            dj,
                            window,
                            reached_bounds,
                            active_tiles_xy,
                        )

            tile_history.append(int(np.count_nonzero(active_tiles_xy)))
            if maximum_reachable_excess <= self.config.tolerance:
                # A transfer in the final sub-phase can alter an edge whose
                # direction was visited earlier in this iteration.  Confirm
                # the post-transfer state before declaring the reachable
                # avalanche exhausted.
                remaining = self._maximum_reachable_excess(
                    h_xy,
                    reached_xy,
                    reached_bounds,
                    neighbors,
                    critical_slope,
                    _paired_views,
                )
                if remaining <= self.config.tolerance:
                    break

        stable_source_yx = np.ascontiguousarray(h_xy.T)
        reached_source_yx = np.ascontiguousarray(reached_xy.T)
        if is_open:
            stable_yx = np.ascontiguousarray(stable_source_yx[1:-1, 1:-1])
            reached_yx = np.ascontiguousarray(reached_source_yx[1:-1, 1:-1])
        else:
            stable_yx = stable_source_yx
            reached_yx = reached_source_yx
        if is_open:
            # Ghost cells are part of the open-boundary stencil, not terrain
            # tiles exposed in canonical diagnostics.
            tile_history[-1] = self._tile_count(reached_source_yx, True)
        stats = RelaxationStats(
            iterations=iterations,
            converged=True,
            max_slope=max_neighbor_slope(
                h_xy, (self.grid.dx, self.grid.dy)
            ),
            volume_before=float(source_yx.sum() * self.grid.cell_area),
            volume_after=float(stable_source_yx.sum() * self.grid.cell_area),
        )
        return stable_yx, reached_yx, iterations, stats, tuple(tile_history)

    def _maximum_reachable_excess(
        self,
        h_xy: np.ndarray,
        reached_xy: np.ndarray,
        reached_bounds: list[int],
        neighbors: list[tuple[int, int, float]],
        critical_slope: float,
        paired_views,
    ) -> float:
        window = self._stencil_window_xy(h_xy.shape, reached_bounds)
        h_crop = h_xy[window]
        reached_crop = reached_xy[window]
        maximum = 0.0
        for di, dj, distance in neighbors:
            a, b = paired_views(h_crop, di, dj)
            reached_a, reached_b = paired_views(reached_crop, di, dj)
            reachable = reached_a | reached_b
            if not np.any(reachable):
                continue
            excess = np.abs(a - b) - critical_slope * distance
            maximum = max(maximum, float(excess[reachable].max(initial=0.0)))
        return maximum

    @staticmethod
    def _global_phase_mask(
        shape: tuple[int, int],
        di: int,
        window: tuple[slice, slice],
        phase: int,
    ) -> np.ndarray:
        if di:
            start = int(window[0].start or 0)
            coordinate = np.arange(start, start + shape[0], dtype=np.int64)[:, None]
        else:
            start = int(window[1].start or 0)
            coordinate = np.arange(start, start + shape[1], dtype=np.int64)[None, :]
        return (coordinate & 1) == phase

    def _stencil_window_xy(
        self, shape: tuple[int, int], bounds: list[int]
    ) -> tuple[slice, slice]:
        """Current reached-tile envelope plus one-cell neighbour stencil."""

        i_min, i_max, j_min, j_max = bounds
        i0 = max(0, (i_min // self.tile_size) * self.tile_size - 1)
        i1 = min(
            shape[0], ((i_max // self.tile_size) + 1) * self.tile_size + 1
        )
        j0 = max(0, (j_min // self.tile_size) * self.tile_size - 1)
        j1 = min(
            shape[1], ((j_max // self.tile_size) + 1) * self.tile_size + 1
        )
        # A 2-D MiniSlope view must have two cells in each dimension.
        if i1 - i0 < 2:
            i0, i1 = max(0, i0 - 1), min(shape[0], i1 + 1)
        if j1 - j0 < 2:
            j0, j1 = max(0, j0 - 1), min(shape[1], j1 + 1)
        return slice(i0, i1), slice(j0, j1)

    def _expand_reached_bounds_and_tiles(
        self,
        transferred: np.ndarray,
        di: int,
        dj: int,
        window: tuple[slice, slice],
        bounds: list[int],
        active_tiles_xy: np.ndarray,
    ) -> None:
        """Batch-promote transfer endpoints into cell and tile frontiers."""

        local_i, local_j = np.nonzero(transferred)
        a_i = local_i + int(window[0].start or 0)
        a_j = local_j + int(window[1].start or 0) + (1 if dj == -1 else 0)
        b_i = a_i + di
        b_j = a_j + dj
        bounds[0] = min(bounds[0], int(a_i.min()), int(b_i.min()))
        bounds[1] = max(bounds[1], int(a_i.max()), int(b_i.max()))
        bounds[2] = min(bounds[2], int(a_j.min()), int(b_j.min()))
        bounds[3] = max(bounds[3], int(a_j.max()), int(b_j.max()))
        active_tiles_xy[a_i // self.tile_size, a_j // self.tile_size] = True
        active_tiles_xy[b_i // self.tile_size, b_j // self.tile_size] = True

    def _tile_count(self, reached_source_yx: np.ndarray, is_open: bool) -> int:
        reached_yx = (
            reached_source_yx[1:-1, 1:-1] if is_open else reached_source_yx
        )
        rows = np.arange(0, reached_yx.shape[0], self.tile_size)
        cols = np.arange(0, reached_yx.shape[1], self.tile_size)
        row_reduced = np.logical_or.reduceat(reached_yx, rows, axis=0)
        tiles = np.logical_or.reduceat(row_reduced, cols, axis=1)
        return int(np.count_nonzero(tiles))

    def _unchanged_stats(self, initial: np.ndarray):
        from slope_model import RelaxationStats, max_neighbor_slope

        return RelaxationStats(
            iterations=0,
            converged=True,
            max_slope=max_neighbor_slope(
                np.ascontiguousarray(initial.T),
                (self.grid.dx, self.grid.dy),
            ),
            volume_before=self.grid.compute_volume(initial),
            volume_after=self.grid.compute_volume(initial),
        )

    def _event_diagnostics(
        self,
        seeds_yx: np.ndarray,
        reached_yx: np.ndarray,
        *,
        iterations: int,
        tile_history: tuple[int, ...],
        classification: str,
    ) -> dict[str, Any]:
        seed_window = ActiveDomainManager.window_from_mask(seeds_yx)
        reached_window = ActiveDomainManager.window_from_mask(reached_yx)
        reached_count = int(np.count_nonzero(reached_yx))
        reached_ratio = reached_count / float(reached_yx.size)
        initial_tiles = tile_history[0] if tile_history else 0
        final_tiles = tile_history[-1] if tile_history else 0
        return {
            "adapter": self.__class__.__name__,
            "backend": self.backend_name,
            "active_domain_policy": "dynamically_expanding_reachable_frontier",
            "frontier_execution": (
                "batched_vector_phases_over_reached_bounding_rectangle"
            ),
            "true_sparse_tile_frontier": False,
            "active_tiles_drive_numerical_work": False,
            "fixed_final_roi": False,
            "serial_python_cell_queue": False,
            "propagation_truncated": False,
            "seed_cell_count": int(np.count_nonzero(seeds_yx)),
            "reached_cell_count": reached_count,
            "reached_cell_ratio": reached_ratio,
            "initial_active_tile_count": initial_tiles,
            "active_tile_count": final_tiles,
            "tile_expansion_count": max(final_tiles - initial_tiles, 0),
            "active_tile_history": list(tile_history),
            "tile_size": self.tile_size,
            "seed_bbox_yx": self._window_values(seed_window),
            "propagation_bbox_yx": self._window_values(reached_window),
            "propagation_crossed_seed_bbox": reached_window != seed_window,
            "avalanche_classification": classification,
            "large_avalanche_iteration_threshold": (
                self.config.large_avalanche_iteration_threshold
            ),
            "numerical_safety_max_iterations": (
                self.config.numerical_safety_max_iterations
            ),
            "actual_iterations_to_convergence": iterations,
            "continued_beyond_configured_iteration_threshold": (
                iterations > self.config.large_avalanche_iteration_threshold
            ),
            "continued_beyond_large_avalanche_threshold": (
                iterations > self.config.large_avalanche_iteration_threshold
            ),
            "future_resting_to_mobile_transfer_required": (
                classification == self.LARGE_SUSTAINED_AVALANCHE
            ),
            "future_resting_to_mobile_interface": (
                "reserved; no dynamic transfer applied in this correctness phase"
            ),
        }

    @staticmethod
    def _window_values(window: GridWindow) -> list[int]:
        return [
            window.row_start,
            window.col_start,
            window.row_stop,
            window.col_stop,
        ]

    def _record_completion(
        self,
        stable: np.ndarray,
        reached: np.ndarray,
        diagnostics: dict[str, Any],
        start: float,
    ) -> None:
        self._reference_height = np.array(stable, copy=True)
        self._pending_changed = np.zeros(self.grid.shape, dtype=bool)
        self.last_active_window = ActiveDomainManager.window_from_mask(reached)
        self.last_active_ratio = float(diagnostics["reached_cell_ratio"])
        self.last_frontier_cell_count = int(diagnostics["seed_cell_count"])
        self.last_reached_cell_count = int(diagnostics["reached_cell_count"])
        self.last_solve_ms = (perf_counter() - start) * 1_000.0
        self.last_classification = str(diagnostics["avalanche_classification"])
        self.last_diagnostics = dict(diagnostics)
        assert self._domain is not None
        self._domain.clear()
        if np.any(reached):
            self._domain.mark_mask(reached, ActiveReason.SLOPE)

    def _clear_diagnostics(self) -> None:
        self.last_active_window = GridWindow(0, 0, 0, 0)
        self.last_active_ratio = 0.0
        self.last_frontier_cell_count = 0
        self.last_reached_cell_count = 0
        self.last_solve_ms = 0.0
        self.last_classification = self.NO_EVENT
        self.last_diagnostics = {}


class NumericalNonconvergenceError(RuntimeError):
    """Explicit fail-fast for numerical nonconvergence, never a physics cap."""

    code = "NUMERICAL_NONCONVERGENCE"

    def __init__(
        self,
        *,
        iterations: int,
        safety_limit: int,
        active_cell_count: int,
        active_tile_count: int,
    ) -> None:
        self.iterations = int(iterations)
        self.safety_limit = int(safety_limit)
        self.active_cell_count = int(active_cell_count)
        self.active_tile_count = int(active_tile_count)
        super().__init__(
            "[NUMERICAL_NONCONVERGENCE] MiniSlope did not converge before the "
            f"numerical safety limit; iterations={self.iterations}, "
            f"safety_limit={self.safety_limit}, "
            f"active_cells={self.active_cell_count}, "
            f"active_tiles={self.active_tile_count}"
        )
