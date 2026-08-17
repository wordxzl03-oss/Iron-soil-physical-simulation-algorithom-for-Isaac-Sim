"""Compact active-tile/active-edge MiniSlope backend."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from ..performance import ActiveDomainManager
from .event_driven_slope import (
    EventDrivenMinimumSlopeAdapter,
    NumericalNonconvergenceError,
)
from .base_solver import RelaxationResult


@dataclass(frozen=True)
class CompactActiveEdgeBatch:
    """Contiguous edge buffers shared by CPU and future Warp/GPU kernels."""

    tile_ids: np.ndarray
    a_i: np.ndarray
    a_j: np.ndarray
    b_i: np.ndarray
    b_j: np.ndarray
    direction_index: int
    phase: int

    def __post_init__(self) -> None:
        arrays = []
        for value in (self.tile_ids, self.a_i, self.a_j, self.b_i, self.b_j):
            array = np.ascontiguousarray(value, dtype=np.int32)
            array.setflags(write=False)
            arrays.append(array)
        lengths = {item.size for item in arrays}
        if len(lengths) != 1:
            raise ValueError("[SparseTileMiniSlope] edge buffer lengths differ")
        for name, value in zip(
            ("tile_ids", "a_i", "a_j", "b_i", "b_j"), arrays
        ):
            object.__setattr__(self, name, value)

    @property
    def edge_count(self) -> int:
        return int(self.a_i.size)

    def kernel_buffers(self) -> dict[str, np.ndarray | int]:
        """Return compact buffers without depending on NumPy CPU execution."""

        return {
            "tile_ids": self.tile_ids,
            "a_i": self.a_i,
            "a_j": self.a_j,
            "b_i": self.b_i,
            "b_j": self.b_j,
            "direction_index": self.direction_index,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class IncrementalSlopeProgress:
    """One bounded compact-frontier MiniSlope scheduling result.

    ``heightmap_m`` is the conservative state after at most the requested
    number of frontier rounds.  A caller may commit that partial state and
    return control to the simulator; ``complete`` is true only when no
    reachable unstable edge remains.
    """

    heightmap_m: np.ndarray
    complete: bool
    iteration_count: int
    active_tile_count: int
    reached_cell_count: int
    result: RelaxationResult | None = None

    def __post_init__(self) -> None:
        height = np.ascontiguousarray(self.heightmap_m, dtype=np.float64)
        height.setflags(write=False)
        object.__setattr__(self, "heightmap_m", height)


@dataclass
class _IncrementalSparseSession:
    initial_yx: np.ndarray
    seeds_yx: np.ndarray
    source_yx: np.ndarray
    h_xy: np.ndarray
    reached_xy: np.ndarray
    frontier: "CompactTileFrontier"
    active_tiles: np.ndarray
    ever_active: np.ndarray
    critical_slope: float
    neighbors: list[tuple[int, int, float]]
    is_open: bool
    started_at: float
    iterations: int = 0
    retired_tile_total: int = 0
    edge_evaluations: int = 0
    peak_active_tiles: int = 0
    active_tile_phase_work: int = 0
    frontier_history: list[int] | None = None
    reached_tile_history: list[int] | None = None

    def __post_init__(self) -> None:
        if self.frontier_history is None:
            self.frontier_history = []
        if self.reached_tile_history is None:
            self.reached_tile_history = [int(np.count_nonzero(self.ever_active))]


class CompactTileFrontier:
    """Build unique edge-owner batches from compact active tile IDs."""

    def __init__(self, shape: tuple[int, int], tile_size: int) -> None:
        self.shape = int(shape[0]), int(shape[1])
        self.tile_size = int(tile_size)
        self.tile_shape = (
            (self.shape[0] + tile_size - 1) // tile_size,
            (self.shape[1] + tile_size - 1) // tile_size,
        )
        local_i, local_j = np.indices((tile_size, tile_size), dtype=np.int32)
        self._local_i = local_i.ravel()
        self._local_j = local_j.ravel()
        self._edge_batch_cache: OrderedDict[
            tuple[bytes, int, int, int], CompactActiveEdgeBatch
        ] = OrderedDict()
        self._edge_batch_cache_limit = 32

    @property
    def tile_count(self) -> int:
        return int(self.tile_shape[0] * self.tile_shape[1])

    def tiles_around_cells(
        self, i: np.ndarray, j: np.ndarray, *, stencil_cells: int = 1
    ) -> np.ndarray:
        """Tiles owning every neighbour edge around changed cell endpoints."""

        source_i = np.asarray(i, dtype=np.int64).ravel()
        source_j = np.asarray(j, dtype=np.int64).ravel()
        if source_i.size == 0:
            return np.empty(0, dtype=np.int32)
        offsets = np.arange(-stencil_cells, stencil_cells + 1, dtype=np.int64)
        candidate_i = (source_i[:, None] + offsets[None, :]).ravel()
        candidate_j = (source_j[:, None] + offsets[None, :]).ravel()
        # Cartesian product of the short stencil offsets for every endpoint.
        candidate_i = np.repeat(candidate_i.reshape(source_i.size, -1), offsets.size, axis=1)
        candidate_j = np.tile(candidate_j.reshape(source_j.size, -1), (1, offsets.size))
        candidate_i = candidate_i.ravel()
        candidate_j = candidate_j.ravel()
        valid = (
            (candidate_i >= 0)
            & (candidate_i < self.shape[0])
            & (candidate_j >= 0)
            & (candidate_j < self.shape[1])
        )
        tile_row = candidate_i[valid] // self.tile_size
        tile_col = candidate_j[valid] // self.tile_size
        return np.unique(
            (tile_row * self.tile_shape[1] + tile_col).astype(np.int32)
        )

    def edge_batch(
        self,
        active_tile_ids: np.ndarray,
        *,
        direction_index: int,
        di: int,
        dj: int,
        phase: int,
    ) -> CompactActiveEdgeBatch:
        tile_ids = np.unique(np.asarray(active_tile_ids, dtype=np.int32))
        cache_key = (tile_ids.tobytes(), int(di), int(dj), int(phase))
        cached = self._edge_batch_cache.get(cache_key)
        if cached is not None:
            self._edge_batch_cache.move_to_end(cache_key)
            return cached
        if tile_ids.size == 0:
            empty = np.empty(0, dtype=np.int32)
            batch = CompactActiveEdgeBatch(
                empty, empty, empty, empty, empty, direction_index, phase
            )
            self._cache_batch(cache_key, batch)
            return batch
        tile_rows = tile_ids // self.tile_shape[1]
        tile_cols = tile_ids % self.tile_shape[1]
        a_i = (
            tile_rows[:, None] * self.tile_size + self._local_i[None, :]
        ).ravel()
        a_j = (
            tile_cols[:, None] * self.tile_size + self._local_j[None, :]
        ).ravel()
        owners = np.repeat(tile_ids, self._local_i.size)
        b_i = a_i + int(di)
        b_j = a_j + int(dj)
        coordinate = a_i if di else a_j
        valid = (
            (a_i < self.shape[0])
            & (a_j < self.shape[1])
            & (b_i >= 0)
            & (b_i < self.shape[0])
            & (b_j >= 0)
            & (b_j < self.shape[1])
            & ((coordinate & 1) == int(phase))
        )
        batch = CompactActiveEdgeBatch(
            owners[valid],
            a_i[valid],
            a_j[valid],
            b_i[valid],
            b_j[valid],
            direction_index,
            phase,
        )
        self._cache_batch(cache_key, batch)
        return batch

    def _cache_batch(
        self,
        key: tuple[bytes, int, int, int],
        batch: CompactActiveEdgeBatch,
    ) -> None:
        self._edge_batch_cache[key] = batch
        self._edge_batch_cache.move_to_end(key)
        while len(self._edge_batch_cache) > self._edge_batch_cache_limit:
            self._edge_batch_cache.popitem(last=False)


class SparseTileFrontierMinimumSlopeAdapter(EventDrivenMinimumSlopeAdapter):
    """MiniSlope whose numerical work is driven by compact active tile lists."""

    backend_name = "CPU_OPTIMIZED_COMPACT_TILE_FRONTIER"

    def __init__(self, *args, **kwargs) -> None:
        self._sparse_run_diagnostics: dict[str, Any] = {}
        self._incremental_session: _IncrementalSparseSession | None = None
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self._sparse_run_diagnostics = {}
        self._incremental_session = None

    @property
    def incremental_active(self) -> bool:
        return self._incremental_session is not None

    def begin_incremental(
        self, heightmap: np.ndarray, changed_mask: np.ndarray
    ) -> None:
        """Start a residual relaxation event without doing convergence work."""

        if self._incremental_session is not None:
            raise RuntimeError("[SparseTileMiniSlope] incremental event already active")
        initial = self._validated_copy(heightmap)
        seeds = np.asarray(changed_mask, dtype=bool)
        if seeds.shape != self.grid.shape:
            raise ValueError("[SparseTileMiniSlope] changed mask shape mismatch")
        if not np.any(seeds):
            raise ValueError("[SparseTileMiniSlope] incremental event needs a seed")
        self._incremental_session = self._create_incremental_session(initial, seeds)

    def advance_incremental(self, *, round_budget: int = 1) -> IncrementalSlopeProgress:
        """Advance a scheduled event by a strict frontier-round budget."""

        if not isinstance(round_budget, int) or round_budget < 1:
            raise ValueError("[SparseTileMiniSlope] round_budget must be >= 1")
        session = self._incremental_session
        if session is None:
            raise RuntimeError("[SparseTileMiniSlope] no incremental event is active")
        self._advance_session(session, round_budget)
        partial = self._session_heightmap(session)
        if session.active_tiles.size:
            return IncrementalSlopeProgress(
                partial,
                False,
                session.iterations,
                int(session.active_tiles.size),
                int(np.count_nonzero(session.reached_xy)),
            )
        stable, reached, iterations, stats, tile_history = self._finish_session(session)
        reached_ratio = float(np.count_nonzero(reached)) / float(reached.size)
        classification = (
            self.LARGE_SUSTAINED_AVALANCHE
            if reached_ratio >= 0.5
            or iterations > self.config.large_avalanche_iteration_threshold
            else self.LOCALIZED_AVALANCHE
        )
        diagnostics = self._event_diagnostics(
            session.seeds_yx,
            reached,
            iterations=iterations,
            tile_history=tile_history,
            classification=classification,
        )
        result = self._make_result(
            session.initial_yx,
            stable,
            sequence=self._endpoint_sequence(session.initial_yx, stable),
            iteration_count=iterations,
            converged=True,
            legacy_stats=stats,
            additional_diagnostics=diagnostics,
        )
        self._record_completion(stable, reached, diagnostics, session.started_at)
        self._last_result = result
        self._incremental_session = None
        return IncrementalSlopeProgress(
            stable,
            True,
            iterations,
            0,
            int(np.count_nonzero(reached)),
            result,
        )

    def _relax_dynamic_frontier(
        self, initial_yx: np.ndarray, seeds_yx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, int, Any, tuple[int, ...]]:
        session = self._create_incremental_session(initial_yx, seeds_yx)
        while session.active_tiles.size:
            self._advance_session(session, 1)
        return self._finish_session(session)

    def _create_incremental_session(
        self, initial_yx: np.ndarray, seeds_yx: np.ndarray
    ) -> _IncrementalSparseSession:
        from slope_model import _neighbor_pairs

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

        h_xy = np.ascontiguousarray(source_yx.T)
        reached_xy = np.ascontiguousarray(source_seeds_yx.T)
        frontier = CompactTileFrontier(h_xy.shape, self.tile_size)
        seed_i, seed_j = np.nonzero(reached_xy)
        active_tiles = frontier.tiles_around_cells(seed_i, seed_j)
        ever_active = np.zeros(frontier.tile_count, dtype=bool)
        ever_active[active_tiles] = True
        critical_slope = float(
            np.tan(np.deg2rad(self.config.critical_angle_deg))
        )
        neighbors = _neighbor_pairs(self.grid.dx, self.grid.dy)
        return _IncrementalSparseSession(
            initial_yx=np.ascontiguousarray(initial_yx),
            seeds_yx=np.ascontiguousarray(seeds_yx),
            source_yx=source_yx,
            h_xy=h_xy,
            reached_xy=reached_xy,
            frontier=frontier,
            active_tiles=active_tiles,
            ever_active=ever_active,
            critical_slope=critical_slope,
            neighbors=neighbors,
            is_open=is_open,
            started_at=perf_counter(),
            peak_active_tiles=int(active_tiles.size),
        )

    def _advance_session(
        self, session: _IncrementalSparseSession, round_budget: int
    ) -> None:
        for _ in range(round_budget):
            if session.active_tiles.size == 0:
                return
            session.iterations += 1
            if session.iterations > self.config.numerical_safety_max_iterations:
                raise NumericalNonconvergenceError(
                    iterations=session.iterations - 1,
                    safety_limit=self.config.numerical_safety_max_iterations,
                    active_cell_count=int(np.count_nonzero(session.reached_xy)),
                    active_tile_count=int(session.active_tiles.size),
                )
            phase_tiles = session.active_tiles
            assert session.frontier_history is not None
            session.frontier_history.append(int(session.active_tiles.size))
            session.peak_active_tiles = max(
                session.peak_active_tiles, int(session.active_tiles.size)
            )

            for direction_index, (di, dj, distance) in enumerate(session.neighbors):
                for phase in (0, 1):
                    session.active_tile_phase_work += int(phase_tiles.size)
                    batch = session.frontier.edge_batch(
                        phase_tiles,
                        direction_index=direction_index,
                        di=di,
                        dj=dj,
                        phase=phase,
                    )
                    session.edge_evaluations += batch.edge_count
                    if batch.edge_count == 0:
                        continue
                    reachable = (
                        session.reached_xy[batch.a_i, batch.a_j]
                        | session.reached_xy[batch.b_i, batch.b_j]
                    )
                    difference = (
                        session.h_xy[batch.a_i, batch.a_j]
                        - session.h_xy[batch.b_i, batch.b_j]
                    )
                    excess = np.maximum(
                        np.abs(difference) - session.critical_slope * distance, 0.0
                    )
                    transfer = 0.5 * excess * np.sign(difference)
                    transfer *= reachable
                    moved = transfer != 0.0
                    if not np.any(moved):
                        continue
                    a_i, a_j = batch.a_i[moved], batch.a_j[moved]
                    b_i, b_j = batch.b_i[moved], batch.b_j[moved]
                    moved_transfer = transfer[moved]
                    session.h_xy[a_i, a_j] -= moved_transfer
                    session.h_xy[b_i, b_j] += moved_transfer
                    session.reached_xy[a_i, a_j] = True
                    session.reached_xy[b_i, b_j] = True
                    triggered = session.frontier.tiles_around_cells(
                        np.concatenate((a_i, b_i)),
                        np.concatenate((a_j, b_j)),
                    )
                    session.ever_active[triggered] = True
                    phase_tiles = np.union1d(phase_tiles, triggered).astype(
                        np.int32, copy=False
                    )

            unstable_owners, scan_edges = self._unstable_owner_tiles(
                session.h_xy,
                session.reached_xy,
                phase_tiles,
                session.frontier,
                session.neighbors,
                session.critical_slope,
                self.config.tolerance,
            )
            session.edge_evaluations += scan_edges
            session.retired_tile_total += int(
                np.setdiff1d(phase_tiles, unstable_owners, assume_unique=True).size
            )
            session.active_tiles = unstable_owners
            assert session.reached_tile_history is not None
            session.reached_tile_history.append(
                int(np.count_nonzero(session.ever_active))
            )

    def _session_heightmap(self, session: _IncrementalSparseSession) -> np.ndarray:
        source_yx = np.ascontiguousarray(session.h_xy.T)
        return (
            np.ascontiguousarray(source_yx[1:-1, 1:-1])
            if session.is_open
            else source_yx
        )

    def _finish_session(
        self, session: _IncrementalSparseSession
    ) -> tuple[np.ndarray, np.ndarray, int, Any, tuple[int, ...]]:
        from slope_model import RelaxationStats, max_neighbor_slope

        stable_source_yx = np.ascontiguousarray(session.h_xy.T)
        reached_source_yx = np.ascontiguousarray(session.reached_xy.T)
        if session.is_open:
            stable_yx = np.ascontiguousarray(stable_source_yx[1:-1, 1:-1])
            reached_yx = np.ascontiguousarray(reached_source_yx[1:-1, 1:-1])
        else:
            stable_yx = stable_source_yx
            reached_yx = reached_source_yx
        reached_window = ActiveDomainManager.window_from_mask(reached_yx)
        full_edges_per_iteration = sum(
            (session.h_xy.shape[0] - abs(di))
            * (session.h_xy.shape[1] - abs(dj))
            for di, dj, _ in session.neighbors
        ) * 2
        self._sparse_run_diagnostics = {
            "peak_active_tile_count": session.peak_active_tiles,
            "ever_active_tile_count": int(np.count_nonzero(session.ever_active)),
            "frontier_tile_history_rle": self._run_length_encode(
                session.frontier_history or []
            ),
            "retired_tile_total": session.retired_tile_total,
            "active_tile_phase_work": session.active_tile_phase_work,
            "compact_edge_evaluation_count": session.edge_evaluations,
            "full_domain_edge_evaluation_equivalent": (
                session.iterations * full_edges_per_iteration
            ),
            "bounding_box_cell_count": reached_window.cell_count,
        }
        stats = RelaxationStats(
            iterations=session.iterations,
            converged=True,
            max_slope=max_neighbor_slope(
                session.h_xy, (self.grid.dx, self.grid.dy)
            ),
            volume_before=float(session.source_yx.sum() * self.grid.cell_area),
            volume_after=float(stable_source_yx.sum() * self.grid.cell_area),
        )
        return (
            stable_yx,
            reached_yx,
            session.iterations,
            stats,
            (
                (session.reached_tile_history or [0])[0],
                (session.reached_tile_history or [0])[-1],
            ),
        )

    @staticmethod
    def _run_length_encode(values: list[int]) -> list[list[int]]:
        if not values:
            return []
        result: list[list[int]] = []
        start = 0
        current = values[0]
        for index, value in enumerate(values[1:], start=1):
            if value == current:
                continue
            result.append([start, index - 1, current])
            start = index
            current = value
        result.append([start, len(values) - 1, current])
        return result

    @staticmethod
    def _unstable_owner_tiles(
        h_xy: np.ndarray,
        reached_xy: np.ndarray,
        candidate_tiles: np.ndarray,
        frontier: CompactTileFrontier,
        neighbors: list[tuple[int, int, float]],
        critical_slope: float,
        tolerance: float,
    ) -> tuple[np.ndarray, int]:
        owners: list[np.ndarray] = []
        evaluated = 0
        for direction_index, (di, dj, distance) in enumerate(neighbors):
            for phase in (0, 1):
                batch = frontier.edge_batch(
                    candidate_tiles,
                    direction_index=direction_index,
                    di=di,
                    dj=dj,
                    phase=phase,
                )
                evaluated += batch.edge_count
                if batch.edge_count == 0:
                    continue
                reachable = (
                    reached_xy[batch.a_i, batch.a_j]
                    | reached_xy[batch.b_i, batch.b_j]
                )
                excess = (
                    np.abs(
                        h_xy[batch.a_i, batch.a_j]
                        - h_xy[batch.b_i, batch.b_j]
                    )
                    - critical_slope * distance
                )
                unstable = reachable & (excess > tolerance)
                if np.any(unstable):
                    owners.append(batch.tile_ids[unstable])
        if not owners:
            return np.empty(0, dtype=np.int32), evaluated
        return np.unique(np.concatenate(owners)).astype(np.int32), evaluated

    def _event_diagnostics(self, *args, **kwargs) -> dict[str, Any]:
        diagnostics = super()._event_diagnostics(*args, **kwargs)
        diagnostics.update(
            {
                "frontier_execution": "compact_active_tile_and_edge_lists",
                "true_sparse_tile_frontier": True,
                "active_tiles_drive_numerical_work": True,
                "gpu_warp_interface": "CompactActiveEdgeBatch.kernel_buffers",
                "tile_activation_policy": (
                    "batched_transfer_endpoint_stencil_activation"
                ),
                "stable_tile_policy": (
                    "retire_until_retriggered_by_adjacent_transfer"
                ),
                **self._sparse_run_diagnostics,
            }
        )
        return diagnostics
