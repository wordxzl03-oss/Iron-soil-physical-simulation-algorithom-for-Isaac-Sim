"""Warp execution primitive for accepted compact MiniSlope edge batches.

The operator consumes :class:`CompactActiveEdgeBatch` instead of rebuilding
ownership or propagation rules.  It is a phase primitive: the existing
frontier controller owns activation, retirement, and numerical safety limits.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..performance import WarpRuntime
from .sparse_tile_slope import CompactActiveEdgeBatch


_KERNELS: dict[int, Any] = {}


def _kernels(wp: Any) -> Any:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.func
    def mark_tile_halo(
        row: int,
        col: int,
        rows: int,
        cols: int,
        tile_size: int,
        tiles_x: int,
        tile_flags: wp.array(dtype=wp.int32),
    ):
        # Exact device equivalent of CompactTileFrontier.tiles_around_cells
        # with the one-cell MiniSlope stencil.
        for offset_row in range(-1, 2):
            for offset_col in range(-1, 2):
                neighbor_row = row + offset_row
                neighbor_col = col + offset_col
                if (
                    neighbor_row >= 0
                    and neighbor_row < rows
                    and neighbor_col >= 0
                    and neighbor_col < cols
                ):
                    tile = (
                        neighbor_row // tile_size * tiles_x
                        + neighbor_col // tile_size
                    )
                    wp.atomic_max(tile_flags, tile, 1)

    @wp.kernel
    def relax_edges(
        height: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        reached: wp.array(dtype=wp.int32),
        tile_ids: wp.array(dtype=wp.int32),
        a_i: wp.array(dtype=wp.int32),
        a_j: wp.array(dtype=wp.int32),
        b_i: wp.array(dtype=wp.int32),
        b_j: wp.array(dtype=wp.int32),
        rows: int,
        cols: int,
        tile_size: int,
        tiles_x: int,
        critical_difference: wp.float64,
        moved_tile_flags: wp.array(dtype=wp.int32),
        moved_count: wp.array(dtype=wp.int32),
    ):
        edge = wp.tid()
        a = a_i[edge] * cols + a_j[edge]
        b = b_i[edge] * cols + b_j[edge]
        if reached[a] == 0 and reached[b] == 0:
            return
        difference = height[a] - height[b]
        excess = wp.abs(difference) - critical_difference
        if excess > 0.0:
            signed_excess = excess
            if difference < 0.0:
                signed_excess = -signed_excess
            weight_a = weights[a]
            weight_b = weights[b]
            weight_sum = weight_a + weight_b
            transfer_a = signed_excess * weight_b / weight_sum
            transfer_b = signed_excess * weight_a / weight_sum
            # CompactActiveEdgeBatch uses a two-colour matching: within one
            # direction/phase no vertex occurs in two edges. Direct paired
            # writes are therefore race-free and preserve the same float64
            # arithmetic as the accepted CPU compact-frontier baseline. The
            # asymmetric endpoint changes conserve the production triangle-
            # mesh control-area integral, including boundary vertices.
            height[a] = height[a] - transfer_a
            height[b] = height[b] + transfer_b
            reached[a] = 1
            reached[b] = 1
            wp.atomic_add(moved_count, 0, 1)
            mark_tile_halo(
                a_i[edge], a_j[edge], rows, cols, tile_size, tiles_x,
                moved_tile_flags,
            )
            mark_tile_halo(
                b_i[edge], b_j[edge], rows, cols, tile_size, tiles_x,
                moved_tile_flags,
            )

    @wp.kernel
    def unstable_edges(
        height: wp.array(dtype=wp.float64),
        reached: wp.array(dtype=wp.int32),
        tile_ids: wp.array(dtype=wp.int32),
        a_i: wp.array(dtype=wp.int32),
        a_j: wp.array(dtype=wp.int32),
        b_i: wp.array(dtype=wp.int32),
        b_j: wp.array(dtype=wp.int32),
        width: int,
        critical_difference: wp.float64,
        tolerance: wp.float64,
        unstable_tile_flags: wp.array(dtype=wp.int32),
    ):
        edge = wp.tid()
        a = a_i[edge] * width + a_j[edge]
        b = b_i[edge] * width + b_j[edge]
        if reached[a] != 0 or reached[b] != 0:
            excess = wp.abs(height[a] - height[b]) - critical_difference
            if excess > tolerance:
                wp.atomic_max(unstable_tile_flags, tile_ids[edge], 1)

    result = (relax_edges, unstable_edges)
    _KERNELS[id(wp)] = result
    return result


@dataclass(frozen=True)
class WarpFrontierPhaseResult:
    moved_edge_count: int
    triggered_tile_ids: np.ndarray
    unstable_owner_tile_ids: np.ndarray


class WarpCompactActiveEdgeOperator:
    """GPU-resident height/reach state with cached compact edge buffers."""

    backend_name = "GPU_WARP_COMPACT_ACTIVE_EDGE_FRONTIER"

    def __init__(
        self,
        shape: tuple[int, int],
        *,
        device: str = "cuda:0",
        runtime: WarpRuntime | None = None,
        tile_size: int = 64,
        batch_cache_limit: int = 32,
    ) -> None:
        self.shape = int(shape[0]), int(shape[1])
        self.tile_size = int(tile_size)
        if self.tile_size < 1:
            raise ValueError("[WarpMiniSlope] tile_size must be positive")
        self.batch_cache_limit = int(batch_cache_limit)
        if self.batch_cache_limit < 1:
            raise ValueError("[WarpMiniSlope] batch_cache_limit must be positive")
        self.tile_shape = (
            (self.shape[0] + self.tile_size - 1) // self.tile_size,
            (self.shape[1] + self.tile_size - 1) // self.tile_size,
        )
        self.tile_count = self.tile_shape[0] * self.tile_shape[1]
        self.runtime = runtime or WarpRuntime(device)
        self._owns_runtime = runtime is None
        self._height_key = "height"
        self._weights_key = "frontier_weights"
        self._reached_key = "reached"
        self._batch_cache: OrderedDict[
            tuple[bytes, int, int], tuple[tuple[str, ...], tuple[Any, ...]]
        ] = OrderedDict()
        self._batch_serial = 0
        self._initialized = False
        wp = self.runtime.wp
        self._moved_tile_flags = wp.zeros(
            self.tile_count, dtype=wp.int32, device=self.runtime.device
        )
        self._unstable_tile_flags = wp.zeros(
            self.tile_count, dtype=wp.int32, device=self.runtime.device
        )
        self._moved_count = wp.zeros(1, dtype=wp.int32, device=self.runtime.device)

    def initialize(
        self,
        height_xy: np.ndarray,
        reached_xy: np.ndarray,
        vertex_weights_xy_m2: np.ndarray | None = None,
    ) -> None:
        height = np.asarray(height_xy, dtype=np.float64)
        reached = np.asarray(reached_xy, dtype=np.int32)
        if height.shape != self.shape or reached.shape != self.shape:
            raise ValueError("[WarpMiniSlope] resident state shape mismatch")
        weights = (
            np.ones(self.shape, dtype=np.float64)
            if vertex_weights_xy_m2 is None
            else np.asarray(vertex_weights_xy_m2, dtype=np.float64)
        )
        if (
            weights.shape != self.shape
            or not np.all(np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("[WarpMiniSlope] vertex weights must be finite/positive")
        self.runtime.upload(self._height_key, height.ravel(), dtype=self.runtime.wp.float64)
        self.runtime.upload(
            self._weights_key, weights.ravel(), dtype=self.runtime.wp.float64
        )
        self.runtime.upload(self._reached_key, reached.ravel(), dtype=self.runtime.wp.int32)
        self._initialized = True

    def bind_device_state(self, state: Any) -> None:
        """Bind the frontier primitive to DeviceBulkState.Resting in place."""

        if getattr(state, "runtime", None) is not self.runtime:
            raise ValueError("[WarpMiniSlope] state/runtime ownership mismatch")
        if tuple(getattr(state, "shape", ())) != self.shape:
            raise ValueError("[WarpMiniSlope] state shape mismatch")
        if int(getattr(state, "tile_size", -1)) != self.tile_size:
            raise ValueError("[WarpMiniSlope] state/frontier tile-size mismatch")
        required = {"resting", "weights", "frontier_reached"}
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(
                f"[WarpMiniSlope] missing shared state arrays: {sorted(missing)}"
            )
        self._height_key = "resting"
        self._weights_key = "weights"
        self._reached_key = "frontier_reached"
        self._initialized = True

    def _resident_batch(self, batch: CompactActiveEdgeBatch) -> tuple[Any, ...]:
        key = (
            batch.a_i.tobytes() + batch.a_j.tobytes(),
            batch.direction_index,
            batch.phase,
        )
        cached = self._batch_cache.get(key)
        if cached is not None:
            self._batch_cache.move_to_end(key)
            return cached[1]
        wp = self.runtime.wp
        prefix = f"edge_{self._batch_serial}"
        self._batch_serial += 1
        values = (
                ("tile", batch.tile_ids),
                ("ai", batch.a_i),
                ("aj", batch.a_j),
                ("bi", batch.b_i),
                ("bj", batch.b_j),
        )
        names = tuple(f"{prefix}_{name}" for name, _ in values)
        arrays = tuple(
            self.runtime.upload(name, value, dtype=wp.int32)
            for name, (_, value) in zip(names, values)
        )
        self._batch_cache[key] = (names, arrays)
        self._batch_cache.move_to_end(key)
        while len(self._batch_cache) > self.batch_cache_limit:
            _, (retired_names, _) = self._batch_cache.popitem(last=False)
            # Every consumer synchronizes before requesting the next batch.
            # Remove both ownership maps so Warp's device pool can reuse the
            # evicted buffers instead of leaking one edge list per round.
            for name in retired_names:
                self.runtime.release(name)
        return arrays

    def run_phase(
        self,
        batch: CompactActiveEdgeBatch,
        *,
        critical_difference_m: float,
        tolerance_m: float,
    ) -> WarpFrontierPhaseResult:
        """Compatibility operation: transfer then scan the same edge batch."""

        moved = self.relax_phase(
            batch, critical_difference_m=critical_difference_m
        )
        unstable = self.scan_phase(
            batch,
            critical_difference_m=critical_difference_m,
            tolerance_m=tolerance_m,
        )
        return WarpFrontierPhaseResult(
            moved_edge_count=moved.moved_edge_count,
            triggered_tile_ids=moved.triggered_tile_ids,
            unstable_owner_tile_ids=unstable.unstable_owner_tile_ids,
        )

    def relax_phase(
        self,
        batch: CompactActiveEdgeBatch,
        *,
        critical_difference_m: float,
    ) -> WarpFrontierPhaseResult:
        if not self._initialized:
            raise RuntimeError("[WarpMiniSlope] initialize resident state first")
        if batch.edge_count == 0:
            return WarpFrontierPhaseResult(
                0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
            )
        wp = self.runtime.wp
        tile, ai, aj, bi, bj = self._resident_batch(batch)
        self._moved_tile_flags.zero_()
        self._moved_count.zero_()
        relax, _ = _kernels(wp)
        self.runtime.launch(
            relax,
            dim=batch.edge_count,
            inputs=[
                self.runtime.arrays[self._height_key],
                self.runtime.arrays[self._weights_key],
                self.runtime.arrays[self._reached_key],
                tile,
                ai,
                aj,
                bi,
                bj,
                self.shape[0],
                self.shape[1],
                self.tile_size,
                self.tile_shape[1],
                float(critical_difference_m),
                self._moved_tile_flags,
                self._moved_count,
            ],
        )
        self.runtime.synchronize()
        moved_tiles_host = np.asarray(
            self._moved_tile_flags.numpy(), dtype=np.int32
        )
        moved_count_host = np.asarray(self._moved_count.numpy(), dtype=np.int32)
        self.runtime.telemetry.record_d2h(moved_tiles_host)
        self.runtime.telemetry.record_d2h(moved_count_host)
        return WarpFrontierPhaseResult(
            moved_edge_count=int(moved_count_host[0]),
            triggered_tile_ids=np.flatnonzero(moved_tiles_host).astype(np.int32),
            unstable_owner_tile_ids=np.empty(0, dtype=np.int32),
        )

    def scan_phase(
        self,
        batch: CompactActiveEdgeBatch,
        *,
        critical_difference_m: float,
        tolerance_m: float,
    ) -> WarpFrontierPhaseResult:
        if not self._initialized:
            raise RuntimeError("[WarpMiniSlope] initialize resident state first")
        if batch.edge_count == 0:
            return WarpFrontierPhaseResult(
                0, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
            )
        wp = self.runtime.wp
        tile, ai, aj, bi, bj = self._resident_batch(batch)
        self._unstable_tile_flags.zero_()
        _, scan = _kernels(wp)
        self.runtime.launch(
            scan,
            dim=batch.edge_count,
            inputs=[
                self.runtime.arrays[self._height_key],
                self.runtime.arrays[self._reached_key],
                tile,
                ai,
                aj,
                bi,
                bj,
                self.shape[1],
                float(critical_difference_m),
                float(tolerance_m),
                self._unstable_tile_flags,
            ],
        )
        self.runtime.synchronize()
        unstable_tiles_host = np.asarray(
            self._unstable_tile_flags.numpy(), dtype=np.int32
        )
        self.runtime.telemetry.record_d2h(unstable_tiles_host)
        return WarpFrontierPhaseResult(
            moved_edge_count=0,
            triggered_tile_ids=np.empty(0, dtype=np.int32),
            unstable_owner_tile_ids=np.flatnonzero(unstable_tiles_host).astype(
                np.int32
            ),
        )

    def download_state(self) -> tuple[np.ndarray, np.ndarray]:
        height = self.runtime.download(self._height_key).reshape(self.shape)
        reached = self.runtime.download(self._reached_key).reshape(self.shape).astype(bool)
        return height, reached

    def diagnostics(self) -> dict[str, object]:
        result = self.runtime.diagnostics(self.backend_name)
        result.update(
            {
                "compact_batch_semantics": "CompactActiveEdgeBatch",
                "resident_state": [
                    self._height_key,
                    self._weights_key,
                    self._reached_key,
                ],
                "cached_edge_batch_count": len(self._batch_cache),
                "edge_batch_cache_limit": self.batch_cache_limit,
                "phase_host_output": "COMPACT_TILE_FLAGS_AND_SCALAR_MOVED_COUNT",
            }
        )
        return result
