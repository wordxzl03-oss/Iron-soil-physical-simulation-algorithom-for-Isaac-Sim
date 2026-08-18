"""Explicit host/device authority for the V2 bulk terrain fields.

The legacy :class:`BulkStateManager` owns immutable NumPy fields and remains
the reference implementation.  This module is deliberately separate: a GPU
runtime has exactly one authoritative copy of spatial bulk state -- the Warp
arrays in :class:`DeviceBulkState`.  Host arrays are created only through an
explicit snapshot/patch API and are never writable back through a view.

Keeping this boundary small is important.  It makes an accidental full-grid
download visible in telemetry instead of silently turning a GPU runtime into a
CPU runtime with a GPU shadow copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from ..performance import WarpRuntime
from ..terrain import TerrainGrid


_KERNELS: dict[int, tuple[Any, ...]] = {}
_DIRTY_KERNELS: dict[int, tuple[Any, Any]] = {}
_QUERY_KERNELS: dict[int, Any] = {}
_INT_SCATTER_KERNELS: dict[int, Any] = {}
_FLOAT_ADD_KERNELS: dict[int, Any] = {}
_EXCHANGE_KERNELS: dict[int, tuple[Any, Any]] = {}


def _state_kernels(wp: Any) -> tuple[Any, ...]:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def scatter_float(
        target: wp.array(dtype=wp.float64),
        indices: wp.array(dtype=wp.int32),
        values: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        target[indices[item]] = values[item]

    @wp.kernel
    def gather_surface(
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        indices: wp.array(dtype=wp.int32),
        output: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        output[item] = resting[indices[item]] + mobile[indices[item]]

    @wp.kernel
    def sum_reservoirs(
        z_base: wp.array(dtype=wp.float64),
        b_eff: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        density: wp.float64,
        summary: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        weight = weights[item]
        wp.atomic_add(summary, 0, wp.max(b_eff[item] - z_base[item], wp.float64(0.0)) * weight)
        wp.atomic_add(summary, 1, mobile[item] * weight)
        wp.atomic_add(summary, 2, density * momentum_x[item] * weight)
        wp.atomic_add(summary, 3, density * momentum_y[item] * weight)

    @wp.kernel
    def gather_scalar(
        source: wp.array(dtype=wp.float64),
        indices: wp.array(dtype=wp.int32),
        output: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        output[item] = source[indices[item]]

    result = (scatter_float, gather_surface, sum_reservoirs, gather_scalar)
    _KERNELS[id(wp)] = result
    return result


def _exchange_kernels(wp: Any) -> tuple[Any, Any]:
    """Authoritative, donor-limited static/mobile exchange primitives.

    Both kernels leave ``b_eff + h_mobile`` invariant cell-by-cell.  Entrainment
    deliberately does not write momentum; deposition reports the momentum
    removed from the Mobile reservoir instead of hiding it.
    """

    cached = _EXCHANGE_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def entrain(
        z_base: wp.array(dtype=wp.float64),
        b_eff: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        indices: wp.array(dtype=wp.int32),
        requested_depth: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        summary: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        index = indices[item]
        available = wp.max(b_eff[index] - z_base[index], wp.float64(0.0))
        depth = wp.min(wp.max(requested_depth[item], wp.float64(0.0)), available)
        b_eff[index] = b_eff[index] - depth
        mobile[index] = mobile[index] + depth
        wp.atomic_add(summary, 0, depth * weights[index])

    @wp.kernel
    def deposit(
        b_eff: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        indices: wp.array(dtype=wp.int32),
        requested_depth: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        density: wp.float64,
        summary: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        index = indices[item]
        before = wp.max(mobile[index], wp.float64(0.0))
        depth = wp.min(wp.max(requested_depth[item], wp.float64(0.0)), before)
        fraction = wp.float64(0.0)
        if before > wp.float64(0.0):
            fraction = depth / before
        removed_x = momentum_x[index] * fraction
        removed_y = momentum_y[index] * fraction
        b_eff[index] = b_eff[index] + depth
        mobile[index] = before - depth
        momentum_x[index] = momentum_x[index] - removed_x
        momentum_y[index] = momentum_y[index] - removed_y
        weight = weights[index]
        wp.atomic_add(summary, 0, depth * weight)
        wp.atomic_add(summary, 1, density * removed_x * weight)
        wp.atomic_add(summary, 2, density * removed_y * weight)

    result = (entrain, deposit)
    _EXCHANGE_KERNELS[id(wp)] = result
    return result


def _dirty_kernels(wp: Any) -> tuple[Any, Any]:
    cached = _DIRTY_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def capture_surface(
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        surface_before: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        surface_before[item] = resting[item] + mobile[item]

    @wp.kernel
    def flag_surface_change(
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        surface_before: wp.array(dtype=wp.float64),
        dirty_mask: wp.array(dtype=wp.int32),
        tile_flags: wp.array(dtype=wp.int32),
        cols: int,
        tile_size: int,
        tiles_x: int,
        tolerance: wp.float64,
    ):
        item = wp.tid()
        if wp.abs(resting[item] + mobile[item] - surface_before[item]) > tolerance:
            dirty_mask[item] = 1
            row = item // cols
            col = item - row * cols
            tile = (row // tile_size) * tiles_x + col // tile_size
            wp.atomic_max(tile_flags, tile, 1)

    result = (capture_surface, flag_surface_change)
    _DIRTY_KERNELS[id(wp)] = result
    return result


def _query_kernel(wp: Any) -> Any:
    cached = _QUERY_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def bilinear_surface(
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        query_rows: wp.array(dtype=wp.float64),
        query_cols: wp.array(dtype=wp.float64),
        row_count: int,
        col_count: int,
        output: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        row = wp.clamp(query_rows[item], wp.float64(0.0), wp.float64(row_count - 1))
        col = wp.clamp(query_cols[item], wp.float64(0.0), wp.float64(col_count - 1))
        r0 = int(wp.floor(row))
        c0 = int(wp.floor(col))
        r1 = wp.min(r0 + 1, row_count - 1)
        c1 = wp.min(c0 + 1, col_count - 1)
        fr = row - wp.float64(r0)
        fc = col - wp.float64(c0)
        v00 = resting[r0 * col_count + c0] + mobile[r0 * col_count + c0]
        v01 = resting[r0 * col_count + c1] + mobile[r0 * col_count + c1]
        v10 = resting[r1 * col_count + c0] + mobile[r1 * col_count + c0]
        v11 = resting[r1 * col_count + c1] + mobile[r1 * col_count + c1]
        output[item] = (
            (wp.float64(1.0) - fr) * (wp.float64(1.0) - fc) * v00
            + (wp.float64(1.0) - fr) * fc * v01
            + fr * (wp.float64(1.0) - fc) * v10
            + fr * fc * v11
        )

    _QUERY_KERNELS[id(wp)] = bilinear_surface
    return bilinear_surface


def _int_scatter_kernel(wp: Any) -> Any:
    cached = _INT_SCATTER_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def scatter_int(
        target: wp.array(dtype=wp.int32),
        source_indices: wp.array(dtype=wp.int32),
        source_values: wp.array(dtype=wp.int32),
    ):
        item = wp.tid()
        target[source_indices[item]] = source_values[item]

    _INT_SCATTER_KERNELS[id(wp)] = scatter_int
    return scatter_int


def _float_add_kernel(wp: Any) -> Any:
    cached = _FLOAT_ADD_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def add_float(
        target: wp.array(dtype=wp.float64),
        source_indices: wp.array(dtype=wp.int32),
        source_values: wp.array(dtype=wp.float64),
    ):
        item = wp.tid()
        wp.atomic_add(target, source_indices[item], source_values[item])

    _FLOAT_ADD_KERNELS[id(wp)] = add_float
    return add_float


class BulkStateAuthority(str, Enum):
    """The only permitted owner of mutable terrain fields for one runtime."""

    HOST = "HOST"
    DEVICE = "DEVICE"


class BulkStateAuthorityError(RuntimeError):
    """Raised when a non-authoritative view attempts to mutate physics state."""


@dataclass(frozen=True)
class DeviceTransferSnapshot:
    """Per-step transfer accounting, including explicit full-field detection."""

    h2d_bytes: int
    d2h_bytes: int
    h2d_transfer_count: int
    d2h_transfer_count: int
    synchronization_count: int
    kernel_launch_count: int
    h2d_blocking_ms: float
    d2h_blocking_ms: float
    full_field_h2d_count: int
    full_field_d2h_count: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "h2d_transfer_count": self.h2d_transfer_count,
            "d2h_transfer_count": self.d2h_transfer_count,
            "synchronization_count": self.synchronization_count,
            "kernel_launch_count": self.kernel_launch_count,
            "h2d_blocking_ms": self.h2d_blocking_ms,
            "d2h_blocking_ms": self.d2h_blocking_ms,
            "full_field_h2d_count": self.full_field_h2d_count,
            "full_field_d2h_count": self.full_field_d2h_count,
        }


@dataclass(frozen=True)
class HostBulkStateView:
    """Read-only, non-authoritative host observation of a DeviceBulkState.

    ``source`` is intentionally recorded so acceptance can distinguish an
    allowed checkpoint/debug snapshot from a normal physics-loop transfer.
    The arrays are write-protected and this class has no commit operation.
    """

    H_resting_m: np.ndarray
    H_mobile_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    timestamp_device_s: float
    reset_generation: int
    source: str
    dirty_tile_ids: np.ndarray
    z_base_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("H_resting_m", "H_mobile_m", "mobile_momentum_m2_s"):
            value = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.mobile_momentum_m2_s.shape != self.H_resting_m.shape + (2,):
            raise ValueError("[HostBulkStateView] momentum shape mismatch")
        if self.H_mobile_m.shape != self.H_resting_m.shape:
            raise ValueError("[HostBulkStateView] mobile shape mismatch")
        dirty = np.ascontiguousarray(np.asarray(self.dirty_tile_ids, dtype=np.int32))
        dirty.setflags(write=False)
        object.__setattr__(self, "dirty_tile_ids", dirty)
        base = np.zeros_like(self.H_resting_m) if self.z_base_m is None else np.asarray(self.z_base_m, dtype=np.float64)
        if base.shape != self.H_resting_m.shape:
            raise ValueError("[HostBulkStateView] z_base shape mismatch")
        base = np.ascontiguousarray(base)
        base.setflags(write=False)
        object.__setattr__(self, "z_base_m", base)

    @property
    def b_eff_m(self) -> np.ndarray:
        """Physical static-bed/flowing-layer interface (authoritative surface)."""
        return self.H_resting_m

    @property
    def h_resting_derived_m(self) -> np.ndarray:
        """Compatibility/ledger depth; never a second terrain geometry."""
        return self.b_eff_m - self.z_base_m

    @property
    def H_free_m(self) -> np.ndarray:
        return self.b_eff_m + self.H_mobile_m

    @property
    def authority(self) -> BulkStateAuthority:
        return BulkStateAuthority.DEVICE

    def commit(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise BulkStateAuthorityError(
            "[HostBulkStateView] host views are non-authoritative and cannot commit physics"
        )


@dataclass(frozen=True)
class HostBulkStatePatch:
    """Compact non-authoritative terrain patch for a legacy host operator."""

    bbox_yx: tuple[int, int, int, int]
    H_resting_m: np.ndarray
    H_mobile_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    timestamp_device_s: float
    source: str
    z_base_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        row0, row1, col0, col1 = (int(value) for value in self.bbox_yx)
        expected = (row1 - row0, col1 - col0)
        for name in ("H_resting_m", "H_mobile_m", "mobile_momentum_m2_s"):
            array = np.ascontiguousarray(np.asarray(getattr(self, name), dtype=np.float64))
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.H_resting_m.shape != expected or self.H_mobile_m.shape != expected:
            raise ValueError("[HostBulkStatePatch] scalar field shape mismatch")
        if self.mobile_momentum_m2_s.shape != expected + (2,):
            raise ValueError("[HostBulkStatePatch] momentum field shape mismatch")
        base = np.zeros(expected, dtype=np.float64) if self.z_base_m is None else np.asarray(self.z_base_m, dtype=np.float64)
        if base.shape != expected:
            raise ValueError("[HostBulkStatePatch] z_base shape mismatch")
        base = np.ascontiguousarray(base)
        base.setflags(write=False)
        object.__setattr__(self, "z_base_m", base)

    @property
    def b_eff_m(self) -> np.ndarray:
        return self.H_resting_m

    @property
    def h_resting_derived_m(self) -> np.ndarray:
        return self.b_eff_m - self.z_base_m

    @property
    def H_free_m(self) -> np.ndarray:
        return self.b_eff_m + self.H_mobile_m


@dataclass(frozen=True)
class DeviceMaterialLedgerSnapshot:
    initial_total_m3: float
    resting_m3: float
    mobile_m3: float
    payload_m3: float
    airborne_m3: float
    outflow_m3: float
    absolute_volume_error_m3: float
    relative_volume_error: float


class DeviceMaterialLedger:
    """Scalar material ledger whose terrain entries come from device reductions."""

    def __init__(
        self,
        state: "DeviceBulkState",
        density_kg_m3: float,
        *,
        payload_m3: float = 0.0,
        airborne_m3: float = 0.0,
        outflow_m3: float = 0.0,
        initial_total_m3: float | None = None,
    ) -> None:
        values = state.reservoir_reduction(density_kg_m3)
        self._density_kg_m3 = float(density_kg_m3)
        computed_total = float(
            values["resting_volume_m3"]
            + values["mobile_volume_m3"]
            + payload_m3
            + airborne_m3
            + outflow_m3
        )
        self._initial_total_m3 = (
            computed_total
            if initial_total_m3 is None
            else float(initial_total_m3)
        )
        if not np.isfinite(self._initial_total_m3) or self._initial_total_m3 < 0.0:
            raise ValueError("[DeviceMaterialLedger] initial total must be finite/non-negative")

    @property
    def initial_total_m3(self) -> float:
        """Conserved reference total, exposed for explicit checkpoint boundaries."""

        return self._initial_total_m3

    def snapshot(
        self,
        state: "DeviceBulkState",
        *,
        payload_m3: float,
        airborne_m3: float,
        outflow_m3: float,
    ) -> DeviceMaterialLedgerSnapshot:
        scalars = state.reservoir_reduction(self._density_kg_m3)
        current = float(
            scalars["resting_volume_m3"]
            + scalars["mobile_volume_m3"]
            + float(payload_m3)
            + float(airborne_m3)
            + float(outflow_m3)
        )
        error = current - self._initial_total_m3
        return DeviceMaterialLedgerSnapshot(
            initial_total_m3=self._initial_total_m3,
            resting_m3=float(scalars["resting_volume_m3"]),
            mobile_m3=float(scalars["mobile_volume_m3"]),
            payload_m3=float(payload_m3),
            airborne_m3=float(airborne_m3),
            outflow_m3=float(outflow_m3),
            absolute_volume_error_m3=error,
            relative_volume_error=abs(error) / max(abs(self._initial_total_m3), 1.0e-12),
        )


class DeviceBulkState:
    """Warp-resident authoritative state shared by GPU bulk operators.

    Spatial fields are allocated once and exposed by stable names in the shared
    :class:`WarpRuntime`.  Existing Warp operators can bind those names rather
    than allocate their own shadow terrain.  Host scalar metadata (payload,
    airborne parcels and task state) deliberately stays outside this class.
    """

    backend_identity = "GPU_RUNTIME_DEVICE_AUTHORITATIVE_WARP"

    _FULL_FIELD_NAMES = frozenset(
        {
            "z_base",
            "b_eff",
            "mobile",
            "momentum_x",
            "momentum_y",
            "initial_b_eff",
            "weights",
        }
    )

    def __init__(
        self,
        grid: TerrainGrid,
        initial_resting_m: np.ndarray,
        *,
        device: str = "cuda:0",
        tile_size: int = 64,
    ) -> None:
        self.grid = grid
        self.shape = grid.shape
        self.size = int(grid.nx * grid.ny)
        self.tile_size = int(tile_size)
        if self.tile_size < 4:
            raise ValueError("[DeviceBulkState] tile_size must be >= 4")
        self.runtime = WarpRuntime(device)
        self.authority = BulkStateAuthority.DEVICE
        self._timestamp_device_s = 0.0
        self._timestamp_host_view_s = -1.0
        self._reset_generation = 0
        self._dirty_tile_ids: set[int] = set()
        self._full_field_h2d_count = 0
        self._full_field_d2h_count = 0
        self._h2d_blocking_ms = 0.0
        self._d2h_blocking_ms = 0.0
        initial = np.asarray(grid.validate_heightmap(initial_resting_m), dtype=np.float64)
        wp = self.runtime.wp
        weights = self._vertex_weights(grid)
        for name, value in (
            ("z_base", np.zeros(self.shape, dtype=np.float64)),
            ("initial_b_eff", initial),
            ("b_eff", initial),
            ("mobile", np.zeros(self.shape, dtype=np.float64)),
            ("momentum_x", np.zeros(self.shape, dtype=np.float64)),
            ("momentum_y", np.zeros(self.shape, dtype=np.float64)),
            ("weights", weights),
        ):
            self._upload_full(name, value, dtype=wp.float64, initialization=True)
        # Explicitly named compatibility aliases.  They reference the same
        # Warp allocations and therefore cannot become a second authoritative
        # geometry.  New production code must use b_eff/z_base.
        self.runtime.arrays["initial_resting"] = self.runtime.arrays["initial_b_eff"]
        self.runtime.arrays["resting"] = self.runtime.arrays["b_eff"]
        # These arrays are part of the explicit device state contract.  They
        # are persistent scratch/diagnostic buffers, not per-frame allocations.
        for name in (
            "active_mask",
            "material_mask",
            "dirty_mask",
            "frontier_reached",
            "frontier_active_tiles",
            "frontier_active_edges",
            "avalanche_unstable",
            "avalanche_parent",
            "avalanche_component_count",
            "avalanche_latch",
            "avalanche_previous_component",
            "avalanche_activation_count",
            "deposition_mask",
            "deposition_exclusion_mask",
        ):
            self.runtime.zeros(name, self.size, dtype=wp.int32)
        for name in (
            "track_rut",
            "avalanche_slope",
            "avalanche_gradient_x",
            "avalanche_gradient_y",
            "avalanche_severity",
            "avalanche_component_area",
            "avalanche_component_volume",
            "avalanche_component_excess",
            "avalanche_first_activation_time",
            "avalanche_last_activation_time",
            "avalanche_owned_surface",
            "avalanche_owned_export_baseline",
            "mobile_export_cumulative",
            "mobile_flux_export_cumulative",
            "avalanche_r2m_cumulative",
            "avalanche_m2r_cumulative",
            "deposition_work",
        ):
            self.runtime.zeros(name, self.size, dtype=wp.float64)
        self.runtime.zeros("avalanche_changed", 1, dtype=wp.int32)
        self.runtime.zeros("avalanche_diag_int", 8, dtype=wp.int32)
        self.runtime.zeros("avalanche_diag_float", 16, dtype=wp.float64)
        self.runtime.zeros("avalanche_persistence_int", 8, dtype=wp.int32)
        self.runtime.zeros("avalanche_persistence_float", 16, dtype=wp.float64)
        self.runtime.zeros(
            "avalanche_activity_tile_flags", self.tile_count, dtype=wp.int32
        )
        self.runtime.zeros("surface_before", self.size, dtype=wp.float64)
        self.runtime.zeros("dirty_tile_flags", self.tile_count, dtype=wp.int32)
        # Tiles represent ``tile_size`` terrain cells, therefore a local mesh
        # or contact chunk needs the shared final vertex row/column as a halo.
        # Reserve that maximum once; normal publication never downloads a full
        # heightmap.
        tile_sample_capacity = (self.tile_size + 1) * (self.tile_size + 1)
        self.runtime.empty("dirty_tile_indices", tile_sample_capacity, dtype=wp.int32)
        self.runtime.empty("dirty_tile_surface", tile_sample_capacity, dtype=wp.float64)
        # Airborne parcels use scalar terrain queries.  The accepted parcel
        # model caps a bucket release well below this initial batch capacity;
        # a larger rare batch grows the reusable buffers once, not per parcel.
        self._terrain_query_capacity = 128
        self.runtime.empty("terrain_query_rows", self._terrain_query_capacity, dtype=wp.float64)
        self.runtime.empty("terrain_query_cols", self._terrain_query_capacity, dtype=wp.float64)
        self.runtime.empty("terrain_query_output", self._terrain_query_capacity, dtype=wp.float64)
        self._initialization_transfers = self._raw_transfer_snapshot()
        self.begin_physics_step()

    @staticmethod
    def _vertex_weights(grid: TerrainGrid) -> np.ndarray:
        # Exact Triangle-A-C vertex weights used by TerrainVolumeIntegrator
        # and DynamicMeshAdapter.  A trapezoidal approximation here would make
        # device-side reservoir reductions disagree with the accepted ledger.
        weights = np.zeros(grid.shape, dtype=np.float64)
        factor = grid.dx * grid.dy / 6.0
        weights[:-1, :-1] += 2.0 * factor
        weights[:-1, 1:] += factor
        weights[1:, 1:] += 2.0 * factor
        weights[1:, :-1] += factor
        return weights

    @property
    def timestamp_device_s(self) -> float:
        return self._timestamp_device_s

    @property
    def reset_generation(self) -> int:
        return self._reset_generation

    @property
    def tile_shape(self) -> tuple[int, int]:
        return (
            (self.shape[0] + self.tile_size - 1) // self.tile_size,
            (self.shape[1] + self.tile_size - 1) // self.tile_size,
        )

    @property
    def tile_count(self) -> int:
        tiles = self.tile_shape
        return int(tiles[0] * tiles[1])

    def _raw_transfer_snapshot(self) -> DeviceTransferSnapshot:
        telemetry = self.runtime.telemetry
        return DeviceTransferSnapshot(
            h2d_bytes=int(telemetry.h2d_bytes),
            d2h_bytes=int(telemetry.d2h_bytes),
            h2d_transfer_count=int(telemetry.h2d_transfer_count),
            d2h_transfer_count=int(telemetry.d2h_transfer_count),
            synchronization_count=int(telemetry.synchronization_count),
            kernel_launch_count=int(telemetry.kernel_launch_count),
            h2d_blocking_ms=self._h2d_blocking_ms,
            d2h_blocking_ms=self._d2h_blocking_ms,
            full_field_h2d_count=self._full_field_h2d_count,
            full_field_d2h_count=self._full_field_d2h_count,
        )

    def begin_physics_step(self) -> None:
        """Reset per-step counters; initialization/reset are excluded."""

        self.runtime.telemetry.reset()
        self._full_field_h2d_count = 0
        self._full_field_d2h_count = 0
        self._h2d_blocking_ms = 0.0
        self._d2h_blocking_ms = 0.0

    def transfer_snapshot(self) -> DeviceTransferSnapshot:
        return self._raw_transfer_snapshot()

    def assert_normal_step_transfer_budget(self) -> None:
        snapshot = self.transfer_snapshot()
        if snapshot.full_field_h2d_count or snapshot.full_field_d2h_count:
            raise BulkStateAuthorityError(
                "[DeviceBulkState] normal GPU physics performed a full-field transfer: "
                f"H2D={snapshot.full_field_h2d_count}, D2H={snapshot.full_field_d2h_count}"
            )

    def _upload_full(
        self,
        name: str,
        value: np.ndarray,
        *,
        dtype: Any,
        initialization: bool = False,
    ) -> Any:
        start = perf_counter()
        result = self.runtime.upload(name, np.asarray(value).ravel(), dtype=dtype)
        self._h2d_blocking_ms += (perf_counter() - start) * 1_000.0
        if not initialization and name in self._FULL_FIELD_NAMES:
            self._full_field_h2d_count += 1
        return result

    def advance_time(self, dt_s: float) -> None:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("[DeviceBulkState] dt_s must be finite/non-negative")
        self._timestamp_device_s += dt

    def mark_dirty_tiles(self, tile_ids: Iterable[int]) -> None:
        for tile_id in np.asarray(tuple(tile_ids), dtype=np.int64).ravel():
            if tile_id < 0 or tile_id >= self.tile_count:
                raise ValueError("[DeviceBulkState] dirty tile ID outside grid")
            self._dirty_tile_ids.add(int(tile_id))

    def mark_dirty_bbox(self, bbox_yx: tuple[int, int, int, int]) -> None:
        row0, row1, col0, col1 = (int(value) for value in bbox_yx)
        if row1 <= row0 or col1 <= col0:
            return
        tiles_y, tiles_x = self.tile_shape
        first_y = max(0, row0 // self.tile_size)
        last_y = min(tiles_y - 1, (row1 - 1) // self.tile_size)
        first_x = max(0, col0 // self.tile_size)
        last_x = min(tiles_x - 1, (col1 - 1) // self.tile_size)
        self.mark_dirty_tiles(
            row * tiles_x + col
            for row in range(first_y, last_y + 1)
            for col in range(first_x, last_x + 1)
        )

    def capture_surface_for_dirty_tracking(self) -> None:
        """Snapshot the resident surface before a mutating device operator."""

        capture, _ = _dirty_kernels(self.runtime.wp)
        self.runtime.launch(
            capture,
            dim=self.size,
            inputs=[
                self.runtime.arrays["resting"],
                self.runtime.arrays["mobile"],
                self.runtime.arrays["surface_before"],
            ],
        )

    def collect_surface_dirty_tiles(
        self, *, tolerance_m: float = 1.0e-10
    ) -> np.ndarray:
        """Collect only compact tile flags after a resident terrain mutation."""

        tolerance = float(tolerance_m)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("[DeviceBulkState] dirty tolerance must be finite/non-negative")
        wp = self.runtime.wp
        self.runtime.arrays["dirty_tile_flags"].zero_()
        _, flag = _dirty_kernels(wp)
        self.runtime.launch(
            flag,
            dim=self.size,
            inputs=[
                self.runtime.arrays["resting"],
                self.runtime.arrays["mobile"],
                self.runtime.arrays["surface_before"],
                self.runtime.arrays["dirty_mask"],
                self.runtime.arrays["dirty_tile_flags"],
                self.shape[1],
                self.tile_size,
                self.tile_shape[1],
                tolerance,
            ],
        )
        self.runtime.synchronize()
        flags = np.asarray(
            self.runtime.arrays["dirty_tile_flags"].numpy(), dtype=np.int32
        )
        self.runtime.telemetry.record_d2h(flags)
        tile_ids = np.flatnonzero(flags).astype(np.int32)
        self.mark_dirty_tiles(tile_ids)
        self.runtime.arrays["dirty_mask"].zero_()
        return tile_ids

    def consume_dirty_tile_ids(self) -> np.ndarray:
        result = np.asarray(sorted(self._dirty_tile_ids), dtype=np.int32)
        self._dirty_tile_ids.clear()
        return result

    def _download(self, name: str, *, full_field: bool) -> np.ndarray:
        start = perf_counter()
        result = self.runtime.download(name)
        self._d2h_blocking_ms += (perf_counter() - start) * 1_000.0
        if full_field:
            self._full_field_d2h_count += 1
        return result

    def _upload_compact(self, name: str, value: np.ndarray, *, dtype: Any) -> Any:
        """Upload a compact command/patch buffer, never a terrain field."""

        start = perf_counter()
        result = self.runtime.upload(name, np.ascontiguousarray(value), dtype=dtype)
        self._h2d_blocking_ms += (perf_counter() - start) * 1_000.0
        return result

    def apply_host_patch(
        self,
        field: str,
        bbox_yx: tuple[int, int, int, int],
        values: np.ndarray,
        *,
        reason: str,
    ) -> None:
        """Commit a compact host-generated patch into device authority.

        This is the bridge for the initially host-side CAD failure geometry,
        intake and airborne parcel logic.  It transfers only the patch values
        and indices; the host never receives a complete current terrain field.
        """

        if field == "resting":
            field = "b_eff"
        if field not in {"b_eff", "mobile", "momentum_x", "momentum_y"}:
            raise ValueError("[DeviceBulkState] unsupported mutable field")
        if not reason:
            raise ValueError("[DeviceBulkState] patch reason must be non-empty")
        row0, row1, col0, col1 = (int(value) for value in bbox_yx)
        if not (0 <= row0 <= row1 <= self.shape[0] and 0 <= col0 <= col1 <= self.shape[1]):
            raise ValueError("[DeviceBulkState] patch bbox outside state")
        patch = np.asarray(values, dtype=np.float64)
        expected = (row1 - row0, col1 - col0)
        if patch.shape != expected or not np.all(np.isfinite(patch)):
            raise ValueError("[DeviceBulkState] patch values shape/finite check failed")
        if patch.size == 0:
            return
        rows, cols = np.indices(expected, dtype=np.int32)
        indices = ((rows + row0) * self.shape[1] + (cols + col0)).ravel()
        wp = self.runtime.wp
        device_indices = self._upload_compact("patch_indices", indices, dtype=wp.int32)
        device_values = self._upload_compact("patch_values", patch.ravel(), dtype=wp.float64)
        scatter, _, _, _ = _state_kernels(wp)
        self.runtime.launch(
            scatter,
            dim=int(patch.size),
            inputs=[self.runtime.arrays[field], device_indices, device_values],
        )
        self.mark_dirty_bbox(bbox_yx)

    def apply_host_indices(
        self,
        field: str,
        flat_indices: np.ndarray,
        values: np.ndarray | int,
        *,
        reason: str,
    ) -> None:
        """Scatter compact activation/mask data into the authoritative state."""

        if field not in {
            "material_mask",
            "active_mask",
            "deposition_mask",
            "frontier_reached",
        }:
            raise ValueError("[DeviceBulkState] unsupported compact integer field")
        if not reason:
            raise ValueError("[DeviceBulkState] compact scatter reason must be non-empty")
        indices = np.asarray(flat_indices, dtype=np.int32).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("[DeviceBulkState] compact scatter index outside state")
        if np.isscalar(values):
            data = np.full(indices.shape, int(values), dtype=np.int32)
        else:
            data = np.asarray(values, dtype=np.int32).reshape(-1)
            if data.shape != indices.shape:
                raise ValueError("[DeviceBulkState] compact scatter values shape mismatch")
        if not indices.size:
            return
        wp = self.runtime.wp
        device_indices = self._upload_compact("compact_indices", indices, dtype=wp.int32)
        device_values = self._upload_compact("compact_int_values", data, dtype=wp.int32)

        self.runtime.launch(
            _int_scatter_kernel(wp),
            dim=int(indices.size),
            inputs=[self.runtime.arrays[field], device_indices, device_values],
        )

    def add_host_indices(
        self,
        field: str,
        flat_indices: np.ndarray,
        values: np.ndarray,
        *,
        reason: str,
    ) -> None:
        """Conservatively add compact source terms (e.g. airborne landing)."""

        if field == "resting":
            field = "b_eff"
        if field not in {"b_eff", "mobile", "momentum_x", "momentum_y"}:
            raise ValueError("[DeviceBulkState] unsupported compact float field")
        if not reason:
            raise ValueError("[DeviceBulkState] compact add reason must be non-empty")
        indices = np.asarray(flat_indices, dtype=np.int32).reshape(-1)
        data = np.asarray(values, dtype=np.float64).reshape(-1)
        if data.shape != indices.shape or not np.all(np.isfinite(data)):
            raise ValueError("[DeviceBulkState] compact add values shape/finite check failed")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("[DeviceBulkState] compact add index outside state")
        if not indices.size:
            return
        wp = self.runtime.wp
        device_indices = self._upload_compact("compact_add_indices", indices, dtype=wp.int32)
        device_values = self._upload_compact("compact_add_values", data, dtype=wp.float64)
        self.runtime.launch(
            _float_add_kernel(wp),
            dim=int(indices.size),
            inputs=[self.runtime.arrays[field], device_indices, device_values],
        )

    def entrain_host_indices(
        self,
        flat_indices: np.ndarray,
        requested_depth_m: np.ndarray,
        *,
        reason: str,
    ) -> float:
        """Apply the one authoritative physical Resting->Mobile operation.

        The compact host arguments describe geometry only.  Donor limitation,
        state mutation and volume accounting all execute on DEVICE.  Momentum
        is intentionally unchanged (zero-momentum mass activation).
        """

        if not reason:
            raise ValueError("[DeviceBulkState] entrainment reason must be non-empty")
        indices = np.asarray(flat_indices, dtype=np.int32).reshape(-1)
        depth = np.asarray(requested_depth_m, dtype=np.float64).reshape(-1)
        if depth.shape != indices.shape or not np.all(np.isfinite(depth)):
            raise ValueError("[DeviceBulkState] invalid entrainment request")
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise ValueError("[DeviceBulkState] entrainment index outside state")
        if not indices.size:
            return 0.0
        wp = self.runtime.wp
        device_indices = self._upload_compact("exchange_indices", indices, dtype=wp.int32)
        device_depth = self._upload_compact("exchange_depth", depth, dtype=wp.float64)
        summary = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
        self.runtime.launch(
            _exchange_kernels(wp)[0], dim=int(indices.size), inputs=[
                self.runtime.arrays["z_base"], self.runtime.arrays["b_eff"],
                self.runtime.arrays["mobile"], device_indices, device_depth,
                self.runtime.arrays["weights"], summary,
            ],
        )
        self.runtime.synchronize()
        value = np.asarray(summary.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(value)
        rows, cols = np.divmod(indices, self.shape[1])
        self.mark_dirty_bbox((int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1))
        return float(value[0])

    def deposit_host_indices(
        self,
        flat_indices: np.ndarray,
        requested_depth_m: np.ndarray,
        *,
        density_kg_m3: float,
        reason: str,
    ) -> dict[str, float | np.ndarray]:
        """Apply authoritative Mobile->Resting deposition with explicit impulse."""

        if not reason:
            raise ValueError("[DeviceBulkState] deposition reason must be non-empty")
        indices = np.asarray(flat_indices, dtype=np.int32).reshape(-1)
        depth = np.asarray(requested_depth_m, dtype=np.float64).reshape(-1)
        density = float(density_kg_m3)
        if depth.shape != indices.shape or not np.all(np.isfinite(depth)):
            raise ValueError("[DeviceBulkState] invalid deposition request")
        if np.any(indices < 0) or np.any(indices >= self.size) or density <= 0.0:
            raise ValueError("[DeviceBulkState] invalid deposition indices/density")
        if not indices.size:
            return {"deposited_volume_m3": 0.0, "removed_mobile_momentum_kg_m_s": np.zeros(3)}
        wp = self.runtime.wp
        device_indices = self._upload_compact("exchange_indices", indices, dtype=wp.int32)
        device_depth = self._upload_compact("exchange_depth", depth, dtype=wp.float64)
        summary = wp.zeros(3, dtype=wp.float64, device=self.runtime.device)
        self.runtime.launch(
            _exchange_kernels(wp)[1], dim=int(indices.size), inputs=[
                self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                device_indices, device_depth, self.runtime.arrays["weights"], density, summary,
            ],
        )
        self.runtime.synchronize()
        value = np.asarray(summary.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(value)
        rows, cols = np.divmod(indices, self.shape[1])
        self.mark_dirty_bbox((int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1))
        return {
            "deposited_volume_m3": float(value[0]),
            "removed_mobile_momentum_kg_m_s": np.asarray([value[1], value[2], 0.0]),
        }

    def reservoir_reduction(self, density_kg_m3: float) -> dict[str, float | np.ndarray]:
        """Return authoritative scalar volumes/momentum without field D2H."""

        density = float(density_kg_m3)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("[DeviceBulkState] density must be finite/positive")
        wp = self.runtime.wp
        summary = wp.zeros(4, dtype=wp.float64, device=self.runtime.device)
        _, _, reduce, _ = _state_kernels(wp)
        self.runtime.launch(
            reduce,
            dim=self.size,
            inputs=[
                self.runtime.arrays["z_base"],
                self.runtime.arrays["b_eff"],
                self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"],
                self.runtime.arrays["momentum_y"],
                self.runtime.arrays["weights"],
                density,
                summary,
            ],
        )
        self.runtime.synchronize()
        values = np.asarray(summary.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(values)
        return {
            "resting_volume_m3": float(values[0]),
            "mobile_volume_m3": float(values[1]),
            "mobile_momentum_kg_m_s": np.asarray([values[2], values[3], 0.0]),
        }

    def download_patch(
        self,
        bbox_yx: tuple[int, int, int, int],
        *,
        source: str,
    ) -> HostBulkStatePatch:
        """Download a bounded patch for a host-side legacy geometry operator."""

        if not source:
            raise ValueError("[DeviceBulkState] patch source must be non-empty")
        row0, row1, col0, col1 = (int(value) for value in bbox_yx)
        if not (0 <= row0 < row1 <= self.shape[0] and 0 <= col0 < col1 <= self.shape[1]):
            raise ValueError("[DeviceBulkState] patch bbox outside state")
        count = (row1 - row0) * (col1 - col0)
        if count > self.tile_size * self.tile_size:
            raise ValueError(
                "[DeviceBulkState] patch exceeds compact bridge capacity; split into tiles"
            )
        rows, cols = np.indices((row1 - row0, col1 - col0), dtype=np.int32)
        indices = ((rows + row0) * self.shape[1] + (cols + col0)).ravel()
        wp = self.runtime.wp
        device_indices = self._upload_compact("patch_read_indices", indices, dtype=wp.int32)
        output = self.runtime.arrays["dirty_tile_surface"]
        _, _, _, gather = _state_kernels(wp)

        def read(name: str) -> np.ndarray:
            self.runtime.launch(
                gather,
                dim=count,
                inputs=[self.runtime.arrays[name], device_indices, output],
            )
            self.runtime.synchronize()
            compact = np.asarray(output.numpy(), dtype=np.float64)[:count].copy()
            self.runtime.telemetry.record_d2h(compact)
            return compact.reshape((row1 - row0, col1 - col0))

        resting = read("b_eff")
        z_base = read("z_base")
        mobile = read("mobile")
        momentum = np.stack((read("momentum_x"), read("momentum_y")), axis=-1)
        self._timestamp_host_view_s = self._timestamp_device_s
        return HostBulkStatePatch(
            (row0, row1, col0, col1),
            resting,
            mobile,
            momentum,
            self._timestamp_device_s,
            source,
            z_base,
        )

    def download_region(
        self,
        bbox_yx: tuple[int, int, int, int],
        *,
        source: str,
    ) -> HostBulkStatePatch:
        """Stitch a geometry-driven compact region from reusable tile reads.

        The caller supplies a physical bbox (sweep, mouth, or failure-wedge
        extent).  It may span several tiles, but a normal runtime request may
        never equal the complete terrain field.  This deliberately avoids a
        fixed ROI/safety crop while retaining the full-field transfer gate.
        """

        row0, row1, col0, col1 = (int(value) for value in bbox_yx)
        if not (0 <= row0 < row1 <= self.shape[0] and 0 <= col0 < col1 <= self.shape[1]):
            raise ValueError("[DeviceBulkState] region bbox outside state")
        if (row1 - row0) * (col1 - col0) >= self.size:
            raise BulkStateAuthorityError(
                "GPU_RUNTIME_FULL_FIELD_TERRAIN_REQUEST: compact bridge requested whole terrain"
            )
        shape = (row1 - row0, col1 - col0)
        resting = np.empty(shape, dtype=np.float64)
        mobile = np.empty(shape, dtype=np.float64)
        momentum = np.empty(shape + (2,), dtype=np.float64)
        z_base = np.empty(shape, dtype=np.float64)
        for start_row in range(row0, row1, self.tile_size):
            stop_row = min(start_row + self.tile_size, row1)
            for start_col in range(col0, col1, self.tile_size):
                stop_col = min(start_col + self.tile_size, col1)
                patch = self.download_patch(
                    (start_row, stop_row, start_col, stop_col), source=source
                )
                target = (
                    slice(start_row - row0, stop_row - row0),
                    slice(start_col - col0, stop_col - col0),
                )
                resting[target] = patch.H_resting_m
                mobile[target] = patch.H_mobile_m
                momentum[target] = patch.mobile_momentum_m2_s
                z_base[target] = patch.z_base_m
        return HostBulkStatePatch(
            (row0, row1, col0, col1),
            resting,
            mobile,
            momentum,
            self._timestamp_device_s,
            source,
            z_base,
        )

    def sample_surface_bilinear(self, row_col: np.ndarray, *, source: str) -> np.ndarray:
        """Batch compact D2H terrain heights for CPU-owned airborne parcels."""

        if not source:
            raise ValueError("[DeviceBulkState] terrain query source must be non-empty")
        queries = np.asarray(row_col, dtype=np.float64)
        if queries.ndim != 2 or queries.shape[1] != 2 or not np.all(np.isfinite(queries)):
            raise ValueError("[DeviceBulkState] terrain queries must be finite [N,2]")
        count = int(queries.shape[0])
        if count == 0:
            return np.empty(0, dtype=np.float64)
        wp = self.runtime.wp
        if count > self._terrain_query_capacity:
            self._terrain_query_capacity = count
            self.runtime.empty("terrain_query_rows", count, dtype=wp.float64)
            self.runtime.empty("terrain_query_cols", count, dtype=wp.float64)
            self.runtime.empty("terrain_query_output", count, dtype=wp.float64)
        rows = self.runtime.arrays["terrain_query_rows"]
        cols = self.runtime.arrays["terrain_query_cols"]
        output = self.runtime.arrays["terrain_query_output"]
        indices = self._upload_compact(
            "terrain_query_indices", np.arange(count, dtype=np.int32), dtype=wp.int32
        )
        rows_in = self._upload_compact(
            "terrain_query_rows_in", queries[:, 0], dtype=wp.float64
        )
        cols_in = self._upload_compact(
            "terrain_query_cols_in", queries[:, 1], dtype=wp.float64
        )
        scatter, _, _, _ = _state_kernels(wp)
        self.runtime.launch(scatter, dim=count, inputs=[rows, indices, rows_in])
        self.runtime.launch(scatter, dim=count, inputs=[cols, indices, cols_in])
        self.runtime.launch(
            _query_kernel(wp),
            dim=count,
            inputs=[
                self.runtime.arrays["resting"], self.runtime.arrays["mobile"],
                rows, cols, self.shape[0], self.shape[1], output,
            ],
        )
        self.runtime.synchronize()
        values = np.asarray(output.numpy(), dtype=np.float64)[:count].copy()
        self.runtime.telemetry.record_d2h(values)
        return values

    def explicit_host_view(self, *, source: str) -> HostBulkStateView:
        """Take a deliberately visible full-state snapshot.

        ``source`` must name an allowed boundary (debug, acceptance, checkpoint
        or reset).  It is intentionally unsuitable for the normal step loop.
        """

        allowed = {"debug", "acceptance", "checkpoint", "reset", "visualization"}
        if source not in allowed:
            raise BulkStateAuthorityError(
                "[DeviceBulkState] full host views are restricted to explicit "
                f"boundaries, got source={source!r}"
            )
        resting = self._download("b_eff", full_field=True).reshape(self.shape)
        z_base = self._download("z_base", full_field=True).reshape(self.shape)
        mobile = self._download("mobile", full_field=True).reshape(self.shape)
        momentum = np.stack(
            (
                self._download("momentum_x", full_field=True).reshape(self.shape),
                self._download("momentum_y", full_field=True).reshape(self.shape),
            ),
            axis=-1,
        )
        dirty = self.consume_dirty_tile_ids()
        self._timestamp_host_view_s = self._timestamp_device_s
        return HostBulkStateView(
            resting,
            mobile,
            momentum,
            self._timestamp_device_s,
            self._reset_generation,
            source,
            dirty,
            z_base,
        )

    def restore_checkpoint_fields(
        self,
        fields: dict[str, np.ndarray],
        *,
        timestamp_s: float,
        source: str,
    ) -> None:
        """Restore an explicit acceptance/checkpoint boundary onto device.

        This is intentionally not a normal-step API. It permits reproducible
        continuation of long GPU acceptance runs without constructing a host
        TerrainState shadow or disguising checkpoint traffic as normal H2D.
        """

        if source not in {"acceptance", "checkpoint"}:
            raise BulkStateAuthorityError(
                "[DeviceBulkState] checkpoint restore requires an explicit boundary"
            )
        float_fields = {
            "z_base", "b_eff", "resting", "mobile", "momentum_x", "momentum_y",
            "track_rut",
            "avalanche_first_activation_time", "avalanche_last_activation_time",
            "avalanche_owned_surface",
            "avalanche_owned_export_baseline",
            "mobile_export_cumulative",
            "mobile_flux_export_cumulative",
            "avalanche_r2m_cumulative", "avalanche_m2r_cumulative",
        }
        int_fields = {
            "avalanche_activation_count", "avalanche_latch",
            "avalanche_previous_component", "frontier_reached",
            "active_mask", "material_mask", "deposition_mask",
            "deposition_exclusion_mask",
        }
        unknown = sorted(set(fields) - float_fields - int_fields)
        if unknown:
            raise BulkStateAuthorityError(
                f"[DeviceBulkState] unsupported checkpoint fields: {unknown}"
            )
        wp = self.runtime.wp
        for name, value in fields.items():
            if name == "resting":
                name = "b_eff"
            array = np.asarray(value)
            if array.shape != self.shape or not np.all(np.isfinite(array)):
                raise BulkStateAuthorityError(
                    f"[DeviceBulkState] invalid checkpoint field {name!r}"
                )
            dtype = wp.float64 if name in float_fields else wp.int32
            self._upload_full(name, array, dtype=dtype, initialization=True)
        self.runtime.arrays["resting"] = self.runtime.arrays["b_eff"]
        self.runtime.arrays["initial_resting"] = self.runtime.arrays["initial_b_eff"]
        # Per-step tool occupancy is transient scratch, not checkpoint physics.
        # Older checkpoints may contain it; accept for compatibility but never
        # resurrect stale deposition exclusion into the next frame.
        self.runtime.arrays["deposition_exclusion_mask"].zero_()
        time_value = float(timestamp_s)
        if not np.isfinite(time_value) or time_value < 0.0:
            raise BulkStateAuthorityError(
                "[DeviceBulkState] checkpoint timestamp must be finite/non-negative"
            )
        self._timestamp_device_s = time_value
        self._timestamp_host_view_s = -1.0
        self.mark_dirty_tiles(range(self.tile_count))
        self.begin_physics_step()

    def download_dirty_tiles(
        self,
        tile_ids: Iterable[int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Download only current-surface tiles for visual/contact publication."""

        selected = (
            np.asarray(tuple(tile_ids), dtype=np.int32)
            if tile_ids is not None
            else self.consume_dirty_tile_ids()
        )
        if selected.size == 0:
            return {}
        tiles_y, tiles_x = self.tile_shape
        result: dict[int, np.ndarray] = {}
        wp = self.runtime.wp
        _, gather, _, _ = _state_kernels(wp)
        for tile in selected:
            tile_id = int(tile)
            if tile_id < 0 or tile_id >= tiles_y * tiles_x:
                raise ValueError("[DeviceBulkState] dirty tile ID outside grid")
            row, col = divmod(tile_id, tiles_x)
            r0 = row * self.tile_size
            c0 = col * self.tile_size
            r1 = min(r0 + self.tile_size + 1, self.shape[0])
            c1 = min(c0 + self.tile_size + 1, self.shape[1])
            rows, cols = np.indices((r1 - r0, c1 - c0), dtype=np.int32)
            indices = ((rows + r0) * self.shape[1] + (cols + c0)).ravel()
            count = int(indices.size)
            device_indices = self._upload_compact(
                "dirty_tile_indices", indices, dtype=wp.int32
            )
            output = self.runtime.arrays["dirty_tile_surface"]
            self.runtime.launch(
                gather,
                dim=count,
                inputs=[
                    self.runtime.arrays["resting"],
                    self.runtime.arrays["mobile"],
                    device_indices,
                    output,
                ],
            )
            self.runtime.synchronize()
            compact = np.asarray(output.numpy(), dtype=np.float64)[:count].copy()
            self.runtime.telemetry.record_d2h(compact)
            result[tile_id] = compact.reshape((r1 - r0, c1 - c0))
        self._timestamp_host_view_s = self._timestamp_device_s
        return result

    def assert_consistent(self) -> None:
        if self.authority is not BulkStateAuthority.DEVICE:
            raise BulkStateAuthorityError("[DeviceBulkState] state authority is not DEVICE")
        required = {
            "z_base", "b_eff", "resting", "mobile", "momentum_x", "momentum_y",
            "initial_b_eff", "initial_resting",
            "weights", "active_mask", "dirty_mask", "frontier_reached",
            "frontier_active_tiles", "frontier_active_edges", "avalanche_slope",
            "avalanche_unstable", "avalanche_activation_count",
            "avalanche_first_activation_time", "avalanche_last_activation_time",
            "avalanche_owned_surface",
            "avalanche_owned_export_baseline",
            "mobile_export_cumulative",
            "mobile_flux_export_cumulative",
            "avalanche_r2m_cumulative", "avalanche_m2r_cumulative",
            "avalanche_persistence_int", "avalanche_persistence_float",
            "avalanche_activity_tile_flags", "deposition_work",
            "deposition_exclusion_mask", "track_rut",
            "surface_before", "dirty_tile_flags",
        }
        missing = sorted(required - set(self.runtime.arrays))
        if missing:
            raise BulkStateAuthorityError(
                f"[DeviceBulkState] missing required resident fields: {missing}"
            )
        if self._timestamp_host_view_s > self._timestamp_device_s:
            raise BulkStateAuthorityError(
                "[DeviceBulkState] host view timestamp is newer than device authority"
            )

    def reset(self) -> None:
        """Restore all device-resident fields; reset is an explicit boundary."""

        wp = self.runtime.wp
        wp.copy(self.runtime.arrays["b_eff"], self.runtime.arrays["initial_b_eff"])
        self.runtime.arrays["z_base"].zero_()
        for name in (
            "mobile", "momentum_x", "momentum_y", "track_rut", "avalanche_slope",
            "avalanche_gradient_x", "avalanche_gradient_y",
            "avalanche_severity", "avalanche_component_area",
            "avalanche_component_volume", "avalanche_component_excess",
            "avalanche_first_activation_time", "avalanche_last_activation_time",
            "avalanche_owned_surface", "avalanche_owned_export_baseline",
            "mobile_export_cumulative",
            "mobile_flux_export_cumulative",
            "avalanche_r2m_cumulative", "avalanche_m2r_cumulative",
            "avalanche_changed", "avalanche_diag_int", "avalanche_diag_float",
            "avalanche_persistence_int", "avalanche_persistence_float",
            "deposition_work", "active_mask", "material_mask", "dirty_mask",
            "frontier_reached", "frontier_active_tiles", "frontier_active_edges",
            "avalanche_unstable", "avalanche_parent",
            "avalanche_component_count", "avalanche_latch",
            "avalanche_previous_component", "avalanche_activation_count",
            "deposition_mask", "deposition_exclusion_mask",
            "surface_before", "dirty_tile_flags",
            "avalanche_activity_tile_flags",
        ):
            self.runtime.arrays[name].zero_()
        self.runtime.synchronize()
        self._timestamp_device_s = 0.0
        self._timestamp_host_view_s = -1.0
        self._dirty_tile_ids = set(range(self.tile_count))
        self._reset_generation += 1
        self.begin_physics_step()

    def diagnostics(self) -> dict[str, object]:
        self.assert_consistent()
        return {
            "backend_identity": self.backend_identity,
            "state_authority": self.authority.value,
            "authoritative_geometry": ["z_base", "b_eff", "mobile"],
            "legacy_H_resting": "COMPATIBILITY_ALIAS_OF_B_EFF_NOT_AUTHORITATIVE",
            "shape_yx": list(self.shape),
            "tile_size": self.tile_size,
            "timestamp_device_s": self._timestamp_device_s,
            "timestamp_host_view_s": self._timestamp_host_view_s,
            "reset_generation": self._reset_generation,
            "pending_dirty_tile_count": len(self._dirty_tile_ids),
            "transfer": self.transfer_snapshot().to_dict(),
            "resident_array_names": sorted(self.runtime.arrays),
        }
