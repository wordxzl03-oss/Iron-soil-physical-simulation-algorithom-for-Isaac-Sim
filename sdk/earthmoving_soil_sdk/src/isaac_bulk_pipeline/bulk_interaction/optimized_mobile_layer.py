"""Sparse active-window backend for the reference mobile-layer equations."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..performance import ActiveDomainManager, ActiveDomainSnapshot, ActiveReason, GridWindow
from ..terrain.terrain_grid import TerrainGrid
from .mobile_layer import MobileLayerConfig, MobileLayerResult, MobileLayerSolver


class OptimizedMobileLayerSolver:
    """Execute the unchanged reference finite-volume equations on a local crop.

    The authoritative state remains a full 0.05 m array. Only a dynamic window
    containing non-dry mobile material, current tool forcing, the reference
    active buffer and a gradient/flux halo is passed to ``MobileLayerSolver``.
    No update-frequency reduction or equation change is used.

    Open-boundary cases currently fall back to the reference backend because a
    rectangular crop cannot label only the true global edges as open. This is
    explicit and prevents artificial outflow at an internal crop boundary.
    """

    backend_name = "OPTIMIZED_ACTIVE_CPU"

    def __init__(
        self,
        config: MobileLayerConfig | None = None,
        *,
        tile_size: int = 32,
        stencil_halo_cells: int = 2,
    ) -> None:
        self.config = config or MobileLayerConfig()
        if not isinstance(stencil_halo_cells, int) or stencil_halo_cells < 2:
            raise ValueError("[OptimizedMobileLayer] stencil halo must be >= 2")
        self.tile_size = int(tile_size)
        self.stencil_halo_cells = stencil_halo_cells
        self._reference = MobileLayerSolver(self.config)
        self._domain: ActiveDomainManager | None = None
        self._shape: tuple[int, int] | None = None
        self.last_active_snapshot: ActiveDomainSnapshot | None = None
        self.last_kernel_time_ms = 0.0
        self.last_total_time_ms = 0.0
        self.last_reference_fallback_reason: str | None = None

    def reset(self) -> None:
        if self._domain is not None:
            self._domain.clear()
        self.last_active_snapshot = None
        self.last_kernel_time_ms = 0.0
        self.last_total_time_ms = 0.0
        self.last_reference_fallback_reason = None

    def step(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        tool_forcing_mask: np.ndarray | None = None,
        tool_velocity_xy_m_s: np.ndarray | None = None,
    ) -> MobileLayerResult:
        total_start = perf_counter()
        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        mobile = np.asarray(mobile_height_m, dtype=np.float64)
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64)
        if mobile.shape != grid.shape or momentum.shape != grid.shape + (2,):
            raise ValueError("[OptimizedMobileLayer] state/grid shape mismatch")
        forcing = (
            np.zeros(grid.shape, dtype=bool)
            if tool_forcing_mask is None
            else np.asarray(tool_forcing_mask, dtype=bool)
        )
        if forcing.shape != grid.shape:
            raise ValueError("[OptimizedMobileLayer] forcing mask shape mismatch")
        self._ensure_domain(grid.shape)
        assert self._domain is not None
        self._domain.clear(ActiveReason.MOBILE | ActiveReason.BUCKET)
        raw_active = (mobile > self.config.dry_tolerance_m) | forcing
        raw_window = self._domain.mark_mask(raw_active, ActiveReason.MOBILE)
        if np.any(forcing):
            self._domain.mark_mask(forcing, ActiveReason.BUCKET)
        halo = self.config.active_buffer_cells + self.stencil_halo_cells
        window = self._domain.combined_window(
            ActiveReason.MOBILE | ActiveReason.BUCKET, halo_cells=halo
        )
        if window.empty:
            result = self._reference.step(
                resting,
                mobile,
                momentum,
                material,
                grid,
                integrator,
                dt_s,
                tool_forcing_mask=forcing,
                tool_velocity_xy_m_s=tool_velocity_xy_m_s,
            )
            self.last_active_snapshot = self._domain.snapshot()
            self.last_kernel_time_ms = 0.0
            self.last_total_time_ms = (perf_counter() - total_start) * 1_000.0
            return result

        if self.config.boundary_condition == "open" and window != GridWindow(
            0, 0, grid.ny, grid.nx
        ):
            self.last_reference_fallback_reason = "OPEN_BOUNDARY_REQUIRES_GLOBAL_EDGE_LABELS"
            result = self._reference.step(
                resting,
                mobile,
                momentum,
                material,
                grid,
                integrator,
                dt_s,
                tool_forcing_mask=forcing,
                tool_velocity_xy_m_s=tool_velocity_xy_m_s,
            )
            self.last_active_snapshot = self._domain.snapshot()
            self.last_kernel_time_ms = self.last_total_time_ms = (
                perf_counter() - total_start
            ) * 1_000.0
            return result

        window = self._ensure_minimum_shape(window, grid.shape)
        slices = window.slices
        local_grid = TerrainGrid(
            nx=window.shape[1],
            ny=window.shape[0],
            dx=grid.dx,
            dy=grid.dy,
            origin_x=grid.origin_x + window.col_start * grid.dx,
            origin_y=grid.origin_y + window.row_start * grid.dy,
            terrain_prim_path=grid.terrain_prim_path,
            terrain_to_world_matrix=grid.terrain_to_world_matrix,
            valid_mask=(None if grid.valid_mask is None else grid.valid_mask[slices]),
        )
        local_integrator = TerrainVolumeIntegrator.from_grid(local_grid)
        kernel_start = perf_counter()
        local_result = self._reference.step(
            resting[slices],
            mobile[slices],
            momentum[slices],
            material,
            local_grid,
            local_integrator,
            dt_s,
            tool_forcing_mask=forcing[slices],
            tool_velocity_xy_m_s=tool_velocity_xy_m_s,
        )
        self.last_kernel_time_ms = (perf_counter() - kernel_start) * 1_000.0

        full_height = np.array(mobile, dtype=np.float64, copy=True, order="C")
        full_momentum = np.array(momentum, dtype=np.float64, copy=True, order="C")
        full_height[slices] = local_result.mobile_height_m
        full_momentum[slices] = local_result.mobile_momentum_m2_s
        initial_volume = integrator.integrate(mobile)
        final_volume = integrator.integrate(full_height)
        balance = initial_volume - final_volume - local_result.outflow_volume_m3
        tolerance = max(1.0e-11, 1.0e-10 * max(initial_volume, 1.0))
        if abs(balance) > tolerance:
            raise RuntimeError(
                "[OptimizedMobileLayer] embedded global balance failed: "
                f"before={initial_volume}, after={final_volume}, "
                f"outflow={local_result.outflow_volume_m3}, error={balance}"
            )
        density = float(material.assumed_bulk_density_kg_m3)
        weights = integrator.vertex_weights_m2
        momentum_before = MobileLayerSolver._integrated_momentum(momentum, weights, density)
        momentum_after = MobileLayerSolver._integrated_momentum(full_momentum, weights, density)
        numerical = (
            momentum_after
            - momentum_before
            - local_result.gravity_pressure_impulse_terrain_ns
            - local_result.basal_friction_impulse_terrain_ns
            - local_result.tool_impulse_on_mobile_terrain_ns
        )
        self._domain.mark_window(window, ActiveReason.MOBILE)
        self.last_active_snapshot = self._domain.snapshot(
            ActiveReason.MOBILE | ActiveReason.BUCKET
        )
        self.last_total_time_ms = (perf_counter() - total_start) * 1_000.0
        return MobileLayerResult(
            mobile_height_m=full_height,
            mobile_momentum_m2_s=full_momentum,
            outflow_volume_m3=local_result.outflow_volume_m3,
            substeps=local_result.substeps,
            cfl_limited=local_result.cfl_limited,
            active_bbox_grid=(
                window.row_start,
                window.col_start,
                window.row_stop,
                window.col_stop,
            ),
            volume_before_m3=initial_volume,
            volume_after_m3=final_volume,
            maximum_speed_m_s=local_result.maximum_speed_m_s,
            minimum_height_m=float(np.min(full_height, initial=0.0)),
            momentum_before_terrain_kg_m_s=momentum_before,
            momentum_after_terrain_kg_m_s=momentum_after,
            gravity_pressure_impulse_terrain_ns=local_result.gravity_pressure_impulse_terrain_ns,
            basal_friction_impulse_terrain_ns=local_result.basal_friction_impulse_terrain_ns,
            tool_impulse_on_mobile_terrain_ns=local_result.tool_impulse_on_mobile_terrain_ns,
            numerical_dissipative_impulse_terrain_ns=numerical,
            yielded_area_m2=local_result.yielded_area_m2,
            moving_mobile_volume_m3=local_result.moving_mobile_volume_m3,
            mobile_velocity_p95_m_s=local_result.mobile_velocity_p95_m_s,
            yield_model_classification=local_result.yield_model_classification,
            _trusted_solver_output=True,
        )

    def _ensure_domain(self, shape: tuple[int, int]) -> None:
        if self._domain is None or self._shape != shape:
            self._domain = ActiveDomainManager(shape, self.tile_size)
            self._shape = shape

    @staticmethod
    def _ensure_minimum_shape(
        window: GridWindow, shape: tuple[int, int]
    ) -> GridWindow:
        result = window
        while min(result.shape) < 2:
            expanded = result.expanded(1, shape)
            if expanded == result:
                break
            result = expanded
        if min(result.shape) < 2:
            return GridWindow(0, 0, shape[0], shape[1])
        return result
