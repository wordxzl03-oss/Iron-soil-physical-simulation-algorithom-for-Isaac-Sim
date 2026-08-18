"""Device-resident physical LargeAvalanche diagnosis and mobilization.

This preserves the accepted reduced-order transition equations while replacing
the CPU full-field connected-component dependency.  Only fixed-size scalar
diagnostics and compact active tile flags cross to host.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bulk_interaction.large_avalanche import LargeAvalancheTransitionConfig
from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain import TerrainGrid
from .bulk_state_authority import DeviceBulkState


_KERNELS: dict[int, tuple[Any, ...]] = {}


def _kernels(wp: Any) -> tuple[Any, ...]:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.func
    def root_of(parent: wp.array(dtype=wp.int32), value: int):
        root = int(value)
        step = int(0)
        while step < 64:
            next_root = parent[root]
            if next_root < 0 or next_root == root:
                return root
            root = next_root
            step = step + 1
        return root

    @wp.func
    def join(
        parent: wp.array(dtype=wp.int32),
        a: int,
        b: int,
        changed: wp.array(dtype=wp.int32),
    ):
        if parent[a] < 0 or parent[b] < 0:
            return
        ra = root_of(parent, a)
        rb = root_of(parent, b)
        if ra == rb:
            return
        low = wp.min(ra, rb)
        high = wp.max(ra, rb)
        previous = wp.atomic_min(parent, high, low)
        if previous > low:
            wp.atomic_max(changed, 0, 1)

    @wp.kernel
    def initialize(
        z_base: wp.array(dtype=wp.float64),
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        unstable: wp.array(dtype=wp.int32),
        parent: wp.array(dtype=wp.int32),
        latch: wp.array(dtype=wp.int32),
        owned_surface: wp.array(dtype=wp.float64),
        owned_export_baseline: wp.array(dtype=wp.float64),
        export_cumulative: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        slope_angle: wp.array(dtype=wp.float64),
        gradient_x: wp.array(dtype=wp.float64),
        gradient_y: wp.array(dtype=wp.float64),
        severity: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        tan_start: wp.float64,
        tan_stop: wp.float64,
        density: wp.float64,
        gravity: wp.float64,
        cohesion_pa: wp.float64,
        mobilization_depth: wp.float64,
        dry_tolerance: wp.float64,
        activity_speed: wp.float64,
        release_settled_latches: int,
    ):
        index = wp.tid()
        row = index // cols
        col = index - row * cols
        left = index
        right = index
        down = index
        up = index
        denom_x = dx
        denom_y = dy
        if col > 0:
            left = index - 1
        if col + 1 < cols:
            right = index + 1
        if col > 0 and col + 1 < cols:
            denom_x = wp.float64(2.0) * dx
        if row > 0:
            down = index - cols
        if row + 1 < rows:
            up = index + cols
        if row > 0 and row + 1 < rows:
            denom_y = wp.float64(2.0) * dy
        gx = (
            resting[right] + mobile[right] - resting[left] - mobile[left]
        ) / denom_x
        gy = (
            resting[up] + mobile[up] - resting[down] - mobile[down]
        ) / denom_y
        magnitude = wp.sqrt(gx * gx + gy * gy)
        gradient_x[index] = gx
        gradient_y[index] = gy
        slope_angle[index] = wp.atan(magnitude)
        layer_depth = wp.min(wp.max(resting[index] - z_base[index], wp.float64(0.0)), mobilization_depth)
        normalization = wp.sqrt(wp.float64(1.0) + magnitude * magnitude)
        drive = density * gravity * layer_depth * magnitude / normalization
        normal = density * gravity * layer_depth / normalization
        start_margin = drive - (cohesion_pa + normal * tan_start)
        stop_margin = drive - (cohesion_pa + normal * tan_stop)
        severity[index] = wp.clamp(
            stop_margin / wp.max(normal * (tan_start - tan_stop), wp.float64(1.0e-12)),
            wp.float64(0.0),
            wp.float64(1.0),
        )
        active = layer_depth > dry_tolerance and start_margin > wp.float64(0.0)
        unstable[index] = 0
        parent[index] = -1
        if active:
            unstable[index] = 1
            parent[index] = index
        speed = wp.float64(0.0)
        if mobile[index] > dry_tolerance:
            vx = momentum_x[index] / mobile[index]
            vy = momentum_y[index] / mobile[index]
            speed = wp.sqrt(vx * vx + vy * vy)
        # A latch owns one mobilized tranche.  An empty local Mobile cell is
        # not, by itself, evidence that the tranche left: deposition can put
        # the same material straight back into Resting in the same cell.  The
        # next tranche becomes eligible only after BOTH (a) conservative
        # shared-face transport has exported material from the owned donor cell
        # and (b) authoritative H_free has actually fallen
        # below the surface captured at activation.  The second condition
        # rejects reversible/numerical face traffic which does not represent
        # net tranche departure.  Physical stabilization under Y_stop remains
        # an independent release path.
        free_surface = resting[index] + mobile[index]
        exported_since_activation = (
            export_cumulative[index] - owned_export_baseline[index]
        )
        resolved_export = (
            exported_since_activation > dry_tolerance * weights[index]
            and free_surface < owned_surface[index] - dry_tolerance
        )
        if release_settled_latches != 0:
            # Conservative tranche departure is sufficient even while the
            # neighbouring Mobile field is still moving.  The alternative
            # stabilization release remains a quiet/Y_stop condition.
            if resolved_export or (
                speed <= activity_speed and stop_margin <= wp.float64(0.0)
            ):
                latch[index] = 0

    @wp.kernel
    def union_pass(
        parent: wp.array(dtype=wp.int32),
        changed: wp.array(dtype=wp.int32),
        rows: int,
        cols: int,
        connectivity: int,
    ):
        index = wp.tid()
        if parent[index] < 0:
            return
        row = index // cols
        col = index - row * cols
        if col + 1 < cols:
            join(parent, index, index + 1, changed)
        if row + 1 < rows:
            join(parent, index, index + cols, changed)
        if connectivity == 8 and row + 1 < rows:
            if col + 1 < cols:
                join(parent, index, index + cols + 1, changed)
            if col > 0:
                join(parent, index, index + cols - 1, changed)

    @wp.kernel
    def compress(
        parent: wp.array(dtype=wp.int32),
        changed: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        if parent[index] < 0:
            return
        root = root_of(parent, index)
        if parent[index] != root:
            parent[index] = root
            wp.atomic_max(changed, 0, 1)

    @wp.kernel
    def aggregate_components(
        parent: wp.array(dtype=wp.int32),
        z_base: wp.array(dtype=wp.float64),
        resting: wp.array(dtype=wp.float64),
        latch: wp.array(dtype=wp.int32),
        weights: wp.array(dtype=wp.float64),
        slope_angle: wp.array(dtype=wp.float64),
        severity: wp.array(dtype=wp.float64),
        counts: wp.array(dtype=wp.int32),
        areas: wp.array(dtype=wp.float64),
        volumes: wp.array(dtype=wp.float64),
        excesses: wp.array(dtype=wp.float64),
        mobilization_depth: wp.float64,
        start_angle_rad: wp.float64,
    ):
        index = wp.tid()
        if parent[index] < 0:
            return
        root = root_of(parent, index)
        weight = weights[index]
        # Component volume is a physical event-scale diagnostic and therefore
        # does not disappear while the current tranche is latched.  Ownership
        # only controls whether this step can transfer a new tranche.
        transfer = wp.min(
            wp.max(resting[index] - z_base[index], wp.float64(0.0)),
            mobilization_depth * severity[index],
        )
        excess = wp.max(slope_angle[index] - start_angle_rad, wp.float64(0.0))
        wp.atomic_add(counts, root, 1)
        wp.atomic_add(areas, root, weight)
        wp.atomic_add(volumes, root, transfer * weight)
        wp.atomic_add(excesses, root, excess)

    @wp.kernel
    def reduce_components(
        counts: wp.array(dtype=wp.int32),
        diagnostics: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        count = counts[index]
        if count > 0:
            wp.atomic_add(diagnostics, 0, 1)
            wp.atomic_add(diagnostics, 3, count)
            wp.atomic_max(diagnostics, 1, count)

    @wp.kernel
    def select_winner(
        counts: wp.array(dtype=wp.int32),
        diagnostics: wp.array(dtype=wp.int32),
    ):
        index = wp.tid()
        if counts[index] == diagnostics[1] and diagnostics[1] > 0:
            wp.atomic_min(diagnostics, 2, index)

    @wp.kernel
    def winner_diagnostics(
        parent: wp.array(dtype=wp.int32),
        previous: wp.array(dtype=wp.int32),
        slope_angle: wp.array(dtype=wp.float64),
        areas: wp.array(dtype=wp.float64),
        volumes: wp.array(dtype=wp.float64),
        excesses: wp.array(dtype=wp.float64),
        diag_i: wp.array(dtype=wp.int32),
        diag_f: wp.array(dtype=wp.float64),
        start_angle_rad: wp.float64,
        stop_angle_rad: wp.float64,
    ):
        index = wp.tid()
        winner = diag_i[2]
        if parent[index] < 0 or root_of(parent, index) != winner:
            return
        wp.atomic_add(diag_i, 4, previous[index])
        excess_start = wp.max(slope_angle[index] - start_angle_rad, wp.float64(0.0))
        excess_stop = wp.max(slope_angle[index] - stop_angle_rad, wp.float64(0.0))
        wp.atomic_add(diag_f, 0, excess_start)
        wp.atomic_max(diag_f, 1, excess_start)
        wp.atomic_add(diag_f, 2, excess_stop)
        wp.atomic_max(diag_f, 3, excess_stop)
        if index == winner:
            diag_f[4] = areas[winner]
            diag_f[5] = volumes[winner]
            diag_f[6] = excesses[winner]

    @wp.kernel
    def update_previous_and_tiles(
        parent: wp.array(dtype=wp.int32),
        previous: wp.array(dtype=wp.int32),
        reached: wp.array(dtype=wp.int32),
        tile_flags: wp.array(dtype=wp.int32),
        diag_i: wp.array(dtype=wp.int32),
        cols: int,
        tile_size: int,
        tiles_x: int,
        keep_candidate: int,
        seed_frontier: int,
    ):
        index = wp.tid()
        unstable_cell = parent[index] >= 0
        belongs = unstable_cell and root_of(parent, index) == diag_i[2]
        previous[index] = 0
        if belongs and keep_candidate != 0:
            previous[index] = 1
        if unstable_cell and seed_frontier != 0:
            reached[index] = 1
            row = index // cols
            col = index - row * cols
            tile = (row // tile_size) * tiles_x + col // tile_size
            wp.atomic_max(tile_flags, tile, 1)

    @wp.kernel
    def mobilize(
        z_base: wp.array(dtype=wp.float64),
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        parent: wp.array(dtype=wp.int32),
        latch: wp.array(dtype=wp.int32),
        owned_surface: wp.array(dtype=wp.float64),
        owned_export_baseline: wp.array(dtype=wp.float64),
        export_cumulative: wp.array(dtype=wp.float64),
        gradient_x: wp.array(dtype=wp.float64),
        gradient_y: wp.array(dtype=wp.float64),
        severity: wp.array(dtype=wp.float64),
        diag_i: wp.array(dtype=wp.int32),
        diag_f: wp.array(dtype=wp.float64),
        activation_count: wp.array(dtype=wp.int32),
        first_activation_time: wp.array(dtype=wp.float64),
        last_activation_time: wp.array(dtype=wp.float64),
        resting_to_mobile_cumulative: wp.array(dtype=wp.float64),
        activation_time_s: wp.float64,
        density: wp.float64,
        mobilization_depth: wp.float64,
        dry_tolerance: wp.float64,
    ):
        index = wp.tid()
        # Y_start is the constitutive Resting->Mobile authorization.  Connected
        # size/persistence classify a large event but never veto local yield.
        if parent[index] < 0 or latch[index] != 0:
            return
        transfer = wp.min(
            wp.max(resting[index] - z_base[index], wp.float64(0.0)),
            mobilization_depth * severity[index],
        )
        if transfer <= dry_tolerance:
            return
        # State activation itself is a zero-momentum exchange.  Gravity and
        # pressure accelerate the new Mobile mass through the V2 source/flux
        # update over finite physical time; no arbitrary launch velocity is
        # assigned here.
        vx = wp.float64(0.0)
        vy = wp.float64(0.0)
        owned_surface[index] = resting[index] + mobile[index]
        owned_export_baseline[index] = export_cumulative[index]
        resting[index] = resting[index] - transfer
        mobile[index] = mobile[index] + transfer
        momentum_x[index] = momentum_x[index] + transfer * vx
        momentum_y[index] = momentum_y[index] + transfer * vy
        latch[index] = 1
        weighted = transfer * weights[index]
        previous_count = activation_count[index]
        activation_count[index] = previous_count + 1
        if previous_count == 0:
            first_activation_time[index] = activation_time_s
            wp.atomic_add(diag_f, 10, weighted)
            wp.atomic_add(diag_f, 12, weights[index])
        else:
            wp.atomic_add(diag_f, 11, weighted)
            wp.atomic_add(diag_f, 13, weights[index])
        last_activation_time[index] = activation_time_s
        resting_to_mobile_cumulative[index] = (
            resting_to_mobile_cumulative[index] + weighted
        )
        wp.atomic_add(diag_f, 7, weighted)
        wp.atomic_add(diag_f, 8, density * weighted * vx)
        wp.atomic_add(diag_f, 9, density * weighted * vy)

    @wp.kernel
    def reduce_persistence(
        initial_resting: wp.array(dtype=wp.float64),
        resting: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        slope_angle: wp.array(dtype=wp.float64),
        unstable: wp.array(dtype=wp.int32),
        latch: wp.array(dtype=wp.int32),
        activation_count: wp.array(dtype=wp.int32),
        resting_to_mobile_cumulative: wp.array(dtype=wp.float64),
        mobile_to_resting_cumulative: wp.array(dtype=wp.float64),
        diagnostics_i: wp.array(dtype=wp.int32),
        diagnostics_f: wp.array(dtype=wp.float64),
        activity_tile_flags: wp.array(dtype=wp.int32),
        cols: int,
        tile_size: int,
        tiles_x: int,
        start_angle_rad: wp.float64,
        stop_angle_rad: wp.float64,
        dry_tolerance: wp.float64,
        activity_speed: wp.float64,
    ):
        index = wp.tid()
        count = activation_count[index]
        h = mobile[index]
        speed = wp.float64(0.0)
        if h > dry_tolerance:
            vx = momentum_x[index] / h
            vy = momentum_y[index] / h
            speed = wp.sqrt(vx * vx + vy * vy)
            wp.atomic_add(diagnostics_i, 2, 1)
            wp.atomic_add(diagnostics_f, 4, h * weights[index])
            wp.atomic_max(diagnostics_f, 5, speed)
            if speed > activity_speed:
                wp.atomic_add(diagnostics_f, 13, h * weights[index])
        if unstable[index] != 0 or (h > dry_tolerance and speed > activity_speed):
            row = index // cols
            col = index - row * cols
            tile = (row // tile_size) * tiles_x + col // tile_size
            wp.atomic_max(activity_tile_flags, tile, 1)
        delta_surface = resting[index] + h - initial_resting[index]
        wp.atomic_add(diagnostics_f, 6, delta_surface * weights[index])
        wp.atomic_add(
            diagnostics_f, 7, wp.abs(delta_surface) * weights[index] * wp.float64(0.5)
        )
        # Throughput is a reservoir-wide quantity. Mobile may travel away
        # from its activation cell before deposition, so M2R must not be
        # restricted to cells whose local activation_count is nonzero.
        wp.atomic_add(diagnostics_f, 0, resting_to_mobile_cumulative[index])
        wp.atomic_add(diagnostics_f, 1, mobile_to_resting_cumulative[index])
        if count <= 0:
            return
        weight = weights[index]
        slope = slope_angle[index]
        wp.atomic_add(diagnostics_i, 0, 1)
        wp.atomic_add(diagnostics_f, 2, weight)
        wp.atomic_add(diagnostics_f, 8, slope)
        wp.atomic_max(diagnostics_f, 9, slope)
        wp.atomic_add(diagnostics_f, 10, slope - stop_angle_rad)
        wp.atomic_add(diagnostics_f, 11, start_angle_rad - slope)
        if count > 1:
            wp.atomic_add(diagnostics_i, 1, 1)
            wp.atomic_add(diagnostics_f, 3, weight)
        retired_candidate = (
            h <= dry_tolerance
            and slope <= stop_angle_rad
            and latch[index] == 0
        )
        if retired_candidate:
            wp.atomic_add(diagnostics_i, 3, 1)
            wp.atomic_add(diagnostics_f, 12, weight)

    result = (
        initialize,
        union_pass,
        compress,
        aggregate_components,
        reduce_components,
        select_winner,
        winner_diagnostics,
        update_previous_and_tiles,
        mobilize,
        reduce_persistence,
    )
    _KERNELS[id(wp)] = result
    return result


@dataclass(frozen=True)
class DeviceLargeAvalancheResult:
    classification: str
    transitioned: bool
    unstable_cell_count: int
    connected_region_count: int
    largest_connected_cell_count: int
    largest_connected_area_m2: float
    largest_connected_mobilizable_volume_m3: float
    mean_excess_start_deg: float
    maximum_excess_start_deg: float
    persistence_s: float
    transferred_volume_m3: float
    gravity_initiation_impulse_kg_m_s: np.ndarray
    residual_seed_tile_ids: np.ndarray
    union_iterations: int
    parameter_basis: str
    sensitivity_case: str
    newly_activated_volume_m3: float
    reactivated_volume_m3: float
    newly_activated_area_m2: float
    reactivated_area_step_m2: float
    cumulative_resting_to_mobile_m3: float
    cumulative_mobile_to_resting_m3: float
    unique_cells_ever_activated: int
    cells_activated_more_than_once: int
    unique_activated_area_m2: float
    reactivated_area_m2: float
    retired_candidate_area_m2: float
    current_mobile_volume_m3: float
    moving_mobile_volume_m3: float
    maximum_mobile_speed_m_s: float
    active_tile_count: int
    net_terrain_volume_change_m3: float
    net_spatial_transfer_m3: float
    mean_activated_slope_deg: float
    maximum_activated_slope_deg: float
    mean_slope_minus_theta_stop_deg: float
    mean_theta_start_minus_slope_deg: float


class DeviceLargeAvalancheBridge:
    """Exact-cell GPU connected components with scalar host diagnostics."""

    backend_identity = "GPU_RUNTIME_DEVICE_CONNECTED_LARGE_AVALANCHE"
    NO_LARGE_EVENT = "NO_LARGE_EVENT"
    LOCAL_YIELD_MOBILE_PATH = "LOCAL_YIELD_MOBILE_PATH"
    LOCAL_STATIC_INSTABILITY = LOCAL_YIELD_MOBILE_PATH  # compatibility alias
    PERSISTING_LARGE_UNSTABLE_REGION = "PERSISTING_LARGE_UNSTABLE_REGION"
    LARGE_AVALANCHE_MOBILE_PATH = "LARGE_AVALANCHE_MOBILE_PATH"

    def __init__(
        self,
        state: DeviceBulkState,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        config: LargeAvalancheTransitionConfig,
    ) -> None:
        if state.grid is not grid or integrator.shape != grid.shape:
            raise ValueError("[GpuLargeAvalanche] state/grid/integrator mismatch")
        self.state = state
        self.material = material
        self.grid = grid
        self.integrator = integrator
        self.config = config
        self._persistence_s = 0.0
        self._has_previous = False

    def reset(self) -> None:
        self._persistence_s = 0.0
        self._has_previous = False

    def observe_and_maybe_mobilize(
        self, dt_s: float, *, release_settled_latches: bool = True
    ) -> DeviceLargeAvalancheResult:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[GpuLargeAvalanche] dt_s must be finite/positive")
        state = self.state
        rt = state.runtime
        wp = rt.wp
        kernels = _kernels(wp)
        size = state.size
        start_rad = float(np.deg2rad(self.material.start_angle_deg))
        stop_rad = float(np.deg2rad(self.material.stop_angle_deg))
        rt.launch(
            kernels[0],
            dim=size,
            inputs=[
                rt.arrays["z_base"], rt.arrays["b_eff"], rt.arrays["mobile"],
                rt.arrays["momentum_x"], rt.arrays["momentum_y"],
                rt.arrays["avalanche_unstable"], rt.arrays["avalanche_parent"],
                rt.arrays["avalanche_latch"], rt.arrays["avalanche_owned_surface"],
                rt.arrays["avalanche_owned_export_baseline"],
                rt.arrays["mobile_export_cumulative"], rt.arrays["weights"],
                rt.arrays["avalanche_slope"],
                rt.arrays["avalanche_gradient_x"], rt.arrays["avalanche_gradient_y"],
                rt.arrays["avalanche_severity"], self.grid.ny, self.grid.nx,
                self.grid.dx, self.grid.dy, float(np.tan(start_rad)),
                float(np.tan(stop_rad)),
                self.material.assumed_bulk_density_kg_m3, 9.81,
                self.material.cohesion_proxy_pa, self.config.mobilization_depth_m,
                self.config.dry_tolerance_m,
                self.config.mobile_activity_speed_m_s,
                int(bool(release_settled_latches)),
            ],
        )
        union_iterations = 0
        for _ in range(max(self.grid.shape)):
            rt.arrays["avalanche_changed"].zero_()
            rt.launch(
                kernels[1], dim=size,
                inputs=[rt.arrays["avalanche_parent"], rt.arrays["avalanche_changed"],
                        self.grid.ny, self.grid.nx, self.config.connectivity],
            )
            rt.launch(
                kernels[2], dim=size,
                inputs=[rt.arrays["avalanche_parent"], rt.arrays["avalanche_changed"]],
            )
            rt.synchronize()
            changed = np.asarray(
                rt.arrays["avalanche_changed"].numpy(), dtype=np.int32
            )
            rt.telemetry.record_d2h(changed)
            union_iterations += 1
            if int(changed[0]) == 0:
                break
        else:
            raise RuntimeError(
                "NUMERICAL_NONCONVERGENCE: device avalanche connected components"
            )
        for name in (
            "avalanche_component_count", "avalanche_component_area",
            "avalanche_component_volume", "avalanche_component_excess",
            "avalanche_diag_int", "avalanche_diag_float",
        ):
            rt.arrays[name].zero_()
        # Winner ID is reduced with atomic_min and therefore starts above every
        # possible flat cell index.
        diag_init = np.asarray([0, 0, size + 1, 0, 0, 0, 0, 0], dtype=np.int32)
        rt.upload("avalanche_diag_int", diag_init, dtype=wp.int32)
        rt.launch(
            kernels[3], dim=size,
            inputs=[
                rt.arrays["avalanche_parent"], rt.arrays["z_base"], rt.arrays["b_eff"],
                rt.arrays["avalanche_latch"], rt.arrays["weights"],
                rt.arrays["avalanche_slope"],
                rt.arrays["avalanche_severity"], rt.arrays["avalanche_component_count"],
                rt.arrays["avalanche_component_area"], rt.arrays["avalanche_component_volume"],
                rt.arrays["avalanche_component_excess"], self.config.mobilization_depth_m,
                start_rad,
            ],
        )
        rt.launch(
            kernels[4], dim=size,
            inputs=[rt.arrays["avalanche_component_count"], rt.arrays["avalanche_diag_int"]],
        )
        rt.launch(
            kernels[5], dim=size,
            inputs=[rt.arrays["avalanche_component_count"], rt.arrays["avalanche_diag_int"]],
        )
        rt.launch(
            kernels[6], dim=size,
            inputs=[
                rt.arrays["avalanche_parent"], rt.arrays["avalanche_previous_component"],
                rt.arrays["avalanche_slope"], rt.arrays["avalanche_component_area"],
                rt.arrays["avalanche_component_volume"], rt.arrays["avalanche_component_excess"],
                rt.arrays["avalanche_diag_int"], rt.arrays["avalanche_diag_float"],
                start_rad, stop_rad,
            ],
        )
        rt.synchronize()
        diag_i = np.asarray(rt.arrays["avalanche_diag_int"].numpy(), dtype=np.int32)
        diag_f = np.asarray(rt.arrays["avalanche_diag_float"].numpy(), dtype=np.float64)
        rt.telemetry.record_d2h(diag_i)
        rt.telemetry.record_d2h(diag_f)
        largest_count = int(diag_i[1])
        area = float(diag_f[4])
        mobilizable = float(diag_f[5])
        mean_excess_deg = (
            float(np.rad2deg(diag_f[0] / largest_count)) if largest_count else 0.0
        )
        max_excess_deg = float(np.rad2deg(diag_f[1]))
        extent_pass = (
            largest_count >= self.config.minimum_connected_cells
            and area >= self.config.minimum_connected_area_m2
        )
        volume_pass = mobilizable >= self.config.minimum_mobilizable_volume_m3
        excess_pass = mean_excess_deg >= self.config.minimum_mean_excess_start_deg
        candidate = extent_pass and volume_pass and excess_pass
        overlap = int(diag_i[4]) > 0
        if candidate:
            self._persistence_s = (
                self._persistence_s + dt
                if overlap or not self._has_previous
                else dt
            )
        else:
            self._persistence_s = 0.0
        large_ready = candidate and self._persistence_s >= self.config.persistence_time_s
        if large_ready:
            classification = self.LARGE_AVALANCHE_MOBILE_PATH
        elif candidate:
            classification = self.PERSISTING_LARGE_UNSTABLE_REGION
        elif int(diag_i[3]) > 0:
            classification = self.LOCAL_YIELD_MOBILE_PATH
        else:
            classification = self.NO_LARGE_EVENT

        state.runtime.arrays["dirty_tile_flags"].zero_()
        rt.launch(
            kernels[7], dim=size,
            inputs=[
                rt.arrays["avalanche_parent"], rt.arrays["avalanche_previous_component"],
                rt.arrays["frontier_reached"], rt.arrays["dirty_tile_flags"],
                rt.arrays["avalanche_diag_int"], self.grid.nx, state.tile_size,
                state.tile_shape[1], int(candidate), int(int(diag_i[3]) > 0),
            ],
        )
        self._has_previous = candidate
        transferred = 0.0
        impulse = np.zeros(2, dtype=np.float64)
        # Constitutive local yielding is immediate; persistence only promotes
        # the event classification to LARGE_AVALANCHE_MOBILE_PATH.
        if int(diag_i[3]) > 0:
            state.capture_surface_for_dirty_tracking()
            rt.launch(
                kernels[8], dim=size,
                inputs=[
                    rt.arrays["z_base"], rt.arrays["b_eff"], rt.arrays["mobile"],
                    rt.arrays["momentum_x"], rt.arrays["momentum_y"],
                    rt.arrays["weights"], rt.arrays["avalanche_parent"],
                    rt.arrays["avalanche_latch"], rt.arrays["avalanche_owned_surface"],
                    rt.arrays["avalanche_owned_export_baseline"],
                    rt.arrays["mobile_export_cumulative"],
                    rt.arrays["avalanche_gradient_x"],
                    rt.arrays["avalanche_gradient_y"], rt.arrays["avalanche_severity"],
                    rt.arrays["avalanche_diag_int"], rt.arrays["avalanche_diag_float"],
                    rt.arrays["avalanche_activation_count"],
                    rt.arrays["avalanche_first_activation_time"],
                    rt.arrays["avalanche_last_activation_time"],
                    rt.arrays["avalanche_r2m_cumulative"],
                    state.timestamp_device_s,
                    self.material.assumed_bulk_density_kg_m3,
                    self.config.mobilization_depth_m,
                    self.config.dry_tolerance_m,
                ],
            )
            rt.synchronize()
            diag_f = np.asarray(rt.arrays["avalanche_diag_float"].numpy(), dtype=np.float64)
            rt.telemetry.record_d2h(diag_f)
            transferred = float(diag_f[7])
            impulse = np.asarray(diag_f[8:10], dtype=np.float64)
            state.collect_surface_dirty_tiles()
        rt.synchronize()
        flags = np.asarray(rt.arrays["dirty_tile_flags"].numpy(), dtype=np.int32)
        rt.telemetry.record_d2h(flags)
        seed_tiles = np.flatnonzero(flags).astype(np.int32)
        persistence = self._persistence_summary(kernels[9], start_rad, stop_rad)
        return DeviceLargeAvalancheResult(
            classification=classification,
            transitioned=transferred > 0.0,
            unstable_cell_count=int(diag_i[3]),
            connected_region_count=int(diag_i[0]),
            largest_connected_cell_count=largest_count,
            largest_connected_area_m2=area,
            largest_connected_mobilizable_volume_m3=mobilizable,
            mean_excess_start_deg=mean_excess_deg,
            maximum_excess_start_deg=max_excess_deg,
            persistence_s=self._persistence_s,
            transferred_volume_m3=transferred,
            gravity_initiation_impulse_kg_m_s=impulse,
            residual_seed_tile_ids=seed_tiles,
            union_iterations=union_iterations,
            parameter_basis=self.config.parameter_basis,
            sensitivity_case=self.config.sensitivity_case,
            newly_activated_volume_m3=float(diag_f[10]),
            reactivated_volume_m3=float(diag_f[11]),
            newly_activated_area_m2=float(diag_f[12]),
            reactivated_area_step_m2=float(diag_f[13]),
            cumulative_resting_to_mobile_m3=persistence["cumulative_r2m_m3"],
            cumulative_mobile_to_resting_m3=persistence["cumulative_m2r_m3"],
            unique_cells_ever_activated=persistence["unique_cells_ever_activated"],
            cells_activated_more_than_once=persistence["cells_activated_more_than_once"],
            unique_activated_area_m2=persistence["unique_activated_area_m2"],
            reactivated_area_m2=persistence["reactivated_area_m2"],
            retired_candidate_area_m2=persistence["retired_candidate_area_m2"],
            current_mobile_volume_m3=persistence["current_mobile_volume_m3"],
            moving_mobile_volume_m3=persistence["moving_mobile_volume_m3"],
            maximum_mobile_speed_m_s=persistence["maximum_mobile_speed_m_s"],
            active_tile_count=persistence["active_tile_count"],
            net_terrain_volume_change_m3=persistence["net_terrain_volume_change_m3"],
            net_spatial_transfer_m3=persistence["net_spatial_transfer_m3"],
            mean_activated_slope_deg=persistence["mean_activated_slope_deg"],
            maximum_activated_slope_deg=persistence["maximum_activated_slope_deg"],
            mean_slope_minus_theta_stop_deg=persistence["mean_slope_minus_theta_stop_deg"],
            mean_theta_start_minus_slope_deg=persistence["mean_theta_start_minus_slope_deg"],
        )

    def _persistence_summary(
        self, kernel: Any, start_angle_rad: float, stop_angle_rad: float
    ) -> dict[str, int | float]:
        """Reduce resident lifecycle history to fixed-size acceptance scalars."""

        state = self.state
        rt = state.runtime
        rt.arrays["avalanche_persistence_int"].zero_()
        rt.arrays["avalanche_persistence_float"].zero_()
        rt.arrays["avalanche_activity_tile_flags"].zero_()
        rt.launch(
            kernel,
            dim=state.size,
            inputs=[
                rt.arrays["initial_resting"], rt.arrays["resting"],
                rt.arrays["mobile"], rt.arrays["momentum_x"],
                rt.arrays["momentum_y"], rt.arrays["weights"],
                rt.arrays["avalanche_slope"], rt.arrays["avalanche_unstable"],
                rt.arrays["avalanche_latch"],
                rt.arrays["avalanche_activation_count"],
                rt.arrays["avalanche_r2m_cumulative"],
                rt.arrays["avalanche_m2r_cumulative"],
                rt.arrays["avalanche_persistence_int"],
                rt.arrays["avalanche_persistence_float"],
                rt.arrays["avalanche_activity_tile_flags"],
                self.grid.nx, state.tile_size, state.tile_shape[1],
                start_angle_rad, stop_angle_rad,
                self.config.dry_tolerance_m,
                self.config.mobile_activity_speed_m_s,
            ],
        )
        rt.synchronize()
        values_i = np.asarray(
            rt.arrays["avalanche_persistence_int"].numpy(), dtype=np.int32
        )
        values_f = np.asarray(
            rt.arrays["avalanche_persistence_float"].numpy(), dtype=np.float64
        )
        tile_flags = np.asarray(
            rt.arrays["avalanche_activity_tile_flags"].numpy(), dtype=np.int32
        )
        for value in (values_i, values_f, tile_flags):
            rt.telemetry.record_d2h(value)
        unique = int(values_i[0])
        return {
            "unique_cells_ever_activated": unique,
            "cells_activated_more_than_once": int(values_i[1]),
            "current_mobile_cell_count": int(values_i[2]),
            "retired_candidate_cell_count": int(values_i[3]),
            "cumulative_r2m_m3": float(values_f[0]),
            "cumulative_m2r_m3": float(values_f[1]),
            "unique_activated_area_m2": float(values_f[2]),
            "reactivated_area_m2": float(values_f[3]),
            "current_mobile_volume_m3": float(values_f[4]),
            "moving_mobile_volume_m3": float(values_f[13]),
            "maximum_mobile_speed_m_s": float(values_f[5]),
            "net_terrain_volume_change_m3": float(values_f[6]),
            "net_spatial_transfer_m3": float(values_f[7]),
            "mean_activated_slope_deg": (
                float(np.rad2deg(values_f[8] / unique)) if unique else 0.0
            ),
            "maximum_activated_slope_deg": float(np.rad2deg(values_f[9])),
            "mean_slope_minus_theta_stop_deg": (
                float(np.rad2deg(values_f[10] / unique)) if unique else 0.0
            ),
            "mean_theta_start_minus_slope_deg": (
                float(np.rad2deg(values_f[11] / unique)) if unique else 0.0
            ),
            "retired_candidate_area_m2": float(values_f[12]),
            "active_tile_count": int(np.count_nonzero(tile_flags)),
        }

    def persistence_checkpoint(self, *, source: str = "acceptance") -> dict[str, np.ndarray]:
        """Download per-cell history only at an explicit diagnostic boundary."""

        if source not in {"acceptance", "checkpoint", "debug"}:
            raise ValueError("[GpuLargeAvalanche] invalid persistence checkpoint source")
        rt = self.state.runtime
        count = rt.download("avalanche_activation_count").reshape(self.grid.shape)
        first = rt.download("avalanche_first_activation_time").reshape(self.grid.shape)
        last = rt.download("avalanche_last_activation_time").reshape(self.grid.shape)
        slope = rt.download("avalanche_slope").reshape(self.grid.shape)
        mobile = rt.download("mobile").reshape(self.grid.shape)
        momentum_x = rt.download("momentum_x").reshape(self.grid.shape)
        momentum_y = rt.download("momentum_y").reshape(self.grid.shape)
        speed = np.divide(
            np.hypot(momentum_x, momentum_y), mobile,
            out=np.zeros_like(mobile), where=mobile > self.config.dry_tolerance_m,
        )
        activated = count > 0
        first = np.where(activated, first, -1.0)
        last = np.where(activated, last, -1.0)
        return {
            "activation_count": count,
            "first_activation_time_s": first,
            "last_activation_time_s": last,
            "resting_to_mobile_cumulative_m3": rt.download(
                "avalanche_r2m_cumulative"
            ).reshape(self.grid.shape),
            "mobile_to_resting_cumulative_m3": rt.download(
                "avalanche_m2r_cumulative"
            ).reshape(self.grid.shape),
            "local_slope_deg": np.rad2deg(slope),
            "slope_minus_theta_stop_deg": np.rad2deg(slope)
            - self.material.stop_angle_deg,
            "theta_start_minus_slope_deg": self.material.start_angle_deg
            - np.rad2deg(slope),
            "mobile_speed_m_s": speed,
            "current_mobile_height_m": mobile,
            "current_resting_height_m": rt.download("resting").reshape(
                self.grid.shape
            ),
            "current_mobile_momentum_x_m2_s": momentum_x,
            "current_mobile_momentum_y_m2_s": momentum_y,
            "avalanche_latch": rt.download("avalanche_latch").reshape(
                self.grid.shape
            ),
            "avalanche_previous_component": rt.download(
                "avalanche_previous_component"
            ).reshape(self.grid.shape),
            "frontier_reached": rt.download("frontier_reached").reshape(
                self.grid.shape
            ),
        }

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend_identity": self.backend_identity,
            "field_authority": "DEVICE",
            "host_transfer": "FIXED_SIZE_SCALARS_AND_TILE_FLAGS_ONLY",
            "parameter_basis": self.config.parameter_basis,
            "sensitivity_case": self.config.sensitivity_case,
        }
