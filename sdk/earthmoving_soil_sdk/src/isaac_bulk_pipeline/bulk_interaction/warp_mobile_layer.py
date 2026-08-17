"""GPU-resident Warp backend for the conservative Mobile Layer equations.

The public resident API avoids downloading 701x701 fields each physics step.
``download_result`` is an explicit acceptance/debug boundary and its traffic is
therefore visible in telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..performance import WarpRuntime
from ..terrain import TerrainGrid
from .mobile_layer import MobileLayerConfig, MobileLayerResult, MobileLayerSolver


_KERNELS: dict[int, Any] = {}


def _kernels(wp: Any) -> Any:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def measure_wave(
        height: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        pressure_gravity: wp.float64,
        dry_tolerance: wp.float64,
        maximum_wave: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        h = height[index]
        if h > dry_tolerance:
            vx = momentum_x[index] / h
            vy = momentum_y[index] / h
            wave = wp.sqrt(vx * vx + vy * vy) + wp.sqrt(pressure_gravity * h)
            wp.atomic_max(maximum_wave, 0, wave)

    @wp.kernel
    def apply_sources(
        resting: wp.array(dtype=wp.float64),
        height: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        forcing: wp.array(dtype=wp.int32),
        weights: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        dt: wp.float64,
        gravity: wp.float64,
        pressure_coefficient: wp.float64,
        friction_coefficient: wp.float64,
        cohesion_pa: wp.float64,
        tan_start_phi: wp.float64,
        tan_stop_phi: wp.float64,
        forcing_blend: wp.float64,
        tool_velocity_x: wp.float64,
        tool_velocity_y: wp.float64,
        dry_tolerance: wp.float64,
        density: wp.float64,
        gravity_impulse: wp.array(dtype=wp.float64),
        friction_impulse: wp.array(dtype=wp.float64),
        tool_impulse: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        row = index // cols
        col = index - row * cols
        h = height[index]
        if h <= dry_tolerance:
            velocity_x[index] = wp.float64(0.0)
            velocity_y[index] = wp.float64(0.0)
            return
        vx0 = momentum_x[index] / h
        vy0 = momentum_y[index] / h
        left = index
        right = index
        down = index
        up = index
        if col > 0:
            left = index - 1
        if col + 1 < cols:
            right = index + 1
        if row > 0:
            down = index - cols
        if row + 1 < rows:
            up = index + cols
        sx_left = resting[left] + height[left]
        sx_right = resting[right] + height[right]
        sy_down = resting[down] + height[down]
        sy_up = resting[up] + height[up]
        hx = dx * wp.float64(2.0)
        hy = dy * wp.float64(2.0)
        if col == 0 or col + 1 == cols:
            hx = dx
        if row == 0 or row + 1 == rows:
            hy = dy
        grad_x = (sx_right - sx_left) / hx
        grad_y = (sy_up - sy_down) / hy
        pressure_x = (height[right] - height[left]) / hx
        pressure_y = (height[up] - height[down]) / hy
        slope_tan = wp.sqrt(grad_x * grad_x + grad_y * grad_y)
        normalization = wp.sqrt(wp.float64(1.0) + slope_tan * slope_tan)
        sin_theta = slope_tan / normalization
        cos_theta = wp.float64(1.0) / normalization
        drive = density * gravity * h * sin_theta
        normal_stress = density * gravity * h * cos_theta
        start_margin = drive - (cohesion_pa + normal_stress * tan_start_phi)
        stop_margin = drive - (cohesion_pa + normal_stress * tan_stop_phi)
        moving = wp.sqrt(vx0 * vx0 + vy0 * vy0) > wp.float64(0.01)
        dynamic = start_margin > wp.float64(0.0)
        if moving and stop_margin > wp.float64(0.0):
            dynamic = True
        if forcing[index] != 0:
            dynamic = True
        vx_g = vx0
        vy_g = vy0
        if dynamic:
            vx_g = vx0 - gravity * (grad_x + pressure_coefficient * pressure_x) * dt
            vy_g = vy0 - gravity * (grad_y + pressure_coefficient * pressure_y) * dt
        mass = density * h * weights[index]
        wp.atomic_add(gravity_impulse, 0, mass * (vx_g - vx0))
        wp.atomic_add(gravity_impulse, 1, mass * (vy_g - vy0))
        speed = wp.sqrt(vx_g * vx_g + vy_g * vy_g)
        factor = wp.float64(0.0)
        if speed > 1.0e-12:
            factor = wp.max(
                wp.float64(0.0),
                wp.float64(1.0) - friction_coefficient * gravity * dt / speed,
            )
        vx_f = vx_g * factor
        vy_f = vy_g * factor
        wp.atomic_add(friction_impulse, 0, mass * (vx_f - vx_g))
        wp.atomic_add(friction_impulse, 1, mass * (vy_f - vy_g))
        vx = vx_f
        vy = vy_f
        if forcing[index] != 0:
            vx = vx_f + forcing_blend * (tool_velocity_x - vx_f)
            vy = vy_f + forcing_blend * (tool_velocity_y - vy_f)
        wp.atomic_add(tool_impulse, 0, mass * (vx - vx_f))
        wp.atomic_add(tool_impulse, 1, mass * (vy - vy_f))
        velocity_x[index] = vx
        velocity_y[index] = vy

    @wp.kernel
    def prepare_transport(
        height: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        volume: wp.array(dtype=wp.float64),
        momentum_volume_x: wp.array(dtype=wp.float64),
        momentum_volume_y: wp.array(dtype=wp.float64),
        outgoing: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        cell_volume = height[index] * weights[index]
        volume[index] = cell_volume
        momentum_volume_x[index] = cell_volume * velocity_x[index]
        momentum_volume_y[index] = cell_volume * velocity_y[index]
        outgoing[index] = wp.float64(0.0)

    @wp.kernel
    def edge_fluxes(
        height: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        donors: wp.array(dtype=wp.int32),
        receivers: wp.array(dtype=wp.int32),
        amounts: wp.array(dtype=wp.float64),
        outgoing: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        face_x: wp.float64,
        face_y: wp.float64,
        dt: wp.float64,
        x_edge_count: int,
    ):
        edge = wp.tid()
        first = 0
        second = 0
        face_velocity = wp.float64(0.0)
        face_length = face_x
        if edge < x_edge_count:
            row = edge // (cols - 1)
            col = edge - row * (cols - 1)
            first = row * cols + col
            second = first + 1
            face_velocity = wp.float64(0.5) * (velocity_x[first] + velocity_x[second])
            face_length = face_x
        else:
            local = edge - x_edge_count
            row = local // cols
            col = local - row * cols
            first = row * cols + col
            second = first + cols
            face_velocity = wp.float64(0.5) * (velocity_y[first] + velocity_y[second])
            face_length = face_y
        donor = first
        receiver = second
        if face_velocity < 0.0:
            donor = second
            receiver = first
        amount = height[donor] * wp.abs(face_velocity) * face_length * dt
        donors[edge] = donor
        receivers[edge] = receiver
        amounts[edge] = amount
        wp.atomic_add(outgoing, donor, amount)

    @wp.kernel
    def apply_fluxes(
        donors: wp.array(dtype=wp.int32),
        receivers: wp.array(dtype=wp.int32),
        amounts: wp.array(dtype=wp.float64),
        outgoing: wp.array(dtype=wp.float64),
        volume: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        momentum_volume_x: wp.array(dtype=wp.float64),
        momentum_volume_y: wp.array(dtype=wp.float64),
    ):
        edge = wp.tid()
        donor = donors[edge]
        receiver = receivers[edge]
        candidate = amounts[edge]
        total = outgoing[donor]
        factor = wp.float64(1.0)
        if total > volume[donor] and total > 0.0:
            factor = volume[donor] / total
        amount = candidate * factor
        carried_x = amount * velocity_x[donor]
        carried_y = amount * velocity_y[donor]
        wp.atomic_add(volume, donor, -amount)
        wp.atomic_add(volume, receiver, amount)
        wp.atomic_add(momentum_volume_x, donor, -carried_x)
        wp.atomic_add(momentum_volume_x, receiver, carried_x)
        wp.atomic_add(momentum_volume_y, donor, -carried_y)
        wp.atomic_add(momentum_volume_y, receiver, carried_y)

    @wp.kernel
    def finalize_transport(
        weights: wp.array(dtype=wp.float64),
        volume: wp.array(dtype=wp.float64),
        momentum_volume_x: wp.array(dtype=wp.float64),
        momentum_volume_y: wp.array(dtype=wp.float64),
        height: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        dry_tolerance: wp.float64,
        velocity_cap: wp.float64,
    ):
        index = wp.tid()
        cell_volume = wp.max(volume[index], wp.float64(0.0))
        h = cell_volume / weights[index]
        height[index] = h
        if h <= dry_tolerance or cell_volume <= 0.0:
            momentum_x[index] = wp.float64(0.0)
            momentum_y[index] = wp.float64(0.0)
            return
        vx = momentum_volume_x[index] / cell_volume
        vy = momentum_volume_y[index] / cell_volume
        speed = wp.sqrt(vx * vx + vy * vy)
        if speed > velocity_cap:
            vx = vx * velocity_cap / speed
            vy = vy * velocity_cap / speed
        momentum_x[index] = h * vx
        momentum_y[index] = h * vy

    @wp.kernel
    def summarize(
        height: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        density: wp.float64,
        summary: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        h = height[index]
        wp.atomic_add(summary, 0, h * weights[index])
        wp.atomic_add(summary, 1, density * momentum_x[index] * weights[index])
        wp.atomic_add(summary, 2, density * momentum_y[index] * weights[index])
        if h > 1.0e-12:
            vx = momentum_x[index] / h
            vy = momentum_y[index] / h
            wp.atomic_max(summary, 3, wp.sqrt(vx * vx + vy * vy))

    result = (
        measure_wave,
        apply_sources,
        prepare_transport,
        edge_fluxes,
        apply_fluxes,
        finalize_transport,
        summarize,
    )
    _KERNELS[id(wp)] = result
    return result


@dataclass(frozen=True)
class WarpMobileStep:
    volume_before_m3: float
    volume_after_m3: float
    momentum_before_terrain_kg_m_s: np.ndarray
    momentum_after_terrain_kg_m_s: np.ndarray
    gravity_pressure_impulse_terrain_ns: np.ndarray
    basal_friction_impulse_terrain_ns: np.ndarray
    tool_impulse_on_mobile_terrain_ns: np.ndarray
    numerical_dissipative_impulse_terrain_ns: np.ndarray
    maximum_speed_m_s: float
    substeps: int
    cfl_limited: bool


class WarpMobileLayerSolver:
    """Closed-boundary Mobile Layer with authoritative arrays on a Warp device."""

    backend_name = "GPU_WARP_MOBILE_LAYER_RESIDENT"

    def __init__(
        self,
        config: MobileLayerConfig | None = None,
        *,
        device: str = "cuda:0",
        runtime: WarpRuntime | None = None,
    ) -> None:
        self.config = config or MobileLayerConfig()
        if self.config.boundary_condition != "closed":
            raise NotImplementedError(
                "[WarpMobileLayer] open boundary GPU outflow is not implemented"
            )
        self.runtime = runtime or WarpRuntime(device)
        self._owns_runtime = runtime is None
        self.shape: tuple[int, int] | None = None
        self.grid: TerrainGrid | None = None
        self.material: MaterialScenario | None = None
        self._last_step: WarpMobileStep | None = None

    def _initialize_workspace(self) -> None:
        """Allocate only solver scratch; primary fields may be externally owned."""

        if self.shape is None or self.grid is None:
            raise RuntimeError("[WarpMobileLayer] grid is not initialized")
        wp = self.runtime.wp
        size = self.grid.nx * self.grid.ny
        x_edges = self.grid.ny * (self.grid.nx - 1)
        edge_count = x_edges + (self.grid.ny - 1) * self.grid.nx
        if "forcing" not in self.runtime.arrays:
            self.runtime.zeros("forcing", size, dtype=wp.int32)
        for name in (
            "velocity_x",
            "velocity_y",
            "volume",
            "momentum_volume_x",
            "momentum_volume_y",
            "outgoing",
        ):
            if name not in self.runtime.arrays:
                self.runtime.empty(name, size, dtype=wp.float64)
        for name in ("donors", "receivers"):
            if name not in self.runtime.arrays:
                self.runtime.empty(name, edge_count, dtype=wp.int32)
        if "amounts" not in self.runtime.arrays:
            self.runtime.empty("amounts", edge_count, dtype=wp.float64)

    def bind_device_state(
        self,
        state: Any,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> None:
        """Bind Mobile to a shared DeviceBulkState without copying terrain.

        ``DeviceBulkState`` owns ``resting``, ``mobile`` and momentum arrays.
        The historical kernel name ``height`` is an alias to ``mobile`` only;
        it never creates a second terrain copy.
        """

        if getattr(state, "runtime", None) is not self.runtime:
            raise ValueError("[WarpMobileLayer] state/runtime ownership mismatch")
        if tuple(getattr(state, "shape", ())) != grid.shape:
            raise ValueError("[WarpMobileLayer] state/grid shape mismatch")
        required = {
            "resting", "mobile", "momentum_x", "momentum_y", "weights",
            "material_mask",
        }
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(f"[WarpMobileLayer] missing shared state arrays: {sorted(missing)}")
        self.runtime.arrays["height"] = self.runtime.arrays["mobile"]
        # Tool activation belongs to the sole DeviceBulkState.  This alias
        # prevents a private per-operator forcing-mask shadow.
        self.runtime.arrays["forcing"] = self.runtime.arrays["material_mask"]
        self.shape = grid.shape
        self.grid = grid
        self.material = material
        self._initialize_workspace()

    def initialize_resident(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> None:
        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        height = np.asarray(mobile_height_m, dtype=np.float64)
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64)
        if height.shape != grid.shape or momentum.shape != grid.shape + (2,):
            raise ValueError("[WarpMobileLayer] state/grid shape mismatch")
        if integrator.shape != grid.shape:
            raise ValueError("[WarpMobileLayer] integrator/grid shape mismatch")
        self.shape = grid.shape
        self.grid = grid
        self.material = material
        wp = self.runtime.wp
        for name, value in (
            ("resting", resting),
            ("height", height),
            ("momentum_x", momentum[..., 0]),
            ("momentum_y", momentum[..., 1]),
            ("weights", integrator.vertex_weights_m2),
        ):
            self.runtime.upload(name, value.ravel(), dtype=wp.float64)
        self._initialize_workspace()

    def _small(self, array: Any) -> np.ndarray:
        self.runtime.synchronize()
        host = np.asarray(array.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(host)
        return host

    def _summary(self, density: float) -> np.ndarray:
        wp = self.runtime.wp
        summary = wp.zeros(4, dtype=wp.float64, device=self.runtime.device)
        summarize = _kernels(wp)[-1]
        assert self.shape is not None
        self.runtime.launch(
            summarize,
            dim=self.shape[0] * self.shape[1],
            inputs=[
                self.runtime.arrays["height"],
                self.runtime.arrays["momentum_x"],
                self.runtime.arrays["momentum_y"],
                self.runtime.arrays["weights"],
                density,
                summary,
            ],
        )
        return self._small(summary)

    def step_resident(
        self,
        dt_s: float,
        *,
        tool_forcing_mask: np.ndarray | None = None,
        tool_velocity_xy_m_s: np.ndarray | None = None,
    ) -> WarpMobileStep:
        if self.shape is None or self.grid is None or self.material is None:
            raise RuntimeError("[WarpMobileLayer] initialize resident state first")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("[WarpMobileLayer] dt_s must be finite/positive")
        wp = self.runtime.wp
        if tool_forcing_mask is not None:
            forcing = np.asarray(tool_forcing_mask, dtype=np.int32)
            if forcing.shape != self.shape:
                raise ValueError("[WarpMobileLayer] forcing mask shape mismatch")
            self.runtime.upload("forcing", forcing.ravel(), dtype=wp.int32)
        tool_velocity = (
            np.zeros(2, dtype=np.float64)
            if tool_velocity_xy_m_s is None
            else np.asarray(tool_velocity_xy_m_s, dtype=np.float64)
        )
        if tool_velocity.shape != (2,):
            raise ValueError("[WarpMobileLayer] tool velocity must have shape (2,)")
        density = float(self.material.assumed_bulk_density_kg_m3)
        before = self._summary(density)
        gravity_impulse = wp.zeros(2, dtype=wp.float64, device=self.runtime.device)
        friction_impulse = wp.zeros(2, dtype=wp.float64, device=self.runtime.device)
        tool_impulse = wp.zeros(2, dtype=wp.float64, device=self.runtime.device)
        kernels = _kernels(wp)
        size = self.shape[0] * self.shape[1]
        x_edge_count = self.grid.ny * (self.grid.nx - 1)
        edge_count = x_edge_count + (self.grid.ny - 1) * self.grid.nx
        remaining = float(dt_s)
        substeps = 0
        cfl_limited = False
        while remaining > 1.0e-14:
            if substeps >= self.config.maximum_substeps:
                raise RuntimeError("[WarpMobileLayer] maximum_substeps exceeded")
            maximum_wave = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
            self.runtime.launch(
                kernels[0],
                dim=size,
                inputs=[
                    self.runtime.arrays["height"],
                    self.runtime.arrays["momentum_x"],
                    self.runtime.arrays["momentum_y"],
                    self.config.pressure_coefficient * self.config.gravity_m_s2,
                    self.config.dry_tolerance_m,
                    maximum_wave,
                ],
            )
            wave = float(self._small(maximum_wave)[0])
            stable_dt = (
                remaining
                if wave <= 1.0e-12
                else self.config.cfl * min(self.grid.dx, self.grid.dy) / wave
            )
            sub_dt = min(remaining, stable_dt)
            cfl_limited |= sub_dt < remaining - 1.0e-14
            forcing_blend = min(1.0, sub_dt / self.config.tool_forcing_relaxation_s)
            self.runtime.launch(
                kernels[1],
                dim=size,
                inputs=[
                    self.runtime.arrays["resting"], self.runtime.arrays["height"],
                    self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                    self.runtime.arrays["forcing"], self.runtime.arrays["weights"],
                    self.runtime.arrays["velocity_x"], self.runtime.arrays["velocity_y"],
                    self.shape[0], self.shape[1], self.grid.dx, self.grid.dy, sub_dt,
                    self.config.gravity_m_s2, self.config.pressure_coefficient,
                    self.material.mobile_friction_coefficient,
                    self.material.cohesion_proxy_pa,
                    float(np.tan(np.deg2rad(self.material.start_angle_deg))),
                    float(np.tan(np.deg2rad(self.material.stop_angle_deg))),
                    forcing_blend,
                    float(tool_velocity[0]), float(tool_velocity[1]),
                    self.config.dry_tolerance_m, density,
                    gravity_impulse, friction_impulse, tool_impulse,
                ],
            )
            self.runtime.launch(
                kernels[2], dim=size,
                inputs=[
                    self.runtime.arrays["height"], self.runtime.arrays["weights"],
                    self.runtime.arrays["velocity_x"], self.runtime.arrays["velocity_y"],
                    self.runtime.arrays["volume"], self.runtime.arrays["momentum_volume_x"],
                    self.runtime.arrays["momentum_volume_y"], self.runtime.arrays["outgoing"],
                ],
            )
            self.runtime.launch(
                kernels[3], dim=edge_count,
                inputs=[
                    self.runtime.arrays["height"], self.runtime.arrays["velocity_x"],
                    self.runtime.arrays["velocity_y"], self.runtime.arrays["donors"],
                    self.runtime.arrays["receivers"], self.runtime.arrays["amounts"],
                    self.runtime.arrays["outgoing"], self.shape[0], self.shape[1],
                    self.grid.dy, self.grid.dx, sub_dt, x_edge_count,
                ],
            )
            self.runtime.launch(
                kernels[4], dim=edge_count,
                inputs=[
                    self.runtime.arrays["donors"], self.runtime.arrays["receivers"],
                    self.runtime.arrays["amounts"], self.runtime.arrays["outgoing"],
                    self.runtime.arrays["volume"], self.runtime.arrays["velocity_x"],
                    self.runtime.arrays["velocity_y"], self.runtime.arrays["momentum_volume_x"],
                    self.runtime.arrays["momentum_volume_y"],
                ],
            )
            self.runtime.launch(
                kernels[5], dim=size,
                inputs=[
                    self.runtime.arrays["weights"], self.runtime.arrays["volume"],
                    self.runtime.arrays["momentum_volume_x"], self.runtime.arrays["momentum_volume_y"],
                    self.runtime.arrays["height"], self.runtime.arrays["momentum_x"],
                    self.runtime.arrays["momentum_y"], self.config.dry_tolerance_m,
                    self.config.velocity_cap_m_s,
                ],
            )
            remaining -= sub_dt
            substeps += 1
        after = self._summary(density)
        gravity = self._small(gravity_impulse)
        friction = self._small(friction_impulse)
        tool = self._small(tool_impulse)
        momentum_before = np.asarray([before[1], before[2], 0.0])
        momentum_after = np.asarray([after[1], after[2], 0.0])
        gravity_3 = np.asarray([gravity[0], gravity[1], 0.0])
        friction_3 = np.asarray([friction[0], friction[1], 0.0])
        tool_3 = np.asarray([tool[0], tool[1], 0.0])
        step = WarpMobileStep(
            volume_before_m3=float(before[0]), volume_after_m3=float(after[0]),
            momentum_before_terrain_kg_m_s=momentum_before,
            momentum_after_terrain_kg_m_s=momentum_after,
            gravity_pressure_impulse_terrain_ns=gravity_3,
            basal_friction_impulse_terrain_ns=friction_3,
            tool_impulse_on_mobile_terrain_ns=tool_3,
            numerical_dissipative_impulse_terrain_ns=(momentum_after - momentum_before - gravity_3 - friction_3 - tool_3),
            maximum_speed_m_s=float(after[3]), substeps=substeps, cfl_limited=cfl_limited,
        )
        tolerance = max(1.0e-10, 1.0e-9 * max(step.volume_before_m3, 1.0))
        if abs(step.volume_after_m3 - step.volume_before_m3) > tolerance:
            raise RuntimeError("[WarpMobileLayer] closed-boundary volume conservation failed")
        self._last_step = step
        return step

    def download_result(self) -> MobileLayerResult:
        if self.shape is None or self._last_step is None:
            raise RuntimeError("[WarpMobileLayer] no completed resident step")
        height = self.runtime.download("height").reshape(self.shape)
        momentum = np.stack(
            (self.runtime.download("momentum_x").reshape(self.shape),
             self.runtime.download("momentum_y").reshape(self.shape)), axis=-1
        )
        step = self._last_step
        active = height > self.config.dry_tolerance_m
        return MobileLayerResult(
            mobile_height_m=height, mobile_momentum_m2_s=momentum,
            outflow_volume_m3=0.0, substeps=step.substeps, cfl_limited=step.cfl_limited,
            active_bbox_grid=MobileLayerSolver._bbox(active),
            volume_before_m3=step.volume_before_m3, volume_after_m3=step.volume_after_m3,
            maximum_speed_m_s=step.maximum_speed_m_s,
            minimum_height_m=float(np.min(height, initial=0.0)),
            momentum_before_terrain_kg_m_s=step.momentum_before_terrain_kg_m_s,
            momentum_after_terrain_kg_m_s=step.momentum_after_terrain_kg_m_s,
            gravity_pressure_impulse_terrain_ns=step.gravity_pressure_impulse_terrain_ns,
            basal_friction_impulse_terrain_ns=step.basal_friction_impulse_terrain_ns,
            tool_impulse_on_mobile_terrain_ns=step.tool_impulse_on_mobile_terrain_ns,
            numerical_dissipative_impulse_terrain_ns=step.numerical_dissipative_impulse_terrain_ns,
        )

    def diagnostics(self) -> dict[str, object]:
        result = self.runtime.diagnostics(self.backend_name)
        result.update({
            "resident_state": ["resting", "height", "momentum_x", "momentum_y", "weights"],
            "boundary_condition": "closed",
            "physics_semantics": "MobileLayerSolver conservative donor-limited edge fluxes",
            "host_sync_policy": "CFL_and_acceptance_reduction_scalars_only",
        })
        return result
