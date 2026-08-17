"""Production shared-state adapter for the accepted Mobile V2 DEVICE core.

The numerical kernels remain defined by the accepted standalone implementation;
this adapter binds them to authoritative ``b_eff/mobile/momentum`` buffers and
returns the existing production momentum-budget record. It never constructs a
host terrain field in the normal step path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..experimental.mobile_v2_reference import MobileV2Config
from ..experimental.mobile_v2_warp import _kernels as v2_kernels
from .warp_mobile_layer import WarpMobileStep


_ADAPTER_KERNELS: dict[int, tuple[Any, Any]] = {}


def _adapter_kernels(wp: Any) -> tuple[Any, Any]:
    cached = _ADAPTER_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def summarize(
        b_eff: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
        qx: wp.array(dtype=wp.float64),
        qy: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        K: wp.float64,
        g: wp.float64,
        density: wp.float64,
        dry: wp.float64,
        output: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        depth = h[i]
        weight = weights[i]
        wp.atomic_add(output, 0, depth * weight)
        wp.atomic_add(output, 1, density * qx[i] * weight)
        wp.atomic_add(output, 2, density * qy[i] * weight)
        if depth > dry:
            vx = qx[i] / depth
            vy = qy[i] / depth
            speed = wp.sqrt(vx * vx + vy * vy)
            wp.atomic_max(output, 3, speed)
            wp.atomic_add(output, 4, wp.float64(0.5) * density * depth * speed * speed * weight)
        wp.atomic_add(output, 5, density * g * depth * b_eff[i] * weight)
        wp.atomic_add(output, 6, density * wp.float64(0.5) * K * g * depth * depth * weight)

    @wp.kernel
    def static_diagnostic(
        b_eff: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
        qx: wp.array(dtype=wp.float64),
        qy: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        diagnostic: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        row = i // cols
        col = i - row * cols
        left = wp.max(col - 1, 0) + row * cols
        right = wp.min(col + 1, cols - 1) + row * cols
        down = wp.max(row - 1, 0) * cols + col
        up = wp.min(row + 1, rows - 1) * cols + col
        denom_x = dx
        denom_y = dy
        if col > 0 and col + 1 < cols:
            denom_x = wp.float64(2.0) * dx
        if row > 0 and row + 1 < rows:
            denom_y = wp.float64(2.0) * dy
        gx = (b_eff[right] + h[right] - b_eff[left] - h[left]) / denom_x
        gy = (b_eff[up] + h[up] - b_eff[down] - h[down]) / denom_y
        wp.atomic_max(diagnostic, 0, wp.sqrt(gx * gx + gy * gy))
        wp.atomic_max(diagnostic, 1, wp.abs(qx[i]))
        wp.atomic_max(diagnostic, 1, wp.abs(qy[i]))

    _ADAPTER_KERNELS[id(wp)] = (summarize, static_diagnostic)
    return summarize, static_diagnostic


class WarpProductionMobileV2Solver:
    """Accepted V2 equations consuming persistent production DEVICE fields."""

    backend_name = "GPU_WARP_MOBILE_V2_PRODUCTION_SHARED_STATE"

    def __init__(self, *, runtime: Any, config: MobileV2Config | None = None) -> None:
        self.runtime = runtime
        self.config = config or MobileV2Config()
        self.state: Any | None = None
        self.material: Any | None = None
        self.grid: Any | None = None

    def bind_device_state(self, state: Any, material: Any, grid: Any, integrator: Any) -> None:
        del integrator
        if state.runtime is not self.runtime:
            raise ValueError("[MobileV2Production] runtime mismatch")
        required = {"b_eff", "mobile", "momentum_x", "momentum_y", "weights"}
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(f"[MobileV2Production] missing state: {sorted(missing)}")
        self.state, self.material, self.grid = state, material, grid
        wp = self.runtime.wp
        for name in ("v2_dh", "v2_dqx", "v2_dqy", "v2_external_x", "v2_external_y"):
            if name not in self.runtime.arrays:
                self.runtime.zeros(name, state.size, dtype=wp.float64)

    def _small(self, value: Any) -> np.ndarray:
        self.runtime.synchronize()
        result = np.asarray(value.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(result)
        return result

    def _summary(self) -> np.ndarray:
        state = self.state
        wp = self.runtime.wp
        output = wp.zeros(7, dtype=wp.float64, device=self.runtime.device)
        self.runtime.launch(_adapter_kernels(wp)[0], dim=state.size, inputs=[
            self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
            self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
            self.runtime.arrays["weights"], self.config.earth_pressure_coefficient,
            self.config.gravity_m_s2, self.material.assumed_bulk_density_kg_m3,
            self.config.dry_tolerance_m, output,
        ])
        return self._small(output)

    def _static(self) -> bool:
        state = self.state
        wp = self.runtime.wp
        output = wp.zeros(2, dtype=wp.float64, device=self.runtime.device)
        self.runtime.launch(_adapter_kernels(wp)[1], dim=state.size, inputs=[
            self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
            self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
            state.shape[0], state.shape[1], state.grid.dx, state.grid.dy, output,
        ])
        value = self._small(output)
        return bool(value[1] <= 1.0e-14 and value[0] <= np.tan(np.deg2rad(self.config.start_angle_deg)) + 1.0e-14)

    def step_resident(self, dt_s: float, **_: Any) -> WarpMobileStep:
        if self.state is None:
            raise RuntimeError("[MobileV2Production] bind state first")
        state = self.state
        config = self.config
        wp = self.runtime.wp
        before = self._summary()
        if self._static():
            return WarpMobileStep(
                volume_before_m3=float(before[0]), volume_after_m3=float(before[0]),
                momentum_before_terrain_kg_m_s=np.asarray([before[1], before[2], 0.0]),
                momentum_after_terrain_kg_m_s=np.asarray([before[1], before[2], 0.0]),
                gravity_pressure_impulse_terrain_ns=np.zeros(3),
                basal_friction_impulse_terrain_ns=np.zeros(3),
                tool_impulse_on_mobile_terrain_ns=np.zeros(3),
                numerical_dissipative_impulse_terrain_ns=np.zeros(3),
                maximum_speed_m_s=float(before[3]), substeps=0, cfl_limited=False,
                kinetic_energy_before_j=float(before[4]), kinetic_energy_after_j=float(before[4]),
                gravity_pressure_work_j=0.0, basal_friction_work_j=0.0,
                tool_work_j=0.0, transport_numerical_energy_residual_j=0.0,
                donor_export_m3=0.0, receiver_import_m3=0.0,
                transport_mass_residual_m3=0.0,
                advected_momentum_crossings_kg_m_s=np.zeros(3),
            )
        kernels = v2_kernels(wp)
        x_edges = state.shape[0] * (state.shape[1] - 1)
        edge_count = x_edges + (state.shape[0] - 1) * state.shape[1]
        remaining = float(dt_s)
        substeps = 0
        cfl_limited = False
        diagnostics_total = np.zeros(8)
        self.runtime.arrays["v2_external_x"].zero_()
        self.runtime.arrays["v2_external_y"].zero_()
        while remaining > 1.0e-14:
            maximum = wp.zeros(1, dtype=wp.float64, device=self.runtime.device)
            self.runtime.launch(kernels[0], dim=state.size, inputs=[
                self.runtime.arrays["mobile"], self.runtime.arrays["momentum_x"],
                self.runtime.arrays["momentum_y"], config.earth_pressure_coefficient,
                config.gravity_m_s2, config.dry_tolerance_m, maximum,
            ])
            wave = float(self._small(maximum)[0])
            stable = remaining if wave <= 1.0e-14 else config.cfl * min(state.grid.dx, state.grid.dy) / wave
            sub_dt = min(remaining, stable)
            cfl_limited |= sub_dt < remaining - 1.0e-14
            self.runtime.launch(kernels[1], dim=state.size, inputs=[
                self.runtime.arrays["v2_dh"], self.runtime.arrays["v2_dqx"], self.runtime.arrays["v2_dqy"],
            ])
            self.runtime.launch(kernels[2], dim=edge_count, inputs=[
                self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                self.runtime.arrays["v2_dh"], self.runtime.arrays["v2_dqx"], self.runtime.arrays["v2_dqy"],
                state.shape[0], state.shape[1], x_edges, state.grid.dx, state.grid.dy,
                sub_dt, config.earth_pressure_coefficient, config.gravity_m_s2,
                config.dry_tolerance_m,
            ])
            diagnostic = wp.zeros(8, dtype=wp.float64, device=self.runtime.device)
            self.runtime.launch(kernels[3], dim=state.size, inputs=[
                self.runtime.arrays["mobile"], self.runtime.arrays["momentum_x"],
                self.runtime.arrays["momentum_y"], self.runtime.arrays["v2_dh"],
                self.runtime.arrays["v2_dqx"], self.runtime.arrays["v2_dqy"],
                self.runtime.arrays["v2_external_x"], self.runtime.arrays["v2_external_y"],
                sub_dt, config.gravity_m_s2, config.basal_friction_coefficient,
                config.dry_tolerance_m, diagnostic,
            ])
            diagnostics_total += self._small(diagnostic)
            remaining -= sub_dt
            substeps += 1
            if substeps > config.maximum_substeps:
                raise RuntimeError("[MobileV2Production] maximum_substeps exceeded")
        after = self._summary()
        density = self.material.assumed_bulk_density_kg_m3
        area = state.grid.dx * state.grid.dy
        gravity = np.asarray([diagnostics_total[0], diagnostics_total[1], 0.0]) * density * area
        friction = np.asarray([diagnostics_total[4], diagnostics_total[5], 0.0]) * density * area
        before_p = np.asarray([before[1], before[2], 0.0])
        after_p = np.asarray([after[1], after[2], 0.0])
        return WarpMobileStep(
            volume_before_m3=float(before[0]), volume_after_m3=float(after[0]),
            momentum_before_terrain_kg_m_s=before_p,
            momentum_after_terrain_kg_m_s=after_p,
            gravity_pressure_impulse_terrain_ns=gravity,
            basal_friction_impulse_terrain_ns=friction,
            tool_impulse_on_mobile_terrain_ns=np.zeros(3),
            numerical_dissipative_impulse_terrain_ns=after_p - before_p - gravity - friction,
            maximum_speed_m_s=float(after[3]), substeps=substeps, cfl_limited=cfl_limited,
            kinetic_energy_before_j=float(before[4]), kinetic_energy_after_j=float(after[4]),
            gravity_pressure_work_j=float((after[5] + after[6]) - (before[5] + before[6])),
            basal_friction_work_j=float(diagnostics_total[7] * density * area),
            tool_work_j=0.0,
            transport_numerical_energy_residual_j=float(
                (after[4] + after[5] + after[6])
                - (before[4] + before[5] + before[6])
                + diagnostics_total[7] * density * area
            ),
            donor_export_m3=0.0, receiver_import_m3=0.0,
            transport_mass_residual_m3=float(after[0] - before[0]),
            advected_momentum_crossings_kg_m_s=np.zeros(3),
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "state": ["z_base", "b_eff", "mobile", "momentum_x", "momentum_y"],
            "pressure_flux": "0.5*K*g*h^2_CONSERVATIVE_FACE_FLUX",
            "K_classification": "ENGINEERING_CLOSURE_UNCALIBRATED",
        }
