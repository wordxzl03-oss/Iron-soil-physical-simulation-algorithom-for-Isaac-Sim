"""Warp DEVICE port of the standalone Mobile V2 reference equations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..performance import WarpRuntime
from .mobile_v2_reference import (
    MobileV2Config,
    MobileV2ReferenceSolver,
    MobileV2State,
    MobileV2Step,
)


_KERNELS: dict[int, tuple[Any, ...]] = {}


def _kernels(wp: Any) -> tuple[Any, ...]:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def measure_wave(
        h: wp.array(dtype=wp.float64),
        qx: wp.array(dtype=wp.float64),
        qy: wp.array(dtype=wp.float64),
        K: wp.float64,
        g: wp.float64,
        dry: wp.float64,
        maximum: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        depth = h[i]
        if depth > dry:
            vx = qx[i] / depth
            vy = qy[i] / depth
            wp.atomic_max(maximum, 0, wp.sqrt(vx * vx + vy * vy) + wp.sqrt(K * g * depth))

    @wp.kernel
    def clear_updates(
        dh: wp.array(dtype=wp.float64),
        dqx: wp.array(dtype=wp.float64),
        dqy: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        dh[i] = wp.float64(0.0)
        dqx[i] = wp.float64(0.0)
        dqy[i] = wp.float64(0.0)

    @wp.kernel
    def faces(
        b: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
        qx: wp.array(dtype=wp.float64),
        qy: wp.array(dtype=wp.float64),
        dh: wp.array(dtype=wp.float64),
        dqx: wp.array(dtype=wp.float64),
        dqy: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        x_edges: int,
        dx: wp.float64,
        dy: wp.float64,
        dt: wp.float64,
        K: wp.float64,
        g: wp.float64,
        dry: wp.float64,
    ):
        edge = wp.tid()
        left = int(0)
        right = int(0)
        normal = int(0)
        spacing = dx
        if edge < x_edges:
            row = edge // (cols - 1)
            col = edge - row * (cols - 1)
            left = row * cols + col
            right = left + 1
            normal = 0
            spacing = dx
        else:
            local = edge - x_edges
            row = local // cols
            col = local - row * cols
            left = row * cols + col
            right = left + cols
            normal = 1
            spacing = dy
        hl = h[left]
        hr = h[right]
        head_l = b[left] + K * hl
        head_r = b[right] + K * hr
        b_face = wp.max(b[left], b[right])
        hsl = wp.max((head_l - b_face) / K, wp.float64(0.0))
        hsr = wp.max((head_r - b_face) / K, wp.float64(0.0))
        scale_l = wp.float64(0.0)
        scale_r = wp.float64(0.0)
        if hl > dry:
            scale_l = hsl / hl
        if hr > dry:
            scale_r = hsr / hr
        qxl = qx[left] * scale_l
        qyl = qy[left] * scale_l
        qxr = qx[right] * scale_r
        qyr = qy[right] * scale_r
        unl = wp.float64(0.0)
        unr = wp.float64(0.0)
        if hsl > dry:
            if normal == 0:
                unl = qxl / hsl
            else:
                unl = qyl / hsl
        if hsr > dry:
            if normal == 0:
                unr = qxr / hsr
            else:
                unr = qyr / hsr
        mass_l = qxl
        mass_r = qxr
        if normal == 1:
            mass_l = qyl
            mass_r = qyr
        vxl = wp.float64(0.0)
        vyl = wp.float64(0.0)
        vxr = wp.float64(0.0)
        vyr = wp.float64(0.0)
        if hsl > dry:
            vxl = qxl / hsl
            vyl = qyl / hsl
        if hsr > dry:
            vxr = qxr / hsr
            vyr = qyr / hsr
        fl_h = mass_l
        fr_h = mass_r
        fl_qx = mass_l * vxl
        fl_qy = mass_l * vyl
        fr_qx = mass_r * vxr
        fr_qy = mass_r * vyr
        if normal == 0:
            fl_qx = fl_qx + wp.float64(0.5) * K * g * hsl * hsl
            fr_qx = fr_qx + wp.float64(0.5) * K * g * hsr * hsr
        else:
            fl_qy = fl_qy + wp.float64(0.5) * K * g * hsl * hsl
            fr_qy = fr_qy + wp.float64(0.5) * K * g * hsr * hsr
        speed = wp.max(
            wp.abs(unl) + wp.sqrt(K * g * hsl),
            wp.abs(unr) + wp.sqrt(K * g * hsr),
        )
        flux_h = wp.float64(0.5) * (fl_h + fr_h) - wp.float64(0.5) * speed * (hsr - hsl)
        flux_qx = wp.float64(0.5) * (fl_qx + fr_qx) - wp.float64(0.5) * speed * (qxr - qxl)
        flux_qy = wp.float64(0.5) * (fl_qy + fr_qy) - wp.float64(0.5) * speed * (qyr - qyl)
        corr_l = wp.float64(0.5) * K * g * (hl * hl - hsl * hsl)
        corr_r = wp.float64(0.5) * K * g * (hr * hr - hsr * hsr)
        flux_l_qx = flux_qx
        flux_l_qy = flux_qy
        flux_r_qx = flux_qx
        flux_r_qy = flux_qy
        if normal == 0:
            flux_l_qx = flux_l_qx + corr_l
            flux_r_qx = flux_r_qx + corr_r
        else:
            flux_l_qy = flux_l_qy + corr_l
            flux_r_qy = flux_r_qy + corr_r
        factor = dt / spacing
        wp.atomic_add(dh, left, -factor * flux_h)
        wp.atomic_add(dh, right, factor * flux_h)
        wp.atomic_add(dqx, left, -factor * flux_l_qx)
        wp.atomic_add(dqx, right, factor * flux_r_qx)
        wp.atomic_add(dqy, left, -factor * flux_l_qy)
        wp.atomic_add(dqy, right, factor * flux_r_qy)

    @wp.kernel
    def apply_update_and_sources(
        h: wp.array(dtype=wp.float64),
        qx: wp.array(dtype=wp.float64),
        qy: wp.array(dtype=wp.float64),
        dh: wp.array(dtype=wp.float64),
        dqx: wp.array(dtype=wp.float64),
        dqy: wp.array(dtype=wp.float64),
        external_x: wp.array(dtype=wp.float64),
        external_y: wp.array(dtype=wp.float64),
        dt: wp.float64,
        g: wp.float64,
        mu: wp.float64,
        dry: wp.float64,
        diagnostics: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        old_qx = qx[i]
        old_qy = qy[i]
        depth = wp.max(h[i] + dh[i], wp.float64(0.0))
        cx = old_qx + dqx[i]
        cy = old_qy + dqy[i]
        wp.atomic_add(diagnostics, 0, cx - old_qx)
        wp.atomic_add(diagnostics, 1, cy - old_qy)
        if depth <= dry:
            h[i] = depth
            qx[i] = wp.float64(0.0)
            qy[i] = wp.float64(0.0)
            return
        vx0 = cx / depth
        vy0 = cy / depth
        vx1 = vx0 + external_x[i] * dt
        vy1 = vy0 + external_y[i] * dt
        wp.atomic_add(diagnostics, 2, depth * (vx1 - vx0))
        wp.atomic_add(diagnostics, 3, depth * (vy1 - vy0))
        wp.atomic_add(
            diagnostics, 6,
            wp.float64(0.5) * depth
            * (vx1 * vx1 + vy1 * vy1 - vx0 * vx0 - vy0 * vy0),
        )
        speed = wp.sqrt(vx1 * vx1 + vy1 * vy1)
        factor = wp.float64(0.0)
        if speed > wp.float64(1.0e-15):
            factor = wp.max(wp.float64(0.0), wp.float64(1.0) - mu * g * dt / speed)
        vx2 = vx1 * factor
        vy2 = vy1 * factor
        wp.atomic_add(diagnostics, 4, depth * (vx2 - vx1))
        wp.atomic_add(diagnostics, 5, depth * (vy2 - vy1))
        wp.atomic_add(
            diagnostics, 7,
            wp.float64(0.5) * depth
            * (vx1 * vx1 + vy1 * vy1 - vx2 * vx2 - vy2 * vy2),
        )
        h[i] = depth
        qx[i] = depth * vx2
        qy[i] = depth * vy2

    result = (measure_wave, clear_updates, faces, apply_update_and_sources)
    _KERNELS[id(wp)] = result
    return result


class WarpMobileV2ReferenceSolver:
    """Standalone DEVICE implementation; never bound to production state."""

    def __init__(
        self,
        config: MobileV2Config | None = None,
        *,
        device: str = "cuda:0",
    ) -> None:
        self.config = config or MobileV2Config()
        self.reference = MobileV2ReferenceSolver(self.config)
        self.runtime = WarpRuntime(device)

    def step(self, state: MobileV2State, dt_s: float) -> MobileV2Step:
        config = self.config
        shape = state.h_m.shape
        rows, cols = shape
        size = rows * cols
        rt = self.runtime
        wp = rt.wp
        for name, value in (
            ("v2_b", state.b_eff_m), ("v2_h", state.h_m),
            ("v2_qx", state.q_m2_s[..., 0]), ("v2_qy", state.q_m2_s[..., 1]),
        ):
            rt.upload(name, np.asarray(value).ravel(), dtype=wp.float64)
        for name in ("v2_dh", "v2_dqx", "v2_dqy"):
            rt.zeros(name, size, dtype=wp.float64)
        rt.zeros("v2_external_x", size, dtype=wp.float64)
        rt.zeros("v2_external_y", size, dtype=wp.float64)
        initial = MobileV2State(
            np.array(state.b_eff_m, copy=True), np.array(state.h_m, copy=True),
            np.array(state.q_m2_s, copy=True),
        )
        k0, g0, i0, e0 = self.reference.energy_terms(initial)
        m0 = self.reference.mass(initial)
        p0 = self.reference.momentum(initial)
        if self.reference._globally_static_below_start(
            initial.b_eff_m, initial.h_m, initial.q_m2_s
        ):
            return self.reference._result(
                initial, initial, float(dt_s), 0, 0.0, m0, p0,
                np.zeros(2), np.zeros(2), np.zeros(2),
                k0, g0, i0, e0, 0.0, 0.0,
            )
        kernels = _kernels(wp)
        x_edges = rows * (cols - 1)
        edge_count = x_edges + (rows - 1) * cols
        remaining = float(dt_s)
        substeps = 0
        max_cfl = 0.0
        diagnostics_total = np.zeros(8)
        while remaining > 1.0e-14:
            if substeps >= config.maximum_substeps:
                raise RuntimeError("Warp MobileV2 maximum_substeps exceeded")
            maximum = wp.zeros(1, dtype=wp.float64, device=rt.device)
            rt.launch(kernels[0], dim=size, inputs=[
                rt.arrays["v2_h"], rt.arrays["v2_qx"], rt.arrays["v2_qy"],
                config.earth_pressure_coefficient, config.gravity_m_s2,
                config.dry_tolerance_m, maximum,
            ])
            rt.synchronize()
            wave = float(np.asarray(maximum.numpy())[0])
            stable = (
                remaining if wave <= 1.0e-14
                else config.cfl * min(config.dx_m, config.dy_m) / wave
            )
            sub_dt = min(remaining, stable)
            max_cfl = max(max_cfl, wave * sub_dt / min(config.dx_m, config.dy_m))
            rt.launch(kernels[1], dim=size, inputs=[
                rt.arrays["v2_dh"], rt.arrays["v2_dqx"], rt.arrays["v2_dqy"],
            ])
            rt.launch(kernels[2], dim=edge_count, inputs=[
                rt.arrays["v2_b"], rt.arrays["v2_h"],
                rt.arrays["v2_qx"], rt.arrays["v2_qy"],
                rt.arrays["v2_dh"], rt.arrays["v2_dqx"], rt.arrays["v2_dqy"],
                rows, cols, x_edges, config.dx_m, config.dy_m, sub_dt,
                config.earth_pressure_coefficient, config.gravity_m_s2,
                config.dry_tolerance_m,
            ])
            diagnostics = wp.zeros(8, dtype=wp.float64, device=rt.device)
            rt.launch(kernels[3], dim=size, inputs=[
                rt.arrays["v2_h"], rt.arrays["v2_qx"], rt.arrays["v2_qy"],
                rt.arrays["v2_dh"], rt.arrays["v2_dqx"], rt.arrays["v2_dqy"],
                rt.arrays["v2_external_x"], rt.arrays["v2_external_y"],
                sub_dt, config.gravity_m_s2, config.basal_friction_coefficient,
                config.dry_tolerance_m, diagnostics,
            ])
            rt.synchronize()
            diagnostics_total += np.asarray(diagnostics.numpy(), dtype=np.float64)
            remaining -= sub_dt
            substeps += 1
        h = rt.download("v2_h").reshape(shape)
        q = np.stack(
            (rt.download("v2_qx").reshape(shape), rt.download("v2_qy").reshape(shape)),
            axis=-1,
        )
        final = MobileV2State(np.array(state.b_eff_m, copy=True), h, q)
        area = config.dx_m * config.dy_m
        return self.reference._result(
            initial, final, float(dt_s), substeps, max_cfl, m0, p0,
            diagnostics_total[0:2] * area,
            diagnostics_total[4:6] * area,
            diagnostics_total[2:4] * area,
            k0, g0, i0, e0,
            float(diagnostics_total[7] * area),
            float(diagnostics_total[6] * area),
        )
