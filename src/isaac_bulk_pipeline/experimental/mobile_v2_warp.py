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
        weights: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        K: wp.float64,
        g: wp.float64,
        dry: wp.float64,
        maxima: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        depth = h[i]
        if depth > dry:
            vx = qx[i] / depth
            vy = qy[i] / depth
            wave = wp.sqrt(vx * vx + vy * vy) + wp.sqrt(K * g * depth)
            row = i // cols
            col = i - row * cols
            # A face flux is per unit face length.  The most restrictive
            # local height rate is wave * L_face / A_i.  This reduces exactly
            # to wave/min(dx,dy) for equal-area interior vertices and becomes
            # appropriately stricter for smaller Triangle-A-C edge/corner CVs.
            inverse_length = wp.float64(0.0)
            if col > 0 or col + 1 < cols:
                inverse_length = wp.max(inverse_length, dy / weights[i])
            if row > 0 or row + 1 < rows:
                inverse_length = wp.max(inverse_length, dx / weights[i])
            wp.atomic_max(maxima, 0, wave * inverse_length)
            wp.atomic_max(maxima, 1, wave)

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
        weights: wp.array(dtype=wp.float64),
        ownership: wp.array(dtype=wp.int32),
        exported_cumulative: wp.array(dtype=wp.float64),
        flux_exported_cumulative: wp.array(dtype=wp.float64),
        transport_crossings: wp.array(dtype=wp.float64),
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
        face_length = dy
        if edge < x_edges:
            row = edge // (cols - 1)
            col = edge - row * (cols - 1)
            left = row * cols + col
            right = left + 1
            normal = 0
            face_length = dy
        else:
            local = edge - x_edges
            row = local // cols
            col = local - row * cols
            left = row * cols + col
            right = left + cols
            normal = 1
            face_length = dx
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
        # One shared face transfer is integrated once, then divided by each
        # endpoint's authoritative dual-control area.  Mass and the shared
        # advective/pressure momentum flux are thus weighted conservative.
        scale_l = dt * face_length / weights[left]
        scale_r = dt * face_length / weights[right]
        wp.atomic_add(dh, left, -scale_l * flux_h)
        wp.atomic_add(dh, right, scale_r * flux_h)
        wp.atomic_add(dqx, left, -scale_l * flux_l_qx)
        wp.atomic_add(dqx, right, scale_r * flux_r_qx)
        wp.atomic_add(dqy, left, -scale_l * flux_l_qy)
        wp.atomic_add(dqy, right, scale_r * flux_r_qy)

        # Bookkeeping is derived from the exact same authoritative shared-face
        # mass transfer used above.  ``transfer_volume`` is already a physical
        # volume [m^3]; it must never be divided by either endpoint's dual area.
        #
        # ``flux_exported_cumulative`` records every donor-side conservative
        # crossing as transport telemetry.  ``exported_cumulative`` is the
        # tranche-departure evidence used by LargeAvalanche and deliberately
        # records the *same real donor-side export even across owned->owned
        # faces*.  Requiring the receiver to be unowned deadlocks multi-cell
        # ownership chains (A->B->C->outside): interior donors would never get
        # departure evidence.  LargeAvalanche independently requires H_free to
        # fall below the activation-time owned surface, so reversible/internal
        # face traffic alone still cannot release a tranche.
        transfer_volume = flux_h * face_length * dt
        if transfer_volume > wp.float64(0.0):
            amount = transfer_volume
            wp.atomic_add(transport_crossings, 0, amount)
            wp.atomic_add(flux_exported_cumulative, left, amount)
            if ownership[left] != 0:
                wp.atomic_add(exported_cumulative, left, amount)
        elif transfer_volume < wp.float64(0.0):
            amount = -transfer_volume
            wp.atomic_add(transport_crossings, 0, amount)
            wp.atomic_add(flux_exported_cumulative, right, amount)
            if ownership[right] != 0:
                wp.atomic_add(exported_cumulative, right, amount)

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
        tool_contact_mask: wp.array(dtype=wp.int32),
        tool_normal_x: wp.array(dtype=wp.float64),
        tool_normal_y: wp.array(dtype=wp.float64),
        tool_normal_z: wp.array(dtype=wp.float64),
        tool_velocity_x: wp.array(dtype=wp.float64),
        tool_velocity_y: wp.array(dtype=wp.float64),
        tool_velocity_z: wp.array(dtype=wp.float64),
        tool_contact_point_x: wp.array(dtype=wp.float64),
        tool_contact_point_y: wp.array(dtype=wp.float64),
        tool_contact_point_z: wp.array(dtype=wp.float64),
        b_eff: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        dt: wp.float64,
        g: wp.float64,
        mu: wp.float64,
        tool_mu: wp.float64,
        tool_reference_x: wp.float64,
        tool_reference_y: wp.float64,
        tool_reference_z: wp.float64,
        dry: wp.float64,
        diagnostics: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        old_qx = qx[i]
        old_qy = qy[i]
        raw_depth = h[i] + dh[i]
        if raw_depth < wp.float64(0.0):
            # Positivity must come from the metric-aware CFL.  This diagnostic
            # makes any non-roundoff violation fatal on the host instead of
            # silently using clipping as a mass correction.
            wp.atomic_add(diagnostics, 8, -raw_depth * weights[i])
        depth = wp.max(raw_depth, wp.float64(0.0))
        cx = old_qx + dqx[i]
        cy = old_qy + dqy[i]
        weight = weights[i]
        wp.atomic_add(diagnostics, 0, (cx - old_qx) * weight)
        wp.atomic_add(diagnostics, 1, (cy - old_qy) * weight)
        if depth <= dry:
            h[i] = depth
            qx[i] = wp.float64(0.0)
            qy[i] = wp.float64(0.0)
            return
        vx0 = cx / depth
        vy0 = cy / depth
        vx_contact = vx0
        vy_contact = vy0
        if tool_contact_mask[i] != 0:
            # Rauter/Tukovic-style 3-D Cartesian kinematics while retaining a
            # depth-integrated state: reconstruct the material world velocity
            # from the local basal/support-surface tangent plane, resolve bucket contact
            # against the true 3-D bucket normal, then project the correction
            # back onto the terrain tangent.  This removes the former failure
            # mode where a near-vertical normal was normalized in XY and became
            # an artificial horizontal bulldozing impulse.
            row = i // cols
            col = i - row * cols
            # Warp 1.5.0 codegen does not support Python conditional
            # expressions (ast.IfExp) inside kernels. Keep the same
            # one-sided/central finite-difference stencil with explicit
            # branch assignments.
            left = i - 1
            if col == 0:
                left = i
            right = i + 1
            if col + 1 == cols:
                right = i
            down = i - cols
            if row == 0:
                down = i
            up = i + cols
            if row + 1 == rows:
                up = i
            denom_x = wp.float64(2.0) * dx
            if col == 0 or col + 1 == cols:
                denom_x = dx
            denom_y = wp.float64(2.0) * dy
            if row == 0 or row + 1 == rows:
                denom_y = dy
            gx = (b_eff[right] - b_eff[left]) / denom_x
            gy = (b_eff[up] - b_eff[down]) / denom_y
            terrain_normal = wp.normalize(wp.vec3d(-gx, -gy, wp.float64(1.0)))
            mobile3 = wp.vec3d(vx0, vy0, vx0 * gx + vy0 * gy)
            tool3 = wp.vec3d(
                tool_velocity_x[i], tool_velocity_y[i], tool_velocity_z[i]
            )
            bucket_normal = wp.normalize(wp.vec3d(
                tool_normal_x[i], tool_normal_y[i], tool_normal_z[i]
            ))
            relative3 = mobile3 - tool3
            closing3 = wp.dot(relative3, bucket_normal)
            if closing3 < wp.float64(0.0):
                normal_delta = -closing3

                # Full 3-D request is retained only as a dimensionality oracle.
                slip3 = relative3 - closing3 * bucket_normal
                slip_speed = wp.length(slip3)
                requested_tangent_delta3 = wp.vec3d(
                    wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
                )
                if slip_speed > wp.float64(1.0e-15):
                    requested_tangent_magnitude = wp.min(
                        slip_speed, tool_mu * normal_delta
                    )
                    requested_tangent_delta3 = (
                        -requested_tangent_magnitude / slip_speed
                    ) * slip3
                requested_delta3 = (
                    normal_delta * bucket_normal + requested_tangent_delta3
                )

                # The authoritative Mobile V2 state stores only q_x,q_y.
                # Contact detection above uses true 3-D geometry and the
                # kinematically reconstructed world velocity, but the accepted
                # reduced impulse must remain in world XY.  Do NOT renormalize
                # the surviving normal components: a near-vertical bucket face
                # must not turn into an artificial unit horizontal push.
                bucket_normal_xy = wp.vec3d(
                    bucket_normal[0], bucket_normal[1], wp.float64(0.0)
                )
                normal_represented_delta3 = normal_delta * bucket_normal_xy
                normal_xy_norm2 = wp.dot(bucket_normal_xy, bucket_normal_xy)

                relative_xy3 = wp.vec3d(
                    relative3[0], relative3[1], wp.float64(0.0)
                )
                slip_represented3 = relative_xy3
                if normal_xy_norm2 > wp.float64(1.0e-30):
                    slip_represented3 = (
                        slip_represented3
                        - bucket_normal_xy
                        * (
                            wp.dot(slip_represented3, bucket_normal_xy)
                            / normal_xy_norm2
                        )
                    )
                slip_represented_speed = wp.length(slip_represented3)
                tangent_represented_delta3 = wp.vec3d(
                    wp.float64(0.0), wp.float64(0.0), wp.float64(0.0)
                )
                represented_normal_magnitude = wp.length(
                    normal_represented_delta3
                )
                if (
                    slip_represented_speed > wp.float64(1.0e-15)
                    and represented_normal_magnitude > wp.float64(0.0)
                ):
                    tangent_represented_magnitude = wp.min(
                        slip_represented_speed,
                        tool_mu * represented_normal_magnitude,
                    )
                    tangent_represented_delta3 = (
                        -tangent_represented_magnitude
                        / slip_represented_speed
                    ) * slip_represented3

                raw_represented_delta3 = (
                    normal_represented_delta3 + tangent_represented_delta3
                )
                represented_norm2 = wp.dot(
                    raw_represented_delta3, raw_represented_delta3
                )
                passivity_scale = wp.float64(0.0)
                if represented_norm2 > wp.float64(1.0e-30):
                    available_work_per_mass = wp.dot(
                        raw_represented_delta3, tool3 - mobile3
                    )
                    if available_work_per_mass > wp.float64(0.0):
                        passivity_scale = wp.min(
                            wp.float64(1.0),
                            wp.float64(2.0) * available_work_per_mass
                            / represented_norm2,
                        )
                normal_accepted_delta3 = (
                    passivity_scale * normal_represented_delta3
                )
                tangent_accepted_delta3 = (
                    passivity_scale * tangent_represented_delta3
                )
                represented_delta3 = (
                    normal_accepted_delta3 + tangent_accepted_delta3
                )
                vx_contact = vx0 + represented_delta3[0]
                vy_contact = vy0 + represented_delta3[1]
                # Contact energy is measured in the authoritative horizontal
                # kinetic-energy state.  The reconstructed z velocity is an
                # oracle for 3-D closing, not stored q_z.
                impulse3 = depth * weight * represented_delta3
                requested_impulse3 = depth * weight * requested_delta3
                normal_impulse = (
                    depth * wp.length(normal_accepted_delta3) * weight
                )
                tangent_impulse = (
                    depth * wp.length(tangent_accepted_delta3) * weight
                )
                impulse_magnitude = wp.length(impulse3)
                kinetic_change = (
                    wp.float64(0.5) * depth
                    * (
                        vx_contact * vx_contact + vy_contact * vy_contact
                        - vx0 * vx0 - vy0 * vy0
                    )
                    * weight
                )
                tool_work = (
                    impulse3[0] * tool3[0] + impulse3[1] * tool3[1]
                )
                # Friction-only telemetry uses the accepted tangential
                # correction after the accepted normal correction.
                vx_after_normal = vx0 + normal_accepted_delta3[0]
                vy_after_normal = vy0 + normal_accepted_delta3[1]
                friction_kinetic_change = (
                    wp.float64(0.5) * depth
                    * (
                        vx_contact * vx_contact + vy_contact * vy_contact
                        - vx_after_normal * vx_after_normal
                        - vy_after_normal * vy_after_normal
                    )
                    * weight
                )
                friction_tool_work = (
                    depth * weight
                    * (
                        tangent_accepted_delta3[0] * tool3[0]
                        + tangent_accepted_delta3[1] * tool3[1]
                    )
                )
                friction_dissipation = wp.max(
                    wp.float64(0.0),
                    friction_tool_work - friction_kinetic_change,
                )
                rx = tool_contact_point_x[i] - tool_reference_x
                ry = tool_contact_point_y[i] - tool_reference_y
                rz = tool_contact_point_z[i] - tool_reference_z
                wp.atomic_add(diagnostics, 9, impulse3[0])
                wp.atomic_add(diagnostics, 10, impulse3[1])
                wp.atomic_add(diagnostics, 11, normal_impulse)
                wp.atomic_add(diagnostics, 12, tangent_impulse)
                wp.atomic_add(diagnostics, 13, kinetic_change)
                wp.atomic_add(diagnostics, 14, tool_work)
                wp.atomic_add(diagnostics, 15, friction_dissipation)
                wp.atomic_add(diagnostics, 16, ry * impulse3[2] - rz * impulse3[1])
                wp.atomic_add(diagnostics, 17, rz * impulse3[0] - rx * impulse3[2])
                wp.atomic_add(diagnostics, 18, rx * impulse3[1] - ry * impulse3[0])
                wp.atomic_add(diagnostics, 19, wp.float64(1.0))
                wp.atomic_add(diagnostics, 20, depth * weight)
                wp.atomic_add(diagnostics, 21, weight)
                wp.atomic_add(diagnostics, 22, tool_contact_point_x[i] * impulse_magnitude)
                wp.atomic_add(diagnostics, 23, tool_contact_point_y[i] * impulse_magnitude)
                wp.atomic_add(diagnostics, 24, tool_contact_point_z[i] * impulse_magnitude)
                wp.atomic_add(diagnostics, 25, impulse_magnitude)
                wp.atomic_add(diagnostics, 26, tool_work - kinetic_change)
                # New dimensionality audit: requested 3-D impulse versus the
                # terrain-tangent impulse representable by this 2.5-D state.
                wp.atomic_add(diagnostics, 27, requested_impulse3[0])
                wp.atomic_add(diagnostics, 28, requested_impulse3[1])
                wp.atomic_add(diagnostics, 29, requested_impulse3[2])
                wp.atomic_add(diagnostics, 30, impulse3[2])
                wp.atomic_add(
                    diagnostics, 31, wp.length(requested_impulse3 - impulse3)
                )
                wp.atomic_add(diagnostics, 32, wp.abs(closing3))

        # General external acceleration remains a separate source.  Tool
        # contact is an impulse above, not a hidden acceleration field.
        vx1 = vx_contact + external_x[i] * dt
        vy1 = vy_contact + external_y[i] * dt
        wp.atomic_add(diagnostics, 2, depth * (vx1 - vx_contact) * weight)
        wp.atomic_add(diagnostics, 3, depth * (vy1 - vy_contact) * weight)
        wp.atomic_add(
            diagnostics, 6,
            wp.float64(0.5) * depth
            * (
                vx1 * vx1 + vy1 * vy1
                - vx_contact * vx_contact - vy_contact * vy_contact
            ) * weight,
        )
        speed = wp.sqrt(vx1 * vx1 + vy1 * vy1)
        factor = wp.float64(0.0)
        if speed > wp.float64(1.0e-15):
            factor = wp.max(wp.float64(0.0), wp.float64(1.0) - mu * g * dt / speed)
        vx2 = vx1 * factor
        vy2 = vy1 * factor
        wp.atomic_add(diagnostics, 4, depth * (vx2 - vx1) * weight)
        wp.atomic_add(diagnostics, 5, depth * (vy2 - vy1) * weight)
        wp.atomic_add(
            diagnostics, 7,
            wp.float64(0.5) * depth
            * (vx1 * vx1 + vy1 * vy1 - vx2 * vx2 - vy2 * vy2) * weight,
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
        rt.zeros("v2_ownership", size, dtype=wp.int32)
        rt.zeros("v2_export_cumulative", size, dtype=wp.float64)
        rt.zeros("v2_flux_export_cumulative", size, dtype=wp.float64)
        rt.upload(
            "v2_weights",
            np.full(size, config.dx_m * config.dy_m, dtype=np.float64),
            dtype=wp.float64,
        )
        rt.zeros("v2_external_x", size, dtype=wp.float64)
        rt.zeros("v2_external_y", size, dtype=wp.float64)
        rt.zeros("v2_tool_contact_mask", size, dtype=wp.int32)
        for name in (
            "v2_tool_normal3_x", "v2_tool_normal3_y", "v2_tool_normal3_z",
            "v2_tool_velocity_x", "v2_tool_velocity_y", "v2_tool_velocity_z",
            "v2_tool_contact_point_x", "v2_tool_contact_point_y",
            "v2_tool_contact_point_z",
        ):
            rt.zeros(name, size, dtype=wp.float64)
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
        diagnostics_total = np.zeros(33)
        transport_crossings = wp.zeros(1, dtype=wp.float64, device=rt.device)
        while remaining > 1.0e-14:
            if substeps >= config.maximum_substeps:
                raise RuntimeError("Warp MobileV2 maximum_substeps exceeded")
            maximum = wp.zeros(2, dtype=wp.float64, device=rt.device)
            rt.launch(kernels[0], dim=size, inputs=[
                rt.arrays["v2_h"], rt.arrays["v2_qx"], rt.arrays["v2_qy"],
                rt.arrays["v2_weights"], rows, cols, config.dx_m, config.dy_m,
                config.earth_pressure_coefficient, config.gravity_m_s2,
                config.dry_tolerance_m, maximum,
            ])
            rt.synchronize()
            maxima = np.asarray(maximum.numpy())
            metric_rate = float(maxima[0])
            wave = float(maxima[1])
            stable = (
                remaining if metric_rate <= 1.0e-14
                else config.cfl / metric_rate
            )
            sub_dt = min(remaining, stable)
            max_cfl = max(max_cfl, metric_rate * sub_dt)
            rt.launch(kernels[1], dim=size, inputs=[
                rt.arrays["v2_dh"], rt.arrays["v2_dqx"], rt.arrays["v2_dqy"],
            ])
            rt.launch(kernels[2], dim=edge_count, inputs=[
                rt.arrays["v2_b"], rt.arrays["v2_h"],
                rt.arrays["v2_qx"], rt.arrays["v2_qy"],
                rt.arrays["v2_weights"],
                rt.arrays["v2_ownership"],
                rt.arrays["v2_export_cumulative"],
                rt.arrays["v2_flux_export_cumulative"],
                transport_crossings,
                rt.arrays["v2_dh"], rt.arrays["v2_dqx"], rt.arrays["v2_dqy"],
                rows, cols, x_edges, config.dx_m, config.dy_m, sub_dt,
                config.earth_pressure_coefficient, config.gravity_m_s2,
                config.dry_tolerance_m,
            ])
            diagnostics = wp.zeros(33, dtype=wp.float64, device=rt.device)
            rt.launch(kernels[3], dim=size, inputs=[
                rt.arrays["v2_h"], rt.arrays["v2_qx"], rt.arrays["v2_qy"],
                rt.arrays["v2_dh"], rt.arrays["v2_dqx"], rt.arrays["v2_dqy"],
                rt.arrays["v2_external_x"], rt.arrays["v2_external_y"],
                rt.arrays["v2_tool_contact_mask"],
                rt.arrays["v2_tool_normal3_x"], rt.arrays["v2_tool_normal3_y"],
                rt.arrays["v2_tool_normal3_z"],
                rt.arrays["v2_tool_velocity_x"], rt.arrays["v2_tool_velocity_y"],
                rt.arrays["v2_tool_velocity_z"],
                rt.arrays["v2_tool_contact_point_x"],
                rt.arrays["v2_tool_contact_point_y"],
                rt.arrays["v2_tool_contact_point_z"],
                rt.arrays["v2_b"], rt.arrays["v2_weights"],
                rows, cols, config.dx_m, config.dy_m,
                sub_dt, config.gravity_m_s2, config.basal_friction_coefficient,
                wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0),
                config.dry_tolerance_m, diagnostics,
            ])
            rt.synchronize()
            diagnostics_total += np.asarray(diagnostics.numpy(), dtype=np.float64)
            positivity_tolerance_m3 = (
                128.0 * np.finfo(np.float64).eps * max(1.0, m0)
            )
            if diagnostics_total[8] > positivity_tolerance_m3:
                raise RuntimeError(
                    "Warp MobileV2 metric-aware CFL positivity failure: "
                    f"clipped_volume_m3={diagnostics_total[8]:.17g}"
                )
            remaining -= sub_dt
            substeps += 1
        h = rt.download("v2_h").reshape(shape)
        q = np.stack(
            (rt.download("v2_qx").reshape(shape), rt.download("v2_qy").reshape(shape)),
            axis=-1,
        )
        final = MobileV2State(np.array(state.b_eff_m, copy=True), h, q)
        return self.reference._result(
            initial, final, float(dt_s), substeps, max_cfl, m0, p0,
            diagnostics_total[0:2],
            diagnostics_total[4:6],
            diagnostics_total[2:4],
            k0, g0, i0, e0,
            float(diagnostics_total[7]),
            float(diagnostics_total[6]),
        )
