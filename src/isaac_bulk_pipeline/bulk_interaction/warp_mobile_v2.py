"""Production shared-state adapter for the accepted Mobile V2 DEVICE core.

The numerical kernels remain defined by the accepted standalone implementation;
this adapter binds them to authoritative ``b_eff/mobile/momentum`` buffers and
returns the existing production momentum-budget record. It never constructs a
host terrain field in the normal step path.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from ..experimental.mobile_v2_reference import MobileV2Config
from ..experimental.mobile_v2_warp import _kernels as v2_kernels
from .warp_mobile_layer import WarpMobileStep


_ADAPTER_KERNELS: dict[int, tuple[Any, Any]] = {}
_CONTACT_SCATTER_KERNELS: dict[int, Any] = {}


def _contact_scatter_kernel(wp: Any) -> Any:
    cached = _CONTACT_SCATTER_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def scatter(
        indices: wp.array(dtype=wp.int32),
        normal_x_in: wp.array(dtype=wp.float64),
        normal_y_in: wp.array(dtype=wp.float64),
        normal_z_in: wp.array(dtype=wp.float64),
        velocity_x_in: wp.array(dtype=wp.float64),
        velocity_y_in: wp.array(dtype=wp.float64),
        velocity_z_in: wp.array(dtype=wp.float64),
        point_x_in: wp.array(dtype=wp.float64),
        point_y_in: wp.array(dtype=wp.float64),
        point_z_in: wp.array(dtype=wp.float64),
        mask: wp.array(dtype=wp.int32),
        normal_x: wp.array(dtype=wp.float64),
        normal_y: wp.array(dtype=wp.float64),
        normal_z: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        velocity_z: wp.array(dtype=wp.float64),
        point_x: wp.array(dtype=wp.float64),
        point_y: wp.array(dtype=wp.float64),
        point_z: wp.array(dtype=wp.float64),
    ):
        source = wp.tid()
        target = indices[source]
        mask[target] = 1
        normal_x[target] = normal_x_in[source]
        normal_y[target] = normal_y_in[source]
        normal_z[target] = normal_z_in[source]
        velocity_x[target] = velocity_x_in[source]
        velocity_y[target] = velocity_y_in[source]
        velocity_z[target] = velocity_z_in[source]
        point_x[target] = point_x_in[source]
        point_y[target] = point_y_in[source]
        point_z[target] = point_z_in[source]

    _CONTACT_SCATTER_KERNELS[id(wp)] = scatter
    return scatter


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
            wp.atomic_add(output, 7, wp.float64(1.0))
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
        # Audit-only callback after each *real fused* device substep.  The
        # production kernel intentionally fuses face/topography, tool impulse,
        # external source and basal friction; the diagnostic vector preserves
        # their conservative integrated contributions without inventing state
        # boundaries that do not exist.
        self.audit_substep_observer: (
            Callable[[int, float, np.ndarray], None] | None
        ) = None

    def bind_device_state(self, state: Any, material: Any, grid: Any, integrator: Any) -> None:
        del integrator
        if state.runtime is not self.runtime:
            raise ValueError("[MobileV2Production] runtime mismatch")
        required = {
            "b_eff", "mobile", "momentum_x", "momentum_y", "weights",
            "avalanche_latch", "mobile_export_cumulative",
            "mobile_flux_export_cumulative",
        }
        missing = required - set(self.runtime.arrays)
        if missing:
            raise ValueError(f"[MobileV2Production] missing state: {sorted(missing)}")
        self.state, self.material, self.grid = state, material, grid
        wp = self.runtime.wp
        for name in ("v2_dh", "v2_dqx", "v2_dqy", "v2_external_x", "v2_external_y"):
            if name not in self.runtime.arrays:
                self.runtime.zeros(name, state.size, dtype=wp.float64)
        if "v2_tool_contact_mask" not in self.runtime.arrays:
            self.runtime.zeros("v2_tool_contact_mask", state.size, dtype=wp.int32)
        for name in (
            "v2_tool_normal3_x", "v2_tool_normal3_y", "v2_tool_normal3_z",
            "v2_tool_velocity_x", "v2_tool_velocity_y", "v2_tool_velocity_z",
            "v2_tool_contact_point_x", "v2_tool_contact_point_y",
            "v2_tool_contact_point_z",
        ):
            if name not in self.runtime.arrays:
                self.runtime.zeros(name, state.size, dtype=wp.float64)

    def _prepare_tool_contact(self, contact: Any | None) -> tuple[int, np.ndarray]:
        runtime = self.runtime
        if contact is not None and bool(getattr(contact, "device_resident", False)):
            if int(contact.runtime_identity) != id(runtime):
                raise ValueError("[MobileV2Production] device contact runtime mismatch")
            return int(contact.cell_count), np.asarray(
                contact.tool_reference_position_terrain_m, dtype=np.float64
            )
        for name in (
            "v2_tool_contact_mask",
            "v2_tool_normal3_x", "v2_tool_normal3_y", "v2_tool_normal3_z",
            "v2_tool_velocity_x", "v2_tool_velocity_y", "v2_tool_velocity_z",
            "v2_tool_contact_point_x", "v2_tool_contact_point_y",
            "v2_tool_contact_point_z",
        ):
            runtime.arrays[name].zero_()
        if contact is None or int(contact.cell_count) == 0:
            return 0, np.zeros(3, dtype=np.float64)
        wp = runtime.wp
        compact = (
            ("v2_contact_indices", contact.flat_indices, wp.int32),
            ("v2_contact_nx", contact.outward_normals_terrain[:, 0], wp.float64),
            ("v2_contact_ny", contact.outward_normals_terrain[:, 1], wp.float64),
            ("v2_contact_nz", contact.outward_normals_terrain[:, 2], wp.float64),
            ("v2_contact_tvx", contact.tool_surface_velocity_terrain_m_s[:, 0], wp.float64),
            ("v2_contact_tvy", contact.tool_surface_velocity_terrain_m_s[:, 1], wp.float64),
            ("v2_contact_tvz", contact.tool_surface_velocity_terrain_m_s[:, 2], wp.float64),
            ("v2_contact_px", contact.closest_points_terrain_m[:, 0], wp.float64),
            ("v2_contact_py", contact.closest_points_terrain_m[:, 1], wp.float64),
            ("v2_contact_pz", contact.closest_points_terrain_m[:, 2], wp.float64),
        )
        arrays = [runtime.upload(name, np.asarray(value), dtype=dtype) for name, value, dtype in compact]
        runtime.launch(
            _contact_scatter_kernel(wp),
            dim=int(contact.cell_count),
            inputs=[
                *arrays,
                runtime.arrays["v2_tool_contact_mask"],
                runtime.arrays["v2_tool_normal3_x"],
                runtime.arrays["v2_tool_normal3_y"],
                runtime.arrays["v2_tool_normal3_z"],
                runtime.arrays["v2_tool_velocity_x"],
                runtime.arrays["v2_tool_velocity_y"],
                runtime.arrays["v2_tool_velocity_z"],
                runtime.arrays["v2_tool_contact_point_x"],
                runtime.arrays["v2_tool_contact_point_y"],
                runtime.arrays["v2_tool_contact_point_z"],
            ],
        )
        return int(contact.cell_count), np.asarray(
            contact.tool_reference_position_terrain_m, dtype=np.float64
        )

    def _small(self, value: Any) -> np.ndarray:
        self.runtime.synchronize()
        result = np.asarray(value.numpy(), dtype=np.float64)
        self.runtime.telemetry.record_d2h(result)
        return result

    def _summary(self) -> np.ndarray:
        state = self.state
        wp = self.runtime.wp
        output = wp.zeros(8, dtype=wp.float64, device=self.runtime.device)
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

    def step_resident(
        self,
        dt_s: float,
        *,
        tool_mobile_contact: Any | None = None,
        tool_mobile_friction_coefficient: float | None = None,
        **_: Any,
    ) -> WarpMobileStep:
        if self.state is None:
            raise RuntimeError("[MobileV2Production] bind state first")
        state = self.state
        config = self.config
        wp = self.runtime.wp
        step_start = perf_counter()
        h2d_bytes_before = int(self.runtime.telemetry.h2d_bytes)
        h2d_count_before = int(self.runtime.telemetry.h2d_transfer_count)
        contact_prepare_start = perf_counter()
        contact_count, tool_reference = self._prepare_tool_contact(tool_mobile_contact)
        contact_prepare_ms = (perf_counter() - contact_prepare_start) * 1_000.0
        contact_h2d_bytes = int(self.runtime.telemetry.h2d_bytes) - h2d_bytes_before
        contact_h2d_count = (
            int(self.runtime.telemetry.h2d_transfer_count) - h2d_count_before
        )
        tool_mu = (
            0.0
            if tool_mobile_friction_coefficient is None
            else float(tool_mobile_friction_coefficient)
        )
        if not np.isfinite(tool_mu) or tool_mu < 0.0:
            raise ValueError("[MobileV2Production] tool friction must be finite/non-negative")
        # Explicit counterfactual audit only.  Geometry/contact evidence remains
        # resident and is still observed, while the fused production kernel
        # consumes an immutable zero mask.  Normal production never sets this.
        diagnostic_tool_ablation = bool(
            getattr(self, "diagnostic_disable_tool_mobile_impulse", False)
        )
        if diagnostic_tool_ablation:
            if "v2_tool_contact_mask_ablation_zero" not in self.runtime.arrays:
                self.runtime.zeros(
                    "v2_tool_contact_mask_ablation_zero", state.size, dtype=wp.int32
                )
            tool_contact_mask = self.runtime.arrays[
                "v2_tool_contact_mask_ablation_zero"
            ]
        else:
            tool_contact_mask = self.runtime.arrays["v2_tool_contact_mask"]
        summary_start = perf_counter()
        before = self._summary()
        summary_sync_ms = (perf_counter() - summary_start) * 1_000.0
        if contact_count == 0 and self._static():
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
                mobile_nonzero_cell_count=int(round(before[7])),
                tool_mobile_contact_prepare_dispatch_ms=contact_prepare_ms,
                tool_mobile_contact_h2d_bytes=contact_h2d_bytes,
                tool_mobile_contact_h2d_transfer_count=contact_h2d_count,
                mobile_summary_sync_ms=summary_sync_ms,
                mobile_step_total_ms=(perf_counter() - step_start) * 1_000.0,
            )
        kernels = v2_kernels(wp)
        x_edges = state.shape[0] * (state.shape[1] - 1)
        edge_count = x_edges + (state.shape[0] - 1) * state.shape[1]
        remaining = float(dt_s)
        substeps = 0
        cfl_limited = False
        diagnostics_total = np.zeros(33)
        transport_crossings = wp.zeros(
            1, dtype=wp.float64, device=self.runtime.device
        )
        contact_substeps: list[dict[str, object]] = []
        cfl_measure_sync_ms = 0.0
        transport_and_source_sync_ms = 0.0
        self.runtime.arrays["v2_external_x"].zero_()
        self.runtime.arrays["v2_external_y"].zero_()
        while remaining > 1.0e-14:
            phase_start = perf_counter()
            maximum = wp.zeros(2, dtype=wp.float64, device=self.runtime.device)
            self.runtime.launch(kernels[0], dim=state.size, inputs=[
                self.runtime.arrays["mobile"], self.runtime.arrays["momentum_x"],
                self.runtime.arrays["momentum_y"], self.runtime.arrays["weights"],
                state.shape[0], state.shape[1], state.grid.dx, state.grid.dy,
                config.earth_pressure_coefficient,
                config.gravity_m_s2, config.dry_tolerance_m, maximum,
            ])
            maxima = self._small(maximum)
            cfl_measure_sync_ms += (perf_counter() - phase_start) * 1_000.0
            metric_rate = float(maxima[0])
            stable = remaining if metric_rate <= 1.0e-14 else config.cfl / metric_rate
            sub_dt = min(remaining, stable)
            cfl_limited |= sub_dt < remaining - 1.0e-14
            phase_start = perf_counter()
            self.runtime.launch(kernels[1], dim=state.size, inputs=[
                self.runtime.arrays["v2_dh"], self.runtime.arrays["v2_dqx"], self.runtime.arrays["v2_dqy"],
            ])
            self.runtime.launch(kernels[2], dim=edge_count, inputs=[
                self.runtime.arrays["b_eff"], self.runtime.arrays["mobile"],
                self.runtime.arrays["momentum_x"], self.runtime.arrays["momentum_y"],
                self.runtime.arrays["weights"],
                self.runtime.arrays["avalanche_latch"],
                self.runtime.arrays["mobile_export_cumulative"],
                self.runtime.arrays["mobile_flux_export_cumulative"],
                transport_crossings,
                self.runtime.arrays["v2_dh"], self.runtime.arrays["v2_dqx"], self.runtime.arrays["v2_dqy"],
                state.shape[0], state.shape[1], x_edges, state.grid.dx, state.grid.dy,
                sub_dt, config.earth_pressure_coefficient, config.gravity_m_s2,
                config.dry_tolerance_m,
            ])
            diagnostic = wp.zeros(33, dtype=wp.float64, device=self.runtime.device)
            self.runtime.launch(kernels[3], dim=state.size, inputs=[
                self.runtime.arrays["mobile"], self.runtime.arrays["momentum_x"],
                self.runtime.arrays["momentum_y"], self.runtime.arrays["v2_dh"],
                self.runtime.arrays["v2_dqx"], self.runtime.arrays["v2_dqy"],
                self.runtime.arrays["v2_external_x"], self.runtime.arrays["v2_external_y"],
                tool_contact_mask,
                self.runtime.arrays["v2_tool_normal3_x"],
                self.runtime.arrays["v2_tool_normal3_y"],
                self.runtime.arrays["v2_tool_normal3_z"],
                self.runtime.arrays["v2_tool_velocity_x"],
                self.runtime.arrays["v2_tool_velocity_y"],
                self.runtime.arrays["v2_tool_velocity_z"],
                self.runtime.arrays["v2_tool_contact_point_x"],
                self.runtime.arrays["v2_tool_contact_point_y"],
                self.runtime.arrays["v2_tool_contact_point_z"],
                self.runtime.arrays["b_eff"], self.runtime.arrays["weights"],
                state.shape[0], state.shape[1], state.grid.dx, state.grid.dy,
                sub_dt, config.gravity_m_s2, config.basal_friction_coefficient,
                tool_mu,
                float(tool_reference[0]), float(tool_reference[1]), float(tool_reference[2]),
                config.dry_tolerance_m, diagnostic,
            ])
            substep_diagnostic = self._small(diagnostic)
            transport_and_source_sync_ms += (
                perf_counter() - phase_start
            ) * 1_000.0
            diagnostics_total += substep_diagnostic
            if self.audit_substep_observer is not None:
                self.audit_substep_observer(
                    int(substeps), float(sub_dt), substep_diagnostic.copy()
                )
            density = self.material.assumed_bulk_density_kg_m3
            contact_substeps.append({
                "mobile_substep_index": substeps,
                "dt_sub_s": float(sub_dt),
                "contact_active_cell_count": int(round(substep_diagnostic[19])),
                "contact_mobile_volume_m3": float(substep_diagnostic[20]),
                "contact_weighted_area_m2": float(substep_diagnostic[21]),
                "requested_tool_to_mobile_impulse_xyz_ns": (
                    density * substep_diagnostic[27:30]
                ).tolist(),
                "accepted_tool_to_mobile_impulse_xyz_ns": (
                    density * np.asarray([
                        substep_diagnostic[9], substep_diagnostic[10],
                        substep_diagnostic[30],
                    ])
                ).tolist(),
                "unresolved_terrain_normal_impulse_ns": float(
                    density * substep_diagnostic[31]
                ),
                "normal_impulse_ns": float(density * substep_diagnostic[11]),
                "tangential_impulse_ns": float(density * substep_diagnostic[12]),
                "mobile_kinetic_energy_change_due_to_contact_j": float(
                    density * substep_diagnostic[13]
                ),
                "tool_to_mobile_work_j": float(density * substep_diagnostic[14]),
                "machine_reaction_work_j": float(-density * substep_diagnostic[14]),
                "frictional_dissipation_j": float(density * substep_diagnostic[15]),
                "total_contact_dissipation_j": float(density * substep_diagnostic[26]),
                "machine_reaction_impulse_xyz_ns": (
                    -density * np.asarray([
                        substep_diagnostic[9], substep_diagnostic[10],
                        substep_diagnostic[30],
                    ])
                ).tolist(),
            })
            positivity_tolerance_m3 = (
                128.0 * np.finfo(np.float64).eps * max(1.0, float(before[0]))
            )
            if diagnostics_total[8] > positivity_tolerance_m3:
                raise RuntimeError(
                    "[MobileV2Production] metric-aware CFL positivity failure: "
                    f"clipped_volume_m3={diagnostics_total[8]:.17g}"
                )
            remaining -= sub_dt
            substeps += 1
            if substeps > config.maximum_substeps:
                raise RuntimeError("[MobileV2Production] maximum_substeps exceeded")
        summary_start = perf_counter()
        after = self._summary()
        summary_sync_ms += (perf_counter() - summary_start) * 1_000.0
        gross_transport_volume_m3 = float(self._small(transport_crossings)[0])
        density = self.material.assumed_bulk_density_kg_m3
        gravity = np.asarray([diagnostics_total[0], diagnostics_total[1], 0.0]) * density
        friction = np.asarray([diagnostics_total[4], diagnostics_total[5], 0.0]) * density
        tool = np.asarray([
            diagnostics_total[9], diagnostics_total[10], diagnostics_total[30]
        ]) * density
        angular_tool = np.asarray(diagnostics_total[16:19]) * density
        before_p = np.asarray([before[1], before[2], 0.0])
        after_p = np.asarray([after[1], after[2], 0.0])
        return WarpMobileStep(
            volume_before_m3=float(before[0]), volume_after_m3=float(after[0]),
            momentum_before_terrain_kg_m_s=before_p,
            momentum_after_terrain_kg_m_s=after_p,
            gravity_pressure_impulse_terrain_ns=gravity,
            basal_friction_impulse_terrain_ns=friction,
            tool_impulse_on_mobile_terrain_ns=tool,
            numerical_dissipative_impulse_terrain_ns=after_p - before_p - gravity - friction - tool,
            maximum_speed_m_s=float(after[3]), substeps=substeps, cfl_limited=cfl_limited,
            kinetic_energy_before_j=float(before[4]), kinetic_energy_after_j=float(after[4]),
            gravity_pressure_work_j=float((after[5] + after[6]) - (before[5] + before[6])),
            basal_friction_work_j=float(diagnostics_total[7] * density),
            tool_work_j=float(diagnostics_total[14] * density),
            transport_numerical_energy_residual_j=float(
                (after[4] + after[5] + after[6])
                - (before[4] + before[5] + before[6])
                + diagnostics_total[7] * density
                + diagnostics_total[26] * density
                - diagnostics_total[14] * density
            ),
            donor_export_m3=gross_transport_volume_m3,
            receiver_import_m3=gross_transport_volume_m3,
            transport_mass_residual_m3=float(after[0] - before[0]),
            advected_momentum_crossings_kg_m_s=np.zeros(3),
            tool_normal_impulse_ns=float(diagnostics_total[11] * density),
            tool_tangential_impulse_ns=float(diagnostics_total[12] * density),
            tool_angular_impulse_on_mobile_about_tool_origin_terrain_nms=angular_tool,
            tool_contact_centroid_terrain_m=(
                np.asarray(diagnostics_total[22:25] / diagnostics_total[25])
                if diagnostics_total[25] > 0.0
                else np.asarray(tool_reference)
            ),
            tool_frictional_dissipation_j=float(diagnostics_total[15] * density),
            tool_contact_dissipation_j=float(diagnostics_total[26] * density),
            machine_reaction_work_j=float(-diagnostics_total[14] * density),
            tool_contact_active_substeps=sum(
                int(record["contact_active_cell_count"] > 0) for record in contact_substeps
            ),
            tool_contact_active_cell_substeps=sum(
                int(record["contact_active_cell_count"]) for record in contact_substeps
            ),
            peak_tool_contact_mobile_volume_m3=max(
                (float(record["contact_mobile_volume_m3"]) for record in contact_substeps),
                default=0.0,
            ),
            tool_contact_substep_diagnostics=tuple(contact_substeps),
            mobile_nonzero_cell_count=int(round(after[7])),
            tool_mobile_contact_prepare_dispatch_ms=contact_prepare_ms,
            tool_mobile_contact_h2d_bytes=contact_h2d_bytes,
            tool_mobile_contact_h2d_transfer_count=contact_h2d_count,
            mobile_summary_sync_ms=summary_sync_ms,
            mobile_cfl_measure_sync_ms=cfl_measure_sync_ms,
            mobile_transport_and_source_sync_ms=transport_and_source_sync_ms,
            tool_mobile_impulse_fused_upper_bound_ms=(
                transport_and_source_sync_ms if contact_count > 0 else 0.0
            ),
            mobile_step_total_ms=(perf_counter() - step_start) * 1_000.0,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "state": ["z_base", "b_eff", "mobile", "momentum_x", "momentum_y"],
            "pressure_flux": "0.5*K*g*h^2_CONSERVATIVE_FACE_FLUX",
            "discrete_measure": "TRIANGLE_A_C_VERTEX_DUAL_CONTROL_AREA",
            "face_transfer": "ONE_SHARED_FLUX_TIMES_FACE_LENGTH_TIMES_DT",
            "cfl_metric": "MAX_LOCAL_WAVE_TIMES_FACE_LENGTH_OVER_DUAL_AREA",
            "K_classification": "ENGINEERING_CLOSURE_UNCALIBRATED",
            "tool_contact_kinematics": (
                "FULL_3D_BUCKET_NORMAL+SURFACE_TANGENT_CLOSING_ORACLE"
                "+STRICT_XY_REPRESENTABLE_IMPULSE"
            ),
            "dimensionality_audit": (
                "REQUESTED_3D_IMPULSE_VS_ACCEPTED_XY_REDUCED_IMPULSE"
            ),
        }
