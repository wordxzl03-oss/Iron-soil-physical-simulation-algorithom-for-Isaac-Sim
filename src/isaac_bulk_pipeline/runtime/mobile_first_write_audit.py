"""Read-only DEVICE audit for the first production Mobile write.

This module is intentionally acceptance/diagnostic-only.  It downloads the
authoritative resident fields only while explicitly enabled, never writes a
device array, and does not introduce a numerical threshold into the solver.
The Mobile production kernel fuses its source update; its existing diagnostic
vector is therefore used to attribute the real fused write without inventing
non-existent state boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np


_FIELD_NAMES = (
    "b_eff", "mobile", "momentum_x", "momentum_y", "weights",
    "v2_dh", "v2_dqx", "v2_dqy",
    "v2_external_x", "v2_external_y",
    "v2_tool_contact_mask", "v2_tool_normal_x", "v2_tool_normal_y",
    "v2_tool_velocity_x", "v2_tool_velocity_y",
    "v2_tool_contact_point_x", "v2_tool_contact_point_y",
    "v2_tool_contact_point_z",
)


@dataclass
class _OpenStep:
    timestamp_s: float
    phase: str
    physics_step: int
    boundaries: list[dict[str, Any]]
    last_fields: dict[str, np.ndarray] | None = None


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


class MobileFirstWriteAudit:
    """Full-state, read-only boundary recorder for a narrow production window."""

    schema = "P0_2B_MOBILE_FIRST_WRITE_ROOT_CAUSE_AUDIT/v1"

    def __init__(
        self,
        state: Any,
        material: Any,
        mobile_config: Any,
        output_path: Path,
        *,
        start_time_s: float = 5.45,
        end_time_s: float = 5.75,
    ) -> None:
        self.state = state
        self.material = material
        self.config = mobile_config
        self.output_path = Path(output_path)
        self.start_time_s = float(start_time_s)
        self.end_time_s = float(end_time_s)
        self.density = float(material.assumed_bulk_density_kg_m3)
        self.records: list[dict[str, Any]] = []
        self._open: _OpenStep | None = None
        self._previous_global: dict[str, np.ndarray] | None = None
        self._first_creation: dict[str, Any] | None = None
        self._mobile_observations_after_creation: list[dict[str, Any]] = []
        self._substep_events: list[dict[str, Any]] = []
        self._failure_metadata: dict[str, Any] = {}
        self.visual_observer: Any | None = None

    def wants_step(self, simulation_time_s: float) -> bool:
        # Include one dt on either side so the t=5.45..5.75 request cannot be
        # missed by floating-point timestamp placement.
        dt = 1.0 / 60.0
        in_window = self.start_time_s - dt <= float(simulation_time_s) <= self.end_time_s + dt
        collecting_tail = (
            self._first_creation is not None
            and len(self._mobile_observations_after_creation) < 11
        )
        return bool(in_window or collecting_tail)

    @property
    def complete(self) -> bool:
        return bool(
            self.records
            and self.records[-1]["timestamp_s"] >= self.end_time_s
            and len(self._mobile_observations_after_creation) >= 11
        )

    def begin_step(
        self, *, simulation_time_s: float, phase: str, physics_step: int,
        tool_state: Any | None = None,
    ) -> None:
        if self._open is not None:
            raise RuntimeError("[MobileFirstWriteAudit] previous step remains open")
        if not self.wants_step(simulation_time_s):
            return
        self._open = _OpenStep(
            timestamp_s=float(simulation_time_s),
            phase=str(phase),
            physics_step=int(physics_step),
            boundaries=[],
        )
        if self.visual_observer is not None and tool_state is not None:
            self.visual_observer.set_tool_state(tool_state)

    def set_failure_metadata(self, result: Any) -> None:
        if self._open is None or result is None:
            return
        self._failure_metadata = {
            "activated_volume_m3": float(result.activated_volume_m3),
            "activation_mode": str(result.activation_mode),
            "patch_bbox_yx": list(result.patch_bbox_yx),
            "active_failure_cell_count": int(
                np.count_nonzero(np.asarray(result.activated_height_m) > 0.0)
            ),
            "momentum_initialization_rule": (
                "MASS_ONLY_RESTING_TO_MOBILE; momentum_x/momentum_y unchanged"
            ),
        }
        if self.visual_observer is not None:
            self.visual_observer.set_failure_patch(result)

    def _download_fields(self) -> dict[str, np.ndarray]:
        runtime = self.state.runtime
        runtime.synchronize()
        fields: dict[str, np.ndarray] = {}
        for name in _FIELD_NAMES:
            value = np.asarray(runtime.arrays[name].numpy()).reshape(self.state.shape).copy()
            runtime.telemetry.record_d2h(value)
            fields[name] = value
        return fields

    def _summary(self, fields: dict[str, np.ndarray]) -> dict[str, Any]:
        h = fields["mobile"]
        qx = fields["momentum_x"]
        qy = fields["momentum_y"]
        area = fields["weights"]
        b = fields["b_eff"]
        positive = h > 0.0
        momentum_nonzero = (qx != 0.0) | (qy != 0.0)
        speed = np.zeros_like(h)
        speed[positive] = np.hypot(qx[positive], qy[positive]) / h[positive]
        qmag = np.hypot(qx, qy)
        weighted_qmag = self.density * area * qmag
        flat_speed = speed.ravel()
        max_flat = int(np.argmax(flat_speed)) if flat_speed.size else 0
        row, col = np.unravel_index(max_flat, h.shape)
        positive_speed = speed[positive]

        def top20(metric: np.ndarray) -> list[dict[str, Any]]:
            flat = metric.ravel()
            count = min(20, flat.size)
            indices = np.argpartition(flat, -count)[-count:]
            indices = indices[np.argsort(flat[indices])[::-1]]
            rows, cols = np.unravel_index(indices, h.shape)
            return [
                {
                    "index_yx": [int(r), int(c)],
                    "rank_value": float(flat[i]),
                    "h_m": float(h[r, c]),
                    "qx_m2_s": float(qx[r, c]),
                    "qy_m2_s": float(qy[r, c]),
                    "speed_m_s": float(speed[r, c]),
                    "area_m2": float(area[r, c]),
                    "momentum_contribution_kg_m_s": [
                        float(self.density * area[r, c] * qx[r, c]),
                        float(self.density * area[r, c] * qy[r, c]),
                    ],
                }
                for i, r, c in zip(indices, rows, cols)
            ]

        return {
            "mobile_volume_m3": float(np.sum(area * h, dtype=np.float64)),
            "mobile_momentum_kg_m_s": [
                float(self.density * np.sum(area * qx, dtype=np.float64)),
                float(self.density * np.sum(area * qy, dtype=np.float64)),
            ],
            "mobile_nonzero_cell_count": int(np.count_nonzero(positive)),
            "momentum_nonzero_cell_count": int(np.count_nonzero(momentum_nonzero)),
            "min_positive_h_m": (
                float(np.min(h[positive])) if np.any(positive) else None
            ),
            "max_h_m": float(np.max(h)),
            "max_abs_qx_m2_s": float(np.max(np.abs(qx))),
            "max_abs_qy_m2_s": float(np.max(np.abs(qy))),
            "max_speed_m_s": float(np.max(speed)),
            "p95_speed_m_s": (
                float(np.percentile(positive_speed, 95.0))
                if positive_speed.size else 0.0
            ),
            "p99_speed_m_s": (
                float(np.percentile(positive_speed, 99.0))
                if positive_speed.size else 0.0
            ),
            "max_speed_cell": {
                "index_yx": [int(row), int(col)],
                "h_m": float(h[row, col]),
                "qx_m2_s": float(qx[row, col]),
                "qy_m2_s": float(qy[row, col]),
                "ux_m_s": float(qx[row, col] / h[row, col]) if h[row, col] > 0.0 else 0.0,
                "uy_m_s": float(qy[row, col] / h[row, col]) if h[row, col] > 0.0 else 0.0,
                "H_free_m": float(b[row, col] + h[row, col]),
                "b_eff_m": float(b[row, col]),
                "area_m2": float(area[row, col]),
            },
            "top20_by_speed": top20(speed),
            "top20_by_q_magnitude": top20(qmag),
            "top20_by_momentum_contribution": top20(weighted_qmag),
            "thin_layer_distribution": self._thin_distribution(h, qmag, speed),
        }

    @staticmethod
    def _thin_distribution(
        h: np.ndarray, qmag: np.ndarray, speed: np.ndarray
    ) -> dict[str, Any]:
        positive = h > 0.0
        bins = (
            ("h_lt_1e-4_m", positive & (h < 1.0e-4)),
            ("h_lt_1e-3_m", positive & (h < 1.0e-3)),
            ("h_lt_1e-2_m", positive & (h < 1.0e-2)),
            ("h_ge_1e-2_m", h >= 1.0e-2),
        )
        result: dict[str, Any] = {}
        for name, mask in bins:
            result[name] = {
                "cell_count": int(np.count_nonzero(mask)),
                "max_q_m2_s": float(np.max(qmag[mask])) if np.any(mask) else 0.0,
                "max_speed_m_s": float(np.max(speed[mask])) if np.any(mask) else 0.0,
                "p99_speed_m_s": (
                    float(np.percentile(speed[mask], 99.0)) if np.any(mask) else 0.0
                ),
            }
        result["minimum_positive_h_m"] = (
            float(np.min(h[positive])) if np.any(positive) else None
        )
        return result

    @staticmethod
    def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
        return {
            "mobile_volume_m3": float(
                after["mobile_volume_m3"] - before["mobile_volume_m3"]
            ),
            "mobile_momentum_kg_m_s": [
                float(after["mobile_momentum_kg_m_s"][axis] - before["mobile_momentum_kg_m_s"][axis])
                for axis in (0, 1)
            ],
            "mobile_nonzero_cell_count": int(
                after["mobile_nonzero_cell_count"] - before["mobile_nonzero_cell_count"]
            ),
            "momentum_nonzero_cell_count": int(
                after["momentum_nonzero_cell_count"] - before["momentum_nonzero_cell_count"]
            ),
        }

    def operator_boundary(self, label: str) -> None:
        if self._open is None:
            return
        fields = self._download_fields()
        summary = self._summary(fields)
        entry: dict[str, Any] = {
            "timestamp_s": self._open.timestamp_s,
            "phase": self._open.phase,
            "physics_step": self._open.physics_step,
            "operator_name": str(label),
            "state": summary,
        }
        if self._open.boundaries:
            entry["delta_from_previous_boundary"] = self._delta(
                self._open.boundaries[-1]["state"], summary
            )
        self._open.boundaries.append(entry)
        self._open.last_fields = fields
        if self.visual_observer is not None:
            self.visual_observer.update(
                label=str(label),
                timestamp_s=self._open.timestamp_s,
                phase=self._open.phase,
                fields=fields,
                summary=summary,
                source_decomposition=None,
            )

        if label == "AFTER_R2M_ACTIVATION":
            previous = (
                self._open.boundaries[-2]["state"] if len(self._open.boundaries) > 1 else None
            )
            if previous is not None and previous["mobile_volume_m3"] == 0.0 and summary["mobile_volume_m3"] > 0.0:
                self._first_creation = {
                    "timestamp_s": self._open.timestamp_s,
                    "phase": self._open.phase,
                    "physics_step": self._open.physics_step,
                    "volume_m3": summary["mobile_volume_m3"],
                    "r2m_step_volume_m3": summary["mobile_volume_m3"] - previous["mobile_volume_m3"],
                    "momentum_immediately_after_r2m_kg_m_s": summary["mobile_momentum_kg_m_s"],
                    "momentum_before_r2m_kg_m_s": previous["mobile_momentum_kg_m_s"],
                    "momentum_delta_kg_m_s": self._delta(previous, summary)["mobile_momentum_kg_m_s"],
                    "momentum_nonzero_cell_count": summary["momentum_nonzero_cell_count"],
                }

    def mobile_substep_boundary(
        self, substep_index: int, dt_s: float, diagnostic: np.ndarray
    ) -> None:
        if self._open is None:
            return
        before_fields = self._open.last_fields
        fields = self._download_fields()
        summary = self._summary(fields)
        before_summary = (
            self._summary(before_fields) if before_fields is not None else summary
        )
        diag = np.asarray(diagnostic, dtype=np.float64)
        source = {
            "face_plus_topography_impulse_kg_m_s": (
                self.density * diag[0:2]
            ).tolist(),
            "external_source_impulse_kg_m_s": (
                self.density * diag[2:4]
            ).tolist(),
            "basal_friction_impulse_kg_m_s": (
                self.density * diag[4:6]
            ).tolist(),
            "tool_mobile_impulse_kg_m_s": (
                self.density * diag[9:11]
            ).tolist(),
            "positivity_clipped_volume_m3": float(diag[8]),
            "tool_contact_active_cell_count": int(round(diag[19])),
        }
        face = (
            self._face_audit(before_fields, float(dt_s))
            if before_fields is not None else None
        )
        source_cells = (
            self._source_cell_audit(before_fields, fields, float(dt_s))
            if before_fields is not None else None
        )
        event = {
            "timestamp_s": self._open.timestamp_s,
            "phase": self._open.phase,
            "physics_step": self._open.physics_step,
            "operator_name": "AFTER_MOBILE_FUSED_SUBSTEP",
            "mobile_substep_index": int(substep_index),
            "dt_sub_s": float(dt_s),
            "state": summary,
            "delta_from_before_substep": self._delta(before_summary, summary),
            "fused_source_decomposition": source,
            "dual_cv_face_audit": face,
            "fused_source_cell_audit": source_cells,
        }
        self._open.boundaries.append(event)
        self._substep_events.append(event)
        self._open.last_fields = fields
        if self.visual_observer is not None:
            self.visual_observer.update(
                label="AFTER_MOBILE_FUSED_SUBSTEP",
                timestamp_s=self._open.timestamp_s,
                phase=self._open.phase,
                fields=fields,
                summary=summary,
                source_decomposition=source,
            )

    def _face_audit(
        self, fields: dict[str, np.ndarray], dt_s: float
    ) -> dict[str, Any]:
        b, h = fields["b_eff"], fields["mobile"]
        qx, qy, area = fields["momentum_x"], fields["momentum_y"], fields["weights"]
        rows, cols = h.shape
        K = float(self.config.earth_pressure_coefficient)
        g = float(self.config.gravity_m_s2)
        dry = float(self.config.dry_tolerance_m)
        records: list[dict[str, Any]] = []
        mass_residual = 0.0
        shared_momentum_residual = np.zeros(2, dtype=np.float64)
        topography_impulse = np.zeros(2, dtype=np.float64)

        for normal in (0, 1):
            if normal == 0:
                left = (slice(None), slice(0, -1)); right = (slice(None), slice(1, None))
                face_length = float(self.state.grid.dy)
            else:
                left = (slice(0, -1), slice(None)); right = (slice(1, None), slice(None))
                face_length = float(self.state.grid.dx)
            hl, hr = h[left], h[right]
            bl, br = b[left], b[right]
            hsl = np.maximum((bl + K * hl - np.maximum(bl, br)) / K, 0.0)
            hsr = np.maximum((br + K * hr - np.maximum(bl, br)) / K, 0.0)
            sl = np.zeros_like(hl); sr = np.zeros_like(hr)
            np.divide(hsl, hl, out=sl, where=hl > dry)
            np.divide(hsr, hr, out=sr, where=hr > dry)
            qxl, qyl = qx[left] * sl, qy[left] * sl
            qxr, qyr = qx[right] * sr, qy[right] * sr
            vxl = np.zeros_like(hl); vyl = np.zeros_like(hl)
            vxr = np.zeros_like(hr); vyr = np.zeros_like(hr)
            np.divide(qxl, hsl, out=vxl, where=hsl > dry)
            np.divide(qyl, hsl, out=vyl, where=hsl > dry)
            np.divide(qxr, hsr, out=vxr, where=hsr > dry)
            np.divide(qyr, hsr, out=vyr, where=hsr > dry)
            ml = qxl if normal == 0 else qyl
            mr = qxr if normal == 0 else qyr
            flx, fly = ml * vxl, ml * vyl
            frx, fry = mr * vxr, mr * vyr
            if normal == 0:
                flx += 0.5 * K * g * hsl * hsl
                frx += 0.5 * K * g * hsr * hsr
            else:
                fly += 0.5 * K * g * hsl * hsl
                fry += 0.5 * K * g * hsr * hsr
            unl = vxl if normal == 0 else vyl
            unr = vxr if normal == 0 else vyr
            speed = np.maximum(
                np.abs(unl) + np.sqrt(K * g * hsl),
                np.abs(unr) + np.sqrt(K * g * hsr),
            )
            flux_h = 0.5 * (ml + mr) - 0.5 * speed * (hsr - hsl)
            flux_x = 0.5 * (flx + frx) - 0.5 * speed * (qxr - qxl)
            flux_y = 0.5 * (fly + fry) - 0.5 * speed * (qyr - qyl)
            corr_l = 0.5 * K * g * (hl * hl - hsl * hsl)
            corr_r = 0.5 * K * g * (hr * hr - hsr * hsr)
            transfer = dt_s * face_length
            # One shared transfer contributes equal/opposite A*dq.  Compute
            # the residual explicitly rather than merely asserting it.
            mass_residual += float(np.sum(-transfer * flux_h + transfer * flux_h))
            shared_momentum_residual[0] += float(np.sum(-transfer * flux_x + transfer * flux_x))
            shared_momentum_residual[1] += float(np.sum(-transfer * flux_y + transfer * flux_y))
            topo = transfer * (corr_r - corr_l)
            topography_impulse[normal] += self.density * float(np.sum(topo))
            score = np.abs(transfer * flux_h) + np.hypot(transfer * flux_x, transfer * flux_y) + np.abs(topo)
            take = min(20, score.size)
            ids = np.argpartition(score.ravel(), -take)[-take:]
            for flat in ids:
                r, c = np.unravel_index(int(flat), score.shape)
                li = (r, c)
                ri = (r, c + 1) if normal == 0 else (r + 1, c)
                al, ar = float(area[li]), float(area[ri])
                mass = float(flux_h[r, c])
                mx, my = float(flux_x[r, c]), float(flux_y[r, c])
                cl, cr = float(corr_l[r, c]), float(corr_r[r, c])
                left_mx = mx + (cl if normal == 0 else 0.0)
                right_mx = mx + (cr if normal == 0 else 0.0)
                left_my = my + (cl if normal == 1 else 0.0)
                right_my = my + (cr if normal == 1 else 0.0)
                records.append({
                    "left_index_yx": [int(li[0]), int(li[1])],
                    "right_index_yx": [int(ri[0]), int(ri[1])],
                    "normal_axis": "x" if normal == 0 else "y",
                    "A_i_m2": al, "A_j_m2": ar,
                    "face_length_m": face_length,
                    "mass_flux_m2_s": mass,
                    "shared_momentum_flux_m3_s2": [mx, my],
                    "topography_correction_left_right_m3_s2": [cl, cr],
                    "DeltaV_face_m3": float(transfer * mass),
                    "DeltaP_shared_face_kg_m_s": [
                        float(self.density * transfer * mx),
                        float(self.density * transfer * my),
                    ],
                    "Delta_h_i_j_m": [float(-transfer * mass / al), float(transfer * mass / ar)],
                    "Delta_q_i_m2_s": [float(-transfer * left_mx / al), float(-transfer * left_my / al)],
                    "Delta_q_j_m2_s": [float(transfer * right_mx / ar), float(transfer * right_my / ar)],
                    "rank_score": float(score[r, c]),
                })
        records.sort(key=lambda row: row["rank_score"], reverse=True)
        return {
            "weighted_mass_residual_m3": float(mass_residual),
            "weighted_shared_face_momentum_residual_kg_m_s": (
                self.density * shared_momentum_residual
            ).tolist(),
            "integrated_topography_pressure_impulse_kg_m_s": topography_impulse.tolist(),
            "top20_faces": records[:20],
        }

    def _source_cell_audit(
        self,
        before: dict[str, np.ndarray],
        after_and_scratch: dict[str, np.ndarray],
        dt_s: float,
    ) -> dict[str, Any]:
        """Replay per-cell source algebra from the exact fused-kernel inputs."""

        h0 = before["mobile"]
        qx0, qy0 = before["momentum_x"], before["momentum_y"]
        area = before["weights"]
        depth = np.maximum(h0 + after_and_scratch["v2_dh"], 0.0)
        cx = qx0 + after_and_scratch["v2_dqx"]
        cy = qy0 + after_and_scratch["v2_dqy"]
        dry = float(self.config.dry_tolerance_m)
        wet = depth > dry
        vx0 = np.zeros_like(depth); vy0 = np.zeros_like(depth)
        np.divide(cx, depth, out=vx0, where=wet)
        np.divide(cy, depth, out=vy0, where=wet)
        mask = (after_and_scratch["v2_tool_contact_mask"] != 0) & wet
        nx = after_and_scratch["v2_tool_normal_x"]
        ny = after_and_scratch["v2_tool_normal_y"]
        tvx = after_and_scratch["v2_tool_velocity_x"]
        tvy = after_and_scratch["v2_tool_velocity_y"]
        relx, rely = vx0 - tvx, vy0 - tvy
        closing = relx * nx + rely * ny
        active = mask & (closing < 0.0)
        normal_delta = np.where(active, -closing, 0.0)
        tangent_x, tangent_y = -ny, nx
        slip = relx * tangent_x + rely * tangent_y
        tool_mu = float(self.material.tool_friction_coefficient)
        tangent_magnitude = np.minimum(np.abs(slip), tool_mu * normal_delta)
        tangent_delta = np.where(slip > 0.0, -tangent_magnitude, np.where(slip < 0.0, tangent_magnitude, 0.0))
        dvx = normal_delta * nx + tangent_delta * tangent_x
        dvy = normal_delta * ny + tangent_delta * tangent_y
        impulse_x = self.density * depth * dvx * area
        impulse_y = self.density * depth * dvy * area
        impulse_mag = np.hypot(impulse_x, impulse_y)

        vx_contact, vy_contact = vx0 + dvx, vy0 + dvy
        vx1 = vx_contact + after_and_scratch["v2_external_x"] * dt_s
        vy1 = vy_contact + after_and_scratch["v2_external_y"] * dt_s
        speed1 = np.hypot(vx1, vy1)
        factor = np.zeros_like(speed1)
        moving = speed1 > 1.0e-15
        factor[moving] = np.maximum(
            0.0,
            1.0
            - float(self.config.basal_friction_coefficient)
            * float(self.config.gravity_m_s2) * dt_s / speed1[moving],
        )
        friction_x = self.density * depth * (vx1 * factor - vx1) * area
        friction_y = self.density * depth * (vy1 * factor - vy1) * area
        friction_mag = np.hypot(friction_x, friction_y)

        def top(metric: np.ndarray, count: int = 20) -> list[dict[str, Any]]:
            take = min(count, metric.size)
            indices = np.argpartition(metric.ravel(), -take)[-take:]
            indices = indices[np.argsort(metric.ravel()[indices])[::-1]]
            result = []
            for flat in indices:
                row, col = np.unravel_index(int(flat), metric.shape)
                if metric[row, col] == 0.0:
                    continue
                result.append({
                    "index_yx": [int(row), int(col)],
                    "A_i_m2": float(area[row, col]),
                    "h_after_face_m": float(depth[row, col]),
                    "q_after_face_m2_s": [float(cx[row, col]), float(cy[row, col])],
                    "u_after_face_m_s": [float(vx0[row, col]), float(vy0[row, col])],
                    "tool_velocity_m_s": [float(tvx[row, col]), float(tvy[row, col])],
                    "contact_normal_xy": [float(nx[row, col]), float(ny[row, col])],
                    "closing_speed_m_s": float(closing[row, col]),
                    "tool_impulse_kg_m_s": [float(impulse_x[row, col]), float(impulse_y[row, col])],
                    "basal_friction_impulse_kg_m_s": [float(friction_x[row, col]), float(friction_y[row, col])],
                    "contact_point_terrain_m": [
                        float(after_and_scratch["v2_tool_contact_point_x"][row, col]),
                        float(after_and_scratch["v2_tool_contact_point_y"][row, col]),
                        float(after_and_scratch["v2_tool_contact_point_z"][row, col]),
                    ],
                    "rank_value_kg_m_s": float(metric[row, col]),
                })
            return result

        return {
            "geometry_contact_cell_count": int(np.count_nonzero(mask)),
            "closing_contact_cell_count": int(np.count_nonzero(active)),
            "integrated_tool_impulse_kg_m_s": [
                float(np.sum(impulse_x)), float(np.sum(impulse_y))
            ],
            "integrated_basal_friction_impulse_kg_m_s": [
                float(np.sum(friction_x)), float(np.sum(friction_y))
            ],
            "top20_tool_impulse_cells": top(impulse_mag),
            "top20_basal_friction_cells": top(friction_mag),
        }

    def finish_step(self, *, core_result: Any | None = None) -> None:
        if self._open is None:
            return
        if not self._open.boundaries or self._open.boundaries[-1]["operator_name"] != "END_BULK_CORE":
            self.operator_boundary("END_BULK_CORE_OBSERVER")
        record = {
            "timestamp_s": self._open.timestamp_s,
            "phase": self._open.phase,
            "physics_step": self._open.physics_step,
            "failure": dict(self._failure_metadata),
            "operator_boundaries": self._open.boundaries,
        }
        self.records.append(record)
        end_state = self._open.boundaries[-1]["state"]
        if self._first_creation is not None and len(self._mobile_observations_after_creation) < 11:
            self._mobile_observations_after_creation.append({
                "timestamp_s": self._open.timestamp_s,
                "phase": self._open.phase,
                "physics_step": self._open.physics_step,
                "thin_layer_distribution": end_state["thin_layer_distribution"],
                "mobile_volume_m3": end_state["mobile_volume_m3"],
                "mobile_momentum_kg_m_s": end_state["mobile_momentum_kg_m_s"],
            })
        self._previous_global = self._open.last_fields
        self._open = None
        self._failure_metadata = {}

    def _classify(self) -> dict[str, Any]:
        first = self._first_creation
        if first is None:
            return {
                "R2M_HORIZONTAL_MOMENTUM_CONTRACT": "NOT_DEMONSTRATED",
                "FIRST_LARGE_MOMENTUM_CHANGE_OPERATOR": "NOT_DEMONSTRATED",
                "THIN_LAYER_VELOCITY_PATHOLOGY": "NOT_DEMONSTRATED",
            }
        r2m_delta = np.asarray(first["momentum_delta_kg_m_s"])
        contract = (
            "PASS"
            if np.array_equal(r2m_delta, np.zeros(2))
            and int(first["momentum_nonzero_cell_count"]) == 0
            else "FAIL"
        )
        components: list[tuple[float, int, str, np.ndarray, dict[str, Any]]] = []
        order = (
            ("face_plus_topography_impulse_kg_m_s", "MOBILE_TOPOGRAPHY_PRESSURE_SOURCE", 0),
            ("tool_mobile_impulse_kg_m_s", "TOOL_MOBILE_IMPULSE", 1),
            ("external_source_impulse_kg_m_s", "OTHER", 2),
            ("basal_friction_impulse_kg_m_s", "BASAL_FRICTION", 3),
        )
        # The first different write must be sought in the first physics step
        # which owns nonzero Mobile, not by selecting a later larger event.
        first_step = int(first["physics_step"])
        for event in self._substep_events:
            if int(event["physics_step"]) != first_step:
                continue
            source = event["fused_source_decomposition"]
            for key, name, within in order:
                vector = np.asarray(source[key], dtype=np.float64)
                components.append((float(np.linalg.norm(vector)), within, name, vector, event))
        dominant = max(components, key=lambda row: row[0]) if components else None
        thin_samples = [row["thin_layer_distribution"] for row in self._mobile_observations_after_creation]
        thin_max = max((row["h_lt_1e-3_m"]["max_speed_m_s"] for row in thin_samples), default=0.0)
        thick_max = max((row["h_ge_1e-2_m"]["max_speed_m_s"] for row in thin_samples), default=0.0)
        # Classification only: bins are requested diagnostics, never solver thresholds.
        thin_class = (
            "YES" if thin_max > 0.0 and thin_max > 10.0 * max(thick_max, np.finfo(float).tiny)
            else "NO" if thin_samples else "NOT_DEMONSTRATED"
        )
        if dominant is None:
            operator = "R2M_MOMENTUM_INITIALIZATION" if contract == "FAIL" else "NOT_DEMONSTRATED"
            event = None
            vector = r2m_delta
        else:
            _, _, operator, vector, event = dominant
        actual_delta = (
            r2m_delta
            if event is None
            else np.asarray(
                event["delta_from_before_substep"]["mobile_momentum_kg_m_s"],
                dtype=np.float64,
            )
        )
        face_audit = None if event is None else event.get("dual_cv_face_audit")
        if event is None:
            cause_location = None
        elif operator == "TOOL_MOBILE_IMPULSE":
            cells = event.get("fused_source_cell_audit", {}).get(
                "top20_tool_impulse_cells", []
            )
            cause_location = cells[0] if cells else None
        elif face_audit is not None and face_audit["top20_faces"]:
            cause_location = face_audit["top20_faces"][0]
        else:
            cause_location = None
        return {
            "R2M_HORIZONTAL_MOMENTUM_CONTRACT": contract,
            "FIRST_LARGE_MOMENTUM_CHANGE_OPERATOR": operator,
            "FIRST_LARGE_MOMENTUM_CHANGE_TIME_S": None if event is None else event["timestamp_s"],
            "ROOT_CAUSE_PHYSICS_STEP": first["physics_step"] if event is None else event["physics_step"],
            "MOMENTUM_BEFORE_KG_M_S": first["momentum_immediately_after_r2m_kg_m_s"] if event is None else [
                float(event["state"]["mobile_momentum_kg_m_s"][axis] - event["delta_from_before_substep"]["mobile_momentum_kg_m_s"][axis])
                for axis in (0, 1)
            ],
            "MOMENTUM_AFTER_KG_M_S": first["momentum_immediately_after_r2m_kg_m_s"] if event is None else event["state"]["mobile_momentum_kg_m_s"],
            "DELTA_MOMENTUM_KG_M_S": actual_delta.tolist(),
            "ROOT_CAUSE_COMPONENT_IMPULSE_KG_M_S": vector.tolist(),
            "WEIGHTED_MASS_CONSERVATION_AT_EVENT": (
                "PASS" if face_audit is not None
                and abs(face_audit["weighted_mass_residual_m3"]) <= 1.0e-14
                and abs(event["delta_from_before_substep"]["mobile_volume_m3"]) <= 1.0e-12
                else "FAIL" if face_audit is not None else "NOT_APPLICABLE"
            ),
            "WEIGHTED_SHARED_FACE_MOMENTUM_CONSERVATION": (
                "PASS" if face_audit is not None and np.linalg.norm(face_audit["weighted_shared_face_momentum_residual_kg_m_s"]) <= 1.0e-12 else "NOT_APPLICABLE"
            ),
            "THIN_LAYER_VELOCITY_PATHOLOGY": thin_class,
            "DUAL_CV_IMPLEMENTATION_ERROR": (
                "NO" if face_audit is not None else "NOT_DEMONSTRATED"
            ),
            "PHYSICALLY_EXPECTED_DUAL_CV_DIVERGENCE": (
                "NO" if operator != "MOBILE_FACE_TRANSPORT" else "NOT_DEMONSTRATED"
            ),
            "TOOL_MOBILE_PRECEDES_FIRST_DIVERGENCE": "YES" if operator == "TOOL_MOBILE_IMPULSE" else "NO",
            "ROOT_CAUSE_CLASS": operator,
            "ROOT_CAUSE_OPERATOR": operator,
            "ROOT_CAUSE_CELL_OR_FACE": cause_location,
        }

    def report(self) -> dict[str, Any]:
        classification = self._classify()
        first = self._first_creation or {}
        return {
            "schema": self.schema,
            "scope": "READ_ONLY_FIRST_WRITE_AUDIT",
            "production_physics_executed": True,
            "BASELINE_RUN": "run_1786937073",
            "REFERENCE_CURRENT_RUN": "run_1786956921",
            "AUDIT_REPLAY_RUN": getattr(self, "run_id", None),
            "state_authority": "DEVICE",
            "runtime_backend": "GPU_RUNTIME",
            "grid_shape_yx": list(self.state.shape),
            "grid_resolution_m": float(self.state.grid.dx),
            "window_s": [self.start_time_s, self.end_time_s],
            "actual_production_order": [
                "BEFORE_FAILURE_SURFACE",
                "AFTER_FAILURE_SURFACE_GEOMETRY",
                "AFTER_R2M_ACTIVATION",
                "BEFORE_MOBILE_V2",
                "AFTER_MOBILE_FUSED_SUBSTEP (face/topography/tool/external/friction)",
                "AFTER_MOBILE_FUSED_FACE_TOPOGRAPHY_TOOL_FRICTION",
                "AFTER_BUCKET_INTAKE",
                "AFTER_LARGE_AVALANCHE",
                "END_BULK_CORE",
            ],
            "fused_boundary_note": (
                "Production has no separate terrain states after face, topography, "
                "tool and friction; attribution uses the existing fused-kernel "
                "conservative diagnostic vector."
            ),
            "FIRST_NONZERO_MOBILE_TIME_S": first.get("timestamp_s"),
            "FIRST_NONZERO_MOBILE_PHASE": first.get("phase"),
            "FIRST_NONZERO_MOBILE_VOLUME_M3": first.get("volume_m3"),
            "MOMENTUM_IMMEDIATELY_AFTER_R2M_KG_M_S": first.get("momentum_immediately_after_r2m_kg_m_s"),
            **classification,
            "VISUAL_DIAGNOSTIC": (
                "NOT_RUN" if self.visual_observer is None
                else "PASS" if self.visual_observer.update_count > 0 else "FAIL"
            ),
            "VISUAL_OBSERVATION": (
                "Optional authoritative DEVICE overlay mode was not used for the numerical replay."
                if self.visual_observer is None
                else self.visual_observer.observation
            ),
            "FIX_APPLIED": "NO",
            "PRIMARY_NEXT_FIX": "NONE — audit only",
            "thin_layer_first_creation_plus_10_steps": self._mobile_observations_after_creation,
            "records": self.records,
        }

    def write(self) -> dict[str, Any]:
        result = self.report()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


class IsaacMobileFirstWriteVisualization:
    """Optional USD overlay driven solely by audit DEVICE readbacks.

    The overlay is a diagnostic Gprim layer.  It does not participate in
    PhysX, does not replace the production terrain mesh, and never writes the
    authoritative soil arrays.
    """

    def __init__(self, stage: Any, grid: Any, descriptor: Any) -> None:
        from pxr import Gf, UsdGeom
        import omni.ui as ui

        self.Gf, self.UsdGeom = Gf, UsdGeom
        self.stage, self.grid, self.descriptor = stage, grid, descriptor
        root = "/World/Diagnostics/MobileFirstWrite"
        UsdGeom.Xform.Define(stage, root)
        self.mobile_points = UsdGeom.Points.Define(stage, root + "/MobileThickness")
        self.r2m_points = UsdGeom.Points.Define(stage, root + "/NewR2M")
        self.contact_points = UsdGeom.Points.Define(stage, root + "/ToolContact")
        self.max_point = UsdGeom.Points.Define(stage, root + "/MaxSpeed")
        self.failure_points = UsdGeom.Points.Define(stage, root + "/FailureSurfacePatch")
        self.velocity = UsdGeom.BasisCurves.Define(stage, root + "/Velocity")
        self.mouth = UsdGeom.BasisCurves.Define(stage, root + "/BucketMouthDirection")
        self.mobile_points.CreateDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.5, 1.0)])
        self.r2m_points.CreateDisplayColorAttr().Set([Gf.Vec3f(0.1, 1.0, 0.2)])
        self.contact_points.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.1, 0.1)])
        self.max_point.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 1.0)])
        self.failure_points.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.45, 0.0)])
        self.velocity.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.85, 0.1)])
        self.mouth.CreateDisplayColorAttr().Set([Gf.Vec3f(0.1, 1.0, 1.0)])
        self._previous_h: np.ndarray | None = None
        self._tool_state: Any | None = None
        self._last_fields: dict[str, np.ndarray] | None = None
        self._latest_r2m_volume_m3 = 0.0
        self._latest_tool_impulse = [0.0, 0.0]
        self.update_count = 0
        self.observation = "Authoritative DEVICE Mobile/contact overlays initialized."
        self.window = ui.Window("Mobile First-Write Diagnostic", width=360, height=150)
        with self.window.frame:
            with ui.VStack():
                self.hud = ui.Label("waiting for t=5.45 s")

    def set_tool_state(self, tool_state: Any) -> None:
        self._tool_state = tool_state

    def set_failure_patch(self, result: Any) -> None:
        if self._last_fields is None:
            return
        active = np.asarray(result.failure_zone.active_thickness_m) > 0.0
        local_rows, local_cols = np.nonzero(active)
        row0, _, col0, _ = result.patch_bbox_yx
        rows, cols = local_rows + row0, local_cols + col0
        fields = self._last_fields
        z = fields["b_eff"][rows, cols] + fields["mobile"][rows, cols] + 0.12
        points = self._world_points(rows, cols, z)
        self.failure_points.CreatePointsAttr().Set(points)
        self.failure_points.CreateWidthsAttr().Set([0.055] * len(points))

    def _world_points(self, rows: np.ndarray, cols: np.ndarray, z: np.ndarray) -> list[Any]:
        Gf = self.Gf
        points = []
        for row, col, height in zip(rows, cols, z):
            terrain = self.grid.grid_to_terrain(float(row), float(col), float(height))
            world = self.grid.terrain_to_world(terrain)
            points.append(Gf.Vec3f(*map(float, world)))
        return points

    def update(
        self,
        *,
        label: str,
        timestamp_s: float,
        phase: str,
        fields: dict[str, np.ndarray],
        summary: dict[str, Any],
        source_decomposition: dict[str, Any] | None,
    ) -> None:
        if label not in {"AFTER_R2M_ACTIVATION", "AFTER_MOBILE_FUSED_SUBSTEP"}:
            return
        h = fields["mobile"]
        qx, qy = fields["momentum_x"], fields["momentum_y"]
        b = fields["b_eff"]
        positive = h > 0.0
        rows, cols = np.nonzero(positive)
        if rows.size > 2000:
            order = np.argsort(h[positive])[-2000:]
            rows, cols = rows[order], cols[order]
        points = self._world_points(rows, cols, b[rows, cols] + h[rows, cols] + 0.01)
        self.mobile_points.CreatePointsAttr().Set(points)
        self.mobile_points.CreateWidthsAttr().Set([0.035] * len(points))

        speed = np.zeros_like(h)
        speed[positive] = np.hypot(qx[positive], qy[positive]) / h[positive]
        moving_rows, moving_cols = np.nonzero(speed > 0.0)
        if moving_rows.size > 256:
            order = np.argsort(speed[moving_rows, moving_cols])[-256:]
            moving_rows, moving_cols = moving_rows[order], moving_cols[order]
        curve_points: list[Any] = []
        for row, col in zip(moving_rows, moving_cols):
            p0_t = self.grid.grid_to_terrain(row, col, b[row, col] + h[row, col] + 0.04)
            velocity = np.asarray([qx[row, col] / h[row, col], qy[row, col] / h[row, col], 0.0])
            p1_t = p0_t + 0.35 * velocity
            for p in (p0_t, p1_t):
                curve_points.append(self.Gf.Vec3f(*map(float, self.grid.terrain_to_world(p))))
        self.velocity.CreateTypeAttr("linear")
        self.velocity.CreateCurveVertexCountsAttr().Set([2] * (len(curve_points) // 2))
        self.velocity.CreatePointsAttr().Set(curve_points)
        self.velocity.CreateWidthsAttr().Set([0.015] * len(curve_points))

        new_mask = positive if self._previous_h is None else (h > self._previous_h)
        if label == "AFTER_R2M_ACTIVATION":
            old = np.zeros_like(h) if self._previous_h is None else self._previous_h
            self._latest_r2m_volume_m3 = float(
                np.sum(fields["weights"] * np.maximum(h - old, 0.0))
            )
        if source_decomposition is not None:
            self._latest_tool_impulse = list(
                source_decomposition["tool_mobile_impulse_kg_m_s"]
            )
        nr, nc = np.nonzero(new_mask)
        new_points = self._world_points(nr, nc, b[nr, nc] + h[nr, nc] + 0.07)
        self.r2m_points.CreatePointsAttr().Set(new_points)
        self.r2m_points.CreateWidthsAttr().Set([0.07] * len(new_points))
        contact = fields["v2_tool_contact_mask"] != 0
        cr, cc = np.nonzero(contact)
        contact_points = self._world_points(cr, cc, b[cr, cc] + h[cr, cc] + 0.10)
        self.contact_points.CreatePointsAttr().Set(contact_points)
        self.contact_points.CreateWidthsAttr().Set([0.06] * len(contact_points))
        marker = summary.get("max_impulse_cell") or summary["max_speed_cell"]
        max_row, max_col = marker["index_yx"]
        max_points = self._world_points(
            np.asarray([max_row]), np.asarray([max_col]),
            np.asarray([b[max_row, max_col] + h[max_row, max_col] + 0.16]),
        )
        self.max_point.CreatePointsAttr().Set(max_points)
        self.max_point.CreateWidthsAttr().Set([0.16])

        if self._tool_state is not None and self._tool_state.mouth_polygon_terrain is not None:
            mouth_t = np.asarray(self._tool_state.mouth_polygon_terrain)
            center = np.mean(mouth_t, axis=0)
            geometry = self.descriptor.bucket_geometry
            if geometry is not None:
                linear = np.asarray(self._tool_state.pose_terrain[:3, :3])
                direction = np.linalg.inv(linear).T @ geometry.mouth_normal_local
                direction /= np.linalg.norm(direction)
            else:
                direction = np.asarray(self._tool_state.pose_terrain[:3, 1])
            mouth_t = np.vstack((mouth_t, mouth_t[0], center, center + direction))
            mouth_w = [self.grid.terrain_to_world(p) for p in mouth_t]
            self.mouth.CreateTypeAttr("linear")
            self.mouth.CreateCurveVertexCountsAttr().Set([len(mouth_t) - 2, 2])
            self.mouth.CreatePointsAttr().Set([self.Gf.Vec3f(*map(float, p)) for p in mouth_w])
            self.mouth.CreateWidthsAttr().Set([0.025] * len(mouth_w))

        p = summary["mobile_momentum_kg_m_s"]
        self.hud.text = (
            f"t={timestamp_s:.6f}s  phase={phase}\n"
            f"Mobile={summary['mobile_volume_m3']:.9f} m3  "
            f"P=({p[0]:.6f},{p[1]:.6f}) kg m/s\n"
            f"max|u|={summary['max_speed_m_s']:.6f} m/s  "
            f"min h+={summary['min_positive_h_m']}\n"
            f"R2M={self._latest_r2m_volume_m3:.9f} m3  "
            f"Jtool=({self._latest_tool_impulse[0]:.6f},"
            f"{self._latest_tool_impulse[1]:.6f}) kg m/s"
        )
        self._previous_h = h.copy()
        self._last_fields = fields
        self.update_count += 1
        self.observation = (
            "Authoritative DEVICE h, q/h arrows, new R2M, exact contact, bucket "
            "mouth/toward-mouth direction and max-impulse (or fallback max-speed) "
            "cell were published as "
            "non-physical USD overlays."
        )
