"""Diagnostic-only physical-validity audit for the P0-2D tool/Mobile closure.

The observer reads compact DEVICE regions selected by the production contact
geometry.  It never changes contact geometry, material parameters, Mobile
equations, trajectory, intake, or task gates.  The sole counterfactual is the
explicit ``OFF`` ablation configured by the runner on the production Mobile
solver; contact geometry is still computed and retained for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np


_GATHER_KERNELS: dict[int, tuple[Any, Any]] = {}


def _gather_kernels(wp: Any) -> tuple[Any, Any]:
    cached = _GATHER_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def gather_float(
        source: wp.array(dtype=wp.float64),
        indices: wp.array(dtype=wp.int32),
        output: wp.array(dtype=wp.float64),
    ):
        i = wp.tid()
        output[i] = source[indices[i]]

    @wp.kernel
    def gather_int(
        source: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        output: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        output[i] = source[indices[i]]

    result = (gather_float, gather_int)
    _GATHER_KERNELS[id(wp)] = result
    return result


@dataclass
class _Step:
    timestamp_s: float
    phase: str
    physics_step: int
    tool_state: Any
    payload_before_m3: float
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    requested_joint_target_rad: np.ndarray
    tool_pose_terrain: np.ndarray
    mouth_basis_terrain: np.ndarray
    bbox_yx: tuple[int, int, int, int] | None = None
    before_fields: dict[str, np.ndarray] | None = None
    substeps: list[dict[str, Any]] | None = None


class ToolMobilePhysicalValidityAudit:
    """Compact contact, 3-D direction and ON/OFF time-series recorder."""

    schema = "P0_2D_TOOL_MOBILE_PHYSICAL_VALIDITY_RUN/v1"
    _float_fields = (
        "b_eff", "mobile", "momentum_x", "momentum_y", "weights",
        "v2_dh", "v2_dqx", "v2_dqy", "v2_external_x", "v2_external_y",
        "v2_tool_normal_x", "v2_tool_normal_y",
        "v2_tool_normal_terrain_z",
        "v2_tool_velocity_x", "v2_tool_velocity_y", "v2_tool_velocity_z",
        "v2_tool_contact_point_x", "v2_tool_contact_point_y",
        "v2_tool_contact_point_z", "v2_tool_signed_distance",
    )
    _int_fields = ("v2_tool_contact_mask", "v2_tool_closest_face_index")

    def __init__(
        self,
        state: Any,
        material: Any,
        mobile_config: Any,
        descriptor: Any,
        output_path: Path,
        *,
        impulse_mode: str,
        observation_horizon_s: float = 4.0,
    ) -> None:
        mode = str(impulse_mode).upper()
        if mode not in {"ON", "OFF"}:
            raise ValueError("[ToolMobileValidity] impulse_mode must be ON or OFF")
        self.state = state
        self.runtime = state.runtime
        self.grid = state.grid
        self.material = material
        self.config = mobile_config
        self.descriptor = descriptor
        self.output_path = Path(output_path)
        self.impulse_mode = mode
        self.horizon_s = float(observation_horizon_s)
        self.density = float(material.assumed_bulk_density_kg_m3)
        self._step: _Step | None = None
        self.creation_time_s: float | None = None
        self.records: list[dict[str, Any]] = []
        self.selected_contacts: list[dict[str, Any]] = []
        self._region: dict[str, dict[str, Any]] = {}
        self._normal_semantic_samples = 0
        self._normal_semantic_violations = 0
        self._mouth_cap_active_samples = 0
        self._contact_geometry_samples = 0
        self.visual_observer: Any | None = None
        self.visual_frames: list[dict[str, Any]] = []
        self.run_id: str | None = None
        self._candidate_flux_cumulative_m3 = 0.0
        self._accepted_flux_cumulative_m3 = 0.0
        self._r2m_cumulative_m3 = 0.0

    @property
    def complete(self) -> bool:
        return bool(
            self.creation_time_s is not None
            and self.records
            and self.records[-1]["tau_from_first_mobile_s"] >= self.horizon_s
        )

    def begin_step(
        self,
        *,
        simulation_time_s: float,
        phase: str,
        physics_step: int,
        tool_state: Any,
        payload_before_m3: float,
        joint_position_rad: np.ndarray,
        joint_velocity_rad_s: np.ndarray,
        requested_joint_target_rad: np.ndarray,
    ) -> None:
        if self._step is not None:
            raise RuntimeError("[ToolMobileValidity] previous step remains open")
        pose = np.asarray(tool_state.pose_terrain, dtype=np.float64)
        geometry = self.descriptor.bucket_geometry
        inward = np.linalg.inv(pose[:3, :3]).T @ geometry.mouth_normal_local
        inward[2] = 0.0
        inward /= max(float(np.linalg.norm(inward)), np.finfo(float).tiny)
        lateral = pose[:3, :3] @ geometry.mouth_local_frame[:3, 0]
        lateral[2] = 0.0
        lateral /= max(float(np.linalg.norm(lateral)), np.finfo(float).tiny)
        vertical = np.asarray([0.0, 0.0, 1.0])
        self._step = _Step(
            float(simulation_time_s), str(phase), int(physics_step), tool_state,
            float(payload_before_m3),
            np.asarray(joint_position_rad, dtype=np.float64).reshape(-1).copy(),
            np.asarray(joint_velocity_rad_s, dtype=np.float64).reshape(-1).copy(),
            np.asarray(requested_joint_target_rad, dtype=np.float64).reshape(-1).copy(),
            pose.copy(), np.stack((inward, lateral, vertical), axis=0), substeps=[],
        )
        if self.visual_observer is not None:
            self.visual_observer.set_tool_state(tool_state)

    def _indices(self, bbox: tuple[int, int, int, int]) -> np.ndarray:
        row0, row1, col0, col1 = bbox
        rows, cols = np.indices((row1 - row0, col1 - col0), dtype=np.int32)
        return np.ascontiguousarray(
            ((rows + row0) * self.grid.nx + (cols + col0)).ravel(), dtype=np.int32
        )

    def _gather(self, bbox: tuple[int, int, int, int]) -> dict[str, np.ndarray]:
        indices = self._indices(bbox)
        count = int(indices.size)
        if count == 0:
            return {}
        wp = self.runtime.wp
        prefix = "tool_mobile_validity"
        self.runtime.upload(prefix + "_indices", indices, dtype=wp.int32)
        self.runtime.empty(prefix + "_float", count, dtype=wp.float64)
        self.runtime.empty(prefix + "_int", count, dtype=wp.int32)
        gather_float, gather_int = _gather_kernels(wp)
        result: dict[str, np.ndarray] = {"flat_indices": indices}
        shape = (bbox[1] - bbox[0], bbox[3] - bbox[2])
        for name in self._float_fields:
            self.runtime.launch(
                gather_float, dim=count,
                inputs=[self.runtime.arrays[name], self.runtime.arrays[prefix + "_indices"],
                        self.runtime.arrays[prefix + "_float"]],
            )
            self.runtime.synchronize()
            host = np.asarray(
                self.runtime.arrays[prefix + "_float"].numpy(), dtype=np.float64
            ).copy().reshape(shape)
            self.runtime.telemetry.record_d2h(host)
            result[name] = host
        for name in self._int_fields:
            self.runtime.launch(
                gather_int, dim=count,
                inputs=[self.runtime.arrays[name], self.runtime.arrays[prefix + "_indices"],
                        self.runtime.arrays[prefix + "_int"]],
            )
            self.runtime.synchronize()
            host = np.asarray(
                self.runtime.arrays[prefix + "_int"].numpy(), dtype=np.int32
            ).copy().reshape(shape)
            self.runtime.telemetry.record_d2h(host)
            result[name] = host
        return result

    def observe_contact_geometry(self, support: Any) -> None:
        if self._step is None or support is None:
            return
        bbox = tuple(int(value) for value in getattr(support, "bbox_yx", (0, 0, 0, 0)))
        if bbox[1] <= bbox[0] or bbox[3] <= bbox[2]:
            return
        self._step.bbox_yx = bbox
        self._step.before_fields = self._gather(bbox)

    def _point_local(self, point: np.ndarray) -> np.ndarray:
        assert self._step is not None
        inv = np.linalg.inv(self._step.tool_pose_terrain)
        return (inv @ np.r_[point, 1.0])[:3]

    def _region_name(self, face: int, point: np.ndarray) -> str:
        geometry = self.descriptor.bucket_geometry
        local = self._point_local(point)
        edge = np.asarray(geometry.cutting_edge_local, dtype=np.float64)
        a, b = edge[0], edge[-1]
        ab = b - a
        t = np.clip(np.dot(local - a, ab) / max(np.dot(ab, ab), 1.0e-30), 0.0, 1.0)
        if np.linalg.norm(local - (a + t * ab)) <= max(self.grid.dx, self.grid.dy):
            return "TOOTH_CUTTING_EDGE"
        if face < 32:
            return "SIDE_WALL"
        segment = (face - 32) // 2
        if segment == 16:
            return "MOUTH_CAP"
        if segment == 17:
            return "INNER_FLOOR"
        if 0 <= segment <= 15:
            return "INNER_BACK"
        return "UNKNOWN"

    def _region_entry(self, name: str) -> dict[str, Any]:
        if name not in self._region:
            self._region[name] = {
                "contact_cell_substep_count": 0,
                "active_2d_cell_substep_count": 0,
                "active_3d_cell_substep_count": 0,
                "sampled_mass_kg": 0.0,
                "j3d_normal_terrain_ns": np.zeros(3),
                "j3d_total_terrain_ns": np.zeros(3),
                "j2d_current_terrain_ns": np.zeros(3),
                "j3d_basis_mouth_lateral_vertical_ns": np.zeros(3),
                "j2d_basis_mouth_lateral_vertical_ns": np.zeros(3),
                "lost_vertical_abs_ns": 0.0,
                "unique_cells": set(),
            }
        return self._region[name]

    def mobile_substep_boundary(
        self, substep_index: int, dt_s: float, diagnostic: np.ndarray
    ) -> None:
        step = self._step
        if step is None or step.bbox_yx is None or step.before_fields is None:
            return
        before = step.before_fields
        after = self._gather(step.bbox_yx)
        if not after:
            return
        h = np.maximum(before["mobile"] + after["v2_dh"], 0.0)
        qx = before["momentum_x"] + after["v2_dqx"]
        qy = before["momentum_y"] + after["v2_dqy"]
        wet = h > float(self.config.dry_tolerance_m)
        ux = np.divide(qx, h, out=np.zeros_like(h), where=wet)
        uy = np.divide(qy, h, out=np.zeros_like(h), where=wet)
        mask = (after["v2_tool_contact_mask"] != 0) & wet
        nx, ny = after["v2_tool_normal_x"], after["v2_tool_normal_y"]
        nz = np.clip(after["v2_tool_normal_terrain_z"], -1.0, 1.0)
        horizontal = np.sqrt(np.maximum(1.0 - nz * nz, 0.0))
        n3 = np.stack((nx * horizontal, ny * horizontal, nz), axis=-1)
        tv = np.stack((after["v2_tool_velocity_x"], after["v2_tool_velocity_y"],
                       after["v2_tool_velocity_z"]), axis=-1)
        mobile3 = np.stack((ux, uy, np.zeros_like(ux)), axis=-1)
        rel3 = mobile3 - tv
        closing3 = np.sum(rel3 * n3, axis=-1)
        active3 = mask & (closing3 < 0.0)
        normal_delta3 = np.where(active3, -closing3, 0.0)
        rel_after_normal = rel3 + normal_delta3[..., None] * n3
        slip3 = rel_after_normal - np.sum(
            rel_after_normal * n3, axis=-1
        )[..., None] * n3
        slip_mag3 = np.linalg.norm(slip3, axis=-1)
        tangent_limit3 = np.minimum(
            slip_mag3, float(self.material.tool_friction_coefficient) * normal_delta3
        )
        tangent_delta3 = np.divide(
            -tangent_limit3[..., None] * slip3,
            slip_mag3[..., None], out=np.zeros_like(slip3),
            where=slip_mag3[..., None] > 0.0,
        )
        mass = self.density * h * before["weights"]
        jnormal3 = mass[..., None] * normal_delta3[..., None] * n3
        jtotal3 = mass[..., None] * (
            normal_delta3[..., None] * n3 + tangent_delta3
        )

        relx, rely = ux - tv[..., 0], uy - tv[..., 1]
        closing2 = relx * nx + rely * ny
        active2 = mask & (closing2 < 0.0)
        normal_delta2 = np.where(active2, -closing2, 0.0)
        tx, ty = -ny, nx
        slip2 = relx * tx + rely * ty
        tangent_mag2 = np.minimum(
            np.abs(slip2), float(self.material.tool_friction_coefficient) * normal_delta2
        )
        tangent_delta2 = np.where(
            slip2 > 0.0, -tangent_mag2, np.where(slip2 < 0.0, tangent_mag2, 0.0)
        )
        j2 = mass[..., None] * np.stack((
            normal_delta2 * nx + tangent_delta2 * tx,
            normal_delta2 * ny + tangent_delta2 * ty,
            np.zeros_like(nx),
        ), axis=-1)

        points = np.stack((after["v2_tool_contact_point_x"],
                           after["v2_tool_contact_point_y"],
                           after["v2_tool_contact_point_z"]), axis=-1)
        rows, cols = np.nonzero(mask)
        candidates: list[dict[str, Any]] = []
        for row, col in zip(rows, cols):
            point = points[row, col]
            face = int(after["v2_tool_closest_face_index"][row, col])
            region = self._region_name(face, point)
            entry = self._region_entry(region)
            entry["contact_cell_substep_count"] += 1
            entry["active_2d_cell_substep_count"] += int(active2[row, col])
            entry["active_3d_cell_substep_count"] += int(active3[row, col])
            entry["sampled_mass_kg"] += float(mass[row, col])
            entry["j3d_normal_terrain_ns"] += jnormal3[row, col]
            entry["j3d_total_terrain_ns"] += jtotal3[row, col]
            entry["j2d_current_terrain_ns"] += j2[row, col]
            basis = step.mouth_basis_terrain
            entry["j3d_basis_mouth_lateral_vertical_ns"] += basis @ jtotal3[row, col]
            entry["j2d_basis_mouth_lateral_vertical_ns"] += basis @ j2[row, col]
            entry["lost_vertical_abs_ns"] += abs(float(jtotal3[row, col, 2]))
            global_index = int(after["flat_indices"].reshape(mask.shape)[row, col])
            entry["unique_cells"].add(global_index)
            center = np.asarray([
                self.grid.origin_x + (step.bbox_yx[2] + col) * self.grid.dx,
                self.grid.origin_y + (step.bbox_yx[0] + row) * self.grid.dy,
                before["b_eff"][row, col] + 0.5 * before["mobile"][row, col],
            ])
            delta = center - point
            if np.linalg.norm(delta) > 1.0e-12:
                self._normal_semantic_samples += 1
                self._normal_semantic_violations += int(
                    float(np.dot(delta, n3[row, col])) < -1.0e-10
                )
            self._contact_geometry_samples += 1
            self._mouth_cap_active_samples += int(
                region == "MOUTH_CAP" and active2[row, col]
            )
            candidate = {
                "timestamp_s": step.timestamp_s,
                "substep_index": int(substep_index),
                "dt_sub_s": float(dt_s),
                "index_yx": [int(step.bbox_yx[0] + row), int(step.bbox_yx[2] + col)],
                "triangle_index": face,
                "bucket_surface_region": region,
                "contact_point_terrain_m": point.tolist(),
                "contact_point_bucket_local_m": self._point_local(point).tolist(),
                "tool_velocity_terrain_m_s": tv[row, col].tolist(),
                "precontact_mobile_velocity_terrain_m_s": mobile3[row, col].tolist(),
                "raw_unit_tool_to_material_normal_terrain": n3[row, col].tolist(),
                "current_projected_normal_xy": [float(nx[row, col]), float(ny[row, col])],
                "closing_speed_3d_m_s": float(closing3[row, col]),
                "closing_speed_2d_m_s": float(closing2[row, col]),
                "implied_3d_normal_impulse_terrain_ns": jnormal3[row, col].tolist(),
                "implied_3d_total_frictional_impulse_terrain_ns": jtotal3[row, col].tolist(),
                "current_2d_impulse_terrain_ns": j2[row, col].tolist(),
                "j3d_basis_mouth_lateral_vertical_ns": (basis @ jtotal3[row, col]).tolist(),
                "j2d_basis_mouth_lateral_vertical_ns": (basis @ j2[row, col]).tolist(),
                "lost_vertical_component_ns": float(jtotal3[row, col, 2]),
                "cell_mobile_mass_kg": float(mass[row, col]),
                "cavity_signed_distance_m": float(after["v2_tool_signed_distance"][row, col]),
            }
            candidate["rank_impulse_ns"] = float(
                max(np.linalg.norm(j2[row, col]), np.linalg.norm(jtotal3[row, col]))
            )
            candidates.append(candidate)
        candidates.sort(key=lambda item: item["rank_impulse_ns"], reverse=True)
        self.selected_contacts.extend(candidates[:20])
        assert step.substeps is not None
        diag = np.asarray(diagnostic, dtype=np.float64)
        step.substeps.append({
            "substep_index": int(substep_index),
            "dt_sub_s": float(dt_s),
            "geometry_contact_count": int(np.count_nonzero(mask)),
            "active_2d_contact_count": int(np.count_nonzero(active2)),
            "active_3d_contact_count": int(np.count_nonzero(active3)),
            "current_2d_impulse_terrain_ns_reconstructed": np.sum(j2, axis=(0, 1)).tolist(),
            "implied_3d_impulse_terrain_ns": np.sum(jtotal3, axis=(0, 1)).tolist(),
            "production_diagnostic_tool_impulse_terrain_ns": (
                self.density * np.r_[diag[9:11], 0.0]
            ).tolist(),
        })
        step.before_fields = after
        if (
            self.visual_observer is not None
            and (
                self.creation_time_s is None
                or step.timestamp_s - self.creation_time_s <= 20.0 / 60.0 + 1.0e-9
            )
        ):
            # Reuse the authoritative overlay; construct the full-grid fields it
            # expects only in this explicitly enabled visual audit.
            full = {}
            for name in ("b_eff", "mobile", "momentum_x", "momentum_y", "weights",
                         "v2_tool_contact_mask", "v2_tool_normal_x", "v2_tool_normal_y",
                         "v2_tool_velocity_x", "v2_tool_velocity_y",
                         "v2_tool_contact_point_x", "v2_tool_contact_point_y",
                         "v2_tool_contact_point_z"):
                value = np.asarray(self.runtime.arrays[name].numpy()).reshape(self.state.shape).copy()
                self.runtime.telemetry.record_d2h(value)
                full[name] = value
            max_impulse_yx = (
                candidates[0]["index_yx"] if candidates else None
            )
            summary = self._visual_summary(full, max_impulse_yx=max_impulse_yx)
            self.visual_observer.update(
                label="AFTER_MOBILE_FUSED_SUBSTEP", timestamp_s=step.timestamp_s,
                phase=step.phase, fields=full, summary=summary,
                source_decomposition={
                    "tool_mobile_impulse_kg_m_s": (
                        self.density * diag[9:11]
                    ).tolist()
                },
            )

    def _visual_summary(
        self,
        fields: dict[str, np.ndarray],
        *,
        max_impulse_yx: list[int] | None = None,
    ) -> dict[str, Any]:
        h, qx, qy = fields["mobile"], fields["momentum_x"], fields["momentum_y"]
        positive = h > 0.0
        speed = np.divide(
            np.hypot(qx, qy), h, out=np.zeros_like(h), where=positive
        )
        flat = int(np.argmax(speed))
        row, col = np.unravel_index(flat, h.shape)
        area = fields["weights"]
        return {
            "mobile_volume_m3": float(np.sum(area * h)),
            "mobile_momentum_kg_m_s": [
                float(self.density * np.sum(area * qx)),
                float(self.density * np.sum(area * qy)),
            ],
            "max_speed_m_s": float(np.max(speed)),
            "min_positive_h_m": float(np.min(h[positive])) if np.any(positive) else None,
            "max_speed_cell": {"index_yx": [int(row), int(col)]},
            "max_impulse_cell": (
                None if max_impulse_yx is None
                else {"index_yx": [int(max_impulse_yx[0]), int(max_impulse_yx[1])]}
            ),
        }

    def finish_step(
        self,
        *,
        core_result: Any,
        payload_after_m3: float,
        reservoir: dict[str, Any],
        cut_record: dict[str, Any] | None,
        base_pose_xy_yaw: np.ndarray,
    ) -> None:
        step = self._step
        if step is None:
            return
        interaction = core_result.interaction
        if self.visual_observer is not None and interaction is not None:
            self.visual_observer.set_failure_patch(interaction.failure_bridge)
        activated = 0.0 if interaction is None else float(interaction.activated_volume_m3)
        if self.creation_time_s is None and activated > 0.0:
            self.creation_time_s = step.timestamp_s
        if interaction is not None:
            self._candidate_flux_cumulative_m3 += float(
                interaction.intake_result.candidate_flux_volume_m3
            )
            self._accepted_flux_cumulative_m3 += float(
                interaction.intake_result.bucket_inflow_volume_m3
            )
            self._r2m_cumulative_m3 += activated
        if self.creation_time_s is not None:
            mobile_step = None if interaction is None else interaction.mobile_result
            impulse = (
                np.zeros(3) if mobile_step is None
                else np.asarray(mobile_step.tool_impulse_on_mobile_terrain_ns)
            )
            basis = step.mouth_basis_terrain
            self.records.append({
                "simulation_time_s": step.timestamp_s,
                "tau_from_first_mobile_s": step.timestamp_s - self.creation_time_s,
                "phase": step.phase,
                "physics_step": step.physics_step,
                "impulse_mode": self.impulse_mode,
                "payload_before_m3": step.payload_before_m3,
                "payload_after_m3": float(payload_after_m3),
                "mobile_volume_m3": float(reservoir["mobile_volume_m3"]),
                "mobile_momentum_kg_m_s": np.asarray(
                    reservoir["mobile_momentum_kg_m_s"]
                ).tolist(),
                "r2m_cumulative_from_creation_m3": self._r2m_cumulative_m3,
                "candidate_mouth_flux_cumulative_m3": self._candidate_flux_cumulative_m3,
                "accepted_mouth_flux_cumulative_m3": self._accepted_flux_cumulative_m3,
                "tool_impulse_on_mobile_terrain_ns": impulse.tolist(),
                "tool_impulse_mouth_lateral_vertical_ns": (basis @ impulse).tolist(),
                "machine_reaction_impulse_terrain_ns": (-impulse).tolist(),
                "joint_position_rad": step.joint_position_rad.tolist(),
                "joint_velocity_rad_s": step.joint_velocity_rad_s.tolist(),
                "requested_joint_target_rad": step.requested_joint_target_rad.tolist(),
                "maximum_penetration_m": (
                    0.0 if interaction is None else float(np.max(
                        interaction.failure_bridge.intersection.penetration_depth_m
                    ))
                ),
                "soil_force_terrain_n": np.asarray(core_result.applied_force_terrain_n).tolist(),
                "mass_balance_error_m3": float(core_result.mass_balance_error_m3),
                "tool_mobile_work_j": 0.0 if mobile_step is None else float(mobile_step.tool_work_j),
                "machine_reaction_work_j": 0.0 if mobile_step is None else float(mobile_step.machine_reaction_work_j),
                "contact_dissipation_j": 0.0 if mobile_step is None else float(mobile_step.tool_contact_dissipation_j),
                "production_tool_contact_substep_diagnostics": (
                    [] if mobile_step is None
                    else list(mobile_step.tool_contact_substep_diagnostics)
                ),
                "substeps": list(step.substeps or ()),
                "mouth_neighborhood": None if cut_record is None else cut_record.get("proximity"),
                "local_mobile": None if cut_record is None else cut_record.get("local_mobile"),
                "base_pose_xy_yaw": np.asarray(
                    base_pose_xy_yaw, dtype=np.float64
                ).tolist(),
                "tool_pose_terrain": step.tool_pose_terrain.tolist(),
            })
        self._step = None

    def _region_report(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in self._region.items():
            j3 = np.asarray(value["j3d_total_terrain_ns"])
            j2 = np.asarray(value["j2d_current_terrain_ns"])
            basis_integrals = np.asarray(
                value["j3d_basis_mouth_lateral_vertical_ns"]
            )
            basis_integrals_2d = np.asarray(
                value["j2d_basis_mouth_lateral_vertical_ns"]
            )
            denominator = max(float(np.sum(np.abs(basis_integrals))), 1.0e-30)
            result[name] = {
                "contact_cell_substep_count": value["contact_cell_substep_count"],
                "active_2d_cell_substep_count": value["active_2d_cell_substep_count"],
                "active_3d_cell_substep_count": value["active_3d_cell_substep_count"],
                "unique_contact_cell_count": len(value["unique_cells"]),
                "sampled_mass_kg_cell_substep_sum": value["sampled_mass_kg"],
                "integrated_3d_normal_impulse_terrain_ns": np.asarray(
                    value["j3d_normal_terrain_ns"]
                ).tolist(),
                "integrated_3d_total_impulse_terrain_ns": j3.tolist(),
                "integrated_current_2d_impulse_terrain_ns": j2.tolist(),
                "integrated_3d_basis_mouth_lateral_vertical_ns": basis_integrals.tolist(),
                "integrated_current_2d_basis_mouth_lateral_vertical_ns": (
                    basis_integrals_2d.tolist()
                ),
                "integrated_3d_direction_fraction_abs": (
                    np.abs(basis_integrals) / denominator
                ).tolist(),
                "lost_vertical_abs_impulse_ns": value["lost_vertical_abs_ns"],
            }
        return result

    def report(self) -> dict[str, Any]:
        selected = sorted(
            self.selected_contacts, key=lambda item: item["rank_impulse_ns"], reverse=True
        )[:200]
        max_mass = max((abs(item["mass_balance_error_m3"]) for item in self.records), default=0.0)
        tool = np.sum(
            [item["tool_impulse_on_mobile_terrain_ns"] for item in self.records], axis=0
        ) if self.records else np.zeros(3)
        reaction = np.sum(
            [item["machine_reaction_impulse_terrain_ns"] for item in self.records], axis=0
        ) if self.records else np.zeros(3)
        energy_residual = sum(
            item["tool_mobile_work_j"] + item["machine_reaction_work_j"]
            for item in self.records
        )
        return {
            "schema": self.schema,
            "scope": "DIAGNOSTIC_ONLY_CONTROLLED_TOOL_IMPULSE_ABLATION",
            "run_id": self.run_id,
            "impulse_mode": self.impulse_mode,
            "runtime_backend": "GPU_RUNTIME",
            "state_authority": "DEVICE",
            "grid_shape_yx": list(self.state.shape),
            "grid_resolution_m": float(self.grid.dx),
            "creation_time_s": self.creation_time_s,
            "observation_horizon_s": self.horizon_s,
            "payload_final_m3": self.records[-1]["payload_after_m3"] if self.records else None,
            "candidate_mouth_flux_m3": self._candidate_flux_cumulative_m3,
            "accepted_mouth_flux_m3": self._accepted_flux_cumulative_m3,
            "r2m_cumulative_m3": self._r2m_cumulative_m3,
            "maximum_mass_error_m3": max_mass,
            "action_reaction_residual_ns": (tool + reaction).tolist(),
            "contact_energy_action_reaction_residual_j": float(energy_residual),
            "normal_semantic_sample_count": self._normal_semantic_samples,
            "normal_semantic_violation_count": self._normal_semantic_violations,
            "mouth_cap_active_cell_substep_count": self._mouth_cap_active_samples,
            "contact_geometry_sample_count": self._contact_geometry_samples,
            "VISUAL_DIAGNOSTIC": (
                "NOT_RUN" if self.visual_observer is None
                else "PASS" if self.visual_frames else "FAIL"
            ),
            "visual_frames": list(self.visual_frames),
            "per_bucket_surface_directional_statistics": self._region_report(),
            "selected_representative_contacts": selected,
            "timeseries": self.records,
            "FIX_APPLIED": "MOUTH_CAP_CONTAINMENT_ONLY_NO_PHYSICAL_CONTACT",
        }

    def write(self) -> dict[str, Any]:
        result = self.report()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
