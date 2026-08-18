"""Read-only causal telemetry for the production CUT_AND_FILL intake path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CutFillPayloadCausalAudit:
    """Accumulate control-surface, proximity, trajectory and force evidence.

    The observer never mutates terrain, payload, trajectory, or parameters.
    Compact DEVICE region reads are explicit acceptance telemetry and are not
    part of the production physics path.
    """

    state: Any
    grid: Any
    descriptor: Any
    phase_target_rad: np.ndarray
    dt_s: float
    records: list[dict[str, Any]] = field(default_factory=list)
    entry_joint_rad: np.ndarray | None = None
    entry_payload_m3: float | None = None
    r2m_start_m3: float | None = None
    m2r_start_m3: float | None = None
    entry_mobile_m3: float | None = None
    failure_zone_r2m_gross_m3: float = 0.0
    first_simulation_time_s: float | None = None

    def _proximity(self, tool_state: Any) -> tuple[dict[str, float], dict[str, Any]]:
        geometry = self.descriptor.bucket_geometry
        pose = np.asarray(tool_state.pose_terrain, dtype=np.float64)
        mouth = geometry.transform_points(pose, geometry.mouth_polygon_local)
        centroid = np.mean(mouth, axis=0)
        radius = 2.05
        col0 = max(0, int(np.floor((centroid[0] - radius - self.grid.origin_x) / self.grid.dx)))
        col1 = min(self.grid.nx, int(np.ceil((centroid[0] + radius - self.grid.origin_x) / self.grid.dx)) + 1)
        row0 = max(0, int(np.floor((centroid[1] - radius - self.grid.origin_y) / self.grid.dy)))
        row1 = min(self.grid.ny, int(np.ceil((centroid[1] + radius - self.grid.origin_y) / self.grid.dy)) + 1)
        if row1 <= row0 or col1 <= col0:
            volumes = {key: 0.0 for key in (
                "radius_0p25_m3", "radius_0p5_m3", "radius_1p0_m3", "radius_2p0_m3",
                "front_m3", "mouth_prism_m3", "bucket_interior_m3",
                "left_bypass_m3", "right_bypass_m3", "behind_m3",
            )}
            return volumes, {
                "mouth_prism_mean_mobile_thickness_m": 0.0,
                "mouth_prism_max_mobile_thickness_m": 0.0,
                "mouth_prism_volume_weighted_mobile_velocity_m_s": [0.0, 0.0],
            }
        patch = self.state.download_region((row0, row1, col0, col1), source="cut_fill_payload_causal_audit")
        h = np.asarray(patch.H_mobile_m)
        rows, cols = np.indices(h.shape)
        x = self.grid.origin_x + (cols + col0) * self.grid.dx
        y = self.grid.origin_y + (rows + row0) * self.grid.dy
        z = np.asarray(patch.b_eff_m) + 0.5 * h
        points = np.stack((x, y, z), axis=-1)
        inv = np.linalg.inv(pose)
        local = (inv @ np.c_[points.reshape(-1, 3), np.ones(points.size // 3)].T).T[:, :3].reshape(points.shape)
        # Exact control weights are immutable grid geometry, not mutable DEVICE
        # physics state; reconstructing them avoids a full-field D2H per step.
        weights = np.asarray(self.state._vertex_weights(self.grid))[row0:row1, col0:col1]
        volume = h * weights
        distance = np.linalg.norm(points[..., :2] - centroid[:2], axis=-1)
        rotation = pose[:3, :3]
        normal = np.linalg.inv(rotation).T @ geometry.mouth_normal_local
        normal_xy = normal[:2] / max(np.linalg.norm(normal[:2]), 1.0e-12)
        signed_front = (points[..., :2] - centroid[:2]) @ normal_xy
        interior = np.asarray(geometry.interior_vertices_local)
        low, high = np.min(interior, axis=0), np.max(interior, axis=0)
        in_interior = np.all((local >= low - 1.0e-9) & (local <= high + 1.0e-9), axis=-1)
        mouth_local = np.asarray(geometry.mouth_polygon_local)
        mouth_low, mouth_high = np.min(mouth_local, axis=0), np.max(mouth_local, axis=0)
        mouth_prism = (
            (local[..., 0] >= mouth_low[0]) & (local[..., 0] <= mouth_high[0])
            & (local[..., 2] >= mouth_low[2]) & (local[..., 2] <= mouth_high[2])
            & (np.abs(local[..., 1] - np.mean(mouth_local[:, 1])) <= max(self.grid.dx, self.grid.dy))
        )
        near = distance <= 2.0
        volumes = {
            "radius_0p25_m3": float(np.sum(volume[distance <= 0.25])),
            "radius_0p5_m3": float(np.sum(volume[distance <= 0.5])),
            "radius_1p0_m3": float(np.sum(volume[distance <= 1.0])),
            "radius_2p0_m3": float(np.sum(volume[near])),
            "front_m3": float(np.sum(volume[near & (signed_front > 0.0)])),
            "mouth_prism_m3": float(np.sum(volume[mouth_prism])),
            "bucket_interior_m3": float(np.sum(volume[in_interior])),
            "left_bypass_m3": float(np.sum(volume[near & (local[..., 0] < low[0])])),
            "right_bypass_m3": float(np.sum(volume[near & (local[..., 0] > high[0])])),
            "behind_m3": float(np.sum(volume[near & (signed_front < 0.0)])),
        }
        momentum = np.asarray(patch.mobile_momentum_m2_s)
        velocity = np.divide(
            momentum,
            h[..., None],
            out=np.zeros_like(momentum),
            where=h[..., None] > 1.0e-12,
        )
        mouth_volume = float(np.sum(volume[mouth_prism]))
        weighted_velocity = (
            np.sum(velocity[mouth_prism] * volume[mouth_prism, None], axis=0)
            / mouth_volume
            if mouth_volume > 1.0e-12
            else np.zeros(2, dtype=np.float64)
        )
        local_mobile = {
            "mouth_prism_mean_mobile_thickness_m": (
                float(np.mean(h[mouth_prism])) if np.any(mouth_prism) else 0.0
            ),
            "mouth_prism_max_mobile_thickness_m": (
                float(np.max(h[mouth_prism])) if np.any(mouth_prism) else 0.0
            ),
            "mouth_prism_volume_weighted_mobile_velocity_m_s": weighted_velocity.tolist(),
        }
        return volumes, local_mobile

    def observe(
        self,
        *,
        simulation_time_s: float,
        tool_state: Any,
        interaction: Any,
        avalanche: Any,
        payload_before_m3: float,
        payload_after_m3: float,
        joint_position_rad: np.ndarray,
        joint_velocity_rad_s: np.ndarray,
        actuator_output: Any,
        quasi_static_force_n: np.ndarray,
        momentum_force_n: np.ndarray,
        applied_force_n: np.ndarray,
        failure_zone_r2m_step_m3: float = 0.0,
        mobile_volume_m3: float | None = None,
        machine_velocity_terrain_m_s: np.ndarray | None = None,
        force_application_point_terrain_m: np.ndarray | None = None,
        requested_joint_target_rad: np.ndarray | None = None,
        phase: str = "CUT_AND_FILL",
        trajectory_stage: str = "LEGACY_FIXED_TARGET",
        mass_balance_error_m3: float = 0.0,
    ) -> None:
        if interaction is None:
            return
        q = np.asarray(joint_position_rad, dtype=np.float64).reshape(-1)
        if self.entry_joint_rad is None:
            self.entry_joint_rad = q.copy()
            self.entry_payload_m3 = float(payload_before_m3)
            self.r2m_start_m3 = float(avalanche.cumulative_resting_to_mobile_m3)
            self.m2r_start_m3 = float(avalanche.cumulative_mobile_to_resting_m3)
            self.entry_mobile_m3 = (
                None if mobile_volume_m3 is None else float(mobile_volume_m3)
            )
            self.first_simulation_time_s = float(simulation_time_s)
        self.failure_zone_r2m_gross_m3 += float(failure_zone_r2m_step_m3)
        target_delta = np.asarray(self.phase_target_rad) - self.entry_joint_rad
        progress = float(
            np.dot(q - self.entry_joint_rad, target_delta)
            / max(float(np.dot(target_delta, target_delta)), 1.0e-12)
        )
        intake = interaction.intake_result
        geometry = self.descriptor.bucket_geometry
        pose = np.asarray(tool_state.pose_terrain)
        rotation = pose[:3, :3]
        normal = np.linalg.inv(rotation).T @ geometry.mouth_normal_local
        normal /= np.linalg.norm(normal)
        proximity, local_mobile = self._proximity(tool_state)
        edge_center = np.mean(tool_state.cutting_edge_terrain, axis=0)
        edge_velocity = np.asarray(tool_state.linear_velocity, dtype=np.float64) + np.cross(
            np.asarray(tool_state.angular_velocity, dtype=np.float64),
            edge_center - pose[:3, 3],
        )
        separation = np.asarray(
            tool_state.separation_plane_direction_terrain, dtype=np.float64
        ).copy()
        separation /= max(float(np.linalg.norm(separation)), 1.0e-12)
        plate_normal = np.linalg.inv(rotation).T @ geometry.bottom_plate_normal_local
        plate_normal /= max(float(np.linalg.norm(plate_normal)), 1.0e-12)
        requested_target = (
            np.asarray(self.phase_target_rad, dtype=np.float64)
            if requested_joint_target_rad is None
            else np.asarray(requested_joint_target_rad, dtype=np.float64)
        )
        mobile_step = interaction.mobile_result
        contact_support = interaction.failure_bridge.tool_mobile_contact
        self.records.append({
            "simulation_time_s": float(simulation_time_s),
            "phase_time_s": float(simulation_time_s - self.first_simulation_time_s),
            "phase": str(phase),
            "trajectory_stage": str(trajectory_stage),
            "payload_before_m3": float(payload_before_m3),
            "gross_intake_m3": float(intake.bucket_inflow_volume_m3),
            "candidate_mouth_flux_m3": float(intake.candidate_flux_volume_m3),
            "accepted_mouth_flux_m3": float(intake.bucket_inflow_volume_m3),
            "capacity_rejected_flux_m3": float(intake.candidate_flux_volume_m3 - intake.bucket_inflow_volume_m3),
            "capacity_limited": bool(intake.capacity_limited),
            "payload_after_m3": float(payload_after_m3),
            "payload_spill_m3": 0.0,
            "mouth_pose_terrain": pose.tolist(),
            "mouth_normal_terrain": normal.tolist(),
            "mouth_area_m2": float(geometry.mouth_area_m2),
            "mouth_intersection_area_m2": float(intake.mouth_intersection_area_m2),
            "mouth_overlap_length_m": float(intake.mouth_overlap_length_m),
            "mean_positive_relative_normal_speed_m_s": float(intake.mean_positive_relative_speed_m_s),
            "bucket_linear_velocity_m_s": np.asarray(tool_state.linear_velocity).tolist(),
            "bucket_angular_velocity_rad_s": np.asarray(tool_state.angular_velocity).tolist(),
            "intake_centroid_terrain_m": np.asarray(intake.intake_centroid_terrain_m).tolist(),
            "proximity": proximity,
            "local_mobile": local_mobile,
            "cutting_edge_center_terrain_m": edge_center.tolist(),
            "cutting_edge_velocity_terrain_m_s": edge_velocity.tolist(),
            "tool_velocity_decomposition": {
                "V_TOOL_TOTAL_M_S": float(np.linalg.norm(edge_velocity)),
                "V_PENETRATION_COMPONENT_M_S": float(np.dot(edge_velocity, plate_normal)),
                "V_SEPARATION_COMPONENT_M_S": float(np.dot(edge_velocity, separation)),
                "plate_normal_terrain": plate_normal.tolist(),
                "separation_tangent_terrain": separation.tolist(),
            },
            "maximum_penetration_m": float(np.max(interaction.failure_bridge.intersection.penetration_depth_m)),
            "joint_position_rad": q.tolist(),
            "joint_velocity_rad_s": np.asarray(joint_velocity_rad_s).reshape(-1).tolist(),
            "requested_joint_target_rad": requested_target.tolist(),
            "BUCKET_CURL_RATE_RAD_S": float(np.asarray(joint_velocity_rad_s).reshape(-1)[3]),
            "STICK_RETRACTION_RATE_RAD_S": float(np.asarray(joint_velocity_rad_s).reshape(-1)[2]),
            "BOOM_RATE_RAD_S": float(np.asarray(joint_velocity_rad_s).reshape(-1)[1]),
            "tool_progress_fraction": progress,
            "quasi_static_force_n": np.asarray(quasi_static_force_n).tolist(),
            "dynamic_momentum_force_n": np.asarray(momentum_force_n).tolist(),
            "tool_mobile_contact": {
                "geometry_confirmed_candidate_cell_count": int(contact_support.cell_count),
                "geometry_confirmed_mobile_volume_m3": float(contact_support.mobile_volume_m3),
                "geometry_performance": dict(
                    contact_support.performance_diagnostics
                ),
                "mobile_nonzero_cell_count": int(
                    mobile_step.mobile_nonzero_cell_count
                ),
                "mobile_substep_count": int(mobile_step.substeps),
                "contact_prepare_dispatch_ms": float(
                    mobile_step.tool_mobile_contact_prepare_dispatch_ms
                ),
                "contact_h2d_bytes": int(
                    mobile_step.tool_mobile_contact_h2d_bytes
                ),
                "contact_h2d_transfer_count": int(
                    mobile_step.tool_mobile_contact_h2d_transfer_count
                ),
                "mobile_transport_and_source_sync_ms": float(
                    mobile_step.mobile_transport_and_source_sync_ms
                ),
                "tool_mobile_impulse_fused_upper_bound_ms": float(
                    mobile_step.tool_mobile_impulse_fused_upper_bound_ms
                ),
                "impulse_timing_scope": str(
                    mobile_step.tool_mobile_impulse_timing_scope
                ),
                "normal_impulse_ns": float(mobile_step.tool_normal_impulse_ns),
                "tangential_impulse_ns": float(mobile_step.tool_tangential_impulse_ns),
                "accepted_tool_to_mobile_impulse_terrain_ns": np.asarray(
                    mobile_step.tool_impulse_on_mobile_terrain_ns
                ).tolist(),
                "machine_reaction_impulse_terrain_ns": (
                    -np.asarray(mobile_step.tool_impulse_on_mobile_terrain_ns)
                ).tolist(),
                "angular_impulse_on_mobile_about_tool_origin_terrain_nms": np.asarray(
                    mobile_step.tool_angular_impulse_on_mobile_about_tool_origin_terrain_nms
                ).tolist(),
                "contact_centroid_terrain_m": np.asarray(
                    mobile_step.tool_contact_centroid_terrain_m
                ).tolist(),
                "tool_to_mobile_work_j": float(mobile_step.tool_work_j),
                "machine_reaction_work_j": float(mobile_step.machine_reaction_work_j),
                "frictional_dissipation_j": float(
                    mobile_step.tool_frictional_dissipation_j
                ),
                "total_contact_dissipation_j": float(
                    mobile_step.tool_contact_dissipation_j
                ),
                "active_substep_count": int(mobile_step.tool_contact_active_substeps),
                "active_cell_substep_count": int(
                    mobile_step.tool_contact_active_cell_substeps
                ),
                "peak_contact_mobile_volume_m3": float(
                    mobile_step.peak_tool_contact_mobile_volume_m3
                ),
                "substeps": list(mobile_step.tool_contact_substep_diagnostics),
            },
            "applied_soil_force_n": np.asarray(applied_force_n).tolist(),
            "soil_power_on_terrain_w": float(np.dot(np.asarray(applied_force_n), np.asarray(tool_state.linear_velocity))),
            "actuator_effort_command_nm": np.asarray(actuator_output.effort_command_nm).tolist(),
            "actuator_effort_saturated": np.asarray(actuator_output.effort_saturated, dtype=bool).tolist(),
            "actuator_shared_power_scale": float(actuator_output.shared_power_scale),
            "machine_velocity_terrain_m_s": (
                [0.0, 0.0, 0.0]
                if machine_velocity_terrain_m_s is None
                else np.asarray(machine_velocity_terrain_m_s, dtype=np.float64).tolist()
            ),
            "force_application_point_terrain_m": (
                None
                if force_application_point_terrain_m is None
                else np.asarray(force_application_point_terrain_m, dtype=np.float64).tolist()
            ),
            "failure_zone_r2m_step_m3": float(failure_zone_r2m_step_m3),
            "mobile_volume_m3": (
                None if mobile_volume_m3 is None else float(mobile_volume_m3)
            ),
            "r2m_cumulative_m3": float(avalanche.cumulative_resting_to_mobile_m3),
            "m2r_cumulative_m3": float(avalanche.cumulative_mobile_to_resting_m3),
            "mass_balance_error_m3": abs(float(mass_balance_error_m3)),
        })

    def report(self, *, failure_code: str | None) -> dict[str, Any]:
        if not self.records:
            return {"status": "NO_CUT_AND_FILL_RECORDS"}
        first, last = self.records[0], self.records[-1]
        gross_m2p = float(sum(item["gross_intake_m3"] for item in self.records))
        candidate = float(sum(item["candidate_mouth_flux_m3"] for item in self.records))
        spill = float(sum(item["payload_spill_m3"] for item in self.records))
        net = float(last["payload_after_m3"] - self.entry_payload_m3)
        avalanche_r2m = float(last["r2m_cumulative_m3"] - self.r2m_start_m3)
        r2m = self.failure_zone_r2m_gross_m3 + avalanche_r2m
        m2r = float(last["m2r_cumulative_m3"] - self.m2r_start_m3)
        mouth_volumes = np.asarray([item["proximity"]["mouth_prism_m3"] for item in self.records])
        overlap = np.asarray([item["mouth_intersection_area_m2"] for item in self.records])
        relative = np.asarray([item["mean_positive_relative_normal_speed_m_s"] for item in self.records])
        saturated = np.asarray([any(item["actuator_effort_saturated"]) for item in self.records])
        power_limited = np.asarray([item["actuator_shared_power_scale"] < 1.0 - 1e-9 for item in self.records])
        accepted = gross_m2p
        capture = accepted / candidate if candidate > 0.0 else 0.0
        positive_relative = relative[relative > 0.0]
        mean_positive_relative = (
            float(np.mean(positive_relative)) if positive_relative.size else 0.0
        )
        mouth_to_r2m = float(np.mean(mouth_volumes)) / r2m if r2m > 0.0 else 0.0
        candidate_to_r2m = candidate / r2m if r2m > 0.0 else 0.0
        if (
            r2m > 0.0
            and mouth_to_r2m < 0.01
            and candidate_to_r2m < 0.01
            and capture > 0.99
        ):
            primary = "MOBILE_NOT_REACHING_MOUTH"
            intake_class = "INTAKE_FLUX_TOO_SMALL"
        elif np.count_nonzero(overlap > 0.0) < max(2, len(overlap) // 10):
            primary = "BUCKET_MOUTH_FLUX_GEOMETRY_ERROR"
            intake_class = "MOUTH_GEOMETRY_MISALIGNED"
        elif np.count_nonzero(relative > 0.0) < max(2, len(relative) // 10):
            primary = "RELATIVE_VELOCITY_SIGN_ERROR"
            intake_class = "RELATIVE_FLUX_WRONG_SIGN"
        elif gross_m2p < 0.02 and np.mean(saturated | power_limited) > 0.5:
            primary = "SOIL_FORCE_COUPLING_STALL"
            intake_class = "INTAKE_FLUX_TOO_SMALL"
        elif capture < 1.0 - 1.0e-9:
            primary = "PAYLOAD_RETENTION_LOSS" if spill > 0.0 else "CAPACITY_LIMIT"
            intake_class = "CAPACITY_LIMIT"
        else:
            primary = "OTHER"
            intake_class = "INTAKE_FLUX_TOO_SMALL"
        peak_force = max(float(np.linalg.norm(item["applied_soil_force_n"])) for item in self.records)
        active = np.asarray(
            [
                item["maximum_penetration_m"] > 0.0
                and item.get("failure_zone_r2m_step_m3", 0.0) > 0.0
                and item["proximity"]["radius_2p0_m3"] > 0.0
                for item in self.records
            ],
            dtype=bool,
        )
        if not np.any(active):
            # Historical v1 records did not retain per-step FailureZone R2M.
            # This fallback is explicitly geometric, never used by physics.
            active = np.asarray(
                [
                    item["maximum_penetration_m"] > 0.0
                    and item["proximity"]["radius_2p0_m3"] > 0.0
                    for item in self.records
                ],
                dtype=bool,
            )
        active_records = [item for item, enabled in zip(self.records, active) if enabled]
        curl = np.asarray([
            item.get("BUCKET_CURL_RATE_RAD_S", item.get("joint_velocity_rad_s", [0.0] * 4)[3])
            for item in self.records
        ])
        retract = np.asarray([
            item.get("STICK_RETRACTION_RATE_RAD_S", item.get("joint_velocity_rad_s", [0.0] * 4)[2])
            for item in self.records
        ])
        penetration_component = np.asarray([
            item.get("tool_velocity_decomposition", {}).get("V_PENETRATION_COMPONENT_M_S", 0.0)
            for item in self.records
        ])
        separation_component = np.asarray([
            item.get("tool_velocity_decomposition", {}).get("V_SEPARATION_COMPONENT_M_S", 0.0)
            for item in self.records
        ])
        active_count = max(int(np.count_nonzero(active)), 1)
        curl_active = curl > 0.01 * 0.70
        retract_active = retract > 0.01 * 0.55
        inward = relative > 0.0
        active_relative = relative[active & inward]
        failure_rates = np.asarray(
            [item.get("failure_zone_r2m_step_m3", 0.0) / self.dt_s for item in self.records]
        )

        def peak_time(values: np.ndarray) -> float | None:
            if values.size == 0:
                return None
            record = self.records[int(np.argmax(values))]
            return float(record.get("simulation_time_s", 0.0))

        trajectory_metrics = {
            "active_interval_definition": (
                "penetration>0 AND FailureZone_R2M_step>0 AND Mobile within 2m; "
                "historical records without per-step R2M use the declared geometric fallback"
            ),
            "joint_activity_definition": "positive rate > 1% of production joint velocity limit",
            "ACTIVE_CUT_DURATION_S": float(np.count_nonzero(active) * self.dt_s),
            "PENETRATION_DOMINATED_FRACTION": float(
                np.count_nonzero(active & (np.abs(penetration_component) > np.abs(separation_component)))
                / active_count
            ),
            "BUCKET_CURL_ACTIVE_FRACTION": float(np.count_nonzero(active & curl_active) / active_count),
            "STICK_RETRACT_ACTIVE_FRACTION": float(np.count_nonzero(active & retract_active) / active_count),
            "SIMULTANEOUS_STICK_RETRACT_BUCKET_CURL_FRACTION": float(
                np.count_nonzero(active & curl_active & retract_active) / active_count
            ),
            "POSITIVE_INWARD_TRANSPORT_FRACTION": float(np.count_nonzero(active & inward) / active_count),
            "PEAK_BUCKET_CURL_RATE_RAD_S": float(np.max(curl[active], initial=0.0)),
            "MEAN_BUCKET_CURL_RATE_DURING_ACTIVE_CUT_RAD_S": float(np.mean(curl[active])) if np.any(active) else 0.0,
            "MEAN_STICK_RETRACTION_RATE_DURING_ACTIVE_CUT_RAD_S": float(np.mean(retract[active])) if np.any(active) else 0.0,
            "MOBILE_MOUTH_P95_M3": float(np.percentile(mouth_volumes[active], 95.0)) if np.any(active) else 0.0,
            "MOBILE_FRONT_P95_M3": float(np.percentile(
                np.asarray([item["proximity"]["front_m3"] for item in self.records])[active], 95.0
            )) if np.any(active) else 0.0,
            "P95_POSITIVE_RELATIVE_NORMAL_SPEED_M_S": float(np.percentile(active_relative, 95.0)) if active_relative.size else 0.0,
            "temporal_ordering": {
                "peak_failure_zone_r2m_rate_time_s": peak_time(failure_rates),
                "peak_mouth_overlap_time_s": peak_time(overlap),
                "peak_mouth_flux_time_s": peak_time(np.asarray([item["candidate_mouth_flux_m3"] for item in self.records])),
                "peak_bucket_curl_rate_time_s": peak_time(curl),
                "peak_penetration_time_s": peak_time(np.asarray([item["maximum_penetration_m"] for item in self.records])),
            },
        }
        empty_contact = {
            "geometry_confirmed_mobile_volume_m3": 0.0,
            "normal_impulse_ns": 0.0,
            "tangential_impulse_ns": 0.0,
            "accepted_tool_to_mobile_impulse_terrain_ns": [0.0, 0.0, 0.0],
            "machine_reaction_impulse_terrain_ns": [0.0, 0.0, 0.0],
            "tool_to_mobile_work_j": 0.0,
            "machine_reaction_work_j": 0.0,
            "frictional_dissipation_j": 0.0,
            "total_contact_dissipation_j": 0.0,
            "active_substep_count": 0,
            "active_cell_substep_count": 0,
            "peak_contact_mobile_volume_m3": 0.0,
            "substeps": [],
        }
        contact_records = [
            item.get("tool_mobile_contact", empty_contact) for item in self.records
        ]
        accepted_impulse = sum(
            (
                np.asarray(item["accepted_tool_to_mobile_impulse_terrain_ns"], dtype=np.float64)
                for item in contact_records
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        machine_impulse = sum(
            (
                np.asarray(item["machine_reaction_impulse_terrain_ns"], dtype=np.float64)
                for item in contact_records
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        action_reaction = accepted_impulse + machine_impulse
        impulse_scale = max(float(np.linalg.norm(accepted_impulse)), 1.0e-30)
        contact_substeps = [
            substep
            for item in contact_records
            for substep in item["substeps"]
            if int(substep["contact_active_cell_count"]) > 0
        ]
        unexplained_energy = float(sum(
            float(substep["mobile_kinetic_energy_change_due_to_contact_j"])
            + float(substep["total_contact_dissipation_j"])
            - float(substep["tool_to_mobile_work_j"])
            for substep in contact_substeps
        ))
        return {
            "schema": "CUT_AND_FILL_PAYLOAD_CAUSAL_REPORT/v1",
            "status": "CAUSAL_AUDIT_COMPLETE",
            "failure_code": failure_code,
            "R2M_GROSS_M3": r2m,
            "FAILURE_ZONE_R2M_GROSS_M3": self.failure_zone_r2m_gross_m3,
            "LARGE_AVALANCHE_R2M_GROSS_M3": avalanche_r2m,
            "M2P_GROSS_M3": gross_m2p,
            "PAYLOAD_SPILL_GROSS_M3": spill,
            "PAYLOAD_TO_AIRBORNE_M3": 0.0,
            "M2R_GROSS_M3": m2r,
            "MOBILE_TO_OUTFLOW_M3": 0.0,
            "NET_PAYLOAD_GAIN_M3": net,
            "PAYLOAD_BUDGET_RESIDUAL_M3": net - gross_m2p + spill,
            "MOBILE_IN_MOUTH_ROI_M3": {
                "maximum": float(np.max(mouth_volumes)),
                "mean": float(np.mean(mouth_volumes)),
            },
            "CANDIDATE_MOUTH_FLUX_M3": candidate,
            "ACCEPTED_MOUTH_FLUX_M3": accepted,
            "CAPACITY_REJECTED_FLUX_M3": candidate - accepted,
            "MOUTH_CAPTURE_RATIO": capture,
            "MEAN_POSITIVE_RELATIVE_NORMAL_SPEED_M_S": mean_positive_relative,
            "P95_POSITIVE_RELATIVE_NORMAL_SPEED_M_S": trajectory_metrics["P95_POSITIVE_RELATIVE_NORMAL_SPEED_M_S"],
            "MEAN_MOUTH_PRISM_TO_R2M_RATIO": mouth_to_r2m,
            "GROSS_CAPTURE_RATIO": gross_m2p / r2m if r2m > 0.0 else None,
            "NET_CAPTURE_RATIO": net / r2m if r2m > 0.0 else None,
            "TOOL_PROGRESS_FRACTION": float(max(item["tool_progress_fraction"] for item in self.records)),
            "MAX_PENETRATION_M": float(max(item["maximum_penetration_m"] for item in self.records)),
            "MEAN_PENETRATION_WHILE_ACTIVE_M": float(np.mean([item["maximum_penetration_m"] for item in self.records if item["maximum_penetration_m"] > 0.0])),
            "PEAK_SOIL_FORCE_N": peak_force,
            "SOIL_FORCE": {
                "peak_quasi_static_n": max(float(np.linalg.norm(item["quasi_static_force_n"])) for item in self.records),
                "peak_dynamic_mobile_n": max(float(np.linalg.norm(item["dynamic_momentum_force_n"])) for item in self.records),
                "effort_saturated_step_fraction": float(np.mean(saturated)),
                "power_limited_step_fraction": float(np.mean(power_limited)),
                "cumulative_soil_work_on_terrain_j": float(sum(item["soil_power_on_terrain_w"] * self.dt_s for item in self.records)),
            },
            "TOOL_MOBILE_COUPLING": {
                "contact_active_step_count": int(sum(
                    item["active_substep_count"] > 0 for item in contact_records
                )),
                "contact_active_substep_count": int(sum(
                    item["active_substep_count"] for item in contact_records
                )),
                "contact_active_cell_substep_count": int(sum(
                    item["active_cell_substep_count"] for item in contact_records
                )),
                "peak_geometry_confirmed_mobile_volume_m3": float(max(
                    item["geometry_confirmed_mobile_volume_m3"] for item in contact_records
                )),
                "peak_contact_mobile_volume_m3": float(max(
                    item["peak_contact_mobile_volume_m3"] for item in contact_records
                )),
                "total_normal_impulse_ns": float(sum(
                    item["normal_impulse_ns"] for item in contact_records
                )),
                "total_tangential_impulse_ns": float(sum(
                    item["tangential_impulse_ns"] for item in contact_records
                )),
                "total_mobile_dynamic_impulse_terrain_ns": accepted_impulse.tolist(),
                "total_machine_reaction_impulse_terrain_ns": machine_impulse.tolist(),
                "action_reaction_residual_terrain_ns": action_reaction.tolist(),
                "action_reaction_residual_norm_ns": float(np.linalg.norm(action_reaction)),
                "action_reaction_relative_residual": float(
                    np.linalg.norm(action_reaction) / impulse_scale
                ),
                "tool_to_mobile_work_j": float(sum(
                    item["tool_to_mobile_work_j"] for item in contact_records
                )),
                "machine_reaction_work_j": float(sum(
                    item["machine_reaction_work_j"] for item in contact_records
                )),
                "frictional_dissipation_j": float(sum(
                    item["frictional_dissipation_j"] for item in contact_records
                )),
                "total_contact_dissipation_j": float(sum(
                    item["total_contact_dissipation_j"] for item in contact_records
                )),
                "unexplained_contact_energy_j": unexplained_energy,
            },
            "SOIL_FORCE_LIMITING_ROLE": (
                "ACTUATOR_OR_POWER_LIMIT_CORRELATED" if np.mean(saturated | power_limited) > 0.5
                else "NOT_FORCE_LIMITED"
            ),
            "INTAKE_CLASSIFICATION": intake_class,
            "PRIMARY_BOTTLENECK": primary,
            "MAX_MASS_ERROR_M3": max(abs(float(item.get("mass_balance_error_m3", 0.0))) for item in self.records),
            "TRAJECTORY_METRICS": trajectory_metrics,
            "proximity_maxima_m3": {
                key: max(item["proximity"][key] for item in self.records)
                for key in first["proximity"]
            },
            "step_count": len(self.records),
            "time_series": self.records,
        }
