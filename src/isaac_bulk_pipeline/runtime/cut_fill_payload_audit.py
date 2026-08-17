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
        self.records.append({
            "simulation_time_s": float(simulation_time_s),
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
            "cutting_edge_center_terrain_m": np.mean(tool_state.cutting_edge_terrain, axis=0).tolist(),
            "maximum_penetration_m": float(np.max(interaction.failure_bridge.intersection.penetration_depth_m)),
            "joint_position_rad": q.tolist(),
            "joint_velocity_rad_s": np.asarray(joint_velocity_rad_s).reshape(-1).tolist(),
            "requested_joint_target_rad": np.asarray(self.phase_target_rad).tolist(),
            "tool_progress_fraction": progress,
            "quasi_static_force_n": np.asarray(quasi_static_force_n).tolist(),
            "dynamic_momentum_force_n": np.asarray(momentum_force_n).tolist(),
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
            "SOIL_FORCE_LIMITING_ROLE": (
                "ACTUATOR_OR_POWER_LIMIT_CORRELATED" if np.mean(saturated | power_limited) > 0.5
                else "NOT_FORCE_LIMITED"
            ),
            "INTAKE_CLASSIFICATION": intake_class,
            "PRIMARY_BOTTLENECK": primary,
            "proximity_maxima_m3": {
                key: max(item["proximity"][key] for item in self.records)
                for key in first["proximity"]
            },
            "step_count": len(self.records),
            "time_series": self.records,
        }
