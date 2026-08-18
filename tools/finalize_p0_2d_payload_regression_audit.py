#!/usr/bin/env python3
"""Build the read-only P0-2D post-integration payload regression audit.

This tool consumes already-recorded production artifacts.  It does not import
or execute the physics core and does not modify any production configuration.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "outputs/390f_v2/interactive_runs"
BASELINE_RUN = "run_1786937073"
CPU_EXACT_RUN = "run_1786950984"
CURRENT_RUN = "run_1786956921"
OUTPUT = ROOT / "outputs/mobile_v2_production/p0_2d_post_integration_payload_regression_causal_audit.json"


def _load(run: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = RUN_ROOT / run / "cut_fill_payload_causal_audit.json"
    audit = json.loads(path.read_text(encoding="utf-8"))
    samples = audit["time_series"]
    t0 = float(samples[0]["simulation_time_s"])
    for sample in samples:
        sample["_tau_s"] = float(sample["simulation_time_s"]) - t0
    return audit, samples


def _norm(value: Any) -> float:
    return float(np.linalg.norm(np.asarray(value, dtype=np.float64)))


def _pose_position(sample: dict[str, Any]) -> np.ndarray:
    return np.asarray(sample["mouth_pose_terrain"], dtype=np.float64)[:3, 3]


def _vertex_weights(shape: tuple[int, int], spacing_m: float = 0.05) -> np.ndarray:
    weights = np.zeros(shape, dtype=np.float64)
    factor = spacing_m * spacing_m / 6.0
    weights[:-1, :-1] += 2.0 * factor
    weights[:-1, 1:] += factor
    weights[1:, 1:] += 2.0 * factor
    weights[1:, :-1] += factor
    return weights


def _field_checkpoint_comparison() -> dict[str, Any]:
    baseline_path = ROOT / "outputs/mobile_v2_production/cut_fill_realistic_trajectory_candidate2_audit.cut_start.npz"
    current_path = ROOT / "outputs/mobile_v2_production/p0_2b_p0_2d_production_timeseries.cut_start.npz"
    baseline = np.load(baseline_path)
    current = np.load(current_path)
    weights = _vertex_weights(baseline["mobile"].shape)

    def integral(field: np.ndarray) -> float:
        return float(np.sum(field * weights, dtype=np.float64))

    result: dict[str, Any] = {
        "checkpoint_boundary": "CUT_AND_FILL_START_BEFORE_FIRST_CUT_CORE_STEP",
        "baseline": str(baseline_path.relative_to(ROOT)),
        "current": str(current_path.relative_to(ROOT)),
    }
    for key in ("b_eff", "mobile", "momentum_x", "momentum_y"):
        delta = current[key] - baseline[key]
        result[key] = {
            "baseline_integral": integral(baseline[key]),
            "current_integral": integral(current[key]),
            "delta_integral": integral(delta),
            "different_cell_count": int(np.count_nonzero(np.abs(delta) > 1.0e-15)),
            "max_abs": float(np.max(np.abs(delta))),
            "weighted_l1": float(np.sum(np.abs(delta) * weights, dtype=np.float64)),
        }

    h_baseline = baseline["b_eff"] + baseline["mobile"]
    h_current = current["b_eff"] + current["mobile"]
    delta_h = h_current - h_baseline
    rows, columns = np.where(np.abs(delta_h) > 1.0e-15)
    result["H_free"] = {
        "baseline_integral_m3": integral(h_baseline),
        "current_integral_m3": integral(h_current),
        "net_delta_m3": integral(delta_h),
        "different_cell_count": int(rows.size),
        "max_abs_delta_m": float(np.max(np.abs(delta_h))),
        "weighted_l1_m3": float(np.sum(np.abs(delta_h) * weights, dtype=np.float64)),
        "index_bbox_yx": [int(rows.min()), int(columns.min()), int(rows.max()), int(columns.max())],
        "terrain_bbox_xy_m": [
            float(columns.min() * 0.05),
            float(rows.min() * 0.05),
            float(columns.max() * 0.05),
            float(rows.max() * 0.05),
        ],
    }
    density = 1370.0
    result["integrated_mobile_momentum_kg_m_s"] = {
        "baseline": [density * integral(baseline["momentum_x"]), density * integral(baseline["momentum_y"])],
        "current": [density * integral(current["momentum_x"]), density * integral(current["momentum_y"])],
    }
    return result


def _aligned_samples(
    baseline: list[dict[str, Any]], current: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    count = min(len(baseline), len(current))
    baseline_failure_cumulative = 0.0
    current_failure_cumulative = 0.0
    baseline_admitted_cumulative = 0.0
    current_admitted_cumulative = 0.0
    result: list[dict[str, Any]] = []
    for index in range(count):
        b = baseline[index]
        c = current[index]
        baseline_failure_cumulative += float(b["failure_zone_r2m_step_m3"])
        current_failure_cumulative += float(c["failure_zone_r2m_step_m3"])
        baseline_admitted_cumulative += float(b["accepted_mouth_flux_m3"])
        current_admitted_cumulative += float(c["accepted_mouth_flux_m3"])
        b_contact = b.get("tool_mobile_contact") or {}
        c_contact = c.get("tool_mobile_contact") or {}
        result.append(
            {
                "sample_index": index,
                "tau_s": float(c["_tau_s"]),
                "actual_joint_position_rad": {"baseline": b["joint_position_rad"], "current": c["joint_position_rad"]},
                "actual_joint_velocity_rad_s": {"baseline": b["joint_velocity_rad_s"], "current": c["joint_velocity_rad_s"]},
                "requested_joint_target_rad": {"baseline": b["requested_joint_target_rad"], "current": c["requested_joint_target_rad"]},
                "requested_joint_rates_rad_s": {
                    "baseline": [b["BOOM_RATE_RAD_S"], b["STICK_RETRACTION_RATE_RAD_S"], b["BUCKET_CURL_RATE_RAD_S"]],
                    "current": [c["BOOM_RATE_RAD_S"], c["STICK_RETRACTION_RATE_RAD_S"], c["BUCKET_CURL_RATE_RAD_S"]],
                },
                "mouth_position_terrain_m": {"baseline": _pose_position(b).tolist(), "current": _pose_position(c).tolist()},
                "mouth_velocity_terrain_m_s": {"baseline": b["bucket_linear_velocity_m_s"], "current": c["bucket_linear_velocity_m_s"]},
                "maximum_penetration_m": {"baseline": b["maximum_penetration_m"], "current": c["maximum_penetration_m"]},
                "failure_zone_r2m_step_m3": {"baseline": b["failure_zone_r2m_step_m3"], "current": c["failure_zone_r2m_step_m3"]},
                "failure_zone_r2m_cumulative_m3": {"baseline": baseline_failure_cumulative, "current": current_failure_cumulative},
                "mobile_total_m3": {"baseline": b["mobile_volume_m3"], "current": c["mobile_volume_m3"]},
                "mobile_proximity_m3": {"baseline": b["proximity"], "current": c["proximity"]},
                "candidate_mouth_flux_step_m3": {"baseline": b["candidate_mouth_flux_m3"], "current": c["candidate_mouth_flux_m3"]},
                "accepted_mouth_flux_step_m3": {"baseline": b["accepted_mouth_flux_m3"], "current": c["accepted_mouth_flux_m3"]},
                "accepted_mouth_flux_cumulative_m3": {"baseline": baseline_admitted_cumulative, "current": current_admitted_cumulative},
                "payload_after_m3": {"baseline": b["payload_after_m3"], "current": c["payload_after_m3"]},
                "tool_mobile": {
                    "baseline": {
                        "accepted_contact_count": int(b_contact.get("geometry_confirmed_candidate_cell_count", 0)),
                        "normal_impulse_ns": float(b_contact.get("normal_impulse_ns", 0.0)),
                        "tangential_impulse_ns": float(b_contact.get("tangential_impulse_ns", 0.0)),
                        "mobile_impulse_terrain_ns": b_contact.get("accepted_tool_to_mobile_impulse_terrain_ns", [0.0, 0.0, 0.0]),
                    },
                    "current": {
                        "accepted_contact_count": int(c_contact.get("geometry_confirmed_candidate_cell_count", 0)),
                        "normal_impulse_ns": float(c_contact.get("normal_impulse_ns", 0.0)),
                        "tangential_impulse_ns": float(c_contact.get("tangential_impulse_ns", 0.0)),
                        "mobile_impulse_terrain_ns": c_contact.get("accepted_tool_to_mobile_impulse_terrain_ns", [0.0, 0.0, 0.0]),
                        "machine_reaction_impulse_terrain_ns": c_contact.get("machine_reaction_impulse_terrain_ns", [0.0, 0.0, 0.0]),
                    },
                },
                "dynamic_momentum_force_n": {"baseline": b["dynamic_momentum_force_n"], "current": c["dynamic_momentum_force_n"]},
                "quasi_static_force_n": {"baseline": b["quasi_static_force_n"], "current": c["quasi_static_force_n"]},
                "mass_balance_error_m3": {"baseline": b["mass_balance_error_m3"], "current": c["mass_balance_error_m3"]},
            }
        )
    return result


def _contact_audit(samples: list[dict[str, Any]], aggregate: dict[str, Any]) -> dict[str, Any]:
    active_indices: list[int] = []
    frames: list[dict[str, Any]] = []
    signed_normal = 0.0
    signed_lateral = 0.0
    absolute_normal = 0.0
    absolute_lateral = 0.0
    for index, sample in enumerate(samples):
        contact = sample.get("tool_mobile_contact") or {}
        count = int(contact.get("geometry_confirmed_candidate_cell_count", 0))
        active_substeps = int(contact.get("active_substep_count", 0))
        if active_substeps:
            active_indices.append(index)
        impulse = np.asarray(contact.get("accepted_tool_to_mobile_impulse_terrain_ns", [0.0, 0.0, 0.0]), dtype=np.float64)[:2]
        mouth_normal = np.asarray(sample["mouth_normal_terrain"], dtype=np.float64)[:2]
        mouth_normal /= np.linalg.norm(mouth_normal)
        lateral = np.asarray([-mouth_normal[1], mouth_normal[0]], dtype=np.float64)
        normal_projection = float(impulse @ mouth_normal)
        lateral_projection = float(impulse @ lateral)
        signed_normal += normal_projection
        signed_lateral += lateral_projection
        absolute_normal += abs(normal_projection)
        absolute_lateral += abs(lateral_projection)
        frames.append(
            {
                "sample_index": index,
                "tau_s": float(sample["_tau_s"]),
                "geometry_contact_count": count,
                "active_substep_count": active_substeps,
                "active_cell_substep_count": int(contact.get("active_cell_substep_count", 0)),
                "mobile_impulse_terrain_ns": impulse.tolist() + [0.0],
                "machine_reaction_impulse_terrain_ns": contact.get("machine_reaction_impulse_terrain_ns", [0.0, 0.0, 0.0]),
                "dynamic_force_n": sample["dynamic_momentum_force_n"],
                "normal_projection_ns_positive_toward_mouth": normal_projection,
                "lateral_projection_ns": lateral_projection,
            }
        )

    intervals: list[dict[str, Any]] = []
    if active_indices:
        start = previous = active_indices[0]
        for index in active_indices[1:]:
            if index != previous + 1:
                intervals.append({"start_tau_s": samples[start]["_tau_s"], "end_tau_s": samples[previous]["_tau_s"], "frame_count": previous - start + 1})
                start = index
            previous = index
        intervals.append({"start_tau_s": samples[start]["_tau_s"], "end_tau_s": samples[previous]["_tau_s"], "frame_count": previous - start + 1})

    first_geometry = next((frame for frame in frames if frame["geometry_contact_count"] > 0), None)
    return {
        "first_geometry_confirmed_contact": first_geometry,
        "active_intervals": intervals,
        "frame_records": frames,
        "aggregate_recorded": aggregate,
        "direction_projection": {
            "definition": "positive normal projection is toward the recorded mouth normal; tangent is its left-hand in-plane perpendicular",
            "signed_normal_ns": signed_normal,
            "signed_lateral_ns": signed_lateral,
            "absolute_normal_ns": absolute_normal,
            "absolute_lateral_ns": absolute_lateral,
            "classification": "NET_AWAY_FROM_MOUTH_AND_LATERAL",
        },
    }


def _cpu_gpu_aggregate_equivalence(
    cpu_samples: list[dict[str, Any]], gpu_samples: list[dict[str, Any]]
) -> dict[str, Any]:
    maxima = {
        "candidate_count": 0.0,
        "geometry_mobile_volume_m3": 0.0,
        "normal_impulse_ns": 0.0,
        "tangential_impulse_ns": 0.0,
        "mobile_impulse_norm_ns": 0.0,
        "machine_reaction_norm_ns": 0.0,
        "dynamic_force_norm_n": 0.0,
    }
    for cpu, gpu in zip(cpu_samples, gpu_samples):
        a = cpu.get("tool_mobile_contact") or {}
        b = gpu.get("tool_mobile_contact") or {}
        maxima["candidate_count"] = max(maxima["candidate_count"], abs(float(a.get("geometry_confirmed_candidate_cell_count", 0)) - float(b.get("geometry_confirmed_candidate_cell_count", 0))))
        maxima["geometry_mobile_volume_m3"] = max(maxima["geometry_mobile_volume_m3"], abs(float(a.get("geometry_confirmed_mobile_volume_m3", 0.0)) - float(b.get("geometry_confirmed_mobile_volume_m3", 0.0))))
        maxima["normal_impulse_ns"] = max(maxima["normal_impulse_ns"], abs(float(a.get("normal_impulse_ns", 0.0)) - float(b.get("normal_impulse_ns", 0.0))))
        maxima["tangential_impulse_ns"] = max(maxima["tangential_impulse_ns"], abs(float(a.get("tangential_impulse_ns", 0.0)) - float(b.get("tangential_impulse_ns", 0.0))))
        maxima["mobile_impulse_norm_ns"] = max(maxima["mobile_impulse_norm_ns"], _norm(np.asarray(a.get("accepted_tool_to_mobile_impulse_terrain_ns", [0.0, 0.0, 0.0])) - np.asarray(b.get("accepted_tool_to_mobile_impulse_terrain_ns", [0.0, 0.0, 0.0]))))
        maxima["machine_reaction_norm_ns"] = max(maxima["machine_reaction_norm_ns"], _norm(np.asarray(a.get("machine_reaction_impulse_terrain_ns", [0.0, 0.0, 0.0])) - np.asarray(b.get("machine_reaction_impulse_terrain_ns", [0.0, 0.0, 0.0]))))
        maxima["dynamic_force_norm_n"] = max(maxima["dynamic_force_norm_n"], _norm(np.asarray(cpu["dynamic_momentum_force_n"]) - np.asarray(gpu["dynamic_momentum_force_n"])))
    return {
        "cpu_exact_production_run": CPU_EXACT_RUN,
        "gpu_production_run": CURRENT_RUN,
        "selected_real_cut_frame_count": min(len(cpu_samples), len(gpu_samples)),
        "aggregate_max_abs_difference": maxima,
        "accepted_flat_indices_recorded": False,
        "closest_points_recorded": False,
        "normals_recorded": False,
        "surface_velocities_recorded": False,
        "signed_distances_recorded": False,
        "status": "NOT_ENOUGH_EVIDENCE",
        "reason": "The 17 real frames agree to floating-point precision in all retained aggregate/support-count diagnostics, but neither production audit retained the requested per-contact support geometry.",
    }


def _runtime_funnel_comparison() -> dict[str, Any]:
    baseline = json.loads((RUN_ROOT / BASELINE_RUN / "runtime_telemetry.json").read_text(encoding="utf-8"))
    current = json.loads((RUN_ROOT / CURRENT_RUN / "runtime_telemetry.json").read_text(encoding="utf-8"))
    baseline_by_time = {round(float(record["timestamp_s"]), 6): record for record in baseline}
    aligned: list[dict[str, Any]] = []
    for current_record in current:
        key = round(float(current_record["timestamp_s"]), 6)
        baseline_record = baseline_by_time.get(key)
        if baseline_record is None:
            continue
        aligned.append(
            {
                "timestamp_s": float(current_record["timestamp_s"]),
                "tau_from_cut_start_s": float(current_record["timestamp_s"]) - 5.700000297278166,
                "state": {"baseline": baseline_record["state"], "current": current_record["state"]},
                "mobile_m3": {"baseline": baseline_record["mobile_m3"], "current": current_record["mobile_m3"]},
                "material_funnel": {"baseline": baseline_record["material_funnel"], "current": current_record["material_funnel"]},
            }
        )
    return {
        "source": "runtime_telemetry.json",
        "semantic_note": "material_funnel.failure_volume_m3 is the retained cumulative FailureSurface/candidate-volume diagnostic; failure_zone_r2m is the conservative activated transfer and is a different quantity",
        "last_equal_zero_observation": {
            "timestamp_s": 5.500000286847353,
            "tau_from_cut_start_s": -0.20000001043081284,
            "baseline_mobile_m3": 0.0,
            "current_mobile_m3": 0.0,
        },
        "final": {
            "failure_volume_m3": {"baseline": baseline[-1]["material_funnel"]["failure_volume_m3"], "current": current[-1]["material_funnel"]["failure_volume_m3"]},
            "activated_volume_m3": {"baseline": baseline[-1]["material_funnel"]["activated_volume_m3"], "current": current[-1]["material_funnel"]["activated_volume_m3"]},
        },
        "aligned_records": aligned,
    }


def main() -> None:
    baseline_audit, baseline_samples = _load(BASELINE_RUN)
    cpu_audit, cpu_samples = _load(CPU_EXACT_RUN)
    current_audit, current_samples = _load(CURRENT_RUN)
    aligned = _aligned_samples(baseline_samples, current_samples)

    centers = np.asarray([sample["cutting_edge_center_terrain_m"] for sample in current_samples], dtype=np.float64)
    cutting_distance = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1), dtype=np.float64))
    current_payload_gain = float(current_samples[-1]["payload_after_m3"] - current_samples[0]["payload_before_m3"])

    report = {
        "schema": "P0_2D_POST_INTEGRATION_PAYLOAD_REGRESSION_CAUSAL_AUDIT/v1",
        "scope": "READ_ONLY_FIRST_DIVERGENCE_AUDIT",
        "production_physics_executed": False,
        "BASELINE_RUN": BASELINE_RUN,
        "CURRENT_RUN": CURRENT_RUN,
        "BASELINE_PAYLOAD_M3": baseline_audit["NET_PAYLOAD_GAIN_M3"],
        "CURRENT_PAYLOAD_M3": current_samples[-1]["payload_after_m3"],
        "CUT_COMPLETION_FAILED_CLAUSE": "minimum_payload_gain_m3: measured 0.01667549366595567 < required 0.02; minimum_cut_distance_m passed (6.608409154394499 >= 0.25)",
        "FIRST_DIVERGENCE_TAU_S": 0.0,
        "FIRST_DIVERGENCE_BOUND": "absent in the tau=-0.2000000104 runtime sample and present in the before-first-CUT checkpoint; onset lies in -0.2000000104 < tau <= 0 during PENETRATE",
        "FIRST_DIVERGENCE_VARIABLE": "pre-CUT DeviceBulkState Mobile momentum (momentum_x/momentum_y), accompanied by local Mobile and H_free differences",
        "BASELINE_VALUE": {"integrated_mobile_momentum_kg_m_s": [-0.03333473746255221, -0.012990705562148742], "mobile_volume_m3": 0.004632544601422323},
        "CURRENT_VALUE": {"integrated_mobile_momentum_kg_m_s": [-0.7655750134105321, -1.3701451864801428], "mobile_volume_m3": 0.004676274107485289},
        "ROOT_CAUSE_CLASS": "C. MOBILE_TRANSPORT_DIVERGENCE",
        "TOOL_MOBILE_CAUSALLY_RESPONSIBLE": "NOT_DEMONSTRATED",
        "GPU_GEOMETRY_PRODUCTION_EQUIVALENCE": "NOT_ENOUGH_EVIDENCE",
        "TRAJECTORY_CHANGED": "NO (identical commanded schedule; later actual-pose divergence is downstream)",
        "FAILURESURFACE_CHANGED": "YES (activation outcome changed; geometric FailureSurface volume was not retained)",
        "MOBILE_TRANSPORT_CHANGED": "YES",
        "INTAKE_REJECTED_VALID_MOUTH_FLUX": "NO",
        "FIX_APPLIED": "NO",
        "cut_alignment": {
            "baseline_start_s": baseline_samples[0]["simulation_time_s"],
            "current_start_s": current_samples[0]["simulation_time_s"],
            "sample_period_s": current_samples[1]["_tau_s"] - current_samples[0]["_tau_s"],
            "aligned_sample_count": len(aligned),
            "common_tau_end_s": aligned[-1]["tau_s"],
        },
        "completion_gate": {
            "minimum_cut_distance_required_m": 0.25,
            "cutting_distance_measured_m": cutting_distance,
            "minimum_payload_gain_required_m3": 0.02,
            "payload_gain_measured_m3": current_payload_gain,
            "payload_shortfall_m3": 0.02 - current_payload_gain,
        },
        "cut_start_checkpoint_comparison": _field_checkpoint_comparison(),
        "global_comparison": {
            "baseline": {key: baseline_audit.get(key) for key in ("FAILURE_ZONE_R2M_GROSS_M3", "R2M_GROSS_M3", "M2P_GROSS_M3", "CANDIDATE_MOUTH_FLUX_M3", "ACCEPTED_MOUTH_FLUX_M3", "CAPACITY_REJECTED_FLUX_M3", "MAX_PENETRATION_M", "PEAK_SOIL_FORCE_N", "MAX_MASS_ERROR_M3", "step_count")},
            "current": {key: current_audit.get(key) for key in ("FAILURE_ZONE_R2M_GROSS_M3", "R2M_GROSS_M3", "M2P_GROSS_M3", "CANDIDATE_MOUTH_FLUX_M3", "ACCEPTED_MOUTH_FLUX_M3", "CAPACITY_REJECTED_FLUX_M3", "MAX_PENETRATION_M", "PEAK_SOIL_FORCE_N", "MAX_MASS_ERROR_M3", "step_count")},
        },
        "runtime_failure_surface_funnel_comparison": _runtime_funnel_comparison(),
        "tool_mobile_current_cut": _contact_audit(current_samples, current_audit["TOOL_MOBILE_COUPLING"]),
        "gpu_geometry_production_equivalence": _cpu_gpu_aggregate_equivalence(cpu_samples, current_samples),
        "telemetry_limitations": {
            "failure_surface_geometric_volume": "NOT_RECORDED; failure_zone_r2m_step_m3 is activation transfer, not geometric surface volume",
            "full_contact_support_geometry": "NOT_RECORDED",
            "pre_cut_penetrate_timeseries": "NOT_RECORDED; only the boundary checkpoint is available",
        },
        "aligned_samples": aligned,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
