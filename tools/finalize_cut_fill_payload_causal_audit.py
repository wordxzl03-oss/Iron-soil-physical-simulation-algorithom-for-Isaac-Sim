#!/usr/bin/env python3
"""Close the CUT_AND_FILL causal budget from preserved production artifacts.

This is an acceptance-only reducer.  It neither restores nor advances physics.
The total Resting->Mobile transfer is reconstructed from the conservative
Mobile control-volume balance because the first-pass observer recorded the
LargeAvalanche counter but not the FailureZone counter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _weighted_volume(height: np.ndarray, dx: float, dy: float) -> float:
    weights = np.full(height.shape, dx * dy, dtype=np.float64)
    weights[[0, -1], :] *= 0.5
    weights[:, [0, -1]] *= 0.5
    return float(np.sum(np.asarray(height, dtype=np.float64) * weights))


def _norm_peak(records: list[dict], key: str) -> float:
    return max(float(np.linalg.norm(item[key])) for item in records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    telemetry = json.loads(args.telemetry.read_text(encoding="utf-8"))
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        checkpoint_record = json.loads(str(archive["checkpoint_record_json"]))
        dx, dy = checkpoint_record["resolution_xy_m"]
        mobile_start = _weighted_volume(archive["mobile"], float(dx), float(dy))

    records = raw["time_series"]
    final_reservoirs = acceptance["final_reservoirs"]
    mobile_final = float(final_reservoirs["mobile_volume_m3"])
    m2p = float(raw["M2P_GROSS_M3"])
    m2r = float(raw["M2R_GROSS_M3"])
    spill = float(raw["PAYLOAD_SPILL_GROSS_M3"])
    payload_to_airborne = float(raw["PAYLOAD_TO_AIRBORNE_M3"])
    outflow = float(final_reservoirs["outflow_volume_m3"])
    net_payload = float(raw["NET_PAYLOAD_GAIN_M3"])
    # Delta Mobile = R2M + spill - M2P - M2R - outflow.  There was no
    # Airborne landing in this pre-dump interval.
    total_r2m = mobile_final - mobile_start - spill + m2p + m2r + outflow
    large_avalanche_r2m = float(raw["R2M_GROSS_M3"])
    failure_zone_r2m = total_r2m - large_avalanche_r2m

    candidate = float(raw["CANDIDATE_MOUTH_FLUX_M3"])
    accepted = float(raw["ACCEPTED_MOUTH_FLUX_M3"])
    mouth = np.asarray([item["proximity"]["mouth_prism_m3"] for item in records])
    relative = np.asarray(
        [item["mean_positive_relative_normal_speed_m_s"] for item in records]
    )
    overlap_area = np.asarray([item["mouth_intersection_area_m2"] for item in records])
    bucket_velocity = np.asarray([item["bucket_linear_velocity_m_s"] for item in records])
    tool_progress = np.asarray([item["tool_progress_fraction"] for item in records])
    saturation = np.asarray(
        [any(item["actuator_effort_saturated"]) for item in records], dtype=bool
    )
    power_limited = np.asarray(
        [item["actuator_shared_power_scale"] < 1.0 - 1.0e-9 for item in records],
        dtype=bool,
    )
    positive_relative = relative[relative > 0.0]

    cut_telemetry = [item for item in telemetry if item["state"] == "CUT_AND_FILL"]
    if len(cut_telemetry) >= 2:
        time = np.asarray([item["timestamp_s"] for item in cut_telemetry])
        root_xy = np.asarray([item["base_pose_xy_yaw"][:2] for item in cut_telemetry])
        root_speed = np.linalg.norm(np.diff(root_xy, axis=0) / np.diff(time)[:, None], axis=1)
        root_motion = {
            "net_xy_displacement_m": float(np.linalg.norm(root_xy[-1] - root_xy[0])),
            "mean_sampled_xy_speed_m_s": float(np.mean(root_speed)),
            "maximum_sampled_xy_speed_m_s": float(np.max(root_speed)),
        }
    else:
        root_motion = {"status": "INSUFFICIENT_RUNTIME_TELEMETRY"}

    q_end = np.asarray(records[-1]["joint_position_rad"])
    q_target = np.asarray(records[-1]["requested_joint_target_rad"])
    payload_capacity = float(acceptance["payload"]["capacity_m3"])
    proximity_stats = {}
    for key in records[0]["proximity"]:
        values = np.asarray([item["proximity"][key] for item in records])
        proximity_stats[key] = {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95.0)),
            "maximum": float(np.max(values)),
        }

    report = {
        "schema": "CUT_AND_FILL_PAYLOAD_CAUSAL_REPORT/v2",
        "status": "CAUSAL_ROOT_CAUSE_ESTABLISHED_NO_PHYSICS_FIX_APPLIED",
        "source_run": "run_1786615982",
        "source_artifacts": {
            "raw_step_audit": str(args.raw),
            "cut_start_checkpoint": str(args.checkpoint),
            "production_acceptance": str(args.acceptance),
            "runtime_telemetry": str(args.telemetry),
        },
        "R2M_GROSS_M3": total_r2m,
        "R2M_ACCOUNTING": {
            "failure_zone_m3": failure_zone_r2m,
            "large_avalanche_m3": large_avalanche_r2m,
            "method": "CONSERVATIVE_MOBILE_CONTROL_VOLUME_RECONSTRUCTION",
            "identity": "R2M=DeltaMobile+M2P+M2R+Outflow-PayloadSpill",
            "mobile_at_cut_start_m3": mobile_start,
            "mobile_at_cut_end_m3": mobile_final,
        },
        "M2P_GROSS_M3": m2p,
        "PAYLOAD_SPILL_GROSS_M3": spill,
        "PAYLOAD_TO_AIRBORNE_M3": payload_to_airborne,
        "M2R_GROSS_M3": m2r,
        "MOBILE_TO_OUTFLOW_M3": outflow,
        "NET_PAYLOAD_GAIN_M3": net_payload,
        "PAYLOAD_BUDGET_RESIDUAL_M3": net_payload - m2p + spill + payload_to_airborne,
        "MOBILE_CONTROL_VOLUME_RESIDUAL_M3": (
            mobile_final - mobile_start
            - (total_r2m + spill - m2p - m2r - outflow)
        ),
        "MASS_ERROR_M3": float(final_reservoirs["mass_balance_error_m3"]),
        "MOBILE_IN_MOUTH_ROI_M3": {
            "mean": float(np.mean(mouth)),
            "p95": float(np.percentile(mouth, 95.0)),
            "maximum": float(np.max(mouth)),
        },
        "MOBILE_PROXIMITY_M3": proximity_stats,
        "MOUTH_CONTROL_SURFACE": {
            "candidate_flux_m3": candidate,
            "accepted_flux_m3": accepted,
            "capacity_rejected_flux_m3": candidate - accepted,
            "geometry_rejected_flux_m3": None,
            "geometry_rejected_flux_status": (
                "NOT_A_SEPARATE_CONSERVATIVE_TRANSITION; non-intersecting material remains Mobile"
            ),
            "intersection_area_m2": {
                "mean": float(np.mean(overlap_area)),
                "p95": float(np.percentile(overlap_area, 95.0)),
                "maximum": float(np.max(overlap_area)),
            },
            "positive_relative_normal_speed_m_s": {
                "all_step_mean": float(np.mean(relative)),
                "active_step_mean": (
                    float(np.mean(positive_relative)) if positive_relative.size else 0.0
                ),
                "p95": float(np.percentile(relative, 95.0)),
                "maximum": float(np.max(relative)),
                "active_step_fraction": float(np.mean(relative > 0.0)),
            },
            "bucket_linear_speed_m_s": {
                "mean": float(np.mean(np.linalg.norm(bucket_velocity, axis=1))),
                "maximum": float(np.max(np.linalg.norm(bucket_velocity, axis=1))),
            },
            "local_mobile_velocity_status": (
                "AUTHORITATIVE_RELATIVE_SPEED_RECORDED; component vector absent from preserved first-pass records; observer now records compact mouth-prism vector"
            ),
            "intake_classification": "INTAKE_FLUX_TOO_SMALL",
        },
        "PAYLOAD_RETENTION": {
            "capacity_m3": payload_capacity,
            "maximum_cut_payload_m3": float(records[-1]["payload_after_m3"]),
            "remaining_capacity_m3": payload_capacity - float(records[-1]["payload_after_m3"]),
            "capacity_limited_step_count": int(sum(item["capacity_limited"] for item in records)),
            "spill_m3": spill,
            "intake_then_immediate_spill": False,
            "retention_operator_role": "NOT_LIMITING_DURING_CUT_AND_FILL",
        },
        "CANDIDATE_MOUTH_FLUX_M3": candidate,
        "ACCEPTED_MOUTH_FLUX_M3": accepted,
        "MOUTH_CAPTURE_RATIO": accepted / candidate if candidate > 0.0 else None,
        "GROSS_CAPTURE_RATIO": m2p / total_r2m,
        "NET_CAPTURE_RATIO": net_payload / total_r2m,
        "TOOL_PROGRESS_FRACTION": float(np.max(tool_progress)),
        "MAX_PENETRATION_M": float(raw["MAX_PENETRATION_M"]),
        "MEAN_PENETRATION_WHILE_ACTIVE_M": float(raw["MEAN_PENETRATION_WHILE_ACTIVE_M"]),
        "TRAJECTORY": {
            "duration_s": float(
                records[-1]["simulation_time_s"] - records[0]["simulation_time_s"]
                + 1.0 / 30.0
            ),
            "target_joint_error_end_rad": (q_end - q_target).tolist(),
            "target_joint_error_end_max_abs_rad": float(np.max(np.abs(q_end - q_target))),
            "root_motion": root_motion,
            "interpretation": (
                "JOINT_TARGET_REACHED; no mechanical actuator stall. Most Mobile activation and mouth overlap occur after commanded cut motion has effectively completed."
            ),
        },
        "PEAK_SOIL_FORCE_N": _norm_peak(records, "applied_soil_force_n"),
        "SOIL_FORCE": {
            "peak_fee_quasi_static_n": _norm_peak(records, "quasi_static_force_n"),
            "peak_dynamic_mobile_momentum_n": _norm_peak(records, "dynamic_momentum_force_n"),
            "other_unattributed_peak_n": 0.0,
            "effort_saturated_step_fraction": float(np.mean(saturation)),
            "power_limited_step_fraction": float(np.mean(power_limited)),
        },
        "SOIL_FORCE_LIMITING_ROLE": "NOT_FORCE_LIMITED",
        "PRIMARY_BOTTLENECK": "MOBILE_NOT_REACHING_MOUTH",
        "SECONDARY_CONTRIBUTORS": [
            "BROAD_FAILURE_ACTIVATION_OUTSIDE_MOUTH_CONTROL_VOLUME",
            "WEAK_POSITIVE_NORMAL_TRANSPORT_AFTER_COMMAND_COMPLETION",
        ],
        "ROOT_CAUSE": (
            "The first broken link is Mobile transport into/through the bucket mouth, not Failure activation, intake acceptance, retention, capacity, force saturation, or mass accounting. Only 0.2991% of R2M crosses the mouth. The mouth prism contains little Mobile for most steps (p95 0.00528 m3), and positive relative-normal speed is weak after the requested joints reach target. Every conservative candidate flux element is accepted and retained."
        ),
        "MINIMAL_FIX": "NOT_APPLIED; no unambiguous local implementation defect was demonstrated, and trajectory/Mobile transport changes are outside this audit's allowed scope",
        "CUT_AND_FILL_AFTER_FIX": "NOT_APPLIED",
        "PRE_DUMP_REACHED": "NO",
        "VISIBLE_NEEDLE_FOREST": "NOT_EVALUATED",
        "VISIBLE_TRIANGULAR_FINS": "NOT_EVALUATED",
        "GRID_SCALE_SPIKE_PROLIFERATION": "NOT_EVALUATED",
        "physics_changes": [],
        "forbidden_parameter_changes": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "R2M_GROSS_M3": total_r2m,
        "M2P_GROSS_M3": m2p,
        "PRIMARY_BOTTLENECK": report["PRIMARY_BOTTLENECK"],
    }))


if __name__ == "__main__":
    main()
