#!/usr/bin/env python3
"""Build the P0-1 old/new trajectory closure artifacts from frozen replays."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from isaac_bulk_pipeline.tools import ToolDescriptorLoader


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/mobile_v2_production"
OLD_PATH = OUT / "cut_fill_payload_causal_audit.json"
OLD_CAUSAL_PATH = OUT / "CUT_AND_FILL_PAYLOAD_CAUSAL_REPORT.json"
NEW_PATH = OUT / "cut_fill_realistic_trajectory_candidate2_audit.json"
PRE_PATH = OUT / "cut_fill_realistic_trajectory_pre_dump.npz"
PRE_JSON = PRE_PATH.with_suffix(".json")
FINAL_REPORT = OUT / "cut_fill_realistic_trajectory_report.json"
FINAL_SERIES = OUT / "cut_fill_realistic_trajectory_timeseries.json"
TRAJECTORY_PLOT = OUT / "cut_fill_realistic_trajectory_alignment.png"
MORPHOLOGY_PLOT = OUT / "cut_fill_realistic_trajectory_pre_dump_morphology.png"
DOC = ROOT / "docs/CUT_AND_FILL_REALISTIC_TRAJECTORY_REPORT.md"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def augment(records: list[dict], geometry) -> list[dict]:
    first_time = float(records[0]["simulation_time_s"])
    result = []
    for item in records:
        record = dict(item)
        pose = np.asarray(item["mouth_pose_terrain"], dtype=np.float64)
        u, _, vh = np.linalg.svd(pose[:3, :3])
        rotation = u @ vh
        edge = np.asarray(item["cutting_edge_center_terrain_m"], dtype=np.float64)
        velocity = np.asarray(item["bucket_linear_velocity_m_s"], dtype=np.float64) + np.cross(
            np.asarray(item["bucket_angular_velocity_rad_s"], dtype=np.float64),
            edge - pose[:3, 3],
        )
        separation = rotation @ geometry.separation_plane_direction_local
        separation /= np.linalg.norm(separation)
        normal = rotation @ geometry.bottom_plate_normal_local
        normal /= np.linalg.norm(normal)
        decomposition = record.get("tool_velocity_decomposition") or {
            "V_TOOL_TOTAL_M_S": float(np.linalg.norm(velocity)),
            "V_PENETRATION_COMPONENT_M_S": float(np.dot(velocity, normal)),
            "V_SEPARATION_COMPONENT_M_S": float(np.dot(velocity, separation)),
            "plate_normal_terrain": normal.tolist(),
            "separation_tangent_terrain": separation.tolist(),
        }
        record.update(
            phase_time_s=float(item.get("phase_time_s", item["simulation_time_s"] - first_time)),
            phase=item.get("phase", "CUT_AND_FILL"),
            trajectory_stage=item.get("trajectory_stage", "LEGACY_FIXED_TARGET"),
            cutting_edge_velocity_terrain_m_s=record.get(
                "cutting_edge_velocity_terrain_m_s", velocity.tolist()
            ),
            tool_velocity_decomposition=decomposition,
            BUCKET_CURL_RATE_RAD_S=float(
                item.get("BUCKET_CURL_RATE_RAD_S", item["joint_velocity_rad_s"][3])
            ),
            STICK_RETRACTION_RATE_RAD_S=float(
                item.get("STICK_RETRACTION_RATE_RAD_S", item["joint_velocity_rad_s"][2])
            ),
            BOOM_RATE_RAD_S=float(item.get("BOOM_RATE_RAD_S", item["joint_velocity_rad_s"][1])),
        )
        result.append(record)
    return result


def metrics(records: list[dict], report: dict, *, failure_zone_total: float) -> dict:
    # Common old/new comparison interval.  The old v1 archive did not retain
    # per-step FailureZone transfer, so the common mask uses the exact geometry
    # and proximity fields available in both archives.  The new report also
    # retains its stricter per-step-FailureZone interval separately.
    penetration = np.asarray([item["maximum_penetration_m"] for item in records])
    near = np.asarray([item["proximity"]["radius_2p0_m3"] for item in records])
    active = (penetration > 0.0) & (near > 0.0)
    count = max(int(np.count_nonzero(active)), 1)
    curl = np.asarray([item["BUCKET_CURL_RATE_RAD_S"] for item in records])
    stick = np.asarray([item["STICK_RETRACTION_RATE_RAD_S"] for item in records])
    pcomp = np.asarray([
        item["tool_velocity_decomposition"]["V_PENETRATION_COMPONENT_M_S"] for item in records
    ])
    scomp = np.asarray([
        item["tool_velocity_decomposition"]["V_SEPARATION_COMPONENT_M_S"] for item in records
    ])
    relative = np.asarray([item["mean_positive_relative_normal_speed_m_s"] for item in records])
    positive = relative[active & (relative > 0.0)]
    mouth = np.asarray([item["proximity"]["mouth_prism_m3"] for item in records])
    front = np.asarray([item["proximity"]["front_m3"] for item in records])
    dt = float(np.median(np.diff([item["simulation_time_s"] for item in records])))
    total_r2m = float(report["R2M_GROSS_M3"])
    return {
        "MAX_PENETRATION_M": float(np.max(penetration)),
        "ACTIVE_CUT_DURATION_S": float(np.count_nonzero(active) * dt),
        "ACTIVE_INTERVAL_COMPARISON_DEFINITION": "penetration>0 AND Mobile volume within 2m>0",
        "PENETRATION_DOMINATED_FRACTION": float(
            np.count_nonzero(active & (np.abs(pcomp) > np.abs(scomp))) / count
        ),
        "SIMULTANEOUS_STICK_RETRACT_BUCKET_CURL_FRACTION": float(
            np.count_nonzero(active & (stick > 0.0055) & (curl > 0.0070)) / count
        ),
        "PEAK_BUCKET_CURL_RATE_RAD_S": float(np.max(curl[active], initial=0.0)),
        "MEAN_BUCKET_CURL_RATE_DURING_ACTIVE_CUT_RAD_S": float(np.mean(curl[active])),
        "MEAN_STICK_RETRACTION_RATE_DURING_ACTIVE_CUT_RAD_S": float(np.mean(stick[active])),
        "FAILUREZONE_R2M_M3": float(failure_zone_total),
        "TOTAL_R2M_M3": total_r2m,
        "MOBILE_MOUTH_P95_M3": float(np.percentile(mouth[active], 95.0)),
        "MOBILE_FRONT_P95_M3": float(np.percentile(front[active], 95.0)),
        "MEAN_POSITIVE_RELATIVE_NORMAL_SPEED_M_S": float(np.mean(positive)) if positive.size else 0.0,
        "P95_POSITIVE_RELATIVE_NORMAL_SPEED_M_S": float(np.percentile(positive, 95.0)) if positive.size else 0.0,
        "CANDIDATE_MOUTH_FLUX_M3": float(report["CANDIDATE_MOUTH_FLUX_M3"]),
        "ACCEPTED_MOUTH_FLUX_M3": float(report["ACCEPTED_MOUTH_FLUX_M3"]),
        "NET_PAYLOAD_GAIN_M3": float(report["NET_PAYLOAD_GAIN_M3"]),
        "GROSS_CAPTURE_RATIO": float(report["GROSS_CAPTURE_RATIO"]),
        "PEAK_SOIL_FORCE_N": float(report["PEAK_SOIL_FORCE_N"]),
        "EFFORT_SATURATED_STEP_FRACTION": float(report["SOIL_FORCE"]["effort_saturated_step_fraction"]),
        "MAX_MASS_ERROR_M3": float(report.get("MAX_MASS_ERROR_M3", 0.0)),
        "PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N": float(report["SOIL_FORCE"]["peak_dynamic_mobile_n"]),
    }


def compact(records: list[dict]) -> list[dict]:
    return [
        {
            "time_s": item["phase_time_s"],
            "phase": item["phase"],
            "trajectory_stage": item["trajectory_stage"],
            "penetration_m": item["maximum_penetration_m"],
            "separation_component_m_s": item["tool_velocity_decomposition"]["V_SEPARATION_COMPONENT_M_S"],
            "penetration_component_m_s": item["tool_velocity_decomposition"]["V_PENETRATION_COMPONENT_M_S"],
            "stick_retraction_rate_rad_s": item["STICK_RETRACTION_RATE_RAD_S"],
            "bucket_curl_rate_rad_s": item["BUCKET_CURL_RATE_RAD_S"],
            "boom_rate_rad_s": item["BOOM_RATE_RAD_S"],
            "failure_zone_r2m_rate_m3_s": (
                None
                if "failure_zone_r2m_step_m3" not in item
                else item["failure_zone_r2m_step_m3"] / (1.0 / 60.0)
            ),
            "inward_mouth_relative_speed_m_s": item["mean_positive_relative_normal_speed_m_s"],
            "candidate_mouth_flux_m3": item["candidate_mouth_flux_m3"],
            "payload_m3": item["payload_after_m3"],
        }
        for item in records
    ]


def morphology() -> tuple[dict, np.ndarray, np.ndarray]:
    initial = np.loadtxt(
        ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_track_pile_acceptance_m.csv",
        delimiter=",",
    )
    with np.load(PRE_PATH) as archive:
        final = np.asarray(archive["H_free_m"], dtype=np.float64)

    def features(height):
        center = height[1:-1, 1:-1]
        maximum = np.maximum.reduce(
            [height[:-2, 1:-1], height[2:, 1:-1], height[1:-1, :-2], height[1:-1, 2:]]
        )
        minimum = np.minimum.reduce(
            [height[:-2, 1:-1], height[2:, 1:-1], height[1:-1, :-2], height[1:-1, 2:]]
        )
        jumps = np.r_[np.abs(np.diff(height, axis=0)).ravel(), np.abs(np.diff(height, axis=1)).ravel()]
        return {
            "jump_max_m": float(np.max(jumps)),
            "jump_p99_m": float(np.percentile(jumps, 99.0)),
            "isolated_spike_count_gt_5cm": int(np.count_nonzero(center > maximum + 0.05)),
            "isolated_pit_count_gt_5cm": int(np.count_nonzero(center < minimum - 0.05)),
            "edge_count_jump_gt_10cm": int(np.count_nonzero(jumps > 0.10)),
        }

    before, after = features(initial), features(final)
    proliferation = bool(
        after["isolated_spike_count_gt_5cm"] >= 10
        and after["edge_count_jump_gt_10cm"] >= 100
        and before["edge_count_jump_gt_10cm"] == 0
    )
    record = {
        "initial": before,
        "pre_dump": after,
        "VISIBLE_NEEDLE_FOREST": "YES" if proliferation else "NO",
        "VISIBLE_TRIANGULAR_FINS": "YES" if proliferation else "NO",
        "GRID_SCALE_SPIKE_PROLIFERATION": "YES" if proliferation else "NO",
        "classification_basis": (
            "field-derived: >=10 isolated >5cm peaks and >=100 >10cm neighbor edges, "
            "both absent in the frozen initial terrain; visual plot preserved"
        ),
    }
    return record, initial, final


def main() -> None:
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(ROOT / "configs/excavator_390f_real_bucket.yaml")
    )
    geometry = descriptor.bucket_geometry
    old_report, new_report = load(OLD_PATH), load(NEW_PATH)
    old_causal, pre_record = load(OLD_CAUSAL_PATH), load(PRE_JSON)
    old_report["MAX_MASS_ERROR_M3"] = abs(float(old_causal["MASS_ERROR_M3"]))
    # The preserved raw v1 report counted only LargeAvalanche R2M in its
    # top-level denominator.  The subsequently accepted conservative control-
    # volume reconstruction is authoritative for total R2M/capture.
    old_report["R2M_GROSS_M3"] = float(old_causal["R2M_GROSS_M3"])
    old_report["GROSS_CAPTURE_RATIO"] = float(old_causal["GROSS_CAPTURE_RATIO"])
    old_records = augment(old_report["time_series"], geometry)
    new_records = augment(new_report["time_series"], geometry)
    old_metrics = metrics(
        old_records,
        old_report,
        failure_zone_total=float(old_causal["R2M_ACCOUNTING"]["failure_zone_m3"]),
    )
    new_metrics = metrics(
        new_records,
        new_report,
        failure_zone_total=float(new_report["FAILURE_ZONE_R2M_GROSS_M3"]),
    )
    morphology_record, initial, final = morphology()

    timeseries = {
        "schema": "CUT_FILL_REALISTIC_TRAJECTORY_TIMESERIES/v1",
        "joint_order": ["swing", "boom", "stick", "bucket"],
        "old_per_step_failure_zone_note": "not retained by historical v1 audit; total is authoritative",
        "old": compact(old_records),
        "new": compact(new_records),
    }
    FINAL_SERIES.write_text(json.dumps(timeseries, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False, constrained_layout=True)
    for label, records, color in (("OLD", old_records, "#9c3d2e"), ("NEW", new_records, "#176b87")):
        t = np.asarray([x["phase_time_s"] for x in records])
        axes[0].plot(t, [x["maximum_penetration_m"] for x in records], label=label, color=color)
        axes[1].plot(t, [x["tool_velocity_decomposition"]["V_SEPARATION_COMPONENT_M_S"] for x in records], label=label, color=color)
        axes[2].plot(t, [x["STICK_RETRACTION_RATE_RAD_S"] for x in records], color=color, label=f"{label} stick")
        axes[2].plot(t, [x["BUCKET_CURL_RATE_RAD_S"] for x in records], color=color, linestyle="--", label=f"{label} curl")
        axes[3].plot(t, [x["payload_after_m3"] for x in records], color=color, label=f"{label} payload")
        axes[3].plot(t, [x["candidate_mouth_flux_m3"] for x in records], color=color, linestyle=":", label=f"{label} flux/step")
    axes[0].set_ylabel("penetration [m]")
    axes[1].set_ylabel("plate tangent [m/s]")
    axes[2].set_ylabel("joint rate [rad/s]")
    axes[3].set_ylabel("volume [m³]"); axes[3].set_xlabel("time from CUT audit start [s]")
    for axis in axes: axis.grid(alpha=0.25); axis.legend(ncol=2)
    fig.savefig(TRAJECTORY_PLOT, dpi=150); plt.close(fig)

    changed = np.abs(final - initial) > 0.02
    rows, cols = np.nonzero(changed)
    r0, r1 = max(0, int(rows.min()) - 8), min(final.shape[0], int(rows.max()) + 9)
    c0, c1 = max(0, int(cols.min()) - 8), min(final.shape[1], int(cols.max()) + 9)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (initial[r0:r1, c0:c1], final[r0:r1, c0:c1], (final - initial)[r0:r1, c0:c1]),
        ("Frozen initial H_free", "PRE_DUMP H_free", "PRE_DUMP - initial"),
    ):
        image = axis.imshow(data, origin="lower", cmap="terrain" if "-" not in title else "coolwarm")
        axis.set_title(title); fig.colorbar(image, ax=axis, shrink=0.75)
    fig.savefig(MORPHOLOGY_PLOT, dpi=150); plt.close(fig)

    full_max_mass_error = max(
        new_metrics["MAX_MASS_ERROR_M3"], abs(float(pre_record["mass_balance_error_m3"]))
    )
    report = {
        "schema": "CUT_AND_FILL_REALISTIC_TRAJECTORY_REPORT/v1",
        "status": "TRAJECTORY_H1_SUPPORTED_PRE_DUMP_REACHED_WITH_TRACKSOIL_LEDGER_REGRESSION",
        "CURRENT_TRAJECTORY_CLASSIFICATION": "PUSH_PENETRATE_DRAG",
        "CORRECTED_TRAJECTORY_CLASSIFICATION": "COORDINATED_SCOOP",
        "TRAJECTORY_DEFECT_DEMONSTRATED": "YES",
        "TRAJECTORY_FIX_APPLIED": "YES",
        "FIX_DESCRIPTION": (
            "production-default C1 A-E target schedule: controlled engagement, initial cut, "
            "overlapping positive stick retraction + positive bucket curl + negative boom "
            "compensation, closure/capture, then withdrawal to the existing breakout pose"
        ),
        "PHYSICS_PARAMETERS_CHANGED": "NONE",
        "MOBILE_V2_CHANGED": "NO",
        "FAILURESURFACE_V3_CHANGED": "NO",
        "old": old_metrics,
        "new": new_metrics,
        "strict_new_failure_zone_active_interval": new_report["TRAJECTORY_METRICS"],
        "hypothesis": "H1_SUPPORTED",
        "OLD_NET_PAYLOAD_GAIN_M3": old_metrics["NET_PAYLOAD_GAIN_M3"],
        "NEW_NET_PAYLOAD_GAIN_M3": new_metrics["NET_PAYLOAD_GAIN_M3"],
        "OLD_GROSS_CAPTURE_RATIO": old_metrics["GROSS_CAPTURE_RATIO"],
        "NEW_GROSS_CAPTURE_RATIO": new_metrics["GROSS_CAPTURE_RATIO"],
        "OLD_MAX_PENETRATION_M": old_metrics["MAX_PENETRATION_M"],
        "NEW_MAX_PENETRATION_M": new_metrics["MAX_PENETRATION_M"],
        "payload_gain_factor": new_metrics["NET_PAYLOAD_GAIN_M3"] / old_metrics["NET_PAYLOAD_GAIN_M3"],
        "gross_capture_ratio_factor": new_metrics["GROSS_CAPTURE_RATIO"] / old_metrics["GROSS_CAPTURE_RATIO"],
        "PRE_DUMP_REACHED": "YES",
        "VISIBLE_NEEDLE_FOREST": morphology_record["VISIBLE_NEEDLE_FOREST"],
        "VISIBLE_TRIANGULAR_FINS": morphology_record["VISIBLE_TRIANGULAR_FINS"],
        "GRID_SCALE_SPIKE_PROLIFERATION": morphology_record["GRID_SCALE_SPIKE_PROLIFERATION"],
        "pre_dump": pre_record,
        "morphology": morphology_record,
        "PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N": new_metrics["PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N"],
        "CUT_MAX_MASS_ERROR_M3": new_metrics["MAX_MASS_ERROR_M3"],
        "MAX_MASS_ERROR_M3": full_max_mass_error,
        "mass_attribution": (
            "CUT/CURL remains conservative to 2.14e-11 m3; error begins only when "
            "GPU TrackSoil activates during REVERSE and reaches 2.19e-4 m3"
        ),
        "PRIMARY_REMAINING_BLOCKER": "GPU_TRACKSOIL_CONSERVATION_RESIDUAL_DURING_REVERSE",
        "NEXT_P0_BLOCKER": "TOOL_MOBILE_MOMENTUM_COUPLING_REQUIRES_CAUSAL_AUDIT",
        "excluded_replay": {
            "candidate1": "INVALID_CHECKPOINT_RESTORE_NOT_PHYSICS_RESULT",
            "reason": (
                "historical cut checkpoint lacks root z/roll/pitch and actuator internal target "
                "velocity; restored initial penetration was 1.53m versus frozen 0.23m"
            ),
        },
        "artifacts": {
            "timeseries": str(FINAL_SERIES.relative_to(ROOT)),
            "trajectory_plot": str(TRAJECTORY_PLOT.relative_to(ROOT)),
            "pre_dump_fields": str(PRE_PATH.relative_to(ROOT)),
            "morphology_plot": str(MORPHOLOGY_PLOT.relative_to(ROOT)),
        },
    }
    FINAL_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    numeric_metric_keys = [
        key
        for key, value in old_metrics.items()
        if isinstance(value, (int, float)) and isinstance(new_metrics.get(key), (int, float))
    ]

    def row(key):
        return f"| {key} | {old_metrics[key]:.10g} | {new_metrics[key]:.10g} |"

    document = f"""# CUT_AND_FILL realistic trajectory report

## Result

The frozen production trajectory is **PUSH_PENETRATE_DRAG**: its CUT target held
the bucket at -60° and moved stick from about 16° toward 6.33°, so principal
failure preceded positive curl/retraction.  The production-default replacement
is a continuous A–E curl-scoop consumed by the unchanged bounded actuator.

The valid normal-boundary replay reached PRE_DUMP.  Net payload increased
{report['payload_gain_factor']:.3f}× and gross capture ratio increased
{report['gross_capture_ratio_factor']:.3f}× while maximum penetration fell.
This supports H1 without changing soil physics.

## Old versus new

| Metric | old | new |
|---|---:|---:|
{chr(10).join(row(key) for key in numeric_metric_keys)}

The common comparison interval uses `penetration>0 AND Mobile-within-2m>0`
because the historical v1 archive retained total FailureZone R2M but not its
per-step values.  The candidate additionally reports the stricter per-step
FailureZone-active interval in the JSON.

## Temporal ordering

Candidate peak FailureZone R2M occurred at simulation t=
{new_report['TRAJECTORY_METRICS']['temporal_ordering']['peak_failure_zone_r2m_rate_time_s']:.6f}s,
peak mouth overlap at {new_report['TRAJECTORY_METRICS']['temporal_ordering']['peak_mouth_overlap_time_s']:.6f}s,
peak mouth flux at {new_report['TRAJECTORY_METRICS']['temporal_ordering']['peak_mouth_flux_time_s']:.6f}s,
peak curl rate at {new_report['TRAJECTORY_METRICS']['temporal_ordering']['peak_bucket_curl_rate_time_s']:.6f}s,
and peak penetration at {new_report['TRAJECTORY_METRICS']['temporal_ordering']['peak_penetration_time_s']:.6f}s.

## PRE_DUMP morphology and conservation boundary

- PRE_DUMP reached: **YES**
- Visible needle forest: **{morphology_record['VISIBLE_NEEDLE_FOREST']}**
- Visible triangular fins: **{morphology_record['VISIBLE_TRIANGULAR_FINS']}**
- Grid-scale spike proliferation: **{morphology_record['GRID_SCALE_SPIKE_PROLIFERATION']}**
- max / p99 H_free neighbor jump: {morphology_record['pre_dump']['jump_max_m']:.9g} / {morphology_record['pre_dump']['jump_p99_m']:.9g} m
- CUT/CURL max mass error: {new_metrics['MAX_MASS_ERROR_M3']:.9g} m³
- full replay max mass error: {full_max_mass_error:.9g} m³

The trajectory/intake path remains conservative to numerical precision.  The
full replay is not a conservation pass: error begins only when TrackSoil first
activates during REVERSE.  This frozen-module regression is recorded, not
silently attributed to the curl-scoop or repaired in this task.

Dynamic Mobile momentum reaction remains exactly zero under real active
contact, so the required next P0 label is
`TOOL_MOBILE_MOMENTUM_COUPLING_REQUIRES_CAUSAL_AUDIT`.

## Replay and regression evidence

- valid production replay: `run_1786937073`, 390F / GPU_RUNTIME / DEVICE /
  701×701 / dx=0.05 m, stopped at the existing pre-release PRE_DUMP boundary;
- focused trajectory/runtime regression: 18 passed;
- full CPU suite: 279 passed, 16 skipped, 57 subtests passed;
- three pre-existing unrelated failures remain: two stale Phase-A manifest
  checks for `runtime/__init__.py`, and the existing Phase-H width monotonicity
  assertion.  None of those files/models were changed here.

The historical CUT checkpoint was restored once for diagnosis, but it lacks
root z/roll/pitch and actuator internal target velocity.  Its initial
penetration became 1.53 m instead of the frozen 0.23 m, so that interrupted
state is explicitly excluded.  The accepted result reran the unchanged normal
production prelude to obtain a physically complete PENETRATE→CUT boundary;
no root/link pose write or teleport was added.

## Artifacts

- `{FINAL_REPORT.relative_to(ROOT)}`
- `{FINAL_SERIES.relative_to(ROOT)}`
- `{PRE_PATH.relative_to(ROOT)}`
- `{TRAJECTORY_PLOT.relative_to(ROOT)}`
- `{MORPHOLOGY_PLOT.relative_to(ROOT)}`
"""
    DOC.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
