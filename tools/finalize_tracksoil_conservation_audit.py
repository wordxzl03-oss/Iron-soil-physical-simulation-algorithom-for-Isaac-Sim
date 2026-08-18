#!/usr/bin/env python3
"""Finalize the frozen P0-2A production ledger into concise artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/mobile_v2_production"
TIMESERIES = OUT / "tracksoil_conservation_timeseries.json"
REPORT_JSON = OUT / "tracksoil_conservation_causal_report.json"
FIRST_BAD_JSON = OUT / "tracksoil_first_bad_step.json"
DOC = ROOT / "docs/TRACKSOIL_CONSERVATION_CAUSAL_REPORT.md"
BEFORE_MAX = 0.00021906799202042748


def main() -> None:
    payload = json.loads(TIMESERIES.read_text(encoding="utf-8"))
    records = payload["records"]
    run_dir = Path(payload["run_dir"])
    events = json.loads((run_dir / "track_soil_events.json").read_text(encoding="utf-8"))
    active = [row for row in records if row["tracksoil_control_volume"]["active"]]
    if len(active) != len(events):
        raise RuntimeError(
            f"active TrackSoil rows/events mismatch: {len(active)} != {len(events)}"
        )
    for row in records:
        declared = row["declared_transfers_m3"]
        observed_m2r = float(
            declared.get(
                "avalanche_persistence_observed_m2r_increment_m3",
                declared.get("large_avalanche_m2r_m3", 0.0),
            )
        )
        declared["large_avalanche_m2r_m3"] = 0.0
        declared[
            "avalanche_persistence_observed_m2r_increment_m3"
        ] = observed_m2r
    # This frozen reverse is straight travel: left/right commands are equal.
    # That makes combined exact Triangle-A-C footprint area sufficient to
    # reconstruct the actual two-track sink request without a terrain read.
    for row, event in zip(active, events, strict=True):
        if abs(row["simulation_time_s"] - event["timestamp_s"]) > 1.0e-6:
            raise RuntimeError("TrackSoil event timestamp mismatch")
        left = float(event["applied_left_command"])
        right = float(event["applied_right_command"])
        if abs(left - right) > 1.0e-12:
            raise RuntimeError("frozen replay is not symmetric straight-track travel")
        slip = abs(left) * 1.20
        slip_effect = max(slip - 0.02, 0.0)
        requested_depth = min(0.004, 0.012 * (1.0 / 30.0) * (1.0 + 0.8 * slip_effect))
        control = row["tracksoil_control_volume"]
        requested = requested_depth * float(control["footprint_area_m2"])
        accepted = float(control["accepted_r2m_m3"])
        control["requested_r2m_m3"] = requested
        control["rejected_r2m_m3"] = max(requested - accepted, 0.0)

    baseline = []
    for row in records:
        if row["tracksoil_control_volume"]["active"]:
            break
        baseline.append(row)
    floor = max(
        abs(float(boundary["state"]["mass_error_m3"]))
        for row in baseline
        for boundary in row["operator_boundaries"]
    )
    threshold = max(100.0 * floor, 1.0e-10)
    ordered = [
        {"row": row, "boundary": boundary}
        for row in records
        for boundary in row["operator_boundaries"][1:]
    ]
    first_index = next(
        index
        for index, item in enumerate(ordered)
        if abs(float(item["boundary"]["state"]["mass_error_m3"])) > threshold
    )
    first = ordered[first_index]
    last = ordered[first_index - 1]
    first_row = first["row"]
    first_boundary = first["boundary"]
    last_boundary = last["boundary"]
    final_error = abs(float(records[-1]["reservoirs_m3"]["mass_error_m3"]))
    maximum_error = max(
        abs(float(row["reservoirs_m3"]["mass_error_m3"])) for row in records
    )
    track_total = sum(
        float(row["declared_transfers_m3"]["tracksoil_r2m_m3"])
        for row in records
    )
    payload_volume = float(records[-1]["reservoirs_m3"]["payload"])
    initial_total = float(records[-1]["reservoirs_m3"]["initial_total_m3"])
    mobile_residual_sum = sum(
        float(row["declared_transfers_m3"]["mobile_transport_mass_residual_m3"])
        for row in records
    )
    track_residual_sum = sum(
        float(row["tracksoil_control_volume"]["tracksoil_local_residual_m3"])
        for row in records
    )
    track_residual_max = max(
        abs(float(row["tracksoil_control_volume"]["tracksoil_local_residual_m3"]))
        for row in records
    )
    boundary_sums: dict[str, float] = {}
    boundary_max: dict[str, float] = {}
    for row in records:
        for boundary in row["operator_boundaries"][1:]:
            label = boundary["label"]
            value = float(boundary["delta_from_previous"]["total"])
            boundary_sums[label] = boundary_sums.get(label, 0.0) + value
            boundary_max[label] = max(boundary_max.get(label, 0.0), abs(value))

    conclusions = {
        "FIRST_BAD_STEP_FOUND": "YES",
        "FIRST_BAD_PHASE": first_row["phase"],
        "FIRST_BAD_TIME_S": first_row["simulation_time_s"],
        "LAST_GOOD_MASS_ERROR_M3": last_boundary["state"]["mass_error_m3"],
        "FIRST_BAD_MASS_ERROR_M3": first_boundary["state"]["mass_error_m3"],
        "FIRST_BAD_DELTA_MASS_ERROR_M3": first_boundary["delta_from_previous"]["mass_error"],
        "TRACKSOIL_ACTIVE_AT_FIRST_BAD_STEP": "YES",
        "ROOT_CAUSE_CLASS": "PHYSICS_STATE_NONCONSERVATION",
        "ROOT_CAUSE": (
            "Production Mobile V2 applies equal-and-opposite height increments at each face, "
            "which conserves an unweighted vertex sum but not the authoritative Triangle-A-C "
            "control-volume weights at the terrain boundary; TrackSoil exposes Mobile there, "
            "and the first Mobile transport step loses weighted material with no outflow."
        ),
        "RESPONSIBLE_OPERATOR": "WarpProductionMobileV2Solver",
        "RESPONSIBLE_FILE_FUNCTION": (
            "src/isaac_bulk_pipeline/experimental/mobile_v2_warp.py::_kernels.faces / "
            "apply_update_and_sources, called by "
            "src/isaac_bulk_pipeline/bulk_interaction/warp_mobile_v2.py::"
            "WarpProductionMobileV2Solver.step_resident"
        ),
        "FIX_APPLIED": "NO",
        "FIX_DESCRIPTION": (
            "None. A valid repair requires a conservative dual-control-volume/weight-aware "
            "Mobile V2 face update; Mobile V2 was explicitly frozen for P0-2A. No TrackSoil "
            "or ledger workaround was applied."
        ),
        "PHYSICS_PARAMETERS_CHANGED": "NONE",
        "MOBILE_V2_CHANGED": "NO",
        "FAILURESURFACE_V3_CHANGED": "NO",
        "CURL_SCOOP_TRAJECTORY_CHANGED": "NO",
        "BEFORE_MAX_MASS_ERROR_M3": BEFORE_MAX,
        "AFTER_MAX_MASS_ERROR_M3": None,
        "RELATIVE_ERROR_VS_DISTURBED_MATERIAL_BEFORE": final_error / track_total,
        "RELATIVE_ERROR_VS_DISTURBED_MATERIAL_AFTER": None,
        "PRE_DUMP_REACHED_AFTER_FIX": "NOT_RUN",
        "VISIBLE_NEEDLE_FOREST": "NOT_EVALUATED",
        "PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N": "UNCHANGED",
        "PRIMARY_REMAINING_BLOCKER": (
            "MOBILE_V2_BOUNDARY_CONTROL_VOLUME_WEIGHT_MISMATCH"
        ),
    }
    report = {
        "schema": "P0-2A_TRACKSOIL_CONSERVATION_CAUSAL_REPORT/v1",
        "status": "AUDIT_ONLY_CAUSALITY_ESTABLISHED_FIX_OUTSIDE_FROZEN_SCOPE",
        "run_dir": str(run_dir),
        "production_replay": {
            "vehicle": "390F",
            "grid_shape_yx": [701, 701],
            "dx_m": 0.05,
            "runtime_backend": "GPU_RUNTIME",
            "state_authority": "DEVICE",
            "pre_dump_reached": True,
            "physics_parameters_changed": [],
        },
        "locator": {
            "numerical_floor_m3": floor,
            "first_bad_threshold_m3": threshold,
            "last_good": last,
            "first_bad": first,
        },
        "first_bad_control_volume": first_row["tracksoil_control_volume"],
        "first_bad_declared_transfers_m3": first_row["declared_transfers_m3"],
        "normalization": {
            "absolute_final_residual_m3": final_error,
            "absolute_maximum_residual_m3": maximum_error,
            "initial_total_m3": initial_total,
            "tracksoil_disturbed_volume_m3": track_total,
            "payload_volume_m3": payload_volume,
            "epsilon_initial": final_error / initial_total,
            "epsilon_disturbed": final_error / track_total,
            "epsilon_payload": final_error / payload_volume,
        },
        "causal_closure": {
            "mobile_transport_residual_sum_m3": mobile_residual_sum,
            "final_mass_error_m3": records[-1]["reservoirs_m3"]["mass_error_m3"],
            "difference_m3": mobile_residual_sum
            - float(records[-1]["reservoirs_m3"]["mass_error_m3"]),
            "tracksoil_local_residual_sum_m3": track_residual_sum,
            "tracksoil_local_residual_max_abs_m3": track_residual_max,
            "boundary_delta_sums_m3": boundary_sums,
            "boundary_delta_max_abs_m3": boundary_max,
        },
        "gpu_authority_audit": {
            "actual_physics_state_nonconservation": True,
            "ledger_telemetry_mismatch": False,
            "stale_cpu_gpu_alias": False,
            "double_application": False,
            "reduction_precision_root_cause": False,
            "z_base_tracksoil_access": "READ_ONLY",
            "b_eff_tracksoil_access": "ONE_DEVICE_WRITE",
            "mobile_tracksoil_access": "ONE_DEVICE_WRITE",
            "legacy_resting_alias": "SAME_WARP_ALLOCATION_AS_B_EFF",
            "kernel_ordering": "SYNCHRONIZED_AT_EACH_SCALAR_AUDIT_BOUNDARY",
        },
        "gpu_boundary_probe": json.loads(
            (OUT / "tracksoil_mobile_boundary_probe.json").read_text(
                encoding="utf-8"
            )
        ),
        "conclusions": conclusions,
        "timeseries": str(TIMESERIES),
        "first_bad_artifact": str(FIRST_BAD_JSON),
    }
    payload["records"] = records
    TIMESERIES.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    FIRST_BAD_JSON.write_text(
        json.dumps(
            {
                "schema": report["schema"],
                "run_dir": str(run_dir),
                "numerical_floor_m3": floor,
                "first_bad_threshold_m3": threshold,
                "last_good": last,
                "first_bad": first,
                "first_bad_control_volume": first_row["tracksoil_control_volume"],
                "first_bad_declared_transfers_m3": first_row["declared_transfers_m3"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    DOC.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict) -> str:
    c = report["conclusions"]
    n = report["normalization"]
    causal = report["causal_closure"]
    first = report["locator"]["first_bad"]
    control = report["first_bad_control_volume"]
    probe = report["gpu_boundary_probe"]
    return f"""# TrackSoil conservation causal report

## Outcome

The first bad production boundary is **Mobile V2 transport, not TrackSoil**.
The frozen 390F / 701×701 / 0.05 m / GPU_RUNTIME / DEVICE replay reached
PRE_DUMP without changing any physics parameter. No conservation fix was
applied because the causal repair is a Mobile V2 control-volume change, and
Mobile V2 is explicitly frozen in P0-2A.

## First bad ledger boundary

- physics step: `{first['row']['physics_step']}`
- simulation time: `{first['row']['simulation_time_s']:.12f} s`
- phase: `{first['row']['phase']}`
- boundary: `{first['boundary']['label']}`
- last-good mass error: `{c['LAST_GOOD_MASS_ERROR_M3']:.12g} m³`
- first-bad mass error: `{c['FIRST_BAD_MASS_ERROR_M3']:.12g} m³`
- first-bad delta: `{c['FIRST_BAD_DELTA_MASS_ERROR_M3']:.12g} m³`
- locator threshold: `{report['locator']['first_bad_threshold_m3']:.12g} m³`

At the immediately preceding TrackSoil boundary, the accepted R→M transfer was
`{control['accepted_r2m_m3']:.12g} m³`: Resting lost
`{control['resting_volume_removed_m3']:.12g} m³`, Mobile gained
`{control['mobile_volume_added_m3']:.12g} m³`, and the local residual was only
`{control['tracksoil_local_residual_m3']:.12g} m³`.

At `AFTER_MOBILE_TRANSPORT`, the observed authoritative delta was:

```text
ΔV_R = {first['boundary']['delta_from_previous']['resting']:.12g} m³
ΔV_M = {first['boundary']['delta_from_previous']['mobile']:.12g} m³
ΔV_P = 0
ΔV_A = 0
ΔV_O = 0
ΔV_total = {first['boundary']['delta_from_previous']['total']:.12g} m³
```

The Mobile step itself reports the same unmatched transport residual. Across
all 398 audited post-cut steps, cumulative Mobile transport residual is
`{causal['mobile_transport_residual_sum_m3']:.12g} m³`, while final ledger
error is `{causal['final_mass_error_m3']:.12g} m³`; their difference is only
`{causal['difference_m3']:.12g} m³`.

## Cause

`experimental/mobile_v2_warp.py::_kernels.faces` applies equal-and-opposite
**height** increments to adjacent vertex samples. That conserves an unweighted
sum of heights. The authoritative terrain/material measure instead uses exact
Triangle-A-C vertex control areas from `DeviceBulkState._vertex_weights`.
Interior weights are uniform, but edge/corner weights are smaller. TrackSoil
creates Mobile on the real heightmap boundary (first active footprint: 383
vertices, exact weighted area `{control['footprint_area_m2']:.12g} m²`), so a
Mobile face transfer across unequal weights changes authoritative material.

This is not stale host data, a second terrain mirror, double application, or
reduction noise: `resting` aliases `b_eff` to the same Warp allocation;
TrackSoil reads `z_base`, writes `b_eff` once and `mobile` once; every audited
boundary synchronizes before reduction. Reduction scatter is O(1e-11 m³),
whereas the coherent Mobile residual reaches O(1e-4 m³).

The focused CUDA probe independently reproduces the measure mismatch:
TrackSoil boundary R→M residual is
`{probe['tracksoil']['authoritative_weighted_residual_m3']:.12g} m³`; Mobile V2
preserves the uniform measure to
`{probe['mobile_v2_boundary_transport']['uniform_residual_m3']:.12g} m³` but
changes the authoritative Triangle-A-C measure by
`{probe['mobile_v2_boundary_transport']['authoritative_weighted_residual_m3']:.12g} m³`.
That value equals the solver-reported transport residual exactly in the probe.

## Normalized severity

- absolute final residual: `{n['absolute_final_residual_m3']:.12g} m³`
- ε_initial: `{n['epsilon_initial']:.12g}`
- disturbed TrackSoil volume: `{n['tracksoil_disturbed_volume_m3']:.12g} m³`
- ε_disturbed: `{n['epsilon_disturbed']:.12g}` ({100*n['epsilon_disturbed']:.6g}%)
- payload: `{n['payload_volume_m3']:.12g} m³`
- ε_payload: `{n['epsilon_payload']:.12g}` ({100*n['epsilon_payload']:.6g}%)

## Fix decision

`FIX_APPLIED = NO`. Fixing the ledger, globally renormalizing terrain, hiding
the residual, or redepositing TrackSoil material directly would be invalid.
The smallest physical fix is a weight-aware/dual-control-volume Mobile V2 face
transfer. That necessarily changes Mobile V2 and therefore is outside the
frozen P0-2A scope. No after-fix replay was run.

## Mandatory conclusions

```text
FIRST_BAD_STEP_FOUND: {c['FIRST_BAD_STEP_FOUND']}
FIRST_BAD_PHASE: {c['FIRST_BAD_PHASE']}
FIRST_BAD_TIME_S: {c['FIRST_BAD_TIME_S']}
LAST_GOOD_MASS_ERROR_M3: {c['LAST_GOOD_MASS_ERROR_M3']}
FIRST_BAD_MASS_ERROR_M3: {c['FIRST_BAD_MASS_ERROR_M3']}
FIRST_BAD_DELTA_MASS_ERROR_M3: {c['FIRST_BAD_DELTA_MASS_ERROR_M3']}
TRACKSOIL_ACTIVE_AT_FIRST_BAD_STEP: {c['TRACKSOIL_ACTIVE_AT_FIRST_BAD_STEP']}
ROOT_CAUSE_CLASS: {c['ROOT_CAUSE_CLASS']}
ROOT_CAUSE: {c['ROOT_CAUSE']}
RESPONSIBLE_OPERATOR: {c['RESPONSIBLE_OPERATOR']}
RESPONSIBLE_FILE_FUNCTION: {c['RESPONSIBLE_FILE_FUNCTION']}
FIX_APPLIED: {c['FIX_APPLIED']}
FIX_DESCRIPTION: {c['FIX_DESCRIPTION']}
PHYSICS_PARAMETERS_CHANGED: {c['PHYSICS_PARAMETERS_CHANGED']}
MOBILE_V2_CHANGED: {c['MOBILE_V2_CHANGED']}
FAILURESURFACE_V3_CHANGED: {c['FAILURESURFACE_V3_CHANGED']}
CURL_SCOOP_TRAJECTORY_CHANGED: {c['CURL_SCOOP_TRAJECTORY_CHANGED']}
BEFORE_MAX_MASS_ERROR_M3: {c['BEFORE_MAX_MASS_ERROR_M3']}
AFTER_MAX_MASS_ERROR_M3: NOT_RUN
RELATIVE_ERROR_VS_DISTURBED_MATERIAL_BEFORE: {c['RELATIVE_ERROR_VS_DISTURBED_MATERIAL_BEFORE']}
RELATIVE_ERROR_VS_DISTURBED_MATERIAL_AFTER: NOT_RUN
PRE_DUMP_REACHED_AFTER_FIX: {c['PRE_DUMP_REACHED_AFTER_FIX']}
VISIBLE_NEEDLE_FOREST: {c['VISIBLE_NEEDLE_FOREST']}
PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N: {c['PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N']}
PRIMARY_REMAINING_BLOCKER: {c['PRIMARY_REMAINING_BLOCKER']}
```
"""


if __name__ == "__main__":
    main()
