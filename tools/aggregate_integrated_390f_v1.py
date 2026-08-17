"""Aggregate the three real-Isaac 390F comparison runs without upgrading claims."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import hashlib

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "integrated_390f_v1"
CASES = ("A", "B", "C")


def _vector(row: dict[str, str], key: str) -> np.ndarray:
    return np.asarray(json.loads(row[key]), dtype=np.float64)


def _norm_stats(values: list[np.ndarray]) -> dict[str, float]:
    norm = np.asarray([np.linalg.norm(item) for item in values], dtype=np.float64)
    return {
        "mean": float(np.mean(norm)),
        "p95": float(np.percentile(norm, 95.0)),
        "maximum": float(np.max(norm)),
    }


def _write_subset(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def main() -> None:
    summaries: dict[str, dict] = {}
    profiles: dict[str, dict] = {}
    by_case: dict[str, list[dict[str, str]]] = {}
    all_rows: list[dict[str, str]] = []
    for case in CASES:
        case_dir = OUTPUT / f"case_{case}"
        failure = OUTPUT / f"case_{case}_failure.json"
        if failure.exists():
            raise RuntimeError(f"case {case} has failure evidence: {failure}")
        summaries[case] = json.loads((case_dir / "cycle_summary.json").read_text())
        profiles[case] = json.loads((case_dir / "runtime_profile.json").read_text())
        with (case_dir / "three_cycle_log.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 648:
            raise RuntimeError(f"case {case}: expected 648 rows, received {len(rows)}")
        by_case[case] = rows
        all_rows.extend(rows)

    with (OUTPUT / "three_cycle_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    identifiers = [
        "case", "cycle", "phase", "physics_step", "robot_state_timestamp_s",
        "soil_force_timestamp_s", "terrain_state_timestamp_s", "visual_mesh_timestamp_s",
        "contact_surface_timestamp_s",
    ]
    _write_subset(
        OUTPUT / "joint_tip_timeseries.csv",
        all_rows,
        identifiers + [
            "joint_target_rad", "joint_position_rad", "joint_velocity_rad_s",
            "joint_effort_nm", "actuator_effort_command_nm",
            "actuator_target_velocity_rad_s", "actuator_positive_power_w",
            "actuator_shared_power_scale", "bucket_tip_terrain_m",
        ],
    )
    _write_subset(
        OUTPUT / "soil_force_timeseries.csv",
        all_rows,
        identifiers + [
            "computed_soil_force_terrain_n", "quasi_static_force_terrain_n",
            "momentum_force_terrain_n", "applied_soil_force_terrain_n",
            "soil_impulse_terrain_ns", "failure_zone_applicability",
            "failure_zone_candidate_strips", "failure_zone_excluded_strips",
            "material_activation_mode",
        ],
    )
    _write_subset(
        OUTPUT / "payload_timeseries.csv",
        all_rows,
        identifiers + [
            "payload_volume_m3", "payload_mass_kg", "payload_geometric_fill_ratio",
            "payload_com_bucket_m", "payload_phase", "airborne_volume_m3",
        ],
    )
    _write_subset(
        OUTPUT / "mass_balance_timeseries.csv",
        all_rows,
        identifiers + [
            "resting_volume_m3", "mobile_volume_m3", "payload_volume_m3",
            "airborne_volume_m3", "outflow_volume_m3", "mass_balance_error_m3",
        ],
    )

    metrics: dict[str, dict] = {}
    for case, rows in by_case.items():
        applied = [_vector(row, "applied_soil_force_terrain_n") for row in rows]
        computed = [_vector(row, "computed_soil_force_terrain_n") for row in rows]
        payload = np.asarray([float(row["payload_volume_m3"]) for row in rows])
        metrics[case] = {
            "applied_force_norm_n": _norm_stats(applied),
            "computed_force_norm_n": _norm_stats(computed),
            "peak_payload_m3": float(np.max(payload)),
            "final_payload_m3": float(payload[-1]),
            "final_mass_balance_error_m3": summaries[case]["final_mass_balance_error_m3"],
            "total_abs_mechanical_work_j": summaries[case]["total_abs_mechanical_work_j"],
            "rtf": profiles[case]["rtf"],
        }
    baseline_q = np.asarray([_vector(row, "joint_position_rad") for row in by_case["A"]])
    baseline_tip = np.asarray([_vector(row, "bucket_tip_terrain_m") for row in by_case["A"]])
    for case in ("B", "C"):
        q = np.asarray([_vector(row, "joint_position_rad") for row in by_case[case]])
        tip = np.asarray([_vector(row, "bucket_tip_terrain_m") for row in by_case[case]])
        metrics[case]["joint_trajectory_rmse_vs_A_rad"] = float(np.sqrt(np.mean((q - baseline_q) ** 2)))
        metrics[case]["bucket_tip_rmse_vs_A_m"] = float(np.sqrt(np.mean((tip - baseline_tip) ** 2)))

    zero_a = max(metrics["A"]["applied_force_norm_n"].values()) <= 1.0e-12
    b_match = all(
        np.allclose(
            _vector(row, "applied_soil_force_terrain_n"),
            _vector(row, "quasi_static_force_terrain_n"),
            atol=1.0e-8,
            rtol=1.0e-10,
        )
        for row in by_case["B"]
    )
    c_match = all(
        np.allclose(
            _vector(row, "applied_soil_force_terrain_n"),
            _vector(row, "computed_soil_force_terrain_n"),
            atol=1.0e-8,
            rtol=1.0e-10,
        )
        for row in by_case["C"]
    )
    conservation = max(
        summary["final_mass_balance_error_m3"] for summary in summaries.values()
    ) <= 1.0e-8
    actuator_bound = all(
        float(row["actuator_positive_power_w"]) <= 391_000.0 + 1.0e-6
        for row in all_rows
    )
    collision = summaries["C"]["collision_evidence"]
    video_source = OUTPUT / "case_C" / "three_cycle_full_physics.mp4"
    video_manifest_source = OUTPUT / "case_C" / "three_cycle_full_physics.json"
    video_ok = video_source.is_file() and video_source.stat().st_size > 0 and video_manifest_source.is_file()
    full_force_feedback = (
        c_match
        and metrics["C"]["applied_force_norm_n"]["maximum"] > 0.0
        and metrics["C"]["joint_trajectory_rmse_vs_A_rad"] > 0.0
    )
    bulk_chain = all(
        summary["custom_solver_call_count"] > 0
        and summary["minislope_call_count"] == 3
        for summary in summaries.values()
    ) and conservation
    physics = {
        "three_continuous_cycles_no_reset": {
            "status": "PASS" if all(
                item["physics_step_count"] == 648
                and not item["terrain_reset_between_cycles"]
                and not item["vehicle_reset_between_cycles"]
                and not item["payload_reset_between_cycles"]
                for item in summaries.values()
            ) else "FAIL"
        },
        "closed_mass_conservation": {"status": "PASS" if conservation else "FAIL"},
        "failure_mobile_intake_fill_spill_deposition": {
            "status": "PASS" if bulk_chain else "FAIL"
        },
        "abc_force_definition": {"status": "PASS" if zero_a and b_match and c_match else "FAIL"},
        "soil_force_applied_to_real_bucket": {
            "status": "PASS" if c_match and metrics["C"]["applied_force_norm_n"]["maximum"] > 0.0 else "FAIL"
        },
        "load_dependent_vehicle_response": {
            "status": "PASS" if full_force_feedback else "FAIL",
            "joint_trajectory_rmse_vs_no_force_rad": metrics["C"].get("joint_trajectory_rmse_vs_A_rad", 0.0),
            "bucket_tip_rmse_vs_no_force_m": metrics["C"].get("bucket_tip_rmse_vs_A_m", 0.0),
        },
        "actuator_command_power_bound": {"status": "PASS" if actuator_bound else "FAIL"},
        "track_support_collision_enabled": {
            "status": "PASS" if (
                collision["support_ground_collision_api"]
                and all(collision["track_collision_api"])
            ) else "FAIL"
        },
        "track_contact_force_sensor": {
            "status": "DEGRADED",
            "reason": collision["track_contact_sensor_status"],
        },
        "deformable_rigid_collision_pair_disabled": {
            "status": "PASS" if (
                not collision["dynamic_bulk_collision_api"]
                and not collision["bucket_bulk_rigid_pair_possible"]
            ) else "FAIL"
        },
        "finite_width_3d_force": {
            "status": "BLOCKED_FULL_PRIMARY_EQUATION_NOT_RECOVERED",
            "reason": "No gain or guessed width correction was introduced.",
        },
        "cavity_expansion_force": {
            "status": "BLOCKED_FULL_PRIMARY_EQUATION_AND_INPUT_SET_NOT_RECOVERED",
        },
        "second_external_bucket_benchmark": {
            "status": "BLOCKED_MATCHED_PUBLIC_DATASET_NOT_RECOVERED",
            "reason": "The Obermayr blade case remains the only equation-complete external benchmark; no unmatched bucket dataset is labelled as validation.",
        },
        "real_bucket_rake_fee_coverage": {
            "status": "FAIL",
            "reason": "Recorded 390F rake states were outside the implemented Luengo closure; CAD sweep displacement is explicitly force-free.",
        },
        "condition_consistent_iron_ore_tuple": {
            "status": "PARTIAL_STRESS_BRACKETED_NOT_SITE_CALIBRATED",
        },
        "real_390f_mass_com_inertia": {
            "status": "PARTIAL_RUNTIME_MESH_DERIVED_NOT_MANUFACTURER_VALIDATED",
            "reason": "Seven runtime bodies have finite mass/COM/inertia, but the 94 t asset mass exceeds the published machine range and source COM/inertia placeholders were invalid.",
        },
        "hydraulic_cylinder_mapping": {
            "status": "BLOCKED_CYLINDER_PIN_GEOMETRY_NOT_RECOVERED",
        },
        "continuous_rendered_three_cycle_video": {
            "status": "PASS" if video_ok else "FAIL",
            "path": "three_cycle_full_physics.mp4",
        },
        "terrain_visual_sync": {
            "status": "PASS" if max(item["maximum_visual_lag_s"] for item in summaries.values()) <= (5.0 / 60.0 + 1.0e-9) else "FAIL",
            "maximum_lag_s": max(item["maximum_visual_lag_s"] for item in summaries.values()),
            "policy": "visual mesh refreshed every six physics steps",
        },
        "terrain_contact_sync": {
            "status": "FAIL_STATIC_TRACK_SUPPORT_SURFACE",
            "reason": "The deforming heightmap is custom-solver/visual state. PhysX track support remains a static collision surface; bucket-bulk rigid collision is intentionally disabled to avoid double counting.",
        },
        "force_torque_work_logging": {"status": "PASS"},
        "published_obermayr_benchmark": {"status": "EXECUTED"},
        "real_time_performance": {
            "status": "FAIL",
            "minimum_rtf": min(profile["rtf"] for profile in profiles.values()),
        },
    }
    software_status_path = OUTPUT / "software_test_status.json"
    software_status = (
        json.loads(software_status_path.read_text(encoding="utf-8"))
        if software_status_path.is_file()
        else {"status": "NOT_RUN"}
    )
    acceptance = {
        "schema": "390F_INTEGRATED_EARTHMOVING_PHYSICS_V1_ACCEPTANCE",
        "overall_status": "DEGRADED_NOT_FULL_V1_ACCEPTANCE",
        "claim": "Integrated reduced-order research simulator; not a field-validated digital twin.",
        "case_metrics": metrics,
        "physics_acceptance": physics,
        "layer_status": {
            "PHYSICS_MODEL_STATUS": "DEGRADED_REAL_BUCKET_RAKE_OUTSIDE_IMPLEMENTED_FEE_DOMAIN",
            "ISAAC_RUNTIME_STATUS": "DEGRADED_STATIC_TRACK_SUPPORT_AND_NO_CONTACT_FORCE_SENSOR",
            "LITERATURE_VALIDATION_STATUS": "EXECUTED_2D_BENCHMARK_FINITE_WIDTH_AND_CAVITY_BLOCKED",
            "PERFORMANCE_STATUS": "FAIL_NOT_REAL_TIME",
        },
        "software_status": software_status,
    }
    runtime_manifest = {
        "asset": "/home/eric/桌面/bulldozer_sim/bulldozer_main.usd",
        "asset_sha256": "db2bb54fd81b43c9ce57f8ab9ad7d7b07dbe81b6dbf4dbd9dbb321eabe747b7c",
        "runner": "isaac_loader/integrated_390f_v1.py",
        "runner_sha256": hashlib.sha256(
            (ROOT / "isaac_loader" / "integrated_390f_v1.py").read_bytes()
        ).hexdigest(),
        "physics_dt_s": 1.0 / 60.0,
        "deterministic_seed": 390,
        "stochastic_model_terms_enabled": False,
        "terrain_grid": {"shape_yx": [701, 701], "spacing_m": 0.05},
        "material": {
            "name": "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED_FLOW_ANGLES_ENGINEERING",
            "assumed_bulk_density_kg_m3": 1370.0,
            "internal_friction_angle_deg": 29.8,
            "cohesion_pa": 800.0,
            "tool_friction_angle_deg": 34.4,
            "theta_start_deg": 38.0,
            "theta_stop_deg": 30.0,
            "soil_force_tuple_status": "SAME_SAMPLE_SAME_MOISTURE_NEAR_MATCHED_LOW_STRESS",
            "flow_angle_status": "ENGINEERING_INPUT_NOT_PUBLISHED_FOR_THIS_DYNAMIC_MODEL"
        },
        "comparison_contract": "same asset, initial heightmap, policy, dt, grid, material and seed; only applied soil-force mode differs",
        "formal_case_step_count": 648,
        "formal_case_count": 3,
        "rendered_video": "outputs/integrated_390f_v1/three_cycle_full_physics.mp4",
    }
    (OUTPUT / "cycle_summary.json").write_text(
        json.dumps({"cases": summaries, "comparison": metrics}, indent=2) + "\n"
    )
    (OUTPUT / "runtime_profile.json").write_text(
        json.dumps({"cases": profiles}, indent=2) + "\n"
    )
    (OUTPUT / "integrated_390f_physics_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n"
    )
    (OUTPUT / "runtime_manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2) + "\n"
    )
    for name in ("terrain_state_sequence.npz", "before_heightmap_m.csv", "after_heightmap_m.csv"):
        source = OUTPUT / "case_C" / name
        if source.exists():
            shutil.copy2(source, OUTPUT / name)
    for name in ("three_cycle_full_physics.mp4", "three_cycle_full_physics.json"):
        source = OUTPUT / "case_C" / name
        if source.exists():
            shutil.copy2(source, OUTPUT / name)
    evidence_copies = {
        ROOT / "configs" / "literature" / "iron_ore_condition_scenarios.json": OUTPUT / "condition_consistent_material_scenarios.json",
        ROOT / "outputs" / "real_390f_runtime_probe.json": OUTPUT / "real_390f_runtime_probe.json",
        ROOT / "outputs" / "real_390f_bucket_audit.json": OUTPUT / "real_390f_bucket_audit.json",
        ROOT / "outputs" / "soil_force_literature_acceptance.json": OUTPUT / "soil_force_literature_acceptance.json",
    }
    for source, destination in evidence_copies.items():
        if source.exists():
            shutil.copy2(source, destination)
    print(json.dumps({"status": acceptance["overall_status"], "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
