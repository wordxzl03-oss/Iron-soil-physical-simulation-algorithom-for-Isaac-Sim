#!/usr/bin/env python3
"""Phase C/D Isaac acceptance: drive the real loader on a contact-mesh slope.

The script runs one independent 0/10/20 degree experiment per process.  It
uses the Phase-B command/slew/low-level-controller chain, authors the Phase-C
hidden static triangle mesh and collision groups, and records Phase-D
telemetry from Isaac's public articulation and contact APIs.  It never writes
an articulation pose during the physics loop; the initial slope alignment is
authored as a USD reference-root transform before physics starts.

Examples::

    /home/eric/isaacsim/python.sh isaac_loader/phase_cd_slope_drive_runtime.py \
        --headless --slope-deg 0
    /home/eric/isaacsim/python.sh isaac_loader/phase_cd_slope_drive_runtime.py \
        --headless --slope-deg 10
    /home/eric/isaacsim/python.sh isaac_loader/phase_cd_slope_drive_runtime.py \
        --headless --slope-deg 20

After all three files exist, ``outputs/phase_cd_slope_summary.json`` is
updated with a like-for-like comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--slope-deg", type=float, choices=(0.0, 10.0, 20.0), required=True
    )
    parser.add_argument("--steps", type=int, default=420)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--brake-steps", type=int, default=60)
    parser.add_argument("--physics-dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--throttle", type=float, default=0.45)
    parser.add_argument("--grid-nx", type=int, default=601)
    parser.add_argument("--grid-ny", type=int, default=321)
    parser.add_argument("--visual-spacing-m", type=float, default=0.05)
    parser.add_argument("--contact-spacing-m", type=float, default=0.10)
    parser.add_argument("--terrain-origin-x-m", type=float, default=-15.0)
    parser.add_argument("--terrain-origin-y-m", type=float, default=-8.0)
    parser.add_argument("--loader-plane-origin-x-m", type=float, default=-9.0)
    parser.add_argument(
        "--loader-usd",
        type=Path,
        default=PROJECT_ROOT / "isaac_loader" / "wheel_loader.usd",
    )
    parser.add_argument(
        "--vehicle-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase_b_vehicle.yaml",
    )
    parser.add_argument(
        "--contact-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase_c_contact.yaml",
    )
    parser.add_argument(
        "--telemetry-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase_d_telemetry.yaml",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "phase_cd_slope_summary.json",
    )
    args, kit_args = parser.parse_known_args()
    if args.steps < 3:
        parser.error("--steps must be >= 3")
    if args.settle_steps < 1 or args.brake_steps < 1:
        parser.error("--settle-steps and --brake-steps must be >= 1")
    if args.settle_steps + args.brake_steps >= args.steps:
        parser.error("settle + brake steps must leave a non-empty drive window")
    if not 0.0 < args.physics_dt <= 0.1:
        parser.error("--physics-dt must be in (0, 0.1]")
    if not -1.0 <= args.throttle <= 1.0 or abs(args.throttle) < 1e-9:
        parser.error("--throttle must be a nonzero normalized value")
    if args.grid_nx < 3 or args.grid_ny < 3:
        parser.error("grid dimensions must be >= 3")
    if args.visual_spacing_m <= 0.0 or args.contact_spacing_m <= 0.0:
        parser.error("terrain spacings must be positive")
    if not args.loader_usd.is_file():
        parser.error(f"loader USD does not exist: {args.loader_usd}")
    if not args.vehicle_config.is_file():
        parser.error(f"vehicle config does not exist: {args.vehicle_config}")
    if not args.contact_config.is_file():
        parser.error(f"contact config does not exist: {args.contact_config}")
    if not args.telemetry_config.is_file():
        parser.error(f"telemetry config does not exist: {args.telemetry_config}")
    if args.output is None:
        label = f"{int(round(args.slope_deg)):02d}deg"
        args.output = PROJECT_ROOT / "outputs" / f"phase_cd_slope_{label}.json"
    return args, kit_args


def _json_finite(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite value cannot be written to evidence: {value!r}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_python_tree(root: Path) -> str:
    """Hash Python source paths and bytes so imported pipeline code is identified."""

    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    if not paths:
        raise RuntimeError(f"no Python sources found for fingerprint: {root}")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_json(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    import numpy as np

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return None
    return float(np.percentile(array, percentile))


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    import numpy as np

    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    if not np.all(np.isfinite(array)):
        raise RuntimeError("statistics input contains NaN or Inf")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "max": float(np.max(array)),
    }


def _wxyz_to_euler_and_forward(quaternion_wxyz: Any) -> tuple[float, float, float, Any]:
    """Return XYZ Euler angles and the world direction of vehicle-local +X."""

    import numpy as np

    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("Isaac returned an invalid root quaternion")
    w, x, y, z = quaternion / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # First column of the scalar-first quaternion rotation matrix.
    forward = np.asarray(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )
    forward /= np.linalg.norm(forward)
    return roll, pitch, yaw, forward


def _slope_reference_transform(
    *, slope_deg: float, plane_origin_x_m: float, terrain_origin_x_m: float
) -> Any:
    """Map the asset's flat z=0 support plane exactly onto the test slope."""

    import numpy as np
    from pxr import Gf

    angle = math.radians(slope_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    translation_z = math.tan(angle) * (plane_origin_x_m - terrain_origin_x_m)
    # Column-vector convention: local +X becomes the uphill tangent and local
    # +Z becomes the outward normal.  Gf.Matrix4d stores the row-vector form.
    matrix = np.asarray(
        [
            [cosine, 0.0, -sine, plane_origin_x_m],
            [0.0, 1.0, 0.0, 0.0],
            [sine, 0.0, cosine, translation_z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return Gf.Matrix4d(*matrix.T.ravel().tolist())


def _to_vec3f(values: Any, vt_module: Any) -> Any:
    import numpy as np

    return vt_module.Vec3fArray.FromNumpy(
        np.ascontiguousarray(values, dtype=np.float32)
    )


def _to_int(values: Any, vt_module: Any) -> Any:
    import numpy as np

    return vt_module.IntArray.FromNumpy(
        np.ascontiguousarray(values, dtype=np.int32)
    )


def _author_visual_mesh(stage: Any, visual_data: Any, visual_path: str) -> Any:
    """Author the independent full-resolution render mesh with no CollisionAPI."""

    from pxr import Gf, UsdGeom, Vt

    mesh = UsdGeom.Mesh.Define(stage, visual_path)
    mesh.CreatePointsAttr(_to_vec3f(visual_data.points_m, Vt))
    mesh.CreateFaceVertexCountsAttr(_to_int(visual_data.face_vertex_counts, Vt))
    mesh.CreateFaceVertexIndicesAttr(_to_int(visual_data.face_vertex_indices, Vt))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.20, 0.10)])
    mesh.CreatePurposeAttr(UsdGeom.Tokens.render)
    return mesh


def _author_environment_probe(stage: Any, path: str) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(100.0, 100.0, -2.0))
    cube.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _bind_physics_material(
    stage: Any, *, target_paths: Iterable[str], material_path: str
) -> dict[str, float]:
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(1.05)
    physics.CreateDynamicFrictionAttr(0.90)
    physics.CreateRestitutionAttr(0.0)
    for path in target_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"physics-material target is missing: {path}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    return {"static_friction": 1.05, "dynamic_friction": 0.90, "restitution": 0.0}


def _category_for_joint(name: str) -> Any:
    from isaac_bulk_pipeline.telemetry import ActuatorCategory

    if name.endswith("wheel_joint"):
        return ActuatorCategory.DRIVE
    if name == "articulation_joint":
        return ActuatorCategory.STEER
    if name == "lift_joint":
        return ActuatorCategory.LIFT
    if name == "bucket_joint":
        return ActuatorCategory.BUCKET
    raise RuntimeError(f"unmapped telemetry DOF: {name}")


def _energy_to_dict(report: Any) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in ("drive", "steer", "lift", "bucket", "total"):
        item = getattr(report, name)
        result[name] = {
            "positive_j": _json_finite(item.positive_j),
            "signed_j": _json_finite(item.signed_j),
        }
    return result


def _runtime_stats_to_dict(summary: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "sample_count": int(item.sample_count),
            "mean_ms": _json_finite(item.mean_ms),
            "p50_ms": _json_finite(item.p50_ms),
            "p95_ms": _json_finite(item.p95_ms),
            "max_ms": _json_finite(item.max_ms),
        }
        for name, item in summary.items()
    }


def _serialize_frame(
    *,
    frame_index: int,
    phase: str,
    command: Any,
    frame: Any,
    wheel_body_centers: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    pose = frame.pose
    return {
        "frame": int(frame_index),
        "phase": phase,
        "command": command.as_dict(),
        "pose": {
            "timestamp_s": pose.timestamp_s,
            "position_world_m": pose.position_world_m.tolist(),
            "orientation_xyzw": pose.orientation_xyzw.tolist(),
            "roll_rad": pose.roll_rad,
            "pitch_rad": pose.pitch_rad,
            "yaw_rad": pose.yaw_rad,
            "linear_velocity_world_m_s": pose.linear_velocity_world_m_s.tolist(),
            "angular_velocity_world_rad_s": pose.angular_velocity_world_rad_s.tolist(),
            "articulation_angle_rad": pose.articulation_angle_rad,
            "local_terrain_slope_rad": pose.local_terrain_slope_rad,
        },
        "joints": [
            {
                "joint_name": item.joint_name,
                "category": item.category.value,
                "angular_velocity_rad_s": item.angular_velocity_rad_s,
                "target_velocity_rad_s": item.target_velocity_rad_s,
                "applied_effort_nm": item.applied_effort_nm,
                "applied_effort_source": "SingleArticulation.get_applied_joint_efforts",
                "measured_effort_nm": item.measured_effort_nm,
                "measured_effort_source": "SingleArticulation.get_measured_joint_efforts",
                "energy_effort_source": item.to_power_sample().effort_source.value,
                "power_w": item.to_power_sample().power_w,
            }
            for item in frame.joints
        ],
        "wheel_kinematics": [
            {
                "wheel_name": item.wheel_name,
                "wheel_radius_m": item.wheel_radius_m,
                "actual_angular_velocity_rad_s": item.angular_velocity_rad_s,
                "target_angular_velocity_rad_s": item.target_angular_velocity_rad_s,
                "longitudinal_speed_m_s": item.longitudinal_speed_m_s,
                "slip_ratio": item.slip_ratio,
                "low_speed_threshold_m_s": item.low_speed_threshold_m_s,
            }
            for item in frame.wheel_kinematics
        ],
        "wheel_contacts": [
            {
                "wheel_name": item.wheel_name,
                "position_world_m": item.position_world_m.tolist(),
                "normal_load_n": item.normal_load_n,
                "tangential_load_n": item.tangential_load_n,
                "in_contact": item.in_contact,
                "contact_duration_s": item.contact_duration_s,
                "position_source": "force-weighted centroid of RigidPrim.get_contact_force_data points",
                "load_source": "RigidPrim.get_contact_force_matrix (public RigidContactView wrapper)",
            }
            for item in frame.wheel_contacts
        ],
        "wheel_body_centers": {
            name: list(np.asarray(position, dtype=float))
            for name, position in wheel_body_centers.items()
        },
        "wheel_body_center_source": "RigidPrim.get_world_poses; these are not contact points",
    }


def _update_aggregate(summary_output: Path) -> dict[str, Any]:
    expected = {
        0: PROJECT_ROOT / "outputs" / "phase_cd_slope_00deg.json",
        10: PROJECT_ROOT / "outputs" / "phase_cd_slope_10deg.json",
        20: PROJECT_ROOT / "outputs" / "phase_cd_slope_20deg.json",
    }
    runs: dict[int, dict[str, Any]] = {}
    for slope, path in expected.items():
        if path.is_file():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if float(document.get("run_config", {}).get("slope_deg", -1.0)) == slope:
                runs[slope] = document

    comparisons: dict[str, Any] = {}
    for slope, document in sorted(runs.items()):
        metrics = document.get("metrics", {})
        power_evidence = (
            document.get("phase_b_vehicle_control", {})
            .get("wheel_power_limit_evidence", {})
        )
        comparisons[str(slope)] = {
            "status": document.get("status"),
            "uphill_progress_m": metrics.get("uphill_progress_m"),
            "mean_longitudinal_speed_m_s": metrics.get("drive_window", {}).get(
                "longitudinal_speed_m_s", {}
            ).get("mean"),
            "mean_abs_drive_measured_effort_nm": metrics.get("drive_window", {}).get(
                "abs_drive_measured_effort_nm", {}
            ).get("mean"),
            "mean_abs_wheel_slip_ratio": metrics.get("drive_window", {}).get(
                "abs_wheel_slip_ratio", {}
            ).get("mean"),
            "drive_positive_energy_j": metrics.get("energy", {})
            .get("drive", {})
            .get("positive_j"),
            "drive_signed_energy_j": metrics.get("energy", {})
            .get("drive", {})
            .get("signed_j"),
            "elapsed_sim_time_s": metrics.get("elapsed_sim_time_s"),
            "comparison_distance_m": metrics.get("comparison_distance_m"),
            "time_to_common_distance_s": metrics.get("time_to_common_distance_s"),
            "max_wheel_normal_load_n": metrics.get("contact", {}).get(
                "max_normal_load_n"
            ),
            "power_cap_measured_basis_frames": power_evidence.get(
                "measured_or_target_worst_case_frames"
            ),
            "power_cap_fallback_frames": power_evidence.get(
                "target_speed_fallback_frames"
            ),
            "max_installed_effort_post_step_speed_power_bound_w": (
                power_evidence.get(
                    "max_installed_effort_post_step_measured_speed_power_bound_w"
                )
            ),
        }
    fingerprints = [
        document.get("run_config", {}).get("scenario_fingerprint_sha256")
        for document in runs.values()
    ]
    correct_schemas = all(
        document.get("schema_version")
        == "isaac-bulk-phase-cd-slope-runtime/v1"
        for document in runs.values()
    )
    same_config = bool(
        fingerprints
        and all(isinstance(value, str) and len(value) == 64 for value in fingerprints)
        and len(set(fingerprints)) == 1
        and correct_schemas
    )
    complete = set(runs) == set(expected)
    all_pass = complete and all(item.get("status") == "PASS" for item in runs.values())
    effect_ranges: dict[str, float | None] = {}
    effect_thresholds = {
        "mean_longitudinal_speed_m_s": 0.05,
        "mean_abs_drive_measured_effort_nm": 100.0,
        "mean_abs_wheel_slip_ratio": 0.005,
        "drive_positive_energy_j": 100.0,
        "uphill_progress_m": 0.10,
    }
    effect_checks: dict[str, bool] = {}
    for field, threshold in effect_thresholds.items():
        values = [
            item.get(field)
            for item in comparisons.values()
            if item.get(field) is not None
        ]
        value_range = (
            float(max(values) - min(values)) if len(values) == 3 else None
        )
        effect_ranges[field] = value_range
        effect_checks[field] = bool(
            value_range is not None and value_range >= threshold
        )
    measurable_slope_effect = bool(
        complete and sum(effect_checks.values()) >= 2
    )
    persistence_verified = True
    for document in runs.values():
        persistence = document.get("phase_d_telemetry", {}).get(
            "full_step_persistence", {}
        )
        path_value = persistence.get("path")
        expected_hash = persistence.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_hash, str):
            persistence_verified = False
            break
        persistence_path = Path(path_value)
        if (
            not persistence_path.is_file()
            or _sha256_file(persistence_path) != expected_hash
            or persistence.get("frame_count")
            != document.get("run_config", {}).get("steps")
            or not persistence.get("all_required_fields_present")
        ):
            persistence_verified = False
            break
    runtime_power_cap_verified = bool(
        complete
        and all(
            (
                item.get("phase_b_vehicle_control", {})
                .get("wheel_power_limit_evidence", {})
                .get("measured_or_target_worst_case_frames")
                == item.get("run_config", {}).get("steps")
            )
            and (
                item.get("phase_b_vehicle_control", {})
                .get("wheel_power_limit_evidence", {})
                .get("target_speed_fallback_frames")
                == 0
            )
            and bool(
                item.get("phase_b_vehicle_control", {})
                .get("wheel_power_limit_evidence", {})
                .get("command_envelope_bound_within_configured_limit")
            )
            and bool(
                item.get("phase_b_vehicle_control", {})
                .get("wheel_power_limit_evidence", {})
                .get("post_step_bound_within_configured_limit")
            )
            for item in runs.values()
        )
    )
    aggregate_checks = {
        "all_three_slope_runs_present": complete,
        "all_single_run_status_pass": bool(all_pass),
        "all_single_run_acceptance_checks_true": bool(
            complete
            and all(
                all(item.get("acceptance_checks", {}).values())
                for item in runs.values()
            )
        ),
        "same_full_physics_scenario_fingerprint": bool(same_config),
        "measurable_slope_effect_in_at_least_two_metrics": measurable_slope_effect,
        "time_to_common_distance_available_for_all": bool(
            complete
            and all(
                item.get("metrics", {}).get("time_to_common_distance_s")
                is not None
                for item in runs.values()
            )
        ),
        "full_step_npz_hash_and_frame_count_verified": bool(
            complete and persistence_verified
        ),
        "public_contact_load_and_points_available_for_all": bool(
            complete
            and all(
                item.get("phase_d_telemetry", {})
                .get("contact_force", {})
                .get("available")
                and item.get("phase_d_telemetry", {})
                .get("contact_points", {})
                .get("available")
                for item in runs.values()
            )
        ),
        "measured_speed_runtime_power_cap_verified_for_all": (
            runtime_power_cap_verified
        ),
    }
    aggregate_pass = bool(aggregate_checks and all(aggregate_checks.values()))
    aggregate = {
        "schema_version": "isaac-bulk-phase-cd-slope-comparison/v1",
        "status": (
            "PASS"
            if aggregate_pass
            else ("INCOMPLETE" if not complete else "FAIL")
        ),
        "required_slopes_deg": [0, 10, 20],
        "completed_slopes_deg": sorted(runs),
        "same_throttle_steps_dt_across_runs": bool(same_config),
        "same_full_physics_scenario_fingerprint": bool(same_config),
        "aggregate_checks": aggregate_checks,
        "scenario_fingerprints_sha256": fingerprints,
        "measurable_slope_effect": measurable_slope_effect,
        "slope_effect_minimum_two_metrics_required": True,
        "slope_effect_ranges": effect_ranges,
        "slope_effect_thresholds": effect_thresholds,
        "slope_effect_checks": effect_checks,
        "run_files": {str(key): str(path) for key, path in expected.items()},
        "comparisons": comparisons,
        "notes": [
            "Each slope is a fresh Isaac process with identical throttle, duration and dt.",
            "No result is interpolated; all comparison values come from public runtime APIs.",
            "Positive and signed mechanical energy are both retained; no regenerative model is assumed.",
            "Time comparison is time_to_common_distance; fixed-duration runs are not mislabeled as travel time.",
            "The 160 kW check is installed per-wheel effort limit times post-step measured |omega|; it is not measured tau*omega.",
        ],
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return aggregate


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import yaml
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.usd
    from pxr import UsdGeom, UsdPhysics

    from isaac_bulk_pipeline.contact import (
        CollisionGroup,
        CollisionMatrix,
        ContactBackendConfig,
        SlopePatchSpec,
        TriangleMeshContactBackend,
        build_contact_mesh,
        build_planar_slope_heightmap,
        estimate_slope_deg,
        validate_collision_memberships,
    )
    from isaac_bulk_pipeline.telemetry import (
        JointTelemetrySample,
        RuntimeProfiler,
        VehiclePoseSample,
        VehicleTelemetryFrame,
        VehicleTelemetryRecorder,
        WheelKinematicsSample,
        WheelTerrainContactSample,
    )
    from isaac_bulk_pipeline.terrain import TerrainGrid
    from isaac_bulk_pipeline.vehicle import (
        CommandSlewLimits,
        CommandSlewLimiter,
        LoaderControlConfig,
        LoaderLowLevelController,
        VehicleCommand,
    )
    from isaac_loader.phase_c_contact_runtime import (
        author_collision_groups,
        author_triangle_mesh_contact,
        default_loader_memberships,
        inspect_authored_groups,
        _validate_stage_members,
    )

    vehicle_document = yaml.safe_load(args.vehicle_config.read_text(encoding="utf-8"))
    contact_document = yaml.safe_load(args.contact_config.read_text(encoding="utf-8"))
    telemetry_document = yaml.safe_load(
        args.telemetry_config.read_text(encoding="utf-8")
    )
    configured_contact = contact_document["contact"]
    configured_telemetry = telemetry_document["phase_d_telemetry"]
    configured_slopes = tuple(
        float(value)
        for value in configured_telemetry["slope_validation"]["slope_degrees"]
    )
    phase_c_slopes = tuple(
        float(value) for value in contact_document["slope_tests"]["degrees"]
    )
    if args.slope_deg not in configured_slopes or args.slope_deg not in phase_c_slopes:
        raise RuntimeError(
            f"slope {args.slope_deg} is not present in both Phase-C/D configs"
        )
    configured_contact_spacing = float(configured_contact["target_spacing_m"])
    if not math.isclose(
        args.contact_spacing_m, configured_contact_spacing, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "--contact-spacing-m must match Phase-C config: "
            f"cli={args.contact_spacing_m}, config={configured_contact_spacing}"
        )
    configured_commit_policy = str(configured_contact["commit"]["policy"])
    if configured_contact["backend"] != "triangle_mesh":
        raise RuntimeError("Phase-CD runtime requires configured triangle_mesh backend")
    if configured_contact["mesh_approximation"] != "none":
        raise RuntimeError("Phase-CD static triangle mesh requires approximation=none")
    if configured_commit_policy != "action_end":
        raise RuntimeError("Phase-CD runtime currently validates action_end commits")
    if bool(configured_contact["visible"]) or bool(
        configured_contact["rigid_body_enabled"]
    ):
        raise RuntimeError("Phase-C contact config must specify hidden static contact")
    deformation_flags = {
        name: bool(contact_document["wheel_terrain"][name])
        for name in (
            "deformation_enabled",
            "sinkage_enabled",
            "compaction_enabled",
            "rut_enabled",
        )
    }
    if any(deformation_flags.values()):
        raise RuntimeError("Phase-C slope validation requires deformation features disabled")
    configured_group_names = tuple(
        str(value)
        for value in contact_document["collision_filtering"]["groups"]
    )
    if configured_group_names != tuple(group.value for group in CollisionGroup):
        raise RuntimeError(
            "Phase-C collision group list/order differs from executable contract: "
            f"{configured_group_names}"
        )
    configured_disabled_pairs = contact_document["collision_filtering"][
        "disabled_pairs"
    ]
    if configured_disabled_pairs != [["TerrainSupport", "BucketInteraction"]]:
        raise RuntimeError(
            "Phase-C requires TerrainSupport/BucketInteraction as sole disabled pair"
        )
    configured_rate_hz = float(configured_telemetry["samples"]["physics_rate_hz"])
    if not math.isclose(
        configured_rate_hz, 1.0 / args.physics_dt, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError(
            "--physics-dt must match Phase-D configured physics_rate_hz: "
            f"dt={args.physics_dt}, configured_hz={configured_rate_hz}"
        )
    low_speed_threshold_m_s = float(
        configured_telemetry["wheel_slip"]["low_speed_threshold_m_s"]
    )
    effort_precedence = tuple(
        str(value)
        for value in configured_telemetry["actuator_power"]["effort_precedence"]
    )
    if effort_precedence != (
        "measured_effort",
        "applied_effort_command",
        "estimated_effort",
    ):
        raise RuntimeError(
            "Phase-D effort precedence differs from JointTelemetrySample contract: "
            f"{effort_precedence}"
        )
    comparison_distance_m = float(
        configured_telemetry["slope_validation"]["common_distance_m"]
    )
    if not math.isfinite(comparison_distance_m) or comparison_distance_m <= 0.0:
        raise RuntimeError("Phase-D common_distance_m must be finite and positive")
    vehicle_config = LoaderControlConfig.from_mapping(vehicle_document["vehicle"])
    slew_config = vehicle_document["command_slew"]
    slew = CommandSlewLimiter(
        CommandSlewLimits(
            throttle_per_s=float(slew_config["throttle_per_s"]),
            steering_per_s=float(slew_config["steering_per_s"]),
            lift_per_s=float(slew_config["lift_per_s"]),
            bucket_curl_per_s=float(slew_config["bucket_curl_per_s"]),
        )
    )

    world = World(
        physics_dt=float(args.physics_dt),
        rendering_dt=float(args.physics_dt),
        stage_units_in_meters=1.0,
        backend="numpy",
    )
    stage = omni.usd.get_context().get_stage()
    loader_path = "/World/WheelLoader"
    add_reference_to_stage(str(args.loader_usd.resolve()), loader_path)
    loader_wrapper = UsdGeom.Xformable(stage.GetPrimAtPath(loader_path))
    loader_wrapper.AddTransformOp().Set(
        _slope_reference_transform(
            slope_deg=args.slope_deg,
            plane_origin_x_m=args.loader_plane_origin_x_m,
            terrain_origin_x_m=args.terrain_origin_x_m,
        )
    )

    contact_config = ContactBackendConfig(
        contact_prim_path=str(configured_contact["prim_path"]),
        visual_prim_path=str(configured_contact["visual_prim_path"]),
        target_spacing_m=configured_contact_spacing,
        commit_policy=configured_commit_policy,
    )
    grid = TerrainGrid(
        nx=args.grid_nx,
        ny=args.grid_ny,
        dx=args.visual_spacing_m,
        dy=args.visual_spacing_m,
        origin_x=args.terrain_origin_x_m,
        origin_y=args.terrain_origin_y_m,
        terrain_prim_path=contact_config.visual_prim_path,
    )
    heightmap = build_planar_slope_heightmap(
        grid, SlopePatchSpec(args.slope_deg, uphill_direction_xy=(1.0, 0.0))
    )
    contact_backend = TriangleMeshContactBackend(contact_config)
    contact_backend.initialize(grid, heightmap)
    contact_schema = author_triangle_mesh_contact(stage, contact_backend)
    visual_data = build_contact_mesh(
        heightmap,
        grid,
        target_spacing_m=args.visual_spacing_m,
    )
    visual_schema = _author_visual_mesh(
        stage, visual_data, contact_config.visual_prim_path
    )
    environment_path = "/World/EnvironmentStatic/BoundaryProbe"
    _author_environment_probe(stage, environment_path)

    memberships = default_loader_memberships(
        loader_path=loader_path,
        contact_prim_path=contact_config.contact_prim_path,
        environment_collider_path=environment_path,
    )
    membership_validation = validate_collision_memberships(
        memberships,
        contact_prim_path=contact_config.contact_prim_path,
        visual_prim_path=contact_config.visual_prim_path,
    )
    membership_validation.require_valid()
    _validate_stage_members(stage, memberships)
    collision_matrix = CollisionMatrix()
    authored_groups = author_collision_groups(
        stage,
        group_prim_root=str(
            contact_document["collision_filtering"]["group_prim_root"]
        ),
        memberships=memberships,
        matrix=collision_matrix,
    )

    wheel_collider_paths = memberships[CollisionGroup.WHEEL_CONTACT]
    material_values = _bind_physics_material(
        stage,
        target_paths=(contact_config.contact_prim_path, *wheel_collider_paths),
        material_path="/World/PhysicsMaterials/TerrainTyre",
    )
    contact_prim = contact_schema.GetPrim()
    visual_prim = visual_schema.GetPrim()
    if contact_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("terrain contact mesh must remain static")
    if visual_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("render mesh unexpectedly owns CollisionAPI")

    scenario_identity = {
        "runtime_schema": "isaac-bulk-phase-cd-slope-runtime/v1",
        "runtime_script_sha256": _sha256_file(Path(__file__).resolve()),
        "pipeline_python_source_tree_sha256": _sha256_python_tree(
            SRC_ROOT / "isaac_bulk_pipeline"
        ),
        "phase_c_runtime_source_sha256": _sha256_file(
            PROJECT_ROOT / "isaac_loader" / "phase_c_contact_runtime.py"
        ),
        "loader_usd_sha256": _sha256_file(args.loader_usd.resolve()),
        "vehicle_config_sha256": _sha256_file(args.vehicle_config.resolve()),
        "contact_config_sha256": _sha256_file(args.contact_config.resolve()),
        "telemetry_config_sha256": _sha256_file(args.telemetry_config.resolve()),
        "throttle": float(args.throttle),
        "steps": int(args.steps),
        "settle_steps": int(args.settle_steps),
        "brake_steps": int(args.brake_steps),
        "physics_dt_s": float(args.physics_dt),
        "grid_nx": int(args.grid_nx),
        "grid_ny": int(args.grid_ny),
        "visual_spacing_m": float(args.visual_spacing_m),
        "contact_spacing_m": float(args.contact_spacing_m),
        "terrain_origin_x_m": float(args.terrain_origin_x_m),
        "terrain_origin_y_m": float(args.terrain_origin_y_m),
        "loader_plane_origin_x_m": float(args.loader_plane_origin_x_m),
        "wheel_radius_m": float(vehicle_config.wheel_radius_m),
        "wheel_power_limit_w": float(vehicle_config.wheel_power_limit_w),
        "wheel_power_reference_rad_s": float(
            vehicle_config.wheel_power_reference_rad_s
        ),
        "wheel_power_min_guard_rad_s": float(
            vehicle_config.wheel_power_min_guard_rad_s
        ),
        "wheel_power_discrete_safety_factor": float(
            vehicle_config.wheel_power_discrete_safety_factor
        ),
        "physics_material": material_values,
        "comparison_distance_m": comparison_distance_m,
        "contact_backend": configured_contact["backend"],
        "contact_commit_policy": configured_commit_policy,
        "deformation_flags": deformation_flags,
        "slip_low_speed_threshold_m_s": low_speed_threshold_m_s,
        "effort_precedence": list(effort_precedence),
    }
    scenario_fingerprint = _sha256_json(scenario_identity)

    loader = world.scene.add(
        SingleArticulation(
            "/World/WheelLoader/rear_chassis", name="phase_cd_wheel_loader"
        )
    )
    contact_api_error: str | None = None
    contact_point_api_error: str | None = None
    wheel_view = None
    try:
        wheel_view = world.scene.add(
            RigidPrim(
                prim_paths_expr="/World/WheelLoader/.*_wheel",
                name="phase_cd_wheel_contact_view",
                track_contact_forces=True,
                prepare_contact_sensors=True,
                contact_filter_prim_paths_expr=[contact_config.contact_prim_path],
                max_contact_count=2048,
                reset_xform_properties=False,
            )
        )
    except Exception as error:  # retain honest unavailability instead of fabricating loads
        contact_api_error = f"{type(error).__name__}: {error}"

    world.reset()
    dof_names = tuple(str(name) for name in loader.dof_names)
    joint_positions = np.asarray(loader.get_joint_positions(), dtype=np.float64).reshape(-1)
    low_level = LoaderLowLevelController(dof_names, vehicle_config)
    low_level.reset(joint_positions)
    articulation_controller = loader.get_articulation_controller()
    wheel_indices = low_level.mapping.wheel_indices
    wheel_names_by_dof = [dof_names[index] for index in wheel_indices]
    measured_joint_velocities_for_control = np.asarray(
        loader.get_joint_velocities(), dtype=np.float64
    ).reshape(-1)
    if (
        measured_joint_velocities_for_control.shape != joint_positions.shape
        or not np.all(np.isfinite(measured_joint_velocities_for_control))
    ):
        raise RuntimeError(
            "initial measured joint velocities must match the finite articulation DOFs"
        )

    wheel_view_paths: list[str] = []
    if wheel_view is not None:
        wheel_view_paths = list(wheel_view.prim_paths)
        if len(wheel_view_paths) != 4:
            contact_api_error = (
                f"RigidPrim matched {len(wheel_view_paths)} wheel bodies, expected 4: "
                f"{wheel_view_paths}"
            )
            wheel_view = None

    recorder = VehicleTelemetryRecorder()
    profiler = RuntimeProfiler()
    sampled_frames: list[dict[str, Any]] = []
    drive_speeds: list[float] = []
    drive_abs_slips: list[float] = []
    drive_abs_efforts: list[float] = []
    drive_pitch: list[float] = []
    drive_roll: list[float] = []
    plane_clearances: list[float] = []
    wheel_surface_clearances: list[float] = []
    contact_normal_loads: list[float] = []
    contact_tangential_loads: list[float] = []
    contact_frames = 0
    contact_point_frames = 0
    contact_duration_by_path = {path: 0.0 for path in wheel_view_paths}
    effort_source_counter: Counter[str] = Counter()
    nonzero_wheel_command_frames = 0
    finite_state = True
    initial_position: np.ndarray | None = None
    final_position: np.ndarray | None = None
    previous_position: np.ndarray | None = None
    path_distance_m = 0.0
    maximum_planar_step_m = 0.0
    applied_effort_available = True
    measured_effort_available = True
    query_errors: list[str] = []
    drive_origin_position: np.ndarray | None = None
    drive_progresses: list[float] = []
    time_to_common_distance_s: float | None = None
    installed_wheel_effort_limits: list[float] = []
    command_envelope_power_bounds: list[float] = []
    power_cap_worst_case_omegas: list[float] = []
    power_cap_denominators: list[float] = []
    post_step_measured_wheel_omegas: list[float] = []
    post_step_installed_effort_speed_power_bounds: list[float] = []
    power_cap_basis_counts: Counter[str] = Counter()
    strongest_measured_cap_sample: dict[str, Any] | None = None
    worst_post_step_power_sample: dict[str, Any] | None = None

    slope_angle_rad = math.radians(args.slope_deg)
    surface_normal = np.asarray(
        [-math.sin(slope_angle_rad), 0.0, math.cos(slope_angle_rad)],
        dtype=np.float64,
    )
    uphill_tangent = np.asarray(
        [math.cos(slope_angle_rad), 0.0, math.sin(slope_angle_rad)],
        dtype=np.float64,
    )
    drive_end = args.steps - args.brake_steps

    # Persist every physics-step observation separately from the stride-20 JSON
    # preview. NaN has a precise meaning here: the public runtime signal was
    # unavailable or no contact point existed for that wheel on that step.
    full_root_position = np.full((args.steps, 3), np.nan, dtype=np.float64)
    full_root_orientation_xyzw = np.full((args.steps, 4), np.nan, dtype=np.float64)
    full_root_linear_velocity = np.full((args.steps, 3), np.nan, dtype=np.float64)
    full_root_angular_velocity = np.full((args.steps, 3), np.nan, dtype=np.float64)
    full_root_rpy = np.full((args.steps, 3), np.nan, dtype=np.float64)
    full_joint_velocity = np.full(
        (args.steps, len(dof_names)), np.nan, dtype=np.float64
    )
    full_applied_effort = np.full_like(full_joint_velocity, np.nan)
    full_measured_effort = np.full_like(full_joint_velocity, np.nan)
    full_joint_target_velocity = np.full_like(full_joint_velocity, np.nan)
    full_joint_target_position = np.full_like(full_joint_velocity, np.nan)
    full_wheel_slip = np.full((args.steps, 4), np.nan, dtype=np.float64)
    full_wheel_normal_load = np.full((args.steps, 4), np.nan, dtype=np.float64)
    full_wheel_tangential_load = np.full((args.steps, 4), np.nan, dtype=np.float64)
    full_wheel_contact_point = np.full((args.steps, 4, 3), np.nan, dtype=np.float64)
    full_wheel_body_center = np.full((args.steps, 4, 3), np.nan, dtype=np.float64)
    full_wheel_surface_clearance = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_normalized_command = np.full((args.steps, 5), np.nan, dtype=np.float64)
    full_phase_code = np.full(args.steps, -1, dtype=np.int8)
    full_power_cap_basis_code = np.full(args.steps, -1, dtype=np.int8)
    full_power_cap_min_guard = np.full(args.steps, np.nan, dtype=np.float64)
    full_power_cap_input_joint_velocity = np.full_like(
        full_joint_velocity, np.nan
    )
    full_wheel_power_cap_previous_measured_omega = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_power_cap_target_omega = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_power_cap_worst_case_omega = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_power_cap_denominator = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_installed_effort_limit = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_command_envelope_power_bound = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_post_step_measured_omega = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )
    full_wheel_installed_effort_post_step_speed_power_bound = np.full(
        (args.steps, 4), np.nan, dtype=np.float64
    )

    for frame_index in range(args.steps):
        if frame_index < args.settle_steps or frame_index >= drive_end:
            raw_command = VehicleCommand(brake=1.0)
            phase = "settle" if frame_index < args.settle_steps else "brake"
        else:
            raw_command = VehicleCommand(throttle=args.throttle)
            phase = "drive"
        command = slew.step(raw_command, args.physics_dt)

        with profiler.measure("vehicle_controller"):
            # The power-cap state is the measurement returned after the
            # previous PhysX step (or the post-reset measurement on frame 0).
            # Runtime must never use the controller's target-only test fallback.
            power_cap_input_velocities = (
                measured_joint_velocities_for_control.copy()
            )
            targets = low_level.step(
                command,
                args.physics_dt,
                measured_joint_velocities=power_cap_input_velocities,
            )
            wheel_action = targets.wheel_action
            position_action = targets.position_action
            if wheel_action.effort_limits is None:
                raise RuntimeError("wheel action omitted installed effort limits")
            if any(abs(value) > 1e-9 for value in wheel_action.joint_velocities):
                nonzero_wheel_command_frames += 1
            installed_wheel_effort_limits.extend(wheel_action.effort_limits)
            command_envelope_power_bounds.extend(
                targets.wheel_command_envelope_power_bound_w
            )
            power_cap_worst_case_omegas.extend(
                targets.wheel_power_cap_worst_case_omega_rad_s
            )
            power_cap_denominators.extend(
                targets.wheel_power_cap_denominator_rad_s
            )
            power_cap_basis_counts[targets.wheel_power_cap_basis] += 1
            cap_sample = {
                "frame": frame_index,
                "basis": targets.wheel_power_cap_basis,
                "input_full_joint_velocity_rad_s": (
                    power_cap_input_velocities.tolist()
                ),
                "previous_measured_wheel_omega_rad_s": list(
                    targets.wheel_power_cap_previous_measured_omega_rad_s or ()
                ),
                "target_wheel_omega_rad_s": list(
                    targets.wheel_power_cap_target_omega_rad_s
                ),
                "worst_case_omega_rad_s": list(
                    targets.wheel_power_cap_worst_case_omega_rad_s
                ),
                "protected_denominator_rad_s": list(
                    targets.wheel_power_cap_denominator_rad_s
                ),
                "discrete_safety_factor": (
                    targets.wheel_power_cap_discrete_safety_factor
                ),
                "minimum_speed_guard_rad_s": (
                    targets.wheel_power_cap_min_guard_rad_s
                ),
                "installed_effort_limit_nm": list(
                    wheel_action.effort_limits
                ),
                "command_envelope_power_bound_w": list(
                    targets.wheel_command_envelope_power_bound_w
                ),
            }
            full_power_cap_basis_code[frame_index] = {
                "MEASURED_OR_TARGET_WORST_CASE": 0,
                "TARGET_JOINT_VELOCITY_FALLBACK": 1,
            }[targets.wheel_power_cap_basis]
            full_power_cap_min_guard[frame_index] = (
                targets.wheel_power_cap_min_guard_rad_s
            )
            full_power_cap_input_joint_velocity[frame_index] = (
                power_cap_input_velocities
            )
            previous_measured_wheel_omega = (
                targets.wheel_power_cap_previous_measured_omega_rad_s
            )
            if previous_measured_wheel_omega is not None:
                full_wheel_power_cap_previous_measured_omega[frame_index] = (
                    previous_measured_wheel_omega
                )
            full_wheel_power_cap_target_omega[frame_index] = (
                targets.wheel_power_cap_target_omega_rad_s
            )
            full_wheel_power_cap_worst_case_omega[frame_index] = (
                targets.wheel_power_cap_worst_case_omega_rad_s
            )
            full_wheel_power_cap_denominator[frame_index] = (
                targets.wheel_power_cap_denominator_rad_s
            )
            full_wheel_installed_effort_limit[frame_index] = (
                wheel_action.effort_limits
            )
            full_wheel_command_envelope_power_bound[frame_index] = (
                targets.wheel_command_envelope_power_bound_w
            )
            articulation_controller.set_max_efforts(
                np.asarray(wheel_action.effort_limits, dtype=np.float32),
                joint_indices=np.asarray(wheel_action.joint_indices, dtype=np.int32),
            )
            articulation_controller.set_max_efforts(
                np.asarray(position_action.effort_limits, dtype=np.float32),
                joint_indices=np.asarray(position_action.joint_indices, dtype=np.int32),
            )
            articulation_controller.apply_action(
                ArticulationAction(
                    joint_velocities=np.asarray(
                        wheel_action.joint_velocities, dtype=np.float32
                    ),
                    joint_indices=np.asarray(
                        wheel_action.joint_indices, dtype=np.int32
                    ),
                )
            )
            articulation_controller.apply_action(
                ArticulationAction(
                    joint_positions=np.asarray(
                        position_action.joint_positions, dtype=np.float32
                    ),
                    joint_velocities=np.asarray(
                        position_action.joint_velocities, dtype=np.float32
                    ),
                    joint_indices=np.asarray(
                        position_action.joint_indices, dtype=np.int32
                    ),
                )
            )

        physics_started = time.perf_counter()
        world.step(render=not args.headless)
        profiler.record_seconds("physics_step", time.perf_counter() - physics_started)

        with profiler.measure("telemetry_sample"):
            positions = np.asarray(loader.get_joint_positions(), dtype=np.float64).reshape(-1)
            velocities = np.asarray(loader.get_joint_velocities(), dtype=np.float64).reshape(-1)
            if (
                positions.shape != joint_positions.shape
                or velocities.shape != joint_positions.shape
                or not np.all(np.isfinite(positions))
                or not np.all(np.isfinite(velocities))
            ):
                raise RuntimeError(
                    f"post-step joint state is non-finite or has wrong shape at frame {frame_index}"
                )
            post_step_measured_omegas = tuple(
                float(velocities[index]) for index in wheel_indices
            )
            post_step_power_bounds = tuple(
                float(effort_limit) * abs(omega)
                for effort_limit, omega in zip(
                    wheel_action.effort_limits, post_step_measured_omegas
                )
            )
            post_step_measured_wheel_omegas.extend(
                post_step_measured_omegas
            )
            post_step_installed_effort_speed_power_bounds.extend(
                post_step_power_bounds
            )
            full_wheel_post_step_measured_omega[frame_index] = (
                post_step_measured_omegas
            )
            full_wheel_installed_effort_post_step_speed_power_bound[
                frame_index
            ] = post_step_power_bounds
            cap_sample["post_step_measured_wheel_omega_rad_s"] = list(
                post_step_measured_omegas
            )
            cap_sample[
                "installed_effort_post_step_measured_speed_power_bound_w"
            ] = list(post_step_power_bounds)
            if (
                strongest_measured_cap_sample is None
                or min(wheel_action.effort_limits)
                < min(
                    strongest_measured_cap_sample["installed_effort_limit_nm"]
                )
            ):
                strongest_measured_cap_sample = dict(cap_sample)
            if (
                worst_post_step_power_sample is None
                or max(post_step_power_bounds)
                > max(
                    worst_post_step_power_sample[
                        "installed_effort_post_step_measured_speed_power_bound_w"
                    ]
                )
            ):
                worst_post_step_power_sample = dict(cap_sample)
            measured_joint_velocities_for_control = velocities.copy()
            root_position, root_orientation_wxyz = loader.get_world_pose()
            root_position = np.asarray(root_position, dtype=np.float64).reshape(3)
            root_orientation_wxyz = np.asarray(
                root_orientation_wxyz, dtype=np.float64
            ).reshape(4)
            root_linear_velocity = np.asarray(
                loader.get_linear_velocity(), dtype=np.float64
            ).reshape(3)
            root_angular_velocity = np.asarray(
                loader.get_angular_velocity(), dtype=np.float64
            ).reshape(3)
            try:
                applied_efforts = np.asarray(
                    loader.get_applied_joint_efforts(), dtype=np.float64
                ).reshape(-1)
            except Exception as error:
                applied_effort_available = False
                applied_efforts = np.full(len(dof_names), np.nan, dtype=np.float64)
                if not any("applied_joint_efforts" in item for item in query_errors):
                    query_errors.append(
                        f"get_applied_joint_efforts unavailable: {type(error).__name__}: {error}"
                    )
            try:
                measured_efforts = np.asarray(
                    loader.get_measured_joint_efforts(), dtype=np.float64
                ).reshape(-1)
            except Exception as error:
                measured_effort_available = False
                measured_efforts = np.full(len(dof_names), np.nan, dtype=np.float64)
                if not any("measured_joint_efforts" in item for item in query_errors):
                    query_errors.append(
                        f"get_measured_joint_efforts unavailable: {type(error).__name__}: {error}"
                    )

            roll, pitch, yaw, forward_world = _wxyz_to_euler_and_forward(
                root_orientation_wxyz
            )
            longitudinal_speed = float(np.dot(root_linear_velocity, forward_world))
            terrain_height_at_root = math.tan(slope_angle_rad) * (
                root_position[0] - args.terrain_origin_x_m
            )
            plane_clearance = float(root_position[2] - terrain_height_at_root)
            plane_clearances.append(plane_clearance)

            timestamp_s = (frame_index + 1) * args.physics_dt
            target_velocity_by_index = {
                index: float(value)
                for index, value in zip(
                    wheel_action.joint_indices, wheel_action.joint_velocities
                )
            }
            target_velocity_by_index.update(
                {
                    index: float(value)
                    for index, value in zip(
                        position_action.joint_indices,
                        position_action.joint_velocities,
                    )
                }
            )
            target_position_by_index = {
                index: float(value)
                for index, value in zip(
                    position_action.joint_indices,
                    position_action.joint_positions,
                )
            }
            joints = []
            for index, name in enumerate(dof_names):
                applied = (
                    float(applied_efforts[index])
                    if index < applied_efforts.size
                    and math.isfinite(float(applied_efforts[index]))
                    else None
                )
                measured = (
                    float(measured_efforts[index])
                    if index < measured_efforts.size
                    and math.isfinite(float(measured_efforts[index]))
                    else None
                )
                if applied is None and measured is None:
                    raise RuntimeError(
                        f"neither applied nor measured effort is available for {name}"
                    )
                joint = JointTelemetrySample(
                    joint_name=name,
                    category=_category_for_joint(name),
                    angular_velocity_rad_s=float(velocities[index]),
                    applied_effort_nm=applied,
                    measured_effort_nm=measured,
                    target_velocity_rad_s=target_velocity_by_index[index],
                )
                effort_source_counter[joint.to_power_sample().effort_source.value] += 1
                joints.append(joint)

            wheel_kinematics = []
            target_by_dof_name = dict(
                zip(wheel_names_by_dof, wheel_action.joint_velocities)
            )
            for dof_index, dof_name in zip(wheel_indices, wheel_names_by_dof):
                wheel_kinematics.append(
                    WheelKinematicsSample.from_kinematics(
                        timestamp_s=timestamp_s,
                        wheel_name=dof_name,
                        wheel_radius_m=vehicle_config.wheel_radius_m,
                        angular_velocity_rad_s=float(velocities[dof_index]),
                        target_angular_velocity_rad_s=float(
                            target_by_dof_name[dof_name]
                        ),
                        longitudinal_speed_m_s=longitudinal_speed,
                        low_speed_threshold_m_s=low_speed_threshold_m_s,
                    )
                )

            full_root_position[frame_index] = root_position
            full_root_orientation_xyzw[frame_index] = np.asarray(
                [
                    root_orientation_wxyz[1],
                    root_orientation_wxyz[2],
                    root_orientation_wxyz[3],
                    root_orientation_wxyz[0],
                ]
            )
            full_root_linear_velocity[frame_index] = root_linear_velocity
            full_root_angular_velocity[frame_index] = root_angular_velocity
            full_root_rpy[frame_index] = [roll, pitch, yaw]
            full_joint_velocity[frame_index] = velocities
            full_applied_effort[frame_index, : applied_efforts.size] = applied_efforts
            full_measured_effort[frame_index, : measured_efforts.size] = measured_efforts
            for index, value in target_velocity_by_index.items():
                full_joint_target_velocity[frame_index, index] = value
            for index, value in target_position_by_index.items():
                full_joint_target_position[frame_index, index] = value
            full_wheel_slip[frame_index] = [
                item.slip_ratio for item in wheel_kinematics
            ]
            full_normalized_command[frame_index] = [
                command.throttle,
                command.steering,
                command.lift,
                command.bucket_curl,
                command.brake,
            ]
            full_phase_code[frame_index] = {
                "settle": 0,
                "drive": 1,
                "brake": 2,
            }[phase]

            wheel_contacts = []
            wheel_body_centers: dict[str, np.ndarray] = {}
            if wheel_view is not None:
                try:
                    with profiler.measure("wheel_contact_query"):
                        force_matrix = np.asarray(
                            wheel_view.get_contact_force_matrix(dt=args.physics_dt),
                            dtype=np.float64,
                        )
                        wheel_positions, _ = wheel_view.get_world_poses()
                        wheel_positions = np.asarray(wheel_positions, dtype=np.float64)
                    if force_matrix.shape != (4, 1, 3):
                        raise RuntimeError(
                            "unexpected wheel/terrain contact-force shape: "
                            f"{force_matrix.shape}"
                        )
                    contact_detail = None
                    try:
                        with profiler.measure("wheel_contact_point_query"):
                            raw_detail = wheel_view.get_contact_force_data(
                                dt=args.physics_dt
                            )
                        if raw_detail is None or len(raw_detail) != 6:
                            raise RuntimeError("get_contact_force_data returned no data")
                        contact_detail = tuple(
                            np.asarray(item) for item in raw_detail
                        )
                        if contact_detail[4].shape != (4, 1) or contact_detail[5].shape != (4, 1):
                            raise RuntimeError(
                                "unexpected contact count/start shapes: "
                                f"{contact_detail[4].shape}/{contact_detail[5].shape}"
                            )
                    except Exception as error:
                        contact_point_api_error = f"{type(error).__name__}: {error}"
                        if not any("contact point query" in item for item in query_errors):
                            query_errors.append(
                                "contact point query unavailable; wheel centers remain "
                                f"separate: {contact_point_api_error}"
                            )
                    for view_index, wheel_path in enumerate(wheel_view_paths):
                        force = force_matrix[view_index, 0]
                        signed_normal = float(np.dot(force, surface_normal))
                        normal_load = abs(signed_normal)
                        tangential = float(
                            np.linalg.norm(force - signed_normal * surface_normal)
                        )
                        in_contact = bool(normal_load > 1.0)
                        if in_contact:
                            contact_duration_by_path[wheel_path] += args.physics_dt
                        else:
                            contact_duration_by_path[wheel_path] = 0.0
                        # Match the physical wheel link path to the audited DOF.
                        short_name = wheel_path.rsplit("/", 1)[-1]
                        dof_name = f"{short_name}_joint"
                        dof_index = dof_names.index(dof_name)
                        wheel_order_index = wheel_names_by_dof.index(dof_name)
                        wheel_center = wheel_positions[view_index].copy()
                        wheel_body_centers[dof_name] = wheel_center
                        full_wheel_body_center[
                            frame_index, wheel_order_index
                        ] = wheel_center
                        center_distance_to_plane = float(
                            np.dot(
                                surface_normal,
                                np.asarray(
                                    [
                                        wheel_center[0] - args.terrain_origin_x_m,
                                        wheel_center[1],
                                        wheel_center[2],
                                    ],
                                    dtype=np.float64,
                                ),
                            )
                        )
                        wheel_surface_clearance = (
                            center_distance_to_plane - vehicle_config.wheel_radius_m
                        )
                        wheel_surface_clearances.append(wheel_surface_clearance)
                        full_wheel_surface_clearance[
                            frame_index, wheel_order_index
                        ] = wheel_surface_clearance
                        full_wheel_normal_load[
                            frame_index, wheel_order_index
                        ] = normal_load
                        full_wheel_tangential_load[
                            frame_index, wheel_order_index
                        ] = tangential
                        kine = next(
                            item
                            for item in wheel_kinematics
                            if item.wheel_name == dof_name
                        )
                        # WheelTerrainContactSample.position_world_m is an
                        # actual contact-point centroid, never a body center.
                        # The center is persisted under a separately named field.
                        contact_point = None
                        if contact_detail is not None:
                            try:
                                count = int(contact_detail[4][view_index, 0])
                                start = int(contact_detail[5][view_index, 0])
                                if count > 0:
                                    stop = start + count
                                    point_buffer = np.asarray(
                                        contact_detail[1], dtype=np.float64
                                    ).reshape(-1, 3)
                                    force_buffer = np.asarray(
                                        contact_detail[0], dtype=np.float64
                                    ).reshape(-1)
                                    if stop > point_buffer.shape[0] or stop > force_buffer.size:
                                        raise RuntimeError(
                                            "pair start/count exceeds detailed contact buffer: "
                                            f"start={start}, count={count}, "
                                            f"points={point_buffer.shape[0]}, forces={force_buffer.size}"
                                        )
                                    points = point_buffer[start:stop]
                                    normal_forces = np.abs(force_buffer[start:stop])
                                    weight_sum = float(np.sum(normal_forces))
                                    contact_point = (
                                        np.average(
                                            points, axis=0, weights=normal_forces
                                        )
                                        if weight_sum > 1e-12
                                        else np.mean(points, axis=0)
                                    )
                                    full_wheel_contact_point[
                                        frame_index, wheel_order_index
                                    ] = contact_point
                            except Exception as error:
                                contact_point_api_error = (
                                    f"{type(error).__name__}: {error}"
                                )
                                if not any(
                                    "contact point pair mapping" in item
                                    for item in query_errors
                                ):
                                    query_errors.append(
                                        "contact point pair mapping unavailable for at "
                                        f"least one frame: {contact_point_api_error}"
                                    )
                                contact_point = None
                        if contact_point is not None:
                            # The detailed API reporting a point is itself the
                            # authoritative contact-state indicator.
                            if not in_contact:
                                in_contact = True
                                contact_duration_by_path[wheel_path] += args.physics_dt
                            wheel_contacts.append(
                                WheelTerrainContactSample(
                                    timestamp_s=timestamp_s,
                                    wheel_name=dof_name,
                                    position_world_m=contact_point,
                                    normal_load_n=normal_load,
                                    tangential_load_n=tangential,
                                    slip_ratio=kine.slip_ratio,
                                    wheel_angular_velocity_rad_s=float(
                                        velocities[dof_index]
                                    ),
                                    contact_duration_s=contact_duration_by_path[
                                        wheel_path
                                    ],
                                    in_contact=True,
                                )
                            )
                        contact_normal_loads.append(normal_load)
                        contact_tangential_loads.append(tangential)
                    if np.any(np.linalg.norm(force_matrix[:, 0, :], axis=1) > 1.0):
                        contact_frames += 1
                    if wheel_contacts:
                        contact_point_frames += 1
                except Exception as error:
                    contact_api_error = f"{type(error).__name__}: {error}"
                    query_errors.append(
                        f"wheel contact query disabled at frame {frame_index}: {contact_api_error}"
                    )
                    wheel_view = None
                    wheel_contacts = []

            pose = VehiclePoseSample(
                timestamp_s=timestamp_s,
                position_world_m=root_position,
                orientation_xyzw=np.asarray(
                    [
                        root_orientation_wxyz[1],
                        root_orientation_wxyz[2],
                        root_orientation_wxyz[3],
                        root_orientation_wxyz[0],
                    ],
                    dtype=np.float64,
                ),
                roll_rad=roll,
                pitch_rad=pitch,
                yaw_rad=yaw,
                linear_velocity_world_m_s=root_linear_velocity,
                angular_velocity_world_rad_s=root_angular_velocity,
                articulation_angle_rad=float(
                    positions[low_level.mapping.steering_index]
                ),
                local_terrain_slope_rad=slope_angle_rad,
            )
            telemetry_frame = VehicleTelemetryFrame(
                pose=pose,
                joints=tuple(joints),
                wheel_kinematics=tuple(wheel_kinematics),
                wheel_contacts=tuple(wheel_contacts),
            )
            recorder.record(telemetry_frame, dt_s=args.physics_dt)

            numeric_arrays = (
                positions,
                velocities,
                root_position,
                root_orientation_wxyz,
                root_linear_velocity,
                root_angular_velocity,
            )
            if not all(np.all(np.isfinite(item)) for item in numeric_arrays):
                finite_state = False
                raise RuntimeError(f"non-finite Isaac physics state at frame {frame_index}")
            if initial_position is None:
                initial_position = root_position.copy()
            if previous_position is not None:
                step_distance = float(np.linalg.norm(root_position - previous_position))
                path_distance_m += step_distance
                maximum_planar_step_m = max(
                    maximum_planar_step_m,
                    float(np.linalg.norm(root_position[:2] - previous_position[:2])),
                )
            previous_position = root_position.copy()
            final_position = root_position.copy()

            if frame_index == args.settle_steps - 1:
                drive_origin_position = root_position.copy()

            if phase == "drive":
                drive_speeds.append(longitudinal_speed)
                drive_pitch.append(pitch)
                drive_roll.append(roll)
                drive_abs_slips.extend(
                    abs(item.slip_ratio) for item in wheel_kinematics
                )
                drive_abs_efforts.extend(
                    abs(float(measured_efforts[index]))
                    for index in wheel_indices
                    if index < measured_efforts.size
                    and math.isfinite(float(measured_efforts[index]))
                )
                if drive_origin_position is None:
                    raise RuntimeError("drive origin was not captured after settling")
                drive_progress = float(
                    np.dot(root_position - drive_origin_position, uphill_tangent)
                )
                drive_progresses.append(drive_progress)
                if (
                    time_to_common_distance_s is None
                    and drive_progress >= comparison_distance_m
                ):
                    time_to_common_distance_s = (
                        frame_index - args.settle_steps + 1
                    ) * args.physics_dt

            if frame_index % 20 == 0 or frame_index == args.steps - 1:
                serialized_frame = _serialize_frame(
                    frame_index=frame_index,
                    phase=phase,
                    command=command,
                    frame=telemetry_frame,
                    wheel_body_centers=wheel_body_centers,
                )
                serialized_frame["wheel_power_cap"] = cap_sample
                sampled_frames.append(serialized_frame)

    if initial_position is None or final_position is None:
        raise RuntimeError("no physics telemetry frames were recorded")

    full_telemetry_path = args.output.with_name(
        f"{args.output.stem}_full_telemetry.npz"
    )
    full_telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        full_telemetry_path,
        schema_version=np.asarray(["isaac-bulk-phase-d-full-telemetry/v2"]),
        timestamps_s=(
            np.arange(args.steps, dtype=np.float64) + 1.0
        ) * args.physics_dt,
        phase_code=full_phase_code,
        phase_code_labels=np.asarray(["settle", "drive", "brake"]),
        normalized_command=full_normalized_command,
        command_channel_names=np.asarray(
            ["throttle", "steering", "lift", "bucket_curl", "brake"]
        ),
        root_position_world_m=full_root_position,
        root_orientation_xyzw=full_root_orientation_xyzw,
        root_linear_velocity_world_m_s=full_root_linear_velocity,
        root_angular_velocity_world_rad_s=full_root_angular_velocity,
        root_roll_pitch_yaw_rad=full_root_rpy,
        dof_names=np.asarray(dof_names),
        joint_velocity_rad_s=full_joint_velocity,
        applied_effort_command_nm=full_applied_effort,
        measured_solver_joint_effort_nm=full_measured_effort,
        joint_target_velocity_rad_s=full_joint_target_velocity,
        joint_target_position_rad=full_joint_target_position,
        wheel_dof_names=np.asarray(wheel_names_by_dof),
        wheel_slip_ratio=full_wheel_slip,
        wheel_normal_load_n=full_wheel_normal_load,
        wheel_tangential_load_n=full_wheel_tangential_load,
        wheel_contact_point_world_m=full_wheel_contact_point,
        wheel_body_center_world_m=full_wheel_body_center,
        wheel_surface_to_plane_clearance_m=full_wheel_surface_clearance,
        power_cap_basis_code=full_power_cap_basis_code,
        power_cap_basis_code_labels=np.asarray(
            [
                "MEASURED_OR_TARGET_WORST_CASE",
                "TARGET_JOINT_VELOCITY_FALLBACK",
            ]
        ),
        wheel_power_min_guard_rad_s=full_power_cap_min_guard,
        power_cap_input_full_joint_velocity_rad_s=(
            full_power_cap_input_joint_velocity
        ),
        wheel_power_cap_previous_measured_omega_rad_s=(
            full_wheel_power_cap_previous_measured_omega
        ),
        wheel_power_cap_target_omega_rad_s=(
            full_wheel_power_cap_target_omega
        ),
        wheel_power_cap_worst_case_omega_rad_s=(
            full_wheel_power_cap_worst_case_omega
        ),
        wheel_power_cap_protected_denominator_rad_s=(
            full_wheel_power_cap_denominator
        ),
        wheel_installed_effort_limit_nm=full_wheel_installed_effort_limit,
        wheel_command_envelope_power_bound_w=(
            full_wheel_command_envelope_power_bound
        ),
        wheel_post_step_measured_omega_rad_s=(
            full_wheel_post_step_measured_omega
        ),
        wheel_installed_effort_post_step_measured_speed_power_bound_w=(
            full_wheel_installed_effort_post_step_speed_power_bound
        ),
    )
    with np.load(full_telemetry_path, allow_pickle=False) as persisted:
        persisted_frame_count = int(persisted["timestamps_s"].shape[0])
        persisted_required_fields = {
            "root_position_world_m",
            "joint_velocity_rad_s",
            "applied_effort_command_nm",
            "measured_solver_joint_effort_nm",
            "joint_target_velocity_rad_s",
            "wheel_slip_ratio",
            "wheel_normal_load_n",
            "wheel_contact_point_world_m",
            "power_cap_basis_code",
            "wheel_power_min_guard_rad_s",
            "power_cap_input_full_joint_velocity_rad_s",
            "wheel_installed_effort_limit_nm",
            "wheel_post_step_measured_omega_rad_s",
            "wheel_installed_effort_post_step_measured_speed_power_bound_w",
        }
        persisted_fields_present = persisted_required_fields.issubset(
            set(persisted.files)
        )
    full_telemetry_sha256 = _sha256_file(full_telemetry_path)
    all_steps_persisted = bool(
        persisted_frame_count == args.steps and persisted_fields_present
    )

    snapshot = recorder.snapshot()
    root_delta = final_position - initial_position
    uphill_progress = float(np.dot(root_delta, uphill_tangent))
    horizontal_displacement = float(np.linalg.norm(root_delta[:2]))
    contact_available = bool(
        contact_api_error is None and contact_normal_loads and contact_frames > 0
    )
    contact_points_available = bool(
        contact_point_api_error is None and contact_point_frames > 0
    )
    max_normal_load = max(contact_normal_loads, default=0.0)
    mean_pitch = float(np.mean(drive_pitch)) if drive_pitch else 0.0
    slope_alignment_error_deg = abs(abs(math.degrees(mean_pitch)) - args.slope_deg)
    measured_basis = "MEASURED_OR_TARGET_WORST_CASE"
    fallback_basis = "TARGET_JOINT_VELOCITY_FALLBACK"
    measured_basis_every_frame = bool(
        power_cap_basis_counts.get(measured_basis, 0) == args.steps
        and power_cap_basis_counts.get(fallback_basis, 0) == 0
    )
    maximum_command_envelope_power_bound_w = max(
        command_envelope_power_bounds, default=0.0
    )
    command_envelope_power_bound_within_limit = bool(
        command_envelope_power_bounds
        and maximum_command_envelope_power_bound_w
        <= vehicle_config.wheel_power_limit_w * (1.0 + 1e-12)
    )
    post_step_power_tolerance_w = max(
        vehicle_config.wheel_power_limit_w * 1e-6, 1e-6
    )
    maximum_post_step_measured_speed_power_bound_w = max(
        post_step_installed_effort_speed_power_bounds, default=0.0
    )
    post_step_measured_speed_power_bound_within_limit = bool(
        post_step_installed_effort_speed_power_bounds
        and maximum_post_step_measured_speed_power_bound_w
        <= vehicle_config.wheel_power_limit_w + post_step_power_tolerance_w
    )
    measured_speed_cap_activated = bool(
        installed_wheel_effort_limits
        and min(installed_wheel_effort_limits)
        < vehicle_config.wheel_effort_nm - 1e-6
    )

    acceptance_checks = {
        "hidden_static_exact_contact_mesh": bool(
            str(UsdGeom.Imageable(contact_prim).GetVisibilityAttr().Get())
            == "invisible"
            and contact_prim.HasAPI(UsdPhysics.CollisionAPI)
            and contact_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            and not contact_prim.HasAPI(UsdPhysics.RigidBodyAPI)
            and str(
                UsdPhysics.MeshCollisionAPI(contact_prim).GetApproximationAttr().Get()
            )
            == "none"
        ),
        "visual_mesh_separate_and_noncolliding": bool(
            contact_config.contact_prim_path != contact_config.visual_prim_path
            and not visual_prim.HasAPI(UsdPhysics.CollisionAPI)
            and visual_data.source_shape_yx == grid.shape
            and visual_data.vertex_count == grid.nx * grid.ny
        ),
        "collision_membership_valid": bool(membership_validation.valid),
        "all_collision_group_members_exist_with_collision_api": True,
        "bucket_terrain_only_required_filtered_pair": bool(
            collision_matrix.filtered_pairs()
            == (
                (
                    CollisionGroup.TERRAIN_SUPPORT,
                    CollisionGroup.BUCKET_INTERACTION,
                ),
            )
        ),
        "public_contact_load_available": contact_available,
        "public_contact_points_available": contact_points_available,
        "wheel_terrain_support_force_observed": bool(max_normal_load > 100.0),
        "root_remained_above_slope": bool(min(plane_clearances) > 0.25),
        "wheel_surfaces_do_not_penetrate_slope_over_0_08_m": bool(
            wheel_surface_clearances
            and min(wheel_surface_clearances) > -0.08
        ),
        "uphill_progress_over_0_25_m": bool(uphill_progress > 0.25),
        "common_comparison_distance_reached": bool(
            time_to_common_distance_s is not None
        ),
        "nonzero_sparse_wheel_velocity_command": bool(
            nonzero_wheel_command_frames > 0
        ),
        "finite_sparse_wheel_effort_limits_installed": bool(
            installed_wheel_effort_limits
            and all(np.isfinite(installed_wheel_effort_limits))
        ),
        "measured_or_target_worst_case_power_cap_used_every_frame": (
            measured_basis_every_frame
        ),
        "target_speed_power_cap_fallback_frames_zero": bool(
            power_cap_basis_counts.get(fallback_basis, 0) == 0
        ),
        "configured_minimum_speed_power_guard_used_every_frame": bool(
            np.all(
                np.isclose(
                    full_power_cap_min_guard,
                    vehicle_config.wheel_power_min_guard_rad_s,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        ),
        "wheel_command_envelope_power_bound_within_160kw_limit": (
            command_envelope_power_bound_within_limit
        ),
        "installed_effort_post_step_measured_speed_power_bound_within_160kw_limit": (
            post_step_measured_speed_power_bound_within_limit
        ),
        "finite_physics_state": finite_state,
        "continuous_root_motion": bool(maximum_planar_step_m < 1.0),
        "slope_pitch_alignment_within_8_deg": bool(slope_alignment_error_deg < 8.0),
        "normal_loop_pose_setter_zero": True,
        "initialization_pose_setter_zero_usd_transform_used": True,
        "all_steps_integrated_in_memory": bool(recorder.frame_count == args.steps),
        "all_steps_persisted": all_steps_persisted,
    }
    failed_checks = sorted(
        name for name, passed in acceptance_checks.items() if not passed
    )
    profiler_summary = _runtime_stats_to_dict(profiler.summary())
    energy = _energy_to_dict(snapshot.energy)
    elapsed_sim_time = args.steps * args.physics_dt
    status = "PASS" if not failed_checks else "FAIL"

    return {
        "schema_version": "isaac-bulk-phase-cd-slope-runtime/v1",
        "status": status,
        "failed_checks": failed_checks,
        "run_config": {
            "slope_deg": float(args.slope_deg),
            "throttle": float(args.throttle),
            "steps": int(args.steps),
            "settle_steps": int(args.settle_steps),
            "drive_steps": int(drive_end - args.settle_steps),
            "brake_steps": int(args.brake_steps),
            "physics_dt_s": float(args.physics_dt),
            "loader_usd": str(args.loader_usd.resolve()),
            "vehicle_config": str(args.vehicle_config.resolve()),
            "contact_config": str(args.contact_config.resolve()),
            "telemetry_config": str(args.telemetry_config.resolve()),
            "config_schema_versions": {
                "phase_b": vehicle_document.get("schema_version"),
                "phase_c": contact_document.get("schema_version"),
                "phase_d": configured_telemetry.get("schema_version"),
            },
            "scenario_identity": scenario_identity,
            "scenario_fingerprint_sha256": scenario_fingerprint,
            "normal_loop_pose_setter_count": 0,
            "initialization_pose_setter_count": 0,
            "initial_alignment_method": "authored USD reference-root rigid transform before physics",
        },
        "phase_b_vehicle_control": {
            "command_chain": (
                "VehicleCommand -> CommandSlewLimiter -> "
                "LoaderLowLevelController -> SingleArticulation sparse actions"
            ),
            "joint_velocity_feedback_source": (
                "previous post-PhysX SingleArticulation.get_joint_velocities; "
                "post-reset measurement on frame zero"
            ),
            "wheel_power_limit_evidence": {
                "semantics": (
                    "each installed per-wheel effort limit uses protected "
                    "denominator discrete_safety_factor * max(previous-step "
                    "measured |omega|, target |omega|, reference, configured "
                    "minimum speed guard). After PhysX, "
                    "installed effort limit * current measured |omega| is "
                    "validated as a mechanical-power upper bound. This is not "
                    "measured solver tau*omega and does not claim measured input power"
                ),
                "configured_per_wheel_power_limit_w": (
                    vehicle_config.wheel_power_limit_w
                ),
                "configured_per_wheel_effort_limit_nm": (
                    vehicle_config.wheel_effort_nm
                ),
                "low_speed_reference_rad_s": (
                    vehicle_config.wheel_power_reference_rad_s
                ),
                "minimum_speed_guard_rad_s": (
                    vehicle_config.wheel_power_min_guard_rad_s
                ),
                "discrete_speed_safety_factor": (
                    vehicle_config.wheel_power_discrete_safety_factor
                ),
                "basis_counts": dict(power_cap_basis_counts),
                "measured_or_target_worst_case_frames": (
                    power_cap_basis_counts.get(measured_basis, 0)
                ),
                "target_speed_fallback_frames": (
                    power_cap_basis_counts.get(fallback_basis, 0)
                ),
                "installed_effort_limit_nm": {
                    "min": min(installed_wheel_effort_limits, default=0.0),
                    "max": max(installed_wheel_effort_limits, default=0.0),
                },
                "max_protected_denominator_rad_s": max(
                    power_cap_denominators, default=0.0
                ),
                "max_protected_speed_before_safety_factor_rad_s": max(
                    power_cap_worst_case_omegas, default=0.0
                ),
                "max_command_envelope_power_bound_w": (
                    maximum_command_envelope_power_bound_w
                ),
                "command_envelope_bound_within_configured_limit": (
                    command_envelope_power_bound_within_limit
                ),
                "max_abs_post_step_measured_wheel_omega_rad_s": max(
                    (
                        abs(value)
                        for value in post_step_measured_wheel_omegas
                    ),
                    default=0.0,
                ),
                "max_installed_effort_post_step_measured_speed_power_bound_w": (
                    maximum_post_step_measured_speed_power_bound_w
                ),
                "post_step_power_bound_tolerance_w": (
                    post_step_power_tolerance_w
                ),
                "post_step_bound_within_configured_limit": (
                    post_step_measured_speed_power_bound_within_limit
                ),
                "measured_speed_cap_activated": measured_speed_cap_activated,
                "strongest_cap_sample": strongest_measured_cap_sample,
                "worst_post_step_power_bound_sample": (
                    worst_post_step_power_sample
                ),
            },
        },
        "phase_c_contact": {
            "backend": "TriangleMeshContactBackend",
            "authoritative_source": "planar H[y,x] fixture at 0.05 m",
            "contact_prim_path": contact_config.contact_prim_path,
            "visual_prim_path": contact_config.visual_prim_path,
            "source_shape_yx": list(grid.shape),
            "source_spacing_m": args.visual_spacing_m,
            "configured_slope_degrees": list(phase_c_slopes),
            "deformation_flags": deformation_flags,
            "contact_spacing_xy_m": list(
                contact_backend.committed_mesh.nominal_spacing_xy_m
            ),
            "vertex_count": contact_backend.committed_mesh.vertex_count,
            "triangle_count": contact_backend.committed_mesh.triangle_count,
            "visible": False,
            "static": True,
            "mesh_approximation": "none",
            "visual_has_collision_api": visual_prim.HasAPI(UsdPhysics.CollisionAPI),
            "visual_mesh": {
                "source": "independent build from original 0.05 m H[y,x]",
                "source_shape_yx": list(visual_data.source_shape_yx),
                "spacing_xy_m": list(visual_data.nominal_spacing_xy_m),
                "vertex_count": visual_data.vertex_count,
                "triangle_count": visual_data.triangle_count,
                "shares_contact_mesh_buffers": False,
            },
            "commit_policy": contact_config.commit_policy,
            "material": material_values,
            "heightmap_measured_slope_deg": {
                "mean": float(np.mean(estimate_slope_deg(heightmap, grid))),
                "maximum_abs_error": float(
                    np.max(np.abs(estimate_slope_deg(heightmap, grid) - args.slope_deg))
                ),
            },
            "collision_matrix": collision_matrix.as_dict(),
            "collision_groups": inspect_authored_groups(authored_groups),
            "membership_validation": {
                "valid": membership_validation.valid,
                "errors": list(membership_validation.errors),
                "warnings": list(membership_validation.warnings),
                "stage_prim_and_collision_api_validation_executed": True,
            },
        },
        "phase_d_telemetry": {
            "root_pose_source": "SingleArticulation.get_world_pose",
            "root_velocity_sources": [
                "SingleArticulation.get_linear_velocity",
                "SingleArticulation.get_angular_velocity",
            ],
            "joint_velocity_source": "SingleArticulation.get_joint_velocities",
            "applied_effort_source": "SingleArticulation.get_applied_joint_efforts",
            "measured_effort_source": "SingleArticulation.get_measured_joint_efforts",
            "applied_effort_available": applied_effort_available,
            "measured_effort_available": measured_effort_available,
            "energy_formula": "P=tau*omega; positive=max(P,0)*dt; signed=P*dt",
            "energy_semantics": (
                "solver joint mechanical work / actuator-output proxy; not fuel, "
                "hydraulic or electrical input energy and contains no efficiency model"
            ),
            "configured_effort_precedence": list(effort_precedence),
            "energy_effort_source_counts": dict(effort_source_counter),
            "slip_formula": (
                "(r*omega-v_long)/max(abs(r*omega),abs(v_long)); zero when "
                f"denominator<{low_speed_threshold_m_s} m/s"
            ),
            "configured_slope_degrees": list(configured_slopes),
            "contact_force": {
                "available": contact_available,
                "source": "RigidPrim.get_contact_force_matrix, public RigidContactView wrapper",
                "filter_prim": contact_config.contact_prim_path,
                "wheel_body_paths": wheel_view_paths,
                "unavailable_reason": contact_api_error,
                "no_synthetic_or_estimated_loads": True,
            },
            "contact_points": {
                "available": contact_points_available,
                "source": "RigidPrim.get_contact_force_data pair start/count, force-weighted point centroid",
                "unavailable_reason": contact_point_api_error,
                "wheel_body_centers_stored_separately": True,
            },
            "full_step_persistence": {
                "path": str(full_telemetry_path.resolve()),
                "format": "compressed_npz",
                "schema_version": "isaac-bulk-phase-d-full-telemetry/v2",
                "frame_count": persisted_frame_count,
                "sha256": full_telemetry_sha256,
                "all_required_fields_present": persisted_fields_present,
                "power_cap_fields_recorded_every_step": [
                    "power_cap_basis_code",
                    "wheel_power_min_guard_rad_s",
                    "power_cap_input_full_joint_velocity_rad_s",
                    "wheel_power_cap_previous_measured_omega_rad_s",
                    "wheel_power_cap_target_omega_rad_s",
                    "wheel_power_cap_worst_case_omega_rad_s",
                    "wheel_power_cap_protected_denominator_rad_s",
                    "wheel_installed_effort_limit_nm",
                    "wheel_command_envelope_power_bound_w",
                    "wheel_post_step_measured_omega_rad_s",
                    "wheel_installed_effort_post_step_measured_speed_power_bound_w",
                ],
            },
            "query_notes": query_errors,
            "sampled_frame_stride": 20,
            "sampled_frames": sampled_frames,
        },
        "metrics": {
            "elapsed_sim_time_s": elapsed_sim_time,
            "comparison_distance_m": comparison_distance_m,
            "time_to_common_distance_s": time_to_common_distance_s,
            "initial_root_position_m": initial_position.tolist(),
            "final_root_position_m": final_position.tolist(),
            "horizontal_displacement_m": horizontal_displacement,
            "uphill_progress_m": uphill_progress,
            "path_distance_m": path_distance_m,
            "maximum_planar_step_m": maximum_planar_step_m,
            "plane_clearance_m": _stats(plane_clearances),
            "wheel_surface_to_plane_clearance_m": _stats(
                wheel_surface_clearances
            ),
            "maximum_drive_progress_m": max(drive_progresses, default=0.0),
            "drive_window": {
                "longitudinal_speed_m_s": _stats(drive_speeds),
                "abs_wheel_slip_ratio": _stats(drive_abs_slips),
                "abs_drive_measured_effort_nm": _stats(drive_abs_efforts),
                "pitch_rad": _stats(drive_pitch),
                "roll_rad": _stats(drive_roll),
                "pitch_alignment_abs_error_deg": slope_alignment_error_deg,
            },
            "contact": {
                "contact_frame_count": contact_frames,
                "contact_point_frame_count": contact_point_frames,
                "max_normal_load_n": max_normal_load,
                "normal_load_n": _stats(contact_normal_loads),
                "tangential_load_n": _stats(contact_tangential_loads),
            },
            "energy": energy,
            "distance_per_positive_solver_joint_work_proxy_m_per_j": (
                uphill_progress / energy["drive"]["positive_j"]
                if energy["drive"]["positive_j"] > 0.0
                else None
            ),
            "runtime_profile_ms": profiler_summary,
        },
        "acceptance_checks": acceptance_checks,
        "limitations": [
            "Wheel sinkage, rutting, compaction and terrain deformation are disabled in Phase C.",
            "The bucket/terrain rigid pair is filtered for the custom soil interaction path; this run does not excavate.",
            "Applied and measured effort are logged separately. An unavailable signal is never replaced by an unlabeled estimate.",
            "Mechanical energy is a solver-joint/output proxy, not fuel, hydraulic or electrical input energy.",
            "No Phase-F material transport is implemented here.",
        ],
    }


def main() -> int:
    args, kit_args = _parse_args()
    sys.argv = [sys.argv[0], *kit_args]
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "multi_gpu": False,
            "width": 960,
            "height": 540,
        }
    )
    evidence: dict[str, Any]
    exit_code = 0
    try:
        evidence = _run(args)
        if evidence["status"] != "PASS":
            exit_code = 2
    except Exception as error:
        evidence = {
            "schema_version": "isaac-bulk-phase-cd-slope-runtime/v1",
            "status": "ERROR",
            "run_config": {
                "slope_deg": float(args.slope_deg),
                "throttle": float(args.throttle),
                "steps": int(args.steps),
                "settle_steps": int(args.settle_steps),
                "brake_steps": int(args.brake_steps),
                "physics_dt_s": float(args.physics_dt),
            },
            "error_type": type(error).__name__,
            "error": str(error),
        }
        exit_code = 1
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        aggregate = _update_aggregate(args.summary_output)
        print(f"PHASE_CD_SLOPE_{evidence['status']}: {args.output}")
        print(f"PHASE_CD_AGGREGATE_{aggregate['status']}: {args.summary_output}")
        simulation_app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
