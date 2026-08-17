#!/usr/bin/env python3
"""Physical-timescale attribution using the production GPU Earthmoving Core.

The runner changes no constitutive or numerical physics setting.  Full-field
reads occur only at named diagnostic/checkpoint boundaries and are excluded
from the normal production-step transfer claim.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (warp_root / "warp").is_dir():
        sys.path.insert(0, str(warp_root))
        break

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    LargeAvalancheTransitionConfig,
    TrackSoilConfig,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario  # noqa: E402
from isaac_bulk_pipeline.runtime import EarthmovingPhysicsCore, SoilForceMode  # noqa: E402
from isaac_bulk_pipeline.runtime.device_checkpoint import (  # noqa: E402
    restore_device_checkpoint,
    sha256_file,
    write_device_checkpoint,
)
from isaac_bulk_pipeline.runtime.gpu_runtime_metadata import (  # noqa: E402
    GpuRuntimeMetadata,
)
from isaac_bulk_pipeline.runtime.v2_config import Interactive390FConfig  # noqa: E402
from isaac_bulk_pipeline.terrain import HeightmapIO, TerrainGrid  # noqa: E402
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState  # noqa: E402


CONFIG_PATH = ROOT / "configs/390f_v2_interactive.yaml"
FORMAL_TERRAIN = ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_track_pile_acceptance_m.csv"
STABLE_TERRAIN = ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_initial_m.csv"
INTERRUPTED_RUN = ROOT / "outputs/390f_v2/interactive_runs/run_1786506718"
OUTPUT_ROOT = ROOT / "outputs/390f_v2"
DUMP_AFTER_CHECKPOINT = OUTPUT_ROOT / "formal_dump_after_device_checkpoint.npz"
DUMP_RELEASE_CHECKPOINT = OUTPUT_ROOT / "formal_dump_release_device_checkpoint.npz"
FORMAL_SCENARIO = "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED"

CASE_FILES = {
    "passive": (
        OUTPUT_ROOT / "track_pile_passive_timeseries.json",
        OUTPUT_ROOT / "track_pile_passive_summary.json",
    ),
    "dump": (
        OUTPUT_ROOT / "track_pile_dump_timeseries.json",
        OUTPUT_ROOT / "track_pile_dump_summary.json",
    ),
    "stable": (
        OUTPUT_ROOT / "stable_pile_dump_timeseries.json",
        OUTPUT_ROOT / "stable_pile_dump_summary.json",
    ),
}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _material(path: Path, scenario_id: str) -> tuple[MaterialScenario, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    source = next(item for item in document["scenarios"] if item["id"] == scenario_id)
    material = MaterialScenario(
        source["id"],
        source["bulk_density_kg_m3"],
        source["internal_friction_angle_deg"],
        source["cohesion_pa"],
        float(np.tan(np.deg2rad(source["tool_wall_friction_angle_deg"]))),
        source["theta_start_deg"],
        source["theta_stop_deg"],
        source["mobile_friction_coefficient"],
    )
    return material, source


def _grid(config: Interactive390FConfig) -> TerrainGrid:
    transform = np.eye(4)
    transform[:3, 3] = config.terrain_translation_world_m
    return TerrainGrid(
        701,
        701,
        config.grid_spacing_m,
        config.grid_spacing_m,
        0.0,
        0.0,
        "/World/Terrain/TimescaleAttribution",
        terrain_to_world_matrix=transform,
    )


def _core(terrain_path: Path) -> tuple[EarthmovingPhysicsCore, Interactive390FConfig, dict[str, Any]]:
    config = Interactive390FConfig.load(CONFIG_PATH)
    if config.runtime_backend != "GPU_RUNTIME":
        raise RuntimeError("TIMESCALE_ATTRIBUTION_REQUIRES_GPU_RUNTIME")
    if config.grid_shape != (701, 701) or config.grid_spacing_m != 0.05:
        raise RuntimeError("FORMAL_GRID_IDENTITY_MISMATCH")
    grid = _grid(config)
    terrain = HeightmapIO.load(terrain_path, grid=grid, source_axis_order="xy")
    material, material_source = _material(
        config.material_scenarios, config.material_scenario_id
    )
    if material.name != FORMAL_SCENARIO:
        raise RuntimeError("FORMAL_MATERIAL_IDENTITY_MISMATCH")
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(config.bucket_descriptor)
    )
    core = EarthmovingPhysicsCore(
        grid=grid,
        descriptor=descriptor,
        material=material,
        initial_heightmap_m=terrain,
        runtime_backend=config.runtime_backend,
        solver_backend=config.solver_backend,
        slope_backend=config.slope_backend,
        tile_size=config.active_tile_size,
        large_avalanche_iteration_threshold=config.large_avalanche_iteration_threshold,
        numerical_safety_max_iterations=config.numerical_safety_max_iterations,
        minislope_round_budget_per_step=config.minislope_round_budget_per_step,
        minislope_tolerance_m=config.minislope_tolerance_m,
        large_avalanche_config=LargeAvalancheTransitionConfig.from_mapping(
            dict(config.large_avalanche_transition)
        ),
        track_soil_config=TrackSoilConfig(
            sinkage_rate_m_s=float(config.track_soil["sinkage_rate_m_s"]),
            slip_gain=float(config.track_soil["slip_gain"]),
            maximum_sinkage_per_step_m=float(
                config.track_soil["maximum_sinkage_per_step_m"]
            ),
            maximum_total_rut_depth_m=float(
                config.track_soil["maximum_total_rut_depth_m"]
            ),
            shoulder_halo_cells=int(config.track_soil["shoulder_halo_cells"]),
            minimum_slip_speed_m_s=float(
                config.track_soil["minimum_slip_speed_m_s"]
            ),
            parameter_status=str(config.track_soil["parameter_status"]),
        ),
    )
    return core, config, material_source


def _stationary_tool(core: EarthmovingPhysicsCore, timestamp_s: float) -> ToolState:
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.0, 20.0]

    def transform(points: np.ndarray) -> np.ndarray:
        return (pose @ np.c_[points, np.ones(len(points))].T).T[:, :3]

    descriptor = core.descriptor
    return ToolState(
        timestamp_s,
        pose,
        pose,
        transform(descriptor.cutting_edge_local),
        transform(descriptor.bottom_profile_local),
        transform(descriptor.left_boundary_local),
        transform(descriptor.right_boundary_local),
        np.zeros(3),
        np.zeros(3),
    )


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0 or float(np.sum(weights)) <= 0.0:
        return 0.0
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    target = float(q) * float(np.sum(weights))
    return float(values[min(int(np.searchsorted(np.cumsum(weights), target)), values.size - 1)])


def _spatial_diagnostic(
    core: EarthmovingPhysicsCore,
    *,
    previous_sample_time_s: float,
    trigger_xy_m: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Explicit attribution boundary; not part of the normal transfer path."""

    state = core.device_state
    if state is None:
        raise RuntimeError("DEVICE_STATE_REQUIRED")
    rt = state.runtime
    mobile = np.asarray(rt.download("mobile"), dtype=np.float64).reshape(state.shape)
    mx = np.asarray(rt.download("momentum_x"), dtype=np.float64).reshape(state.shape)
    my = np.asarray(rt.download("momentum_y"), dtype=np.float64).reshape(state.shape)
    activation = np.asarray(
        rt.download("avalanche_activation_count"), dtype=np.int32
    ).reshape(state.shape)
    first = np.asarray(
        rt.download("avalanche_first_activation_time"), dtype=np.float64
    ).reshape(state.shape)
    speed = np.divide(
        np.hypot(mx, my),
        mobile,
        out=np.zeros_like(mobile),
        where=mobile > core.avalanche_controller.config.dry_tolerance_m,
    )
    weights = core.integrator.vertex_weights_m2
    volume_weights = mobile * weights
    mobile_mask = mobile > core.avalanche_controller.config.dry_tolerance_m
    moving_mask = speed >= core.avalanche_controller.config.mobile_activity_speed_m_s
    mobile_volume = float(np.sum(volume_weights))
    moving_volume = float(np.sum(volume_weights[moving_mask]))
    mean_speed = (
        float(np.sum(speed * volume_weights) / mobile_volume)
        if mobile_volume > 0.0
        else 0.0
    )
    active = activation > 0
    newly = active & (first > previous_sample_time_s + 1.0e-12)
    rows, cols = np.nonzero(active)
    new_rows, new_cols = np.nonzero(newly)
    unique_area = float(np.sum(weights[active]))
    new_area = float(np.sum(weights[newly]))
    if trigger_xy_m is None and new_rows.size:
        trigger_xy_m = np.asarray(
            [float(np.mean(new_cols) * core.grid.dx), float(np.mean(new_rows) * core.grid.dy)]
        )
    front_centroid = (
        [float(np.mean(new_cols) * core.grid.dx), float(np.mean(new_rows) * core.grid.dy)]
        if new_rows.size
        else None
    )
    bbox = (
        [
            float(np.min(cols) * core.grid.dx),
            float(np.min(rows) * core.grid.dy),
            float(np.max(cols) * core.grid.dx),
            float(np.max(rows) * core.grid.dy),
        ]
        if rows.size
        else None
    )
    max_distance = 0.0
    if trigger_xy_m is not None and rows.size:
        max_distance = float(
            np.max(
                np.hypot(
                    cols * core.grid.dx - trigger_xy_m[0],
                    rows * core.grid.dy - trigger_xy_m[1],
                )
            )
        )
    values = speed[mobile_mask]
    value_weights = volume_weights[mobile_mask]
    return {
        "mobile_volume_m3": mobile_volume,
        "moving_mobile_volume_m3": moving_volume,
        "mobile_mean_velocity_m_s": mean_speed,
        "mobile_velocity_p50_m_s": _weighted_quantile(values, value_weights, 0.50),
        "mobile_velocity_p95_m_s": _weighted_quantile(values, value_weights, 0.95),
        "mobile_velocity_p99_m_s": _weighted_quantile(values, value_weights, 0.99),
        "mobile_velocity_max_m_s_diagnostic_only": float(np.max(values)) if values.size else 0.0,
        "unique_activated_area_m2": unique_area,
        "newly_activated_area_m2": new_area,
        "front_centroid_xy_m": front_centroid,
        "front_bounding_box_xyxy_m": bbox,
        "maximum_propagation_distance_m": max_distance,
        "diagnostic_transfer_boundary": "EXPLICIT_FULL_FIELD_ACCEPTANCE_SAMPLE",
    }, trigger_xy_m


def _read_checkpoint_record(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["checkpoint_record_json"]))


def _install_equivalent_dump(
    core: EarthmovingPhysicsCore,
    release_checkpoint: Path,
) -> dict[str, Any]:
    record = _read_checkpoint_record(release_checkpoint)
    metadata_record = dict(record["metadata"])
    release_time = float(metadata_record["timestamp_s"])
    metadata_record["timestamp_s"] = 0.0
    for parcel in metadata_record.get("airborne", []):
        parcel["timestamp_s"] = max(0.0, float(parcel["timestamp_s"]) - release_time)
    assert core.device_state is not None
    core.gpu_metadata = GpuRuntimeMetadata.from_checkpoint_record(
        core.device_state,
        metadata_record,
        preserve_ledger_reference=False,
    )
    return {
        "source_checkpoint": str(release_checkpoint.relative_to(ROOT)),
        "source_checkpoint_hash": sha256_file(release_checkpoint),
        "airborne_volume_m3": sum(
            float(parcel["volume_m3"])
            for parcel in metadata_record.get("airborne", [])
        ),
        "airborne_parcel_count": len(metadata_record.get("airborne", [])),
        "spatial_footprint": "SAME_PRODUCTION_AIRBORNE_PARCELS_AND_WORLD_XY",
        "temporal_profile": "SAME_PRODUCTION_BALLISTIC_PARCELS_AND_PHYSICS_DT",
    }


def _initial_accumulators() -> dict[str, Any]:
    return {
        "elapsed_s": 0.0,
        "wall_time_s": 0.0,
        "T_large_avalanche_active_s": 0.0,
        "T_mobile_nonzero_s": 0.0,
        "T_mobile_actively_moving_s": 0.0,
        "T_deposition_dominated_s": 0.0,
        "T_residual_minislope_only_s": 0.0,
        "T_quiet_to_settled_s": 0.0,
        "mobile_travel_distance_m": 0.0,
        "trigger_count": 0,
        "trigger_times_s": [],
        "trigger_volumes_m3": [],
        "trigger_areas_m2": [],
        "trigger_depths_m": [],
        "solver_iterations_start": 0,
        "full_field_normal_h2d": 0,
        "full_field_normal_d2h": 0,
    }


def _run_case(
    case: str,
    core: EarthmovingPhysicsCore,
    config: Interactive390FConfig,
    *,
    max_elapsed_s: float,
    sample_interval_s: float,
    minimum_observation_s: float,
    input_evidence: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    timeseries_path, summary_path = CASE_FILES[case]
    latest_checkpoint = OUTPUT_ROOT / f"{case}_timescale_latest.npz"
    resume_path = OUTPUT_ROOT / f"{case}_timescale_resume.json"
    rows: list[dict[str, Any]] = []
    accum = _initial_accumulators()
    if resume and latest_checkpoint.is_file() and resume_path.is_file():
        restore_device_checkpoint(core, latest_checkpoint)
        resume_record = json.loads(resume_path.read_text(encoding="utf-8"))
        rows = list(resume_record["rows"])
        accum = dict(resume_record["accumulators"])
    core.initialize_tool(_stationary_tool(core, core.device_state.timestamp_device_s))
    dt = config.physics_dt_s
    sample_stride = max(1, int(round(sample_interval_s / dt)))
    checkpoint_stride = max(1, int(round(30.0 / dt)))
    step_index = int(round(float(accum["elapsed_s"]) / dt))
    previous_sample_elapsed = float(rows[-1]["t_s"]) if rows else -dt
    previous_unique_area = float(rows[-1]["unique_activated_area_m2"]) if rows else 0.0
    previous_front_length = float(rows[-1]["L_front_m"]) if rows else 0.0
    previous_cumulative_r2m = float(rows[-1]["cumulative_R2M_m3"]) if rows else 0.0
    previous_cumulative_m2r = float(rows[-1]["cumulative_M2R_m3"]) if rows else 0.0
    trigger_xy = (
        np.asarray(rows[-1]["avalanche_trigger_location_xy_m"], dtype=np.float64)
        if rows and rows[-1]["avalanche_trigger_location_xy_m"] is not None
        else None
    )
    previous_mean_speed = float(rows[-1]["mobile_mean_velocity_m_s"]) if rows else 0.0
    stable_streak_s = 0.0
    previous_settled = False
    step_start_wall = perf_counter()
    last_result = None
    while float(accum["elapsed_s"]) + 0.5 * dt < max_elapsed_s:
        tool = _stationary_tool(
            core, core.device_state.timestamp_device_s + dt
        )
        result = core.step(
            tool,
            phase="deposition",
            cycle=1,
            dt_s=dt,
            soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        )
        last_result = result
        transition = result.avalanche_transition
        if transition is None:
            raise RuntimeError("GPU_LARGE_AVALANCHE_DIAGNOSTIC_MISSING")
        elapsed = float(accum["elapsed_s"]) + dt
        accum["elapsed_s"] = elapsed
        classification = transition.classification
        large_active = classification in {
            core.gpu_chain.large_avalanche.PERSISTING_LARGE_UNSTABLE_REGION,
            core.gpu_chain.large_avalanche.LARGE_AVALANCHE_MOBILE_PATH,
        }
        mobile_nonzero = (
            transition.current_mobile_volume_m3
            > core.avalanche_controller.config.settled_mobile_volume_m3
        )
        mobile_moving = (
            transition.moving_mobile_volume_m3
            > core.avalanche_controller.config.mobile_activity_volume_m3
        )
        deposition_volume = (
            0.0
            if result.dump_advance is None or result.dump_advance.deposition is None
            else result.dump_advance.deposition.deposited_volume_m3
        )
        deposition_dominated = bool(
            deposition_volume > transition.transferred_volume_m3 + 1.0e-15
        )
        residual_only = bool(
            result.static_relaxation_pending
            and not large_active
            and not mobile_nonzero
            and result.scalar_state.airborne_volume_m3 <= 1.0e-10
        )
        quiet_pending = bool(
            not large_active and not mobile_moving and not result.terrain_settled
        )
        accum["T_large_avalanche_active_s"] += dt * large_active
        accum["T_mobile_nonzero_s"] += dt * mobile_nonzero
        accum["T_mobile_actively_moving_s"] += dt * mobile_moving
        accum["T_deposition_dominated_s"] += dt * deposition_dominated
        accum["T_residual_minislope_only_s"] += dt * residual_only
        accum["T_quiet_to_settled_s"] += dt * quiet_pending
        if transition.transitioned:
            area = transition.newly_activated_area_m2 + transition.reactivated_area_step_m2
            volume = transition.newly_activated_volume_m3 + transition.reactivated_volume_m3
            accum["trigger_count"] += 1
            accum["trigger_times_s"].append(elapsed)
            accum["trigger_volumes_m3"].append(volume)
            accum["trigger_areas_m2"].append(area)
            accum["trigger_depths_m"].append(volume / max(area, 1.0e-12))
        if result.terrain_settled:
            stable_streak_s += dt
        else:
            stable_streak_s = 0.0
        step_index += 1
        should_sample = (
            step_index % sample_stride == 0
            or large_active
            or mobile_nonzero
            or transition.transitioned
            or deposition_volume > 0.0
            or result.terrain_settled != previous_settled
            or elapsed + 0.5 * dt >= max_elapsed_s
        )
        if should_sample:
            spatial, trigger_xy = _spatial_diagnostic(
                core,
                previous_sample_time_s=(
                    core.device_state.timestamp_device_s - (elapsed - previous_sample_elapsed)
                ),
                trigger_xy_m=trigger_xy,
            )
            interval = elapsed - max(previous_sample_elapsed, 0.0)
            unique_rate = (
                spatial["unique_activated_area_m2"] - previous_unique_area
            ) / max(interval, dt)
            front_length = spatial["maximum_propagation_distance_m"]
            front_speed = (front_length - previous_front_length) / max(interval, dt)
            r2m_rate = (
                transition.cumulative_resting_to_mobile_m3 - previous_cumulative_r2m
            ) / max(interval, dt)
            m2r_rate = (
                transition.cumulative_mobile_to_resting_m3 - previous_cumulative_m2r
            ) / max(interval, dt)
            accum["mobile_travel_distance_m"] += (
                0.5 * (previous_mean_speed + spatial["mobile_mean_velocity_m_s"])
                * interval
            )
            row = {
                "evidence_status": "OBSERVED",
                "t_s": elapsed,
                "absolute_device_time_s": core.device_state.timestamp_device_s,
                **spatial,
                "newly_activated_area_rate_m2_s": unique_rate,
                "active_avalanche_tiles": transition.active_tile_count,
                "active_residual_tiles": result.static_relaxation_active_tiles,
                "connected_unstable_area_m2": transition.largest_connected_area_m2,
                "Y_start_cell_count": transition.unstable_cell_count,
                "Y_start_component_count": transition.connected_region_count,
                "largest_Y_start_component_cells": transition.largest_connected_cell_count,
                "eligible_mobilizable_volume_m3": (
                    transition.largest_connected_mobilizable_volume_m3
                ),
                "mean_excess_Y_start_deg": transition.mean_excess_start_deg,
                "maximum_excess_Y_start_deg": transition.maximum_excess_start_deg,
                "cumulative_R2M_m3": transition.cumulative_resting_to_mobile_m3,
                "cumulative_M2R_m3": transition.cumulative_mobile_to_resting_m3,
                "R2M_rate_m3_s": r2m_rate,
                "M2R_rate_m3_s": m2r_rate,
                "mass_error_m3": result.mass_balance_error_m3,
                "terrain_settled": result.terrain_settled,
                "large_avalanche_classification": classification,
                "L_front_m": front_length,
                "v_front_m_s": front_speed,
                "avalanche_trigger_location_xy_m": (
                    None if trigger_xy is None else trigger_xy.tolist()
                ),
                "trigger_count_cumulative": accum["trigger_count"],
                "mobile_travel_distance_m_sampled_integral": accum["mobile_travel_distance_m"],
                "residual_minislope_iterations": result.static_relaxation_iterations,
                "core_wall_time_ms": result.timings_ms["physics_core_total"],
            }
            rows.append(row)
            previous_sample_elapsed = elapsed
            previous_unique_area = spatial["unique_activated_area_m2"]
            previous_front_length = front_length
            previous_cumulative_r2m = transition.cumulative_resting_to_mobile_m3
            previous_cumulative_m2r = transition.cumulative_mobile_to_resting_m3
            previous_mean_speed = spatial["mobile_mean_velocity_m_s"]
            timeseries_path.write_text(
                json.dumps(rows, indent=2) + "\n", encoding="utf-8"
            )
        previous_settled = result.terrain_settled
        if step_index % checkpoint_stride == 0:
            write_device_checkpoint(
                core,
                latest_checkpoint,
                provenance={
                    "case": case,
                    "elapsed_s": elapsed,
                    "semantics": "RESUMABLE_ATTRIBUTION_BOUNDARY_NOT_NORMAL_PHYSICS",
                },
            )
            resume_path.write_text(
                json.dumps({"rows": rows, "accumulators": accum}, indent=2) + "\n",
                encoding="utf-8",
            )
        if (
            result.terrain_settled
            and elapsed >= minimum_observation_s
            and stable_streak_s >= min(1.0, minimum_observation_s)
        ):
            break
    accum["wall_time_s"] = float(accum["wall_time_s"]) + (
        perf_counter() - step_start_wall
    )
    if last_result is None or not rows:
        raise RuntimeError("ATTRIBUTION_CASE_PRODUCED_NO_SAMPLES")
    trigger_times = np.asarray(accum["trigger_times_s"], dtype=np.float64)
    trigger_intervals = np.diff(trigger_times)
    velocities = np.asarray(
        [row["mobile_mean_velocity_m_s"] for row in rows], dtype=np.float64
    )
    propagation = float(max(row["L_front_m"] for row in rows))
    dynamic_velocities = velocities[velocities > 0.0]
    characteristic_speed = float(
        np.percentile(dynamic_velocities, 50.0)
        if dynamic_velocities.size
        else 0.0
    )
    transferred = float(rows[-1]["cumulative_R2M_m3"])
    deposited = float(rows[-1]["cumulative_M2R_m3"])
    summary = {
        "schema": "390F_PHYSICAL_TIMESCALE_CASE/v1",
        "case": case,
        "status": (
            "SETTLED" if rows[-1]["terrain_settled"] else "FINITE_HORIZON_NOT_SETTLED"
        ),
        "evidence_status": "OBSERVED",
        "production_path": {
            "core": type(core).__name__,
            "runtime_backend": core.runtime_backend,
            "state_authority": core.device_state.authority.value,
            "operator_chain": type(core.gpu_chain).__name__,
            "machine_motion": "DISABLED",
            "render_ui": "DISABLED",
        },
        "formal_identity": {
            "shape_yx": list(core.grid.shape),
            "dx_m": core.grid.dx,
            "physics_dt_s": config.physics_dt_s,
            "material_scenario": core.material.name,
            "config_hash_sha256": sha256_file(CONFIG_PATH),
            "physics_parameters_modified": False,
        },
        "input_evidence": input_evidence,
        "physical_time_decomposition": {
            "T_total_s": accum["elapsed_s"],
            "T_large_avalanche_active_s": accum["T_large_avalanche_active_s"],
            "T_mobile_nonzero_s": accum["T_mobile_nonzero_s"],
            "T_mobile_actively_moving_s": accum["T_mobile_actively_moving_s"],
            "T_deposition_dominated_s": accum["T_deposition_dominated_s"],
            "T_residual_minislope_only_s": accum["T_residual_minislope_only_s"],
            "T_quiet_to_settled_s": accum["T_quiet_to_settled_s"],
            "durations_are_overlapping_diagnostics": True,
        },
        "computational_time": {
            "wall_clock_s": accum["wall_time_s"],
            "mean_RTF": float(accum["elapsed_s"]) / max(float(accum["wall_time_s"]), 1.0e-12),
            "solver_iterations": rows[-1]["residual_minislope_iterations"],
            "solver_iterations_are_not_material_motion_time": True,
        },
        "timescale_metrics": {
            "v_mean_time_sampled_m_s": (
                float(np.mean(dynamic_velocities))
                if dynamic_velocities.size else 0.0
            ),
            "v_p50_time_sampled_m_s": characteristic_speed,
            "v_p95_time_sampled_m_s": (
                float(np.percentile(dynamic_velocities, 95.0))
                if dynamic_velocities.size else 0.0
            ),
            "L_mobile_sampled_integral_m": accum["mobile_travel_distance_m"],
            "maximum_front_propagation_m": propagation,
            "T_transport_L_over_v_s": (
                propagation / characteristic_speed if characteristic_speed > 0.0 else None
            ),
            "Q_R2M_mean_m3_s": transferred / max(float(accum["elapsed_s"]), dt),
            "Q_M2R_mean_m3_s": deposited / max(float(accum["elapsed_s"]), dt),
            "dA_unique_mean_m2_s": rows[-1]["unique_activated_area_m2"] / max(float(accum["elapsed_s"]), dt),
            "v_front_mean_m_s": propagation / max(float(accum["elapsed_s"]), dt),
        },
        "event_scheduling": {
            "trigger_count": accum["trigger_count"],
            "mean_interval_between_triggers_s": (
                float(np.mean(trigger_intervals)) if trigger_intervals.size else None
            ),
            "median_activated_volume_per_trigger_m3": (
                float(np.median(accum["trigger_volumes_m3"]))
                if accum["trigger_volumes_m3"] else None
            ),
            "median_activated_area_per_trigger_m2": (
                float(np.median(accum["trigger_areas_m2"]))
                if accum["trigger_areas_m2"] else None
            ),
            "mean_mobilization_depth_per_trigger_m": (
                float(np.mean(accum["trigger_depths_m"]))
                if accum["trigger_depths_m"] else None
            ),
        },
        "final": rows[-1],
        "maximum_abs_mass_error_m3": float(
            max(abs(row["mass_error_m3"]) for row in rows)
        ),
        "diagnostic_sampling": {
            "interval_s": sample_interval_s,
            "velocity_quantiles": "MOBILE_VOLUME_WEIGHTED_EXACT_AT_SAMPLE_BOUNDARIES",
            "full_field_reads": "EXPLICIT_ATTRIBUTION_ONLY_NOT_NORMAL_PRODUCTION_STEP",
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_device_checkpoint(
        core,
        latest_checkpoint,
        provenance={
            "case": case,
            "elapsed_s": accum["elapsed_s"],
            "final_case_status": summary["status"],
        },
    )
    resume_path.write_text(
        json.dumps({"rows": rows, "accumulators": accum}, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _interrupted_evidence() -> dict[str, Any]:
    marker = INTERRUPTED_RUN / "USER_INTERRUPTED_PHYSICS_NOT_SETTLED_NOT_ACCEPTED.json"
    return {
        "source_run": "run_1786506718",
        "marker": str(marker.relative_to(ROOT)),
        "marker_hash_sha256": sha256_file(marker),
        "runtime_telemetry_hash_sha256": sha256_file(
            INTERRUPTED_RUN / "runtime_telemetry.json"
        ),
        "runtime_manifest_hash_sha256": sha256_file(
            INTERRUPTED_RUN / "runtime_manifest.json"
        ),
        "state_transitions_hash_sha256": sha256_file(
            INTERRUPTED_RUN / "state_transitions.json"
        ),
        "acceptance_disposition": "DIAGNOSTIC_VALID_FINAL_FULL_CYCLE_NOT_ACCEPTED",
    }


def _classify_and_report() -> dict[str, Any]:
    summaries = {
        case: json.loads(path.read_text(encoding="utf-8"))
        for case, (_, path) in CASE_FILES.items()
    }
    passive = summaries["passive"]
    dump = summaries["dump"]
    stable = summaries["stable"]
    dtime = dump["physical_time_decomposition"]
    metrics = dump["timescale_metrics"]
    events = dump["event_scheduling"]
    total = float(dtime["T_total_s"])
    factors: list[dict[str, Any]] = []
    if passive["status"] != "SETTLED" or passive["final"]["unique_activated_area_m2"] > 0.08:
        factors.append({"factor": "INITIAL_TERRAIN_STABILITY_PROBLEM", "rank_score": 4.0, "evidence": "passive formal terrain remained active or exceeded one formal large-event area"})
    if events["trigger_count"] > 0:
        median_volume = events["median_activated_volume_per_trigger_m3"] or 0.0
        if median_volume < 0.01:
            factors.append({"factor": "MOBILIZATION_RATE_LIMITED", "rank_score": 2.5, "evidence": f"median activation {median_volume:.6g} m3/trigger"})
    if metrics["v_p50_time_sampled_m_s"] < 0.05 and dtime["T_mobile_nonzero_s"] > 0.5 * total:
        factors.append({"factor": "MOBILE_TRANSPORT_TIMESCALE_LIMITED", "rank_score": 3.0, "evidence": f"p50 sampled mean speed {metrics['v_p50_time_sampled_m_s']:.6g} m/s"})
    if metrics["Q_M2R_mean_m3_s"] < metrics["Q_R2M_mean_m3_s"] and dtime["T_deposition_dominated_s"] > 0.5 * total:
        factors.append({"factor": "DEPOSITION_RATE_LIMITED", "rank_score": 3.0, "evidence": "M2R throughput trails R2M while deposition dominates physical time"})
    interval = events["mean_interval_between_triggers_s"]
    persistence = Interactive390FConfig.load(CONFIG_PATH).large_avalanche_transition["persistence_time_s"]
    if interval is not None and events["trigger_count"] >= 5 and interval <= 3.0 * persistence:
        factors.append({"factor": "EVENT_SCHEDULING_FRAGMENTATION", "rank_score": 3.5, "evidence": f"{events['trigger_count']} triggers with mean interval {interval:.6g}s near persistence gate {persistence}s"})
    if dtime["T_residual_minislope_only_s"] > 0.5 * total:
        factors.append({"factor": "RESIDUAL_SOLVER_SCHEDULING_LIMITED", "rank_score": 4.0, "evidence": "residual-only duration exceeded half total physical time"})
    stable_time = float(stable["physical_time_decomposition"]["T_total_s"])
    if dump["status"] != "SETTLED" and stable["status"] == "SETTLED" and stable_time < 0.25 * total:
        factors.append({"factor": "LONG_NEAR_CRITICAL_CONNECTED_SLOPE", "rank_score": 4.5, "evidence": "equivalent production dump settles on stable pile in less than one quarter of formal replay horizon"})
    factors.sort(key=lambda item: float(item["rank_score"]), reverse=True)
    mapped = [item["factor"] for item in factors]
    official = [
        value for value in mapped if value in {
            "INITIAL_TERRAIN_STABILITY_PROBLEM", "MOBILIZATION_RATE_LIMITED",
            "MOBILE_TRANSPORT_TIMESCALE_LIMITED", "DEPOSITION_RATE_LIMITED",
            "EVENT_SCHEDULING_FRAGMENTATION", "RESIDUAL_SOLVER_SCHEDULING_LIMITED",
        }
    ]
    primary = official[0] if len(official) == 1 else "MIXED_CAUSE"
    if not official:
        primary = "PHYSICAL_LONG_RANGE_PROPAGATION_WITH_REASONABLE_TIMESCALE"
    flags = {
        "TINY_VOLUME_PER_TRIGGER": bool((events["median_activated_volume_per_trigger_m3"] or np.inf) < 0.01),
        "VERY_LOW_MOBILE_SPEED": bool(metrics["v_p50_time_sampled_m_s"] < 0.05),
        "DEPOSITION_RATE_LIMITED": "DEPOSITION_RATE_LIMITED" in mapped,
        "PERSISTENCE_GATE_FRAGMENTATION": "EVENT_SCHEDULING_FRAGMENTATION" in mapped,
        "LONG_NEAR_CRITICAL_CONNECTED_SLOPE": "LONG_NEAR_CRITICAL_CONNECTED_SLOPE" in mapped,
        "RESIDUAL_ONLY_SLOWDOWN": "RESIDUAL_SOLVER_SCHEDULING_LIMITED" in mapped,
    }
    classification = {
        "schema": "390F_PHYSICS_TIMESCALE_CLASSIFICATION/v1",
        "status": "ATTRIBUTION_COMPLETE",
        "primary_classification": primary,
        "ranked_factors": factors,
        "diagnostic_flags": flags,
        "claim_status": {
            "runner_and_outputs": "IMPLEMENTED",
            "three_case_measurements": "OBSERVED",
            "causal_ranking": "INFERRED",
            "site_calibrated_timescale": "NOT_YET_VALIDATED",
        },
        "physics_modified": False,
    }
    (OUTPUT_ROOT / "timescale_classification.json").write_text(
        json.dumps(classification, indent=2) + "\n", encoding="utf-8"
    )
    aggregate = {
        "schema": "390F_PHYSICS_TIMESCALE_ATTRIBUTION/v1",
        "status": "COMPLETE",
        "interrupted_run": _interrupted_evidence(),
        "cases": summaries,
        "classification": classification,
        "production_clock_attribution": {
            "evidence_status": "OBSERVED",
            "checkpoint_world_time_s": 39.90000208094716,
            "checkpoint_device_terrain_time_s": 19.95000000000015,
            "world_to_terrain_time_ratio": 2.000000104307612,
            "interpretation": "INTERRUPTED_RUN_WORLD_TIME_IS_NOT_TERRAIN_MATERIAL_INTEGRATION_TIME",
        },
        "safe_computational_acceleration_candidates": [
            "CUDA graph capture for the unchanged resident operator sequence",
            "device-side compact diagnostic reductions instead of explicit acceptance full-field samples",
            "asynchronous checkpoint compression outside the physics thread",
        ],
        "physics_changing_acceleration_not_implemented": [
            "velocity/mobilization/deposition multipliers",
            "changed persistence gate or material parameters",
            "relaxed MiniSlope tolerance or avalanche truncation",
        ],
    }
    (OUTPUT_ROOT / "physics_timescale_attribution_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 390F Physics Timescale Attribution Report",
        "",
        "## Outcome",
        "",
        f"Primary classification: **{primary}** (`INFERRED`).",
        "",
        "The interrupted full-cycle state remains diagnostic evidence only and is not accepted as a completed physics result (`OBSERVED`). No physics parameter, 0.05 m resolution, physics dt, conservation rule, Mobile equation, deposition rule, persistence threshold, or MiniSlope tolerance was changed (`IMPLEMENTED`).",
        "",
        "At the exact dump-after checkpoint, Isaac world time is 39.900002 s while the authoritative Device terrain timestamp is 19.950000 s: a 2.0000001:1 clock ratio (`OBSERVED`). Therefore the interrupted run's 1259 s world timestamp must not be reported as 1259 s of terrain-material integration.",
        "",
        "## Case comparison",
        "",
        "| Case | Status | Physical time (s) | Wall time (s) | Mobile-active (s) | Residual-only (s) | Max front (m) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, summary in summaries.items():
        p = summary["physical_time_decomposition"]
        lines.append(
            f"| {case} | {summary['status']} | {p['T_total_s']:.3f} | "
            f"{summary['computational_time']['wall_clock_s']:.3f} | "
            f"{p['T_mobile_actively_moving_s']:.3f} | "
            f"{p['T_residual_minislope_only_s']:.3f} | "
            f"{summary['timescale_metrics']['maximum_front_propagation_m']:.3f} |"
        )
    lines.extend(["", "## Ranked attribution", ""])
    for index, factor in enumerate(factors, 1):
        lines.append(f"{index}. `{factor['factor']}` — {factor['evidence']} (`INFERRED`).")
    lines.extend([
        "",
        "## Claim boundaries",
        "",
        "- Measurements and generated artifacts: `OBSERVED`.",
        "- Production checkpoint/runner integration: `IMPLEMENTED`.",
        "- Causal factor ranking: `INFERRED`.",
        "- Site-calibrated iron-ore timing: `NOT_YET_VALIDATED`.",
        "- Solver iterations are reported separately and are not interpreted as material physical time.",
        "",
        "## Acceleration boundary",
        "",
        "Safe computational candidates preserve the exact physics and include CUDA graph capture, compact device diagnostics, and asynchronous checkpoint I/O. Velocity, deposition, mobilization, persistence, tolerance, grid, or timestep multipliers are physics-changing and were not implemented.",
    ])
    (OUTPUT_ROOT / "physics_timescale_attribution_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("passive", "dump", "stable", "all"), default="all")
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    parser.add_argument("--passive-horizon-s", type=float, default=30.0)
    parser.add_argument("--dump-horizon-s", type=float, default=1300.0)
    parser.add_argument("--stable-horizon-s", type=float, default=400.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    selected = ("passive", "dump", "stable") if args.case == "all" else (args.case,)
    for case in selected:
        if case == "passive":
            core, config, material_source = _core(FORMAL_TERRAIN)
            input_evidence = {
                "terrain": str(FORMAL_TERRAIN.relative_to(ROOT)),
                "terrain_hash_sha256": sha256_file(FORMAL_TERRAIN),
                "disturbance": "NONE",
                "material_parameter_basis": material_source.get("start_stop_angle_status"),
            }
            horizon, minimum = args.passive_horizon_s, 5.0
        elif case == "dump":
            if not DUMP_AFTER_CHECKPOINT.is_file():
                raise RuntimeError(
                    f"FORMAL_DUMP_AFTER_CHECKPOINT_MISSING: {DUMP_AFTER_CHECKPOINT}"
                )
            core, config, material_source = _core(FORMAL_TERRAIN)
            checkpoint = restore_device_checkpoint(core, DUMP_AFTER_CHECKPOINT)
            input_evidence = {
                **_interrupted_evidence(),
                "CHECKPOINT_SOURCE_RUN": checkpoint["provenance"]["checkpoint_source_run"],
                "CHECKPOINT_SIM_TIME": checkpoint["provenance"]["checkpoint_sim_time_s"],
                "CHECKPOINT_HASH": checkpoint["checkpoint_hash_sha256"],
                "TERRAIN_HASH": checkpoint["terrain_hash_sha256"],
                "CONFIG_HASH": checkpoint["provenance"]["config_hash_sha256"],
                "checkpoint_boundary": checkpoint["provenance"]["checkpoint_boundary"],
            }
            horizon, minimum = args.dump_horizon_s, 5.0
        else:
            if not DUMP_RELEASE_CHECKPOINT.is_file():
                raise RuntimeError(
                    f"FORMAL_DUMP_RELEASE_CHECKPOINT_MISSING: {DUMP_RELEASE_CHECKPOINT}"
                )
            core, config, material_source = _core(STABLE_TERRAIN)
            equivalent = _install_equivalent_dump(core, DUMP_RELEASE_CHECKPOINT)
            input_evidence = {
                "terrain": str(STABLE_TERRAIN.relative_to(ROOT)),
                "terrain_hash_sha256": sha256_file(STABLE_TERRAIN),
                "stable_without_disturbance_basis": "SOURCE_PILE_MAX_CENTRAL_SLOPE_BELOW_THETA_START_AND_TEST_A_COMPANION_COMPARISON",
                "equivalent_dump": equivalent,
            }
            horizon, minimum = args.stable_horizon_s, 5.0
        summary = _run_case(
            case,
            core,
            config,
            max_elapsed_s=horizon,
            sample_interval_s=args.sample_interval_s,
            minimum_observation_s=minimum,
            input_evidence=input_evidence,
            resume=args.resume,
        )
        print(json.dumps({"case": case, "status": summary["status"], "physical_time_s": summary["physical_time_decomposition"]["T_total_s"], "wall_time_s": summary["computational_time"]["wall_clock_s"]}), flush=True)
    if all(summary_path.is_file() for _, summary_path in CASE_FILES.values()):
        aggregate = _classify_and_report()
        print(json.dumps({"status": aggregate["status"], "classification": aggregate["classification"]["primary_classification"]}), flush=True)


if __name__ == "__main__":
    main()
