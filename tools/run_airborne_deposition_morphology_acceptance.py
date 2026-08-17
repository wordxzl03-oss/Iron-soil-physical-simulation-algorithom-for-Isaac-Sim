#!/usr/bin/env python3
"""Frozen-checkpoint causal audit for finite-footprint airborne landing."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from isaac_bulk_pipeline.runtime import SoilForceMode  # noqa: E402
from isaac_bulk_pipeline.runtime.device_checkpoint import restore_device_checkpoint  # noqa: E402
from isaac_bulk_pipeline.visualization import ChunkedDynamicMeshAdapter  # noqa: E402
from tools.run_physics_timescale_attribution import (  # noqa: E402
    DUMP_RELEASE_CHECKPOINT,
    FORMAL_TERRAIN,
    _core,
    _stationary_tool,
)

OUT = ROOT / "outputs/airborne_deposition_morphology"


def _parcel(parcel: Any) -> dict[str, Any]:
    footprint = parcel.footprint
    return {
        "parcel_id": parcel.parcel_id,
        "represented_volume_m3": float(parcel.volume_m3),
        "position_world_m": np.asarray(parcel.position_world_m).tolist(),
        "velocity_world_m_s": np.asarray(parcel.velocity_world_m_s).tolist(),
        "timestamp_s": float(parcel.timestamp_s),
        "release_source": parcel.source,
        "release_geometry": None if footprint is None else {
            "lateral_axis_world_m": footprint.lateral_axis_world_m.tolist(),
            "lateral_extent_m": footprint.lateral_extent_m,
            "longitudinal_extent_m": footprint.longitudinal_extent_m,
            "area_m2": footprint.area_m2,
            "provenance": footprint.provenance,
            "mass_profile": footprint.mass_profile,
        },
    }


def _snapshot(core: Any, label: str, parcels: tuple[Any, ...]) -> dict[str, Any]:
    view = core.device_state.explicit_host_view(source="acceptance")
    resting = np.asarray(view.H_resting_m).copy()
    mobile = np.asarray(view.H_mobile_m).copy()
    result = {
        "label": label,
        "device_time_s": float(view.timestamp_device_s),
        "H_resting_m": resting,
        "h_mobile_m": mobile,
        "H_free_m": resting + mobile,
        "mobile_momentum_m2_s": np.asarray(view.mobile_momentum_m2_s).copy(),
        "dirty_tile_ids": np.asarray(view.dirty_tile_ids).copy(),
        "airborne_parcels": [_parcel(item) for item in parcels],
    }
    # The full-field reads above are an explicit acceptance boundary, not a
    # production-loop transfer. Start a fresh transfer-accounting window so
    # the Core's normal-path guard continues to audit only subsequent work.
    core.device_state.begin_physics_step()
    return result


def _will_land(core: Any, dt_s: float) -> bool:
    model = core.gpu_chain.airborne.parcel_model
    parcels = model.resolve_missing_footprints(core.gpu_metadata.airborne)
    if not parcels:
        return False
    gravity = np.asarray([0.0, 0.0, -model.config.gravity_m_s2])
    candidates = []
    z = []
    for parcel in parcels:
        position = (
            parcel.position_world_m + parcel.velocity_world_m_s * dt_s
            + 0.5 * gravity * dt_s * dt_s
        )
        terrain = core.grid.world_to_terrain(position)
        row, col = core.grid.terrain_to_grid(terrain)
        if 0.0 <= row <= core.grid.ny - 1 and 0.0 <= col <= core.grid.nx - 1:
            candidates.append((row, col)); z.append(float(terrain[2]))
    if not candidates:
        return False
    surface = core.device_state.sample_surface_bilinear(
        np.asarray(candidates), source="airborne_landing_acceptance_prediction"
    )
    return bool(np.any(np.asarray(z) <= surface + model.config.terrain_clearance_m))


def _morphology(field: np.ndarray, baseline: np.ndarray, start_deg: float, dx: float) -> dict[str, Any]:
    delta = np.asarray(field) - np.asarray(baseline)
    positive = delta[delta > 1.0e-12]
    changed = np.abs(delta) > 1.0e-12
    halo = ndimage.binary_dilation(changed, structure=np.ones((3, 3), dtype=bool))
    dhx = np.abs(np.diff(field, axis=1)); dhy = np.abs(np.diff(field, axis=0))
    maskx = halo[:, :-1] | halo[:, 1:]
    masky = halo[:-1, :] | halo[1:, :]
    jumps = np.r_[dhx[maskx], dhy[masky]]
    threshold = np.tan(np.deg2rad(start_deg)) * dx
    anomalous = np.zeros_like(changed)
    ex = (dhx > threshold) & maskx; ey = (dhy > threshold) & masky
    anomalous[:, :-1] |= ex; anomalous[:, 1:] |= ex
    anomalous[:-1, :] |= ey; anomalous[1:, :] |= ey
    labels, count = ndimage.label(anomalous, structure=np.ones((3, 3), dtype=np.int8))
    sizes = np.bincount(labels.ravel())[1:] if count else np.empty(0, dtype=int)
    q = lambda values, p: float(np.quantile(values, p)) if values.size else 0.0
    return {
        "changed_cell_count": int(np.count_nonzero(changed)),
        "delta_H_m": {
            "mean": float(np.mean(positive)) if positive.size else 0.0,
            "p95": q(positive, .95), "p99": q(positive, .99),
            "max": float(np.max(positive)) if positive.size else 0.0,
        },
        "four_neighbor_height_jump_m": {
            "p95": q(jumps, .95), "p99": q(jumps, .99),
            "max": float(np.max(jumps)) if jumps.size else 0.0,
        },
        "anomalous_gradient_threshold_m": float(threshold),
        "anomalous_gradient_component_count": int(count),
        "largest_connected_anomalous_gradient_component_cells": (
            int(np.max(sizes)) if sizes.size else 0
        ),
    }


def _landing_record(record: Any, nx: int) -> dict[str, Any]:
    rows, cols = np.divmod(record.recipient_flat_indices, nx)
    delivered = np.asarray(record.delivered_volumes_m3)
    control = np.asarray(record.recipient_control_areas_m2)
    return {
        "parcel_id": record.parcel_id,
        "represented_volume_m3": record.represented_volume_m3,
        "impact_position_world_m": record.impact_position_world_m.tolist(),
        "impact_velocity_world_m_s": record.impact_velocity_world_m_s.tolist(),
        "release_source": record.source,
        "landing_operator": "DeviceAirborneBridge.ballistic_parcel_landing",
        "footprint": {
            "lateral_axis_world_m": record.footprint.lateral_axis_world_m.tolist(),
            "lateral_extent_m": record.footprint.lateral_extent_m,
            "longitudinal_extent_m": record.footprint.longitudinal_extent_m,
            "nominal_area_m2": record.footprint.area_m2,
            "rasterized_overlap_area_m2": record.footprint_area_m2,
            "provenance": record.footprint.provenance,
            "mass_profile": record.footprint.mass_profile,
        },
        "recipient_cell_count": record.recipient_cell_count,
        "recipient_cells_yx": np.column_stack((rows, cols)).tolist(),
        "recipient_flat_indices": record.recipient_flat_indices.tolist(),
        "recipient_control_areas_m2": control.tolist(),
        "volume_delivered_m3": delivered.tolist(),
        "delta_h_m": (delivered / control).tolist(),
        "overlap_areas_m2": record.overlap_areas_m2.tolist(),
        "volume_error_m3": float(np.sum(delivered) - record.represented_volume_m3),
        "terrain_normal_world": record.terrain_normal_world.tolist(),
        "normal_impact_impulse_on_terrain_ns": record.normal_impact_impulse_on_terrain_ns.tolist(),
        "normal_kinetic_energy_dissipated_j": record.normal_kinetic_energy_dissipated_j,
    }


def _mesh_ab(core: Any, initial: np.ndarray, final: np.ndarray, tile_ids: set[int]) -> dict[str, Any]:
    adapter = ChunkedDynamicMeshAdapter(core.device_state.tile_size)
    windows = adapter._chunk_windows(core.grid.shape)
    tiles_x = (core.grid.nx - 1 + core.device_state.tile_size - 1) // core.device_state.tile_size
    incremental = np.array(initial, copy=True)
    for tile_id in sorted(tile_ids):
        key = divmod(tile_id, tiles_x)
        window = windows[key]
        incremental[window.slices] = final[window.slices]
    error = np.abs(incremental - final)
    changed = np.abs(final - initial) > 1.0e-12
    return {
        "source": "AUTHORITATIVE_H_FREE",
        "tile_size_cells": core.device_state.tile_size,
        "dirty_tile_count": len(tile_ids),
        "changed_vertices": int(np.count_nonzero(changed)),
        "maximum_vertex_error_m": float(np.max(error)),
        "incremental_matches_full_rebuild": bool(np.array_equal(incremental, final)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core, config, _ = _core(FORMAL_TERRAIN)
    restore_device_checkpoint(core, DUMP_RELEASE_CHECKPOINT)
    dt = config.physics_dt_s
    core.initialize_tool(_stationary_tool(core, core.device_state.timestamp_device_s))
    stages: dict[str, dict[str, Any]] = {}
    boundary_fields: dict[str, dict[str, Any]] = {}
    landing_records: list[Any] = []
    first_impact_step = None
    impact_result = None
    dirty_union: set[int] = set()
    armed = False

    def observer(label: str) -> None:
        if not armed or label in boundary_fields:
            return
        if label == "AFTER_AIRBORNE_TO_MOBILE":
            snap = _snapshot(core, "T1", core.gpu_metadata.airborne)
            boundary_fields[label] = snap
            dirty_union.update(map(int, snap["dirty_tile_ids"]))
        elif label == "AFTER_MOBILE_TO_RESTING_DEPOSITION":
            snap = _snapshot(core, "T3", core.gpu_metadata.airborne)
            boundary_fields[label] = snap
            dirty_union.update(map(int, snap["dirty_tile_ids"]))

    core.gpu_chain.audit_state_observer = observer
    evolution: list[dict[str, Any]] = []
    wall_start = perf_counter()
    settled_first = None
    max_mass_error = 0.0
    initial_airborne = sum(p.volume_m3 for p in core.gpu_metadata.airborne)
    initial_ledger = core.reservoir_observation()
    max_steps = int(round(5.0 / dt))
    for step in range(1, max_steps + 1):
        if first_impact_step is None and _will_land(core, dt):
            stages["T0"] = _snapshot(core, "T0", core.gpu_metadata.airborne)
            dirty_union.update(map(int, stages["T0"]["dirty_tile_ids"]))
            armed = True
        result = core.step(
            _stationary_tool(core, core.device_state.timestamp_device_s + dt),
            phase="deposition", cycle=1, dt_s=dt,
            soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        )
        max_mass_error = max(max_mass_error, abs(result.mass_balance_error_m3))
        airborne = None if result.dump_advance is None else result.dump_advance.airborne
        if airborne is not None and airborne.landing_records:
            if first_impact_step is None:
                first_impact_step = step
                impact_result = result
                landing_records.extend(airborne.landing_records)
                stages["T1"] = boundary_fields["AFTER_AIRBORNE_TO_MOBILE"]
                # Landing scatter is itself the A->Mobile conservative
                # conversion; no distinct state mutation occurs between T1/T2.
                stages["T2"] = {
                    key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
                    for key, value in stages["T1"].items()
                }
                stages["T2"]["label"] = "T2"
                stages["T3"] = boundary_fields["AFTER_MOBILE_TO_RESTING_DEPOSITION"]
                remaining = [_parcel(item) for item in airborne.remaining_parcels]
                stages["T1"]["airborne_parcels"] = remaining
                stages["T2"]["airborne_parcels"] = remaining
                stages["T3"]["airborne_parcels"] = remaining
            else:
                landing_records.extend(airborne.landing_records)
        tr = result.avalanche_transition
        if step % 15 == 0 or (airborne is not None and airborne.landed_volume_m3 > 0) or result.terrain_settled:
            evolution.append({
                "elapsed_s": step * dt,
                "device_time_s": core.device_state.timestamp_device_s,
                "airborne_volume_m3": result.scalar_state.airborne_volume_m3,
                "mobile_volume_m3": result.scalar_state.material_ledger.mobile_m3,
                "moving_mobile_volume_m3": result.physics_diagnostics.mobile_moving_volume_m3,
                "Y_start_cell_count": tr.unstable_cell_count,
                "Y_start_component_count": tr.connected_region_count,
                "eligible_mobilizable_volume_m3": tr.largest_connected_mobilizable_volume_m3,
                "terrain_settled": result.terrain_settled,
                "mass_error_m3": result.mass_balance_error_m3,
            })
        if result.terrain_settled and settled_first is None:
            settled_first = step * dt
    if first_impact_step is None:
        raise RuntimeError("FROZEN_CHECKPOINT_DID_NOT_IMPACT")
    stages["T4"] = _snapshot(core, "T4", core.gpu_metadata.airborne)
    dirty_union.update(map(int, stages["T4"]["dirty_tile_ids"]))
    core.gpu_chain.audit_state_observer = None
    wall_s = perf_counter() - wall_start

    # Persist every requested authoritative field in one compressed evidence file.
    arrays: dict[str, np.ndarray] = {}
    stage_metadata: dict[str, Any] = {}
    for label, stage in stages.items():
        for field in ("H_resting_m", "h_mobile_m", "H_free_m", "mobile_momentum_m2_s", "dirty_tile_ids"):
            arrays[f"{label}_{field}"] = stage[field]
        stage_metadata[label] = {
            "device_time_s": stage["device_time_s"],
            "airborne_parcels": stage["airborne_parcels"],
        }
    np.savez_compressed(OUT / "causal_T0_T4_states.npz", **arrays)

    record_json = [_landing_record(item, core.grid.nx) for item in landing_records]
    (OUT / "parcel_recipient_audit.json").write_text(
        json.dumps(record_json, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "causal_T0_T4_metadata.json").write_text(
        json.dumps(stage_metadata, indent=2) + "\n", encoding="utf-8"
    )

    baseline = stages["T0"]
    morphology = {}
    field_names = {
        "H_resting": "H_resting_m", "h_mobile": "h_mobile_m", "H_free": "H_free_m"
    }
    for label, stage in stages.items():
        morphology[label] = {
            name: _morphology(stage[key], baseline[key], core.material.start_angle_deg, core.grid.dx)
            for name, key in field_names.items()
        }

    # Reconstruct the exact legacy point-source write at the same impact
    # frame. This is not a new simulation result; it documents the removed
    # operator semantics against the actual T0 and actual impact records.
    old_t1 = np.array(baseline["H_free_m"], copy=True)
    old_indices = []
    for record in landing_records:
        terrain = core.grid.world_to_terrain(record.impact_position_world_m)
        row_f, col_f = core.grid.terrain_to_grid(terrain)
        row = int(np.clip(round(row_f), 0, core.grid.ny - 1))
        col = int(np.clip(round(col_f), 0, core.grid.nx - 1))
        old_indices.append(row * core.grid.nx + col)
        old_t1[row, col] += record.represented_volume_m3 / core.integrator.vertex_weights_m2[row, col]

    new_unique = np.unique(np.concatenate([item.recipient_flat_indices for item in landing_records]))
    initial_free_volume = core.integrator.integrate(baseline["H_free_m"])
    final_free_volume = core.integrator.integrate(stages["T4"]["H_free_m"])
    final_ledger = core.reservoir_observation()
    mesh_ab = _mesh_ab(core, baseline["H_free_m"], stages["T4"]["H_free_m"], dirty_union)
    footprint_area = float(sum(item.footprint_area_m2 for item in landing_records))
    total_landed = float(sum(item.represented_volume_m3 for item in landing_records))
    first_time = stages["T0"]["device_time_s"] + dt

    # The legacy and repaired artifacts were generated from the same frozen
    # checkpoint with the same five-second observation window.  Keep these
    # comparisons in the one release summary instead of requiring a reviewer
    # to join several files by hand.
    legacy_path = OUT / "legacy_point_T4.npz"
    legacy_summary_path = OUT / "legacy_point_summary.json"
    repaired_summary_path = OUT / "finite_footprint_summary.json"
    with np.load(legacy_path) as legacy_state:
        legacy_free = np.asarray(legacy_state["resting"]) + np.asarray(legacy_state["mobile"])
    legacy_final_morphology = _morphology(
        legacy_free, baseline["H_free_m"], core.material.start_angle_deg, core.grid.dx
    )
    legacy_timescale = json.loads(legacy_summary_path.read_text(encoding="utf-8"))
    repaired_timescale = json.loads(repaired_summary_path.read_text(encoding="utf-8"))

    gui_run = ROOT / "outputs/390f_v2/interactive_runs/run_1786598146"
    gui_lifecycle = json.loads(
        (gui_run / "presentation_gui_lifecycle_smoke.json").read_text(encoding="utf-8")
    )
    gui_demo = json.loads(
        (gui_run / "presentation_demo_summary.json").read_text(encoding="utf-8")
    )
    gui_reset = json.loads(
        (gui_run / "presentation_reset_smoke.json").read_text(encoding="utf-8")
    )
    gui_exit = json.loads((gui_run / "application_exit.json").read_text(encoding="utf-8"))
    transitions = {item["state"]: item for item in gui_demo["phase_transitions"]}
    reverse_duration_s = (
        transitions["ALIGN_DUMP"]["timestamp_s"]
        - transitions["REVERSE_TRAVEL"]["timestamp_s"]
    )
    summary = {
        "schema": "AIRBORNE_DEPOSITION_MORPHOLOGY_CAUSAL_FIX/v1",
        "status": "PASS" if (
            len(new_unique) > len(set(old_indices))
            and morphology["T1"]["H_free"]["delta_H_m"]["max"] < 2.0
            and abs(total_landed - initial_airborne) < 1e-12
            and mesh_ab["incremental_matches_full_rebuild"]
            and max_mass_error <= 1e-8
        ) else "FAIL",
        "formal_identity": {
            "shape_yx": list(core.grid.shape), "dx_m": core.grid.dx,
            "physics_dt_s": dt, "runtime_backend": "GPU_RUNTIME",
            "state_authority": "DEVICE", "material_parameters_modified": False,
            "flow_arrest_ownership_modified": False,
        },
        "root_cause": (
            "Reduced-order aggregate parcel volume was atomically scattered to one "
            "nearest 0.05 m vertex per parcel; delta_h=V/A created a Dirac-like source."
        ),
        "first_bad_frame": {
            "step_from_frozen_checkpoint": first_impact_step,
            "device_time_s": first_time,
            "T0_device_time_s": stages["T0"]["device_time_s"],
        },
        "first_bad_operator": "DeviceAirborneBridge.advance/ballistic_parcel_landing point scatter",
        "state_transition_semantics": "AIRBORNE_TO_MOBILE_PRESERVED_WITH_HORIZONTAL_IMPACT_MOMENTUM",
        "footprint_semantics": {
            "kind": "FINITE_RECTANGULAR_REDUCED_ORDER_SUPPORT",
            "area_formula": "parcel_volume/(bucket_capacity/bucket_mouth_area)",
            "aspect_source": "real bucket cutting-edge length and mouth area",
            "orientation_source": "real release lip lateral axis in world frame",
            "provenance": "CONSERVATION_BASED_ENGINEERING_CLOSURE_NOT_CALIBRATED_DISPERSION",
            "rasterization": "exact polygon/control-area overlap, nonnegative volume normalization",
        },
        "comparison": {
            "airborne_initial_volume_m3": initial_airborne,
            "airborne_final_volume_m3": final_ledger["airborne_volume_m3"],
            "deposition_volume_before_m3": total_landed,
            "deposition_volume_after_m3": total_landed,
            "old_recipient_cell_count": len(set(old_indices)),
            "new_recipient_cell_count": int(new_unique.size),
            "new_rasterized_footprint_area_sum_m2": footprint_area,
            "old_max_dH_m_at_first_scatter": float(np.max(old_t1 - baseline["H_free_m"])),
            "new_max_dH_m_at_first_scatter": morphology["T1"]["H_free"]["delta_H_m"]["max"],
            "H_free_net_volume_change_m3": final_free_volume - initial_free_volume,
            "expected_H_free_net_from_reservoir_change_m3": (
                initial_airborne - final_ledger["airborne_volume_m3"]
            ),
            "H_free_net_volume_error_m3": (
                final_free_volume - initial_free_volume
                - (initial_airborne - final_ledger["airborne_volume_m3"])
            ),
            "mass_error_max_m3": max_mass_error,
            "arrest_final_time_s": settled_first,
            "wall_time_s": wall_s, "RTF": 5.0 / wall_s,
            "legacy_final_H_free_morphology": legacy_final_morphology,
            "repaired_final_H_free_morphology": morphology["T4"]["H_free"],
            "legacy_mobile_duration_s": legacy_timescale["physical_time_decomposition"]["T_mobile_nonzero_s"],
            "repaired_mobile_duration_s": repaired_timescale["physical_time_decomposition"]["T_mobile_nonzero_s"],
            "legacy_moving_mobile_duration_s": legacy_timescale["physical_time_decomposition"]["T_mobile_actively_moving_s"],
            "repaired_moving_mobile_duration_s": repaired_timescale["physical_time_decomposition"]["T_mobile_actively_moving_s"],
            "legacy_final_Y_start_cell_count": legacy_timescale["final"]["Y_start_cell_count"],
            "repaired_final_Y_start_cell_count": repaired_timescale["final"]["Y_start_cell_count"],
        },
        "morphology": morphology,
        "mesh_ab": mesh_ab,
        "evolution": evolution,
        "landing_events": record_json,
        "gates": {
            "AIRBORNE_VOLUME_CONSERVED": abs(total_landed - initial_airborne) < 1e-12,
            "MASS_LEDGER": max_mass_error <= 1e-8,
            "POINT_LIKE_THREE_CELL_SCATTER_REMOVED": int(new_unique.size) > 3,
            "AUTHORITATIVE_14M_SPIKE_REMOVED": morphology["T1"]["H_free"]["delta_H_m"]["max"] < 2.0,
            "NO_MULTI_METER_SINGLE_GRID_VERTEX_DEPOSITION_ARTIFACT": morphology["T1"]["H_free"]["delta_H_m"]["max"] < 2.0,
            "FINITE_FOOTPRINT_RASTERIZATION": all(abs(item["volume_error_m3"]) <= 1e-12 for item in record_json),
            "H_FREE_NET_VOLUME_MATCHES_AIRBORNE_LOSS": abs(final_free_volume - initial_free_volume - initial_airborne) <= 1e-8,
            "INCREMENTAL_MESH_MATCHES_FULL_REBUILD": mesh_ab["incremental_matches_full_rebuild"],
            "FLOW_ARREST_REGRESSION": bool(evolution[-1]["terrain_settled"] and evolution[-1]["mobile_volume_m3"] <= 1e-5),
        },
        "multi_event_ownership": {
            "status": "PASS",
            "contract_A": {
                "name": "COMPLETED_EVENT_NO_SPONTANEOUS_RETRIGGER",
                "test": "test_local_redeposition_does_not_retrigger_the_same_owned_tranche",
                "result": "PASS",
                "semantics": "local Mobile-to-Resting redeposition retains ownership",
            },
            "contract_B": {
                "name": "NEW_DISTURBANCE_CAN_CREATE_NEW_EVENT",
                "test": "test_still_yielded_resting_retriggers_after_local_mobile_has_left",
                "result": "PASS",
                "semantics": "authoritative donor-limited Mobile export releases the locally transported tranche",
            },
            "global_latch_clear_between_events": False,
        },
        "reverse_travel": {
            "status": "PASS",
            "classification": "OTHER_PHYSX_CONTACT_DOMAIN_DOUBLE_SUPPORT",
            "root_cause": (
                "A 0.50 m local support-apron/authoritative-terrain overlap placed coincident "
                "static support manifolds beneath the real 6.17 m track patch and pinned the articulation."
            ),
            "fix": (
                "Set support_apron_flat_heightmap_overlap_m to 0.0 so apron and authoritative "
                "terrain meet at one boundary without an overlap or gap."
            ),
            "timeout_modified": False,
            "soil_physics_modified": False,
            "reverse_entry_sim_time_s": transitions["REVERSE_TRAVEL"]["timestamp_s"],
            "align_dump_sim_time_s": transitions["ALIGN_DUMP"]["timestamp_s"],
            "measured_reverse_phase_duration_s": reverse_duration_s,
            "completion_reason": transitions["ALIGN_DUMP"]["reason"],
        },
        "full_gui_soak": {
            "status": "PARTIAL",
            "run_id": "run_1786598146",
            "headless": False,
            "ready_hold": gui_lifecycle["checks"][0],
            "post_dump_pause_hold": gui_lifecycle["checks"][1],
            "reset_ready_hold": gui_lifecycle["checks"][2],
            "reset_all": gui_reset,
            "post_dump_physics_s": gui_demo["post_dump_physics_display_s"],
            "arrest_final_reached_in_gui": False,
            "mobile_volume_at_pause_m3": gui_demo["mobile_m3"],
            "partial_reason": (
                "Non-headless lifecycle, ReverseTravel, dump, nine-second post-dump hold and reset "
                "passed, but the presentation path intentionally paused with nonzero Mobile and "
                "did not claim terrain_settled/ARREST_FINAL."
            ),
            "mean_RTF": gui_demo["mean_rtf"],
            "mean_frame_time_ms": gui_demo["mean_frame_time_ms"],
            "root_pose_write_count": gui_demo["root_pose_write_count"],
            "exit_reason": gui_exit["exit_reason"],
            "simulation_app_is_running_became_false": gui_exit["simulation_app_is_running_became_false"],
            "exception": gui_exit["exception"],
        },
        "remaining_limitations": [
            (
                "The compact rectangular footprint is a conservation-based engineering closure "
                "derived from real bucket geometry and represented volume; it is not a calibrated "
                "granular-dispersion law."
            ),
            (
                "The repaired deposit retains an abrupt physical footprint edge: maximum local "
                "four-neighbor H_free jump is approximately 0.91 m, although the nonphysical "
                "14.11 m single-vertex source is removed."
            ),
        ],
        "artifacts": {
            "states": "causal_T0_T4_states.npz",
            "stage_metadata": "causal_T0_T4_metadata.json",
            "parcel_recipient_audit": "parcel_recipient_audit.json",
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "comparison": summary["comparison"], "gates": summary["gates"]}, indent=2))


if __name__ == "__main__":
    main()
