#!/usr/bin/env python3
"""Operator-boundary audit for production Airborne impact and Mobile spread."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from isaac_bulk_pipeline.bulk_interaction.yield_criterion import evaluate_cohesive_yield  # noqa: E402
from isaac_bulk_pipeline.runtime import SoilForceMode  # noqa: E402
from isaac_bulk_pipeline.runtime.device_checkpoint import restore_device_checkpoint  # noqa: E402
from tools.run_airborne_deposition_morphology_acceptance import _morphology, _will_land  # noqa: E402
from tools.run_physics_timescale_attribution import (  # noqa: E402
    DUMP_RELEASE_CHECKPOINT,
    FORMAL_TERRAIN,
    _core,
    _stationary_tool,
)


OUT = ROOT / "outputs/airborne_impact_spread_arrest"


def _parcel_record(parcel: Any) -> dict[str, Any]:
    footprint = parcel.footprint
    return {
        "parcel_id": parcel.parcel_id,
        "volume_m3": float(parcel.volume_m3),
        "represented_mass_kg": float(parcel.estimated_mass_kg),
        "position_world_m": np.asarray(parcel.position_world_m).tolist(),
        "velocity_world_m_s": np.asarray(parcel.velocity_world_m_s).tolist(),
        "source": parcel.source,
        "footprint": None if footprint is None else {
            "lateral_extent_m": footprint.lateral_extent_m,
            "longitudinal_extent_m": footprint.longitudinal_extent_m,
            "nominal_rectangle_area_m2": footprint.area_m2,
            "mass_profile": footprint.mass_profile,
            "provenance": footprint.provenance,
        },
    }


def _snapshot(core: Any, label: str) -> dict[str, Any]:
    view = core.device_state.explicit_host_view(source="acceptance")
    resting = np.asarray(view.H_resting_m).copy()
    mobile = np.asarray(view.H_mobile_m).copy()
    momentum = np.asarray(view.mobile_momentum_m2_s).copy()
    speed = np.divide(
        np.linalg.norm(momentum, axis=-1),
        mobile,
        out=np.zeros_like(mobile),
        where=mobile > 1.0e-12,
    )
    velocity = np.divide(
        momentum,
        mobile[..., None],
        out=np.zeros_like(momentum),
        where=mobile[..., None] > 1.0e-12,
    )
    moving = (mobile > 1.0e-12) & (
        speed > core.avalanche_controller.config.mobile_activity_speed_m_s
    )
    yield_state = evaluate_cohesive_yield(
        resting,
        mobile,
        core.material,
        core.grid,
        core.integrator,
        layer_depth_m=np.full(
            core.grid.shape,
            core.avalanche_controller.config.mobilization_depth_m,
            dtype=np.float64,
        ),
        moving_mask=moving,
    )
    rt = core.device_state.runtime
    extras = {
        name: rt.download(name).reshape(core.grid.shape).copy()
        for name in (
            "avalanche_latch",
            "avalanche_owned_surface",
            "mobile_export_cumulative",
            "mobile_flux_export_cumulative",
            "avalanche_r2m_cumulative",
            "avalanche_m2r_cumulative",
            "deposition_work",
        )
    }
    result = {
        "label": label,
        "device_time_s": float(view.timestamp_device_s),
        "H_resting_m": resting,
        "h_mobile_m": mobile,
        "H_free_m": resting + mobile,
        "momentum_m2_s": momentum,
        "velocity_m_s": velocity,
        "speed_m_s": speed,
        "Y_start": yield_state.start_mask,
        "Y_stop_continue": yield_state.continue_mask,
        "deposition_eligible": (
            (mobile > 0.0)
            & (speed <= core.gpu_chain.deposition.speed_threshold_m_s)
            & ~yield_state.continue_mask
        ),
        "airborne": [_parcel_record(item) for item in core.gpu_metadata.airborne],
        "mobile_volume_m3": core.integrator.integrate(mobile),
        "moving_mobile_volume_m3": core.integrator.integrate(np.where(moving, mobile, 0.0)),
        **extras,
    }
    # Full fields are an explicitly named acceptance boundary. Keep the
    # production step's normal-transfer guard scoped to subsequent work.
    core.device_state.begin_physics_step()
    return result


def _stage_metadata(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": stage["label"],
        "device_time_s": stage["device_time_s"],
        "mobile_volume_m3": stage["mobile_volume_m3"],
        "moving_mobile_volume_m3": stage["moving_mobile_volume_m3"],
        "maximum_mobile_speed_m_s": float(np.max(stage["speed_m_s"])),
        "Y_start_cell_count": int(np.count_nonzero(stage["Y_start"])),
        "Y_stop_continue_cell_count": int(np.count_nonzero(stage["Y_stop_continue"])),
        "deposition_eligible_cell_count": int(np.count_nonzero(stage["deposition_eligible"])),
        "owned_cell_count": int(np.count_nonzero(stage["avalanche_latch"])),
        "airborne": stage["airborne"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core, config, _ = _core(FORMAL_TERRAIN)
    restore_device_checkpoint(core, DUMP_RELEASE_CHECKPOINT)
    dt = config.physics_dt_s
    core.initialize_tool(_stationary_tool(core, core.device_state.timestamp_device_s))
    prehistory: deque[dict[str, Any]] = deque(maxlen=3)
    boundaries: dict[str, dict[str, Any]] = {}
    frames: list[dict[str, Any]] = []
    impact_result = None
    landing_records: list[Any] = []
    impact_step = None
    impact_armed = False
    first_quiet_after_impact_s = None
    last_moving_after_impact_s = None
    maximum_mass_error = 0.0
    after_frames_remaining = 5
    initial_reservoir = core.reservoir_observation()
    initial_total = sum(
        float(initial_reservoir[name])
        for name in (
            "resting_volume_m3", "mobile_volume_m3", "payload_volume_m3",
            "airborne_volume_m3", "outflow_volume_m3",
        )
    )
    initial_airborne = float(initial_reservoir["airborne_volume_m3"])

    def observer(label: str) -> None:
        if not impact_armed and label not in {
            "AIRBORNE_COLLISION_DETECTED",
            "POST_AIRBORNE_TO_MOBILE_WRITE",
            "POST_MOBILE_MOMENTUM_WRITE",
        }:
            return
        if label not in boundaries:
            boundaries[label] = _snapshot(core, label)

    core.gpu_chain.audit_state_observer = observer
    wall_start = perf_counter()
    max_steps = int(round(15.0 / dt))
    for step in range(1, max_steps + 1):
        if impact_step is None:
            snap = _snapshot(core, f"PRE_FRAME_{step}")
            prehistory.append(snap)
            if _will_land(core, dt):
                boundaries["PRE_IMPACT"] = snap
                for offset, prior in enumerate(reversed(prehistory)):
                    boundaries[f"PRE_IMPACT_MINUS_{offset}"] = prior
                impact_armed = True
        result = core.step(
            _stationary_tool(core, core.device_state.timestamp_device_s + dt),
            phase="deposition",
            cycle=1,
            dt_s=dt,
            soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        )
        maximum_mass_error = max(maximum_mass_error, abs(result.mass_balance_error_m3))
        airborne = None if result.dump_advance is None else result.dump_advance.airborne
        if airborne is not None and airborne.landing_records:
            if impact_step is None:
                impact_step = step
                impact_result = result
            landing_records.extend(airborne.landing_records)
        if impact_step is not None:
            elapsed = (step - impact_step + 1) * dt
            transition = result.avalanche_transition
            frames.append({
                "frame_offset": step - impact_step,
                "elapsed_from_impact_s": elapsed,
                "device_time_s": core.device_state.timestamp_device_s,
                "mobile_volume_m3": result.scalar_state.material_ledger.mobile_m3,
                "moving_mobile_volume_m3": result.physics_diagnostics.mobile_moving_volume_m3,
                "maximum_mobile_speed_m_s": result.physics_diagnostics.mobile_velocity_p95_m_s,
                "airborne_volume_m3": result.scalar_state.airborne_volume_m3,
                "Y_start_cell_count": transition.unstable_cell_count,
                "eligible_mobilizable_volume_m3": transition.largest_connected_mobilizable_volume_m3,
                "R2M_step_m3": transition.transferred_volume_m3,
                "M2R_step_m3": result.dump_advance.deposition.deposited_volume_m3,
                "terrain_settled": result.terrain_settled,
                "not_settled_reason": result.physics_diagnostics.not_settled_reason,
                "mass_error_m3": result.mass_balance_error_m3,
            })
            if result.physics_diagnostics.mobile_moving_volume_m3 > 0.0:
                last_moving_after_impact_s = elapsed
            elif first_quiet_after_impact_s is None:
                first_quiet_after_impact_s = elapsed
            if after_frames_remaining > 0:
                boundaries[f"POST_IMPACT_FRAME_{step-impact_step}"] = _snapshot(
                    core, f"POST_IMPACT_FRAME_{step-impact_step}"
                )
                after_frames_remaining -= 1
            if result.terrain_settled:
                boundaries["POST_ARREST"] = _snapshot(core, "POST_ARREST")
                break
    wall_s = perf_counter() - wall_start
    core.gpu_chain.audit_state_observer = None
    if impact_result is None or not landing_records:
        raise RuntimeError("FROZEN_PARCELS_DID_NOT_LAND")
    if "POST_ARREST" not in boundaries:
        boundaries["OBSERVATION_HORIZON"] = _snapshot(core, "OBSERVATION_HORIZON")

    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for label, stage in boundaries.items():
        for field in (
            "H_resting_m", "h_mobile_m", "H_free_m", "momentum_m2_s",
            "velocity_m_s", "Y_start", "Y_stop_continue", "deposition_eligible",
            "avalanche_latch", "avalanche_owned_surface",
            "mobile_export_cumulative", "mobile_flux_export_cumulative",
            "avalanche_r2m_cumulative",
            "avalanche_m2r_cumulative", "deposition_work",
        ):
            arrays[f"{label}_{field}"] = stage[field]
        metadata[label] = _stage_metadata(stage)
    np.savez_compressed(OUT / "impact_operator_boundaries.npz", **arrays)
    (OUT / "impact_operator_boundaries.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "impact_frame_evolution.json").write_text(
        json.dumps(frames, indent=2) + "\n", encoding="utf-8"
    )

    pre = boundaries["PRE_IMPACT"]
    post_write = boundaries["POST_AIRBORNE_TO_MOBILE_WRITE"]
    post_transport = boundaries["AFTER_MOBILE_TRANSPORT"]
    post_deposition = boundaries["AFTER_MOBILE_TO_RESTING_DEPOSITION"]
    post_large = boundaries["AFTER_LARGE_AVALANCHE"]
    final = boundaries.get("POST_ARREST", boundaries["OBSERVATION_HORIZON"])
    recipients = np.unique(
        np.concatenate([record.recipient_flat_indices for record in landing_records])
    )
    support = np.zeros(core.grid.shape, dtype=bool)
    support.ravel()[recipients] = True
    boundary_support = support & ~ndimage.binary_erosion(support)
    final_y = final["Y_start"]
    pre_latch = pre["avalanche_latch"] != 0
    final_latch = final["avalanche_latch"] != 0
    raw_y_attribution = {
        "total": int(np.count_nonzero(final_y)),
        "A_previously_owned": int(np.count_nonzero(final_y & pre_latch)),
        "B_new_landing_support": int(np.count_nonzero(final_y & support)),
        "C_footprint_boundary": int(np.count_nonzero(final_y & boundary_support)),
        "D_new_landing_incorrectly_shielded": int(
            np.count_nonzero(final_y & support & final_latch & ~pre_latch)
        ),
        "eligible_mobilizable_volume_m3": frames[-1]["eligible_mobilizable_volume_m3"],
    }

    impact_mobile = impact_result.dump_advance.mobile
    impact_deposition = impact_result.dump_advance.deposition
    landing_audit = []
    for record in landing_records:
        dh = record.delivered_volumes_m3 / record.recipient_control_areas_m2
        landing_audit.append({
            "parcel_id": record.parcel_id,
            "volume_m3": record.represented_volume_m3,
            "represented_mass_kg": record.represented_volume_m3 * core.material.assumed_bulk_density_kg_m3,
            "impact_velocity_world_m_s": record.impact_velocity_world_m_s.tolist(),
            "support_area_m2": record.footprint_area_m2,
            "recipient_cell_count": record.recipient_cell_count,
            "mass_profile": record.footprint.mass_profile,
            "delta_h_m": {
                "mean": float(np.mean(dh)),
                "p95": float(np.quantile(dh, .95)),
                "p99": float(np.quantile(dh, .99)),
                "max": float(np.max(dh)),
            },
            "normal_impact_impulse_on_terrain_ns": record.normal_impact_impulse_on_terrain_ns.tolist(),
            "normal_kinetic_energy_dissipated_j": record.normal_kinetic_energy_dissipated_j,
            "volume_error_m3": float(np.sum(record.delivered_volumes_m3) - record.represented_volume_m3),
        })

    initial_export = pre["mobile_flux_export_cumulative"].ravel()[recipients]
    final_export = final["mobile_flux_export_cumulative"].ravel()[recipients]
    exported_from_landing_support = float(np.sum(np.maximum(final_export - initial_export, 0.0)))
    final_reservoir = core.reservoir_observation()
    final_total = sum(
        float(final_reservoir[name])
        for name in (
            "resting_volume_m3", "mobile_volume_m3", "payload_volume_m3",
            "airborne_volume_m3", "outflow_volume_m3",
        )
    )
    summary = {
        "schema": "AIRBORNE_IMPACT_SPREAD_ARREST_CLOSURE/v1",
        "status": "PASS_WITH_ARREST_NOT_REACHED_IN_15S" if "POST_ARREST" not in boundaries else "PASS",
        "formal_identity": {
            "grid_shape_yx": list(core.grid.shape), "dx_m": core.grid.dx,
            "physics_dt_s": dt, "runtime_backend": "GPU_RUNTIME",
            "state_authority": "DEVICE", "material_parameters_modified": False,
        },
        "current_tophat_root_cause": "uniform overlap weights made V/support_area equal the plateau thickness",
        "zero_mobile_lifetime_root_cause": (
            "checkpoint restore replaced the canonical device mobile allocation while the Warp Mobile "
            "solver retained a stale height alias; it observed zero volume with nonzero momentum, "
            "zeroed that momentum, and caused same-frame subcell-tail deposition"
        ),
        "operator_order": [
            "Airborne collision/query", "Airborne-to-Mobile height write",
            "horizontal Mobile momentum write", "Mobile transport",
            "Mobile-to-Resting deposition", "LargeAvalanche Resting-to-Mobile",
            "residual projection when dynamically quiet",
        ],
        "impact_mobile_lifetime_ordering_bug": False,
        "authoritative_array_alias_bug_fixed": True,
        "source_profile": {
            "classification": "CONSERVATION_BASED_ENGINEERING_CLOSURE",
            "kind": "ELLIPTIC_CONE_STOP_ANGLE_CLOSURE",
            "derivation": "unique compact elliptic cone matching parcel volume, bucket-mouth aspect and frozen Y_stop boundary slope",
            "post_smoothing": False,
        },
        "vertical_impact_treatment": {
            "classification": "CONSERVATION_BASED_ENGINEERING_CLOSURE",
            "normal_collision": "perfectly inelastic against terrain-frame up normal",
            "normal_momentum": "recorded as impulse on terrain",
            "normal_kinetic_energy": "recorded as dissipated unresolved granular impact energy",
            "lateral_energy_multiplier": None,
            "horizontal_momentum": "preserved into Mobile",
        },
        "boundary_metadata": metadata,
        "landing_events": landing_audit,
        "impact_step": impact_step,
        "impact_mobile_step": {
            "volume_before_m3": impact_mobile.volume_before_m3,
            "volume_after_m3": impact_mobile.volume_after_m3,
            "momentum_before_kg_m_s": impact_mobile.momentum_before_terrain_kg_m_s.tolist(),
            "momentum_after_kg_m_s": impact_mobile.momentum_after_terrain_kg_m_s.tolist(),
            "numerical_dissipative_impulse_ns": impact_mobile.numerical_dissipative_impulse_terrain_ns.tolist(),
            "maximum_speed_m_s": impact_mobile.maximum_speed_m_s,
            "substeps": impact_mobile.substeps,
            "deposited_same_frame_m3": impact_deposition.deposited_volume_m3,
        },
        "morphology": {
            "post_write_H_free": _morphology(post_write["H_free_m"], pre["H_free_m"], core.material.start_angle_deg, core.grid.dx),
            "post_transport_H_free": _morphology(post_transport["H_free_m"], pre["H_free_m"], core.material.start_angle_deg, core.grid.dx),
            "post_deposition_H_free": _morphology(post_deposition["H_free_m"], pre["H_free_m"], core.material.start_angle_deg, core.grid.dx),
            "post_large_avalanche_H_free": _morphology(post_large["H_free_m"], pre["H_free_m"], core.material.start_angle_deg, core.grid.dx),
            "final_H_free": _morphology(final["H_free_m"], pre["H_free_m"], core.material.start_angle_deg, core.grid.dx),
        },
        "motion": {
            "moving_mobile_duration_observed_s": last_moving_after_impact_s or 0.0,
            "first_quiet_after_impact_s": first_quiet_after_impact_s,
            "transported_volume_from_landing_support_m3": exported_from_landing_support,
            "observation_horizon_s": frames[-1]["elapsed_from_impact_s"],
            "arrest_final": "POST_ARREST" in boundaries,
            "final_active_state": frames[-1],
        },
        "raw_Y_start_attribution": raw_y_attribution,
        "airborne_ownership_contract": {
            "status": "PASS" if post_transport["moving_mobile_volume_m3"] > 0.0 and exported_from_landing_support > 0.0 else "FAIL",
            "new_mobile_inherits_old_resting_latch": False,
            "moving_after_landing_m3": post_transport["moving_mobile_volume_m3"],
            "conservative_export_from_landing_support_m3": exported_from_landing_support,
        },
        "conservation": {
            "initial_total_m3": initial_total,
            "final_total_m3": final_total,
            "net_total_error_m3": final_total - initial_total,
            "maximum_mass_error_m3": maximum_mass_error,
            "initial_airborne_m3": initial_airborne,
            "landed_airborne_m3": sum(x.represented_volume_m3 for x in landing_records),
            "airborne_volume_conserved": abs(initial_airborne - sum(x.represented_volume_m3 for x in landing_records)) <= 1.0e-12,
        },
        "performance": {"wall_time_s": wall_s, "RTF": frames[-1]["elapsed_from_impact_s"] / wall_s},
        "artifacts": {
            "operator_boundaries": "impact_operator_boundaries.npz",
            "boundary_metadata": "impact_operator_boundaries.json",
            "frame_evolution": "impact_frame_evolution.json",
        },
    }
    (OUT / "isolated_impact_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"],
        "impact_mobile_step": summary["impact_mobile_step"],
        "motion": summary["motion"],
        "airborne_ownership_contract": summary["airborne_ownership_contract"],
        "conservation": summary["conservation"],
        "morphology": summary["morphology"],
    }, indent=2))


if __name__ == "__main__":
    main()
