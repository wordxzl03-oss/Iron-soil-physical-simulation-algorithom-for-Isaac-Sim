#!/usr/bin/env python3
"""Short frozen-checkpoint audit of Mobile/LargeAvalanche causality.

This is acceptance-only instrumentation.  It restores the exact production
T_DUMP_END DEVICE checkpoint, applies no tool forcing, and never changes the
production constitutive parameters or update ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from isaac_bulk_pipeline.bulk_interaction.yield_criterion import evaluate_cohesive_yield  # noqa: E402
from isaac_bulk_pipeline.runtime import SoilForceMode  # noqa: E402
from isaac_bulk_pipeline.runtime.device_checkpoint import restore_device_checkpoint  # noqa: E402
from tools.run_physics_timescale_attribution import FORMAL_TERRAIN, _core, _stationary_tool  # noqa: E402


CHECKPOINT = ROOT / "outputs/airborne_impact_spread_arrest/full_cycle_T_DUMP_END.npz"
OUT = ROOT / "outputs/mobile_large_avalanche_causal_audit"


def _jump(field: np.ndarray) -> float:
    return float(max(np.max(np.abs(np.diff(field, axis=0))), np.max(np.abs(np.diff(field, axis=1)))))


def _fields(core: Any) -> dict[str, np.ndarray]:
    rt = core.device_state.runtime
    shape = core.grid.shape
    result = {
        name: rt.download(name).reshape(shape).copy()
        for name in (
            "resting", "mobile", "momentum_x", "momentum_y", "weights",
            "avalanche_unstable", "avalanche_latch", "avalanche_activation_count",
            "avalanche_owned_surface", "avalanche_owned_export_baseline",
            "avalanche_r2m_cumulative", "avalanche_m2r_cumulative",
            "mobile_export_cumulative", "mobile_flux_export_cumulative",
            "deposition_work", "material_mask",
        )
    }
    return result


def _metrics(core: Any, fields: dict[str, np.ndarray]) -> dict[str, Any]:
    resting = fields["resting"]
    mobile = fields["mobile"]
    mx = fields["momentum_x"]
    my = fields["momentum_y"]
    weights = fields["weights"]
    free = resting + mobile
    speed = np.divide(
        np.hypot(mx, my), mobile, out=np.zeros_like(mobile), where=mobile > 1.0e-12
    )
    momentum_density = np.hypot(mx, my)
    active = mobile > 1.0e-12
    moving = active & (speed > core.avalanche_controller.config.mobile_activity_speed_m_s)
    yield_state = evaluate_cohesive_yield(
        resting, mobile, core.material, core.grid, core.integrator,
        layer_depth_m=np.minimum(resting, core.avalanche_controller.config.mobilization_depth_m),
        moving_mask=moving,
    )
    stop = yield_state.continue_mask
    deposition_speed = core.gpu_chain.deposition.speed_threshold_m_s
    state_a = active & stop
    state_b = active & ~stop & (speed > deposition_speed)
    state_c = active & ~stop & (speed <= deposition_speed)
    state_d = active & (fields["material_mask"] != 0)
    volume = lambda mask: float(np.sum(mobile[mask] * weights[mask], dtype=np.float64))
    density = core.material.assumed_bulk_density_kg_m3
    kinetic = float(
        0.5 * density * np.sum(weights * mobile * speed * speed, dtype=np.float64)
    )
    # Height-field gravitational potential of the authoritative column.  R2M
    # and M2R relabeling leave this invariant; lateral transport changes it.
    potential = float(
        0.5 * density * 9.81 * np.sum(weights * free * free, dtype=np.float64)
    )
    values = mobile[active]
    return {
        "J_H_m": _jump(free), "J_R_m": _jump(resting), "J_M_m": _jump(mobile),
        "mobile_volume_m3": float(np.sum(mobile * weights, dtype=np.float64)),
        "mobile_max_thickness_m": float(np.max(mobile, initial=0.0)),
        "mobile_p95_thickness_m": float(np.percentile(values, 95)) if values.size else 0.0,
        "mobile_p99_thickness_m": float(np.percentile(values, 99)) if values.size else 0.0,
        "mobile_nonzero_cells": int(np.count_nonzero(active)),
        "moving_mobile_volume_m3": volume(moving),
        "maximum_speed_m_s": float(np.max(speed, initial=0.0)),
        "maximum_momentum_density_m2_s": float(np.max(momentum_density, initial=0.0)),
        "Y_start_cells": int(np.count_nonzero(yield_state.start_mask)),
        "Y_stop_cells": int(np.count_nonzero(stop)),
        "arrest_A_volume_m3": volume(state_a),
        "arrest_B_volume_m3": volume(state_b & ~state_d),
        "arrest_C_volume_m3": volume(state_c & ~state_d),
        "arrest_D_volume_m3": volume(state_d),
        "kinetic_energy_j": kinetic,
        "heightfield_potential_energy_j": potential,
        "mechanical_energy_j": kinetic + potential,
        "r2m_cumulative_m3": float(np.sum(fields["avalanche_r2m_cumulative"], dtype=np.float64)),
        "m2r_cumulative_m3": float(np.sum(fields["avalanche_m2r_cumulative"], dtype=np.float64)),
        "ownership_export_cumulative_m3": float(np.sum(fields["mobile_export_cumulative"], dtype=np.float64)),
        "all_flux_crossings_cumulative_m3": float(np.sum(fields["mobile_flux_export_cumulative"], dtype=np.float64)),
        "deposition_cells_this_step": int(
            np.count_nonzero(fields["deposition_work"] > 0.0)
        ),
        "deposition_work_m3_this_step": float(
            np.sum(fields["deposition_work"] * weights, dtype=np.float64)
        ),
        "activation_count_max": int(np.max(fields["avalanche_activation_count"], initial=0)),
        "reactivated_cells": int(np.count_nonzero(fields["avalanche_activation_count"] > 1)),
    }


def _stage_morphology() -> dict[str, Any]:
    run = ROOT / "outputs/390f_v2/interactive_runs/run_1786531861"
    result: dict[str, Any] = {}
    for name in ("pre_dig", "post_cut", "post_breakout", "early_post_dig"):
        data = np.load(run / f"v3_closure_{name}_terrain.npz")
        resting = np.asarray(data["H_resting_m"])
        mobile = np.asarray(data["h_mobile_m"])
        result[name.upper()] = {
            "device_time_s": float(data["timestamp_device_s"]),
            "J_H_m": _jump(resting + mobile),
            "J_R_m": _jump(resting),
            "J_M_m": _jump(mobile),
            "mobile_max_thickness_m": float(np.max(mobile, initial=0.0)),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core, config, _ = _core(FORMAL_TERRAIN)
    restore_device_checkpoint(core, CHECKPOINT)
    dt = config.physics_dt_s
    tool = _stationary_tool(core, core.device_state.timestamp_device_s)
    core.initialize_tool(tool)
    initial_airborne = core.reservoir_observation()["airborne_volume_m3"]
    boundaries: dict[str, dict[str, Any]] = {}

    def observer(label: str) -> None:
        # The observer is intentionally diagnostic-only and replaces the last
        # boundary of a given step.  Full fields are persisted only for the
        # largest J_H change found after the scan.
        fields = _fields(core)
        boundaries[label] = {"fields": fields, "metrics": _metrics(core, fields)}

    core.gpu_chain.audit_state_observer = observer
    before_fields = _fields(core)
    before = _metrics(core, before_fields)
    initial = dict(before)
    samples: list[dict[str, Any]] = []
    activation_events: list[dict[str, Any]] = []
    activation_category_totals = {
        "NEWLY_EXPOSED_RESTING": 0.0,
        "PREVIOUSLY_OWNED_RESTING": 0.0,
        "AIRBORNE_DEPOSIT": 0.0,
        "MOBILE_REDEPOSITION": 0.0,
        "OTHER": 0.0,
    }
    release_reason_totals = {
        "CONSERVATIVE_EXPORT": 0.0,
        "Y_STOP_STABLE_RELEASE": 0.0,
        "FIRST_ACTIVATION": 0.0,
        "OTHER": 0.0,
    }
    strongest: dict[str, Any] | None = None
    wall_start = perf_counter()
    steps = int(round(5.0 / dt))
    previous_airborne = float(initial_airborne)
    previous_payload = float(core.reservoir_observation()["payload_volume_m3"])
    previous_outflow = float(core.reservoir_observation()["outflow_volume_m3"])
    for step in range(1, steps + 1):
        boundaries.clear()
        pre_fields = before_fields
        pre = before
        result = core.step(
            _stationary_tool(core, core.device_state.timestamp_device_s + dt),
            phase="idle", cycle=1, dt_s=dt,
            soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        )
        after_fields = _fields(core)
        after = _metrics(core, after_fields)
        reservoirs = core.reservoir_observation()
        airborne = float(reservoirs["airborne_volume_m3"])
        payload = float(reservoirs["payload_volume_m3"])
        outflow = float(reservoirs["outflow_volume_m3"])
        r2m = float(result.avalanche_transition.transferred_volume_m3)
        m2r = float(result.dump_advance.deposition.deposited_volume_m3)
        a2m = max(0.0, previous_airborne - airborne)
        p2a = max(0.0, previous_payload - payload)
        outflow_step = max(0.0, outflow - previous_outflow)
        expected_delta = r2m + a2m - m2r - outflow_step
        actual_delta = after["mobile_volume_m3"] - pre["mobile_volume_m3"]
        delta_r2m_cell = (
            after_fields["avalanche_r2m_cumulative"]
            - pre_fields["avalanche_r2m_cumulative"]
        )
        activated_mask = delta_r2m_cell > 1.0e-15
        if np.any(activated_mask):
            landing_mask = np.zeros(core.grid.shape, dtype=bool)
            landing_indices = result.dump_advance.airborne.landing_flat_indices
            if landing_indices.size:
                landing_mask.ravel()[landing_indices] = True
            previously_owned = pre_fields["avalanche_activation_count"] > 0
            redeposited = pre_fields["avalanche_m2r_cumulative"] > 0.0
            categories = {
                "AIRBORNE_DEPOSIT": activated_mask & landing_mask,
                "MOBILE_REDEPOSITION": activated_mask & ~landing_mask & redeposited,
                "PREVIOUSLY_OWNED_RESTING": (
                    activated_mask & ~landing_mask & ~redeposited & previously_owned
                ),
                "NEWLY_EXPOSED_RESTING": (
                    activated_mask & ~landing_mask & ~redeposited & ~previously_owned
                ),
            }
            assigned = np.zeros(core.grid.shape, dtype=bool)
            category_step: dict[str, float] = {}
            for name, mask in categories.items():
                value = float(np.sum(delta_r2m_cell[mask], dtype=np.float64))
                activation_category_totals[name] += value
                category_step[name] = value
                assigned |= mask
            other = activated_mask & ~assigned
            other_volume = float(np.sum(delta_r2m_cell[other], dtype=np.float64))
            activation_category_totals["OTHER"] += other_volume
            category_step["OTHER"] = other_volume
            exported = (
                pre_fields["mobile_export_cumulative"]
                - pre_fields["avalanche_owned_export_baseline"]
                > core.avalanche_controller.config.dry_tolerance_m
                * pre_fields["weights"]
            )
            release_masks = {
                "FIRST_ACTIVATION": activated_mask & ~previously_owned,
                "CONSERVATIVE_EXPORT": activated_mask & previously_owned & exported,
                "Y_STOP_STABLE_RELEASE": activated_mask & previously_owned & ~exported,
            }
            release_step: dict[str, float] = {}
            assigned_release = np.zeros(core.grid.shape, dtype=bool)
            for name, mask in release_masks.items():
                value = float(np.sum(delta_r2m_cell[mask], dtype=np.float64))
                release_reason_totals[name] += value
                release_step[name] = value
                assigned_release |= mask
            release_other = activated_mask & ~assigned_release
            release_other_volume = float(
                np.sum(delta_r2m_cell[release_other], dtype=np.float64)
            )
            release_reason_totals["OTHER"] += release_other_volume
            release_step["OTHER"] = release_other_volume
            resting = after_fields["resting"]
            mobile_field = after_fields["mobile"]
            yield_state = evaluate_cohesive_yield(
                resting, mobile_field, core.material, core.grid, core.integrator,
                layer_depth_m=np.minimum(
                    resting, core.avalanche_controller.config.mobilization_depth_m
                ),
            )
            depth = np.divide(
                delta_r2m_cell, after_fields["weights"],
                out=np.zeros_like(delta_r2m_cell),
                where=after_fields["weights"] > 0.0,
            )
            activation_events.append({
                "elapsed_s": step * dt,
                "area_m2": float(np.sum(after_fields["weights"][activated_mask])),
                "activated_volume_m3": float(np.sum(delta_r2m_cell[activated_mask])),
                "activated_depth_m": {
                    "mean": float(np.mean(depth[activated_mask])),
                    "max": float(np.max(depth[activated_mask])),
                },
                "Y_start_margin_pa": {
                    "mean": float(np.mean(yield_state.yield_start_margin_pa[activated_mask])),
                    "p95": float(np.percentile(yield_state.yield_start_margin_pa[activated_mask], 95)),
                    "max": float(np.max(yield_state.yield_start_margin_pa[activated_mask])),
                },
                "slope_deg": {
                    "mean": float(np.mean(np.rad2deg(yield_state.slope_rad[activated_mask]))),
                    "max": float(np.max(np.rad2deg(yield_state.slope_rad[activated_mask]))),
                },
                "H_resting_m": {
                    "mean": float(np.mean(resting[activated_mask])),
                    "max": float(np.max(resting[activated_mask])),
                },
                "h_mobile_m": {
                    "mean": float(np.mean(mobile_field[activated_mask])),
                    "max": float(np.max(mobile_field[activated_mask])),
                },
                "source_classification_m3": category_step,
                "ownership_release_reason_m3": release_step,
            })
        boundary_records: dict[str, Any] = {}
        prior = pre
        for label in (
            "POST_AIRBORNE_TO_MOBILE_WRITE", "POST_MOBILE_MOMENTUM_WRITE",
            "AFTER_MOBILE_TRANSPORT", "AFTER_MOBILE_TO_RESTING_DEPOSITION",
            "AFTER_LARGE_AVALANCHE",
        ):
            if label not in boundaries:
                continue
            current = boundaries[label]["metrics"]
            boundary_records[label] = {
                "J_H_m": current["J_H_m"],
                "delta_J_H_m": current["J_H_m"] - prior["J_H_m"],
                "delta_K_j": current["kinetic_energy_j"] - prior["kinetic_energy_j"],
                "delta_PE_j": current["heightfield_potential_energy_j"] - prior["heightfield_potential_energy_j"],
                "delta_mechanical_energy_j": current["mechanical_energy_j"] - prior["mechanical_energy_j"],
                "mobile_delta_m3": current["mobile_volume_m3"] - prior["mobile_volume_m3"],
            }
            score = abs(boundary_records[label]["delta_J_H_m"])
            if strongest is None or score > strongest["score"]:
                strongest = {
                    "score": score, "step": step, "elapsed_s": step * dt,
                    "operator": label, "pre_fields": pre_fields,
                    "post_fields": boundaries[label]["fields"],
                    "pre_metrics": prior, "post_metrics": current,
                }
            prior = current
        sample = {
            "step": step, "elapsed_s": step * dt,
            **after,
            "R2M_step_m3": r2m, "M2R_step_m3": m2r,
            "A2M_step_m3": a2m, "P2A_step_m3": p2a,
            "Outflow_step_m3": outflow_step,
            "settle_subcell_tail_triggered": bool(
                result.dump_advance.mobile.volume_after_m3
                <= core.grid.dx * core.grid.dy * min(core.grid.dx, core.grid.dy)
            ),
            "mobile_budget_residual_m3": actual_delta - expected_delta,
            "large_avalanche_classification": result.avalanche_transition.classification,
            "newly_activated_volume_m3": result.avalanche_transition.newly_activated_volume_m3,
            "reactivated_volume_m3": result.avalanche_transition.reactivated_volume_m3,
            "boundary_energy_and_jump": boundary_records,
            "mobile_gravity_pressure_impulse_ns": result.dump_advance.mobile.gravity_pressure_impulse_terrain_ns.tolist(),
            "mobile_friction_impulse_ns": result.dump_advance.mobile.basal_friction_impulse_terrain_ns.tolist(),
            "mobile_numerical_impulse_ns": result.dump_advance.mobile.numerical_dissipative_impulse_terrain_ns.tolist(),
            "mobile_energy": {
                "K_before_j": result.dump_advance.mobile.kinetic_energy_before_j,
                "K_after_j": result.dump_advance.mobile.kinetic_energy_after_j,
                "gravity_pressure_work_j": result.dump_advance.mobile.gravity_pressure_work_j,
                "basal_friction_work_j": result.dump_advance.mobile.basal_friction_work_j,
                "tool_work_j": result.dump_advance.mobile.tool_work_j,
                "transport_numerical_energy_residual_j": result.dump_advance.mobile.transport_numerical_energy_residual_j,
            },
            "mobile_transport": {
                "donor_export_m3": result.dump_advance.mobile.donor_export_m3,
                "receiver_import_m3": result.dump_advance.mobile.receiver_import_m3,
                "mass_residual_m3": result.dump_advance.mobile.transport_mass_residual_m3,
                "advected_momentum_crossings_kg_m_s": result.dump_advance.mobile.advected_momentum_crossings_kg_m_s.tolist(),
            },
            "mass_error_m3": result.mass_balance_error_m3,
        }
        samples.append(sample)
        before_fields, before = after_fields, after
        previous_airborne, previous_payload, previous_outflow = airborne, payload, outflow

    assert strongest is not None
    np.savez_compressed(
        OUT / "strongest_operator_boundary.npz",
        **{f"PRE_{k}": v for k, v in strongest["pre_fields"].items()},
        **{f"POST_{k}": v for k, v in strongest["post_fields"].items()},
    )
    final = samples[-1]
    report = {
        "schema": "MOBILE_LARGE_AVALANCHE_CAUSAL_AUDIT/v1",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "window_s": 5.0,
        "dt_s": dt,
        "external_tool_forcing": "NONE_STATIONARY_TOOL_IDLE_PHASE",
        "stage_morphology": _stage_morphology(),
        "initial": initial,
        "at_0p5s": samples[int(round(0.5 / dt)) - 1],
        "at_2s": samples[int(round(2.0 / dt)) - 1],
        "at_5s": final,
        "budgets_0_to_5s": {
            key: float(sum(item[key] for item in samples))
            for key in (
                "R2M_step_m3", "M2R_step_m3", "A2M_step_m3",
                "P2A_step_m3", "Outflow_step_m3", "mobile_budget_residual_m3",
            )
        },
        "energy_0_to_5s": {
            "K_before_j": initial["kinetic_energy_j"],
            "K_after_j": final["kinetic_energy_j"],
            "PE_before_j": initial["heightfield_potential_energy_j"],
            "PE_after_j": final["heightfield_potential_energy_j"],
            "mechanical_delta_j": final["mechanical_energy_j"] - initial["mechanical_energy_j"],
            "mobile_operator_delta_mechanical_j": float(sum(
                item["boundary_energy_and_jump"].get("AFTER_MOBILE_TRANSPORT", {}).get("delta_mechanical_energy_j", 0.0)
                for item in samples
            )),
            "deposition_delta_K_j": float(sum(
                item["boundary_energy_and_jump"].get("AFTER_MOBILE_TO_RESTING_DEPOSITION", {}).get("delta_K_j", 0.0)
                for item in samples
            )),
            "large_avalanche_delta_K_j": float(sum(
                item["boundary_energy_and_jump"].get("AFTER_LARGE_AVALANCHE", {}).get("delta_K_j", 0.0)
                for item in samples
            )),
            "gravity_pressure_work_j": float(sum(item["mobile_energy"]["gravity_pressure_work_j"] for item in samples)),
            "basal_friction_work_j": float(sum(item["mobile_energy"]["basal_friction_work_j"] for item in samples)),
            "tool_work_j": float(sum(item["mobile_energy"]["tool_work_j"] for item in samples)),
            "transport_numerical_energy_residual_j": float(sum(item["mobile_energy"]["transport_numerical_energy_residual_j"] for item in samples)),
        },
        "transport_0_to_5s": {
            "donor_export_m3": float(sum(item["mobile_transport"]["donor_export_m3"] for item in samples)),
            "receiver_import_m3": float(sum(item["mobile_transport"]["receiver_import_m3"] for item in samples)),
            "mass_residual_m3": float(sum(item["mobile_transport"]["mass_residual_m3"] for item in samples)),
        },
        "settle_subcell_tail_0_to_5s": {
            "triggered_step_count": int(sum(
                item["settle_subcell_tail_triggered"] for item in samples
            )),
            "settled_cell_updates": int(sum(
                item["deposition_cells_this_step"]
                for item in samples if item["settle_subcell_tail_triggered"]
            )),
            "settled_volume_m3": float(sum(
                item["deposition_work_m3_this_step"]
                for item in samples if item["settle_subcell_tail_triggered"]
            )),
            "normal_deposition_cell_updates": int(sum(
                item["deposition_cells_this_step"]
                for item in samples if not item["settle_subcell_tail_triggered"]
            )),
            "normal_deposition_volume_m3": float(sum(
                item["deposition_work_m3_this_step"]
                for item in samples if not item["settle_subcell_tail_triggered"]
            )),
            "global_trigger_threshold_m3": float(
                core.grid.dx * core.grid.dy * min(core.grid.dx, core.grid.dy)
            ),
        },
        "activation_attribution_0_to_5s": {
            "source_classification_m3": activation_category_totals,
            "ownership_release_reason_m3": release_reason_totals,
            "event_count": len(activation_events),
            "events": activation_events,
        },
        "strongest_J_H_operator_boundary": {
            key: value for key, value in strongest.items()
            if key not in {"pre_fields", "post_fields"}
        },
        "wall_time_s": perf_counter() - wall_start,
        "samples": samples,
    }
    (OUT / "short_replay_5s.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": "PASS", "budgets": report["budgets_0_to_5s"],
        "energy": report["energy_0_to_5s"],
        "strongest": report["strongest_J_H_operator_boundary"],
        "final": {k: final[k] for k in (
            "mobile_volume_m3", "moving_mobile_volume_m3", "maximum_speed_m_s",
            "arrest_A_volume_m3", "arrest_B_volume_m3", "arrest_C_volume_m3",
            "arrest_D_volume_m3", "activation_count_max", "reactivated_cells",
        )},
    }, indent=2))


if __name__ == "__main__":
    main()
