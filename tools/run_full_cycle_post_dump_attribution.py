#!/usr/bin/env python3
"""Replay the exact full-cycle dump boundary without GUI/state-machine effects."""

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
from isaac_bulk_pipeline.runtime.device_checkpoint import restore_device_checkpoint, write_device_checkpoint  # noqa: E402
from tools.run_physics_timescale_attribution import FORMAL_TERRAIN, _core, _stationary_tool  # noqa: E402


OUT = ROOT / "outputs/airborne_impact_spread_arrest"
T_END = OUT / "full_cycle_T_DUMP_END.npz"
T_PLUS_9 = OUT / "full_cycle_T_DUMP_PLUS_9S.npz"


def _archive_state(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(str(data["checkpoint_record_json"]))
        arrays = {
            name: np.asarray(data[name]).copy()
            for name in data.files
            if name != "checkpoint_record_json"
        }
    return {"record": record, "arrays": arrays}


def _metrics(core: Any, resting: np.ndarray, mobile: np.ndarray, mx: np.ndarray, my: np.ndarray, record: dict[str, Any]) -> dict[str, Any]:
    momentum = np.stack((mx, my), axis=-1)
    speed = np.divide(
        np.hypot(mx, my), mobile, out=np.zeros_like(mobile), where=mobile > 1.0e-12
    )
    moving = speed > core.avalanche_controller.config.mobile_activity_speed_m_s
    yield_state = evaluate_cohesive_yield(
        resting, mobile, core.material, core.grid, core.integrator,
        layer_depth_m=np.full(
            core.grid.shape, core.avalanche_controller.config.mobilization_depth_m
        ),
        moving_mask=moving,
    )
    weights = core.integrator.vertex_weights_m2
    density = core.material.assumed_bulk_density_kg_m3
    return {
        "device_time_s": record["device_timestamp_s"],
        "resting_volume_m3": core.integrator.integrate(resting),
        "mobile_volume_m3": core.integrator.integrate(mobile),
        "moving_mobile_volume_m3": core.integrator.integrate(np.where(moving, mobile, 0.0)),
        "maximum_mobile_speed_m_s": float(np.max(speed)),
        "integrated_mobile_momentum_kg_m_s": (
            density * np.sum(momentum * weights[..., None], axis=(0, 1))
        ).tolist(),
        "airborne_volume_m3": float(sum(x["volume_m3"] for x in record["metadata"]["airborne"])),
        "airborne_parcel_count": len(record["metadata"]["airborne"]),
        "payload_volume_m3": float(record["metadata"]["payload"]["volume_m3"]),
        "Y_start_cell_count": int(np.count_nonzero(yield_state.start_mask)),
        "Y_stop_continue_cell_count": int(np.count_nonzero(yield_state.continue_mask)),
        "owned_cell_count": int(np.count_nonzero(record.get("arrays", {}).get("avalanche_latch", 0))),
    }


def main() -> None:
    end = _archive_state(T_END)
    plus = _archive_state(T_PLUS_9)
    core, config, _ = _core(FORMAL_TERRAIN)
    end_metrics = _metrics(
        core, end["arrays"]["resting"], end["arrays"]["mobile"],
        end["arrays"]["momentum_x"], end["arrays"]["momentum_y"],
        {**end["record"], "arrays": end["arrays"]},
    )
    plus_metrics = _metrics(
        core, plus["arrays"]["resting"], plus["arrays"]["mobile"],
        plus["arrays"]["momentum_x"], plus["arrays"]["momentum_y"],
        {**plus["record"], "arrays": plus["arrays"]},
    )

    restore_device_checkpoint(core, T_END)
    core.initialize_tool(_stationary_tool(core, core.device_state.timestamp_device_s))
    dt = config.physics_dt_s
    horizon_s = 60.0
    samples: list[dict[str, Any]] = []
    replay_9_arrays = None
    replay_9_result = None
    settled_time_s = None
    max_mass_error = 0.0
    wall_start = perf_counter()
    for step in range(1, int(round(horizon_s / dt)) + 1):
        result = core.step(
            _stationary_tool(core, core.device_state.timestamp_device_s + dt),
            phase="deposition", cycle=1, dt_s=dt,
            soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        )
        elapsed = step * dt
        max_mass_error = max(max_mass_error, abs(result.mass_balance_error_m3))
        if step % 15 == 0 or step == 1 or result.terrain_settled:
            transition = result.avalanche_transition
            samples.append({
                "elapsed_s": elapsed,
                "mobile_volume_m3": result.scalar_state.material_ledger.mobile_m3,
                "moving_mobile_volume_m3": result.physics_diagnostics.mobile_moving_volume_m3,
                "maximum_mobile_speed_m_s": result.physics_diagnostics.mobile_velocity_p95_m_s,
                "airborne_volume_m3": result.scalar_state.airborne_volume_m3,
                "Y_start_cell_count": transition.unstable_cell_count,
                "eligible_mobilizable_volume_m3": transition.largest_connected_mobilizable_volume_m3,
                "classification": transition.classification,
                "R2M_step_m3": transition.transferred_volume_m3,
                "M2R_step_m3": result.dump_advance.deposition.deposited_volume_m3,
                "terrain_settled": result.terrain_settled,
                "not_settled_reason": result.physics_diagnostics.not_settled_reason,
                "mass_error_m3": result.mass_balance_error_m3,
            })
        if abs(elapsed - 9.0) <= 0.5 * dt:
            view = core.device_state.explicit_host_view(source="acceptance")
            replay_9_arrays = {
                "resting": np.asarray(view.H_resting_m).copy(),
                "mobile": np.asarray(view.H_mobile_m).copy(),
                "momentum_x": np.asarray(view.mobile_momentum_m2_s[..., 0]).copy(),
                "momentum_y": np.asarray(view.mobile_momentum_m2_s[..., 1]).copy(),
            }
            replay_9_result = samples[-1] if samples else None
            core.device_state.begin_physics_step()
        if result.terrain_settled:
            settled_time_s = elapsed
            break
    wall_s = perf_counter() - wall_start
    final_result = result
    write_device_checkpoint(
        core, OUT / "full_cycle_replay_final.npz",
        provenance={
            "checkpoint_boundary": "HEADLESS_REPLAY_FINAL",
            "source_checkpoint": str(T_END.relative_to(ROOT)),
            "elapsed_physics_s": samples[-1]["elapsed_s"],
            "arrest_final": settled_time_s is not None,
        },
    )
    if replay_9_arrays is None:
        raise RuntimeError("REPLAY_DID_NOT_CAPTURE_9S")
    comparison = {}
    for field in ("resting", "mobile", "momentum_x", "momentum_y"):
        error = np.abs(replay_9_arrays[field] - plus["arrays"][field])
        comparison[field] = {
            "max_abs_error": float(np.max(error)),
            "L1": float(np.sum(error)),
            "changed_cell_count": int(np.count_nonzero(error > 1.0e-12)),
        }
    classification = []
    if end_metrics["mobile_volume_m3"] > 0.1:
        classification.append("MORE_MOBILE_INITIAL_CONDITION")
    if np.linalg.norm(end_metrics["integrated_mobile_momentum_kg_m_s"]) > 1.0:
        classification.append("DIFFERENT_MOMENTUM_STATE")
    if end_metrics["airborne_volume_m3"] > 0.0:
        classification.append("CONTINUING_AIRBORNE_INPUT")
    if comparison["resting"]["changed_cell_count"] or comparison["mobile"]["changed_cell_count"]:
        classification.append("GUI_STATE_MACHINE_SIDE_EFFECT")
    summary = {
        "schema": "FULL_CYCLE_POST_DUMP_ATTRIBUTION/v1",
        "T_DUMP_END": end_metrics,
        "T_DUMP_PLUS_9S": plus_metrics,
        "headless_replay_plus_9": replay_9_result,
        "full_cycle_vs_replay_plus_9": comparison,
        "classification": classification,
        "replay": {
            "arrest_final": settled_time_s is not None,
            "arrest_final_time_s": settled_time_s,
            "observation_horizon_s": samples[-1]["elapsed_s"],
            "final_active_state": samples[-1],
            "maximum_mass_error_m3": max_mass_error,
            "wall_time_s": wall_s,
            "RTF": samples[-1]["elapsed_s"] / wall_s,
        },
        "samples": samples,
    }
    (OUT / "full_cycle_post_dump_attribution.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: summary[key] for key in (
        "T_DUMP_END", "T_DUMP_PLUS_9S", "headless_replay_plus_9",
        "full_cycle_vs_replay_plus_9", "classification", "replay"
    )}, indent=2))


if __name__ == "__main__":
    main()
