#!/usr/bin/env python3
"""Formal 701x701 GPU replay for LargeAvalanche lifecycle persistence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (warp_root / "warp").is_dir():
        sys.path.insert(0, str(warp_root))
        break

from isaac_bulk_pipeline.bulk_interaction.large_avalanche import (  # noqa: E402
    LargeAvalancheTransitionConfig,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.runtime import DeviceBulkState, GpuBulkOperatorChain  # noqa: E402
from isaac_bulk_pipeline.runtime.avalanche_persistence import (  # noqa: E402
    PersistenceSample,
    classify_persistence,
)
from isaac_bulk_pipeline.runtime.v2_config import Interactive390FConfig  # noqa: E402
from isaac_bulk_pipeline.solvers import CompactTileFrontier  # noqa: E402
from isaac_bulk_pipeline.terrain import HeightmapIO, TerrainGrid  # noqa: E402
from slope_model import _neighbor_pairs  # noqa: E402

FORMAL_TERRAIN = ROOT / "continuous_heightmap_25m_closed_dataset/sequence_000_H0_track_pile_acceptance_m.csv"
CONFIG_PATH = ROOT / "configs/390f_v2_interactive.yaml"
OLD_RUN = ROOT / "outputs/390f_v2/interactive_runs/run_1786444706"
OUTPUT = ROOT / "outputs/390f_v2/large_avalanche_persistence_acceptance.json"
SAMPLES_OUTPUT = ROOT / "outputs/390f_v2/large_avalanche_persistence_samples.json"
CHECKPOINT_OUTPUT = ROOT / "outputs/390f_v2/large_avalanche_trigger_checkpoint.npz"
FINAL_CHECKPOINT_OUTPUT = ROOT / "outputs/390f_v2/large_avalanche_persistence_latest.npz"
FORMAL_SCENARIO = "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED"
FORMAL_DUMP_VOLUME_M3 = 0.10582992558428088


def material_from_config(path: Path, scenario_id: str):
    document = json.loads(path.read_text(encoding="utf-8"))
    source = next((x for x in document["scenarios"] if x["id"] == scenario_id), None)
    if source is None:
        raise ValueError(f"material scenario not found: {scenario_id}")
    required = {"bulk_density_kg_m3", "internal_friction_angle_deg", "cohesion_pa",
                "tool_wall_friction_angle_deg", "theta_start_deg", "theta_stop_deg",
                "mobile_friction_coefficient"}
    missing = sorted(required - source.keys())
    if missing:
        raise ValueError(f"incomplete material scenario: {missing}")
    material = MaterialScenario(
        scenario_id, source["bulk_density_kg_m3"],
        source["internal_friction_angle_deg"], source["cohesion_pa"],
        float(np.tan(np.deg2rad(source["tool_wall_friction_angle_deg"]))),
        source["theta_start_deg"], source["theta_stop_deg"],
        source["mobile_friction_coefficient"],
    )
    return material, source


def gpu_memory_mb() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0.0
    for line in result.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            if len(parts) == 2 and int(parts[0]) == os.getpid():
                total += float(parts[1])
        except ValueError:
            pass
    return total if total > 0.0 else None


def audit_interrupted_run() -> dict[str, object]:
    rows = json.loads((OLD_RUN / "runtime_telemetry.json").read_text(encoding="utf-8"))
    final = rows[-1]
    avalanche = [row.get("large_avalanche", {}) for row in rows]
    rtfs = [float(row["core_rtf"]) for row in rows if row.get("core_rtf") is not None]
    funnel = final.get("material_funnel", {})
    return {
        "run_directory": str(OLD_RUN.relative_to(ROOT)),
        "sample_count": len(rows),
        "simulation_time_s": float(final["timestamp_s"]),
        "final_state": final.get("state"),
        "max_mobile_m3": max(float(x.get("mobile_m3", 0.0)) for x in rows),
        "final_mobile_m3": float(final.get("mobile_m3", 0.0)),
        "max_connected_unstable_area_m2": max(
            float(x.get("largest_connected_area_m2", 0.0)) for x in avalanche
        ),
        "max_mass_error_m3": max(abs(float(x.get("mass_error_m3", 0.0))) for x in rows),
        "mean_core_rtf": float(np.mean(rtfs)) if rtfs else None,
        "dump_released_volume_m3": funnel.get("dump_released_volume_m3"),
        "deposited_after_dump_throughput_m3": funnel.get("deposited_after_dump_m3"),
        "final_large_avalanche": final.get("large_avalanche"),
        "acceptance_status": "USER_INTERRUPTED_PHYSICS_NOT_SETTLED_NOT_ACCEPTED",
        "not_accepted_marker_present": (
            OLD_RUN / "USER_INTERRUPTED_PHYSICS_NOT_SETTLED_NOT_ACCEPTED.json"
        ).is_file(),
        "diagnostic_gap": (
            "NO_PER_CELL_ACTIVATION_HISTORY; CONSERVATION_AND_LONG_DURATION "
            "CANNOT_EXCLUDE_REMOBILIZATION"
        ),
    }


def compact_dump_source(grid, integrator, volume_m3):
    x, y, sigma = 14.0, 25.0, 0.15
    radius = 3.0 * sigma
    c0, c1 = max(0, int((x-radius)/grid.dx)), min(grid.nx, int(np.ceil((x+radius)/grid.dx))+1)
    r0, r1 = max(0, int((y-radius)/grid.dy)), min(grid.ny, int(np.ceil((y+radius)/grid.dy))+1)
    rr, cc = np.indices((r1-r0, c1-c0), dtype=np.float64)
    xx, yy = (cc+c0)*grid.dx, (rr+r0)*grid.dy
    shape = np.exp(-0.5*(((xx-x)/sigma)**2 + ((yy-y)/sigma)**2))
    weights = integrator.vertex_weights_m2[r0:r1, c0:c1]
    height = shape * (volume_m3 / float(np.sum(shape*weights)))
    indices = ((rr.astype(np.int32)+r0)*grid.nx + cc.astype(np.int32)+c0).ravel()
    return indices, height.ravel(), (r0, r1, c0, c1), float(np.sum(height*weights))


def make_sample(t, result, new, repeat, new_area, mass_error, settled):
    return PersistenceSample(
        simulation_time_s=t,
        newly_activated_volume_m3=new,
        reactivated_volume_m3=repeat,
        cumulative_r2m_m3=result.cumulative_resting_to_mobile_m3,
        cumulative_m2r_m3=result.cumulative_mobile_to_resting_m3,
        net_terrain_volume_change_m3=result.net_terrain_volume_change_m3,
        net_spatial_transfer_m3=result.net_spatial_transfer_m3,
        current_mobile_volume_m3=result.current_mobile_volume_m3,
        maximum_mobile_speed_m_s=result.maximum_mobile_speed_m_s,
        newly_activated_area_m2=new_area,
        reactivated_area_m2=result.reactivated_area_m2,
        retired_candidate_area_m2=result.retired_candidate_area_m2,
        active_tile_count=result.active_tile_count,
        connected_unstable_area_m2=result.largest_connected_area_m2,
        unique_activated_area_m2=result.unique_activated_area_m2,
        unique_cells_ever_activated=result.unique_cells_ever_activated,
        cells_activated_more_than_once=result.cells_activated_more_than_once,
        mass_error_m3=mass_error,
        terrain_settled=settled,
    )


def advance_residual_frontier(chain, frontier, active_tiles, material, grid, tolerance):
    """Run one production-equivalent compact GPU MiniSlope frontier round."""

    critical = float(np.tan(np.deg2rad(material.stop_angle_deg)))
    phase_tiles = np.asarray(active_tiles, dtype=np.int32)
    triggered = []
    for direction, (di, dj, distance) in enumerate(_neighbor_pairs(grid.dx, grid.dy)):
        for phase in (0, 1):
            batch = frontier.edge_batch(
                phase_tiles, direction_index=direction, di=di, dj=dj, phase=phase
            )
            result = chain.transfer_frontier_phase(
                batch, critical_difference_m=critical*distance
            )
            if result.triggered_tile_ids.size:
                triggered.append(result.triggered_tile_ids)
                phase_tiles = np.union1d(phase_tiles, triggered[-1]).astype(np.int32)
    unstable = []
    for direction, (di, dj, distance) in enumerate(_neighbor_pairs(grid.dx, grid.dy)):
        for phase in (0, 1):
            batch = frontier.edge_batch(
                phase_tiles, direction_index=direction, di=di, dj=dj, phase=phase
            )
            result = chain.scan_frontier_phase(
                batch,
                critical_difference_m=critical*distance,
                tolerance_m=tolerance,
            )
            if result.unstable_owner_tile_ids.size:
                unstable.append(result.unstable_owner_tile_ids)
    if unstable:
        return np.unique(np.concatenate(unstable)).astype(np.int32)
    chain.state.runtime.arrays["frontier_reached"].zero_()
    return np.empty(0, dtype=np.int32)


def run(max_simulation_time_s: float, sample_interval_s: float, *, resume: bool) -> dict[str, object]:
    config = Interactive390FConfig.load(CONFIG_PATH)
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/Terrain/FormalReplay")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    terrain = HeightmapIO.load(FORMAL_TERRAIN, grid=grid, source_axis_order="xy")
    material, material_source = material_from_config(
        config.material_scenarios, config.material_scenario_id
    )
    avalanche_config = LargeAvalancheTransitionConfig.from_mapping(
        dict(config.large_avalanche_transition)
    )
    preflight = {
        "runtime_backend": config.runtime_backend,
        "state_authority": "DEVICE",
        "grid_shape_yx": list(grid.shape),
        "grid_resolution_m": grid.dx,
        "terrain_file": str(FORMAL_TERRAIN.relative_to(ROOT)),
        "terrain_file_exact_match": FORMAL_TERRAIN.is_file(),
        "material_scenario": material.name,
        "material_scenario_exact_match": material.name == FORMAL_SCENARIO,
        "operator_chain": "GpuBulkOperatorChain_USED_BY_EarthmovingPhysicsCore",
        "full_field_normal_transfer_disabled": True,
        "replay_scope": "POST_DUMP_LARGE_AVALANCHE_LIFECYCLE_ONLY",
    }
    if not (config.runtime_backend == "GPU_RUNTIME" and grid.shape == (701, 701)
            and grid.dx == 0.05 and preflight["terrain_file_exact_match"]
            and preflight["material_scenario_exact_match"]):
        raise RuntimeError(f"PERSISTENCE_PRE_FLIGHT_REJECTED: {preflight}")

    state = DeviceBulkState(grid, terrain, tile_size=config.active_tile_size)
    indices, heights, bbox, injected = compact_dump_source(
        grid, integrator, FORMAL_DUMP_VOLUME_M3
    )
    resumed = bool(resume and FINAL_CHECKPOINT_OUTPUT.is_file())
    previous_wall_time = 0.0
    previous_max_mass_error = 0.0
    active_frontier_tiles = np.empty(0, dtype=np.int32)
    residual_rounds = 0
    if resumed:
        with np.load(FINAL_CHECKPOINT_OUTPUT, allow_pickle=False) as archive:
            mapping = {
                "resting": "current_resting_height_m",
                "mobile": "current_mobile_height_m",
                "momentum_x": "current_mobile_momentum_x_m2_s",
                "momentum_y": "current_mobile_momentum_y_m2_s",
                "avalanche_activation_count": "activation_count",
                "avalanche_first_activation_time": "first_activation_time_s",
                "avalanche_last_activation_time": "last_activation_time_s",
                "avalanche_r2m_cumulative": "resting_to_mobile_cumulative_m3",
                "avalanche_m2r_cumulative": "mobile_to_resting_cumulative_m3",
                "avalanche_latch": "avalanche_latch",
                "avalanche_previous_component": "avalanche_previous_component",
                "frontier_reached": "frontier_reached",
            }
            restore = {target: np.array(archive[source], copy=True)
                       for target, source in mapping.items()}
            restored_time = float(archive["simulation_time_s"])
            injected = float(archive["injected_dump_volume_m3"])
            active_frontier_tiles = np.asarray(
                archive["active_frontier_tiles"], dtype=np.int32
            )
            residual_rounds = int(archive["residual_minislope_rounds"])
            previous_wall_time = float(archive["cumulative_wall_time_s"])
            previous_max_mass_error = float(archive["max_mass_error_m3"])
        state.restore_checkpoint_fields(
            restore, timestamp_s=restored_time, source="checkpoint"
        )
        setup_transfer = state.transfer_snapshot().to_dict()
    else:
        state.begin_physics_step()
        state.add_host_indices("resting", indices, heights,
                               reason="formal_post_dump_persistence_replay_source")
        setup_transfer = state.transfer_snapshot().to_dict()
        state.mark_dirty_bbox(bbox)
    chain = GpuBulkOperatorChain(state, material, grid, integrator, avalanche_config)

    dt = config.physics_dt_s
    stride = max(1, int(round(sample_interval_s/dt)))
    remaining_time = max(0.0, max_simulation_time_s-state.timestamp_device_s)
    max_steps = int(np.ceil(remaining_time/dt))
    if resumed and SAMPLES_OUTPUT.is_file():
        detailed = json.loads(SAMPLES_OUTPUT.read_text(encoding="utf-8"))
        samples = [PersistenceSample(**{
            name: row[name] for name in PersistenceSample.__dataclass_fields__
        }) for row in detailed]
    else:
        samples, detailed = [], []
    new = repeat = new_area = 0.0
    max_mass_error = previous_max_mass_error
    memory_start = gpu_memory_mb()
    memories = [memory_start] if memory_start is not None else []
    checkpoint_written = resumed or CHECKPOINT_OUTPUT.is_file()
    full_h2d = full_d2h = h2d_bytes = d2h_bytes = 0
    trigger_time = (float(samples[0].simulation_time_s) if resumed and samples else None)
    frontier = CompactTileFrontier(grid.shape, config.active_tile_size)
    peak_residual_tiles = 0
    start_wall = perf_counter()

    for step in range(max_steps):
        state.begin_physics_step()
        mobile_step = chain.step_mobile(dt)
        deposition = chain.step_deposition(
            dt,
            # Exactly the production EarthmovingPhysicsCore tail rule: once
            # less than one formal cell-volume remains, deposition owns the
            # conservative Mobile->Resting closure before residual MiniSlope.
            settle_subcell_tail=(
                mobile_step.volume_after_m3
                <= grid.dx * grid.dy * min(grid.dx, grid.dy)
            ),
        )
        reservoir = state.reservoir_reduction(material.assumed_bulk_density_kg_m3)
        dynamic_quiet = bool(
            reservoir["mobile_volume_m3"] <= avalanche_config.settled_mobile_volume_m3
            and mobile_step.maximum_speed_m_s <= avalanche_config.settled_speed_m_s
        )
        residual_advanced = False
        if dynamic_quiet and active_frontier_tiles.size:
            active_frontier_tiles = advance_residual_frontier(
                chain, frontier, active_frontier_tiles, material, grid,
                config.minislope_tolerance_m,
            )
            residual_rounds += 1
            residual_advanced = True
            result = chain.advance_large_avalanche(
                dt, release_settled_latches=active_frontier_tiles.size == 0
            )
        else:
            result = chain.advance_large_avalanche(dt, release_settled_latches=True)
        if result.residual_seed_tile_ids.size:
            active_frontier_tiles = np.union1d(
                active_frontier_tiles, result.residual_seed_tile_ids
            ).astype(np.int32)
        peak_residual_tiles = max(peak_residual_tiles, int(active_frontier_tiles.size))
        new += result.newly_activated_volume_m3
        repeat += result.reactivated_volume_m3
        new_area += result.newly_activated_area_m2
        mass_error = result.net_terrain_volume_change_m3 - injected
        max_mass_error = max(max_mass_error, abs(mass_error))
        transfer = state.transfer_snapshot()
        full_h2d += transfer.full_field_h2d_count
        full_d2h += transfer.full_field_d2h_count
        h2d_bytes += transfer.h2d_bytes
        d2h_bytes += transfer.d2h_bytes
        state.assert_normal_step_transfer_budget()
        if result.transitioned and trigger_time is None:
            trigger_time = state.timestamp_device_s
        if result.transitioned and not checkpoint_written:
            checkpoint = chain.large_avalanche.persistence_checkpoint(source="checkpoint")
            checkpoint.update({"simulation_time_s": np.asarray(state.timestamp_device_s),
                               "injected_dump_volume_m3": np.asarray(injected),
                               "material_scenario": np.asarray(material.name)})
            CHECKPOINT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(CHECKPOINT_OUTPUT, **checkpoint)
            checkpoint_written = True
        if (step+1) % stride and step+1 != max_steps:
            continue
        settled = bool(
            result.current_mobile_volume_m3 <= avalanche_config.settled_mobile_volume_m3
            and result.maximum_mobile_speed_m_s <= avalanche_config.settled_speed_m_s
            and result.active_tile_count == 0
            and active_frontier_tiles.size == 0
            and result.classification not in {
                chain.large_avalanche.PERSISTING_LARGE_UNSTABLE_REGION,
                chain.large_avalanche.LARGE_AVALANCHE_MOBILE_PATH,
            }
        )
        sample = make_sample(state.timestamp_device_s, result, new, repeat,
                             new_area, mass_error, settled)
        samples.append(sample)
        memory = gpu_memory_mb()
        if memory is not None:
            memories.append(memory)
        detailed.append({**asdict(sample), "classification": result.classification,
                         "deposited_this_step_m3": deposition.deposited_volume_m3,
                         "moving_mobile_volume_m3": result.moving_mobile_volume_m3,
                         "mean_activated_slope_deg": result.mean_activated_slope_deg,
                         "maximum_activated_slope_deg": result.maximum_activated_slope_deg,
                         "mean_slope_minus_theta_stop_deg": result.mean_slope_minus_theta_stop_deg,
                         "mean_theta_start_minus_slope_deg": result.mean_theta_start_minus_slope_deg})
        detailed[-1].update({
            "residual_minislope_active_tiles": int(active_frontier_tiles.size),
            "residual_minislope_rounds": residual_rounds,
            "residual_advanced_this_sample_step": residual_advanced,
        })
        new = repeat = new_area = 0.0
        decision = classify_persistence(samples)
        if (decision.classification == "REMOBILIZATION_LIMIT_CYCLE"
                and state.timestamp_device_s >= 8.0 and len(samples) >= 8):
            break
        if settled:
            break

    wall_time_segment = perf_counter() - start_wall
    wall_time = previous_wall_time + wall_time_segment
    decision = classify_persistence(samples)
    metrics = decision.metrics
    memory_end = gpu_memory_mb()
    if memory_end is not None:
        memories.append(memory_end)
    final_checkpoint = chain.large_avalanche.persistence_checkpoint(source="checkpoint")
    final_checkpoint.update({
        "simulation_time_s": np.asarray(state.timestamp_device_s),
        "injected_dump_volume_m3": np.asarray(injected),
        "material_scenario": np.asarray(material.name),
        "active_frontier_tiles": active_frontier_tiles,
        "residual_minislope_rounds": np.asarray(residual_rounds),
        "cumulative_wall_time_s": np.asarray(wall_time),
        "max_mass_error_m3": np.asarray(max_mass_error),
    })
    np.savez_compressed(FINAL_CHECKPOINT_OUTPUT, **final_checkpoint)
    report = {
        "schema": "LARGE_AVALANCHE_PERSISTENCE_ACCEPTANCE/v1",
        "status": decision.status,
        "classification": decision.classification,
        "reason": decision.reason,
        "simulation_time_s": metrics["simulation_time_s"],
        "wall_time_s": wall_time,
        "wall_time_latest_segment_s": wall_time_segment,
        "max_mobile_m3": metrics["max_mobile_m3"],
        "final_mobile_m3": metrics["final_mobile_m3"],
        "max_connected_unstable_area_m2": metrics["max_connected_unstable_area_m2"],
        "unique_activated_area_m2": metrics["unique_activated_area_m2"],
        "reactivated_area_m2": metrics["reactivated_area_m2"],
        "permanently_retired_area_m2": (
            metrics["unique_activated_area_m2"] if metrics["terrain_settled"] else 0.0
        ),
        "reactivation_ratio": metrics["reactivation_ratio"],
        "circulation_ratio": metrics["circulation_ratio"],
        "cumulative_r2m_m3": metrics["cumulative_r2m_m3"],
        "cumulative_m2r_m3": metrics["cumulative_m2r_m3"],
        "net_terrain_volume_change_m3": metrics["net_terrain_volume_change_m3"],
        "max_mass_error_m3": max_mass_error,
        "gpu_memory_start_mb": memory_start,
        "gpu_memory_peak_mb": max(memories) if memories else None,
        "gpu_memory_end_mb": memory_end,
        "mean_rtf": float(metrics["simulation_time_s"])/max(wall_time, 1e-12),
        "active_tiles_final": metrics["active_tiles_final"],
        "terrain_settled": metrics["terrain_settled"],
        "unique_cells_ever_activated": metrics["unique_cells_ever_activated"],
        "cells_activated_more_than_once": metrics["cells_activated_more_than_once"],
        "trigger_time_s": trigger_time,
        "residual_minislope": {
            "backend": "WarpCompactActiveEdgeOperator",
            "rounds": residual_rounds,
            "peak_active_tiles": peak_residual_tiles,
            "final_active_tiles": int(active_frontier_tiles.size),
            "tolerance_m": config.minislope_tolerance_m,
            "fixed_iteration_cap_used_as_settled": False,
        },
        "replay_input": {
            "type": "DECLARED_COMPACT_POST_DUMP_PERTURBATION",
            "resumed_from_checkpoint": resumed,
            "center_terrain_xy_m": [14.0, 25.0], "bbox_yx": list(bbox),
            "requested_volume_m3": FORMAL_DUMP_VOLUME_M3,
            "injected_volume_m3": injected,
            "claim_boundary": "LIFECYCLE_INPUT_NOT_INTERRUPTED_FULL_STATE_RECONSTRUCTION",
        },
        "preflight": {**preflight, "status": "PASS"},
        "material_parameter_basis": material_source.get("start_stop_angle_status"),
        "diagnostic_thresholds_not_physics": decision.thresholds,
        "diagnostic_metrics": metrics,
        "normal_step_transfer": {
            "full_field_h2d_count": full_h2d, "full_field_d2h_count": full_d2h,
            "h2d_bytes": h2d_bytes, "d2h_bytes": d2h_bytes,
            "status": "PASS" if full_h2d == 0 and full_d2h == 0 else "FAIL",
            "checkpoint_full_field_transfer_excluded_from_normal_path": True,
        },
        "setup_transfer": setup_transfer,
        "checkpoint": {"written": checkpoint_written,
                       "path": str(CHECKPOINT_OUTPUT.relative_to(ROOT)),
                       "semantics": "EXPLICIT_ACCEPTANCE_BOUNDARY_NOT_NORMAL_PHYSICS"},
        "continuation_checkpoint": {
            "path": str(FINAL_CHECKPOINT_OUTPUT.relative_to(ROOT)),
            "simulation_time_s": state.timestamp_device_s,
            "active_frontier_tiles": int(active_frontier_tiles.size),
        },
        "interrupted_run_audit": audit_interrupted_run(),
        "evidence_summary": {
            "new_area_growth": "PERSISTENT_FIRST_ACTIVATION_HISTORY",
            "repeat_activation": "PERSISTENT_PER_CELL_ACTIVATION_COUNT",
            "retirement_claim": "PERMANENT_ONLY_AFTER_WHOLE_EVENT_SETTLED",
            "finite_horizon_policy": "PHYSICS_NOT_SETTLED_NOT_NUMERICAL_FAIL",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    SAMPLES_OUTPUT.write_text(json.dumps(detailed, indent=2)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-simulation-time-s", type=float, default=120.0)
    parser.add_argument("--sample-interval-s", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_simulation_time_s <= 0 or args.sample_interval_s <= 0:
        parser.error("durations must be positive")
    report = run(
        args.max_simulation_time_s, args.sample_interval_s, resume=args.resume
    )
    print(json.dumps({"status": report["status"],
                      "classification": report["classification"],
                      "simulation_time_s": report["simulation_time_s"],
                      "wall_time_s": report["wall_time_s"],
                      "output": str(OUTPUT)}))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
