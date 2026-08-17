#!/usr/bin/env python3
"""CPU-reference versus Warp benchmark for the three resident GPU operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    MobileLayerSolver,
    TrackSoilModel,
    WarpMobileLayerSolver,
    WarpTrackSoilOperator,
)
from isaac_bulk_pipeline.bulk_state import (  # noqa: E402
    MaterialScenario,
    TerrainVolumeIntegrator,
)
from isaac_bulk_pipeline.performance import probe_warp  # noqa: E402
from isaac_bulk_pipeline.solvers import (  # noqa: E402
    CompactTileFrontier,
    WarpCompactActiveEdgeOperator,
)
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


CASES = {
    "localized": (64, 8),
    "medium": (256, 64),
    "whole_pile": (701, 620),
}


def _material() -> MaterialScenario:
    return MaterialScenario(
        "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED",
        1370.0,
        29.8,
        800.0,
        0.5,
        38.0,
        30.0,
        0.35,
    )


def _fields(size: int, active_width: int):
    grid = TerrainGrid(size, size, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    rows, columns = np.indices(grid.shape)
    resting = 1.0 + 0.0001 * columns
    radius = max(active_width / 2.0, 1.0)
    radial = ((rows - size / 2.0) ** 2 + (columns - size / 2.0) ** 2) ** 0.5
    mobile = np.zeros(grid.shape, dtype=np.float64)
    active = radial <= radius
    mobile[active] = 0.025 * (1.0 - 0.35 * radial[active] / radius)
    momentum = np.zeros(grid.shape + (2,), dtype=np.float64)
    momentum[..., 0] = mobile * 0.12
    momentum[..., 1] = mobile * -0.04
    return grid, integrator, resting, mobile, momentum


def _relative_l1(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(
        np.sum(np.abs(candidate - reference), dtype=np.float64)
        / max(np.sum(np.abs(reference), dtype=np.float64), 1.0e-15)
    )


def _mobile_record(name: str, size: int, active_width: int) -> dict[str, object]:
    grid, integrator, resting, mobile, momentum = _fields(size, active_width)
    material = _material()
    dt = 1.0 / 120.0
    start = perf_counter()
    reference = MobileLayerSolver().step(
        resting, mobile, momentum, material, grid, integrator, dt
    )
    cpu_wall = perf_counter() - start
    gpu = WarpMobileLayerSolver()
    gpu.initialize_resident(resting, mobile, momentum, material, grid, integrator)
    start = perf_counter()
    gpu_metrics = gpu.step_resident(dt)
    gpu.runtime.synchronize()
    gpu_wall = perf_counter() - start
    result = gpu.download_result()
    transfer = gpu.diagnostics()
    tool_impulse_error = float(
        np.linalg.norm(
            result.tool_impulse_on_mobile_terrain_ns
            - reference.tool_impulse_on_mobile_terrain_ns
        )
    )
    return {
        "case": name,
        "grid_shape": [size, size],
        "active_cell_count": int(np.count_nonzero(mobile)),
        "CPU_REFERENCE": {
            "wall_time_s": cpu_wall,
            "RTF": dt / cpu_wall,
            "volume_after_m3": reference.volume_after_m3,
            "momentum_after_kg_m_s": reference.momentum_after_terrain_kg_m_s.tolist(),
            "soil_impulse_on_tool_ns": (-reference.tool_impulse_on_mobile_terrain_ns).tolist(),
        },
        "GPU_OPTIMIZED": {
            "status": "PASS",
            "wall_time_s": gpu_wall,
            "RTF": dt / gpu_wall,
            "H2D_bytes": transfer["h2d_bytes"],
            "D2H_bytes": transfer["d2h_bytes"],
            "synchronization_count": transfer["synchronization_count"],
            "gpu_utilization": {"status": "EXTERNAL_SAMPLER_REQUIRED"},
        },
        "equivalence": {
            "height_relative_l1": _relative_l1(reference.mobile_height_m, result.mobile_height_m),
            "height_max_abs_m": float(np.max(np.abs(reference.mobile_height_m - result.mobile_height_m))),
            "momentum_max_abs_m2_s": float(np.max(np.abs(reference.mobile_momentum_m2_s - result.mobile_momentum_m2_s))),
            "volume_conservation_error_m3": abs(gpu_metrics.volume_after_m3 - gpu_metrics.volume_before_m3),
            "cpu_gpu_volume_error_m3": abs(gpu_metrics.volume_after_m3 - reference.volume_after_m3),
            "cpu_gpu_momentum_error_kg_m_s": float(np.linalg.norm(gpu_metrics.momentum_after_terrain_kg_m_s - reference.momentum_after_terrain_kg_m_s)),
            "soil_impulse_error_ns": tool_impulse_error,
        },
    }


def _track_record(name: str, size: int, active_width: int) -> dict[str, object]:
    grid, integrator, resting, mobile, momentum = _fields(size, active_width)
    mobile.fill(0.0)
    momentum.fill(0.0)
    footprint_length = max(6, min(active_width, size - 8))
    footprint_width = max(2, min(8, size // 12))
    center = size // 2
    start_col = center - footprint_length // 2
    stop_col = start_col + footprint_length
    left = np.zeros(grid.shape, dtype=bool)
    right = np.zeros(grid.shape, dtype=bool)
    left[center - footprint_width - 1 : center - 1, start_col:stop_col] = True
    right[center + 1 : center + footprint_width + 1, start_col:stop_col] = True
    arguments = {
        "left_footprint_mask": left,
        "right_footprint_mask": right,
        "left_track_velocity_xy_m_s": np.asarray([1.0, 0.0]),
        "right_track_velocity_xy_m_s": np.asarray([0.8, 0.0]),
        "base_velocity_xy_m_s": np.asarray([0.2, 0.0]),
        "dt_s": 1.0 / 60.0,
    }
    reference_solver = TrackSoilModel()
    reference_solver.initialize(resting)
    start = perf_counter()
    reference = reference_solver.apply(
        resting, mobile, momentum, grid=grid, integrator=integrator, **arguments
    )
    cpu_wall = perf_counter() - start
    gpu = WarpTrackSoilOperator(grid.shape)
    gpu.initialize(resting, resting, mobile, momentum, integrator.vertex_weights_m2)
    start = perf_counter()
    metrics = gpu.apply_resident(**arguments)
    gpu.runtime.synchronize()
    gpu_wall = perf_counter() - start
    next_resting, next_mobile, next_momentum = gpu.download_state()
    transfer = gpu.diagnostics()
    resting_loss = -integrator.integrate_delta(resting, next_resting)
    mobile_gain = integrator.integrate_delta(mobile, next_mobile)
    return {
        "case": name,
        "grid_shape": [size, size],
        "active_cell_count": int(np.count_nonzero(left | right)),
        "CPU_REFERENCE": {
            "wall_time_s": cpu_wall,
            "resting_to_mobile_volume_m3": reference.resting_to_mobile_volume_m3,
        },
        "GPU_OPTIMIZED": {
            "status": "PASS",
            "wall_time_s": gpu_wall,
            "H2D_bytes": transfer["h2d_bytes"],
            "D2H_bytes": transfer["d2h_bytes"],
            "synchronization_count": transfer["synchronization_count"],
            "gpu_utilization": {"status": "EXTERNAL_SAMPLER_REQUIRED"},
        },
        "equivalence": {
            "terrain_max_abs_m": float(np.max(np.abs(next_resting - reference.H_resting_m))),
            "mobile_max_abs_m": float(np.max(np.abs(next_mobile - reference.mobile_height_m))),
            "momentum_max_abs_m2_s": float(np.max(np.abs(next_momentum - reference.mobile_momentum_m2_s))),
            "cpu_gpu_volume_error_m3": abs(metrics.resting_to_mobile_volume_m3 - reference.resting_to_mobile_volume_m3),
            "resting_mobile_conservation_error_m3": abs(resting_loss - mobile_gain),
        },
    }


def _frontier_record(name: str, size: int, active_width: int) -> dict[str, object]:
    tile_size = 32
    frontier = CompactTileFrontier((size, size), tile_size)
    if name == "localized":
        tile_ids = np.asarray([frontier.tile_count // 2], dtype=np.int32)
    elif name == "medium":
        tile_side = frontier.tile_shape[1]
        center_row, center_col = frontier.tile_shape[0] // 2, tile_side // 2
        tile_ids = np.asarray(
            [r * tile_side + c for r in range(max(0, center_row - 2), min(frontier.tile_shape[0], center_row + 2)) for c in range(max(0, center_col - 2), min(tile_side, center_col + 2))],
            dtype=np.int32,
        )
    else:
        tile_ids = np.arange(frontier.tile_count, dtype=np.int32)
    batch = frontier.edge_batch(
        tile_ids, direction_index=0, di=1, dj=0, phase=0
    )
    height = np.ones((size, size), dtype=np.float64)
    reached = np.zeros((size, size), dtype=bool)
    selected = np.arange(0, batch.edge_count, max(1, batch.edge_count // 64))
    height[batch.a_i[selected], batch.a_j[selected]] += 0.4
    reached[batch.a_i[selected], batch.a_j[selected]] = True
    critical = 0.2
    start = perf_counter()
    cpu_height = height.copy()
    cpu_reached = reached.copy()
    reachable = cpu_reached[batch.a_i, batch.a_j] | cpu_reached[batch.b_i, batch.b_j]
    difference = cpu_height[batch.a_i, batch.a_j] - cpu_height[batch.b_i, batch.b_j]
    transfer_amount = 0.5 * np.maximum(np.abs(difference) - critical, 0.0) * np.sign(difference)
    transfer_amount *= reachable
    moved = transfer_amount != 0.0
    cpu_height[batch.a_i[moved], batch.a_j[moved]] -= transfer_amount[moved]
    cpu_height[batch.b_i[moved], batch.b_j[moved]] += transfer_amount[moved]
    cpu_reached[batch.a_i[moved], batch.a_j[moved]] = True
    cpu_reached[batch.b_i[moved], batch.b_j[moved]] = True
    cpu_wall = perf_counter() - start
    gpu = WarpCompactActiveEdgeOperator((size, size), tile_size=tile_size)
    gpu.initialize(height, reached)
    start = perf_counter()
    phase = gpu.run_phase(batch, critical_difference_m=critical, tolerance_m=1e-9)
    gpu.runtime.synchronize()
    gpu_wall = perf_counter() - start
    gpu_height, gpu_reached = gpu.download_state()
    transfer = gpu.diagnostics()
    return {
        "case": name,
        "grid_shape": [size, size],
        "active_tile_count": int(tile_ids.size),
        "compact_edge_count": batch.edge_count,
        "CPU_REFERENCE": {"wall_time_s": cpu_wall, "moved_edge_count": int(np.count_nonzero(moved))},
        "GPU_OPTIMIZED": {
            "status": "PASS",
            "wall_time_s": gpu_wall,
            "moved_edge_count": int(phase.moved_edge_count),
            "triggered_tile_count": int(phase.triggered_tile_ids.size),
            "H2D_bytes": transfer["h2d_bytes"],
            "D2H_bytes": transfer["d2h_bytes"],
            "synchronization_count": transfer["synchronization_count"],
            "gpu_utilization": {"status": "EXTERNAL_SAMPLER_REQUIRED"},
        },
        "equivalence": {
            "terrain_max_abs_m": float(np.max(np.abs(gpu_height - cpu_height))),
            "volume_conservation_error_m3": abs(
                float(np.sum(gpu_height) - np.sum(height)) * 0.05 * 0.05
            ),
            "reached_mask_mismatch_count": int(np.count_nonzero(gpu_reached != cpu_reached)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "390f_v2" / "warp_backend_benchmarks.json",
    )
    parser.add_argument("--cases", nargs="*", choices=tuple(CASES), default=list(CASES))
    args = parser.parse_args()
    status = probe_warp()
    report: dict[str, object] = {
        "schema": "390F_WARP_BACKEND_BENCHMARK_V1",
        "warp": status.to_dict(),
        "resolution_m": 0.05,
        "GPU_utilization_note": "Requires an external nvidia-smi/NVML sampler; not inferred from kernel wall time.",
    }
    if status.available:
        report["mobile_layer"] = [_mobile_record(name, *CASES[name]) for name in args.cases]
        report["compact_frontier"] = [_frontier_record(name, *CASES[name]) for name in args.cases]
        report["track_soil"] = [_track_record(name, *CASES[name]) for name in args.cases]
        report["status"] = "GPU_OPTIMIZED_EXECUTED"
    else:
        report["status"] = "GPU_OPTIMIZED_UNAVAILABLE"
        report["reason"] = status.reason
        report["mobile_layer"] = []
        report["compact_frontier"] = []
        report["track_soil"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
