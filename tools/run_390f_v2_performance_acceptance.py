"""Deterministic 0.05 m reference/optimized computability benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    MobileLayerSolver,
    OptimizedMobileLayerSolver,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.config import SolverConfig  # noqa: E402
from isaac_bulk_pipeline.performance import PerformanceProfiler  # noqa: E402
from isaac_bulk_pipeline.solvers import (  # noqa: E402
    EventDrivenMinimumSlopeAdapter,
    MinimumSlopeAdapter,
    SparseTileFrontierMinimumSlopeAdapter,
)
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


OUTPUT = ROOT / "outputs" / "390f_v2"


def _command(args: list[str]) -> str | None:
    try:
        return subprocess.run(
            args, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _hardware_manifest() -> dict[str, object]:
    gpu_query = _command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu: dict[str, object] = {"status": "UNAVAILABLE"}
    if gpu_query:
        name, memory, driver = (item.strip() for item in gpu_query.splitlines()[0].split(","))
        gpu = {
            "status": "AVAILABLE",
            "name": name,
            "vram_total_mb": float(memory),
            "driver_version": driver,
        }
    cuda = None
    try:
        import torch

        cuda = {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except ImportError:
        cuda = {"cuda_available": False, "reason": "torch unavailable"}
    memory_kb: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory_kb[key] = int(value.strip().split()[0])
    except OSError:
        pass
    isaac_version = None
    for candidate in (
        Path("/home/eric/isaacsim/VERSION"),
        Path("/home/eric/isaacsim/VERSION.txt"),
    ):
        if candidate.is_file():
            isaac_version = candidate.read_text().strip()
            break
    return {
        "cpu": {
            "model": platform.processor() or _command(["uname", "-p"]),
            "logical_core_count": os.cpu_count(),
            "architecture": platform.machine(),
        },
        "ram": {key.lower() + "_kb": value for key, value in memory_kb.items()},
        "gpu": gpu,
        "cuda": cuda,
        "isaac_sim_version": isaac_version or "4.5.0_INSTALLATION_PATH_DECLARED",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _statistics(values_ms: list[float]) -> dict[str, float | int]:
    data = np.asarray(values_ms, dtype=np.float64)
    return {
        "call_count": int(data.size),
        "mean_ms": float(np.mean(data)),
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.percentile(data, 95.0)),
        "p99_ms": float(np.percentile(data, 99.0)),
        "maximum_ms": float(np.max(data)),
        "total_wall_time_ms": float(np.sum(data)),
    }


def _mobile_case(grid: TerrainGrid):
    rows, columns = np.indices(grid.shape)
    resting = 1.0 + 0.001 * columns
    mobile = np.zeros(grid.shape, dtype=np.float64)
    local = np.exp(-((rows - 350.0) ** 2 + (columns - 350.0) ** 2) / 180.0)
    mobile[local > 0.01] = 0.08 * local[local > 0.01]
    momentum = np.zeros(grid.shape + (2,), dtype=np.float64)
    momentum[..., 0] = mobile * 0.4
    return resting, mobile, momentum


def _run_mobile(solver, grid, integrator, material, steps=5):
    resting, mobile, momentum = _mobile_case(grid)
    samples: list[float] = []
    result = None
    for _ in range(steps):
        start = perf_counter()
        result = solver.step(
            resting, mobile, momentum, material, grid, integrator, 1.0 / 60.0
        )
        samples.append((perf_counter() - start) * 1_000.0)
        mobile = np.asarray(result.mobile_height_m)
        momentum = np.asarray(result.mobile_momentum_m2_s)
    assert result is not None
    return samples, result


def _relative_error(reference: float, optimized: float) -> float:
    return abs(optimized - reference) / max(abs(reference), 1.0e-12)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    material = MaterialScenario(
        "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED",
        1370.0,
        29.8,
        800.0,
        float(np.tan(np.deg2rad(34.4))),
        38.0,
        30.0,
        0.35,
    )
    reference_times, reference_result = _run_mobile(
        MobileLayerSolver(), grid, integrator, material
    )
    tile_results: dict[str, dict[str, object]] = {}
    optimized_runs = {}
    for tile_size in (16, 32, 64):
        solver = OptimizedMobileLayerSolver(tile_size=tile_size)
        values, result = _run_mobile(solver, grid, integrator, material)
        tile_results[str(tile_size)] = {
            **_statistics(values),
            "last_active_ratio": solver.last_active_snapshot.active_ratio,
            "last_kernel_time_ms": solver.last_kernel_time_ms,
        }
        optimized_runs[tile_size] = (values, result, solver)
    selected_tile = min(
        optimized_runs,
        key=lambda size: np.median(optimized_runs[size][0]),
    )
    optimized_times, optimized_result, optimized_solver = optimized_runs[selected_tile]

    baseline = np.ones(grid.shape)
    disturbed = baseline.copy()
    disturbed[350, 350] += 0.30
    slope_config = SolverConfig(
        critical_angle_deg=30.0,
        max_iterations=1000,
        tolerance=1.0e-8,
        boundary_condition="closed",
        conservation_tolerance_m3=1.0e-8,
    )
    slope_reference = MinimumSlopeAdapter(grid, slope_config)
    start = perf_counter()
    reference_slope_result = slope_reference.solve(disturbed)
    reference_slope_ms = (perf_counter() - start) * 1_000.0
    slope_bbox_baseline = EventDrivenMinimumSlopeAdapter(
        grid, slope_config, tile_size=selected_tile
    )
    slope_bbox_baseline.set_reference_height(baseline)
    start = perf_counter()
    bbox_slope_result = slope_bbox_baseline.solve(disturbed)
    bbox_slope_ms = (perf_counter() - start) * 1_000.0
    slope_optimized = SparseTileFrontierMinimumSlopeAdapter(
        grid, slope_config, tile_size=selected_tile
    )
    slope_optimized.set_reference_height(baseline)
    start = perf_counter()
    optimized_slope_result = slope_optimized.solve(disturbed)
    optimized_slope_ms = (perf_counter() - start) * 1_000.0

    terrain_error = float(
        np.sum(np.abs(reference_slope_result.heightmap_stable - optimized_slope_result.heightmap_stable))
        / max(np.sum(np.abs(reference_slope_result.heightmap_stable - baseline)), 1.0e-12)
    )
    impulse_reference = np.linalg.norm(reference_result.momentum_after_terrain_kg_m_s)
    impulse_optimized = np.linalg.norm(optimized_result.momentum_after_terrain_kg_m_s)
    mass_error = abs(
        optimized_result.volume_before_m3
        - optimized_result.volume_after_m3
        - optimized_result.outflow_volume_m3
    )
    v1_profile_path = ROOT / "outputs" / "integrated_390f_v1" / "case_C" / "runtime_profile.json"
    v1_profile = json.loads(v1_profile_path.read_text()) if v1_profile_path.is_file() else None
    v1_mobile_ms = (
        v1_profile["module_timings"]["mobile_layer"]["mean_ms"] if v1_profile else None
    )
    algorithmic_speedup = (
        v1_mobile_ms / np.mean(optimized_times) if v1_mobile_ms is not None else None
    )
    hardware = _hardware_manifest()
    (OUTPUT / "runtime_hardware_manifest.json").write_text(
        json.dumps(hardware, indent=2) + "\n", encoding="utf-8"
    )

    measured_modules = {
        "mobile_layer_reference": _statistics(reference_times),
        "mobile_layer_optimized": _statistics(optimized_times),
        "minislope_reference": _statistics([reference_slope_ms]),
        "minislope_reachable_bbox_cpu_baseline": _statistics([bbox_slope_ms]),
        "minislope_optimized": _statistics([optimized_slope_ms]),
    }
    total_measured = sum(item["total_wall_time_ms"] for item in measured_modules.values())
    for item in measured_modules.values():
        item["percentage_of_total"] = 100.0 * item["total_wall_time_ms"] / total_measured
    unmeasured_integrated_modules = {
        name: {"status": "REQUIRES_INTEGRATED_ISAAC_V2_RUN"}
        for name in (
            "isaac_physx_step",
            "tool_state",
            "bucket_terrain_intersection",
            "failure_zone",
            "failure_rasterization",
            "resting_to_mobile_activation",
            "bucket_intake",
            "internal_fill",
            "retention_spill",
            "airborne_deposition",
            "track_soil",
            "terrain_visual_update",
            "terrain_contact_update",
            "logging",
            "rendering",
            "usd_fabric_synchronization",
        )
    }
    breakdown = {
        "schema": "390F_V2_PERFORMANCE_BREAKDOWN_V1",
        "grid": {"shape": [701, 701], "spacing_m": 0.05},
        "measured_modules": measured_modules,
        "integrated_modules": unmeasured_integrated_modules,
        "tile_size_benchmark": tile_results,
        "selected_tile_size": selected_tile,
        "process": {
            "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "gpu": {
            "solver_backend": "CPU_ACTIVE_DOMAIN",
            "gpu_solver_status": "NOT_YET_IMPLEMENTED",
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "transfer_count": 0,
            "synchronization_count": 0,
            "note": "No fake GPU claim: this Stage-A backend executes on CPU.",
        },
    }
    (OUTPUT / "performance_breakdown.json").write_text(
        json.dumps(breakdown, indent=2) + "\n", encoding="utf-8"
    )

    equivalence = {
        "terrain_disturbed_normalized_l1": terrain_error,
        "integrated_mobile_momentum_relative_error": _relative_error(
            impulse_reference, impulse_optimized
        ),
        "mobile_volume_relative_error": _relative_error(
            reference_result.volume_after_m3, optimized_result.volume_after_m3
        ),
        "mass_balance_error_m3": mass_error,
        "payload_error": {"status": "REQUIRES_INTEGRATED_RUN"},
        "payload_com_error_m": {"status": "REQUIRES_INTEGRATED_RUN"},
        "bucket_tip_trajectory_rmse_m": {"status": "REQUIRES_INTEGRATED_RUN"},
        "joint_trajectory_error": {"status": "REQUIRES_INTEGRATED_RUN"},
        "track_rut_volume_error": {"status": "REQUIRES_TRACK_SOIL_BACKEND"},
    }
    stage_a_pass = bool(
        algorithmic_speedup is not None
        and algorithmic_speedup >= 5.0
        and terrain_error <= 0.02
        and equivalence["integrated_mobile_momentum_relative_error"] <= 0.02
        and equivalence["mobile_volume_relative_error"] <= 0.02
        and mass_error <= 1.0e-9
    )
    acceptance = {
        "schema": "390F_V2_PERFORMANCE_ACCEPTANCE_V1",
        "COMPUTATIONAL_FEASIBILITY_STATUS": "DEGRADED",
        "FULL_V2_ACCEPTANCE": False,
        "hardware": hardware,
        "reference_backend_runtime": measured_modules["mobile_layer_reference"],
        "optimized_backend_runtime": measured_modules["mobile_layer_optimized"],
        "speedup_vs_v1_mobile_layer": algorithmic_speedup,
        "stage_a_algorithmic_optimization": "PASS" if stage_a_pass else "FAIL",
        "stage_b_integrated_headless": "NOT_EXECUTED",
        "RTF_headless": None,
        "RTF_rendered": None,
        "minislope_reference_ms": reference_slope_ms,
        "minislope_reachable_bbox_cpu_baseline_ms": bbox_slope_ms,
        "minislope_optimized_ms": optimized_slope_ms,
        "minislope_sparse_701_benchmark": {
            "status": "AVAILABLE"
            if (OUTPUT / "minislope_sparse_701_benchmark.json").is_file()
            else "NOT_EXECUTED",
            "path": str(OUTPUT / "minislope_sparse_701_benchmark.json"),
        },
        "reference_vs_optimized": equivalence,
        "selected_tile_size": selected_tile,
        "peak_ram_mb": breakdown["process"]["peak_ram_mb"],
        "gpu_memory_peak_mb": None,
        "H2D_bytes": 0,
        "D2H_bytes": 0,
        "performance_regression_status": "BASELINE_CREATED",
        "real_time_status": "NOT_REAL_TIME_INTEGRATED_RTF_NOT_MEASURED",
        "blocking_reasons": [
            "Integrated three-cycle optimized Isaac run has not yet been executed.",
            "Track-soil, dirty contact chunks and GPU-resident backend are not yet accepted.",
        ],
    }
    (OUTPUT / "performance_acceptance.json").write_text(
        json.dumps(acceptance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": acceptance["COMPUTATIONAL_FEASIBILITY_STATUS"],
        "stage_a": acceptance["stage_a_algorithmic_optimization"],
        "speedup_vs_v1_mobile": algorithmic_speedup,
        "output": str(OUTPUT),
    }))


if __name__ == "__main__":
    main()
