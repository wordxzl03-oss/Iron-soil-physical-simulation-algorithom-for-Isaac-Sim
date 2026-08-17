#!/usr/bin/env python3
"""Benchmark CPU, resident Warp and physical large-avalanche paths at 0.05 m."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    LargeAvalancheTransitionConfig,
    LargeAvalancheTransitionController,
    MobileLayerSolver,
    WarpMobileLayerSolver,
)
from isaac_bulk_pipeline.bulk_state import (  # noqa: E402
    MaterialScenario,
    TerrainVolumeIntegrator,
)
from isaac_bulk_pipeline.performance import (  # noqa: E402
    TerrainAcceptancePath,
    TerrainAcceptanceStatus,
    TerrainHierarchyAcceptanceReport,
    TerrainPathMetrics,
    probe_warp,
)
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs" / "390f_v2" / "terrain_hierarchy_acceptance.json"
DT_S = 1.0 / 60.0
GPU_TERRAIN_LINF_TOLERANCE_M = 1.0e-3
GPU_TERRAIN_RMSE_TOLERANCE_M = 1.0e-5
GPU_VOLUME_TOLERANCE_M3 = 1.0e-8
GPU_MOMENTUM_CLOSURE_TOLERANCE_KG_M_S = 1.0e-6
GPU_SOIL_IMPULSE_ABSOLUTE_TOLERANCE_NS = 1.0e-5
GPU_SOIL_IMPULSE_RELATIVE_TOLERANCE = 1.0e-7


def material() -> MaterialScenario:
    return MaterialScenario(
        "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED",
        1370.0,
        29.8,
        800.0,
        float(np.tan(np.deg2rad(34.4))),
        38.0,
        30.0,
        0.35,
    )


def case_height(name: str, size: int) -> tuple[TerrainGrid, np.ndarray]:
    grid = TerrainGrid(size, size, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    rows, cols = np.indices(grid.shape)
    x = (cols - (size - 1) / 2.0) * grid.dx
    y = (rows - (size - 1) / 2.0) * grid.dy
    if name == "localized":
        height = 1.0 + 0.55 * np.exp(-(x * x + y * y) / (2.0 * 0.22**2))
    elif name == "medium":
        height = 2.0 + 2.2 * np.exp(
            -(x * x / (2.0 * 1.20**2) + y * y / (2.0 * 0.80**2))
        )
    elif name == "whole_pile":
        # Positive full-domain near-planar failure; its connected instability
        # is intentionally able to span the complete formal 701 grid.
        height = 40.0 - np.tan(np.deg2rad(43.0)) * (x - x.min())
        height += 0.01 * np.sin(0.4 * y)
    else:
        raise ValueError(name)
    return grid, np.asarray(height, dtype=np.float64)


def gpu_utilization_sample() -> float | None:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()[0]
        return float(output.strip())
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def closure_error(result) -> np.ndarray:
    return np.asarray(
        result.momentum_after_terrain_kg_m_s
        - result.momentum_before_terrain_kg_m_s
        - result.gravity_pressure_impulse_terrain_ns
        - result.basal_friction_impulse_terrain_ns
        - result.tool_impulse_on_mobile_terrain_ns
        - result.numerical_dissipative_impulse_terrain_ns,
        dtype=np.float64,
    )


def soil_impulse(result) -> np.ndarray:
    return np.asarray(
        result.gravity_pressure_impulse_terrain_ns
        + result.basal_friction_impulse_terrain_ns
        + result.tool_impulse_on_mobile_terrain_ns,
        dtype=np.float64,
    )


def run_case(
    name: str,
    size: int,
    *,
    steps: int,
    device: str,
) -> tuple[dict[str, object], bool]:
    grid, initial_resting = case_height(name, size)
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    scenario = material()
    zero_mobile = np.zeros(grid.shape, dtype=np.float64)
    zero_momentum = np.zeros(grid.shape + (2,), dtype=np.float64)
    transition_config = LargeAvalancheTransitionConfig(
        persistence_time_s=DT_S,
        sensitivity_case="nominal_uncalibrated_benchmark",
    )
    transition_controller = LargeAvalancheTransitionController(transition_config)
    transition_started = perf_counter()
    transition = transition_controller.observe_and_maybe_mobilize(
        initial_resting,
        zero_mobile,
        zero_momentum,
        scenario,
        grid,
        integrator,
        DT_S,
    )
    transition_wall_s = perf_counter() - transition_started
    if not transition.transitioned:
        raise RuntimeError(
            f"[TerrainHierarchyBenchmark] {name} did not enter large mobile path"
        )

    reference_solver = MobileLayerSolver()
    reference_mobile = np.asarray(transition.mobile_height_m)
    reference_momentum = np.asarray(transition.mobile_momentum_m2_s)
    cpu_results = []
    cpu_started = perf_counter()
    for _ in range(steps):
        reference_result = reference_solver.step(
            transition.H_resting_m,
            reference_mobile,
            reference_momentum,
            scenario,
            grid,
            integrator,
            DT_S,
        )
        cpu_results.append(reference_result)
        reference_mobile = np.asarray(reference_result.mobile_height_m)
        reference_momentum = np.asarray(reference_result.mobile_momentum_m2_s)
    cpu_wall_s = perf_counter() - cpu_started
    cpu_final = cpu_results[-1]
    reference_surface = np.asarray(transition.H_resting_m) + reference_mobile
    cpu_volume_error = (
        cpu_final.volume_before_m3
        - cpu_final.volume_after_m3
        - cpu_final.outflow_volume_m3
    )
    cpu_impulse = np.sum([soil_impulse(row) for row in cpu_results], axis=0)
    cpu_closure = np.sum([closure_error(row) for row in cpu_results], axis=0)

    cpu_pass = bool(
        abs(cpu_volume_error) <= 1.0e-9
        and np.linalg.norm(cpu_closure) <= 1.0e-6
    )
    cpu_metrics = TerrainPathMetrics(
        path=TerrainAcceptancePath.CPU_REFERENCE,
        status=(
            TerrainAcceptanceStatus.PASS
            if cpu_pass
            else TerrainAcceptanceStatus.FAIL
        ),
        case_name=name,
        resolution_m=0.05,
        simulated_time_s=steps * DT_S,
        wall_time_s=cpu_wall_s,
        final_terrain_linf_error_m=0.0,
        final_terrain_rmse_m=0.0,
        volume_balance_error_m3=cpu_volume_error,
        momentum_balance_error_kg_m_s=cpu_closure,
        soil_impulse_terrain_ns=cpu_impulse,
        soil_impulse_error_ns=np.zeros(3),
        backend_identity="CPU_REFERENCE_MOBILE_LAYER",
        failure_reason=None if cpu_pass else "CPU_CONSERVATION_CHECK_FAILED",
    )

    transition_total_before = integrator.integrate(initial_resting)
    transition_total_after = integrator.integrate(
        transition.H_resting_m + transition.mobile_height_m
    )
    large_volume_error = transition_total_after - transition_total_before
    large_pass = bool(
        transition.transitioned
        and abs(large_volume_error) <= 1.0e-9
        and np.linalg.norm(transition.momentum_closure_error_kg_m_s) <= 1.0e-7
    )
    large_failure_reason = None
    if not transition.transitioned:
        large_failure_reason = "PHYSICAL_TRANSITION_NOT_TRIGGERED"
    elif abs(large_volume_error) > 1.0e-9:
        large_failure_reason = "RESTING_MOBILE_VOLUME_CONSERVATION_FAILED"
    elif np.linalg.norm(transition.momentum_closure_error_kg_m_s) > 1.0e-7:
        large_failure_reason = "MOMENTUM_INITIALIZATION_CLOSURE_FAILED"
    large_metrics = TerrainPathMetrics(
        path=TerrainAcceptancePath.LARGE_AVALANCHE_MOBILE_PATH,
        status=(
            TerrainAcceptanceStatus.PASS
            if large_pass
            else TerrainAcceptanceStatus.FAIL
        ),
        case_name=name,
        resolution_m=0.05,
        simulated_time_s=DT_S,
        wall_time_s=transition_wall_s,
        final_terrain_linf_error_m=0.0,
        final_terrain_rmse_m=0.0,
        volume_balance_error_m3=large_volume_error,
        momentum_balance_error_kg_m_s=np.r_[
            transition.momentum_closure_error_kg_m_s, 0.0
        ],
        soil_impulse_terrain_ns=np.r_[
            transition.gravity_initiation_impulse_kg_m_s, 0.0
        ],
        soil_impulse_error_ns=np.zeros(3),
        backend_identity="PHYSICAL_RESTING_TO_MOBILE_TRANSITION",
        failure_reason=large_failure_reason,
    )

    warp_status = probe_warp(device)
    gpu_details: dict[str, object]
    if not warp_status.available:
        gpu_metrics = TerrainPathMetrics(
            path=TerrainAcceptancePath.GPU_OPTIMIZED,
            status=TerrainAcceptanceStatus.UNAVAILABLE,
            case_name=name,
            resolution_m=0.05,
            simulated_time_s=0.0,
            wall_time_s=0.0,
            final_terrain_linf_error_m=0.0,
            final_terrain_rmse_m=0.0,
            volume_balance_error_m3=0.0,
            momentum_balance_error_kg_m_s=np.zeros(3),
            soil_impulse_terrain_ns=np.zeros(3),
            soil_impulse_error_ns=np.zeros(3),
            backend_identity="WARP_UNAVAILABLE",
            failure_reason=warp_status.reason or "WARP_UNAVAILABLE",
        )
        gpu_details = {"warp": warp_status.to_dict()}
    else:
        utilization: list[float] = []
        gpu_started = perf_counter()
        gpu_solver = WarpMobileLayerSolver(device=device)
        gpu_solver.initialize_resident(
            transition.H_resting_m,
            transition.mobile_height_m,
            transition.mobile_momentum_m2_s,
            scenario,
            grid,
            integrator,
        )
        gpu_steps = []
        for _ in range(steps):
            gpu_steps.append(gpu_solver.step_resident(DT_S))
            sample = gpu_utilization_sample()
            if sample is not None:
                utilization.append(sample)
        gpu_final = gpu_solver.download_result()
        gpu_wall_s = perf_counter() - gpu_started
        gpu_surface = np.asarray(transition.H_resting_m) + np.asarray(
            gpu_final.mobile_height_m
        )
        terrain_difference = gpu_surface - reference_surface
        gpu_volume_error = (
            gpu_final.volume_before_m3
            - gpu_final.volume_after_m3
            - gpu_final.outflow_volume_m3
        )
        gpu_impulse = np.sum([soil_impulse(row) for row in gpu_steps], axis=0)
        gpu_closure = np.sum([closure_error(row) for row in gpu_steps], axis=0)
        impulse_error = gpu_impulse - cpu_impulse
        diagnostics = gpu_solver.diagnostics()
        linf = float(np.max(np.abs(terrain_difference), initial=0.0))
        rmse = float(np.sqrt(np.mean(terrain_difference * terrain_difference)))
        impulse_tolerance = max(
            GPU_SOIL_IMPULSE_ABSOLUTE_TOLERANCE_NS,
            GPU_SOIL_IMPULSE_RELATIVE_TOLERANCE * np.linalg.norm(cpu_impulse),
        )
        gpu_pass = bool(
            linf <= GPU_TERRAIN_LINF_TOLERANCE_M
            and rmse <= GPU_TERRAIN_RMSE_TOLERANCE_M
            and abs(gpu_volume_error) <= GPU_VOLUME_TOLERANCE_M3
            and np.linalg.norm(gpu_closure)
            <= GPU_MOMENTUM_CLOSURE_TOLERANCE_KG_M_S
            and np.linalg.norm(impulse_error) <= impulse_tolerance
        )
        gpu_metrics = TerrainPathMetrics(
            path=TerrainAcceptancePath.GPU_OPTIMIZED,
            status=(
                TerrainAcceptanceStatus.PASS
                if gpu_pass
                else TerrainAcceptanceStatus.FAIL
            ),
            case_name=name,
            resolution_m=0.05,
            simulated_time_s=steps * DT_S,
            wall_time_s=gpu_wall_s,
            final_terrain_linf_error_m=linf,
            final_terrain_rmse_m=rmse,
            volume_balance_error_m3=gpu_volume_error,
            momentum_balance_error_kg_m_s=gpu_closure,
            soil_impulse_terrain_ns=gpu_impulse,
            soil_impulse_error_ns=impulse_error,
            gpu_utilization_percent_mean=(
                float(np.mean(utilization)) if utilization else 0.0
            ),
            h2d_bytes=int(diagnostics["h2d_bytes"]),
            d2h_bytes=int(diagnostics["d2h_bytes"]),
            synchronization_count=int(diagnostics["synchronization_count"]),
            arrays_resident_on_device=True,
            backend_identity=str(diagnostics["backend"]),
            failure_reason=None if gpu_pass else "GPU_REFERENCE_EQUIVALENCE_FAILED",
        )
        gpu_details = diagnostics

    report = TerrainHierarchyAcceptanceReport.build(
        name, [cpu_metrics, gpu_metrics, large_metrics]
    )
    sensitivity = []
    for sensitivity_config in transition_config.sensitivity_ensemble(0.20):
        controller = LargeAvalancheTransitionController(
            LargeAvalancheTransitionConfig(
                **{
                    **sensitivity_config.__dict__,
                    "persistence_time_s": DT_S,
                }
            )
        )
        value = controller.observe_and_maybe_mobilize(
            initial_resting,
            zero_mobile,
            zero_momentum,
            scenario,
            grid,
            integrator,
            DT_S,
        )
        sensitivity.append(
            {
                "case": sensitivity_config.sensitivity_case,
                "transitioned": value.transitioned,
                "mobilized_volume_m3": value.transferred_volume_m3,
                "gravity_initiation_impulse_kg_m_s": (
                    value.gravity_initiation_impulse_kg_m_s.tolist()
                ),
                "largest_connected_area_m2": (
                    value.diagnostics.largest_connected_area_m2
                ),
            }
        )
    record = report.to_dict()
    record.update(
        {
            "grid_shape_yx": list(grid.shape),
            "comparison_horizon": "FINITE_PHYSICAL_TIME_MOBILE_EVOLUTION",
            "comparison_steps": steps,
            "large_avalanche_diagnostics": {
                key: value
                for key, value in transition.diagnostics.__dict__.items()
                if key not in {"largest_connected_mask", "slope_angle_deg"}
            },
            "large_avalanche_transfer": {
                "volume_m3": transition.transferred_volume_m3,
                "gravity_initiation_impulse_kg_m_s": (
                    transition.gravity_initiation_impulse_kg_m_s.tolist()
                ),
            },
            "sensitivity": sensitivity,
            "gpu_backend_diagnostics": gpu_details,
            "gpu_equivalence_tolerances": {
                "terrain_linf_m": GPU_TERRAIN_LINF_TOLERANCE_M,
                "terrain_rmse_m": GPU_TERRAIN_RMSE_TOLERANCE_M,
                "volume_balance_m3": GPU_VOLUME_TOLERANCE_M3,
                "momentum_closure_kg_m_s": (
                    GPU_MOMENTUM_CLOSURE_TOLERANCE_KG_M_S
                ),
                "soil_impulse_absolute_ns": (
                    GPU_SOIL_IMPULSE_ABSOLUTE_TOLERANCE_NS
                ),
                "soil_impulse_relative": GPU_SOIL_IMPULSE_RELATIVE_TOLERANCE,
            },
        }
    )
    return record, report.overall_status == "PASS"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    cases = []
    all_pass = True
    for name, size in (("localized", 129), ("medium", 351), ("whole_pile", 701)):
        record, passed = run_case(
            name, size, steps=args.steps, device=args.device
        )
        cases.append(record)
        all_pass &= passed
    output = {
        "schema": "terrain-hierarchy-benchmark/v1",
        "formal_resolution_m": 0.05,
        "cases": cases,
        "acceptance": "PASS" if all_pass else "NOT_PASSED",
        "claim_boundary": (
            "Numerical equivalence and reduced-order sensitivity only; not "
            "site-calibrated iron-ore predictive validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "acceptance": output["acceptance"],
                "output": str(args.output),
                "warp": probe_warp(args.device).to_dict(),
            }
        )
    )
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
