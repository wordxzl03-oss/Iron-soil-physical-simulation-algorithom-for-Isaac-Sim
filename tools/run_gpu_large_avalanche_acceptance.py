#!/usr/bin/env python3
"""CPU-reference equivalence for device LargeAvalanche transition."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

_isaac_root = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for _warp_root in sorted(_isaac_root.glob("extscache/omni.warp.core-*")):
    if (_warp_root / "warp").is_dir():
        sys.path.insert(0, str(_warp_root))
        break

from isaac_bulk_pipeline.bulk_interaction.large_avalanche import (
    LargeAvalancheTransitionConfig,
    LargeAvalancheTransitionController,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.runtime import DeviceBulkState, DeviceLargeAvalancheBridge
from isaac_bulk_pipeline.terrain import TerrainGrid


OUTPUT = ROOT / "outputs/390f_v2/gpu_large_avalanche_acceptance.json"


def _case(name: str, size: int) -> dict[str, object]:
    grid = TerrainGrid(size, size, 0.05, 0.05, 0.0, 0.0, f"/Terrain/{name}")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    material = MaterialScenario(
        name, 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35
    )
    config = LargeAvalancheTransitionConfig(persistence_time_s=1.0 / 60.0)
    rows, cols = np.indices(grid.shape, dtype=np.float64)
    # One connected above-start plane.  This is a whole-domain stress case at
    # 701 and tests that device preprocessing is not a cropped ROI.
    resting = 1.0 + np.tan(np.deg2rad(42.0)) * cols * grid.dx
    mobile = np.zeros(grid.shape, dtype=np.float64)
    momentum = np.zeros(grid.shape + (2,), dtype=np.float64)

    reference = LargeAvalancheTransitionController(config)
    start = perf_counter()
    cpu = reference.observe_and_maybe_mobilize(
        resting,
        mobile,
        momentum,
        material,
        grid,
        integrator,
        1.0 / 60.0,
    )
    cpu_ms = (perf_counter() - start) * 1_000.0

    state = DeviceBulkState(grid, resting, tile_size=64)
    state.begin_physics_step()
    bridge = DeviceLargeAvalancheBridge(
        state, material, grid, integrator, config
    )
    start = perf_counter()
    gpu = bridge.observe_and_maybe_mobilize(1.0 / 60.0)
    gpu_ms = (perf_counter() - start) * 1_000.0
    transfer = state.transfer_snapshot().to_dict()
    state.assert_normal_step_transfer_budget()
    cpu_second = reference.observe_and_maybe_mobilize(
        cpu.H_resting_m,
        cpu.mobile_height_m,
        cpu.mobile_momentum_m2_s,
        material,
        grid,
        integrator,
        1.0 / 60.0,
    )
    state.begin_physics_step()
    gpu_second = bridge.observe_and_maybe_mobilize(1.0 / 60.0)
    second_transfer = state.transfer_snapshot().to_dict()
    state.assert_normal_step_transfer_budget()
    view = state.explicit_host_view(source="acceptance")
    errors = {
        "resting_linf_m": float(np.max(np.abs(view.H_resting_m - cpu.H_resting_m))),
        "mobile_linf_m": float(np.max(np.abs(view.H_mobile_m - cpu.mobile_height_m))),
        "momentum_linf_m2_s": float(
            np.max(np.abs(view.mobile_momentum_m2_s - cpu.mobile_momentum_m2_s))
        ),
        "transferred_volume_m3": float(
            gpu.transferred_volume_m3 - cpu.transferred_volume_m3
        ),
        "initiation_impulse_linf_kg_m_s": float(
            np.max(
                np.abs(
                    gpu.gravity_initiation_impulse_kg_m_s
                    - cpu.gravity_initiation_impulse_kg_m_s
                )
            )
        ),
        "largest_connected_cell_count": int(
            gpu.largest_connected_cell_count
            - cpu.diagnostics.largest_connected_cell_count
        ),
        "connected_region_count": int(
            gpu.connected_region_count - cpu.diagnostics.connected_region_count
        ),
    }
    impulse_scale = max(
        float(np.max(np.abs(cpu.gravity_initiation_impulse_kg_m_s))), 1.0e-12
    )
    errors["initiation_impulse_relative_linf"] = float(
        errors["initiation_impulse_linf_kg_m_s"] / impulse_scale
    )
    passed = bool(
        gpu.classification == cpu.diagnostics.classification
        and gpu.transitioned == cpu.transitioned
        and abs(errors["resting_linf_m"]) <= 2.0e-9
        and abs(errors["mobile_linf_m"]) <= 2.0e-9
        and abs(errors["momentum_linf_m2_s"]) <= 2.0e-9
        and abs(errors["transferred_volume_m3"]) <= 2.0e-8
        and abs(errors["initiation_impulse_relative_linf"]) <= 3.0e-8
        and errors["largest_connected_cell_count"] == 0
        and errors["connected_region_count"] == 0
        and cpu_second.diagnostics.classification
        == reference.LOCAL_STATIC_INSTABILITY
        and gpu_second.classification == cpu_second.diagnostics.classification
        and not cpu_second.transitioned
        and not gpu_second.transitioned
        and cpu_second.diagnostics.largest_connected_mobilizable_volume_m3 == 0.0
        and gpu_second.largest_connected_mobilizable_volume_m3 == 0.0
        and transfer["full_field_h2d_count"] == 0
        and transfer["full_field_d2h_count"] == 0
        and second_transfer["full_field_h2d_count"] == 0
        and second_transfer["full_field_d2h_count"] == 0
    )
    return {
        "case": name,
        "shape": [size, size],
        "resolution_m": 0.05,
        "status": "PASS" if passed else "FAIL",
        "classification": gpu.classification,
        "connected_cells": gpu.largest_connected_cell_count,
        "connected_regions": gpu.connected_region_count,
        "transferred_volume_m3": gpu.transferred_volume_m3,
        "union_iterations": gpu.union_iterations,
        "latched_second_observation": {
            "cpu_classification": cpu_second.diagnostics.classification,
            "gpu_classification": gpu_second.classification,
            "cpu_mobilizable_volume_m3": cpu_second.diagnostics.largest_connected_mobilizable_volume_m3,
            "gpu_mobilizable_volume_m3": gpu_second.largest_connected_mobilizable_volume_m3,
            "cpu_transitioned": cpu_second.transitioned,
            "gpu_transitioned": gpu_second.transitioned,
            "normal_physics_transfer": second_transfer,
        },
        "cpu_wall_time_ms": cpu_ms,
        "gpu_wall_time_ms": gpu_ms,
        "errors": errors,
        "normal_physics_transfer": transfer,
    }


def main() -> None:
    cases = [
        _case("localized_65", 65),
        _case("medium_257", 257),
        _case("whole_701", 701),
    ]
    report = {
        "schema": "390F_GPU_LARGE_AVALANCHE/v1",
        "status": "PASS" if all(x["status"] == "PASS" for x in cases) else "FAIL",
        "parameter_basis": "LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED",
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT)}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
