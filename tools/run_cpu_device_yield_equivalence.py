#!/usr/bin/env python3
"""Explicit CPU/DEVICE cohesive Y_start equivalence for the stale 45° fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# Isaac/Kit normally exposes the bundled Warp extension while starting the
# application.  This focused headless audit intentionally does not create a
# SimulationApp, so expose the exact same shipped Python package explicitly.
ISAAC_ROOT = Path("/home/eric/isaacsim")
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    sys.path.insert(0, str(warp_root))

from isaac_bulk_pipeline.bulk_interaction.large_avalanche import (  # noqa: E402
    LargeAvalancheTransitionConfig,
    LargeAvalancheTransitionController,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.runtime.bulk_state_authority import DeviceBulkState  # noqa: E402
from isaac_bulk_pipeline.runtime.gpu_large_avalanche import DeviceLargeAvalancheBridge  # noqa: E402
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


def main() -> None:
    grid = TerrainGrid(21, 21, 0.05, 0.05, 0.0, 0.0, "/World/Test")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    x = np.arange(grid.nx) * grid.dx
    resting = np.repeat((2.0 - np.tan(np.deg2rad(45.0)) * x)[None, :], grid.ny, axis=0)
    material = MaterialScenario(
        name="cohesive_semantic_equivalence_fixture",
        assumed_bulk_density_kg_m3=2200.0,
        internal_friction_angle_deg=34.0,
        cohesion_proxy_pa=1500.0,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )
    config = LargeAvalancheTransitionConfig.from_mapping({
        "minimum_connected_cells": 16,
        "minimum_connected_area_m2": 0.02,
        "minimum_mobilizable_volume_m3": 0.001,
        "persistence_time_s": 0.01,
        "mobilization_depth_m": 0.05,
        "gravity_velocity_length_m": 0.20,
        "velocity_efficiency": 0.5,
    })
    zeros = np.zeros(grid.shape)
    cpu = LargeAvalancheTransitionController(config).diagnose(
        resting, zeros, np.zeros(grid.shape + (2,)), material, grid, integrator
    )
    state = DeviceBulkState(grid, resting, device="cuda:0", tile_size=8)
    device = DeviceLargeAvalancheBridge(
        state, material, grid, integrator, config
    ).observe_and_maybe_mobilize(0.01)
    slope = np.deg2rad(45.0)
    depth = 0.05
    drive = material.assumed_bulk_density_kg_m3 * 9.81 * depth * np.sin(slope)
    normal = material.assumed_bulk_density_kg_m3 * 9.81 * depth * np.cos(slope)
    resistance = material.cohesion_proxy_pa + normal * np.tan(
        np.deg2rad(material.start_angle_deg)
    )
    report = {
        "schema": "CPU_DEVICE_COHESIVE_YIELD_EQUIVALENCE/v1",
        "slope_deg": 45.0,
        "cohesion_pa": 1500.0,
        "mobilizable_depth_m": depth,
        "rho_kg_m3": material.assumed_bulk_density_kg_m3,
        "phi_start_deg": material.start_angle_deg,
        "phi_stop_deg": material.stop_angle_deg,
        "tau_drive_pa": float(drive),
        "tau_resist_start_pa": float(resistance),
        "Y_start_margin_pa": float(drive - resistance),
        "analytical_classification": "COHESIVE_STABLE",
        "CPU": {
            "unstable_cells": cpu.unstable_cell_count,
            "classification": cpu.classification,
        },
        "DEVICE": {
            "unstable_cells": device.unstable_cell_count,
            "classification": device.classification,
        },
        "CPU_DEVICE_YIELD_EQUIVALENCE": (
            "PASS" if cpu.unstable_cell_count == device.unstable_cell_count == 0
            and cpu.classification == device.classification else "FAIL"
        ),
        "old_fixture_classification": "STALE_TEST_FIXTURE",
    }
    out = ROOT / "outputs/mobile_large_avalanche_causal_audit/cpu_device_yield_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["CPU_DEVICE_YIELD_EQUIVALENCE"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
