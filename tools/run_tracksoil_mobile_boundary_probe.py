#!/usr/bin/env python3
"""Narrow DEVICE probe for the P0-2A boundary control-volume diagnosis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaacsim import SimulationApp


APP = SimulationApp({"headless": True})

from isaac_bulk_pipeline.bulk_interaction import WarpTrackSoilOperator
from isaac_bulk_pipeline.bulk_interaction.warp_mobile_v2 import (
    WarpProductionMobileV2Solver,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.experimental.mobile_v2_reference import MobileV2Config
from isaac_bulk_pipeline.runtime import DeviceBulkState
from isaac_bulk_pipeline.terrain import TerrainGrid


def material() -> MaterialScenario:
    return MaterialScenario("boundary_probe", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)


def main() -> None:
    grid = TerrainGrid(16, 16, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    mat = material()
    resting = np.ones(grid.shape)

    track_state = DeviceBulkState(grid, resting, tile_size=4)
    track = WarpTrackSoilOperator(grid.shape, runtime=track_state.runtime)
    track.bind_device_state(track_state)
    mask = np.zeros(grid.shape, dtype=bool)
    mask[5:11, 0:3] = True
    empty = np.zeros(grid.shape, dtype=bool)
    track_before = track_state.reservoir_reduction(mat.assumed_bulk_density_kg_m3)
    track_step = track.apply_resident(
        left_footprint_mask=mask,
        right_footprint_mask=empty,
        left_track_velocity_xy_m_s=np.asarray([0.5, 0.0]),
        right_track_velocity_xy_m_s=np.zeros(2),
        base_velocity_xy_m_s=np.zeros(2),
        dt_s=1.0 / 30.0,
    )
    track_after = track_state.reservoir_reduction(mat.assumed_bulk_density_kg_m3)
    track_residual = (
        track_after["resting_volume_m3"]
        + track_after["mobile_volume_m3"]
        - track_before["resting_volume_m3"]
        - track_before["mobile_volume_m3"]
    )

    mobile_state = DeviceBulkState(grid, resting, tile_size=4)
    h0 = np.zeros(grid.shape)
    q0 = np.zeros(grid.shape + (2,))
    h0[6:10, 0] = 0.04
    q0[6:10, 0, 0] = 0.04 * 0.4
    mobile_state.apply_host_patch("mobile", (0, 16, 0, 16), h0, reason="probe_setup")
    mobile_state.apply_host_patch("momentum_x", (0, 16, 0, 16), q0[..., 0], reason="probe_setup")
    solver = WarpProductionMobileV2Solver(
        runtime=mobile_state.runtime,
        config=MobileV2Config(dx_m=grid.dx, dy_m=grid.dy),
    )
    solver.bind_device_state(mobile_state, mat, grid, integrator)
    mobile_before = mobile_state.reservoir_reduction(mat.assumed_bulk_density_kg_m3)
    mobile_step = solver.step_resident(1.0 / 30.0)
    mobile_after = mobile_state.reservoir_reduction(mat.assumed_bulk_density_kg_m3)
    view = mobile_state.explicit_host_view(source="acceptance")
    uniform_before = float(np.sum(h0, dtype=np.float64) * grid.dx * grid.dy)
    uniform_after = float(
        np.sum(view.H_mobile_m, dtype=np.float64) * grid.dx * grid.dy
    )
    weighted_residual = float(
        mobile_after["mobile_volume_m3"] - mobile_before["mobile_volume_m3"]
    )
    result = {
        "schema": "P0-2A_BOUNDARY_CONTROL_VOLUME_PROBE/v1",
        "device": str(mobile_state.runtime.device),
        "tracksoil": {
            "accepted_r2m_m3": track_step.resting_to_mobile_volume_m3,
            "authoritative_weighted_residual_m3": track_residual,
            "status": "PASS" if abs(track_residual) <= 1.0e-10 else "FAIL",
        },
        "mobile_v2_boundary_transport": {
            "uniform_volume_before_m3": uniform_before,
            "uniform_volume_after_m3": uniform_after,
            "uniform_residual_m3": uniform_after - uniform_before,
            "authoritative_weighted_volume_before_m3": mobile_before["mobile_volume_m3"],
            "authoritative_weighted_volume_after_m3": mobile_after["mobile_volume_m3"],
            "authoritative_weighted_residual_m3": weighted_residual,
            "solver_reported_transport_residual_m3": mobile_step.transport_mass_residual_m3,
            "causal_match_m3": weighted_residual - mobile_step.transport_mass_residual_m3,
            "diagnosis": "UNWEIGHTED_HEIGHT_CONSERVED_BUT_TRIANGLE_A_C_VOLUME_NOT_CONSERVED",
        },
    }
    output = ROOT / "outputs/mobile_v2_production/tracksoil_mobile_boundary_probe.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    APP.close()


if __name__ == "__main__":
    main()
