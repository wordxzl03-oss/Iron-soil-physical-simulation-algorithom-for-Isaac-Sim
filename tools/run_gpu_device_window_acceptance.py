#!/usr/bin/env python3
"""Controlled shared-DeviceBulkState CPU/GPU checkpoint comparison.

This is intentionally a short synchronized window, not a claim that the
complete 390F task has migrated to GPU authority.  It exercises the already
ported TrackSoil -> Mobile -> incremental Deposition chain with a single shared
device state and proves normal-step transfer accounting.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    DepositionOperator,
    MobileLayerSolver,
    TrackSoilModel,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.performance import probe_warp  # noqa: E402
from isaac_bulk_pipeline.runtime import DeviceBulkState, GpuBulkOperatorChain  # noqa: E402
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


OUTPUT = ROOT / "outputs" / "390f_v2" / "gpu_device_window_acceptance.json"

# These are deliberately much tighter than the project-level field tolerance:
# this window executes matched reduced-order operators from the same initial
# state.  They are acceptance criteria, not merely values printed for review.
EQUIVALENCE_TOLERANCES = {
    "resting_linf_m": 1.0e-10,
    "mobile_linf_m": 1.0e-10,
    "momentum_linf_m2_s": 1.0e-10,
    "volume_error_m3": 1.0e-9,
}


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


def main() -> None:
    status = probe_warp()
    if not status.available:
        payload = {
            "schema": "390F_GPU_DEVICE_WINDOW/v1",
            "status": "UNAVAILABLE",
            "reason": status.reason,
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit(2)

    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    scenario = material()
    resting = np.ones(grid.shape, dtype=np.float64)
    mobile = np.zeros(grid.shape, dtype=np.float64)
    mobile[330:370, 330:370] = 0.025
    momentum = np.zeros(grid.shape + (2,), dtype=np.float64)
    momentum[..., 0] = mobile * 0.10
    momentum[..., 1] = mobile * -0.02
    left = np.zeros(grid.shape, dtype=bool)
    right = np.zeros(grid.shape, dtype=bool)
    left[340:348, 300:360] = True
    right[354:362, 300:360] = True
    velocities = {
        "left_track_velocity_xy_m_s": np.asarray([1.0, 0.0]),
        "right_track_velocity_xy_m_s": np.asarray([0.9, 0.0]),
        "base_velocity_xy_m_s": np.asarray([0.2, 0.0]),
        "dt_s": 1.0 / 60.0,
    }

    # Host reference uses identical order/inputs.
    track_reference = TrackSoilModel()
    track_reference.initialize(resting)
    tracked = track_reference.apply(
        resting,
        mobile,
        momentum,
        left_footprint_mask=left,
        right_footprint_mask=right,
        grid=grid,
        integrator=integrator,
        **velocities,
    )
    moved = MobileLayerSolver().step(
        tracked.H_resting_m,
        tracked.mobile_height_m,
        tracked.mobile_momentum_m2_s,
        scenario,
        grid,
        integrator,
        velocities["dt_s"],
    )
    reference = DepositionOperator().apply(
        tracked.H_resting_m,
        moved.mobile_height_m,
        moved.mobile_momentum_m2_s,
        scenario,
        grid,
        integrator,
        velocities["dt_s"],
    )

    device = DeviceBulkState(grid, resting, tile_size=64)
    # Setup is an initialization boundary and is excluded from normal-step
    # transfer counters.  A production runtime seeds from initial state/compact
    # tool commands rather than this test fixture.
    device.apply_host_patch("mobile", (0, 701, 0, 701), mobile, reason="acceptance_initialization")
    device.apply_host_patch("momentum_x", (0, 701, 0, 701), momentum[..., 0], reason="acceptance_initialization")
    device.apply_host_patch("momentum_y", (0, 701, 0, 701), momentum[..., 1], reason="acceptance_initialization")
    device.begin_physics_step()
    chain = GpuBulkOperatorChain(device, scenario, grid, integrator)
    track_gpu = chain.apply_track_soil(
        left_footprint_mask=left,
        right_footprint_mask=right,
        **velocities,
    )
    mobile_gpu = chain.step_mobile(velocities["dt_s"])
    deposition_gpu = chain.step_deposition(velocities["dt_s"])
    chain_records = chain.diagnostics()["records"]
    normal_transfer = device.transfer_snapshot().to_dict()
    device.assert_normal_step_transfer_budget()
    view = device.explicit_host_view(source="acceptance")
    terrain_difference = view.H_resting_m - reference.H_resting_m
    mobile_difference = view.H_mobile_m - reference.mobile_height_m
    momentum_difference = view.mobile_momentum_m2_s - reference.mobile_momentum_m2_s
    ledger = device.reservoir_reduction(scenario.assumed_bulk_density_kg_m3)
    equivalence = {
        "resting_linf_m": float(np.max(np.abs(terrain_difference))),
        "resting_rmse_m": float(np.sqrt(np.mean(terrain_difference**2))),
        "mobile_linf_m": float(np.max(np.abs(mobile_difference))),
        "mobile_rmse_m": float(np.sqrt(np.mean(mobile_difference**2))),
        "momentum_linf_m2_s": float(np.max(np.abs(momentum_difference))),
        "track_volume_error_m3": float(track_gpu.resting_to_mobile_volume_m3 - tracked.resting_to_mobile_volume_m3),
        "deposition_volume_error_m3": float(deposition_gpu.deposited_volume_m3 - reference.deposited_volume_m3),
        "mobile_volume_error_m3": float(ledger["mobile_volume_m3"] - integrator.integrate(reference.mobile_height_m)),
    }
    passed = bool(
        normal_transfer["full_field_h2d_count"] == 0
        and normal_transfer["full_field_d2h_count"] == 0
        and equivalence["resting_linf_m"] <= EQUIVALENCE_TOLERANCES["resting_linf_m"]
        and equivalence["mobile_linf_m"] <= EQUIVALENCE_TOLERANCES["mobile_linf_m"]
        and equivalence["momentum_linf_m2_s"] <= EQUIVALENCE_TOLERANCES["momentum_linf_m2_s"]
        and max(
            abs(equivalence["track_volume_error_m3"]),
            abs(equivalence["deposition_volume_error_m3"]),
            abs(equivalence["mobile_volume_error_m3"]),
        ) <= EQUIVALENCE_TOLERANCES["volume_error_m3"]
    )
    payload = {
        "schema": "390F_GPU_DEVICE_WINDOW/v1",
        "status": "PASS" if passed else "FAIL",
        "scope": "CONTROLLED_TRACK_MOBILE_DEPOSITION_WINDOW_NOT_FULL_390F_CYCLE",
        "resolution_m": 0.05,
        "grid_shape_yx": [701, 701],
        "backend_identity": chain.backend_identity,
        "state_authority": "DEVICE",
        "normal_physics_transfer": normal_transfer,
        "acceptance_tolerances": EQUIVALENCE_TOLERANCES,
        "equivalence": equivalence,
        "device_scalars": {
            "resting_volume_m3": ledger["resting_volume_m3"],
            "mobile_volume_m3": ledger["mobile_volume_m3"],
            "mobile_momentum_kg_m_s": ledger["mobile_momentum_kg_m_s"].tolist(),
        },
        "operator_steps": {
            "track_resting_to_mobile_m3": track_gpu.resting_to_mobile_volume_m3,
            "mobile_substeps": mobile_gpu.substeps,
            "deposition_m3": deposition_gpu.deposited_volume_m3,
            "dirty_tile_counts": {
                str(record["module"]): int(record["dirty_tile_count"])
                for record in chain_records
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
