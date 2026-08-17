#!/usr/bin/env python3
"""CUDA payload→airborne→device landing→incremental deposition window."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from isaac_bulk_pipeline.bulk_exchange.airborne import AirborneParcelModel
from isaac_bulk_pipeline.bulk_interaction import DepositionOperator
from isaac_bulk_pipeline.bulk_state import MaterialParcel, MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.performance import probe_warp
from isaac_bulk_pipeline.runtime import DeviceAirborneBridge, DeviceBulkState, GpuBulkOperatorChain
from isaac_bulk_pipeline.terrain import TerrainGrid


OUTPUT = ROOT / "outputs" / "390f_v2" / "gpu_dump_window_acceptance.json"


def main() -> None:
    if not probe_warp().available:
        raise SystemExit("GPU_RUNTIME_BACKEND_UNAVAILABLE")
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    material = MaterialScenario("window", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)
    resting = np.ones(grid.shape)
    mobile = np.zeros(grid.shape)
    momentum = np.zeros(grid.shape + (2,))
    released_payload_m3 = 0.020
    parcel = MaterialParcel(
        "payload_release_000", released_payload_m3, material.assumed_bulk_density_kg_m3,
        np.asarray([15.0, 15.0, 1.005]), np.zeros(3), 0.0, "bucket_spill_or_dump",
    )
    dt = 1.0 / 60.0
    cpu_airborne = AirborneParcelModel().advance((parcel,), resting, mobile, momentum, grid, integrator, dt)
    cpu_deposition = DepositionOperator().apply(resting, cpu_airborne.mobile_height_m, cpu_airborne.mobile_momentum_m2_s, material, grid, integrator, dt)

    state = DeviceBulkState(grid, resting, tile_size=64)
    state.begin_physics_step()
    airborne_gpu = DeviceAirborneBridge(state, grid, integrator).advance((parcel,), dt)
    chain = GpuBulkOperatorChain(state, material, grid, integrator)
    deposition_gpu = chain.step_deposition(dt)
    transfer = state.transfer_snapshot().to_dict()
    state.assert_normal_step_transfer_budget()
    view = state.explicit_host_view(source="acceptance")
    errors = {
        "resting_linf_m": float(np.max(np.abs(view.H_resting_m - cpu_deposition.H_resting_m))),
        "mobile_linf_m": float(np.max(np.abs(view.H_mobile_m - cpu_deposition.mobile_height_m))),
        "momentum_linf_m2_s": float(np.max(np.abs(view.mobile_momentum_m2_s - cpu_deposition.mobile_momentum_m2_s))),
        "landed_volume_m3": float(airborne_gpu.landed_volume_m3 - cpu_airborne.landed_volume_m3),
        "deposition_volume_m3": float(deposition_gpu.deposited_volume_m3 - cpu_deposition.deposited_volume_m3),
        "payload_to_airborne_error_m3": float(released_payload_m3 - parcel.volume_m3),
    }
    passed = max(abs(value) for value in errors.values()) <= 1.0e-10 and transfer["full_field_h2d_count"] == 0 and transfer["full_field_d2h_count"] == 0
    result = {
        "schema": "390F_GPU_DUMP_WINDOW/v1",
        "status": "PASS" if passed else "FAIL",
        "state_authority": "DEVICE",
        "scope": "PAYLOAD_AIRBORNE_BATCHED_DEVICE_LANDING_INCREMENTAL_DEPOSITION",
        "resolution_m": 0.05,
        "funnel": {"payload_before_release_m3": released_payload_m3, "payload_to_airborne_m3": parcel.volume_m3, "airborne_landed_m3": airborne_gpu.landed_volume_m3, "deposited_m3": deposition_gpu.deposited_volume_m3, "airborne_remaining_m3": sum(item.volume_m3 for item in airborne_gpu.remaining_parcels)},
        "normal_physics_transfer": transfer,
        "equivalence": errors,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
