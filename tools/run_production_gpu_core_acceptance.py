#!/usr/bin/env python3
"""Exercise every resident operator through production EarthmovingPhysicsCore."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# Standalone Isaac's python.sh does not expose extension packages until Kit is
# started.  Acceptance tools do not need a SimulationApp, so make the bundled
# Warp package importable directly when running them from the command line.
_isaac_root = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for _warp_root in sorted(_isaac_root.glob("extscache/omni.warp.core-*")):
    if (_warp_root / "warp").is_dir():
        sys.path.insert(0, str(_warp_root))
        break

from isaac_bulk_pipeline.bulk_state import MaterialParcel, MaterialScenario, PayloadState
from isaac_bulk_pipeline.runtime import (
    EarthmovingPhysicsCore,
    GpuRuntimeMetadata,
    ResetLevel,
    SoilForceMode,
)
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


OUTPUT = ROOT / "outputs/390f_v2/production_gpu_core_acceptance.json"


def _tool(descriptor, x_m: float, z_m: float, timestamp_s: float) -> ToolState:
    pose = np.eye(4)
    pose[:3, 3] = [x_m, 15.0, z_m]

    def transform(points: np.ndarray) -> np.ndarray:
        return (pose @ np.c_[points, np.ones(len(points))].T).T[:, :3]

    return ToolState(
        timestamp_s,
        pose,
        pose,
        transform(descriptor.cutting_edge_local),
        transform(descriptor.bottom_profile_local),
        transform(descriptor.left_boundary_local),
        transform(descriptor.right_boundary_local),
        np.asarray([0.2, 0.0, 0.0]),
        np.zeros(3),
    )


def main() -> None:
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/Terrain")
    material = MaterialScenario(
        "production_core", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35
    )
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(
            ROOT / "configs/excavator_390f_real_bucket.yaml"
        )
    )
    core = EarthmovingPhysicsCore(
        grid=grid,
        descriptor=descriptor,
        material=material,
        initial_heightmap_m=np.ones(grid.shape),
        runtime_backend="GPU_RUNTIME",
        tile_size=64,
    )
    previous = _tool(descriptor, 15.00, 0.95, 0.0)
    current = _tool(descriptor, 15.01, 0.94, 1.0 / 60.0)
    core.initialize_tool(previous)
    dig = core.step(
        current,
        phase="penetrate",
        cycle=1,
        dt_s=1.0 / 60.0,
        soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
    )
    footprint = np.zeros(grid.shape, dtype=bool)
    footprint[300:305, 280:320] = True
    core.apply_track_soil(
        left_footprint_mask=footprint,
        right_footprint_mask=np.zeros_like(footprint),
        left_track_velocity_xy_m_s=np.asarray([0.5, 0.0]),
        right_track_velocity_xy_m_s=np.zeros(2),
        base_velocity_xy_m_s=np.zeros(2),
        dt_s=1.0 / 60.0,
    )
    # Clear dynamic material at the explicit reset boundary; operator records
    # intentionally remain as the call-graph audit for this Core instance.
    core.reset(ResetLevel.ALL)

    # A conservative tiny Payload→Airborne setup makes landing, deposition and
    # residual frontier execute in one bounded production step.
    assert core.device_state is not None
    loaded = PayloadState(
        1.0e-6,
        descriptor.effective_capacity_m3,
        material.assumed_bulk_density_kg_m3,
        np.zeros(3),
    )
    core.gpu_metadata = GpuRuntimeMetadata(core.device_state, loaded)
    parcel = MaterialParcel(
        "acceptance_landing",
        loaded.volume_m3,
        material.assumed_bulk_density_kg_m3,
        grid.terrain_to_world(np.asarray([15.0, 15.0, 1.0])),
        np.asarray([0.0, 0.0, -0.1]),
        core.device_state.timestamp_device_s,
        "production_gpu_core_acceptance",
    )
    core.gpu_metadata.commit_payload_to_airborne(
        loaded.with_volume(0.0), (parcel,), loaded.volume_m3
    )
    advance = core.step(
        current,
        phase="deposition",
        cycle=1,
        dt_s=1.0 / 60.0,
        soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
    )
    modules = [item.module for item in core.gpu_chain._records]
    required = {
        "FailureZoneActivation",
        "BucketIntake",
        "Mobile",
        "TrackSoil",
        "AirborneLanding",
        "Deposition",
        "LargeAvalancheDeviceTransition",
    }
    missing = sorted(required - set(modules))
    transfer = advance.device_transfer.to_dict()
    passed = bool(
        dig.state is None
        and advance.state is None
        and not missing
        and float(np.linalg.norm(dig.computed_force_terrain_n)) > 0.0
        and float(np.linalg.norm(dig.applied_force_terrain_n)) > 0.0
        and transfer["full_field_h2d_count"] == 0
        and transfer["full_field_d2h_count"] == 0
        and abs(advance.mass_balance_error_m3) <= 1.0e-8
    )
    report = {
        "schema": "390F_PRODUCTION_GPU_CORE/v1",
        "status": "PASS" if passed else "FAIL",
        "runtime_backend": advance.runtime_backend,
        "state_authority": core.device_state.authority.value,
        "production_core_class": type(core).__name__,
        "operators_called_by_production_core": modules,
        "missing_required_operators": missing,
        "minislope_call_status": (
            "QUIET_GATED_NO_PHYSICAL_RESIDUAL_SEED"
            if "MiniSlopeCompactFrontier" not in modules
            else "QUASI_STATIC_BED_TRANSPORT_EXECUTED"
        ),
        "terrain_state_returned": advance.state is not None,
        "normal_physics_transfer": transfer,
        "mass_balance_error_m3": advance.mass_balance_error_m3,
        "soil_force": {
            "failure_zone_applicability": (
                dig.interaction.failure_zone.applicability_status
            ),
            "quasi_static_resultant_n": float(
                np.linalg.norm(dig.quasi_static_force_terrain_n)
            ),
            "momentum_resultant_n": float(
                np.linalg.norm(dig.momentum_force_terrain_n)
            ),
            "computed_resultant_n": float(
                np.linalg.norm(dig.computed_force_terrain_n)
            ),
            "applied_resultant_n": float(
                np.linalg.norm(dig.applied_force_terrain_n)
            ),
        },
        "static_relaxation": {
            "iterations": advance.static_relaxation_iterations,
            "active_tiles": advance.static_relaxation_active_tiles,
            "pending": advance.static_relaxation_pending,
        },
        "launcher_selection_gate": (
            "EXPLICIT_GPU_RUNTIME_SELECTION_ENABLED_PENDING_REAL_CYCLE_ACCEPTANCE"
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
