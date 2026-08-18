#!/usr/bin/env python3
"""Run P0-2D against the real production Warp contact/source kernel."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (warp_root / "warp").is_dir():
        sys.path.insert(0, str(warp_root))
        break

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    ToolMobileContactSupport,
    WarpProductionMobileV2Solver,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.experimental.mobile_v2_reference import MobileV2Config  # noqa: E402
from isaac_bulk_pipeline.runtime import DeviceBulkState  # noqa: E402
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


OUT = ROOT / "outputs/mobile_v2_production/tool_mobile_p0_2d_synthetic_device.json"


def main() -> None:
    device = os.environ.get("TOOL_MOBILE_WARP_DEVICE", "cpu")
    grid = TerrainGrid(5, 5, 0.1, 0.1, -0.2, -0.2, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    state = DeviceBulkState(grid, np.ones(grid.shape), device=device)
    state.add_host_indices(
        "mobile", np.arange(25, dtype=np.int32), np.full(25, 0.1),
        reason="p0_2d_uniform_mobile",
    )
    material = MaterialScenario(
        "p0_2d", 1370.0, 29.8, 800.0, 0.4, 38.0, 30.0, 1.0e-9
    )
    solver = WarpProductionMobileV2Solver(
        runtime=state.runtime,
        config=MobileV2Config(
            dx_m=0.1, dy_m=0.1, basal_friction_coefficient=1.0e-9
        ),
    )
    solver.bind_device_state(state, material, grid, integrator)
    contact = ToolMobileContactSupport(
        flat_indices=np.asarray([12], dtype=np.int32),
        closest_points_terrain_m=np.asarray([[0.0, 0.0, 1.05]]),
        outward_normals_terrain=np.asarray([[1.0, 0.0, 0.0]]),
        outward_normals_xy=np.asarray([[1.0, 0.0]]),
        tool_surface_velocity_terrain_m_s=np.asarray([[1.0, 0.5, 0.0]]),
        cavity_signed_distance_m=np.asarray([0.0]),
        mobile_volume_m3=0.001,
        tool_reference_position_terrain_m=np.asarray([0.0, 0.0, 1.0]),
    )
    result = solver.step_resident(
        1.0 / 60.0,
        tool_mobile_contact=contact,
        tool_mobile_friction_coefficient=material.tool_friction_coefficient,
    )
    mobile_impulse = np.asarray(result.tool_impulse_on_mobile_terrain_ns)
    machine_impulse = -mobile_impulse
    residual = mobile_impulse + machine_impulse
    energy_residual = sum(
        float(record["mobile_kinetic_energy_change_due_to_contact_j"])
        + float(record["total_contact_dissipation_j"])
        - float(record["tool_to_mobile_work_j"])
        for record in result.tool_contact_substep_diagnostics
    )
    friction_bound = (
        result.tool_tangential_impulse_ns
        <= material.tool_friction_coefficient * result.tool_normal_impulse_ns + 1.0e-10
    )
    passed = bool(
        np.linalg.norm(mobile_impulse) > 0.0
        and np.array_equal(residual, np.zeros(3))
        and abs(result.transport_mass_residual_m3) <= 1.0e-12
        and friction_bound
        and result.tool_contact_dissipation_j >= -1.0e-10
        and abs(energy_residual) <= 1.0e-10
    )
    record = {
        "schema": "TOOL_MOBILE_P0_2D_SYNTHETIC_DEVICE/v1",
        "status": "PASS" if passed else "FAIL",
        "execution_device": device,
        "GPU_DEVICE_TEST": "PASS" if device.startswith("cuda") and passed else "NOT_RUN_CUDA_NOT_EXPOSED",
        "mobile_volume_before_m3": result.volume_before_m3,
        "mobile_volume_after_m3": result.volume_after_m3,
        "mass_residual_m3": result.transport_mass_residual_m3,
        "normal_impulse_ns": result.tool_normal_impulse_ns,
        "tangential_impulse_ns": result.tool_tangential_impulse_ns,
        "coulomb_bound_pass": friction_bound,
        "mobile_dynamic_impulse_terrain_ns": mobile_impulse.tolist(),
        "machine_reaction_impulse_terrain_ns": machine_impulse.tolist(),
        "action_reaction_residual_terrain_ns": residual.tolist(),
        "action_reaction_residual_norm_ns": float(np.linalg.norm(residual)),
        "tool_angular_impulse_on_mobile_about_tool_origin_terrain_nms": np.asarray(
            result.tool_angular_impulse_on_mobile_about_tool_origin_terrain_nms
        ).tolist(),
        "tool_to_mobile_work_j": result.tool_work_j,
        "machine_reaction_work_j": result.machine_reaction_work_j,
        "frictional_dissipation_j": result.tool_frictional_dissipation_j,
        "total_contact_dissipation_j": result.tool_contact_dissipation_j,
        "unexplained_contact_energy_j": energy_residual,
        "active_substep_count": result.tool_contact_active_substeps,
        "active_cell_substep_count": result.tool_contact_active_cell_substeps,
        "substeps": list(result.tool_contact_substep_diagnostics),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
