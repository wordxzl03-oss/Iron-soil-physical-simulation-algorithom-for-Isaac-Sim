#!/usr/bin/env python3
"""CUDA compact FailureZone→activation→bucket-intake acceptance window."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from isaac_bulk_pipeline.bulk_interaction import BucketIntakeModel, FailureZoneModel, ToolTerrainIntersectionModel
from isaac_bulk_pipeline.bulk_state import MaterialScenario, PayloadState, TerrainVolumeIntegrator
from isaac_bulk_pipeline.interaction import SweepResult
from isaac_bulk_pipeline.performance import probe_warp
from isaac_bulk_pipeline.runtime import DeviceBucketIntakeBridge, DeviceBulkState, DeviceFailureZoneBridge
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


OUTPUT = ROOT / "outputs" / "390f_v2" / "gpu_excavation_window_acceptance.json"


def main() -> None:
    if not probe_warp().available:
        raise SystemExit("GPU_RUNTIME_BACKEND_UNAVAILABLE")
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    material = MaterialScenario("window", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(ROOT / "configs/excavator_390f_real_bucket.yaml")
    )
    pose = np.eye(4)
    pose[:3, 3] = [15.0, 15.0, 0.0]

    def transform(points: np.ndarray) -> np.ndarray:
        return (pose @ np.c_[points, np.ones(len(points))].T).T[:, :3]

    tool = ToolState(0.0, pose, pose, transform(descriptor.cutting_edge_local), transform(descriptor.bottom_profile_local), transform(descriptor.left_boundary_local), transform(descriptor.right_boundary_local), np.asarray([0.5, 0.0, 0.0]), np.zeros(3))
    resting = np.ones(grid.shape)
    mobile = np.zeros(grid.shape)
    mobile[299:302, 270:331] = 0.20
    momentum = np.zeros(grid.shape + (2,))
    momentum[..., 1] = mobile * -0.5
    mask = np.zeros(grid.shape, dtype=bool)
    mask[300:308, 290:306] = True
    cut = np.full(grid.shape, np.inf)
    cut[mask] = 0.95
    sweep = SweepResult((300, 290, 308, 306), mask, cut, (pose,))
    payload = PayloadState(0.0, descriptor.effective_capacity_m3, material.assumed_bulk_density_kg_m3, np.zeros(3))

    # Host reference, preserving the accepted FailureZone and intake formulas.
    intersection = ToolTerrainIntersectionModel().compute(resting, sweep, tool, grid, integrator)
    failure = FailureZoneModel().compute(intersection, resting, material, grid, integrator, fallback_approach_direction_xy=tool.pose_terrain[:2, 1])
    activated = np.minimum(failure.active_thickness_m, resting)
    ref_resting = resting - activated
    ref_mobile = mobile + activated
    ref_momentum = momentum + activated[..., None] * intersection.cutting_edge_velocity_terrain_m_s[:2]
    reference = BucketIntakeModel().apply(ref_mobile, ref_momentum, payload, tool, descriptor, grid, integrator, 1.0 / 60.0, terrain_surface_m=ref_resting)

    state = DeviceBulkState(grid, resting, tile_size=64)
    state.apply_host_patch("mobile", (0, 701, 0, 701), mobile, reason="acceptance_initialization")
    state.apply_host_patch("momentum_y", (0, 701, 0, 701), momentum[..., 1], reason="acceptance_initialization")
    state.begin_physics_step()
    failure_gpu = DeviceFailureZoneBridge(state, material, grid, integrator).execute(sweep, tool)
    intake_gpu = DeviceBucketIntakeBridge(state, grid, integrator).execute(payload, tool, descriptor, 1.0 / 60.0)
    transfers = state.transfer_snapshot().to_dict()
    state.assert_normal_step_transfer_budget()
    view = state.explicit_host_view(source="acceptance")
    errors = {
        "resting_linf_m": float(np.max(np.abs(view.H_resting_m - ref_resting))),
        "mobile_linf_m": float(np.max(np.abs(view.H_mobile_m - reference.mobile_height_m))),
        "momentum_linf_m2_s": float(np.max(np.abs(view.mobile_momentum_m2_s - reference.mobile_momentum_m2_s))),
        "activated_volume_m3": float(failure_gpu.activated_volume_m3 - integrator.integrate(activated)),
        "payload_volume_m3": float(intake_gpu.intake.payload.volume_m3 - reference.payload.volume_m3),
    }
    passed = max(abs(value) for value in errors.values()) <= 1.0e-10 and transfers["full_field_h2d_count"] == 0 and transfers["full_field_d2h_count"] == 0
    result = {
        "schema": "390F_GPU_EXCAVATION_WINDOW/v1",
        "status": "PASS" if passed else "FAIL",
        "state_authority": "DEVICE",
        "scope": "REAL_CAD_FAILUREZONE_ACTIVATION_BUCKET_INTAKE_WINDOW",
        "resolution_m": 0.05,
        "failure_patch": {"requested_bbox_yx": failure_gpu.requested_bbox_yx, "final_bbox_yx": failure_gpu.patch_bbox_yx, "cells": failure_gpu.patch_cell_count, "fraction": failure_gpu.patch_fraction_of_terrain, "expansions": failure_gpu.expansion_count},
        "intake_patch": {"bbox_yx": intake_gpu.patch_bbox_yx, "cells": intake_gpu.patch_cell_count, "fraction": intake_gpu.patch_fraction_of_terrain},
        "funnel": {"intersection_m3": intersection.candidate_intersection_volume_m3, "failure_candidate_m3": failure.active_volume_m3, "activated_m3": failure_gpu.activated_volume_m3, "admitted_m3": intake_gpu.intake.bucket_inflow_volume_m3, "payload_retained_m3": intake_gpu.intake.payload.volume_m3},
        "normal_physics_transfer": transfers,
        "equivalence": errors,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(OUTPUT)}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
