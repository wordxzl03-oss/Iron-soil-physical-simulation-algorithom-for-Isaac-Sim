"""Compact device-patch equivalence for the real CAD FailureZone model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaac_bulk_pipeline.bulk_interaction import FailureZoneModel, ToolTerrainIntersectionModel
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.interaction import SweepResult
from isaac_bulk_pipeline.performance import probe_warp
from isaac_bulk_pipeline.runtime import DeviceBulkState, DeviceFailureZoneBridge
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


ROOT = Path(__file__).resolve().parents[1]
GPU_REQUIRED = pytest.mark.skipif(
    not probe_warp().available, reason=probe_warp().reason or "Warp CUDA unavailable"
)


@GPU_REQUIRED
def test_failure_zone_uses_physical_compact_patch_and_scatter_matches_reference():
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    material = MaterialScenario("test", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(ROOT / "configs/excavator_390f_real_bucket.yaml")
    )
    pose = np.eye(4)
    pose[:3, 3] = [15.0, 15.0, 0.0]

    def transform(points: np.ndarray) -> np.ndarray:
        return (pose @ np.c_[points, np.ones(len(points))].T).T[:, :3]

    tool = ToolState(
        timestamp=0.0,
        pose_world=pose,
        pose_terrain=pose,
        cutting_edge_terrain=transform(descriptor.cutting_edge_local),
        bottom_profile_terrain=transform(descriptor.bottom_profile_local),
        left_boundary_terrain=transform(descriptor.left_boundary_local),
        right_boundary_terrain=transform(descriptor.right_boundary_local),
        linear_velocity=np.asarray([0.5, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
    )
    sweep_mask = np.zeros(grid.shape, dtype=bool)
    sweep_mask[300:308, 290:306] = True
    cut = np.full(grid.shape, np.inf)
    cut[sweep_mask] = 0.95
    sweep = SweepResult((300, 290, 308, 306), sweep_mask, cut, (pose,))
    resting = np.ones(grid.shape)
    reference_intersection = ToolTerrainIntersectionModel().compute(
        resting, sweep, tool, grid, integrator
    )
    reference_failure = FailureZoneModel().compute(
        reference_intersection,
        resting,
        material,
        grid,
        integrator,
        fallback_approach_direction_xy=tool.pose_terrain[:2, 1],
    )
    reference = np.minimum(reference_failure.active_thickness_m, resting)

    state = DeviceBulkState(grid, resting, tile_size=64)
    state.begin_physics_step()
    result = DeviceFailureZoneBridge(state, material, grid, integrator).execute(sweep, tool)
    observed = np.zeros(grid.shape)
    row0, row1, col0, col1 = result.patch_bbox_yx
    observed[row0:row1, col0:col1] = result.activated_height_m
    np.testing.assert_allclose(observed, reference, atol=5.0e-12)
    assert result.patch_cell_count < grid.nx * grid.ny
    assert result.patch_fraction_of_terrain < 0.1
    assert result.activated_volume_m3 == pytest.approx(integrator.integrate(reference), abs=5e-12)
    assert state.transfer_snapshot().full_field_h2d_count == 0
    assert state.transfer_snapshot().full_field_d2h_count == 0
