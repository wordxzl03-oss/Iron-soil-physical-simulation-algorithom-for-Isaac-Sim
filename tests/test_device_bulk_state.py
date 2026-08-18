"""GPU-authoritative state ownership and shared-operator regressions."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_bulk_pipeline.bulk_interaction import (
    DepositionOperator,
    WarpMobileLayerSolver,
    WarpTrackSoilOperator,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.performance import probe_warp
from isaac_bulk_pipeline.runtime import (
    BulkStateAuthority,
    BulkStateAuthorityError,
    DeviceBulkState,
    DeviceMaterialLedger,
    GpuBulkOperatorChain,
)
from isaac_bulk_pipeline.solvers import WarpCompactActiveEdgeOperator
from isaac_bulk_pipeline.terrain import TerrainGrid


GPU_REQUIRED = pytest.mark.skipif(
    not probe_warp().available, reason=probe_warp().reason or "Warp CUDA unavailable"
)


def _grid() -> TerrainGrid:
    return TerrainGrid(16, 16, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")


def _material() -> MaterialScenario:
    return MaterialScenario("device_state", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)


@GPU_REQUIRED
def test_device_state_is_the_only_authority_and_host_view_cannot_commit():
    grid = _grid()
    state = DeviceBulkState(grid, np.ones(grid.shape), tile_size=4)
    assert state.authority is BulkStateAuthority.DEVICE
    view = state.explicit_host_view(source="debug")
    assert view.authority is BulkStateAuthority.DEVICE
    assert not view.H_resting_m.flags.writeable
    with pytest.raises(BulkStateAuthorityError):
        view.commit()
    assert state.diagnostics()["state_authority"] == "DEVICE"


@GPU_REQUIRED
def test_shared_mobile_track_and_frontier_bind_no_shadow_terrain():
    grid = _grid()
    state = DeviceBulkState(grid, np.ones(grid.shape), tile_size=4)
    material = _material()
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    mobile = np.zeros(grid.shape)
    mobile[6:10, 6:10] = 0.03
    # Initialization is an allowed full input boundary.  Reset the per-step
    # counter before asserting the normal loop budget.
    state.apply_host_patch("mobile", (0, 16, 0, 16), mobile, reason="test_setup")
    state.begin_physics_step()
    mobile_solver = WarpMobileLayerSolver(runtime=state.runtime)
    mobile_solver.bind_device_state(state, material, grid, integrator)
    track_solver = WarpTrackSoilOperator(grid.shape, runtime=state.runtime)
    track_solver.bind_device_state(state)
    frontier = WarpCompactActiveEdgeOperator(grid.shape, runtime=state.runtime)
    frontier.bind_device_state(state)
    assert state.runtime.arrays["height"] is state.runtime.arrays["mobile"]
    assert frontier.diagnostics()["resident_state"] == ["resting", "frontier_reached"]
    before = state.reservoir_reduction(material.assumed_bulk_density_kg_m3)
    mobile_solver.step_resident(1.0 / 120.0)
    after = state.reservoir_reduction(material.assumed_bulk_density_kg_m3)
    assert after["mobile_volume_m3"] == pytest.approx(before["mobile_volume_m3"], abs=1e-11)
    state.assert_normal_step_transfer_budget()


@GPU_REQUIRED
def test_dirty_tile_extraction_is_compact_and_reset_restores_baseline():
    grid = _grid()
    state = DeviceBulkState(grid, np.ones(grid.shape), tile_size=4)
    baseline = state.explicit_host_view(source="acceptance")
    state.begin_physics_step()
    state.apply_host_patch(
        "resting", (4, 6, 5, 7), np.full((2, 2), 0.75), reason="local_failure"
    )
    tiles = state.download_dirty_tiles()
    assert set(tiles) == {5}
    # A 4-cell tile includes its final shared vertex row/column so a local
    # mesh/contact chunk can be published without a full-heightmap download.
    assert tiles[5].shape == (5, 5)
    transfer = state.transfer_snapshot()
    assert transfer.full_field_h2d_count == 0
    assert transfer.full_field_d2h_count == 0
    state.reset()
    restored = state.explicit_host_view(source="reset")
    np.testing.assert_allclose(restored.H_resting_m, baseline.H_resting_m, atol=0.0)
    np.testing.assert_allclose(restored.H_mobile_m, baseline.H_mobile_m, atol=0.0)
    np.testing.assert_allclose(restored.mobile_momentum_m2_s, baseline.mobile_momentum_m2_s, atol=0.0)
    assert restored.reset_generation == baseline.reset_generation + 1


@GPU_REQUIRED
def test_compact_patch_bridge_never_records_a_full_field_download():
    grid = _grid()
    state = DeviceBulkState(grid, np.arange(grid.nx * grid.ny, dtype=np.float64).reshape(grid.shape), tile_size=4)
    state.begin_physics_step()
    patch = state.download_patch((1, 3, 2, 4), source="failure_geometry")
    np.testing.assert_allclose(patch.H_resting_m, [[18.0, 19.0], [34.0, 35.0]])
    assert patch.mobile_momentum_m2_s.shape == (2, 2, 2)
    assert state.transfer_snapshot().full_field_d2h_count == 0


@GPU_REQUIRED
def test_compact_region_and_batched_surface_query_do_not_become_full_field_reads():
    grid = _grid()
    height = np.arange(grid.nx * grid.ny, dtype=np.float64).reshape(grid.shape)
    state = DeviceBulkState(grid, height, tile_size=4)
    state.begin_physics_step()
    region = state.download_region((1, 9, 2, 10), source="failure_zone_geometry")
    np.testing.assert_allclose(region.H_resting_m, height[1:9, 2:10])
    values = state.sample_surface_bilinear(
        np.asarray([[1.0, 2.0], [2.5, 3.5]]), source="airborne_landing"
    )
    np.testing.assert_allclose(values, [18.0, 43.5])
    assert state.transfer_snapshot().full_field_d2h_count == 0
    with pytest.raises(BulkStateAuthorityError, match="FULL_FIELD"):
        state.download_region((0, grid.ny, 0, grid.nx), source="invalid_runtime")


@GPU_REQUIRED
def test_device_material_ledger_uses_authoritative_reductions_not_host_terrain():
    grid = _grid()
    state = DeviceBulkState(grid, np.ones(grid.shape), tile_size=4)
    ledger = DeviceMaterialLedger(state, 1370.0)
    state.begin_physics_step()
    patch = np.full((2, 2), 0.75)
    state.apply_host_patch("resting", (4, 6, 4, 6), patch, reason="test_mutation")
    # This mutation intentionally changes material, so ledger exposes it rather
    # than masking it through a stale host snapshot.
    snapshot = ledger.snapshot(state, payload_m3=0.0, airborne_m3=0.0, outflow_m3=0.0)
    assert snapshot.absolute_volume_error_m3 < 0.0
    assert state.transfer_snapshot().full_field_d2h_count == 0


@GPU_REQUIRED
def test_device_deposition_reads_and_updates_the_same_authoritative_fields():
    grid = _grid()
    material = _material()
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    resting = np.ones(grid.shape)
    mobile = np.zeros(grid.shape)
    mobile[6:10, 6:10] = 0.03
    momentum = np.zeros(grid.shape + (2,))
    reference = DepositionOperator().apply(
        resting, mobile, momentum, material, grid, integrator, 1.0 / 60.0
    )
    state = DeviceBulkState(grid, resting, tile_size=4)
    state.apply_host_patch("mobile", (0, 16, 0, 16), mobile, reason="test_setup")
    state.begin_physics_step()
    chain = GpuBulkOperatorChain(state, material, grid, integrator)
    result = chain.step_deposition(1.0 / 60.0)
    view = state.explicit_host_view(source="acceptance")
    np.testing.assert_allclose(view.H_resting_m, reference.H_resting_m, atol=1e-13)
    np.testing.assert_allclose(view.H_mobile_m, reference.mobile_height_m, atol=1e-13)
    assert result.deposited_volume_m3 == pytest.approx(
        reference.deposited_volume_m3, abs=1e-13
    )


@GPU_REQUIRED
def test_device_deposition_settles_mobile_blanket_over_steep_substrate():
    grid = _grid()
    material = _material()
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    columns = np.arange(grid.nx, dtype=np.float64)[None, :]
    resting = np.repeat(0.08 * columns, grid.ny, axis=0)
    mobile = np.max(resting) - resting + 0.02
    momentum = np.zeros(grid.shape + (2,))
    reference = DepositionOperator().apply(
        resting, mobile, momentum, material, grid, integrator, 1.0 / 60.0
    )
    state = DeviceBulkState(grid, resting, tile_size=4)
    state.apply_host_patch("mobile", (0, 16, 0, 16), mobile, reason="test_setup")
    state.begin_physics_step()
    chain = GpuBulkOperatorChain(state, material, grid, integrator)
    result = chain.step_deposition(1.0 / 60.0)
    view = state.explicit_host_view(source="acceptance")
    assert result.deposited_volume_m3 > 0.0
    np.testing.assert_allclose(view.H_resting_m, reference.H_resting_m, atol=1e-13)
    np.testing.assert_allclose(view.H_mobile_m, reference.mobile_height_m, atol=1e-13)


@GPU_REQUIRED
def test_device_deposition_does_not_force_settle_unresolved_fast_tail():
    grid = _grid()
    material = _material()
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    resting = np.zeros(grid.shape)
    resting[8, 8] = 0.20
    mobile = np.zeros(grid.shape)
    mobile[8, 9] = 0.01
    momentum = np.zeros(grid.shape + (2,))
    momentum[8, 9, 0] = 0.02
    reference = DepositionOperator().apply(
        resting, mobile, momentum, material, grid, integrator, 1.0 / 60.0
    )
    state = DeviceBulkState(grid, resting, tile_size=4)
    state.apply_host_patch("mobile", (0, 16, 0, 16), mobile, reason="test_setup")
    state.apply_host_patch(
        "momentum_x", (0, 16, 0, 16), momentum[..., 0], reason="test_setup"
    )
    state.begin_physics_step()
    chain = GpuBulkOperatorChain(state, material, grid, integrator)
    result = chain.step_deposition(1.0 / 60.0)
    view = state.explicit_host_view(source="acceptance")
    assert result.deposited_volume_m3 == pytest.approx(0.0, abs=1e-13)
    np.testing.assert_allclose(view.H_resting_m, reference.H_resting_m, atol=1e-13)
    np.testing.assert_allclose(view.H_mobile_m, reference.mobile_height_m, atol=1e-13)
