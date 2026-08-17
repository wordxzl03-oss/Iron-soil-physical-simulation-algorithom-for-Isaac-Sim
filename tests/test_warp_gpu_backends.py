"""Optional GPU smoke/equivalence tests; unavailable Warp is an explicit skip."""

import numpy as np
import pytest

from isaac_bulk_pipeline.bulk_interaction import (
    MobileLayerSolver,
    TrackSoilModel,
    WarpMobileLayerSolver,
    WarpTrackSoilOperator,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.performance import WarpBackendUnavailable, probe_warp
from isaac_bulk_pipeline.solvers import CompactTileFrontier, WarpCompactActiveEdgeOperator
from isaac_bulk_pipeline.solvers import SparseTileFrontierMinimumSlopeAdapter
from isaac_bulk_pipeline.config import SolverConfig
from isaac_bulk_pipeline.terrain import TerrainGrid


WARP_STATUS = probe_warp()
GPU_REQUIRED = pytest.mark.skipif(
    not WARP_STATUS.available, reason=WARP_STATUS.reason or "Warp CUDA unavailable"
)


def _material() -> MaterialScenario:
    return MaterialScenario("warp_test", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)


def test_warp_probe_never_relabels_unavailable_backend_as_gpu_pass():
    assert WARP_STATUS.status in {"AVAILABLE", "UNAVAILABLE"}
    assert WARP_STATUS.available == (WARP_STATUS.status == "AVAILABLE")
    if not WARP_STATUS.available:
        assert WARP_STATUS.reason
        with pytest.raises(WarpBackendUnavailable):
            WarpMobileLayerSolver()


@GPU_REQUIRED
def test_warp_mobile_resident_step_matches_cpu_reference_and_conserves_volume():
    grid = TerrainGrid(32, 32, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    resting = np.ones(grid.shape)
    mobile = np.zeros(grid.shape)
    mobile[12:20, 13:19] = 0.03
    momentum = np.zeros(grid.shape + (2,))
    momentum[..., 0] = mobile * 0.1
    reference = MobileLayerSolver().step(
        resting, mobile, momentum, _material(), grid, integrator, 1.0 / 120.0
    )
    solver = WarpMobileLayerSolver()
    solver.initialize_resident(
        resting, mobile, momentum, _material(), grid, integrator
    )
    metrics = solver.step_resident(1.0 / 120.0)
    result = solver.download_result()
    np.testing.assert_allclose(result.mobile_height_m, reference.mobile_height_m, atol=1e-13)
    np.testing.assert_allclose(
        result.mobile_momentum_m2_s,
        reference.mobile_momentum_m2_s,
        atol=1e-13,
    )
    assert abs(metrics.volume_after_m3 - metrics.volume_before_m3) <= 1e-12
    diagnostics = solver.diagnostics()
    assert diagnostics["execution_status"] == "GPU_OPTIMIZED"
    assert diagnostics["h2d_bytes"] > 0
    assert diagnostics["d2h_bytes"] > 0


@GPU_REQUIRED
def test_warp_track_soil_matches_cpu_reference_conservative_transfer():
    grid = TerrainGrid(32, 32, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    resting = np.ones(grid.shape)
    mobile = np.zeros(grid.shape)
    momentum = np.zeros(grid.shape + (2,))
    left = np.zeros(grid.shape, dtype=bool)
    right = np.zeros(grid.shape, dtype=bool)
    left[12:15, 8:20] = True
    right[18:21, 8:20] = True
    arguments = {
        "left_footprint_mask": left,
        "right_footprint_mask": right,
        "left_track_velocity_xy_m_s": np.asarray([1.0, 0.0]),
        "right_track_velocity_xy_m_s": np.asarray([0.8, 0.0]),
        "base_velocity_xy_m_s": np.asarray([0.2, 0.0]),
        "dt_s": 1.0 / 60.0,
    }
    reference_solver = TrackSoilModel()
    reference_solver.initialize(resting)
    reference = reference_solver.apply(
        resting, mobile, momentum, grid=grid, integrator=integrator, **arguments
    )
    solver = WarpTrackSoilOperator(grid.shape)
    solver.initialize(
        resting, resting, mobile, momentum, integrator.vertex_weights_m2
    )
    metrics = solver.apply_resident(**arguments)
    next_resting, next_mobile, next_momentum = solver.download_state()
    np.testing.assert_allclose(next_resting, reference.H_resting_m, atol=1e-13)
    np.testing.assert_allclose(next_mobile, reference.mobile_height_m, atol=1e-13)
    np.testing.assert_allclose(next_momentum, reference.mobile_momentum_m2_s, atol=1e-13)
    assert metrics.resting_to_mobile_volume_m3 == pytest.approx(
        reference.resting_to_mobile_volume_m3, abs=1e-13
    )


@GPU_REQUIRED
def test_warp_compact_edge_phase_preserves_volume_and_consumes_accepted_batch():
    shape = (8, 8)
    frontier = CompactTileFrontier(shape, tile_size=4)
    batch = frontier.edge_batch(
        np.asarray([0], dtype=np.int32),
        direction_index=0,
        di=1,
        dj=0,
        phase=0,
    )
    height = np.ones(shape)
    reached = np.zeros(shape, dtype=bool)
    ai, aj = int(batch.a_i[0]), int(batch.a_j[0])
    height[ai, aj] = 2.0
    reached[ai, aj] = True
    solver = WarpCompactActiveEdgeOperator(shape, tile_size=4)
    solver.initialize(height, reached)
    phase = solver.run_phase(
        batch, critical_difference_m=0.2, tolerance_m=1e-9
    )
    next_height, next_reached = solver.download_state()
    assert phase.moved_edge_count == 1
    assert phase.triggered_tile_ids.size > 0
    assert np.sum(next_height) == pytest.approx(np.sum(height), abs=1e-13)
    assert np.count_nonzero(next_reached) == 2
    assert solver.diagnostics()["compact_batch_semantics"] == "CompactActiveEdgeBatch"


@GPU_REQUIRED
def test_warp_compact_boundary_edge_conserves_triangle_mesh_control_area():
    shape = (8, 8)
    grid = TerrainGrid(8, 8, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    weights_yx = TerrainVolumeIntegrator.from_grid(grid).vertex_weights_m2
    frontier = CompactTileFrontier(shape, tile_size=4)
    batch = frontier.edge_batch(
        np.asarray([0], dtype=np.int32),
        direction_index=0,
        di=1,
        dj=0,
        phase=0,
    )
    height_xy = np.ones(shape, dtype=np.float64)
    reached_xy = np.zeros(shape, dtype=bool)
    ai, aj = int(batch.a_i[0]), int(batch.a_j[0])
    height_xy[ai, aj] = 2.0
    reached_xy[ai, aj] = True
    weights_xy = np.ascontiguousarray(weights_yx.T)
    volume_before = float(np.sum(height_xy * weights_xy, dtype=np.float64))
    solver = WarpCompactActiveEdgeOperator(shape, tile_size=4)
    solver.initialize(height_xy, reached_xy, weights_xy)
    solver.run_phase(batch, critical_difference_m=0.2, tolerance_m=1.0e-9)
    next_height_xy, _ = solver.download_state()
    volume_after = float(np.sum(next_height_xy * weights_xy, dtype=np.float64))
    assert volume_after == pytest.approx(volume_before, abs=1.0e-13)


@GPU_REQUIRED
def test_warp_compact_frontier_full_event_matches_cpu_and_retires_tiles():
    """Exercise production transfer-then-final-scan round semantics to rest."""

    from slope_model import _neighbor_pairs

    grid = TerrainGrid(33, 33, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    config = SolverConfig(
        critical_angle_deg=30.0,
        max_iterations=2_000,
        large_avalanche_iteration_threshold=1_000,
        numerical_safety_max_iterations=2_000,
        tolerance=1.0e-8,
        boundary_condition="closed",
        conservation_tolerance_m3=1.0e-7,
    )
    baseline = np.ones(grid.shape, dtype=np.float64)
    disturbed = baseline.copy()
    disturbed[16, 16] += 0.30
    seeds = np.zeros(grid.shape, dtype=bool)
    seeds[16, 16] = True

    cpu = SparseTileFrontierMinimumSlopeAdapter(grid, config, tile_size=8)
    cpu.set_reference_height(baseline)
    expected = cpu.solve(disturbed)

    # CompactActiveEdgeBatch coordinates use the solver's x/y storage order.
    height_xy = np.ascontiguousarray(disturbed.T)
    reached_xy = np.ascontiguousarray(seeds.T)
    frontier = CompactTileFrontier(height_xy.shape, tile_size=8)
    seed_i, seed_j = np.nonzero(reached_xy)
    active_tiles = frontier.tiles_around_cells(seed_i, seed_j)
    solver = WarpCompactActiveEdgeOperator(height_xy.shape, tile_size=8)
    solver.initialize(height_xy, reached_xy)
    critical = float(np.tan(np.deg2rad(config.critical_angle_deg)))

    iteration_count = 0
    while active_tiles.size:
        iteration_count += 1
        assert iteration_count <= config.numerical_safety_max_iterations
        phase_tiles = active_tiles
        for direction_index, (di, dj, distance) in enumerate(
            _neighbor_pairs(grid.dx, grid.dy)
        ):
            for phase in (0, 1):
                batch = frontier.edge_batch(
                    phase_tiles,
                    direction_index=direction_index,
                    di=di,
                    dj=dj,
                    phase=phase,
                )
                result = solver.relax_phase(
                    batch, critical_difference_m=critical * distance
                )
                if result.triggered_tile_ids.size:
                    phase_tiles = np.union1d(
                        phase_tiles, result.triggered_tile_ids
                    ).astype(np.int32, copy=False)

        unstable_owners = []
        for direction_index, (di, dj, distance) in enumerate(
            _neighbor_pairs(grid.dx, grid.dy)
        ):
            for phase in (0, 1):
                batch = frontier.edge_batch(
                    phase_tiles,
                    direction_index=direction_index,
                    di=di,
                    dj=dj,
                    phase=phase,
                )
                result = solver.scan_phase(
                    batch,
                    critical_difference_m=critical * distance,
                    tolerance_m=config.tolerance,
                )
                if result.unstable_owner_tile_ids.size:
                    unstable_owners.append(result.unstable_owner_tile_ids)
        active_tiles = (
            np.unique(np.concatenate(unstable_owners)).astype(np.int32)
            if unstable_owners
            else np.empty(0, dtype=np.int32)
        )

    actual_xy, reached_actual_xy = solver.download_state()
    np.testing.assert_allclose(
        actual_xy.T, expected.heightmap_stable, rtol=0.0, atol=2.0e-8
    )
    assert iteration_count == expected.iteration_count
    assert np.count_nonzero(reached_actual_xy) > 1
    assert np.sum(actual_xy) == pytest.approx(np.sum(disturbed), abs=1.0e-11)
    diagnostics = solver.diagnostics()
    assert diagnostics["cached_edge_batch_count"] <= diagnostics["edge_batch_cache_limit"]


@GPU_REQUIRED
def test_warp_compact_frontier_evicts_transient_device_edge_batches():
    shape = (33, 33)
    frontier = CompactTileFrontier(shape, tile_size=8)
    solver = WarpCompactActiveEdgeOperator(
        shape, tile_size=8, batch_cache_limit=4
    )
    height = np.ones(shape, dtype=np.float64)
    reached = np.ones(shape, dtype=bool)
    solver.initialize(height, reached)
    for tile in range(min(frontier.tile_count, 10)):
        batch = frontier.edge_batch(
            np.asarray([tile], dtype=np.int32),
            direction_index=0,
            di=1,
            dj=0,
            phase=tile & 1,
        )
        solver.scan_phase(
            batch, critical_difference_m=0.01, tolerance_m=1.0e-8
        )
    diagnostics = solver.diagnostics()
    assert diagnostics["cached_edge_batch_count"] == 4
    assert diagnostics["edge_batch_cache_limit"] == 4
    edge_names = [name for name in solver.runtime.arrays if name.startswith("edge_")]
    assert len(edge_names) == 4 * 5
