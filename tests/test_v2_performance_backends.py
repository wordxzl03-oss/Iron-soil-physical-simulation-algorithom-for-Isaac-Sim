import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    MobileLayerSolver,
    OptimizedMobileLayerSolver,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.config import SolverConfig
from isaac_bulk_pipeline.contact import ChunkedContactMeshBackend
from isaac_bulk_pipeline.performance import (
    ActiveDomainManager,
    ActiveReason,
    PerformanceProfiler,
)
from isaac_bulk_pipeline.solvers import (
    EventDrivenMinimumSlopeAdapter,
    MinimumSlopeAdapter,
)
from isaac_bulk_pipeline.terrain import TerrainGrid


class V2PerformanceBackendTests(unittest.TestCase):
    @staticmethod
    def material():
        return MaterialScenario("v2_test", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)

    def test_active_domain_is_dynamic_tiled_and_reasoned(self):
        manager = ActiveDomainManager((701, 701), tile_size=32)
        mask = np.zeros((701, 701), dtype=bool)
        mask[300:305, 410:420] = True
        window = manager.mark_mask(mask, ActiveReason.BUCKET, halo_cells=4)
        self.assertEqual(window.shape, (13, 18))
        snapshot = manager.snapshot()
        self.assertLess(snapshot.active_ratio, 0.01)
        self.assertGreater(snapshot.active_tile_count, 0)
        self.assertTrue(np.all(snapshot.tile_flags[snapshot.tile_flags != 0] & ActiveReason.BUCKET))
        manager.clear(ActiveReason.BUCKET)
        self.assertEqual(manager.snapshot().active_cell_count, 0)

    def test_optimized_mobile_matches_reference_exactly_on_local_event(self):
        grid = TerrainGrid(128, 128, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        integrator = TerrainVolumeIntegrator.from_grid(grid)
        resting = np.ones(grid.shape)
        mobile = np.zeros(grid.shape)
        mobile[54:74, 50:78] = 0.07
        momentum = np.zeros(grid.shape + (2,))
        momentum[..., 0] = mobile * 0.45
        reference = MobileLayerSolver().step(
            resting, mobile, momentum, self.material(), grid, integrator, 1.0 / 60.0
        )
        solver = OptimizedMobileLayerSolver(tile_size=32)
        optimized = solver.step(
            resting, mobile, momentum, self.material(), grid, integrator, 1.0 / 60.0
        )
        np.testing.assert_array_equal(optimized.mobile_height_m, reference.mobile_height_m)
        np.testing.assert_array_equal(
            optimized.mobile_momentum_m2_s, reference.mobile_momentum_m2_s
        )
        self.assertAlmostEqual(optimized.volume_after_m3, reference.volume_after_m3, 14)
        self.assertIsNotNone(solver.last_active_snapshot)
        self.assertLess(solver.last_active_snapshot.active_ratio, 0.25)

    def test_event_driven_minislope_matches_reference_for_connected_event(self):
        grid = TerrainGrid(512, 512, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        config = SolverConfig(
            critical_angle_deg=30.0,
            max_iterations=1000,
            tolerance=1.0e-8,
            boundary_condition="closed",
            conservation_tolerance_m3=1.0e-8,
        )
        baseline = np.ones(grid.shape)
        disturbed = baseline.copy()
        disturbed[256, 256] += 0.3
        reference = MinimumSlopeAdapter(grid, config).solve(disturbed)
        solver = EventDrivenMinimumSlopeAdapter(grid, config)
        solver.set_reference_height(baseline)
        optimized = solver.solve(disturbed)
        np.testing.assert_allclose(optimized.heightmap_stable, reference.heightmap_stable, atol=1e-12)
        self.assertAlmostEqual(optimized.volume_after_m3, reference.volume_after_m3, 12)
        self.assertLess(solver.last_active_ratio, 0.25)

    def test_performance_breakdown_has_required_distribution_fields(self):
        profiler = PerformanceProfiler()
        for value in (1.0, 2.0, 3.0, 4.0):
            profiler.record_ms("mobile_layer", value)
        report = profiler.report()
        module = report["modules"]["mobile_layer"]
        self.assertEqual(
            set(module),
            {
                "call_count",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "p99_ms",
                "maximum_ms",
                "total_wall_time_ms",
                "percentage_of_total",
            },
        )
        self.assertEqual(module["call_count"], 4)

    def test_contact_backend_updates_only_dirty_chunks(self):
        grid = TerrainGrid(129, 129, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        initial = np.ones(grid.shape)
        backend = ChunkedContactMeshBackend(chunk_size_cells=64)
        backend.initialize(grid, initial)
        self.assertEqual(backend.chunk_count, 4)
        changed = initial.copy()
        changed[10:14, 10:14] -= 0.1
        mask = changed != initial
        update = backend.update(changed, changed_mask=mask)
        self.assertEqual(update.updated_keys, ((0, 0),))
        self.assertEqual(backend.snapshot((0, 0)).revision, 1)
        self.assertEqual(backend.snapshot((1, 1)).revision, 0)


if __name__ == "__main__":
    unittest.main()
