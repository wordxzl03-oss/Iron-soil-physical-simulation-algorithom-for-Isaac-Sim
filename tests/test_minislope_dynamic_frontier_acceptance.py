"""Acceptance tests for dynamically expanding MiniSlope activity."""

from __future__ import annotations

import unittest

import numpy as np

from isaac_bulk_pipeline.config import SolverConfig
from isaac_bulk_pipeline.solvers import (
    EventDrivenMinimumSlopeAdapter,
    MinimumSlopeAdapter,
)
from isaac_bulk_pipeline.terrain import TerrainGrid


def _config(*, max_iterations: int = 5_000) -> SolverConfig:
    return SolverConfig(
        critical_angle_deg=30.0,
        max_iterations=max_iterations,
        large_avalanche_iteration_threshold=max_iterations,
        numerical_safety_max_iterations=max(10_000, max_iterations),
        tolerance=1.0e-8,
        boundary_condition="closed",
        conservation_tolerance_m3=1.0e-7,
    )


def _near_critical_plane(
    shape: tuple[int, int], dx: float, *, slope_fraction: float
) -> np.ndarray:
    critical_drop = np.tan(np.deg2rad(30.0)) * dx
    profile = 1.0 + np.arange(shape[1], dtype=np.float64) * (
        slope_fraction * critical_drop
    )
    return np.repeat(profile[None, :], shape[0], axis=0)


def _edge_slope_distribution(height_yx: np.ndarray, dx: float, dy: float) -> np.ndarray:
    diagonal = float(np.hypot(dx, dy))
    return np.concatenate(
        (
            (np.abs(height_yx[1:, :] - height_yx[:-1, :]) / dy).ravel(),
            (np.abs(height_yx[:, 1:] - height_yx[:, :-1]) / dx).ravel(),
            (np.abs(height_yx[1:, 1:] - height_yx[:-1, :-1]) / diagonal).ravel(),
            (np.abs(height_yx[1:, :-1] - height_yx[:-1, 1:]) / diagonal).ravel(),
        )
    )


def _changed_bbox(before: np.ndarray, after: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(np.abs(after - before) > 1.0e-12)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


class MiniSlopeDynamicFrontierAcceptanceTests(unittest.TestCase):
    def test_small_local_collapse_matches_full_grid_reference(self):
        grid = TerrainGrid(96, 96, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        baseline = np.ones(grid.shape, dtype=np.float64)
        disturbed = baseline.copy()
        disturbed[47, 53] += 0.30

        reference = MinimumSlopeAdapter(grid, _config()).solve(disturbed)
        solver = EventDrivenMinimumSlopeAdapter(grid, _config(), tile_size=8)
        solver.set_reference_height(baseline)
        optimized = solver.solve(disturbed)

        np.testing.assert_allclose(
            optimized.heightmap_stable,
            reference.heightmap_stable,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(optimized.volume_after_m3, reference.volume_after_m3, 12)
        self.assertTrue(optimized.converged)
        self.assertLess(solver.last_active_ratio, 0.05)
        self.assertFalse(optimized.diagnostics["fixed_final_roi"])
        self.assertFalse(optimized.diagnostics["serial_python_cell_queue"])

    def test_toe_excavation_propagates_beyond_seed_and_across_tiles(self):
        grid = TerrainGrid(65, 65, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        baseline = _near_critical_plane(grid.shape, grid.dx, slope_fraction=0.99)
        disturbed = baseline.copy()
        disturbed[28:37, 0] -= 0.10

        reference = MinimumSlopeAdapter(grid, _config()).solve(disturbed)
        solver = EventDrivenMinimumSlopeAdapter(grid, _config(), tile_size=8)
        solver.set_reference_height(baseline)
        optimized = solver.solve(disturbed)

        np.testing.assert_allclose(
            optimized.heightmap_stable,
            reference.heightmap_stable,
            rtol=0.0,
            atol=2.0e-12,
        )
        diagnostics = optimized.diagnostics
        seed_bbox = diagnostics["seed_bbox_yx"]
        propagation_bbox = diagnostics["propagation_bbox_yx"]
        self.assertEqual(seed_bbox, [28, 0, 37, 1])
        self.assertGreaterEqual(propagation_bbox[3], 24)
        self.assertTrue(diagnostics["propagation_crossed_seed_bbox"])
        self.assertGreater(diagnostics["active_tile_count"], diagnostics["initial_active_tile_count"] + 2)
        self.assertGreater(diagnostics["tile_expansion_count"], 2)
        self.assertFalse(diagnostics["propagation_truncated"])

    def test_whole_pile_stress_is_not_truncated_and_matches_full_domain(self):
        grid = TerrainGrid(33, 33, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        baseline = _near_critical_plane(grid.shape, grid.dx, slope_fraction=1.0)
        disturbed = baseline.copy()
        disturbed[:, 0] -= 0.10

        reference = MinimumSlopeAdapter(grid, _config(max_iterations=5_000)).solve(
            disturbed
        )
        # The threshold deliberately lies below the required quasi-static
        # rounds.  The optimized backend must classify and continue, not stop.
        solver = EventDrivenMinimumSlopeAdapter(
            grid, _config(max_iterations=100), tile_size=8
        )
        solver.set_reference_height(baseline)
        optimized = solver.solve(disturbed)

        self.assertTrue(reference.converged)
        self.assertTrue(optimized.converged)
        self.assertGreater(optimized.iteration_count, 100)
        np.testing.assert_allclose(
            optimized.heightmap_stable,
            reference.heightmap_stable,
            rtol=0.0,
            atol=2.0e-12,
        )
        self.assertAlmostEqual(optimized.volume_before_m3, reference.volume_before_m3, 12)
        self.assertAlmostEqual(optimized.volume_after_m3, reference.volume_after_m3, 12)
        self.assertAlmostEqual(optimized.volume_before_m3, optimized.volume_after_m3, 10)

        reference_slopes = _edge_slope_distribution(
            reference.heightmap_stable, grid.dx, grid.dy
        )
        optimized_slopes = _edge_slope_distribution(
            optimized.heightmap_stable, grid.dx, grid.dy
        )
        np.testing.assert_allclose(
            np.quantile(optimized_slopes, [0.0, 0.5, 0.9, 0.99, 1.0]),
            np.quantile(reference_slopes, [0.0, 0.5, 0.9, 0.99, 1.0]),
            rtol=0.0,
            atol=2.0e-12,
        )
        bins = np.linspace(0.0, np.tan(np.deg2rad(30.0)) * 1.01, 33)
        np.testing.assert_array_equal(
            np.histogram(optimized_slopes, bins=bins)[0],
            np.histogram(reference_slopes, bins=bins)[0],
        )
        self.assertEqual(
            _changed_bbox(disturbed, optimized.heightmap_stable),
            _changed_bbox(disturbed, reference.heightmap_stable),
        )
        self.assertEqual(optimized.diagnostics["propagation_bbox_yx"], [0, 0, 33, 33])
        self.assertEqual(
            optimized.diagnostics["avalanche_classification"],
            EventDrivenMinimumSlopeAdapter.LARGE_SUSTAINED_AVALANCHE,
        )
        self.assertTrue(
            optimized.diagnostics["continued_beyond_large_avalanche_threshold"]
        )
        self.assertTrue(
            optimized.diagnostics["future_resting_to_mobile_transfer_required"]
        )
        self.assertFalse(optimized.diagnostics["propagation_truncated"])


if __name__ == "__main__":
    unittest.main()
