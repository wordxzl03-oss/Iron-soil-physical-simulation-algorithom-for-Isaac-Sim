"""Correctness and execution-contract tests for compact tile MiniSlope."""

from __future__ import annotations

import unittest

import numpy as np

from isaac_bulk_pipeline.config import SolverConfig
from isaac_bulk_pipeline.solvers import (
    CompactTileFrontier,
    MinimumSlopeAdapter,
    NumericalNonconvergenceError,
    SparseTileFrontierMinimumSlopeAdapter,
)
from isaac_bulk_pipeline.terrain import TerrainGrid


def _config(
    *, classification_threshold: int = 1_000, safety_limit: int = 20_000
) -> SolverConfig:
    return SolverConfig(
        critical_angle_deg=30.0,
        max_iterations=20_000,
        large_avalanche_iteration_threshold=classification_threshold,
        numerical_safety_max_iterations=safety_limit,
        tolerance=1.0e-8,
        boundary_condition="closed",
        conservation_tolerance_m3=1.0e-7,
    )


class SparseTileSlopeTests(unittest.TestCase):
    def test_compact_edge_batches_cover_each_full_grid_edge_once(self):
        shape = (17, 19)
        frontier = CompactTileFrontier(shape, tile_size=8)
        all_tiles = np.arange(frontier.tile_count, dtype=np.int32)
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        expected = (16 * 19, 17 * 18, 16 * 18, 16 * 18)
        for direction_index, ((di, dj), expected_count) in enumerate(
            zip(directions, expected)
        ):
            edges = []
            for phase in (0, 1):
                batch = frontier.edge_batch(
                    all_tiles,
                    direction_index=direction_index,
                    di=di,
                    dj=dj,
                    phase=phase,
                )
                self.assertTrue(batch.a_i.flags.c_contiguous)
                self.assertEqual(batch.kernel_buffers()["direction_index"], direction_index)
                edges.extend(
                    zip(
                        batch.a_i.tolist(),
                        batch.a_j.tolist(),
                        batch.b_i.tolist(),
                        batch.b_j.tolist(),
                    )
                )
            self.assertEqual(len(edges), expected_count)
            self.assertEqual(len(set(edges)), expected_count)

    def test_sparse_local_event_matches_reference_and_uses_compact_work(self):
        grid = TerrainGrid(129, 129, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        baseline = np.ones(grid.shape)
        disturbed = baseline.copy()
        disturbed[64, 64] += 0.30
        reference = MinimumSlopeAdapter(grid, _config()).solve(disturbed)
        solver = SparseTileFrontierMinimumSlopeAdapter(
            grid, _config(), tile_size=16
        )
        solver.set_reference_height(baseline)
        result = solver.solve(disturbed)

        np.testing.assert_allclose(
            result.heightmap_stable,
            reference.heightmap_stable,
            rtol=0.0,
            atol=2.0e-8,
        )
        diagnostics = result.diagnostics
        self.assertTrue(diagnostics["true_sparse_tile_frontier"])
        self.assertTrue(diagnostics["active_tiles_drive_numerical_work"])
        self.assertGreater(diagnostics["retired_tile_total"], 0)
        self.assertLess(
            diagnostics["compact_edge_evaluation_count"],
            diagnostics["full_domain_edge_evaluation_equivalent"],
        )
        self.assertLess(diagnostics["reached_cell_count"], diagnostics["bounding_box_cell_count"])

    def test_incremental_frontier_budget_matches_complete_solver(self):
        grid = TerrainGrid(65, 65, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        baseline = np.ones(grid.shape)
        disturbed = baseline.copy()
        disturbed[32, 32] += 0.30
        seeds = np.zeros(grid.shape, dtype=bool)
        seeds[32, 32] = True

        complete_solver = SparseTileFrontierMinimumSlopeAdapter(
            grid, _config(), tile_size=8
        )
        complete_solver.set_reference_height(baseline)
        expected = complete_solver.solve(disturbed)

        scheduled = SparseTileFrontierMinimumSlopeAdapter(
            grid, _config(), tile_size=8
        )
        scheduled.set_reference_height(baseline)
        scheduled.begin_incremental(disturbed, seeds)
        progress = scheduled.advance_incremental(round_budget=1)
        self.assertEqual(progress.iteration_count, 1)
        self.assertFalse(progress.complete)
        initial_volume = grid.compute_volume(disturbed)
        self.assertAlmostEqual(grid.compute_volume(progress.heightmap_m), initial_volume)
        while not progress.complete:
            previous_iteration = progress.iteration_count
            progress = scheduled.advance_incremental(round_budget=1)
            self.assertLessEqual(progress.iteration_count - previous_iteration, 1)
            self.assertAlmostEqual(
                grid.compute_volume(progress.heightmap_m), initial_volume
            )

        self.assertIsNotNone(progress.result)
        np.testing.assert_allclose(
            progress.heightmap_m,
            expected.heightmap_stable,
            rtol=0.0,
            atol=2.0e-8,
        )

    def test_toe_frontier_expands_tiles_then_retires_stable_tiles(self):
        grid = TerrainGrid(65, 65, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        critical_drop = np.tan(np.deg2rad(30.0)) * grid.dx
        baseline = np.repeat(
            (1.0 + np.arange(grid.nx) * 0.99 * critical_drop)[None, :],
            grid.ny,
            axis=0,
        )
        disturbed = baseline.copy()
        disturbed[28:37, 0] -= 0.10
        reference = MinimumSlopeAdapter(grid, _config()).solve(disturbed)
        solver = SparseTileFrontierMinimumSlopeAdapter(
            grid, _config(classification_threshold=100), tile_size=8
        )
        solver.set_reference_height(baseline)
        result = solver.solve(disturbed)

        np.testing.assert_allclose(
            result.heightmap_stable,
            reference.heightmap_stable,
            rtol=0.0,
            atol=2.0e-8,
        )
        diagnostics = result.diagnostics
        self.assertGreater(diagnostics["ever_active_tile_count"], diagnostics["initial_active_tile_count"])
        self.assertGreater(diagnostics["retired_tile_total"], 0)
        self.assertGreater(len(diagnostics["frontier_tile_history_rle"]), 1)
        self.assertEqual(diagnostics["avalanche_classification"], "LARGE_SUSTAINED_AVALANCHE")
        self.assertFalse(diagnostics["propagation_truncated"])

    def test_classification_threshold_does_not_truncate_but_safety_limit_fails(self):
        grid = TerrainGrid(33, 33, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
        baseline = np.ones(grid.shape)
        disturbed = baseline.copy()
        disturbed[16, 16] += 0.30

        classified = SparseTileFrontierMinimumSlopeAdapter(
            grid,
            _config(classification_threshold=1, safety_limit=100),
            tile_size=8,
        )
        classified.set_reference_height(baseline)
        result = classified.solve(disturbed)
        self.assertTrue(result.converged)
        self.assertEqual(
            result.diagnostics["avalanche_classification"],
            "LARGE_SUSTAINED_AVALANCHE",
        )

        guarded = SparseTileFrontierMinimumSlopeAdapter(
            grid,
            _config(classification_threshold=1, safety_limit=1),
            tile_size=8,
        )
        guarded.set_reference_height(baseline)
        with self.assertRaises(NumericalNonconvergenceError) as captured:
            guarded.solve(disturbed)
        self.assertEqual(captured.exception.code, "NUMERICAL_NONCONVERGENCE")
        self.assertEqual(captured.exception.safety_limit, 1)


if __name__ == "__main__":
    unittest.main()
