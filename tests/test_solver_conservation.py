import unittest

import numpy as np

from isaac_bulk_pipeline.config import SolverConfig
from isaac_bulk_pipeline.solvers import MinimumSlopeAdapter
from isaac_bulk_pipeline.terrain import TerrainGrid
from slope_model import relax_critical_slope


class MinimumSlopeAdapterTests(unittest.TestCase):
    @staticmethod
    def _grid(nx=31, ny=23, dx=0.10, dy=0.16):
        return TerrainGrid(
            nx=nx,
            ny=ny,
            dx=dx,
            dy=dy,
            origin_x=-1.0,
            origin_y=-2.0,
            terrain_prim_path="/World/Terrain",
        )

    def test_closed_boundary_matches_explicit_legacy_axis_conversion(self) -> None:
        grid = self._grid()
        yy, xx = np.meshgrid(
            np.arange(grid.ny) * grid.dy,
            np.arange(grid.nx) * grid.dx,
            indexing="ij",
        )
        initial = 1.8 * np.exp(-((xx - 1.9) / 0.32) ** 2 - ((yy - 1.1) / 0.55) ** 2)
        initial[3, 22] += 0.7
        original = initial.copy()
        config = SolverConfig(
            critical_angle_deg=34.0,
            max_iterations=600,
            tolerance=1e-8,
            boundary_condition="closed",
            conservation_tolerance_m3=1e-9,
        )
        result = MinimumSlopeAdapter(grid, config).solve(initial)
        expected_xy, expected_stats = relax_critical_slope(
            initial.T,
            (grid.dx, grid.dy),
            critical_angle_deg=config.critical_angle_deg,
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
        )
        np.testing.assert_allclose(result.heightmap_stable, expected_xy.T, atol=1e-12)
        np.testing.assert_array_equal(initial, original)
        self.assertEqual(result.converged, expected_stats.converged)
        self.assertAlmostEqual(result.volume_before_m3, result.volume_after_m3, 10)
        self.assertEqual(result.boundary_outflow_m3, 0.0)
        self.assertLess(abs(result.diagnostics["volume_balance_error_m3"]), 1e-10)
        self.assertEqual(result.diagnostics["axis_conversion"], "transpose_at_adapter_boundary")

    def test_sequence_contains_real_independent_copies(self) -> None:
        grid = self._grid(nx=17, ny=15, dx=0.1, dy=0.1)
        initial = np.zeros(grid.shape)
        initial[7, 8] = 2.0
        config = SolverConfig(
            critical_angle_deg=30.0,
            max_iterations=300,
            tolerance=1e-7,
            sequence_enabled=True,
            sequence_stride=2,
            sequence_max_frames=16,
            sequence_dtype="float32",
            sequence_memory_limit_mb=8.0,
            boundary_condition="closed",
            conservation_tolerance_m3=1e-9,
        )
        result = MinimumSlopeAdapter(grid, config).solve_sequence(initial)
        self.assertTrue(result.converged)
        self.assertGreater(len(result.heightmap_sequence), 2)
        self.assertTrue(
            all(
                not np.shares_memory(left, right)
                for left, right in zip(
                    result.heightmap_sequence,
                    result.heightmap_sequence[1:],
                )
            )
        )
        np.testing.assert_array_equal(result.heightmap_sequence[0], initial)
        np.testing.assert_allclose(
            result.heightmap_sequence[-1], result.heightmap_stable, atol=1e-6
        )
        self.assertLessEqual(len(result.heightmap_sequence), 16)
        self.assertTrue(
            all(item.dtype == np.float32 for item in result.heightmap_sequence)
        )

    def test_open_boundary_records_outflow_and_balances_volume(self) -> None:
        grid = self._grid(nx=25, ny=21, dx=0.1, dy=0.1)
        initial = np.zeros(grid.shape)
        initial[0:3, 9:13] = 1.5
        config = SolverConfig(
            critical_angle_deg=28.0,
            max_iterations=300,
            tolerance=1e-7,
            sequence_enabled=False,
            sequence_stride=5,
            boundary_condition="open",
            boundary_height_m=0.0,
            conservation_tolerance_m3=1e-8,
        )
        result = MinimumSlopeAdapter(grid, config).solve_sequence(initial)
        self.assertGreater(result.boundary_outflow_m3, 0.0)
        self.assertAlmostEqual(
            result.volume_before_m3,
            result.volume_after_m3 + result.boundary_outflow_m3,
            places=10,
        )
        self.assertTrue(np.all(result.heightmap_stable >= 0.0))
        self.assertEqual(len(result.heightmap_sequence), 2)

    def test_sequence_memory_budget_is_enforced_before_solving(self) -> None:
        grid = self._grid(nx=701, ny=701, dx=0.05, dy=0.05)
        config = SolverConfig(
            sequence_enabled=True,
            sequence_stride=1,
            sequence_max_frames=100,
            sequence_dtype="float64",
            sequence_memory_limit_mb=8.0,
        )
        with self.assertRaisesRegex(ValueError, "exceeds memory budget"):
            MinimumSlopeAdapter(grid, config)

    def test_step_and_reset_have_no_hidden_height_state(self) -> None:
        grid = self._grid(nx=11, ny=11, dx=0.1, dy=0.1)
        initial = np.zeros(grid.shape)
        initial[5, 5] = 1.0
        solver = MinimumSlopeAdapter(
            grid,
            SolverConfig(max_iterations=10, boundary_condition="closed"),
        )
        one = solver.step(initial)
        self.assertEqual(one.iteration_count, 1)
        self.assertIs(solver.last_result, one)
        solver.reset()
        self.assertIsNone(solver.last_result)
        np.testing.assert_array_equal(initial[5, 5], 1.0)


if __name__ == "__main__":
    unittest.main()
