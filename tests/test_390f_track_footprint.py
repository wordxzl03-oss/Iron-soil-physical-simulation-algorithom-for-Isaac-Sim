import unittest

import numpy as np

from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.vehicle import TrackFootprintConfig, TrackFootprintRasterizer


class TrackFootprintRasterizerTests(unittest.TestCase):
    def setUp(self):
        transform = np.eye(4)
        transform[:3, 3] = [-5.0, -5.0, 0.0]
        self.grid = TerrainGrid(201, 201, 0.05, 0.05, 0.0, 0.0, "/Terrain", transform)
        self.config = TrackFootprintConfig(length_m=6.0, width_m=0.8)
        self.rasterizer = TrackFootprintRasterizer(self.grid, self.config)

    def test_world_transform_and_area(self):
        result = self.rasterizer.rasterize(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
        self.assertTrue(result.mask.flags.writeable is False)
        self.assertLess(abs(np.count_nonzero(result.mask) * self.grid.cell_area - 4.8), 0.35)
        np.testing.assert_allclose(result.center_terrain_m, [5.0, 5.0, 1.0])
        np.testing.assert_allclose(result.forward_terrain_xy, [1.0, 0.0])

    def test_rotated_footprint_is_not_axis_aligned_bbox(self):
        direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        result = self.rasterizer.rasterize(np.array([0.0, 0.0, 1.0]), direction)
        r0, c0, r1, c1 = result.active_bbox_grid
        self.assertGreater((r1 - r0) * (c1 - c0), np.count_nonzero(result.mask) * 2)
        self.assertLess(abs(np.count_nonzero(result.mask) * self.grid.cell_area - 4.8), 0.35)

    def test_outside_domain_returns_empty_mask(self):
        result = self.rasterizer.rasterize(np.array([-20.0, -20.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.assertFalse(np.any(result.mask))
        self.assertEqual(result.active_bbox_grid, (0, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
