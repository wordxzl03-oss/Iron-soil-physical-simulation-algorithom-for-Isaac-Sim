import unittest

import numpy as np

from isaac_bulk_pipeline.terrain import TerrainGrid


class TerrainGridTests(unittest.TestCase):
    def setUp(self) -> None:
        angle = np.deg2rad(30.0)
        cosine, sine = np.cos(angle), np.sin(angle)
        transform = np.array(
            [
                [cosine, -sine, 0.0, 4.0],
                [sine, cosine, 0.0, -2.0],
                [0.0, 0.0, 1.0, 1.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.grid = TerrainGrid(
            nx=7,
            ny=5,
            dx=0.25,
            dy=0.40,
            origin_x=-0.75,
            origin_y=-0.80,
            terrain_prim_path="/World/Terrain/DynamicSurface",
            terrain_to_world_matrix=transform,
        )

    def test_world_terrain_roundtrip(self) -> None:
        terrain_point = np.array([0.35, -0.10, 2.20])
        world_point = self.grid.terrain_to_world(terrain_point)
        recovered = self.grid.world_to_terrain(world_point)
        np.testing.assert_allclose(recovered, terrain_point, atol=1e-10)

    def test_grid_mapping_uses_row_y_column_x(self) -> None:
        point = self.grid.grid_to_terrain(row=3, column=4, height=1.25)
        np.testing.assert_allclose(point, [0.25, 0.40, 1.25], atol=1e-12)
        row_column = self.grid.terrain_to_grid(point)
        np.testing.assert_allclose(row_column, [3.0, 4.0], atol=1e-12)

    def test_inside_mask_volume_and_shape(self) -> None:
        mask = np.ones((5, 7), dtype=bool)
        mask[1, 2] = False
        grid = TerrainGrid(
            nx=7,
            ny=5,
            dx=0.25,
            dy=0.40,
            origin_x=0.0,
            origin_y=0.0,
            terrain_prim_path="/World/Terrain/DynamicSurface",
            valid_mask=mask,
        )
        height = np.full(grid.shape, 2.0)
        self.assertEqual(grid.shape, (5, 7))
        self.assertAlmostEqual(grid.cell_area, 0.10)
        self.assertFalse(grid.is_inside(1, 2))
        self.assertTrue(grid.is_inside(4, 6))
        self.assertFalse(grid.is_inside(5, 6))
        self.assertAlmostEqual(grid.compute_volume(height), 34 * 2.0 * 0.10)


if __name__ == "__main__":
    unittest.main()
