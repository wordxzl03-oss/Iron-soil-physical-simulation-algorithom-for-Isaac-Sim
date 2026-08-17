import unittest

import numpy as np

from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.visualization import build_mesh_arrays, compute_vertex_normals


class DynamicMeshArrayTests(unittest.TestCase):
    def test_asymmetric_heightmap_is_not_transposed_or_mirrored(self) -> None:
        grid = TerrainGrid(
            nx=4,
            ny=3,
            dx=0.5,
            dy=0.25,
            origin_x=-1.0,
            origin_y=2.0,
            terrain_prim_path="/World/Terrain/DynamicSurface",
        )
        height = np.array(
            [[10.0, 1.0, 2.0, 3.0], [4.0, 5.0, 17.0, 6.0], [7.0, 8.0, 9.0, 0.5]]
        )
        points, counts, indices = build_mesh_arrays(height, grid)
        np.testing.assert_allclose(points[0], [-1.0, 2.0, 10.0])
        np.testing.assert_allclose(points[1 * grid.nx + 2], [0.0, 2.25, 17.0])
        np.testing.assert_allclose(points[-1], [0.5, 2.5, 0.5])
        self.assertEqual(counts.shape, (2 * 2 * 3,))
        self.assertEqual(indices.shape, (2 * 2 * 3 * 3,))

    def test_flat_mesh_winding_and_normals_point_up(self) -> None:
        grid = TerrainGrid(
            nx=3,
            ny=3,
            dx=1.0,
            dy=1.0,
            origin_x=0.0,
            origin_y=0.0,
            terrain_prim_path="/World/Terrain/DynamicSurface",
        )
        height = np.zeros(grid.shape)
        points, _, indices = build_mesh_arrays(height, grid)
        triangle = points[indices[:3]]
        face_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        self.assertGreater(face_normal[2], 0.0)
        normals = compute_vertex_normals(height, grid)
        np.testing.assert_allclose(normals, np.tile([0.0, 0.0, 1.0], (9, 1)))

    def test_128_256_512_use_same_array_builder(self) -> None:
        physical_span_m = 12.75
        for size in (128, 256, 512):
            with self.subTest(size=size):
                spacing = physical_span_m / (size - 1)
                grid = TerrainGrid(
                    nx=size,
                    ny=size,
                    dx=spacing,
                    dy=spacing,
                    origin_x=-physical_span_m / 2,
                    origin_y=-physical_span_m / 2,
                    terrain_prim_path="/World/Terrain/DynamicSurface",
                )
                height = np.zeros(grid.shape, dtype=np.float32)
                points, counts, indices = build_mesh_arrays(height, grid)
                self.assertEqual(points.shape, (size * size, 3))
                self.assertEqual(counts.shape, (2 * (size - 1) ** 2,))
                self.assertEqual(indices.shape, (6 * (size - 1) ** 2,))
                self.assertAlmostEqual(points[0, 0], -physical_span_m / 2, places=5)
                self.assertAlmostEqual(points[-1, 0], physical_span_m / 2, places=5)


if __name__ == "__main__":
    unittest.main()
