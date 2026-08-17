import unittest

import numpy as np

from isaac_bulk_pipeline.contact import build_contact_mesh
from isaac_bulk_pipeline.terrain import TerrainGrid


class PhaseCTriangleMeshTests(unittest.TestCase):
    def make_grid(self, *, nx: int = 8, ny: int = 6, valid_mask=None) -> TerrainGrid:
        return TerrainGrid(
            nx=nx,
            ny=ny,
            dx=0.05,
            dy=0.05,
            origin_x=-0.2,
            origin_y=-0.1,
            terrain_prim_path="/World/TerrainVisual",
            valid_mask=valid_mask,
        )

    def test_downsample_is_independent_and_retains_boundaries(self) -> None:
        grid = self.make_grid()
        height = np.arange(grid.nx * grid.ny, dtype=np.float64).reshape(grid.shape)
        mesh = build_contact_mesh(height, grid, target_spacing_m=0.10)
        np.testing.assert_array_equal(mesh.sampled_rows, [0, 2, 4, 5])
        np.testing.assert_array_equal(mesh.sampled_columns, [0, 2, 4, 6, 7])
        self.assertEqual(mesh.sampled_shape_yx, (4, 5))
        self.assertEqual(mesh.vertex_count, 20)
        self.assertEqual(mesh.triangle_count, 24)
        np.testing.assert_allclose(mesh.points_m[0], [-0.2, -0.1, 0.0])
        np.testing.assert_allclose(mesh.points_m[-1], [0.15, 0.15, 47.0])
        self.assertFalse(np.shares_memory(mesh.points_m, height))

    def test_triangle_winding_points_up(self) -> None:
        grid = self.make_grid(nx=5, ny=5)
        mesh = build_contact_mesh(np.zeros(grid.shape), grid, target_spacing_m=0.10)
        triangle = mesh.points_m[mesh.face_vertex_indices[:3]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        self.assertGreater(float(normal[2]), 0.0)
        self.assertTrue(np.all(mesh.face_vertex_counts == 3))

    def test_non_rectangular_mask_omits_invalid_quads(self) -> None:
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        grid = self.make_grid(nx=5, ny=5, valid_mask=mask)
        mesh = build_contact_mesh(np.zeros(grid.shape), grid, target_spacing_m=0.05)
        # Four quads touch the invalid vertex, each contributing two triangles.
        self.assertEqual(mesh.triangle_count, 2 * 4 * 4 - 8)

    def test_contact_buffers_are_read_only(self) -> None:
        grid = self.make_grid(nx=4, ny=4)
        mesh = build_contact_mesh(np.zeros(grid.shape), grid, target_spacing_m=0.10)
        with self.assertRaises(ValueError):
            mesh.points_m[0, 2] = 5.0


if __name__ == "__main__":
    unittest.main()
