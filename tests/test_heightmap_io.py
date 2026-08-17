import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.terrain import HeightmapIO, TerrainGrid, validate_heightmap


class HeightmapIOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TerrainGrid(
            nx=4,
            ny=3,
            dx=0.5,
            dy=0.25,
            origin_x=-1.0,
            origin_y=2.0,
            terrain_prim_path="/World/Terrain/DynamicSurface",
        )
        # Deliberately asymmetric: detects transpose, mirrors and 90° rotation.
        self.height_yx = np.array(
            [[10.0, 1.0, 2.0, 3.0], [4.0, 5.0, 17.0, 6.0], [7.0, 8.0, 9.0, 0.5]],
            dtype=np.float32,
        )

    def test_npy_roundtrip_preserves_yx_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "height.npy"
            HeightmapIO.save(path, self.height_yx)
            loaded = HeightmapIO.load(path, grid=self.grid)
        np.testing.assert_array_equal(loaded, self.height_yx)
        self.assertEqual(loaded.shape, (3, 4))
        self.assertEqual(loaded[0, 0], 10.0)
        self.assertEqual(loaded[1, 2], 17.0)
        self.assertEqual(loaded[2, 3], 0.5)

    def test_legacy_xy_requires_explicit_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npy"
            np.save(path, self.height_yx.T)
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                HeightmapIO.load(path, grid=self.grid)
            loaded = HeightmapIO.load(
                path, grid=self.grid, source_axis_order="xy"
            )
        np.testing.assert_array_equal(loaded, self.height_yx)

    def test_invalid_values_are_rejected(self) -> None:
        invalid = self.height_yx.astype(float)
        invalid[1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            validate_heightmap(invalid)


if __name__ == "__main__":
    unittest.main()
