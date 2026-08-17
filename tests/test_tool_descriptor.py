import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.config import ToolConfig
from isaac_bulk_pipeline.tools import ToolDescriptorLoader


class ToolDescriptorTests(unittest.TestCase):
    def test_small_medium_large_change_geometry_only_through_config(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        expected = {
            "bucket_small.yaml": (2.0, 1.55),
            "bucket_medium.yaml": (3.2, 2.4),
            "bucket_large.yaml": (4.5, 3.1),
        }
        radii = []
        for filename, (width, depth) in expected.items():
            with self.subTest(filename=filename):
                config = ToolDescriptorLoader.load_config(
                    repository_root / "configs" / filename
                )
                descriptor = ToolDescriptorLoader.load(config)
                self.assertAlmostEqual(descriptor.nominal_width_m, width)
                self.assertAlmostEqual(
                    np.linalg.norm(
                        descriptor.cutting_edge_local[-1]
                        - descriptor.cutting_edge_local[0]
                    ),
                    width,
                )
                self.assertAlmostEqual(
                    -float(descriptor.bottom_profile_local[0, 1]), depth
                )
                radii.append(descriptor.proxy_radius_m)
        self.assertTrue(np.all(np.diff(radii) > 0.0))

    def test_parameter_frame_maps_existing_bucket_axes_explicitly(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config = ToolDescriptorLoader.load_config(
            repository_root / "configs" / "bucket_medium.yaml"
        )
        descriptor = ToolDescriptorLoader.load(config)
        origin_in_link = descriptor.tool_to_link_matrix @ np.array([0, 0, 0, 1])
        left_in_link = descriptor.tool_to_link_matrix @ np.array([-1.6, 0, 0, 1])
        right_in_link = descriptor.tool_to_link_matrix @ np.array([1.6, 0, 0, 1])
        np.testing.assert_allclose(origin_in_link[:3], [1.30, 0.0, 0.0])
        np.testing.assert_allclose(left_in_link[:3], [1.30, 1.60, 0.0])
        np.testing.assert_allclose(right_in_link[:3], [1.30, -1.60, 0.0])

    def test_file_json_and_npz_roundtrip(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        descriptor = ToolDescriptorLoader.load(
            ToolDescriptorLoader.load_config(
                repository_root / "configs" / "bucket_medium.yaml"
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            for suffix in ("json", "npz"):
                with self.subTest(suffix=suffix):
                    path = Path(directory) / f"bucket.{suffix}"
                    ToolDescriptorLoader.save(descriptor, path)
                    config = ToolConfig(
                        descriptor_source="file",
                        tool_frame_prim=descriptor.tool_frame_prim,
                        descriptor_path=path,
                        proxy_level="L1",
                        actual_proxy_type="ExtrudedProfileBucket_L1",
                    )
                    loaded = ToolDescriptorLoader.load(config)
                    np.testing.assert_array_equal(
                        loaded.cutting_edge_local, descriptor.cutting_edge_local
                    )
                    np.testing.assert_array_equal(
                        loaded.tool_to_link_matrix, descriptor.tool_to_link_matrix
                    )
                    self.assertEqual(loaded.nominal_width_m, 3.2)
                    self.assertEqual(
                        loaded.actual_proxy_type, "ExtrudedProfileBucket_L1"
                    )


if __name__ == "__main__":
    unittest.main()
