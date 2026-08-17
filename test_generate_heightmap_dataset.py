import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_heightmap_dataset import ACTION_FIELDS, generate_dataset


class HeightmapDatasetTests(unittest.TestCase):
    def test_small_dataset_is_valid_and_reproducible(self):
        common = {
            "count": 2,
            "seed": 73,
            "grid_size": 25,
            "grid_spacing": 0.15,
            "peak_height_min": 0.8,
            "peak_height_max": 1.1,
            "repose_angle_min": 30.0,
            "repose_angle_max": 36.0,
            "bulk_density": 1800.0,
        }
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_args = argparse.Namespace(output_dir=Path(first_dir), **common)
            second_args = argparse.Namespace(output_dir=Path(second_dir), **common)
            first_manifest = generate_dataset(first_args)
            second_manifest = generate_dataset(second_args)

            self.assertTrue(first_manifest["all_samples_valid"])
            self.assertEqual(first_manifest["axis_convention"], second_manifest["axis_convention"])
            with np.load(Path(first_dir) / "heightmap_dataset.npz") as first, np.load(
                Path(second_dir) / "heightmap_dataset.npz"
            ) as second:
                self.assertEqual(first["initial_heightmaps_m"].shape, (2, 25, 25))
                self.assertEqual(first["actions"].shape, (2, len(ACTION_FIELDS)))
                np.testing.assert_array_equal(
                    first["initial_heightmaps_m"], second["initial_heightmaps_m"]
                )
                np.testing.assert_array_equal(
                    first["final_heightmaps_m"], second["final_heightmaps_m"]
                )
                np.testing.assert_allclose(
                    first["delta_heightmaps_m"],
                    first["final_heightmaps_m"] - first["initial_heightmaps_m"],
                    atol=1e-6,
                )
                self.assertTrue(np.all(first["initial_heightmaps_m"] >= 0))
                self.assertTrue(np.all(first["final_heightmaps_m"] >= 0))

            parsed = json.loads((Path(first_dir) / "manifest.json").read_text())
            self.assertEqual(parsed["schema_version"], "minslope-heightmap-v1")
            self.assertEqual(len(parsed["samples"]), 2)


if __name__ == "__main__":
    unittest.main()
