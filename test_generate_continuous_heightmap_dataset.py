import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generate_continuous_heightmap_dataset import (
    ACTION_FIELDS,
    generate_continuous_dataset,
)


class ContinuousHeightmapDatasetTests(unittest.TestCase):
    def test_sequence_is_continuous_valid_and_reproducible(self):
        common = {
            "sequence_count": 1,
            "scoops": 6,
            "seed": 81,
            "workspace_size": 8.0,
            "pile_span": 6.0,
            "grid_spacing": 0.20,
            "peak_height_min": 1.4,
            "peak_height_max": 1.7,
            "repose_angle_min": 33.0,
            "repose_angle_max": 37.0,
            "bulk_density": 1800.0,
            "reference_video": None,
        }
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_args = argparse.Namespace(output_dir=Path(first_dir), **common)
            second_args = argparse.Namespace(output_dir=Path(second_dir), **common)
            first_manifest = generate_continuous_dataset(first_args)
            second_manifest = generate_continuous_dataset(second_args)

            self.assertTrue(first_manifest["all_sequences_valid"])
            self.assertTrue(first_manifest["closed_footprint_required"])
            self.assertTrue(first_manifest["sequences"][0]["all_states_closed"])
            initial_peak = first_manifest["sequences"][0]["pile"][
                "initial_peak_height_m"
            ]
            self.assertGreaterEqual(initial_peak, common["peak_height_min"] - 1e-9)
            self.assertLessEqual(initial_peak, common["peak_height_max"] + 1e-9)
            with np.load(
                Path(first_dir) / "continuous_heightmap_dataset.npz",
                allow_pickle=False,
            ) as first, np.load(
                Path(second_dir) / "continuous_heightmap_dataset.npz",
                allow_pickle=False,
            ) as second:
                self.assertEqual(first["heightmaps_m"].shape, (1, 7, 41, 41))
                self.assertEqual(first["actions"].shape, (1, 6, len(ACTION_FIELDS)))
                self.assertEqual(first["active_masks"].shape, (1, 7, 41, 41))
                np.testing.assert_array_equal(
                    first["heightmaps_m"], second["heightmaps_m"]
                )
                np.testing.assert_allclose(
                    first["delta_heightmaps_m"],
                    np.diff(first["heightmaps_m"], axis=1),
                    atol=1e-6,
                )
                state_volumes = first["heightmaps_m"].sum(axis=(2, 3))
                self.assertTrue(np.all(np.diff(state_volumes, axis=1) < 0.0))
                self.assertTrue(np.all(first["heightmaps_m"] >= 0.0))
                self.assertLess(float(first["active_masks"][0, 0].mean()), 0.95)
                heightmaps = first["heightmaps_m"][0]
                self.assertTrue(np.all(heightmaps[:, 0, :] == 0.0))
                self.assertTrue(np.all(heightmaps[:, -1, :] == 0.0))
                self.assertTrue(np.all(heightmaps[:, :, 0] == 0.0))
                self.assertTrue(np.all(heightmaps[:, :, -1] == 0.0))
                active = first["active_masks"][0]
                self.assertFalse(np.any(active[:, 0, :]))
                self.assertFalse(np.any(active[:, -1, :]))
                self.assertFalse(np.any(active[:, :, 0]))
                self.assertFalse(np.any(active[:, :, -1]))


if __name__ == "__main__":
    unittest.main()
