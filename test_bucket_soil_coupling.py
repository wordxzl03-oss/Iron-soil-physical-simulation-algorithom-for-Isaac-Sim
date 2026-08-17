import unittest

import numpy as np

from bucket_soil_coupling import (
    BucketGeometry,
    HeightfieldSoil,
    frame_from_transform,
    generate_realistic_closed_pile,
    heightfield_mesh,
    run_offline_joint_coupled_demo,
)


class BucketSoilCouplingTests(unittest.TestCase):
    def test_realistic_pile_is_irregular_repeatable_and_closed(self):
        first, info = generate_realistic_closed_pile(
            workspace_size_m=18.0,
            spacing_m=0.20,
            seed=17,
            peak_height_m=4.2,
        )
        second, _ = generate_realistic_closed_pile(
            workspace_size_m=18.0,
            spacing_m=0.20,
            seed=17,
            peak_height_m=4.2,
        )
        np.testing.assert_array_equal(first, second)
        self.assertTrue(info["closed_footprint"])
        self.assertEqual(float(first[0].max()), 0.0)
        self.assertEqual(float(first[-1].max()), 0.0)
        self.assertEqual(float(first[:, 0].max()), 0.0)
        self.assertEqual(float(first[:, -1].max()), 0.0)
        self.assertGreater(len(info["old_bucket_scars"]), 1)
        # An elongated irregular ridge must not be radially symmetric.
        self.assertGreater(float(np.mean(np.abs(first - first.T))), 0.08)

    def test_actual_bucket_floor_and_edge_sweep_remove_soil(self):
        height = np.full((101, 101), 2.0)
        height[[0, -1], :] = 0.0
        height[:, [0, -1]] = 0.0
        soil = HeightfieldSoil(height, 0.10)
        transform = np.eye(4)
        transform[:3, 3] = [4.0, 4.0, 0.55]
        first = frame_from_transform(
            transform,
            joint_positions_rad={"lift_joint": -0.1, "bucket_joint": 0.0},
        )
        soil.apply_bucket_frame(first, cutting_enabled=False)
        transform[:3, 3] = [5.5, 4.0, 0.55]
        second = frame_from_transform(
            transform,
            joint_positions_rad={"lift_joint": -0.1, "bucket_joint": 0.2},
            time_s=0.1,
        )
        result = soil.apply_bucket_frame(second, cutting_enabled=True)
        self.assertGreater(result.removed_volume_m3, 0.0)
        self.assertGreater(result.changed_cell_count, 100)
        self.assertEqual(len(soil.event_log), 1)
        self.assertEqual(
            soil.event_log[0]["joint_positions_rad"]["bucket_joint"], 0.2
        )

    def test_six_scoops_are_continuous_and_mesh_topology_is_static(self):
        initial, _ = generate_realistic_closed_pile(
            workspace_size_m=18.0,
            spacing_m=0.20,
            seed=23,
            peak_height_m=4.0,
        )
        states, scoops, events = run_offline_joint_coupled_demo(
            initial, 0.20, scoop_count=6
        )
        self.assertEqual(states.shape, (7, 91, 91))
        self.assertEqual(len(scoops), 6)
        self.assertGreater(len(events), 6)
        volumes = states.sum(axis=(1, 2)) * 0.20**2
        self.assertTrue(np.all(np.diff(volumes) < 0.0))
        self.assertTrue(all(item["removed_volume_m3"] > 0 for item in scoops))
        points, faces = heightfield_mesh(states[0], 0.20)
        self.assertEqual(points.shape, (91 * 91, 3))
        self.assertEqual(faces.shape, (2 * 90 * 90, 3))
        self.assertTrue(np.all(states[:, 0, :] == 0.0))
        self.assertTrue(np.all(states[:, -1, :] == 0.0))


if __name__ == "__main__":
    unittest.main()
