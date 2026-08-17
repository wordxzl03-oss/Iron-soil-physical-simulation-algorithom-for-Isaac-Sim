import unittest
from pathlib import Path

import numpy as np

from render_obj_loader_scooping import (
    _prepare_model,
    ballistic_particle,
    infer_landmarks,
    load_obj_parts,
    make_flatter_stockpile,
    make_payload_particles,
    partition_payload_volume,
    rotate_yz,
    transform_point,
)
from render_multi_obj_loader_scooping import (
    cut_state_offset,
    sample_trajectories,
)


class ObjLoaderModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = load_obj_parts(Path("wheel_buck.obj"))
        cls.raw = raw
        cls.parts, cls.points = _prepare_model(raw, infer_landmarks(raw))

    def test_required_components_are_separate(self):
        self.assertEqual(
            set(self.raw),
            {
                "loader_frame_world.stl",
                "loader_boom_world.stl",
                "loader_bucket_world.stl",
            },
        )
        for part in self.raw.values():
            self.assertGreater(len(part.vertices), 100)
            self.assertGreater(len(part.faces), 100)

    def test_inferred_pins_match_adjacent_parts(self):
        root = self.points["root_pin"]
        pin = self.points["bucket_pin"]
        edge = self.points["cutting_edge"]
        self.assertGreater(root[2], pin[2])
        self.assertGreater(pin[1], root[1])
        self.assertGreater(edge[1], pin[1])

    def test_curl_is_circular_about_real_bucket_pin(self):
        root = self.points["root_pin"]
        pin = self.points["bucket_pin"]
        edge = self.points["cutting_edge"]
        low_angle = -5.754
        low_pin = transform_point(pin, root, low_angle)
        low_edge = transform_point(edge, root, low_angle)
        radii = []
        heights = []
        for curl in np.linspace(0, 48, 13):
            curled_edge = rotate_yz(low_edge[None, :], low_pin, curl)[0]
            radii.append(np.linalg.norm(curled_edge[1:3] - low_pin[1:3]))
            heights.append(curled_edge[2])
        np.testing.assert_allclose(radii, radii[0], atol=1e-12)
        self.assertGreater(heights[-1], heights[0])

    def test_flatter_stockpile_has_requested_scale_and_slope(self):
        _, _, height = make_flatter_stockpile(
            resolution=0.25, slope_angle_deg=42.0
        )
        self.assertAlmostEqual(float(height.max()), 10.0)
        center = height.shape[0] // 2
        positive = height[center] > 0
        profile = height[center, positive]
        slopes = np.abs(np.diff(profile)) / 0.25
        nonzero = slopes[slopes > 1e-8]
        self.assertLessEqual(
            float(nonzero.max()), np.tan(np.deg2rad(42.0)) + 1e-10
        )

    def test_payload_particles_fit_bucket_width(self):
        pin = self.points["bucket_pin"]
        width = float(
            np.ptp(self.parts["loader_bucket_world.stl"].vertices[:, 0])
        )
        particles = make_payload_particles(pin, width, count=250, seed=4)
        self.assertEqual(particles.shape, (250, 3))
        self.assertLessEqual(float(np.abs(particles[:, 0]).max()), 0.43 * width)

    def test_spill_ballistics_and_volume_accounting(self):
        airborne, landed = ballistic_particle(
            np.array([0.0, 0.0, 2.0]),
            0.2,
            np.array([0.0, -0.2, 0.0]),
            0.0,
        )
        self.assertFalse(landed)
        self.assertLess(airborne[2], 2.0)
        retained, spilled = partition_payload_volume(1.8, 180, 18)
        self.assertAlmostEqual(retained + spilled, 1.8)
        self.assertAlmostEqual(spilled, 0.18)

    def test_multi_trajectory_sampling_is_repeatable_and_varied(self):
        first = sample_trajectories(4, 2.7, seed=2042)
        second = sample_trajectories(4, 2.7, seed=2042)
        self.assertEqual(first, second)
        offsets = [record["lateral_offset_m"] for record in first]
        self.assertEqual(offsets, sorted(offsets))
        self.assertGreater(max(offsets) - min(offsets), 5.0)
        self.assertGreater(
            max(record["penetration_m"] for record in first)
            - min(record["penetration_m"] for record in first),
            0.1,
        )

    def test_offset_cuts_affect_different_tracks(self):
        xx, yy, height = make_flatter_stockpile(resolution=0.25)
        _, left = cut_state_offset(
            height, xx, yy, 1.0, -2.5, -11.15, 1.3, 2.7, 1.0
        )
        _, right = cut_state_offset(
            height, xx, yy, 1.0, 2.5, -11.15, 1.3, 2.7, 1.0
        )
        left_center = float((xx * left).sum() / left.sum())
        right_center = float((xx * right).sum() / right.sum())
        self.assertLess(left_center, -2.0)
        self.assertGreater(right_center, 2.0)


if __name__ == "__main__":
    unittest.main()
