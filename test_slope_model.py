import unittest

import numpy as np

from slope_model import (
    fractal_perlin_noise,
    max_neighbor_slope,
    random_pile,
    relax_critical_slope,
    relax_hysteretic_avalanche,
    relax_localized_failure_wedge,
    scoop_loader_bucket,
    scoop_ellipsoid,
)


class CriticalSlopeTests(unittest.TestCase):
    def test_relaxation_conserves_volume_and_limits_slope(self):
        h = np.zeros((31, 31))
        h[15, 15] = 10.0
        relaxed, stats = relax_critical_slope(h, (0.1, 0.1), 30.0)
        self.assertTrue(stats.converged)
        self.assertAlmostEqual(h.sum(), relaxed.sum(), places=10)
        self.assertLessEqual(
            max_neighbor_slope(relaxed, (0.1, 0.1)),
            np.tan(np.deg2rad(30.0)) + 1e-7,
        )
        self.assertGreaterEqual(relaxed.min(), 0.0)

    def test_stable_plane_is_unchanged(self):
        x = np.arange(20)[:, None] * 0.02
        h = np.broadcast_to(x, (20, 17)).copy()
        relaxed, _ = relax_critical_slope(h, (0.1, 0.1), 30.0)
        np.testing.assert_allclose(relaxed, h)

    def test_scoop_reports_volume(self):
        h = np.ones((40, 40))
        result, volume = scoop_ellipsoid(
            h, (2.0, 2.0), (0.1, 0.1), (0.5, 0.4), 0.3
        )
        self.assertAlmostEqual(volume, float((h - result).sum() * 0.01))
        self.assertTrue(np.all(result >= 0))

    def test_random_pile_is_repeatable_and_bounded(self):
        first = random_pile(41, 41, 0.15, 0.15, seed=12)
        second = random_pile(41, 41, 0.15, 0.15, seed=12)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(first.max()), 2.0)
        self.assertGreaterEqual(float(first.min()), 0.0)

    def test_fractal_noise_and_piles_vary_across_seeds(self):
        noise = fractal_perlin_noise(51, 47, seed=9)
        self.assertEqual(noise.shape, (51, 47))
        self.assertLessEqual(float(np.max(np.abs(noise))), 1.0 + 1e-12)
        piles = [
            random_pile(61, 61, 0.12, 0.12, seed=seed)
            for seed in (2, 3, 4)
        ]
        shape_differences = [
            float(np.max(np.abs(piles[0] - pile))) for pile in piles[1:]
        ]
        self.assertTrue(all(difference > 0.25 for difference in shape_differences))
        self.assertTrue(all(np.count_nonzero(pile == 0.0) > 0 for pile in piles))

    def test_loader_scoop_conserves_reported_removal(self):
        h = np.ones((50, 50))
        result, volume, removed = scoop_loader_bucket(
            h, (2.5, 1.0), (0.1, 0.1), heading_deg=13
        )
        self.assertAlmostEqual(volume, float(removed.sum() * 0.01))
        self.assertAlmostEqual(volume, float((h - result).sum() * 0.01))
        self.assertTrue(np.all(result >= 0))

    def test_hysteresis_preserves_metastable_face(self):
        # 50 degree plane: above the 38 degree repose angle, below 60 degree
        # start angle. Strong cohesion keeps it metastable.
        x = np.arange(25)[:, None] * 0.1 * np.tan(np.deg2rad(50.0))
        h = np.broadcast_to(x, (25, 11)).copy()
        relaxed, stats, _, _ = relax_hysteretic_avalanche(
            h, (0.1, 0.1), cohesion_pa=1e7, stochasticity=0.0
        )
        self.assertFalse(stats.triggered)
        np.testing.assert_allclose(relaxed, h)

    def test_hysteretic_avalanche_conserves_mass_and_settles(self):
        h = np.zeros((41, 31))
        h[:20, :] = 4.0
        relaxed, stats, damage, history = relax_hysteretic_avalanche(
            h,
            (0.2, 0.2),
            start_angle_deg=60.0,
            stop_angle_deg=38.0,
            cohesion_pa=500.0,
            stochasticity=0.0,
            record_history=True,
        )
        self.assertTrue(stats.triggered)
        self.assertTrue(stats.converged)
        self.assertAlmostEqual(stats.volume_before, stats.volume_after, places=9)
        self.assertGreater(stats.moved_volume, 0.0)
        self.assertGreater(len(history), 2)
        self.assertEqual(damage.shape, h.shape)
        self.assertLessEqual(
            max_neighbor_slope(relaxed, (0.2, 0.2)),
            np.tan(np.deg2rad(38.0)) + 1e-5,
        )

    def test_failure_mask_preserves_unexposed_region(self):
        h = np.zeros((31, 21))
        h[:15] = 3.0
        mask = np.zeros_like(h, dtype=bool)
        mask[:22] = True
        relaxed, stats, _, _ = relax_hysteretic_avalanche(
            h,
            (0.2, 0.2),
            cohesion_pa=0.0,
            stochasticity=0.0,
            failure_mask=mask,
        )
        self.assertTrue(stats.triggered)
        np.testing.assert_array_equal(relaxed[22:], h[22:])

    def test_local_failure_is_bounded_conservative_and_layer_limited(self):
        coordinates = np.linspace(-8, 8, 65)
        _, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
        h = np.clip((yy + 5) * np.tan(np.deg2rad(42.0)), 0, 6)
        result, stats, disturbance, safety, mask, history = (
            relax_localized_failure_wedge(
                h,
                (0.25, 0.25),
                (0.0, -4.8),
                origin_xy=(-8.0, -8.0),
                collision_force_n=200_000,
                collision_displacement_m=0.5,
                cohesion_pa=5_000,
                max_propagation_radius_m=4.0,
                active_layer_depth_m=0.45,
                record_history=True,
            )
        )
        self.assertTrue(stats.triggered)
        self.assertAlmostEqual(
            stats.volume_before_m3, stats.volume_after_m3, places=10
        )
        self.assertLessEqual(stats.max_propagation_distance_m, 4.0 + 1e-12)
        self.assertLessEqual(stats.max_mobilized_depth_m, 0.45 + 1e-12)
        self.assertLessEqual(stats.energy_used_j, stats.collision_energy_j + 1e-9)
        np.testing.assert_array_equal(result[~mask], h[~mask])
        self.assertGreater(len(history), 1)
        self.assertEqual(disturbance.shape, h.shape)
        self.assertEqual(safety.shape, h.shape)


if __name__ == "__main__":
    unittest.main()
