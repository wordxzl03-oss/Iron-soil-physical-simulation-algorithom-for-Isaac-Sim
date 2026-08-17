import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_state import (
    SurfaceTopology,
    TerrainVolumeIntegrator,
)


class PhaseEVolumeIntegratorTests(unittest.TestCase):
    def test_nonplanar_quad_distinguishes_fixed_triangle_from_bilinear(self) -> None:
        # a=1, b=2, c=8, d=4. Current mesh diagonal is a-c.
        height = np.asarray([[1.0, 2.0], [4.0, 8.0]])
        triangle = TerrainVolumeIntegrator(nx=2, ny=2, dx_m=1.0, dy_m=1.0)
        bilinear = TerrainVolumeIntegrator(
            nx=2,
            ny=2,
            dx_m=1.0,
            dy_m=1.0,
            topology=SurfaceTopology.BILINEAR_TRAPEZOID,
        )
        self.assertAlmostEqual(
            triangle.integrate(height),
            (2.0 * 1.0 + 2.0 + 2.0 * 8.0 + 4.0) / 6.0,
        )
        self.assertAlmostEqual(bilinear.integrate(height), 15.0 / 4.0)
        self.assertNotEqual(triangle.integrate(height), bilinear.integrate(height))

    def test_constant_and_planar_fields_are_analytic_at_all_resolutions(self) -> None:
        for nx, ny in ((2, 2), (5, 7), (17, 11), (128, 64)):
            lx, ly = 3.0, 2.0
            dx, dy = lx / (nx - 1), ly / (ny - 1)
            x = np.linspace(0.0, lx, nx)
            y = np.linspace(0.0, ly, ny)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            height = 1.5 + 0.4 * xx + 0.7 * yy
            expected = (
                1.5 * lx * ly
                + 0.4 * lx**2 * ly / 2.0
                + 0.7 * ly**2 * lx / 2.0
            )
            for topology in SurfaceTopology:
                integrator = TerrainVolumeIntegrator(
                    nx=nx,
                    ny=ny,
                    dx_m=dx,
                    dy_m=dy,
                    topology=topology,
                )
                self.assertAlmostEqual(integrator.domain_area_m2, lx * ly, places=12)
                self.assertAlmostEqual(
                    integrator.integrate(height),
                    expected,
                    places=11,
                )

    def test_quadratic_resolution_error_decreases(self) -> None:
        lx, ly = 2.0, 1.0
        exact = lx * ly + ly * lx**3 / 3.0 + lx * ly**3 / 3.0
        errors = []
        for n in (5, 9, 17, 33):
            nx, ny = 2 * n - 1, n
            x = np.linspace(0.0, lx, nx)
            y = np.linspace(0.0, ly, ny)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            height = 1.0 + xx**2 + yy**2
            integrator = TerrainVolumeIntegrator(
                nx=nx,
                ny=ny,
                dx_m=lx / (nx - 1),
                dy_m=ly / (ny - 1),
            )
            errors.append(abs(integrator.integrate(height) - exact))
        self.assertTrue(all(a > b for a, b in zip(errors, errors[1:])), errors)
        self.assertLess(errors[-1], errors[0] / 20.0)

    def test_required_128_256_512_701_resolutions_share_one_integrator(self) -> None:
        side_m = 25.0
        expected = 2.0 * side_m * side_m
        for resolution in (128, 256, 512, 701):
            integrator = TerrainVolumeIntegrator(
                nx=resolution,
                ny=resolution,
                dx_m=side_m / (resolution - 1),
                dy_m=side_m / (resolution - 1),
            )
            height = np.full((resolution, resolution), 2.0, dtype=np.float32)
            self.assertAlmostEqual(
                integrator.integrate(height),
                expected,
                places=9,
            )

    def test_cell_mask_and_signed_delta_use_same_weights(self) -> None:
        mask = np.asarray([[True, False], [False, True]])
        integrator = TerrainVolumeIntegrator(
            nx=3,
            ny=3,
            dx_m=0.5,
            dy_m=2.0,
            valid_cell_mask=mask,
        )
        height = np.ones((3, 3))
        self.assertAlmostEqual(integrator.domain_area_m2, 2.0)
        self.assertAlmostEqual(integrator.integrate(height), 2.0)
        self.assertAlmostEqual(
            integrator.integrate_delta(height, 1.25 * height),
            0.5,
        )

    def test_invalid_shape_nonfinite_and_negative_are_rejected(self) -> None:
        integrator = TerrainVolumeIntegrator(nx=3, ny=3, dx_m=1.0, dy_m=1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            integrator.integrate(np.ones((2, 2)))
        invalid = np.ones((3, 3))
        invalid[1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            integrator.integrate(invalid)
        with self.assertRaisesRegex(ValueError, "negative"):
            integrator.integrate(-np.ones((3, 3)))


if __name__ == "__main__":
    unittest.main()
