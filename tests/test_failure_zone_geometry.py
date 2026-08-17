import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    FailureStripGeometry,
    FailureZoneModel,
    ToolTerrainIntersection,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.terrain import TerrainGrid


class FailureZoneGeometryTests(unittest.TestCase):
    @staticmethod
    def scenario(phi=32.0, cohesion=500.0, delta_deg=19.29):
        return MaterialScenario(
            name="failure_zone_UNCALIBRATED",
            assumed_bulk_density_kg_m3=1800.0,
            internal_friction_angle_deg=float(phi),
            cohesion_proxy_pa=float(cohesion),
            tool_friction_coefficient=float(np.tan(np.deg2rad(delta_deg))),
            start_angle_deg=50.0,
            stop_angle_deg=30.0,
            mobile_friction_coefficient=0.35,
        )

    @staticmethod
    def solve(
        *,
        spacing=0.1,
        depth=0.25,
        phi=32.0,
        cohesion=500.0,
        delta_deg=19.29,
        rake_deg=90.0,
        slope_deg=0.0,
        terrain_height=2.0,
    ):
        count = int(round(6.0 / spacing)) + 1
        grid = TerrainGrid(count, count, spacing, spacing, -3.0, -3.0, "/World/Terrain")
        integrator = TerrainVolumeIntegrator.from_grid(grid)
        rows, columns = np.indices(grid.shape)
        x = grid.origin_x + columns * grid.dx
        y = grid.origin_y + rows * grid.dy
        grade = np.tan(np.deg2rad(slope_deg))
        height = terrain_height + grade * x
        mask = (np.abs(x) <= 0.51 * spacing) & (np.abs(y) <= 1.001)
        penetration = np.where(mask, depth, 0.0)
        surface = np.where(mask, height - depth, np.inf)
        normal = np.asarray([-grade, 0.0, 1.0])
        normal /= np.linalg.norm(normal)
        rake = np.deg2rad(rake_deg)
        separation = np.asarray([np.cos(rake), 0.0, np.sin(rake)])
        intersection = ToolTerrainIntersection(
            affected_mask=mask,
            penetration_depth_m=penetration,
            cutting_surface_m=surface,
            bucket_velocity_terrain_m_s=np.asarray([1.0, 0.0, 0.0]),
            cutting_edge_velocity_terrain_m_s=np.asarray([1.0, 0.0, 0.0]),
            local_terrain_normal=normal,
            local_slope_rad=abs(np.deg2rad(slope_deg)),
            candidate_intersection_volume_m3=integrator.integrate(penetration),
            affected_bbox_grid=(0, 0, grid.ny, grid.nx),
            cutting_edge_points_terrain_m=np.asarray(
                [[0.0, -1.0, terrain_height - depth], [0.0, 1.0, terrain_height - depth]]
            ),
            separation_plane_direction_terrain=separation,
        )
        material = FailureZoneGeometryTests.scenario(phi, cohesion, delta_deg)
        failure = FailureZoneModel().compute(
            intersection,
            height,
            material,
            grid,
            integrator,
        )
        return grid, integrator, height, intersection, material, failure

    def test_historical_70_degree_saturation_is_removed(self):
        angles = []
        hits = []
        for phi in (20, 25, 30, 35, 40, 45):
            for cohesion in (0, 500, 2000, 5000):
                *_, failure = self.solve(phi=phi, cohesion=cohesion)
                angles.extend(strip.failure_angle_deg for strip in failure.strip_geometries)
                hits.extend(strip.failure_angle_boundary_hit for strip in failure.strip_geometries)
        self.assertTrue(np.all(np.isfinite(angles)))
        self.assertGreater(np.ptp(angles), 5.0)
        self.assertFalse(np.allclose(angles, 70.0))
        self.assertLess(float(np.mean(hits)), 0.05)

    def test_phi_cohesion_delta_depth_rake_and_slope_change_geometry(self):
        groups = {
            "phi": [self.solve(phi=value)[-1] for value in (20, 25, 30, 35, 40, 45)],
            "cohesion": [self.solve(cohesion=value)[-1] for value in (0, 500, 2000, 5000)],
            "delta": [self.solve(delta_deg=value)[-1] for value in (8, 18, 28)],
            "depth": [self.solve(depth=value)[-1] for value in (0.10, 0.25, 0.45)],
            "rake": [self.solve(rake_deg=value)[-1] for value in (60, 75, 90, 105)],
            "slope": [self.solve(slope_deg=value)[-1] for value in (-10, 0, 10)],
        }
        for name, failures in groups.items():
            beta = np.asarray([item.estimated_failure_angle_deg for item in failures])
            volume = np.asarray([item.analytical_wedge_volume_m3 for item in failures])
            self.assertTrue(np.all(np.isfinite(beta)), name)
            self.assertTrue(np.all(np.isfinite(volume)), name)
            if name == "rake":
                applicable = np.asarray(
                    [item.applicability_status == "APPLICABLE" for item in failures]
                )
                self.assertTrue(np.all(volume[applicable] > 0.0), name)
                self.assertTrue(np.all(volume[~applicable] == 0.0), name)
                self.assertEqual(
                    [item.applicability_status for item in failures if item.applicability_status != "APPLICABLE"],
                    ["REVERSE_SEPARATION_INACTIVE"],
                )
            else:
                self.assertTrue(np.all(volume > 0.0), name)
            self.assertGreater(np.ptp(beta) + np.ptp(volume), 1.0e-6, name)
            self.assertLess(float(np.mean([item.failure_angle_boundary_hit_rate for item in failures])), 0.25, name)
        depth_volumes = [item.analytical_wedge_volume_m3 for item in groups["depth"]]
        self.assertTrue(all(right > left for left, right in zip(depth_volumes, depth_volumes[1:])))

    def test_geometry_raster_and_availability_invariants(self):
        grid, integrator, height, _, _, failure = self.solve(rake_deg=75, slope_deg=10)
        self.assertGreater(len(failure.strip_geometries), 0)
        for strip in failure.strip_geometries:
            self.assertIsInstance(strip, FailureStripGeometry)
            self.assertGreater(strip.penetration_depth_m, 0.0)
            self.assertGreater(strip.longitudinal_extent_m, 0.0)
            self.assertGreaterEqual(strip.wedge_volume_m3, 0.0)
            vertices = strip.cross_section_vertices_terrain_m
            self.assertGreater(np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])), 0.0)
            self.assertAlmostEqual(
                strip.wedge_volume_m3,
                strip.cross_section_area_m2 * strip.width_m,
                places=12,
            )
            self.assertTrue(np.all(strip.raster_coverage_fractions >= 0.0))
            self.assertTrue(np.all(strip.raster_coverage_fractions <= 1.0))
            self.assertTrue(np.all(strip.raster_indices_yx[:, 0] >= 0))
            self.assertTrue(np.all(strip.raster_indices_yx[:, 0] < grid.ny))
            self.assertTrue(np.all(strip.raster_indices_yx[:, 1] >= 0))
            self.assertTrue(np.all(strip.raster_indices_yx[:, 1] < grid.nx))
        self.assertTrue(np.all(failure.active_thickness_m <= height + 1.0e-12))
        self.assertAlmostEqual(
            failure.active_volume_m3,
            integrator.integrate(failure.active_thickness_m),
            places=12,
        )
        self.assertAlmostEqual(
            failure.rasterized_requested_volume_m3,
            failure.analytical_wedge_volume_m3,
            places=10,
        )
        self.assertAlmostEqual(
            failure.active_volume_m3,
            sum(strip.rasterized_activated_volume_m3 for strip in failure.strip_geometries),
            places=10,
        )

        # The low terrain has insufficient material. Clipping must remain
        # explicit, conservative and never dig below zero.
        _, low_integrator, low_height, _, _, clipped = self.solve(
            depth=0.25,
            terrain_height=0.08,
        )
        self.assertTrue(any(strip.availability_clipped for strip in clipped.strip_geometries))
        self.assertTrue(np.all(clipped.active_thickness_m <= low_height + 1.0e-12))
        self.assertLess(clipped.active_volume_m3, clipped.analytical_wedge_volume_m3)
        self.assertAlmostEqual(clipped.active_volume_m3, low_integrator.integrate(clipped.active_thickness_m), places=12)

    def test_failure_activation_is_an_equal_resting_to_mobile_transfer(self):
        _, integrator, resting_before, _, _, failure = self.solve()
        mobile_before = np.zeros_like(resting_before)
        resting_after = resting_before - failure.active_thickness_m
        mobile_after = mobile_before + failure.active_thickness_m
        resting_decrement = -integrator.integrate_delta(resting_before, resting_after)
        mobile_increment = integrator.integrate_delta(mobile_before, mobile_after)
        self.assertAlmostEqual(resting_decrement, failure.active_volume_m3, places=12)
        self.assertAlmostEqual(mobile_increment, failure.active_volume_m3, places=12)
        self.assertAlmostEqual(resting_decrement - mobile_increment, 0.0, places=12)

    def test_conservative_rasterization_converges_with_resolution(self):
        errors = []
        volumes = []
        for spacing in (0.20, 0.10, 0.05):
            *_, failure = self.solve(spacing=spacing, rake_deg=75, slope_deg=7)
            error = abs(failure.active_volume_m3 - failure.analytical_wedge_volume_m3)
            errors.append(error)
            volumes.append(failure.active_volume_m3)
        self.assertLess(max(errors), 1.0e-9)
        self.assertLessEqual(errors[1], errors[0] + 1.0e-12)
        self.assertLessEqual(errors[2], errors[1] + 1.0e-12)
        self.assertLess(max(volumes) - min(volumes), 1.0e-8)


if __name__ == "__main__":
    unittest.main()
