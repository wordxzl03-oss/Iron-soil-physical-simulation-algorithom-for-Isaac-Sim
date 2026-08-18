import inspect
import unittest
from dataclasses import replace

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import FailureZoneModel, ToolTerrainIntersectionModel
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.interaction.continuous_sweep import SweepResult
from isaac_bulk_pipeline.soil_force import (
    IsaacSoilForceAdapter,
    MobileMomentumBudget,
    SoilForceConfig,
    SoilForceModel,
)
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptor, ToolState


class _FakeRigidPrim:
    def __init__(self):
        self.calls = []

    def apply_forces_and_torques_at_pos(self, **kwargs):
        self.calls.append(kwargs)


class PhaseHSoilForceTests(unittest.TestCase):
    def setUp(self):
        self.grid = TerrainGrid(61, 61, 0.1, 0.1, -3, -3, "/World/Terrain")
        self.integrator = TerrainVolumeIntegrator.from_grid(self.grid)
        self.material = MaterialScenario(
            "numerical",
            1800.0,
            32.0,
            500.0,
            0.35,
            38.0,
            30.0,
            0.35,
        )
        self.descriptor = ToolDescriptor(
            tool_type="bucket",
            tool_frame_prim="/World/Bucket/ToolFrame",
            cutting_edge_local=np.asarray([[-1.0, 0, 0], [1.0, 0, 0]]),
            bottom_profile_local=np.asarray([[0, -1, 0], [0, 0, 0]]),
            left_boundary_local=np.asarray([[-1, -1, 0], [-1, 0, 0]]),
            right_boundary_local=np.asarray([[1, -1, 0], [1, 0, 0]]),
            interior_profile_local=np.asarray([[0, -1, 0], [0, -1, 1], [0, 0, 0]]),
            nominal_width_m=2.0,
            nominal_capacity_m3=1.0,
        )

    def tool(self, speed=1.0):
        pose = np.eye(4)
        # local +Y and cutting motion point along terrain +X
        pose[:3, :3] = np.asarray([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        pose[2, 3] = 0.8

        def tx(points):
            h = np.column_stack([points, np.ones(len(points))])
            return (pose @ h.T).T[:, :3]

        return ToolState(
            0.0,
            pose,
            pose,
            tx(self.descriptor.cutting_edge_local),
            tx(self.descriptor.bottom_profile_local),
            tx(self.descriptor.left_boundary_local),
            tx(self.descriptor.right_boundary_local),
            np.asarray([speed, 0.0, 0.0]),
            np.zeros(3),
        )

    def interaction(self, depth=0.2, half_width=0.9, speed=1.0, material=None):
        H = np.ones(self.grid.shape)
        rows, cols = np.indices(self.grid.shape)
        x = self.grid.origin_x + cols * self.grid.dx
        y = self.grid.origin_y + rows * self.grid.dy
        mask = (np.abs(x) <= 0.08) & (np.abs(y) <= half_width)
        surface = np.full(self.grid.shape, np.inf)
        surface[mask] = 1.0 - depth
        sweep = SweepResult((20, 20, 41, 41), mask, surface, (np.eye(4),))
        tool = self.tool(speed)
        intersection = ToolTerrainIntersectionModel().compute(
            H, sweep, tool, self.grid, self.integrator
        )
        failure = FailureZoneModel().compute(
            intersection,
            H,
            material or self.material,
            self.grid,
            self.integrator,
            fallback_approach_direction_xy=np.asarray([1.0, 0.0]),
        )
        return intersection, failure, tool

    def force(self, depth=0.2, half_width=0.9, speed=1.0, material=None, config=None):
        selected_material = material or self.material
        intersection, failure, tool = self.interaction(
            depth,
            half_width,
            speed,
            selected_material,
        )
        return SoilForceModel(config).compute(
            failure,
            intersection,
            selected_material,
            self.descriptor,
            tool,
        )

    def test_penetration_width_and_cohesion_increase_quasi_static_resistance(self):
        shallow = self.force(depth=0.1)
        deep = self.force(depth=0.3)
        narrow = self.force(half_width=0.4)
        wide = self.force(half_width=1.2)
        slow = self.force(speed=0.5)
        fast = self.force(speed=2.0)
        cohesive = MaterialScenario(
            **{
                **self.material.__dict__,
                "name": "cohesive",
                "cohesion_proxy_pa": 5000.0,
            }
        )
        high_cohesion = self.force(material=cohesive)
        self.assertGreater(deep.cutting_resistance_n, shallow.cutting_resistance_n)
        self.assertGreater(wide.cutting_resistance_n, narrow.cutting_resistance_n)
        self.assertAlmostEqual(fast.cutting_resistance_n, slow.cutting_resistance_n)
        self.assertGreater(high_cohesion.cutting_resistance_n, self.force().cutting_resistance_n)

    def test_force_opposes_cut_and_uses_cutting_edge_strip_resultants(self):
        intersection, failure, tool = self.interaction()
        result = SoilForceModel().compute(
            failure, intersection, self.material, self.descriptor, tool
        )
        self.assertLess(np.dot(result.force_terrain_n, np.asarray([1.0, 0.0, 0.0])), 0.0)
        self.assertFalse(result.shared_failure_zone_centroid)
        self.assertEqual(result.application_point_model, "CUTTING_EDGE_STRIP_RESULTANT")
        self.assertEqual(len(result.strip_results), len(failure.strip_geometries))
        for force_strip, geometry in zip(result.strip_results, failure.strip_geometries):
            self.assertEqual(force_strip.strip_index, geometry.strip_index)
            self.assertAlmostEqual(force_strip.failure_angle_deg, geometry.failure_angle_deg)
            self.assertAlmostEqual(force_strip.depth_m, geometry.penetration_depth_m)
            self.assertAlmostEqual(force_strip.width_m, geometry.width_m)
            self.assertTrue(np.allclose(
                force_strip.application_point_terrain_m,
                geometry.effective_cutting_edge_terrain_m,
            ))
        reconstructed_torque = (
            np.cross(
                result.application_point_terrain_m - tool.pose_terrain[:3, 3],
                result.force_terrain_n,
            )
            + result.residual_couple_terrain_nm
        )
        self.assertTrue(np.allclose(
            result.torque_about_tool_origin_terrain_nm, reconstructed_torque
        ))

    def test_static_wedge_at_zero_speed_retains_quasi_static_force(self):
        result = self.force(speed=0.0)
        self.assertGreater(result.quasi_static_resultant_force_n, 0.0)
        self.assertEqual(result.active_momentum_resultant_force_n, 0.0)

    def test_zero_penetration_has_zero_quasi_static_force(self):
        result = self.force(depth=0.0, speed=0.0)
        self.assertEqual(result.quasi_static_resultant_force_n, 0.0)
        self.assertEqual(len(result.strip_results), 0)

    def test_motion_opposite_one_sided_separation_plate_has_no_failure_wedge(self):
        intersection, _, tool = self.interaction(depth=0.2, speed=1.0)
        reverse = replace(
            intersection,
            separation_plane_direction_terrain=np.asarray([-1.0, 0.0, 0.1]),
        )
        zone = FailureZoneModel().compute(
            reverse,
            np.ones(self.grid.shape),
            self.material,
            self.grid,
            self.integrator,
            fallback_approach_direction_xy=np.asarray([1.0, 0.0]),
        )
        self.assertEqual(zone.active_volume_m3, 0.0)
        self.assertEqual(len(zone.strip_geometries), 0)
        self.assertEqual(zone.applicability_status, "REVERSE_SEPARATION_INACTIVE")

    def test_nonfinite_fee_domain_is_reported_without_inventing_a_wedge(self):
        intersection, _, _ = self.interaction(depth=0.2, speed=1.0)
        steep_back_face = replace(
            intersection,
            separation_plane_direction_terrain=np.asarray([1.0, 0.0, -1.0]),
        )
        zone = FailureZoneModel().compute(
            steep_back_face,
            np.ones(self.grid.shape),
            self.material,
            self.grid,
            self.integrator,
            fallback_approach_direction_xy=np.asarray([1.0, 0.0]),
        )
        self.assertEqual(zone.active_volume_m3, 0.0)
        self.assertEqual(zone.applicability_status, "OUTSIDE_FEE_DOMAIN")
        self.assertGreater(zone.excluded_strip_count, 0)
        self.assertTrue(zone.exclusion_diagnostics)

    def test_active_momentum_force_uses_measured_impulse_and_action_reaction(self):
        intersection, failure, tool = self.interaction(speed=0.0)
        budget = MobileMomentumBudget(
            momentum_before_terrain_kg_m_s=np.zeros(3),
            momentum_after_terrain_kg_m_s=np.asarray([10.0, 0.0, 0.0]),
            gravity_pressure_impulse_terrain_ns=np.zeros(3),
            basal_friction_impulse_terrain_ns=np.zeros(3),
            numerical_dissipative_impulse_terrain_ns=np.zeros(3),
            tool_impulse_on_mobile_terrain_ns=np.asarray([10.0, 0.0, 0.0]),
            integration_window_s=0.5,
        )
        result = SoilForceModel().compute(
            failure, intersection, self.material, self.descriptor, tool, budget
        )
        self.assertTrue(np.allclose(result.active_momentum_force_terrain_n, [-20, 0, 0]))
        self.assertTrue(np.allclose(budget.balance_residual_terrain_ns, 0.0))
        self.assertTrue(np.allclose(budget.action_reaction_residual_terrain_ns, 0.0))

    def test_no_arbitrary_inertial_coefficient_and_finite_resultants(self):
        self.assertNotIn("inertial_coefficient", SoilForceConfig.__dataclass_fields__)
        result = self.force()
        self.assertTrue(np.all(np.isfinite(result.force_terrain_n)))
        self.assertTrue(np.all(np.isfinite(result.torque_about_tool_origin_terrain_nm)))
        self.assertGreaterEqual(result.cutting_resistance_n, 0.0)
        self.assertGreaterEqual(result.normal_resistance_n, 0.0)
        self.assertTrue(all(strip.inertial_component_n == 0.0 for strip in result.strip_results))

    def test_force_limit_is_explicit_not_infinite(self):
        result = self.force(
            depth=0.4,
            speed=3.0,
            config=SoilForceConfig(maximum_resultant_force_n=10_000.0),
        )
        self.assertTrue(result.force_was_limited)
        self.assertAlmostEqual(result.resultant_force_n, 10_000.0)
        self.assertGreater(result.unclipped_resultant_force_n, result.resultant_force_n)
        self.assertTrue(np.allclose(
            sum(
                (strip.force_terrain_n for strip in result.strip_results),
                start=np.zeros(3),
            ),
            result.quasi_static_force_terrain_n,
        ))

    def test_force_limit_never_scales_conservative_mobile_reaction(self):
        intersection, failure, tool = self.interaction(depth=0.4, speed=0.0)
        budget = MobileMomentumBudget(
            momentum_before_terrain_kg_m_s=np.zeros(3),
            momentum_after_terrain_kg_m_s=np.asarray([20_000.0, 0.0, 0.0]),
            gravity_pressure_impulse_terrain_ns=np.zeros(3),
            basal_friction_impulse_terrain_ns=np.zeros(3),
            numerical_dissipative_impulse_terrain_ns=np.zeros(3),
            tool_impulse_on_mobile_terrain_ns=np.asarray([20_000.0, 0.0, 0.0]),
            integration_window_s=0.01,
        )
        result = SoilForceModel(
            SoilForceConfig(maximum_resultant_force_n=10_000.0)
        ).compute(
            failure, intersection, self.material, self.descriptor, tool, budget
        )
        self.assertTrue(result.force_was_limited)
        self.assertAlmostEqual(result.active_momentum_resultant_force_n, 2_000_000.0)
        self.assertGreater(result.resultant_force_n, 1_990_000.0)
        self.assertTrue(np.allclose(budget.action_reaction_residual_terrain_ns, 0.0))
        self.assertTrue(np.all(np.isfinite(result.active_momentum_force_terrain_n)))

    def test_public_isaac_force_adapter_uses_array_api(self):
        fake = _FakeRigidPrim()
        result = self.force()
        IsaacSoilForceAdapter(fake).apply(result, np.eye(4))
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(fake.calls[0]["forces"].shape, (1, 3))
        self.assertEqual(fake.calls[0]["positions"].shape, (1, 3))
        self.assertEqual(fake.calls[0]["torques"].shape, (1, 3))
        self.assertTrue(fake.calls[0]["is_global"])

    def test_isaac_adapter_source_avoids_private_physx_interface(self):
        source = inspect.getsource(IsaacSoilForceAdapter)
        self.assertIn("apply_forces_and_torques_at_pos", source)
        self.assertNotIn("get_physx_simulation_interface", source)
        self.assertNotIn("acquire_physx", source)


if __name__ == "__main__":
    unittest.main()
