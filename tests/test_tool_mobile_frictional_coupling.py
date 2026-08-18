import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    ToolMobileContactSupport,
    build_tool_mobile_contact_support,
    resolve_frictional_wall_impulse,
)
from isaac_bulk_pipeline.bulk_state import TerrainVolumeIntegrator
from isaac_bulk_pipeline.bulk_state import MaterialScenario
from isaac_bulk_pipeline.bulk_interaction import WarpProductionMobileV2Solver
from isaac_bulk_pipeline.experimental.mobile_v2_reference import MobileV2Config
from isaac_bulk_pipeline.performance import probe_warp
from isaac_bulk_pipeline.runtime.bulk_state_authority import DeviceBulkState
from isaac_bulk_pipeline.soil_force import (
    ToolMobileFrameContractLedger,
    ToolMobileSubstepContract,
)
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import (
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolState,
)


class FrictionalImpulseLawTests(unittest.TestCase):
    def test_moving_flat_wall_pushes_mobile_and_reaction_is_opposite(self):
        result = resolve_frictional_wall_impulse(
            mobile_mass_kg=2.0,
            mobile_velocity_xy_m_s=np.zeros(2),
            tool_velocity_xy_m_s=np.asarray([1.0, 0.0]),
            outward_normal_xy=np.asarray([1.0, 0.0]),
            tool_mobile_friction_coefficient=0.0,
        )
        np.testing.assert_allclose(result.total_impulse_xy_ns, [2.0, 0.0])
        np.testing.assert_allclose(result.machine_reaction_impulse_xy_ns, [-2.0, 0.0])
        self.assertGreaterEqual(result.tool_to_mobile_work_j, result.mobile_kinetic_energy_change_j)

    def test_wall_moving_away_has_no_attractive_impulse(self):
        result = resolve_frictional_wall_impulse(
            mobile_mass_kg=2.0,
            mobile_velocity_xy_m_s=np.zeros(2),
            tool_velocity_xy_m_s=np.asarray([-1.0, 0.0]),
            outward_normal_xy=np.asarray([1.0, 0.0]),
            tool_mobile_friction_coefficient=0.5,
        )
        np.testing.assert_array_equal(result.total_impulse_xy_ns, np.zeros(2))

    def test_coulomb_tangent_opposes_slip_and_is_normal_limited(self):
        result = resolve_frictional_wall_impulse(
            mobile_mass_kg=1.0,
            mobile_velocity_xy_m_s=np.asarray([0.0, 2.0]),
            tool_velocity_xy_m_s=np.asarray([1.0, 0.0]),
            outward_normal_xy=np.asarray([1.0, 0.0]),
            tool_mobile_friction_coefficient=0.5,
        )
        np.testing.assert_allclose(result.normal_impulse_xy_ns, [1.0, 0.0])
        np.testing.assert_allclose(result.tangential_impulse_xy_ns, [0.0, -0.5])
        self.assertLessEqual(
            np.linalg.norm(result.tangential_impulse_xy_ns),
            0.5 * np.linalg.norm(result.normal_impulse_xy_ns) + 1.0e-15,
        )
        self.assertGreater(result.frictional_dissipation_j, 0.0)

    def test_zero_relative_contact_does_not_inject_energy(self):
        result = resolve_frictional_wall_impulse(
            mobile_mass_kg=3.0,
            mobile_velocity_xy_m_s=np.asarray([0.4, -0.2]),
            tool_velocity_xy_m_s=np.asarray([0.4, -0.2]),
            outward_normal_xy=np.asarray([0.0, 1.0]),
            tool_mobile_friction_coefficient=0.7,
        )
        np.testing.assert_array_equal(result.total_impulse_xy_ns, np.zeros(2))
        self.assertEqual(result.mobile_kinetic_energy_change_j, 0.0)

    def test_multi_cell_action_reaction_is_machine_precision(self):
        cases = [
            (1.0, [0.0, 0.4], [1.0, 0.0], [1.0, 0.0]),
            (2.0, [-0.3, 0.0], [0.0, 0.8], [0.0, 1.0]),
            (0.5, [0.1, -0.2], [0.7, 0.1], [1.0, 0.0]),
        ]
        mobile = np.zeros(2)
        machine = np.zeros(2)
        for mass, velocity, tool, normal in cases:
            result = resolve_frictional_wall_impulse(
                mobile_mass_kg=mass,
                mobile_velocity_xy_m_s=np.asarray(velocity),
                tool_velocity_xy_m_s=np.asarray(tool),
                outward_normal_xy=np.asarray(normal),
                tool_mobile_friction_coefficient=0.35,
            )
            mobile += result.total_impulse_xy_ns
            machine += result.machine_reaction_impulse_xy_ns
        np.testing.assert_array_equal(mobile + machine, np.zeros(2))

    def test_multi_substep_machine_frame_sums_impulse_once(self):
        frame = ToolMobileFrameContractLedger()
        for dt, impulse in (
            (0.001, np.asarray([1.0, 2.0])),
            (0.004, np.asarray([-0.5, 0.25])),
            (0.007, np.asarray([3.0, -1.0])),
        ):
            frame.append(
                ToolMobileSubstepContract(
                    accepted_tool_to_mobile_impulse_xy_ns=impulse,
                    measured_mobile_source_delta_xy_ns=impulse,
                    contact_active=True,
                    dt_sub_s=dt,
                )
            )
        np.testing.assert_array_equal(
            frame.machine_reaction_impulse_xy_ns,
            -frame.accepted_tool_to_mobile_impulse_xy_ns,
        )

    def test_unequal_dual_areas_use_physical_mass_not_uniform_cell_area(self):
        density = 1370.0
        # Equal A*h represents equal physical Mobile mass despite unequal A.
        representations = ((0.25, 0.4), (1.0, 0.1))
        impulses = []
        velocity_changes = []
        for area, depth in representations:
            mass = density * area * depth
            result = resolve_frictional_wall_impulse(
                mobile_mass_kg=mass,
                mobile_velocity_xy_m_s=np.zeros(2),
                tool_velocity_xy_m_s=np.asarray([1.0, 0.0]),
                outward_normal_xy=np.asarray([1.0, 0.0]),
                tool_mobile_friction_coefficient=0.0,
            )
            impulses.append(result.total_impulse_xy_ns)
            delta_q = result.total_impulse_xy_ns / (density * area)
            velocity_changes.append(delta_q / depth)
        np.testing.assert_allclose(impulses[0], impulses[1])
        np.testing.assert_allclose(velocity_changes[0], velocity_changes[1])


class ToolContactGeometryTests(unittest.TestCase):
    def setUp(self):
        cutting = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bottom = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
        interior = np.asarray(
            [[0.0, -1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]]
        )
        geometry = BucketGeometryDescriptor.from_extruded_profile(
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            interior_profile_local=interior,
            top_edge_local=np.asarray([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0]]),
            rated_capacity_m3=1.0,
            geometry_source=GeometrySource.EXPLICIT_PROFILE,
            geometry_quality=GeometryQuality.REDUCED_ORDER,
        )
        self.descriptor = ToolDescriptor(
            tool_type="bucket",
            tool_frame_prim="/World/Bucket/ToolFrame",
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            left_boundary_local=np.asarray([[-1.0, -1.0, 0.0], [-1.0, 0.0, 0.0]]),
            right_boundary_local=np.asarray([[1.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
            interior_profile_local=interior,
            nominal_width_m=2.0,
            nominal_capacity_m3=1.0,
            proxy_level="L1",
            actual_proxy_type="ExtrudedProfileBucket_L1",
            bucket_geometry=geometry,
        )
        self.grid = TerrainGrid(
            nx=7, ny=7, dx=0.1, dy=0.1,
            origin_x=-0.3, origin_y=-1.3,
            terrain_prim_path="/World/Terrain",
        )
        self.integrator = TerrainVolumeIntegrator.from_grid(self.grid)

    def test_prism_triangle_intersection_and_rigid_surface_velocity(self):
        pose = np.eye(4)
        geometry = self.descriptor.bucket_geometry
        tool = ToolState(
            timestamp=0.0,
            pose_world=pose,
            pose_terrain=pose,
            cutting_edge_terrain=geometry.transform_points(pose, geometry.cutting_edge_local),
            bottom_profile_terrain=geometry.transform_points(pose, self.descriptor.bottom_profile_local),
            left_boundary_terrain=geometry.transform_points(pose, self.descriptor.left_boundary_local),
            right_boundary_terrain=geometry.transform_points(pose, self.descriptor.right_boundary_local),
            linear_velocity=np.asarray([0.0, 1.0, 0.0]),
            angular_velocity=np.asarray([0.0, 0.0, 0.5]),
            mouth_polygon_terrain=geometry.transform_points(pose, geometry.mouth_polygon_local),
            top_edge_terrain=geometry.transform_points(pose, geometry.top_edge_local),
        )
        candidate = np.zeros((3, 3), dtype=bool)
        candidate[2, 1] = True
        bed = np.zeros((3, 3))
        mobile = np.zeros((3, 3))
        mobile[2, 1] = 0.3
        # Global row 3 / col 2 is x=-0.1, y=-1.0 and its prism intersects the
        # bucket's vertical rear wall at y=-1.0.
        support = build_tool_mobile_contact_support(
            candidate_mask=candidate,
            b_eff_after_activation_m=bed,
            mobile_after_activation_m=mobile,
            bbox_yx=(1, 4, 1, 4),
            tool_state=tool,
            descriptor=self.descriptor,
            grid=self.grid,
            integrator=self.integrator,
        )
        self.assertIsInstance(support, ToolMobileContactSupport)
        self.assertEqual(support.cell_count, 1)
        self.assertGreater(support.mobile_volume_m3, 0.0)
        expected_velocity = tool.linear_velocity + np.cross(
            tool.angular_velocity,
            support.closest_points_terrain_m[0] - tool.pose_terrain[:3, 3],
        )
        np.testing.assert_allclose(
            support.tool_surface_velocity_terrain_m_s[0], expected_velocity
        )
        self.assertAlmostEqual(np.linalg.norm(support.outward_normals_xy[0]), 1.0)
        performance = support.performance_diagnostics
        self.assertEqual(performance["mobile_candidate_count"], 1)
        self.assertEqual(
            performance["cad_triangle_count"],
            len(self.descriptor.bucket_geometry.interior_faces),
        )
        self.assertEqual(
            performance["broadphase_pair_count"],
            performance["mobile_candidate_count"]
            * performance["cad_triangle_count"],
        )
        self.assertGreater(performance["triangle_aabb_test_count"], 0)
        self.assertEqual(
            performance["ray_triangle_test_count"],
            performance["cad_triangle_count"],
        )
        self.assertEqual(
            performance["closest_point_query_count"],
            performance["physical_contact_triangle_count"],
        )
        self.assertEqual(performance["accepted_contact_count"], 1)


class WarpToolContactSourceTests(unittest.TestCase):
    def test_device_source_writes_mobile_impulse_and_exact_reaction_budget(self):
        status = probe_warp("cpu")
        if not status.available:
            self.skipTest(status.reason or "Warp CPU device unavailable")
        grid = TerrainGrid(
            nx=5, ny=5, dx=0.1, dy=0.1,
            origin_x=-0.2, origin_y=-0.2,
            terrain_prim_path="/World/Terrain",
        )
        integrator = TerrainVolumeIntegrator.from_grid(grid)
        state = DeviceBulkState(grid, np.ones(grid.shape), device="cpu")
        state.add_host_indices(
            "mobile",
            np.arange(25, dtype=np.int32),
            np.full(25, 0.1),
            reason="synthetic_uniform_mobile",
        )
        material = MaterialScenario(
            name="synthetic",
            assumed_bulk_density_kg_m3=1370.0,
            internal_friction_angle_deg=30.0,
            cohesion_proxy_pa=0.0,
            tool_friction_coefficient=0.4,
            start_angle_deg=38.0,
            stop_angle_deg=30.0,
            mobile_friction_coefficient=1.0e-9,
        )
        solver = WarpProductionMobileV2Solver(
            runtime=state.runtime,
            config=MobileV2Config(
                dx_m=0.1,
                dy_m=0.1,
                basal_friction_coefficient=1.0e-9,
            ),
        )
        solver.bind_device_state(state, material, grid, integrator)
        center = 12
        contact = ToolMobileContactSupport(
            flat_indices=np.asarray([center], dtype=np.int32),
            closest_points_terrain_m=np.asarray([[0.0, 0.0, 1.05]]),
            outward_normals_terrain=np.asarray([[1.0, 0.0, 0.0]]),
            outward_normals_xy=np.asarray([[1.0, 0.0]]),
            tool_surface_velocity_terrain_m_s=np.asarray([[1.0, 0.5, 0.0]]),
            cavity_signed_distance_m=np.asarray([0.0]),
            mobile_volume_m3=0.001,
            tool_reference_position_terrain_m=np.asarray([0.0, 0.0, 1.0]),
        )
        result = solver.step_resident(
            1.0 / 60.0,
            tool_mobile_contact=contact,
            tool_mobile_friction_coefficient=material.tool_friction_coefficient,
        )
        self.assertGreater(result.tool_normal_impulse_ns, 0.0)
        self.assertGreater(result.tool_tangential_impulse_ns, 0.0)
        self.assertLessEqual(
            result.tool_tangential_impulse_ns,
            material.tool_friction_coefficient * result.tool_normal_impulse_ns + 1.0e-10,
        )
        self.assertGreater(np.linalg.norm(result.tool_impulse_on_mobile_terrain_ns), 0.0)
        self.assertLessEqual(abs(result.transport_mass_residual_m3), 1.0e-12)
        self.assertGreater(result.tool_contact_active_substeps, 0)
        self.assertGreaterEqual(result.tool_contact_dissipation_j, -1.0e-10)


if __name__ == "__main__":
    unittest.main()
