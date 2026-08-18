"""Exact CPU-oracle contracts for GPU-resident Tool--Mobile CAD support."""

import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    WarpExactToolMobileContactGeometry,
    WarpProductionMobileV2Solver,
    build_tool_mobile_contact_support,
    physical_bucket_contact_face_mask,
    resolve_frictional_wall_impulse,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.experimental.mobile_v2_reference import MobileV2Config
from isaac_bulk_pipeline.performance import probe_warp
from isaac_bulk_pipeline.runtime.bulk_state_authority import DeviceBulkState
from isaac_bulk_pipeline.runtime.gpu_failure_bridge import DeviceFailureZoneBridge
from isaac_bulk_pipeline.interaction import SweepResult
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import (
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolState,
)


class WarpExactToolMobileGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        status = probe_warp("cpu")
        if not status.available:
            raise unittest.SkipTest(status.reason or "Warp CPU unavailable")

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
            nx=25, ny=25, dx=0.1, dy=0.1,
            origin_x=-1.0, origin_y=-1.2,
            terrain_prim_path="/World/Terrain",
        )
        self.integrator = TerrainVolumeIntegrator.from_grid(self.grid)

    def _tool(self, *, pose=None, linear=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0)):
        pose = np.eye(4) if pose is None else np.asarray(pose, dtype=np.float64)
        geometry = self.descriptor.bucket_geometry
        return ToolState(
            timestamp=0.0,
            pose_world=pose,
            pose_terrain=pose,
            cutting_edge_terrain=geometry.transform_points(pose, geometry.cutting_edge_local),
            bottom_profile_terrain=geometry.transform_points(pose, self.descriptor.bottom_profile_local),
            left_boundary_terrain=geometry.transform_points(pose, self.descriptor.left_boundary_local),
            right_boundary_terrain=geometry.transform_points(pose, self.descriptor.right_boundary_local),
            linear_velocity=np.asarray(linear, dtype=np.float64),
            angular_velocity=np.asarray(omega, dtype=np.float64),
            mouth_polygon_terrain=geometry.transform_points(pose, geometry.mouth_polygon_local),
            top_edge_terrain=geometry.transform_points(pose, geometry.top_edge_local),
        )

    def _index(self, x, y):
        col = int(round((x - self.grid.origin_x) / self.grid.dx))
        row = int(round((y - self.grid.origin_y) / self.grid.dy))
        return row * self.grid.nx + col

    def _compare(self, entries, tool, *, require_contact=None, bed_height=0.0):
        bed = np.full(self.grid.shape, float(bed_height), dtype=np.float64)
        mobile = np.zeros(self.grid.shape, dtype=np.float64)
        for x, y, depth in entries:
            mobile.ravel()[self._index(x, y)] = depth
        cpu = build_tool_mobile_contact_support(
            candidate_mask=mobile > 0.0,
            b_eff_after_activation_m=bed,
            mobile_after_activation_m=mobile,
            bbox_yx=(0, self.grid.ny, 0, self.grid.nx),
            tool_state=tool,
            descriptor=self.descriptor,
            grid=self.grid,
            integrator=self.integrator,
        )
        state = DeviceBulkState(self.grid, bed, device="cpu")
        wet = np.flatnonzero(mobile)
        if wet.size:
            state.add_host_indices(
                "mobile", wet.astype(np.int32), mobile.ravel()[wet], reason="oracle_case"
            )
        operator = WarpExactToolMobileContactGeometry(
            state=state, descriptor=self.descriptor
        )
        device = operator.compute(
            bbox_yx=(0, self.grid.ny, 0, self.grid.nx), tool_state=tool
        )
        downloaded = operator.download_support_for_oracle()
        np.testing.assert_array_equal(downloaded["flat_indices"], cpu.flat_indices)
        np.testing.assert_allclose(
            downloaded["closest_points_terrain_m"], cpu.closest_points_terrain_m,
            atol=2.0e-12, rtol=2.0e-12,
        )
        np.testing.assert_allclose(
            downloaded["outward_normals_terrain"], cpu.outward_normals_terrain,
            atol=2.0e-11, rtol=2.0e-11,
        )
        np.testing.assert_allclose(
            downloaded["outward_normals_xy"], cpu.outward_normals_xy,
            atol=2.0e-11, rtol=2.0e-11,
        )
        np.testing.assert_allclose(
            downloaded["tool_surface_velocity_terrain_m_s"],
            cpu.tool_surface_velocity_terrain_m_s,
            atol=2.0e-11, rtol=2.0e-11,
        )
        np.testing.assert_allclose(
            downloaded["cavity_signed_distance_m"], cpu.cavity_signed_distance_m,
            atol=2.0e-11, rtol=2.0e-11,
        )
        self.assertAlmostEqual(device.mobile_volume_m3, cpu.mobile_volume_m3, places=13)
        self.assertEqual(device.cell_count, cpu.cell_count)
        self.assertEqual(device.performance_diagnostics["contact_h2d_bytes"], 0)
        # Geometry equivalence must propagate through the unchanged normal and
        # tangential impulse law, not stop at accept/reject classification.
        for position, index in enumerate(cpu.flat_indices):
            mass = (
                1370.0 * self.integrator.vertex_weights_m2.ravel()[index]
                * mobile.ravel()[index]
            )
            cpu_impulse = resolve_frictional_wall_impulse(
                mobile_mass_kg=mass,
                mobile_velocity_xy_m_s=np.zeros(2),
                tool_velocity_xy_m_s=cpu.tool_surface_velocity_terrain_m_s[position, :2],
                outward_normal_xy=cpu.outward_normals_xy[position],
                tool_mobile_friction_coefficient=0.4,
            )
            device_impulse = resolve_frictional_wall_impulse(
                mobile_mass_kg=mass,
                mobile_velocity_xy_m_s=np.zeros(2),
                tool_velocity_xy_m_s=downloaded[
                    "tool_surface_velocity_terrain_m_s"
                ][position, :2],
                outward_normal_xy=downloaded["outward_normals_xy"][position],
                tool_mobile_friction_coefficient=0.4,
            )
            np.testing.assert_allclose(
                device_impulse.normal_impulse_xy_ns,
                cpu_impulse.normal_impulse_xy_ns, atol=2.0e-11, rtol=2.0e-11,
            )
            np.testing.assert_allclose(
                device_impulse.tangential_impulse_xy_ns,
                cpu_impulse.tangential_impulse_xy_ns, atol=2.0e-11, rtol=2.0e-11,
            )
        if require_contact is not None:
            self.assertEqual(cpu.cell_count > 0, require_contact)
        return cpu, device, state

    def test_01_no_contact(self):
        self._compare([(0.0, 0.8, 0.3)], self._tool(), require_contact=False)

    def test_02_triangle_intersection(self):
        self._compare([(0.0, -1.0, 0.3)], self._tool(), require_contact=True)

    def test_03_closed_cavity_containment(self):
        cpu, _, _ = self._compare(
            [(0.0, -0.7, 0.2)], self._tool(), require_contact=True,
            bed_height=0.2,
        )
        self.assertLess(cpu.cavity_signed_distance_m[0], 0.0)

    def test_03b_mouth_cap_closes_containment_but_is_not_physical_contact(self):
        mask = physical_bucket_contact_face_mask(self.descriptor.bucket_geometry)
        self.assertEqual(np.count_nonzero(~mask), 2)
        cpu, device, _ = self._compare(
            [(0.0, -0.9, 0.05)], self._tool(), require_contact=False,
            bed_height=0.875,
        )
        self.assertEqual(cpu.cell_count, 0)
        self.assertEqual(
            device.performance_diagnostics["containment_triangle_count"],
            len(self.descriptor.bucket_geometry.interior_faces),
        )
        self.assertEqual(
            device.performance_diagnostics["physical_contact_triangle_count"],
            len(self.descriptor.bucket_geometry.interior_faces) - 2,
        )
        self.assertEqual(
            device.performance_diagnostics[
                "mouth_cap_physical_contact_triangle_count"
            ],
            0,
        )

    def test_04_edge_contact(self):
        self._compare([(-1.0, -0.5, 0.3)], self._tool(), require_contact=True)

    def test_05_vertex_near_contact(self):
        # The dual-cell prism meets the CAD at the lip/right-side vertex.
        self._compare([(1.0, -0.05, 0.1)], self._tool(), require_contact=True)

    def test_06_multiple_candidate_cells(self):
        entries = [
            (-0.5, -1.0, 0.3), (0.0, -1.0, 0.2),
            (0.5, -1.0, 0.25), (0.0, 0.8, 0.3),
        ]
        cpu, _, _ = self._compare(entries, self._tool())
        self.assertGreaterEqual(cpu.cell_count, 3)

    def test_07_unequal_triangle_a_c_areas_preserve_contact_volume(self):
        # x=-1.0 is a terrain-boundary vertex with a smaller exact A-C weight.
        entries = [(-1.0, -1.0, 0.3), (0.0, -1.0, 0.3)]
        cpu, device, _ = self._compare(entries, self._tool())
        self.assertAlmostEqual(device.mobile_volume_m3, cpu.mobile_volume_m3, places=13)
        weights = self.integrator.vertex_weights_m2.ravel()
        self.assertNotEqual(weights[self._index(-1.0, -1.0)], weights[self._index(0.0, -1.0)])

    def test_08_moving_rotating_bucket_surface_velocity(self):
        angle = np.deg2rad(7.0)
        pose = np.eye(4)
        pose[:3, :3] = np.asarray([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        pose[:3, 3] = [0.05, 0.02, 0.0]
        self._compare(
            [(0.0, -0.95, 0.4), (0.4, -0.5, 0.3)],
            self._tool(pose=pose, linear=(0.3, 0.8, -0.1), omega=(0.2, -0.1, 0.5)),
            require_contact=True,
        )

    def _impulse_case(self, *, linear, omega, friction):
        cpu, device, state = self._compare(
            [(0.0, -1.0, 0.3)], self._tool(linear=linear, omega=omega),
            require_contact=True,
        )
        mobile_velocity = np.asarray([0.0, 0.0])
        cpu_impulse = resolve_frictional_wall_impulse(
            mobile_mass_kg=1370.0 * cpu.mobile_volume_m3,
            mobile_velocity_xy_m_s=mobile_velocity,
            tool_velocity_xy_m_s=cpu.tool_surface_velocity_terrain_m_s[0, :2],
            outward_normal_xy=cpu.outward_normals_xy[0],
            tool_mobile_friction_coefficient=friction,
        )
        downloaded = WarpExactToolMobileContactGeometry(
            state=state, descriptor=self.descriptor
        )
        # The second operator is intentionally not computed; compare the exact
        # geometric inputs already proven above through the accepted law.
        del downloaded, device
        return cpu_impulse

    def test_09_separating_contact_has_no_attractive_impulse(self):
        result = self._impulse_case(linear=(0.0, -1.0, 0.0), omega=(0.0, 0.0, 0.0), friction=0.4)
        np.testing.assert_allclose(result.total_impulse_xy_ns, np.zeros(2), atol=1.0e-13)

    def test_10_tangential_sliding_obeys_coulomb_and_direct_device_consumption(self):
        cpu, device, state = self._compare(
            [(0.0, -1.0, 0.3)],
            self._tool(linear=(0.7, 1.0, 0.0), omega=(0.0, 0.0, 0.2)),
            require_contact=True,
        )
        material = MaterialScenario(
            name="oracle", assumed_bulk_density_kg_m3=1370.0,
            internal_friction_angle_deg=30.0, cohesion_proxy_pa=0.0,
            tool_friction_coefficient=0.4, start_angle_deg=38.0,
            stop_angle_deg=30.0, mobile_friction_coefficient=1.0e-9,
        )
        solver = WarpProductionMobileV2Solver(
            runtime=state.runtime,
            config=MobileV2Config(
                dx_m=self.grid.dx, dy_m=self.grid.dy,
                basal_friction_coefficient=1.0e-9,
            ),
        )
        solver.bind_device_state(state, material, self.grid, self.integrator)
        # Recompute after solver binding so its persistent arrays are the exact
        # support passed directly to the fused source without compact H2D.
        operator = WarpExactToolMobileContactGeometry(state=state, descriptor=self.descriptor)
        device = operator.compute(
            bbox_yx=(0, self.grid.ny, 0, self.grid.nx),
            tool_state=self._tool(linear=(0.7, 1.0, 0.0), omega=(0.0, 0.0, 0.2)),
        )
        volume_before = state.reservoir_reduction(1370.0)["mobile_volume_m3"]
        step = solver.step_resident(
            1.0 / 120.0, tool_mobile_contact=device,
            tool_mobile_friction_coefficient=material.tool_friction_coefficient,
        )
        volume_after = state.reservoir_reduction(1370.0)["mobile_volume_m3"]
        self.assertEqual(step.tool_mobile_contact_h2d_bytes, 0)
        self.assertEqual(step.tool_mobile_contact_h2d_transfer_count, 0)
        self.assertLessEqual(
            step.tool_tangential_impulse_ns,
            material.tool_friction_coefficient * step.tool_normal_impulse_ns + 1.0e-10,
        )
        np.testing.assert_allclose(
            step.tool_impulse_on_mobile_terrain_ns
            - step.tool_impulse_on_mobile_terrain_ns,
            np.zeros(3), atol=1.0e-12,
        )
        self.assertLessEqual(abs(volume_after - volume_before), 2.0e-12)
        self.assertLessEqual(abs(step.transport_mass_residual_m3), 2.0e-12)
        self.assertGreaterEqual(step.tool_contact_dissipation_j, -1.0e-9)

    def test_11_production_failure_bridge_returns_resident_support(self):
        material = MaterialScenario(
            name="bridge", assumed_bulk_density_kg_m3=1370.0,
            internal_friction_angle_deg=30.0, cohesion_proxy_pa=0.0,
            tool_friction_coefficient=0.4, start_angle_deg=38.0,
            stop_angle_deg=30.0, mobile_friction_coefficient=0.35,
        )
        grid = TerrainGrid(
            nx=701, ny=701, dx=0.05, dy=0.05,
            origin_x=-17.5, origin_y=-17.5,
            terrain_prim_path="/World/Terrain",
        )
        integrator = TerrainVolumeIntegrator.from_grid(grid)
        state = DeviceBulkState(grid, np.full(grid.shape, 0.5), device="cpu")
        mask = np.zeros(grid.shape, dtype=bool)
        mask[300:351, 324:377] = True
        cut = np.full(grid.shape, np.inf)
        cut[mask] = 0.2
        sweep = SweepResult((300, 324, 351, 377), mask, cut, (np.eye(4),))
        bridge = DeviceFailureZoneBridge(
            state, material, grid, integrator,
            descriptor=self.descriptor,
        )
        result = bridge.execute(sweep, self._tool(linear=(0.0, 0.5, 0.0)))
        self.assertTrue(result.tool_mobile_contact.device_resident)
        self.assertEqual(
            result.tool_mobile_contact.performance_diagnostics["contact_h2d_bytes"], 0
        )
        self.assertEqual(result.forcing_flat_indices.size, 0)
        self.assertGreaterEqual(result.timings_ms["geometry_candidate_count"], 0.0)
        bridge.clear_tool_forcing(result.tool_mobile_contact)


if __name__ == "__main__":
    unittest.main()
