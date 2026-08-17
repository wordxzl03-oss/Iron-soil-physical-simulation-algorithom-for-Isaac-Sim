import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import MobileLayerSolver

from isaac_bulk_pipeline.bulk_exchange import (
    AirborneParcelModel,
    BucketRetentionSpillModel,
    DumpTarget,
    TerrainDumpOperator,
)
from isaac_bulk_pipeline.bulk_state import (
    AirborneParcelFootprint,
    BulkStateManager,
    MaterialParcel,
    MaterialScenario,
    PayloadState,
    TerrainState,
    TerrainVolumeIntegrator,
)
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import (
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolState,
)


class PhaseGSpillDumpTests(unittest.TestCase):
    def setUp(self):
        self.grid = TerrainGrid(
            nx=61,
            ny=61,
            dx=0.1,
            dy=0.1,
            origin_x=-3.0,
            origin_y=-3.0,
            terrain_prim_path="/World/TerrainVisual",
        )
        self.integrator = TerrainVolumeIntegrator.from_grid(self.grid)
        self.material = MaterialScenario(
            name="numerical",
            assumed_bulk_density_kg_m3=1800.0,
            internal_friction_angle_deg=32.0,
            cohesion_proxy_pa=0.0,
            tool_friction_coefficient=0.35,
            start_angle_deg=38.0,
            stop_angle_deg=30.0,
            mobile_friction_coefficient=0.25,
        )
        cutting = np.asarray([[-1.0, 0, 0], [1.0, 0, 0]])
        bottom = np.asarray([[0, -1, 0], [0, 0, 0]])
        interior = np.asarray([[0, -1, 0], [0, -1, 1], [0, 0, 0]])
        geometry = BucketGeometryDescriptor.from_extruded_profile(
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            interior_profile_local=interior,
            top_edge_local=np.asarray([[-1, -1, 1], [1, -1, 1]]),
            rated_capacity_m3=1.0,
            geometry_source=GeometrySource.EXPLICIT_PROFILE,
            geometry_quality=GeometryQuality.REDUCED_ORDER,
        )
        self.descriptor = ToolDescriptor(
            tool_type="bucket",
            tool_frame_prim="/World/Bucket/ToolFrame",
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            left_boundary_local=geometry.left_side_wall_local,
            right_boundary_local=geometry.right_side_wall_local,
            interior_profile_local=interior,
            nominal_width_m=2.0,
            nominal_capacity_m3=1.0,
            proxy_level="L1",
            actual_proxy_type="ExtrudedProfileBucket_L1",
            bucket_geometry=geometry,
        )

    def tool(self, curl_deg, z=1.0, velocity=(0.0, 0.0, 0.0)):
        angle = np.deg2rad(curl_deg)
        rotation = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[2, 3] = z

        def transform(points):
            homogeneous = np.column_stack([points, np.ones(len(points))])
            return (pose @ homogeneous.T).T[:, :3]

        return ToolState(
            timestamp=0.0,
            pose_world=pose,
            pose_terrain=pose,
            cutting_edge_terrain=transform(self.descriptor.cutting_edge_local),
            bottom_profile_terrain=transform(self.descriptor.bottom_profile_local),
            left_boundary_terrain=transform(self.descriptor.left_boundary_local),
            right_boundary_terrain=transform(self.descriptor.right_boundary_local),
            linear_velocity=np.asarray(velocity),
            angular_velocity=np.zeros(3),
        )

    def manager(self, payload_volume=0.8, boundary="closed"):
        H = np.zeros(self.grid.shape)
        payload = PayloadState(
            volume_m3=payload_volume,
            capacity_m3=1.0,
            assumed_bulk_density_kg_m3=self.material.assumed_bulk_density_kg_m3,
            center_of_mass_bucket_frame_m=np.asarray([0.0, -0.5, 0.35]),
        )
        state = TerrainState(
            H_resting_m=H,
            mobile_height_m=np.zeros_like(H),
            mobile_momentum_m2_s=np.zeros(H.shape + (2,)),
            payload=payload,
            airborne_parcels=(),
            material=self.material,
            outflow_volume_m3=0.0,
            timestamp_s=0.0,
            action_index=0,
        )
        return BulkStateManager(state, self.integrator, boundary_condition=boundary)

    def test_retention_changes_continuously_with_bucket_orientation(self):
        payload = self.manager().snapshot().payload
        model = BucketRetentionSpillModel()
        retained = []
        for angle in (-60, 0, 30):
            result = model.evaluate(
                payload,
                self.descriptor,
                self.tool(angle),
                material_stop_angle_deg=self.material.stop_angle_deg,
            )
            retained.append(result.retainable_volume_m3)
        self.assertLess(retained[0], retained[1])
        self.assertLess(retained[1], retained[2])
        self.assertEqual(retained[0], 0.0)
        self.assertAlmostEqual(retained[2], 1.0)

    def test_effective_gravity_changes_retainable_volume(self):
        payload = self.manager().snapshot().payload
        model = BucketRetentionSpillModel()
        static = model.evaluate(
            payload,
            self.descriptor,
            self.tool(0),
            material_stop_angle_deg=self.material.stop_angle_deg,
        )
        accelerated = model.evaluate(
            payload,
            self.descriptor,
            self.tool(0),
            material_stop_angle_deg=self.material.stop_angle_deg,
            bucket_linear_acceleration_terrain_m_s2=np.asarray([0.0, 4.0, 0.0]),
        )
        self.assertNotAlmostEqual(
            static.retainable_volume_m3,
            accelerated.retainable_volume_m3,
        )

    def test_parcel_split_and_ballistic_landing_preserve_volume(self):
        model = AirborneParcelModel()
        parcels = model.create_from_bucket_release(
            0.23,
            self.material,
            self.tool(-60, z=1.2),
            self.descriptor,
            timestamp_s=0.0,
            source="test",
            id_prefix="parcel",
        )
        self.assertAlmostEqual(sum(item.volume_m3 for item in parcels), 0.23)
        mobile = np.zeros(self.grid.shape)
        momentum = np.zeros(self.grid.shape + (2,))
        remaining = parcels
        landed = 0.0
        for _ in range(30):
            result = model.advance(
                remaining,
                np.zeros(self.grid.shape),
                mobile,
                momentum,
                self.grid,
                self.integrator,
                0.1,
            )
            remaining = result.remaining_parcels
            mobile = result.mobile_height_m
            momentum = result.mobile_momentum_m2_s
            landed += result.landed_volume_m3
            if not remaining:
                break
        self.assertEqual(len(remaining), 0)
        self.assertAlmostEqual(landed, 0.23)
        self.assertAlmostEqual(self.integrator.integrate(mobile), 0.23)

    def test_finite_footprint_rasterizes_over_exact_control_areas(self):
        footprint = AirborneParcelFootprint(
            lateral_axis_world_m=np.asarray([1.0, 0.0, 0.0]),
            lateral_extent_m=0.42,
            longitudinal_extent_m=0.28,
            provenance="CONSERVATION_BASED_ENGINEERING_CLOSURE",
        )
        parcel = MaterialParcel(
            "finite_support", 0.037, self.material.assumed_bulk_density_kg_m3,
            np.asarray([0.0, 0.0, 0.0]), np.asarray([0.3, -0.1, -1.0]),
            0.0, "test", footprint,
        )
        record = AirborneParcelModel.rasterize_landing(
            parcel, parcel.position_world_m, parcel.velocity_world_m_s,
            self.grid, self.integrator,
        )
        self.assertGreater(record.recipient_cell_count, 1)
        self.assertTrue(np.all(record.delivered_volumes_m3 >= 0.0))
        self.assertAlmostEqual(np.sum(record.delivered_volumes_m3), 0.037, places=12)
        rows, cols = np.divmod(record.recipient_flat_indices, self.grid.nx)
        delta_h = record.delivered_volumes_m3 / record.recipient_control_areas_m2
        height = np.zeros(self.grid.shape)
        height[rows, cols] += delta_h
        self.assertAlmostEqual(self.integrator.integrate(height), 0.037, places=12)

    def test_finite_footprint_uses_boundary_half_and_quarter_control_areas(self):
        footprint = AirborneParcelFootprint(
            lateral_axis_world_m=np.asarray([1.0, 0.0, 0.0]),
            lateral_extent_m=0.18,
            longitudinal_extent_m=0.18,
            provenance="CONSERVATION_BASED_ENGINEERING_CLOSURE",
        )
        corner_world = self.grid.terrain_to_world(np.asarray([-3.0, -3.0, 0.0]))
        parcel = MaterialParcel(
            "boundary_support", 0.01, self.material.assumed_bulk_density_kg_m3,
            corner_world, np.asarray([0.0, 0.0, -1.0]), 0.0, "test", footprint,
        )
        record = AirborneParcelModel.rasterize_landing(
            parcel, corner_world, parcel.velocity_world_m_s, self.grid, self.integrator,
        )
        expected = self.integrator.vertex_weights_m2.ravel()[record.recipient_flat_indices]
        np.testing.assert_allclose(record.recipient_control_areas_m2, expected, atol=0.0)
        self.assertAlmostEqual(np.sum(record.delivered_volumes_m3), 0.01, places=12)

    def test_stop_angle_cone_source_profile_is_compact_nonuniform_and_conservative(self):
        model = AirborneParcelModel(
            descriptor=self.descriptor, material=self.material
        )
        footprint = model.footprint_from_geometry(
            0.035,
            self.descriptor,
            np.asarray([1.0, 0.0, 0.0]),
            self.material,
        )
        self.assertEqual(
            footprint.mass_profile, "ELLIPTIC_CONE_STOP_ANGLE_CLOSURE"
        )
        parcel = MaterialParcel(
            "cone_source", 0.035, self.material.assumed_bulk_density_kg_m3,
            np.asarray([0.0, 0.0, 0.0]), np.asarray([0.8, 0.0, -10.0]),
            0.0, "test", footprint,
        )
        record = model.rasterize_landing(
            parcel, parcel.position_world_m, parcel.velocity_world_m_s,
            self.grid, self.integrator,
        )
        dh = record.delivered_volumes_m3 / record.recipient_control_areas_m2
        self.assertGreater(record.recipient_cell_count, 10)
        self.assertGreater(float(np.max(dh)), float(np.min(dh)))
        self.assertLess(float(np.max(dh)), 0.5)
        self.assertAlmostEqual(np.sum(record.delivered_volumes_m3), 0.035, places=12)

    def test_airborne_landing_can_mobilize_and_conservatively_cross_cells(self):
        """New Airborne Mobile is not shielded by Resting ownership semantics."""

        model = AirborneParcelModel(
            descriptor=self.descriptor, material=self.material
        )
        footprint = model.footprint_from_geometry(
            0.035,
            self.descriptor,
            np.asarray([1.0, 0.0, 0.0]),
            self.material,
        )
        parcel = MaterialParcel(
            "airborne_dynamic", 0.035, self.material.assumed_bulk_density_kg_m3,
            np.asarray([0.0, 0.0, 0.0]), np.asarray([0.8, 0.2, -10.0]),
            0.0, "test", footprint,
        )
        landing = model.advance(
            (parcel,), np.zeros(self.grid.shape), np.zeros(self.grid.shape),
            np.zeros(self.grid.shape + (2,)), self.grid, self.integrator, 0.01,
        )
        solver = MobileLayerSolver()
        result = solver.step(
            np.zeros(self.grid.shape), landing.mobile_height_m,
            landing.mobile_momentum_m2_s, self.material, self.grid,
            self.integrator, 1.0 / 60.0,
        )
        self.assertGreater(result.maximum_speed_m_s, 0.0)
        self.assertGreater(float(np.sum(solver.last_conservative_export_m3)), 0.0)
        self.assertAlmostEqual(result.volume_before_m3, 0.035, places=12)
        self.assertAlmostEqual(result.volume_after_m3, 0.035, places=12)

    def test_terrain_dump_full_cycle_closes_ledger(self):
        manager = self.manager()
        operator = TerrainDumpOperator()
        release = operator.release(
            manager,
            self.tool(-60, z=1.2),
            self.descriptor,
            target=DumpTarget.TERRAIN,
        )
        self.assertAlmostEqual(release.released_volume_m3, 0.8)
        self.assertEqual(release.state.payload.volume_m3, 0.0)
        self.assertGreater(release.created_parcel_count, 0)
        for _ in range(40):
            advance = operator.advance_airborne(
                manager, self.grid, self.integrator, 0.1
            )
            if not advance.state.airborne_parcels and self.integrator.integrate(
                advance.state.mobile_height_m
            ) < 1e-8:
                break
        final = manager.snapshot()
        self.assertEqual(len(final.airborne_parcels), 0)
        self.assertAlmostEqual(final.payload.volume_m3, 0.0)
        resting_volume = self.integrator.integrate(final.H_resting_m)
        mobile_volume = self.integrator.integrate(final.mobile_height_m)
        self.assertGreater(resting_volume, 0.7)
        self.assertAlmostEqual(resting_volume + mobile_volume, 0.8, places=8)
        self.assertLess(manager.ledger_snapshot().balance.absolute_volume_error_m3, 1e-9)

    def test_export_target_accounts_outflow_without_parcels(self):
        manager = self.manager(boundary="open")
        result = TerrainDumpOperator().release(
            manager,
            self.tool(0),
            self.descriptor,
            target=DumpTarget.RECEIVER,
        )
        self.assertEqual(result.created_parcel_count, 0)
        self.assertAlmostEqual(result.exported_volume_m3, 0.8)
        self.assertAlmostEqual(result.state.outflow_volume_m3, 0.8)
        self.assertLess(manager.ledger_snapshot().balance.absolute_volume_error_m3, 1e-9)

    def test_post_dump_minislope_hook_is_strictly_volume_checked(self):
        manager = self.manager()
        operator = TerrainDumpOperator()
        operator.release(
            manager,
            self.tool(-60, z=1.2),
            self.descriptor,
            target=DumpTarget.TERRAIN,
        )
        result = operator.advance_airborne(
            manager,
            self.grid,
            self.integrator,
            0.1,
            resting_relaxation=lambda height: (height.copy(), 0.0),
        )
        self.assertAlmostEqual(
            result.minislope_volume_before_m3,
            result.minislope_volume_after_m3,
            places=12,
        )
        self.assertEqual(result.minislope_boundary_outflow_m3, 0.0)


if __name__ == "__main__":
    unittest.main()
