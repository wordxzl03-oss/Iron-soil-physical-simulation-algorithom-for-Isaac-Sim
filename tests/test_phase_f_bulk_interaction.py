import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_interaction import (
    BucketIntakeModel,
    BulkMaterialInteractionModel,
    DepositionOperator,
    FailureZoneModel,
    MobileLayerConfig,
    MobileLayerSolver,
    ToolTerrainIntersectionModel,
)
from isaac_bulk_pipeline.bulk_state import (
    BulkStateManager,
    MaterialScenario,
    PayloadState,
    TerrainState,
    TerrainVolumeIntegrator,
)
from isaac_bulk_pipeline.interaction.continuous_sweep import SweepResult
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import (
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolState,
)


class PhaseFBulkInteractionTests(unittest.TestCase):
    def setUp(self):
        self.grid = TerrainGrid(
            nx=41,
            ny=41,
            dx=0.1,
            dy=0.1,
            origin_x=-2.0,
            origin_y=-2.0,
            terrain_prim_path="/World/TerrainVisual",
        )
        self.integrator = TerrainVolumeIntegrator.from_grid(self.grid)
        self.material = MaterialScenario(
            name="numerical_dry",
            assumed_bulk_density_kg_m3=1800.0,
            internal_friction_angle_deg=32.0,
            cohesion_proxy_pa=500.0,
            tool_friction_coefficient=0.35,
            start_angle_deg=38.0,
            stop_angle_deg=30.0,
            mobile_friction_coefficient=0.25,
        )
        half = 1.0
        cutting = np.asarray([[-half, 0, 0], [half, 0, 0]])
        bottom = np.asarray([[0, -1, 0], [0, 0, 0]])
        interior = np.asarray([[0, -1, 0], [0, -1, 1], [0, 0, 0]])
        geometry = BucketGeometryDescriptor.from_extruded_profile(
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            interior_profile_local=interior,
            top_edge_local=np.asarray([[-half, -1, 1], [half, -1, 1]]),
            rated_capacity_m3=1.0,
            geometry_source=GeometrySource.EXPLICIT_PROFILE,
            geometry_quality=GeometryQuality.REDUCED_ORDER,
        )
        self.descriptor = ToolDescriptor(
            tool_type="bucket",
            tool_frame_prim="/World/Bucket/ToolFrame",
            cutting_edge_local=cutting,
            bottom_profile_local=bottom,
            left_boundary_local=np.asarray([[-half, -1, 0], [-half, 0, 0]]),
            right_boundary_local=np.asarray([[half, -1, 0], [half, 0, 0]]),
            interior_profile_local=interior,
            nominal_width_m=2.0,
            nominal_capacity_m3=1.0,
            proxy_level="L1",
            actual_proxy_type="ExtrudedProfileBucket_L1",
            bucket_geometry=geometry,
        )

    @staticmethod
    def tool_state(*, velocity=(0.0, 1.0, 0.0), z=0.0):
        pose = np.eye(4)
        pose[2, 3] = z
        edge = np.asarray([[-1.0, 0.0, z], [1.0, 0.0, z]])
        return ToolState(
            timestamp=1.0,
            pose_world=pose,
            pose_terrain=pose,
            cutting_edge_terrain=edge,
            bottom_profile_terrain=np.asarray([[0.0, -1.0, z], [0.0, 0.0, z]]),
            left_boundary_terrain=np.asarray([[-1.0, -1.0, z], [-1.0, 0.0, z]]),
            right_boundary_terrain=np.asarray([[1.0, -1.0, z], [1.0, 0.0, z]]),
            linear_velocity=np.asarray(velocity),
            angular_velocity=np.zeros(3),
        )

    def sweep(self, depth=0.2):
        rows, cols = np.indices(self.grid.shape)
        x = self.grid.origin_x + cols * self.grid.dx
        y = self.grid.origin_y + rows * self.grid.dy
        mask = (np.abs(x) <= 0.9) & (np.abs(y) <= 0.08)
        surface = np.full(self.grid.shape, np.inf)
        surface[mask] = 1.0 - depth
        return SweepResult(
            affected_bbox_grid=(19, 11, 22, 30),
            affected_mask=mask,
            cut_surface=surface,
            sampled_poses=(np.eye(4),),
        )

    def test_intersection_is_candidate_only_and_topology_integrated(self):
        H = np.ones(self.grid.shape)
        before = H.copy()
        result = ToolTerrainIntersectionModel().compute(
            H, self.sweep(0.2), self.tool_state(), self.grid, self.integrator
        )
        self.assertTrue(np.array_equal(H, before))
        self.assertGreater(result.candidate_intersection_volume_m3, 0.0)
        self.assertAlmostEqual(
            result.candidate_intersection_volume_m3,
            self.integrator.integrate(result.penetration_depth_m),
        )

    def test_failure_wedge_is_material_dependent_and_not_payload(self):
        H = np.ones(self.grid.shape)
        intersection = ToolTerrainIntersectionModel().compute(
            H, self.sweep(0.25), self.tool_state(), self.grid, self.integrator
        )
        failure = FailureZoneModel().compute(
            intersection, H, self.material, self.grid, self.integrator
        )
        self.assertGreater(failure.active_volume_m3, 0.0)
        self.assertGreater(len(failure.strip_results), 1)
        self.assertAlmostEqual(
            failure.active_volume_m3,
            self.integrator.integrate(failure.active_thickness_m),
        )
        high_cohesion = MaterialScenario(
            **{
                **self.material.__dict__,
                "name": "high_cohesion",
                "cohesion_proxy_pa": 5000.0,
            }
        )
        stronger = FailureZoneModel().compute(
            intersection, H, high_cohesion, self.grid, self.integrator
        )
        self.assertGreater(
            stronger.estimated_total_resistance_n,
            failure.estimated_total_resistance_n,
        )

    def test_mobile_finite_volume_is_positive_closed_and_heading_sensitive(self):
        H = np.zeros(self.grid.shape)
        h = np.zeros(self.grid.shape)
        h[18:23, 18:23] = 0.25
        q = np.zeros(self.grid.shape + (2,))
        q[18:23, 18:23, 0] = h[18:23, 18:23] * 1.0
        solver = MobileLayerSolver(
            MobileLayerConfig(boundary_condition="closed", active_buffer_cells=2)
        )
        before = self.integrator.integrate(h)
        result = solver.step(
            H, h, q, self.material, self.grid, self.integrator, 0.1
        )
        self.assertGreaterEqual(result.minimum_height_m, -1e-12)
        self.assertAlmostEqual(result.volume_after_m3, before, places=10)
        cols = np.indices(self.grid.shape)[1]
        before_center = float(np.sum(cols * h) / np.sum(h))
        after_center = float(np.sum(cols * result.mobile_height_m) / np.sum(result.mobile_height_m))
        self.assertGreater(after_center, before_center)
        delta = (
            result.momentum_after_terrain_kg_m_s
            - result.momentum_before_terrain_kg_m_s
        )
        accounted = (
            result.gravity_pressure_impulse_terrain_ns
            + result.basal_friction_impulse_terrain_ns
            + result.tool_impulse_on_mobile_terrain_ns
            + result.numerical_dissipative_impulse_terrain_ns
        )
        self.assertTrue(np.allclose(delta, accounted, atol=1e-10))

    def test_mobile_tool_impulse_matches_known_mass_velocity_change(self):
        H = np.zeros(self.grid.shape)
        h = np.full(self.grid.shape, 0.1)
        q = np.zeros(self.grid.shape + (2,))
        forcing = np.ones(self.grid.shape, dtype=bool)
        dt = 0.03
        solver = MobileLayerSolver(
            MobileLayerConfig(boundary_condition="closed", active_buffer_cells=0)
        )
        result = solver.step(
            H,
            h,
            q,
            self.material,
            self.grid,
            self.integrator,
            dt,
            tool_forcing_mask=forcing,
            tool_velocity_xy_m_s=np.asarray([1.0, 0.0]),
        )
        mass = (
            self.material.assumed_bulk_density_kg_m3
            * self.integrator.integrate(h)
        )
        # dt / default 0.12 s relaxation = 0.25 velocity increment.
        self.assertAlmostEqual(
            result.tool_impulse_on_mobile_terrain_ns[0], 0.25 * mass, places=8
        )
        self.assertAlmostEqual(result.tool_impulse_on_mobile_terrain_ns[1], 0.0)

    def test_bucket_intake_is_relative_flux_and_capacity_limited(self):
        mobile = np.zeros(self.grid.shape)
        rows = np.arange(self.grid.ny)
        row0 = int(np.argmin(np.abs(self.grid.origin_y + rows * self.grid.dy)))
        mobile[row0 - 1 : row0 + 2, 11:30] = 0.2
        momentum = np.zeros(self.grid.shape + (2,))
        payload = PayloadState(
            volume_m3=0.97,
            capacity_m3=1.0,
            assumed_bulk_density_kg_m3=self.material.assumed_bulk_density_kg_m3,
            center_of_mass_bucket_frame_m=np.zeros(3),
        )
        before_mobile = self.integrator.integrate(mobile)
        result = BucketIntakeModel().apply(
            mobile,
            momentum,
            payload,
            self.tool_state(velocity=(0.0, 1.0, 0.0)),
            self.descriptor,
            self.grid,
            self.integrator,
            0.2,
            terrain_surface_m=np.zeros(self.grid.shape),
        )
        self.assertTrue(result.capacity_limited)
        self.assertAlmostEqual(result.bucket_inflow_volume_m3, 0.03)
        self.assertAlmostEqual(result.payload.volume_m3, 1.0)
        self.assertAlmostEqual(
            before_mobile - self.integrator.integrate(result.mobile_height_m),
            result.bucket_inflow_volume_m3,
            places=10,
        )
        full = BucketIntakeModel().apply(
            mobile,
            momentum,
            result.payload,
            self.tool_state(velocity=(0.0, 1.0, 0.0)),
            self.descriptor,
            self.grid,
            self.integrator,
            0.2,
            terrain_surface_m=np.zeros(self.grid.shape),
        )
        self.assertEqual(full.bucket_inflow_volume_m3, 0.0)
        self.assertAlmostEqual(self.integrator.integrate(full.mobile_height_m), before_mobile)

    def test_deposition_transfers_equal_volume(self):
        H = np.zeros(self.grid.shape)
        mobile = np.zeros(self.grid.shape)
        mobile[15:26, 15:26] = 0.1
        momentum = np.zeros(self.grid.shape + (2,))
        result = DepositionOperator().apply(
            H, mobile, momentum, self.material, self.grid, self.integrator, 0.5
        )
        self.assertGreater(result.deposited_volume_m3, 0.0)
        self.assertAlmostEqual(
            self.integrator.integrate(result.H_resting_m),
            result.deposited_volume_m3,
            places=10,
        )
        self.assertAlmostEqual(
            self.integrator.integrate(mobile),
            self.integrator.integrate(result.mobile_height_m) + result.deposited_volume_m3,
            places=10,
        )

    def test_deposition_uses_free_surface_not_buried_resting_slope(self):
        columns = np.arange(self.grid.nx, dtype=np.float64)[None, :]
        resting = np.repeat(0.08 * columns, self.grid.ny, axis=0)
        # The Resting substrate is steep, but the static Mobile blanket makes
        # the physical free surface horizontal. Relabelling part of that
        # blanket as Resting is therefore physically stable and conservative.
        mobile = np.max(resting) - resting + 0.02
        momentum = np.zeros(self.grid.shape + (2,))
        result = DepositionOperator().apply(
            resting, mobile, momentum, self.material, self.grid,
            self.integrator, 1.0 / 60.0,
        )
        self.assertGreater(result.deposited_volume_m3, 0.0)
        np.testing.assert_allclose(
            result.H_resting_m + result.mobile_height_m,
            resting + mobile,
            atol=1.0e-14,
        )

    def test_subcell_mobile_tail_is_conservatively_relabelled(self):
        resting = np.zeros(self.grid.shape)
        # Below one dx*dy*min(dx,dy) voxel on the formal grid, placed on a
        # deliberately steep local substrate to exercise the tail closure.
        resting[20, 20] = 0.20
        mobile = np.zeros(self.grid.shape)
        mobile[20, 21] = 0.01
        momentum = np.zeros(self.grid.shape + (2,))
        momentum[20, 21, 0] = 0.01 * 2.0
        before = self.integrator.integrate(resting + mobile)
        result = DepositionOperator().apply(
            resting, mobile, momentum, self.material, self.grid,
            self.integrator, 1.0 / 60.0,
        )
        self.assertEqual(self.integrator.integrate(result.mobile_height_m), 0.0)
        self.assertAlmostEqual(
            self.integrator.integrate(result.H_resting_m), before, places=12
        )

    def test_full_step_commits_exact_ledger_and_swept_is_not_payload(self):
        H = np.ones(self.grid.shape)
        payload = PayloadState(
            volume_m3=0.0,
            capacity_m3=1.0,
            assumed_bulk_density_kg_m3=self.material.assumed_bulk_density_kg_m3,
            center_of_mass_bucket_frame_m=np.zeros(3),
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
        manager = BulkStateManager(state, self.integrator)
        intersection = ToolTerrainIntersectionModel().compute(
            H, self.sweep(0.2), self.tool_state(), self.grid, self.integrator
        )
        result = BulkMaterialInteractionModel().advance(
            manager,
            intersection,
            self.tool_state(),
            self.descriptor,
            self.grid,
            self.integrator,
            0.05,
        )
        balance = manager.ledger_snapshot().balance
        self.assertLess(balance.absolute_volume_error_m3, 1e-9)
        self.assertLessEqual(result.state.payload.volume_m3, result.state.payload.capacity_m3)
        self.assertNotAlmostEqual(
            intersection.candidate_intersection_volume_m3,
            result.state.payload.volume_m3,
        )


if __name__ == "__main__":
    unittest.main()
