from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_state import BulkStateManager, MaterialScenario, PayloadState, TerrainState, TerrainVolumeIntegrator
from isaac_bulk_pipeline.interaction import SweepResult
from isaac_bulk_pipeline.operation import ContinuousLoadingCycleCoordinator, LoaderOperationStateMachine, OperationObservation, OperationTargets
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import BucketGeometryDescriptor, GeometryQuality, GeometrySource, ToolDescriptor, ToolState


def obs(t, pose, penetration=0.0, fill=0.0, payload=0.0):
    return OperationObservation(t, np.asarray(pose), 0.0, 0.0, 0.0, penetration, fill, payload, 0.0, 0)


class TestPhaseICycleCoordinator(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TerrainGrid(31, 31, 0.1, 0.1, -1.5, -1.5, "/World/Terrain")
        self.integrator = TerrainVolumeIntegrator.from_grid(self.grid)
        material = MaterialScenario("UNCALIBRATED", 1800.0, 32.0, 500.0, 0.35, 38.0, 30.0, 0.25)
        payload = PayloadState(0.0, 1.0, 1800.0, np.zeros(3))
        H = np.ones(self.grid.shape)
        state = TerrainState(H, np.zeros_like(H), np.zeros(H.shape + (2,)), payload, (), material, 0.0, 0.0, 0)
        manager = BulkStateManager(state, self.integrator)
        cutting = np.array([[-1, 0, 0], [1, 0, 0]])
        bottom = np.array([[0, -1, 0], [0, 0, 0]])
        interior = np.array([[0, -1, 0], [0, -1, 1], [0, 0, 0]])
        geometry = BucketGeometryDescriptor.from_extruded_profile(
            cutting_edge_local=cutting, bottom_profile_local=bottom,
            interior_profile_local=interior,
            top_edge_local=np.array([[-1, -1, 1], [1, -1, 1]]),
            rated_capacity_m3=1.0,
            geometry_source=GeometrySource.EXPLICIT_PROFILE,
            geometry_quality=GeometryQuality.REDUCED_ORDER,
        )
        self.descriptor = ToolDescriptor(
            "bucket", "/World/Bucket/ToolFrame", cutting, bottom,
            geometry.left_side_wall_local, geometry.right_side_wall_local, 2.0,
            interior_profile_local=interior, nominal_capacity_m3=1.0,
            proxy_level="L1", actual_proxy_type="ExtrudedProfileBucket_L1",
            bucket_geometry=geometry,
        )
        targets = OperationTargets(np.array([0, 0, 0]), np.array([0.5, 0, 0]), np.array([1, 0, 0]), np.array([-1, 0, np.pi]))
        self.coordinator = ContinuousLoadingCycleCoordinator(
            operation=LoaderOperationStateMachine(targets), state_manager=manager,
            grid=self.grid, integrator=self.integrator, descriptor=self.descriptor,
        )
        pose = np.eye(4)
        self.tool = ToolState(
            0.0, pose, pose, cutting, bottom,
            geometry.left_side_wall_local, geometry.right_side_wall_local,
            np.array([1.0, 0.0, 0.0]), np.zeros(3),
            mouth_polygon_terrain=geometry.mouth_polygon_local,
            top_edge_terrain=geometry.top_edge_local,
            separation_plane_direction_terrain=geometry.separation_plane_direction_local,
            separation_plane_source="BUCKET_GEOMETRY_BOTTOM_PLATE",
            tool_plus_z_separation_fallback_used=False,
        )

    def test_non_dig_step_advances_authoritative_state(self) -> None:
        self.coordinator.start(0.0)
        result = self.coordinator.step(obs(0.1, (-1, 0, 0)), self.tool, 0.1)
        self.assertEqual(result.material_mode, "settling")
        self.assertAlmostEqual(result.terrain_state.timestamp_s, 0.1)
        self.assertLess(self.coordinator.state_manager.ledger_snapshot().balance.absolute_volume_error_m3, 1e-9)

    def test_dig_transition_requires_sweep_not_hidden_deletion(self) -> None:
        self.coordinator.start(0.0)
        self.coordinator.step(obs(0.1, (0, 0, 0)), self.tool, 0.1)
        self.coordinator.step(obs(0.2, (0, 0, 0)), self.tool, 0.1)
        with self.assertRaisesRegex(ValueError, "continuous sweep"):
            self.coordinator.step(obs(0.3, (0.5, 0, 0)), self.tool, 0.1)

    def test_dig_step_returns_shared_wedge_soil_force(self) -> None:
        self.coordinator.start(0.0)
        self.coordinator.step(obs(0.1, (0, 0, 0)), self.tool, 0.05)
        self.coordinator.step(obs(0.2, (0, 0, 0)), self.tool, 0.05)
        rows, cols = np.indices(self.grid.shape)
        mask = (np.abs(rows - 15) <= 1) & (np.abs(cols - 15) <= 8)
        cut = np.full(self.grid.shape, np.inf); cut[mask] = 0.8
        sweep = SweepResult((14, 7, 17, 24), mask, cut, (np.eye(4),))
        result = self.coordinator.step(obs(0.3, (0.5, 0, 0)), self.tool, 0.05, sweep=sweep)
        self.assertEqual(result.material_mode, "dig_interaction")
        self.assertIsNotNone(result.soil_force)
        self.assertFalse(result.soil_force.shared_failure_zone_centroid)
        self.assertEqual(
            result.soil_force.application_point_model,
            "CUTTING_EDGE_STRIP_RESULTANT",
        )
        self.assertIsNotNone(result.soil_force.momentum_budget)
        self.assertLess(self.coordinator.state_manager.ledger_snapshot().balance.absolute_volume_error_m3, 1e-9)

    def test_coordinator_has_no_pose_write_calls(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "src/isaac_bulk_pipeline/operation/coordinator.py").read_text()
        tree = ast.parse(source)
        attributes = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertTrue({"set_world_pose", "set_local_pose"}.isdisjoint(attributes))


if __name__ == "__main__":
    unittest.main()
