import unittest
import warnings
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolKinematicsAdapter


def _rotation_z(angle: float) -> np.ndarray:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.eye(4)
    result[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return result


class ToolTransformTests(unittest.TestCase):
    @staticmethod
    def _descriptor(filename: str = "bucket_medium.yaml"):
        root = Path(__file__).resolve().parents[1]
        return ToolDescriptorLoader.load(
            ToolDescriptorLoader.load_config(root / "configs" / filename)
        )

    def test_transform_chain_matches_manual_world_and_terrain_calculation(self) -> None:
        terrain_to_world = _rotation_z(np.deg2rad(30.0))
        terrain_to_world[:3, 3] = [10.0, -2.0, 0.5]
        grid = TerrainGrid(
            nx=50,
            ny=40,
            dx=0.05,
            dy=0.05,
            origin_x=-1.0,
            origin_y=-1.0,
            terrain_prim_path="/World/Terrain",
            terrain_to_world_matrix=terrain_to_world,
        )
        descriptor = self._descriptor()
        link_pose = _rotation_z(np.deg2rad(-20.0))
        link_pose[:3, 3] = [12.0, 1.0, 2.0]

        state = ToolKinematicsAdapter(descriptor, grid).update(link_pose, 3.0)
        expected_world = link_pose @ descriptor.tool_to_link_matrix
        expected_terrain = np.linalg.inv(terrain_to_world) @ expected_world
        cutting_h = np.column_stack(
            (descriptor.cutting_edge_local, np.ones(len(descriptor.cutting_edge_local)))
        )
        expected_cutting = (expected_terrain @ cutting_h.T).T[:, :3]

        np.testing.assert_allclose(state.pose_world, expected_world, atol=1e-12)
        np.testing.assert_allclose(state.pose_terrain, expected_terrain, atol=1e-12)
        np.testing.assert_allclose(
            state.cutting_edge_terrain, expected_cutting, atol=1e-12
        )
        np.testing.assert_allclose(state.linear_velocity, 0.0)
        np.testing.assert_allclose(state.angular_velocity, 0.0)

        tool_origin_world = state.pose_world[:3, 3]
        np.testing.assert_allclose(
            grid.terrain_to_world(state.pose_terrain[:3, 3]), tool_origin_world
        )
        row_column = grid.terrain_to_grid(state.pose_terrain[:3, 3])
        roundtrip = grid.grid_to_terrain(
            row_column[0], row_column[1], state.pose_terrain[2, 3]
        )
        np.testing.assert_allclose(roundtrip, state.pose_terrain[:3, 3])

    def test_translation_rotation_velocity_and_reset(self) -> None:
        grid = TerrainGrid(
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
            origin_x=0.0,
            origin_y=0.0,
            terrain_prim_path="/World/Terrain",
        )
        adapter = ToolKinematicsAdapter(self._descriptor(), grid)
        first_pose = np.eye(4)
        adapter.update(first_pose, 1.0)

        second_pose = _rotation_z(np.pi / 2.0)
        second_pose[:3, 3] = [2.0, -1.0, 0.5]
        second = adapter.update(second_pose, 3.0)
        # Tool origin contains the link-to-tool offset, so derive expected
        # velocity from the two composed poses rather than from link translation.
        first_origin = (
            first_pose @ adapter.descriptor.tool_to_link_matrix
        )[:3, 3]
        second_origin = (
            second_pose @ adapter.descriptor.tool_to_link_matrix
        )[:3, 3]
        np.testing.assert_allclose(
            second.linear_velocity, (second_origin - first_origin) / 2.0
        )
        np.testing.assert_allclose(
            second.angular_velocity, [0.0, 0.0, np.pi / 4.0], atol=1e-9
        )

        adapter.reset()
        reset_state = adapter.update(second_pose, 10.0)
        np.testing.assert_allclose(reset_state.linear_velocity, 0.0)
        np.testing.assert_allclose(reset_state.angular_velocity, 0.0)

    def test_three_bucket_sizes_use_same_kinematics_code(self) -> None:
        grid = TerrainGrid(
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
            origin_x=0.0,
            origin_y=0.0,
            terrain_prim_path="/World/Terrain",
        )
        link_pose = _rotation_z(0.4)
        measured = []
        for filename in (
            "bucket_small.yaml",
            "bucket_medium.yaml",
            "bucket_large.yaml",
        ):
            state = ToolKinematicsAdapter(self._descriptor(filename), grid).update(
                link_pose, 0.0
            )
            measured.append(
                np.linalg.norm(
                    state.cutting_edge_terrain[-1]
                    - state.cutting_edge_terrain[0]
                )
            )
        np.testing.assert_allclose(measured, [2.0, 3.2, 4.5], atol=1e-12)

    def test_uniform_scale_is_supported_and_nonuniform_scale_warns(self) -> None:
        grid = TerrainGrid(
            nx=10,
            ny=10,
            dx=0.1,
            dy=0.1,
            origin_x=0.0,
            origin_y=0.0,
            terrain_prim_path="/World/Terrain",
        )
        descriptor = self._descriptor()
        uniform = np.diag([2.0, 2.0, 2.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            uniform_state = ToolKinematicsAdapter(descriptor, grid).update(uniform, 0.0)
        self.assertEqual(caught, [])
        self.assertFalse(uniform_state.diagnostics["non_uniform_scale"])
        self.assertAlmostEqual(uniform_state.diagnostics["uniform_scale"], 2.0)

        nonuniform = np.diag([1.0, 2.0, 1.0, 1.0])
        with self.assertWarnsRegex(RuntimeWarning, "non-uniform scale"):
            state = ToolKinematicsAdapter(descriptor, grid).update(nonuniform, 0.0)
        self.assertTrue(state.diagnostics["non_uniform_scale"])


if __name__ == "__main__":
    unittest.main()
