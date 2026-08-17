import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.interaction import ContinuousSweepBuilder
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


def _transform(pose, points):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (pose @ homogeneous.T).T[:, :3]


def _state(descriptor, pose, timestamp):
    return ToolState(
        timestamp=timestamp,
        pose_world=pose,
        pose_terrain=pose,
        cutting_edge_terrain=_transform(pose, descriptor.cutting_edge_local),
        bottom_profile_terrain=_transform(pose, descriptor.bottom_profile_local),
        left_boundary_terrain=_transform(pose, descriptor.left_boundary_local),
        right_boundary_terrain=_transform(pose, descriptor.right_boundary_local),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )


class ModelScalabilityTests(unittest.TestCase):
    def test_visual_mesh_face_count_cannot_change_sweep(self) -> None:
        root = Path(__file__).resolve().parents[1]
        descriptor = ToolDescriptorLoader.load(
            ToolDescriptorLoader.load_config(root / "configs" / "bucket_medium.yaml")
        )
        low_visual = replace(descriptor, metadata={"visual_mesh_faces": 120})
        high_visual = replace(descriptor, metadata={"visual_mesh_faces": 2_000_000})
        grid = TerrainGrid(
            nx=161,
            ny=161,
            dx=0.05,
            dy=0.05,
            origin_x=-4.0,
            origin_y=-4.0,
            terrain_prim_path="/World/Terrain",
        )
        start = np.eye(4)
        end = np.eye(4)
        start[:3, 3] = [-0.5, 0.0, 0.7]
        end[:3, 3] = [0.8, 1.0, 0.5]
        builder = ContinuousSweepBuilder()
        low = builder.build(
            _state(low_visual, start, 0.0),
            _state(low_visual, end, 1.0),
            grid,
            low_visual,
        )
        high = builder.build(
            _state(high_visual, start, 0.0),
            _state(high_visual, end, 1.0),
            grid,
            high_visual,
        )
        np.testing.assert_array_equal(low.affected_mask, high.affected_mask)
        np.testing.assert_array_equal(low.cut_surface, high.cut_surface)
        self.assertNotIn("visual_mesh_faces", low.diagnostics)


if __name__ == "__main__":
    unittest.main()
