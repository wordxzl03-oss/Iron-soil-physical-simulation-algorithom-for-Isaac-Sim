import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.config import SweepConfig
from isaac_bulk_pipeline.interaction import ContinuousSweepBuilder
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


def _descriptor(filename: str = "bucket_medium.yaml"):
    root = Path(__file__).resolve().parents[1]
    return ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(root / "configs" / filename)
    )


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:2, :2] = [[cosine, -sine], [sine, cosine]]
    return pose


def _transform(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (pose @ homogeneous.T).T[:, :3]


def _state(descriptor, pose: np.ndarray, timestamp: float) -> ToolState:
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


class ContinuousSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TerrainGrid(
            nx=241,
            ny=241,
            dx=0.05,
            dy=0.05,
            origin_x=-6.0,
            origin_y=-6.0,
            terrain_prim_path="/World/Terrain",
        )
        self.descriptor = _descriptor("bucket_small.yaml")
        self.builder = ContinuousSweepBuilder(
            SweepConfig(
                max_translation_step_grid_fraction=0.5,
                max_rotation_step_deg=2.0,
            )
        )

    def test_fast_large_timestep_matches_slow_substeps_without_gaps(self) -> None:
        start_pose = np.eye(4)
        start_pose[:3, 3] = [-2.5, -1.0, 0.7]
        end_pose = np.eye(4)
        end_pose[:3, 3] = [2.5, 2.0, 0.7]
        fast = self.builder.build(
            _state(self.descriptor, start_pose, 0.0),
            _state(self.descriptor, end_pose, 1.0),
            self.grid,
            self.descriptor,
        )
        distance = np.linalg.norm(end_pose[:3, 3] - start_pose[:3, 3])
        maximum_step = 0.5 * min(self.grid.dx, self.grid.dy)
        self.assertGreaterEqual(
            fast.diagnostics["interval_count"], int(np.ceil(distance / maximum_step))
        )

        slow_mask = np.zeros(self.grid.shape, dtype=bool)
        slow_surface = np.full(self.grid.shape, np.inf)
        previous_pose = start_pose
        substep_count = 20
        for index in range(1, substep_count + 1):
            fraction = index / substep_count
            current_pose = np.eye(4)
            current_pose[:3, 3] = (
                (1.0 - fraction) * start_pose[:3, 3]
                + fraction * end_pose[:3, 3]
            )
            result = self.builder.build(
                _state(self.descriptor, previous_pose, (index - 1) / substep_count),
                _state(self.descriptor, current_pose, fraction),
                self.grid,
                self.descriptor,
            )
            slow_mask |= result.affected_mask
            np.minimum(slow_surface, result.cut_surface, out=slow_surface)
            previous_pose = current_pose

        union = fast.affected_mask | slow_mask
        intersection = fast.affected_mask & slow_mask
        # Boundary pixels differ by at most one raster cell because the two
        # temporal partitions do not share identical sub-sample fractions.
        self.assertGreater(intersection.sum() / union.sum(), 0.99)
        np.testing.assert_allclose(
            fast.cut_surface[intersection], slow_surface[intersection], atol=1e-10
        )
        # The swept footprint must occupy every column between its endpoints.
        active_columns = np.flatnonzero(np.any(fast.affected_mask, axis=0))
        np.testing.assert_array_equal(
            active_columns,
            np.arange(active_columns[0], active_columns[-1] + 1),
        )

    def test_rotation_sampling_accounts_for_tool_radius(self) -> None:
        first_pose = _rotation_z(0.0)
        second_pose = _rotation_z(np.pi / 2.0)
        first_pose[:3, 3] = [0.0, 0.0, 0.8]
        second_pose[:3, 3] = [0.0, 0.0, 0.8]
        result = self.builder.build(
            _state(self.descriptor, first_pose, 0.0),
            _state(self.descriptor, second_pose, 0.1),
            self.grid,
            self.descriptor,
        )
        radius_bound = int(
            np.ceil(
                (np.pi / 2.0) * self.descriptor.proxy_radius_m
                / (0.5 * min(self.grid.dx, self.grid.dy))
            )
        )
        self.assertGreaterEqual(result.diagnostics["interval_count"], radius_bound)
        self.assertGreater(result.diagnostics["affected_cell_count"], 1000)


if __name__ == "__main__":
    unittest.main()
