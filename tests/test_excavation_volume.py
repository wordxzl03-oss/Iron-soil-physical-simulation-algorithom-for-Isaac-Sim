import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.config import ExcavationConfig
from isaac_bulk_pipeline.interaction import ContinuousSweepBuilder, ExcavationOperator
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


def _descriptor(filename: str):
    root = Path(__file__).resolve().parents[1]
    return ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(root / "configs" / filename)
    )


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


class ExcavationVolumeTests(unittest.TestCase):
    @staticmethod
    def _grid(spacing: float, span: float = 12.75) -> TerrainGrid:
        size = int(round(span / spacing)) + 1
        actual_spacing = span / (size - 1)
        return TerrainGrid(
            nx=size,
            ny=size,
            dx=actual_spacing,
            dy=actual_spacing,
            origin_x=-span / 2.0,
            origin_y=-span / 2.0,
            terrain_prim_path="/World/Terrain",
        )

    def test_removed_volume_equals_height_difference_integral(self) -> None:
        grid = self._grid(0.05)
        descriptor = _descriptor("bucket_medium.yaml")
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 1.2, 0.75]
        sweep = ContinuousSweepBuilder().build(
            _state(descriptor, pose, 0.0),
            _state(descriptor, pose, 0.1),
            grid,
            descriptor,
        )
        initial = np.full(grid.shape, 2.0, dtype=np.float64)
        result = ExcavationOperator(
            ExcavationConfig(minimum_cut_depth_m=0.002)
        ).apply(initial, sweep, grid)
        expected = float(
            (initial - result.heightmap_excavated).sum() * grid.cell_area
        )
        self.assertAlmostEqual(result.removed_volume_m3, expected, places=12)
        self.assertGreater(result.removed_volume_m3, 0.0)
        self.assertTrue(np.all(result.heightmap_excavated <= initial))
        self.assertTrue(np.all(result.heightmap_excavated >= 0.0))

    def test_tool_size_changes_width_and_volume_without_operator_changes(self) -> None:
        grid = self._grid(0.05)
        initial = np.full(grid.shape, 2.0)
        operator = ExcavationOperator()
        widths = []
        volumes = []
        for filename in (
            "bucket_small.yaml",
            "bucket_medium.yaml",
            "bucket_large.yaml",
        ):
            descriptor = _descriptor(filename)
            pose = np.eye(4)
            pose[:3, 3] = [0.0, 1.0, 0.7]
            sweep = ContinuousSweepBuilder().build(
                _state(descriptor, pose, 0.0),
                _state(descriptor, pose, 0.1),
                grid,
                descriptor,
            )
            result = operator.apply(initial, sweep, grid)
            rows, columns = np.nonzero(result.affected_mask)
            widths.append((columns.max() - columns.min()) * grid.dx)
            volumes.append(result.removed_volume_m3)
        self.assertTrue(np.all(np.diff(widths) > 0.5))
        self.assertTrue(np.all(np.diff(volumes) > 1.0))

    def test_128_256_512_resolution_converges_in_physical_volume(self) -> None:
        descriptor = _descriptor("bucket_medium.yaml")
        volumes = []
        physical_widths = []
        for size in (128, 256, 512):
            span = 12.75
            spacing = span / (size - 1)
            grid = TerrainGrid(
                nx=size,
                ny=size,
                dx=spacing,
                dy=spacing,
                origin_x=-span / 2.0,
                origin_y=-span / 2.0,
                terrain_prim_path="/World/Terrain",
            )
            start = np.eye(4)
            end = np.eye(4)
            start[:3, 3] = [-0.5, 0.0, 0.8]
            end[:3, 3] = [0.5, 0.7, 0.8]
            sweep = ContinuousSweepBuilder().build(
                _state(descriptor, start, 0.0),
                _state(descriptor, end, 1.0),
                grid,
                descriptor,
            )
            result = ExcavationOperator().apply(
                np.full(grid.shape, 2.0), sweep, grid
            )
            volumes.append(result.removed_volume_m3)
            _, columns = np.nonzero(result.affected_mask)
            physical_widths.append((columns.max() - columns.min()) * spacing)
        self.assertLess(max(volumes) - min(volumes), 0.08 * np.mean(volumes))
        self.assertLess(
            max(physical_widths) - min(physical_widths),
            0.15,
        )
        self.assertLess(abs(volumes[2] - volumes[1]), abs(volumes[1] - volumes[0]))

    def test_minimum_cut_depth_suppresses_numerical_grazing(self) -> None:
        grid = self._grid(0.1)
        descriptor = _descriptor("bucket_small.yaml")
        pose = np.eye(4)
        pose[:3, 3] = [0.0, 0.0, 0.999]
        sweep = ContinuousSweepBuilder().build(
            _state(descriptor, pose, 0.0),
            _state(descriptor, pose, 0.1),
            grid,
            descriptor,
        )
        result = ExcavationOperator(
            ExcavationConfig(minimum_cut_depth_m=0.002)
        ).apply(np.ones(grid.shape), sweep, grid)
        self.assertEqual(result.removed_volume_m3, 0.0)
        self.assertFalse(np.any(result.affected_mask))


if __name__ == "__main__":
    unittest.main()
