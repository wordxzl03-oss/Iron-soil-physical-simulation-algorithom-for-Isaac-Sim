import unittest

import numpy as np

from isaac_bulk_pipeline.config import RobotConfig
from isaac_bulk_pipeline.robot import RobotAdapter, usd_matrix_to_numpy_m


class _Prim:
    def __init__(self, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid


class _Stage:
    def __init__(self, valid_paths) -> None:
        self.valid_paths = set(valid_paths)

    def GetPrimAtPath(self, path: str) -> _Prim:
        return _Prim(path in self.valid_paths)


class _Articulation:
    dof_names = ["steering", "lift", "bucket"]

    def __init__(self) -> None:
        self.positions = np.array([0.1, 0.2, -0.3])
        self.velocities = np.array([0.4, 0.5, -0.6])
        self.reset_count = 0

    def get_joint_positions(self):
        return self.positions

    def get_joint_velocities(self):
        return self.velocities

    def post_reset(self) -> None:
        self.reset_count += 1


class RobotAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RobotConfig(
            robot_root_prim="/World/Loader",
            articulation_root_prim="/World/Loader/chassis",
            tool_link_prim="/World/Loader/bucket",
        )
        self.stage = _Stage(
            {
                self.config.robot_root_prim,
                self.config.articulation_root_prim,
                self.config.tool_link_prim,
            }
        )

    def test_reads_joint_and_tool_link_state_without_geometry_assumptions(self) -> None:
        articulation = _Articulation()
        pose = np.eye(4)
        pose[:3, 3] = [4.0, 5.0, 6.0]
        adapter = RobotAdapter(
            articulation=articulation,
            time_source=lambda: 12.5,
            pose_reader=lambda stage, path: pose,
        )
        adapter.initialize(self.stage, self.config)

        joints = adapter.get_joint_state()
        self.assertEqual(joints.names, ("steering", "lift", "bucket"))
        self.assertEqual(joints.timestamp, 12.5)
        np.testing.assert_array_equal(joints.positions_rad, articulation.positions)
        np.testing.assert_array_equal(joints.velocities_rad_s, articulation.velocities)
        np.testing.assert_array_equal(adapter.get_tool_link_pose_world(), pose)

        joints.positions_rad[0] = 99.0
        observed_pose = adapter.get_tool_link_pose_world()
        observed_pose[0, 3] = 99.0
        self.assertEqual(articulation.positions[0], 0.1)
        self.assertEqual(pose[0, 3], 4.0)

        adapter.reset()
        self.assertEqual(articulation.reset_count, 1)

    def test_reports_missing_prim_path_with_context(self) -> None:
        stage = _Stage(
            {self.config.robot_root_prim, self.config.articulation_root_prim}
        )
        adapter = RobotAdapter(articulation=_Articulation())
        with self.assertRaisesRegex(ValueError, "tool_link_prim"):
            adapter.initialize(stage, self.config)

    def test_usd_row_matrix_conversion_transposes_and_converts_units(self) -> None:
        usd_row_matrix = np.eye(4)
        usd_row_matrix[3, :3] = [100.0, 200.0, 300.0]
        converted = usd_matrix_to_numpy_m(usd_row_matrix, 0.01)
        np.testing.assert_allclose(converted[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(converted[3], [0.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
