import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from obj_scooping_env import ObjScoopingTrajectoryEnv


class ObjScoopingTrajectoryEnvTests(unittest.TestCase):
    def test_uses_obj_geometry_and_capacity(self):
        env = ObjScoopingTrajectoryEnv(grid_size=41, observation_grid=9)
        self.assertAlmostEqual(env.obj_bucket_width_m, 2.7, places=6)
        self.assertGreater(env.obj_bucket_length_m, 1.9)
        self.assertEqual(env.bucket_capacity_m3, 3.0)
        check_env(env, warn=True)

    def test_reported_load_does_not_exceed_obj_capacity(self):
        env = ObjScoopingTrajectoryEnv(grid_size=41, observation_grid=9)
        env.reset(seed=19)
        _, reward, _, _, info = env.step(np.ones(7, dtype=np.float32))
        self.assertLessEqual(info["loaded_volume_m3"], 3.0)
        self.assertLessEqual(info["fill_factor"], 1.0)
        self.assertTrue(np.isfinite(reward))

    def test_simple_wheel_loader_groups_are_supported(self):
        env = ObjScoopingTrajectoryEnv(
            obj_path="simple_wheel_loader.obj",
            grid_size=41,
            observation_grid=9,
        )
        self.assertIn("simple_chassis", env.obj_source_part_names)
        self.assertIn("simple_cab", env.obj_source_part_names)
        self.assertIn("wheel_front_left", env.obj_source_part_names)
        self.assertAlmostEqual(env.obj_bucket_width_m, 2.7, places=6)
        self.assertGreater(len(env.obj_parts["loader_frame_world.stl"].faces), 300)


if __name__ == "__main__":
    unittest.main()
