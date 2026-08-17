import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from rl_scooping_env import ScoopingTrajectoryEnv


class ScoopingTrajectoryEnvTests(unittest.TestCase):
    def test_sb3_environment_contract(self):
        env = ScoopingTrajectoryEnv(grid_size=41, observation_grid=9, seed=7)
        check_env(env, warn=True)

    def test_scoop_reports_positive_material_volume(self):
        env = ScoopingTrajectoryEnv(grid_size=41, observation_grid=9)
        observation, info = env.reset(seed=11)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertGreater(info["peak_height_m"], 0.0)
        _, reward, terminated, truncated, result = env.step(
            np.zeros(7, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertGreater(result["loaded_volume_m3"], 0.0)
        self.assertTrue(np.isfinite(reward))

    def test_action_maps_to_documented_limits(self):
        env = ScoopingTrajectoryEnv()
        low = env.action_to_trajectory(-np.ones(7, dtype=np.float32))
        high = env.action_to_trajectory(np.ones(7, dtype=np.float32))
        self.assertAlmostEqual(low.heading_deg, -28.0)
        self.assertAlmostEqual(high.heading_deg, 28.0)
        self.assertAlmostEqual(low.travel_length, 1.7)
        self.assertAlmostEqual(high.travel_length, 3.3)
        self.assertAlmostEqual(low.max_depth, 0.45)
        self.assertAlmostEqual(high.max_depth, 1.25)


if __name__ == "__main__":
    unittest.main()

