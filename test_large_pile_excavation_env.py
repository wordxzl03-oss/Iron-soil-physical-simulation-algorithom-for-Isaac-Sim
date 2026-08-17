import unittest

import numpy as np

from large_pile_excavation_env import LargePileExcavationEnv


class LargePileExcavationEnvTests(unittest.TestCase):
    def test_pile_is_above_twenty_metres_and_has_label(self):
        env = LargePileExcavationEnv(grid_size=41, observation_grid=9)
        observation, info = env.reset(seed=31)
        self.assertGreater(env._initial.max(), 20.0)
        self.assertGreater(info["minimum_scoop_label"], 0)
        self.assertEqual(observation.shape, env.observation_space.shape)

    def test_scoop_updates_remaining_volume(self):
        env = LargePileExcavationEnv(grid_size=41, observation_grid=9)
        env.reset(seed=32)
        before = env.remaining_volume_m3
        _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
        self.assertLess(info["remaining_volume_m3"], before)
        self.assertEqual(info["scoop_count"], 1)


if __name__ == "__main__":
    unittest.main()
