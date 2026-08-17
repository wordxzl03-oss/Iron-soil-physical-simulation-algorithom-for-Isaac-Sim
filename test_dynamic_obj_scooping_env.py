import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from dynamic_obj_scooping_env import DynamicObjScoopingTrajectoryEnv


class DynamicObjScoopingTrajectoryEnvTests(unittest.TestCase):
    def test_dynamic_environment_uses_obj_wheel_geometry(self):
        env = DynamicObjScoopingTrajectoryEnv(
            grid_size=41, observation_grid=9, seed=5
        )
        self.assertAlmostEqual(env.vehicle_parameters.wheelbase_m, 2.30)
        self.assertAlmostEqual(env.vehicle_parameters.track_width_m, 2.84)
        self.assertAlmostEqual(env.vehicle_parameters.wheel_radius_m, 0.72)
        check_env(env, warn=True)

    def test_step_records_ground_fitted_vehicle_states(self):
        env = DynamicObjScoopingTrajectoryEnv(
            grid_size=41, observation_grid=9
        )
        env.reset(seed=8)
        _, reward, terminated, _, info = env.step(
            np.zeros(7, dtype=np.float32)
        )
        self.assertTrue(terminated)
        self.assertTrue(env.last_vehicle_states)
        self.assertGreaterEqual(min(s.z_m for s in env.last_vehicle_states), 0.72)
        self.assertGreater(info["dynamic_penetration_m"], 0.0)
        self.assertTrue(np.isfinite(reward))


if __name__ == "__main__":
    unittest.main()
