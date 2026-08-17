import unittest

import numpy as np

from animate_loader_3d import LoaderTrajectory
from physics_aware_trajectory import (
    MachineLimits,
    MaterialParameters,
    excavation_resistance,
    plan_resistance_aware_dig,
)


class PhysicsAwareTrajectoryTests(unittest.TestCase):
    def test_resistance_increases_with_depth(self):
        material = MaterialParameters()
        shallow = excavation_resistance(0.3, 3.0, 0.5, 1.0, material)
        deep = excavation_resistance(0.8, 3.0, 0.5, 1.0, material)
        self.assertGreater(deep, shallow)

    def test_planner_respects_force_and_ground_limits(self):
        height = np.zeros((61, 61))
        height[:, 25:] = np.linspace(0, 8, 36)[None, :]
        trajectory = LoaderTrajectory(
            approach_distance=3.0,
            travel_length=3.0,
            bucket_width=3.0,
            max_depth=1.5,
            lift_height=4.0,
        )
        limits = MachineLimits(max_resistance_n=180_000)
        plan = plan_resistance_aware_dig(
            height,
            (0.5, 0.5),
            np.array([12.0, 15.0]),
            np.array([1.0, 0.0]),
            trajectory,
            MaterialParameters(),
            limits,
        )
        self.assertLessEqual(plan.resistance_n.max(), 180_000 + 1)
        self.assertGreaterEqual(plan.cutting_edge_z.min(), 0.12)
        self.assertTrue(np.all(plan.path[:, 2] >= 0))


if __name__ == "__main__":
    unittest.main()
