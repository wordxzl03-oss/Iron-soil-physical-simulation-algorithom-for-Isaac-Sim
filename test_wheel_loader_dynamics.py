import unittest

import numpy as np

from wheel_loader_dynamics import (
    VehicleParameters,
    VehicleState,
    WheelLoaderDynamicsEnv,
    fit_vehicle_to_terrain,
    grid_height_function,
    integrate_bicycle,
    vehicle_force_balance,
)


class WheelLoaderDynamicsTests(unittest.TestCase):
    def test_drive_accelerates_and_resistance_reduces_acceleration(self):
        parameters = VehicleParameters()
        state = VehicleState()
        free, _ = integrate_bicycle(
            state, 0.7, 0.0, 0.0, 0.1, parameters
        )
        digging, _ = integrate_bicycle(
            state, 0.7, 0.0, 0.0, 0.1, parameters,
            excavation_resistance_n=100_000,
        )
        self.assertGreater(free.speed_m_s, digging.speed_m_s)
        self.assertGreater(digging.speed_m_s, 0.0)

    def test_traction_force_is_limited_and_slip_reported(self):
        parameters = VehicleParameters(
            max_drive_force_n=400_000,
            tire_ground_friction=0.5,
        )
        forces = vehicle_force_balance(
            VehicleState(), 1.0, 0.0, 0.0, 0.0, parameters
        )
        self.assertLessEqual(
            abs(forces.applied_drive_n), forces.traction_limit_n
        )
        self.assertGreater(forces.slip_ratio, 0.0)

    def test_steering_turns_vehicle_toward_positive_x(self):
        state = VehicleState(speed_m_s=1.5)
        updated, _ = integrate_bicycle(
            state, 0.0, 0.5, 0.0, 0.2, VehicleParameters()
        )
        self.assertGreater(updated.heading_rad, 0.0)
        self.assertGreater(updated.x_m, 0.0)
        self.assertGreater(updated.y_m, 0.0)

    def test_grade_and_rolling_resistance_oppose_forward_motion(self):
        forces = vehicle_force_balance(
            VehicleState(speed_m_s=1.0),
            0.0,
            0.0,
            np.deg2rad(5.0),
            0.0,
            VehicleParameters(),
        )
        self.assertLess(forces.rolling_n, 0.0)
        self.assertLess(forces.grade_n, 0.0)
        self.assertLess(forces.velocity_n, 0.0)

    def test_env_has_gymnasium_style_signatures(self):
        env = WheelLoaderDynamicsEnv(dt=0.1, max_steps=2)
        observation, info = env.reset(options={"y_m": -5.0})
        self.assertEqual(observation.shape, (8,))
        self.assertIn("forces", info)
        result = env.step(np.array([0.5, 0.0, 0.0]))
        self.assertEqual(len(result), 5)
        self.assertFalse(result[2])
        self.assertFalse(result[3])
        result = env.step(np.array([0.5, 0.0, 0.0]))
        self.assertTrue(result[3])

    def test_four_wheel_contacts_set_height_pitch_and_roll(self):
        parameters = VehicleParameters()
        state = VehicleState(x_m=2.0, y_m=3.0)
        plane = lambda x, y: 0.05 * x + 0.10 * y
        pose = fit_vehicle_to_terrain(state, plane, parameters)
        self.assertAlmostEqual(pose.z_m, 1.12, places=10)
        self.assertAlmostEqual(pose.pitch_rad, np.arctan(0.10), places=10)
        self.assertAlmostEqual(pose.roll_rad, np.arctan(0.05), places=10)
        updated, forces = integrate_bicycle(
            VehicleState(speed_m_s=1.0),
            0.5, 0.0, 0.0, 0.1, parameters,
            terrain_height_function=plane,
        )
        self.assertAlmostEqual(updated.pitch_rad, np.arctan(0.10), places=10)
        self.assertLess(forces.grade_n, 0.0)

    def test_grid_height_sampler_is_bilinear(self):
        height = np.array([[0.0, 2.0], [1.0, 3.0]])
        sampler = grid_height_function(height, (0.0, 0.0), (1.0, 1.0))
        self.assertAlmostEqual(sampler(0.5, 0.5), 1.5)
        self.assertAlmostEqual(sampler(-2.0, 3.0), 2.0)

    def test_nonplanar_contacts_never_put_a_wheel_below_ground(self):
        parameters = VehicleParameters(wheelbase_m=2.30)
        heights = {
            (-1, 1): 0.0, (1, 1): 0.4,
            (-1, -1): 0.2, (1, -1): 0.0,
        }
        def terrain(x, y):
            return heights[(-1 if x < 0 else 1, -1 if y < 0 else 1)]
        pose = fit_vehicle_to_terrain(VehicleState(), terrain, parameters)
        half_l = parameters.wheelbase_m / 2
        half_w = parameters.track_width_m / 2
        offsets = (
            half_l * np.tan(pose.pitch_rad) - half_w * np.tan(pose.roll_rad),
            half_l * np.tan(pose.pitch_rad) + half_w * np.tan(pose.roll_rad),
            -half_l * np.tan(pose.pitch_rad) - half_w * np.tan(pose.roll_rad),
            -half_l * np.tan(pose.pitch_rad) + half_w * np.tan(pose.roll_rad),
        )
        wheel_bottoms = [
            pose.z_m - parameters.wheel_radius_m + offset for offset in offsets
        ]
        for bottom, ground in zip(wheel_bottoms, pose.wheel_heights_m):
            self.assertGreaterEqual(bottom + 1e-12, ground)


if __name__ == "__main__":
    unittest.main()
