import unittest

import numpy as np

from isaac_bulk_pipeline.vehicle import (
    ExcavatorActuatorConfig,
    ExcavatorActuatorModel,
)


class ExcavatorActuatorTests(unittest.TestCase):
    def setUp(self):
        self.config = ExcavatorActuatorConfig.cat_390f_l_mass_configuration()
        self.names = tuple(item.joint_name for item in self.config.joints)
        self.model = ExcavatorActuatorModel(self.config, self.names)

    def test_velocity_acceleration_effort_and_shared_power_are_bounded(self):
        self.model.reset(np.zeros(4), np.zeros(4))
        output = self.model.step(
            np.full(4, 10.0), np.zeros(4), np.full(4, 10.0), 1.0 / 60.0
        )
        for index, limit in enumerate(self.config.joints):
            self.assertLessEqual(
                abs(output.target_velocity_rad_s[index]),
                limit.acceleration_limit_rad_s2 / 60.0 + 1e-12,
            )
            self.assertLessEqual(
                abs(output.effort_command_nm[index]), limit.effort_limit_nm + 1e-9
            )
        self.assertLessEqual(
            output.positive_mechanical_power_w,
            self.config.shared_positive_power_limit_w + 1e-9,
        )

    def test_measured_static_hold_effort_is_seeded_and_clamped(self):
        requested = np.asarray([1e9, -1e9, 100.0, -100.0])
        self.model.reset(np.zeros(4), requested)
        output = self.model.step(np.zeros(4), np.zeros(4), np.zeros(4), 0.01)
        for index, limit in enumerate(self.config.joints):
            self.assertAlmostEqual(
                output.effort_command_nm[index],
                float(np.clip(requested[index], -limit.effort_limit_nm, limit.effort_limit_nm)),
            )

    def test_measured_overspeed_overrides_stale_hold_bias_with_full_braking(self):
        hold = np.asarray([item.effort_limit_nm for item in self.config.joints])
        velocity = np.asarray(
            [2.0 * item.velocity_limit_rad_s for item in self.config.joints]
        )
        self.model.reset(np.zeros(4), hold)
        output = self.model.step(np.ones(4), np.zeros(4), velocity, 1.0 / 60.0)
        for index, limit in enumerate(self.config.joints):
            self.assertAlmostEqual(output.effort_command_nm[index], -limit.effort_limit_nm)
            self.assertTrue(output.velocity_saturated[index])

    def test_position_target_velocity_obeys_acceleration_stopping_envelope(self):
        self.model.reset(np.zeros(4), np.zeros(4))
        error = np.full(4, 0.01)
        output = self.model.step(error, np.zeros(4), np.zeros(4), 1.0)
        for index, limit in enumerate(self.config.joints):
            stopping_speed = np.sqrt(
                2.0 * limit.acceleration_limit_rad_s2 * error[index]
            )
            self.assertLessEqual(
                abs(output.target_velocity_rad_s[index]),
                stopping_speed + 1e-12,
            )


if __name__ == "__main__":
    unittest.main()
