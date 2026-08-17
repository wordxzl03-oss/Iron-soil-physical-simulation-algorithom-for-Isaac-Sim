import math
import unittest

from isaac_bulk_pipeline.vehicle import (
    CommandSlewLimits,
    CommandSlewLimiter,
    VehicleCommand,
)


class PhaseBVehicleCommandTests(unittest.TestCase):
    def test_normalization_clamps_all_channels_and_brake_has_priority(self):
        command = VehicleCommand(4.0, 2.0, -3.0, 8.0, -9.0).normalized()
        self.assertEqual(
            command,
            VehicleCommand(throttle=0.0, brake=1.0, steering=-1.0, lift=1.0, bucket_curl=-1.0),
        )

    def test_non_finite_input_is_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite"):
                VehicleCommand(throttle=value)

    def test_four_signed_channels_are_dt_based_not_frame_based(self):
        limits = CommandSlewLimits(0.4, 0.6, 0.8, 1.0)
        target = VehicleCommand(1.0, 0.0, -1.0, 1.0, -1.0)
        one_step = CommandSlewLimiter(limits).step(target, 0.5)
        many = CommandSlewLimiter(limits)
        for _ in range(10):
            many_step = many.step(target, 0.05)
        for channel in ("throttle", "steering", "lift", "bucket_curl"):
            self.assertAlmostEqual(getattr(one_step, channel), getattr(many_step, channel), places=12)

    def test_brake_is_immediate_and_throttle_resumes_through_ramp(self):
        limiter = CommandSlewLimiter(CommandSlewLimits(1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(limiter.step(VehicleCommand(throttle=1.0), 0.4).throttle, 0.4)
        braking = limiter.step(VehicleCommand(throttle=1.0, brake=1.0), 0.01)
        self.assertEqual(braking.brake, 1.0)
        self.assertEqual(braking.throttle, 0.0)
        released = limiter.step(VehicleCommand(throttle=1.0), 0.1)
        self.assertAlmostEqual(released.throttle, 0.1)


if __name__ == "__main__":
    unittest.main()
