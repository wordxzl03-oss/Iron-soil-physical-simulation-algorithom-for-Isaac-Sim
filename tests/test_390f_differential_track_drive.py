import unittest

import numpy as np

from isaac_bulk_pipeline.vehicle import (
    DifferentialTrackDriveConfig,
    DifferentialTrackDriveModel,
    IsaacDifferentialTrackDriveAdapter,
)


class _FakeRigidBody:
    def __init__(self, position=(0.0, 0.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float64)
        self.calls = []

    def get_world_poses(self):
        return self.position.reshape(1, 3), np.asarray([[1.0, 0.0, 0.0, 0.0]])

    def get_linear_velocities(self):
        return np.zeros((1, 3), dtype=np.float64)

    def get_angular_velocities(self):
        return np.zeros((1, 3), dtype=np.float64)

    def apply_forces_and_torques_at_pos(self, **kwargs):
        self.calls.append(kwargs)


class DifferentialTrackDriveTests(unittest.TestCase):
    def test_isaac_adapter_applies_equivalent_track_wrench_to_articulation_root(self):
        left = _FakeRigidBody((0.0, 1.0, 0.0))
        right = _FakeRigidBody((0.0, -1.0, 0.0))
        lower = _FakeRigidBody()
        model = DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(command_slew_per_s=100.0)
        )
        adapter = IsaacDifferentialTrackDriveAdapter(
            left,
            right,
            lower,
            model,
            left_application_offset_lower_local_m=np.asarray([0.0, 1.0, 0.0]),
            right_application_offset_lower_local_m=np.asarray([0.0, -1.0, 0.0]),
        )

        result = adapter.apply(
            0.5,
            0.5,
            1.0 / 60.0,
            left_contact_active=True,
            right_contact_active=True,
        )

        self.assertEqual(len(lower.calls), 2)
        self.assertEqual(left.calls, [])
        self.assertEqual(right.calls, [])
        np.testing.assert_allclose(
            lower.calls[0]["forces"].reshape(3), result.left_force_world_n
        )
        np.testing.assert_allclose(
            lower.calls[1]["forces"].reshape(3), result.right_force_world_n
        )
        np.testing.assert_allclose(
            lower.calls[0]["positions"].reshape(3), [0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(
            lower.calls[1]["positions"].reshape(3), [0.0, -1.0, 0.0]
        )

    def test_equal_commands_generate_equal_forward_physical_forces(self):
        model = DifferentialTrackDriveModel(DifferentialTrackDriveConfig(command_slew_per_s=100.0))
        out = model.step(.5, .5, np.eye(3), .1, left_contact_active=True, right_contact_active=True)
        np.testing.assert_allclose(out.left_force_world_n, [60000, 0, 0])
        np.testing.assert_allclose(out.right_force_world_n, [60000, 0, 0])

    def test_airborne_track_cannot_generate_traction(self):
        model = DifferentialTrackDriveModel(DifferentialTrackDriveConfig(command_slew_per_s=100.0))
        out = model.step(1, 1, np.eye(3), .1, left_contact_active=False, right_contact_active=True)
        np.testing.assert_allclose(out.left_force_world_n, 0)
        self.assertGreater(np.linalg.norm(out.right_force_world_n), 0)

    def test_drive_force_closes_belt_speed_error_instead_of_accelerating_forever(self):
        model = DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(command_slew_per_s=100.0)
        )
        at_target = model.step(
            .5,
            .5,
            np.eye(3),
            .1,
            left_contact_active=True,
            right_contact_active=True,
            lower_body_linear_velocity_world=np.array([.6, 0.0, 0.0]),
        )
        np.testing.assert_allclose(at_target.left_force_world_n, 0.0, atol=1e-9)
        np.testing.assert_allclose(at_target.right_force_world_n, 0.0, atol=1e-9)

    def test_commands_slew_and_rotated_body_sets_force_direction(self):
        model = DifferentialTrackDriveModel(DifferentialTrackDriveConfig(command_slew_per_s=1.0))
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        out = model.step(1, -1, rotation, .1, left_contact_active=True, right_contact_active=True)
        self.assertAlmostEqual(out.applied_left_command, .1)
        self.assertAlmostEqual(out.applied_right_command, -.1)
        np.testing.assert_allclose(out.left_force_world_n, [0, 12000, 0], atol=1e-8)
        np.testing.assert_allclose(out.right_force_world_n, [0, -12000, 0], atol=1e-8)

    def test_hold_brake_applies_force_opposing_measured_base_velocity(self):
        model = DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(
                command_slew_per_s=100.0,
                maximum_braking_force_per_track_n=160000.0,
                braking_full_force_speed_m_s=0.5,
            )
        )
        out = model.step(
            0.0,
            0.0,
            np.eye(3),
            0.1,
            left_contact_active=True,
            right_contact_active=True,
            braking=True,
            lower_body_linear_velocity_world=np.array([0.25, 0.0, 0.0]),
        )
        np.testing.assert_allclose(out.left_force_world_n, [-80000.0, 0.0, 0.0])
        np.testing.assert_allclose(out.right_force_world_n, [-80000.0, 0.0, 0.0])

    def test_hold_brake_resists_lateral_velocity_and_yaw_at_track_bodies(self):
        model = DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(
                command_slew_per_s=100.0,
                maximum_braking_force_per_track_n=160000.0,
            )
        )
        out = model.step(
            0.0,
            0.0,
            np.eye(3),
            0.1,
            left_contact_active=True,
            right_contact_active=True,
            braking=True,
            lower_body_linear_velocity_world=np.array([0.0, 0.125, 0.0]),
            lower_body_angular_velocity_world=np.array([0.0, 0.0, 0.15]),
        )
        # Both tracks oppose lateral motion. Opposed longitudinal components
        # create a yaw moment without applying a root pose or direct root torque.
        np.testing.assert_allclose(out.left_force_world_n, [80000.0, -80000.0, 0.0])
        np.testing.assert_allclose(out.right_force_world_n, [-80000.0, -80000.0, 0.0])

    def test_hold_brake_restores_static_position_and_yaw_errors_with_forces(self):
        model = DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(
                command_slew_per_s=100.0,
                maximum_braking_force_per_track_n=160000.0,
                braking_position_full_force_error_m=0.20,
                braking_yaw_full_force_error_rad=0.20,
            )
        )
        out = model.step(
            0.0,
            0.0,
            np.eye(3),
            0.1,
            left_contact_active=True,
            right_contact_active=True,
            braking=True,
            brake_position_error_world=np.array([0.05, 0.05, 0.0]),
            brake_yaw_error_rad=0.05,
        )
        # Both tracks restore +x/+y displacement; their opposed longitudinal
        # components additionally restore positive yaw through real lever arms.
        np.testing.assert_allclose(out.left_force_world_n, [0.0, -40000.0, 0.0])
        np.testing.assert_allclose(out.right_force_world_n, [-80000.0, -40000.0, 0.0])

    def test_yaw_brake_uses_real_force_pair_moment_sign(self):
        model = DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(
                command_slew_per_s=100.0,
                maximum_braking_force_per_track_n=160000.0,
            )
        )
        out = model.step(
            0.0,
            0.0,
            np.eye(3),
            0.1,
            left_contact_active=True,
            right_contact_active=True,
            braking=True,
            brake_yaw_error_rad=0.1,
            positive_yaw_force_pair_torque_sign=1.0,
        )
        self.assertLess(out.left_force_world_n[0], 0.0)
        self.assertGreater(out.right_force_world_n[0], 0.0)


if __name__ == "__main__":
    unittest.main()
