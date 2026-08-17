import json
import math
from pathlib import Path
import unittest

import yaml

from isaac_bulk_pipeline.vehicle import (
    LoaderControlConfig,
    LoaderLowLevelController,
    PositionActuatorLimit,
    VehicleCommand,
    audit_usd_inventory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PhaseBLowLevelControllerTests(unittest.TestCase):
    def setUp(self):
        self.dof_names = [
            "bucket_joint",
            "rear_right_wheel_joint",
            "articulation_joint",
            "front_left_wheel_joint",
            "lift_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
        ]

    def test_dof_mapping_directions_and_sparse_modalities(self):
        config = LoaderControlConfig(wheel_directions=(1, -1, 1, -1))
        controller = LoaderLowLevelController(self.dof_names, config)
        output = controller.step(VehicleCommand(throttle=0.5), 0.1)

        self.assertEqual(output.wheel_action.joint_indices, (3, 5, 6, 1))
        self.assertEqual(output.wheel_action.joint_velocities, (5.0, -5.0, 5.0, -5.0))
        self.assertIsNone(output.wheel_action.joint_positions)
        self.assertEqual(output.position_action.joint_indices, (2, 4, 0))
        self.assertTrue(set(output.wheel_action.joint_indices).isdisjoint(output.position_action.joint_indices))
        self.assertTrue(all(math.isfinite(value) for value in output.position_action.joint_positions))

    def test_target_speed_fallback_envelope_is_explicit(self):
        config = LoaderControlConfig(
            wheel_velocity_rad_s=10.0,
            wheel_effort_nm=20_000.0,
            wheel_power_limit_w=100_000.0,
            wheel_power_reference_rad_s=1.0,
            wheel_power_min_guard_rad_s=1.0,
            wheel_power_discrete_safety_factor=1.0,
        )
        controller = LoaderLowLevelController(self.dof_names, config)
        fast = controller.step(VehicleCommand(throttle=1.0), 0.1)
        self.assertEqual(
            fast.wheel_power_cap_basis, "TARGET_JOINT_VELOCITY_FALLBACK"
        )
        self.assertIsNone(fast.wheel_power_cap_previous_measured_omega_rad_s)
        self.assertEqual(fast.wheel_power_cap_target_omega_rad_s, (10.0,) * 4)
        self.assertEqual(fast.wheel_power_cap_worst_case_omega_rad_s, (10.0,) * 4)
        self.assertEqual(fast.wheel_power_cap_denominator_rad_s, (10.0,) * 4)
        self.assertEqual(fast.wheel_power_cap_discrete_safety_factor, 1.0)
        self.assertEqual(fast.wheel_action.effort_limits, (10_000.0,) * 4)
        self.assertEqual(
            fast.wheel_command_envelope_power_bound_w, (100_000.0,) * 4
        )
        slow = controller.step(VehicleCommand(throttle=0.05), 0.1)
        self.assertEqual(slow.wheel_action.effort_limits, (20_000.0,) * 4)
        braking = controller.step(VehicleCommand(throttle=1.0, brake=1.0), 0.1)
        self.assertEqual(braking.wheel_action.joint_velocities, (0.0,) * 4)
        self.assertEqual(braking.wheel_action.effort_limits, (20_000.0,) * 4)

    def test_measured_wheel_speed_caps_each_effort_including_braking(self):
        config = LoaderControlConfig(
            wheel_velocity_rad_s=10.0,
            wheel_effort_nm=20_000.0,
            wheel_power_limit_w=100_000.0,
            wheel_power_reference_rad_s=1.0,
            wheel_power_min_guard_rad_s=1.0,
            wheel_power_discrete_safety_factor=1.0,
        )
        controller = LoaderLowLevelController(self.dof_names, config)
        # Full articulation order. Mapping wheel order is front-left,
        # front-right, rear-left, rear-right -> 2, 5, 10, -20 rad/s.
        measured = [0.0, -20.0, 0.0, 2.0, 0.0, 5.0, 10.0]
        output = controller.step(
            VehicleCommand(throttle=1.0, brake=1.0),
            0.1,
            measured_joint_velocities=measured,
        )
        self.assertEqual(output.wheel_action.joint_velocities, (0.0,) * 4)
        self.assertEqual(
            output.wheel_power_cap_basis,
            "MEASURED_OR_TARGET_WORST_CASE",
        )
        self.assertEqual(
            output.wheel_power_cap_previous_measured_omega_rad_s,
            (2.0, 5.0, 10.0, -20.0),
        )
        self.assertEqual(output.wheel_power_cap_target_omega_rad_s, (0.0,) * 4)
        self.assertEqual(
            output.wheel_power_cap_worst_case_omega_rad_s,
            (2.0, 5.0, 10.0, 20.0),
        )
        self.assertEqual(
            output.wheel_power_cap_denominator_rad_s, (2.0, 5.0, 10.0, 20.0)
        )
        self.assertEqual(
            output.wheel_action.effort_limits,
            (20_000.0, 20_000.0, 10_000.0, 5_000.0),
        )
        self.assertEqual(
            output.wheel_command_envelope_power_bound_w,
            (40_000.0, 100_000.0, 100_000.0, 100_000.0),
        )

        startup = controller.step(
            VehicleCommand(throttle=1.0),
            0.1,
            measured_joint_velocities=[0.0] * 7,
        )
        self.assertEqual(startup.wheel_power_cap_denominator_rad_s, (10.0,) * 4)
        self.assertEqual(startup.wheel_action.effort_limits, (10_000.0,) * 4)

    def test_minimum_speed_guard_protects_braking_contact_transient(self):
        config = LoaderControlConfig(
            wheel_effort_nm=20_000.0,
            wheel_power_limit_w=160_000.0,
            wheel_power_reference_rad_s=1.0,
            wheel_power_min_guard_rad_s=10.0,
            wheel_power_discrete_safety_factor=1.15,
        )
        controller = LoaderLowLevelController(self.dof_names, config)
        output = controller.step(
            VehicleCommand(brake=1.0),
            1.0 / 60.0,
            measured_joint_velocities=[0.0] * 7,
        )
        self.assertEqual(output.wheel_power_cap_worst_case_omega_rad_s, (10.0,) * 4)
        self.assertEqual(output.wheel_power_cap_denominator_rad_s, (11.5,) * 4)
        self.assertEqual(output.wheel_power_cap_min_guard_rad_s, 10.0)
        expected_effort = 160_000.0 / 11.5
        for effort in output.wheel_action.effort_limits:
            self.assertAlmostEqual(effort, expected_effort)
            self.assertLessEqual(effort * 11.25, 160_000.0)

    def test_measured_velocity_shape_and_finiteness_are_strict(self):
        controller = LoaderLowLevelController(self.dof_names)
        with self.assertRaisesRegex(ValueError, "expected 7 measured velocities"):
            controller.step(
                VehicleCommand(), 0.1, measured_joint_velocities=[0.0] * 6
            )
        invalid = [0.0] * 7
        invalid[3] = math.nan
        with self.assertRaisesRegex(ValueError, "measured_joint_velocity must be finite"):
            controller.step(
                VehicleCommand(), 0.1, measured_joint_velocities=invalid
            )
        with self.assertRaisesRegex(ValueError, "safety_factor must be >= 1"):
            LoaderControlConfig(wheel_power_discrete_safety_factor=0.99)
        with self.assertRaisesRegex(ValueError, "min_guard_rad_s must be positive"):
            LoaderControlConfig(wheel_power_min_guard_rad_s=0.0)

    def test_steer_lift_and_curl_integrate_and_saturate_at_finite_limits(self):
        small = PositionActuatorLimit(-0.1, 0.1, 0.2, 50.0)
        config = LoaderControlConfig(steering=small, lift=small, bucket=small)
        controller = LoaderLowLevelController(self.dof_names, config)
        output = None
        for _ in range(20):
            output = controller.step(VehicleCommand(steering=1, lift=-1, bucket_curl=1), 0.1)
        assert output is not None
        self.assertEqual(output.position_action.joint_positions, (0.1, -0.1, 0.1))
        self.assertEqual(output.position_action.effort_limits, (50.0, 50.0, 50.0))
        self.assertTrue(all(abs(value) <= 0.2 + 1e-12 for value in output.position_action.joint_velocities))

    def test_phase_b_yaml_has_provenance_and_matches_inventory(self):
        document = yaml.safe_load(
            (REPOSITORY_ROOT / "configs/phase_b_vehicle.yaml").read_text(encoding="utf-8")
        )
        provenance = document["vehicle"]["provenance"]
        self.assertEqual(provenance["wheel_radius_m"], "KNOWN_GEOMETRY")
        self.assertEqual(provenance["wheel_power_limit_w"], "UNCALIBRATED")
        self.assertEqual(
            provenance["wheel_power_discrete_safety_factor"],
            "UNCALIBRATED_DISCRETE_GUARD",
        )
        self.assertEqual(
            provenance["wheel_power_min_guard_rad_s"],
            "UNCALIBRATED_GUARD",
        )
        config = LoaderControlConfig.from_mapping(document["vehicle"])
        self.assertEqual(config.wheel_power_min_guard_rad_s, 10.0)
        self.assertEqual(config.wheel_power_discrete_safety_factor, 1.15)
        inventory = json.loads(
            (REPOSITORY_ROOT / "outputs/phase_a_usd_inventory.json").read_text(encoding="utf-8")
        )
        self.assertEqual(audit_usd_inventory(inventory, config), ())

    def test_missing_or_duplicate_dof_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing expected DOFs"):
            LoaderLowLevelController(self.dof_names[:-1])
        duplicate = self.dof_names[:-1] + [self.dof_names[0]]
        with self.assertRaisesRegex(ValueError, "unique"):
            LoaderLowLevelController(duplicate)


if __name__ == "__main__":
    unittest.main()
