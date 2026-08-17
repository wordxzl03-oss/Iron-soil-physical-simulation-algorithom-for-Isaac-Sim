import unittest
from pathlib import Path

import numpy as np
import yaml

from isaac_bulk_pipeline.telemetry import (
    ActuatorCategory,
    ActuatorPowerSample,
    EffortSource,
    IsaacTelemetryInputAdapter,
    JointTelemetrySample,
    MechanicalEnergyAccumulator,
    RuntimeProfiler,
    SlopeTrialMetrics,
    VehiclePoseSample,
    VehicleTelemetryFrame,
    VehicleTelemetryRecorder,
    WheelKinematicsSample,
    WheelTerrainContactSample,
    compute_longitudinal_slip_ratio,
    validate_comparable_slope_trials,
)


class PhaseDTelemetryTests(unittest.TestCase):
    @staticmethod
    def _pose(timestamp_s: float = 0.0, position=(0.0, 0.0, 0.0)):
        return VehiclePoseSample(
            timestamp_s=timestamp_s,
            position_world_m=np.asarray(position, dtype=np.float64),
            orientation_xyzw=np.asarray([0.0, 0.0, 0.0, 1.0]),
            roll_rad=0.01,
            pitch_rad=0.02,
            yaw_rad=0.03,
            linear_velocity_world_m_s=np.asarray([1.0, 0.0, 0.0]),
            angular_velocity_world_rad_s=np.asarray([0.0, 0.0, 0.1]),
            articulation_angle_rad=0.05,
            local_terrain_slope_rad=np.deg2rad(10.0),
        )

    def test_tau_omega_positive_and_signed_energy_by_category(self) -> None:
        accumulator = MechanicalEnergyAccumulator()
        accumulator.add_step(
            [
                ActuatorPowerSample(
                    "left_wheel",
                    ActuatorCategory.DRIVE,
                    torque_nm=10.0,
                    angular_velocity_rad_s=2.0,
                    effort_source=EffortSource.MEASURED,
                ),
                ActuatorPowerSample(
                    "right_wheel",
                    ActuatorCategory.DRIVE,
                    torque_nm=-4.0,
                    angular_velocity_rad_s=3.0,
                    effort_source=EffortSource.MEASURED,
                ),
                ActuatorPowerSample(
                    "steer",
                    ActuatorCategory.STEER,
                    torque_nm=5.0,
                    angular_velocity_rad_s=-2.0,
                ),
                ActuatorPowerSample(
                    "lift",
                    ActuatorCategory.LIFT,
                    torque_nm=8.0,
                    angular_velocity_rad_s=0.5,
                ),
                ActuatorPowerSample(
                    "bucket",
                    ActuatorCategory.BUCKET,
                    torque_nm=6.0,
                    angular_velocity_rad_s=1.0,
                ),
            ],
            dt_s=0.5,
        )
        report = accumulator.snapshot()
        self.assertAlmostEqual(report.drive.positive_j, 10.0)
        self.assertAlmostEqual(report.drive.signed_j, 4.0)
        self.assertAlmostEqual(report.steer.positive_j, 0.0)
        self.assertAlmostEqual(report.steer.signed_j, -5.0)
        self.assertAlmostEqual(report.lift.positive_j, 2.0)
        self.assertAlmostEqual(report.bucket.positive_j, 3.0)
        self.assertAlmostEqual(report.total.positive_j, 15.0)
        self.assertAlmostEqual(report.total.signed_j, 4.0)

    def test_measured_applied_and_estimated_effort_are_not_conflated(self) -> None:
        measured = JointTelemetrySample(
            joint_name="wheel_joint",
            category=ActuatorCategory.DRIVE,
            angular_velocity_rad_s=2.0,
            applied_effort_nm=100.0,
            measured_effort_nm=5.0,
            estimated_effort_nm=7.0,
        ).to_power_sample()
        self.assertEqual(measured.effort_source, EffortSource.MEASURED)
        self.assertEqual(measured.torque_nm, 5.0)
        self.assertEqual(measured.power_w, 10.0)

        estimated = JointTelemetrySample(
            joint_name="unobserved_joint",
            category=ActuatorCategory.BUCKET,
            angular_velocity_rad_s=3.0,
            estimated_effort_nm=4.0,
        ).to_power_sample()
        self.assertEqual(estimated.effort_source, EffortSource.ESTIMATED)
        self.assertEqual(estimated.power_w, 12.0)

    def test_isaac_array_adapter_preserves_joint_channels(self) -> None:
        adapter = IsaacTelemetryInputAdapter(
            {
                "wheel": ActuatorCategory.DRIVE,
                "steer": ActuatorCategory.STEER,
                "lift": ActuatorCategory.LIFT,
                "bucket": ActuatorCategory.BUCKET,
            }
        )
        frame = adapter.build_frame(
            pose=self._pose(),
            joint_names=["wheel", "steer", "lift", "bucket"],
            angular_velocities_rad_s=[2.0, 0.2, 0.4, -0.5],
            applied_efforts_nm=[100.0, 20.0, 30.0, 40.0],
            measured_efforts_nm=[90.0, 18.0, 28.0, 35.0],
            target_velocities_rad_s=[3.0, 0.3, 0.5, -0.6],
        )
        self.assertEqual(len(frame.joints), 4)
        self.assertEqual(frame.joints[0].applied_effort_nm, 100.0)
        self.assertEqual(frame.joints[0].measured_effort_nm, 90.0)
        self.assertEqual(frame.power_samples()[0].effort_source, EffortSource.MEASURED)
        with self.assertRaisesRegex(KeyError, "missing joint classifications"):
            adapter.build_joint_samples(
                joint_names=["unknown"],
                angular_velocities_rad_s=[0.0],
                applied_efforts_nm=[0.0],
            )

    def test_slip_formula_and_low_speed_guard(self) -> None:
        self.assertEqual(
            compute_longitudinal_slip_ratio(
                wheel_radius_m=0.5,
                wheel_angular_velocity_rad_s=0.1,
                longitudinal_speed_m_s=0.04,
                low_speed_threshold_m_s=0.1,
            ),
            0.0,
        )
        slip = compute_longitudinal_slip_ratio(
            wheel_radius_m=0.5,
            wheel_angular_velocity_rad_s=4.0,
            longitudinal_speed_m_s=1.5,
        )
        self.assertAlmostEqual(slip, 0.25)
        sample = WheelKinematicsSample.from_kinematics(
            timestamp_s=1.0,
            wheel_name="front_left",
            wheel_radius_m=0.5,
            angular_velocity_rad_s=4.0,
            target_angular_velocity_rad_s=5.0,
            longitudinal_speed_m_s=1.5,
        )
        self.assertAlmostEqual(sample.slip_ratio, 0.25)
        with self.assertRaisesRegex(ValueError, "documented formula"):
            WheelKinematicsSample(
                timestamp_s=1.0,
                wheel_name="front_left",
                wheel_radius_m=0.5,
                angular_velocity_rad_s=4.0,
                target_angular_velocity_rad_s=5.0,
                longitudinal_speed_m_s=1.5,
                slip_ratio=0.0,
            )

    def test_pose_contact_schema_is_finite_and_synchronized(self) -> None:
        contact = WheelTerrainContactSample(
            timestamp_s=0.0,
            wheel_name="front_left",
            position_world_m=np.asarray([1.0, 2.0, 0.5]),
            normal_load_n=20_000.0,
            tangential_load_n=2_000.0,
            slip_ratio=0.1,
            wheel_angular_velocity_rad_s=3.0,
            contact_duration_s=0.5,
        )
        frame = VehicleTelemetryFrame(
            pose=self._pose(),
            wheel_contacts=(contact,),
        )
        self.assertEqual(frame.wheel_contacts[0].normal_load_n, 20_000.0)
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            VehiclePoseSample(
                timestamp_s=0.0,
                position_world_m=np.zeros(3),
                orientation_xyzw=np.zeros(4),
                roll_rad=0.0,
                pitch_rad=0.0,
                yaw_rad=0.0,
                linear_velocity_world_m_s=np.zeros(3),
                angular_velocity_world_rad_s=np.zeros(3),
                articulation_angle_rad=0.0,
                local_terrain_slope_rad=0.0,
            )
        with self.assertRaisesRegex(ValueError, "NaN or Inf|finite"):
            self._pose(position=(np.nan, 0.0, 0.0))

    def test_recorder_deep_snapshot_distance_energy_and_reset(self) -> None:
        joint0 = JointTelemetrySample(
            "wheel",
            ActuatorCategory.DRIVE,
            angular_velocity_rad_s=2.0,
            measured_effort_nm=10.0,
        )
        joint1 = JointTelemetrySample(
            "wheel",
            ActuatorCategory.DRIVE,
            angular_velocity_rad_s=3.0,
            measured_effort_nm=10.0,
        )
        recorder = VehicleTelemetryRecorder()
        recorder.record(
            VehicleTelemetryFrame(pose=self._pose(0.0), joints=(joint0,)),
            dt_s=0.1,
        )
        recorder.record(
            VehicleTelemetryFrame(
                pose=self._pose(0.1, position=(3.0, 4.0, 0.0)),
                joints=(joint1,),
            ),
            dt_s=0.1,
        )
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot.elapsed_time_s, 0.2)
        self.assertEqual(snapshot.distance_travelled_m, 5.0)
        self.assertAlmostEqual(snapshot.energy.drive.positive_j, 5.0)
        snapshot.frames[0].pose.position_world_m.setflags(write=True)
        snapshot.frames[0].pose.position_world_m[0] = 99.0
        self.assertEqual(recorder.snapshot().frames[0].pose.position_world_m[0], 0.0)
        recorder.reset()
        reset = recorder.snapshot()
        self.assertEqual(recorder.frame_count, 0)
        self.assertEqual(reset.elapsed_time_s, 0.0)
        self.assertEqual(reset.distance_travelled_m, 0.0)
        self.assertEqual(reset.energy.total.positive_j, 0.0)
        self.assertEqual(reset.energy.total.signed_j, 0.0)

    def test_runtime_profiler_statistics_and_reset(self) -> None:
        profiler = RuntimeProfiler()
        for elapsed_s in (0.001, 0.002, 0.003, 0.004):
            profiler.record_seconds("vehicle_physics", elapsed_s)
        stats = profiler.summary()["vehicle_physics"]
        self.assertEqual(stats.sample_count, 4)
        self.assertAlmostEqual(stats.mean_ms, 2.5)
        self.assertAlmostEqual(stats.p50_ms, 2.5)
        self.assertAlmostEqual(stats.p95_ms, 3.85)
        self.assertEqual(stats.max_ms, 4.0)
        profiler.reset()
        self.assertEqual(profiler.summary(), {})
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            profiler.record_seconds("bad", np.nan)

    def test_slope_trials_require_same_throttle_and_preserve_metrics(self) -> None:
        def trial(slope: float) -> SlopeTrialMetrics:
            return SlopeTrialMetrics(
                slope_deg=slope,
                throttle=0.5,
                mean_speed_m_s=2.0 - slope / 20.0,
                mean_abs_drive_torque_nm=100.0 + slope,
                mean_abs_slip_ratio=slope / 100.0,
                positive_drive_energy_j=1_000.0 + slope * 10.0,
                travel_time_s=5.0 + slope / 10.0,
            )

        ordered = validate_comparable_slope_trials(
            [trial(20.0), trial(0.0), trial(10.0)]
        )
        self.assertEqual([item.slope_deg for item in ordered], [0.0, 10.0, 20.0])
        incompatible = SlopeTrialMetrics(
            slope_deg=30.0,
            throttle=0.7,
            mean_speed_m_s=1.0,
            mean_abs_drive_torque_nm=1.0,
            mean_abs_slip_ratio=0.1,
            positive_drive_energy_j=1.0,
            travel_time_s=1.0,
        )
        with self.assertRaisesRegex(ValueError, "same throttle"):
            validate_comparable_slope_trials([trial(0.0), incompatible])

    def test_phase_d_yaml_declares_provenance_and_effort_precedence(self) -> None:
        path = Path(__file__).parents[1] / "configs" / "phase_d_telemetry.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))[
            "phase_d_telemetry"
        ]
        self.assertEqual(config["schema_version"], "phase_d.vehicle_telemetry.v1")
        self.assertEqual(
            config["actuator_power"]["effort_precedence"],
            ["measured_effort", "applied_effort_command", "estimated_effort"],
        )
        self.assertFalse(
            config["actuator_power"]["regenerative_efficiency_assumed"]
        )


if __name__ == "__main__":
    unittest.main()
