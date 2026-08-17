from __future__ import annotations

import ast
from pathlib import Path
import unittest

import numpy as np

from isaac_bulk_pipeline.operation import (
    LoaderOperationConfig,
    LoaderOperationState,
    LoaderOperationStateMachine,
    OperationObservation,
    OperationTargets,
)
from isaac_bulk_pipeline.vehicle import VehicleCommand


ROOT = Path(__file__).resolve().parents[1]


def observation(
    t: float,
    pose=(0.0, 0.0, 0.0),
    *,
    penetration=0.0,
    fill=0.0,
    payload=0.0,
    mobile=0.0,
    parcels=0,
) -> OperationObservation:
    return OperationObservation(
        timestamp_s=t,
        vehicle_pose_xy_yaw=np.asarray(pose, dtype=float),
        longitudinal_speed_m_s=0.0,
        lift_angle_rad=0.0,
        bucket_angle_rad=0.0,
        penetration_depth_m=penetration,
        payload_fill_ratio=fill,
        payload_volume_m3=payload,
        mobile_volume_m3=mobile,
        airborne_parcel_count=parcels,
    )


class TestPhaseIOperation(unittest.TestCase):
    def setUp(self) -> None:
        self.targets = OperationTargets(
            pre_dig_pose_xy_yaw=np.array([1.0, 0.0, 0.0]),
            dig_pose_xy_yaw=np.array([2.0, 0.0, 0.0]),
            dump_pose_xy_yaw=np.array([5.0, 1.0, 0.0]),
            return_pose_xy_yaw=np.array([0.0, 0.0, np.pi]),
        )
        self.machine = LoaderOperationStateMachine(
            self.targets,
            LoaderOperationConfig(reverse_distance_m=1.0),
        )

    def test_idle_start_and_reset(self) -> None:
        idle = self.machine.step(observation(0.0))
        self.assertEqual(idle.state, LoaderOperationState.IDLE)
        self.assertEqual(idle.command, VehicleCommand(brake=1.0))
        self.machine.start(0.1)
        self.assertEqual(self.machine.state, LoaderOperationState.DRIVE_TO_PRE_DIG)
        self.machine.reset()
        self.assertEqual(self.machine.state, LoaderOperationState.IDLE)

    def test_observation_driven_complete_cycle(self) -> None:
        self.machine.start(0.0)
        states = []

        def take(obs: OperationObservation) -> None:
            decision = self.machine.step(obs)
            self.assertIsInstance(decision.command, VehicleCommand)
            states.append(decision.state)

        take(observation(0.1, (1.0, 0.0, 0.0)))  # pre-dig -> ALIGN
        take(observation(0.2, (1.0, 0.0, 0.0)))  # ALIGN -> APPROACH
        take(observation(0.3, (2.0, 0.0, 0.0)))  # APPROACH -> PENETRATE
        take(observation(0.4, (2.0, 0.0, 0.0), penetration=0.5))
        take(observation(0.5, (2.0, 0.0, 0.0), penetration=0.5, fill=0.9, payload=1.0))
        take(observation(2.6, (2.0, 0.0, 0.0), fill=0.9, payload=1.0))
        take(observation(3.7, (2.0, 0.0, 0.0), fill=0.9, payload=1.0))
        take(observation(3.8, (0.8, 0.0, 0.0), fill=0.9, payload=1.0))
        take(observation(3.9, (5.0, 1.0, 0.0), fill=0.9, payload=1.0))
        take(observation(4.0, (5.0, 1.0, 0.0), fill=0.0, payload=0.0))
        take(observation(4.1, (0.0, 0.0, np.pi), payload=0.0))
        take(observation(4.7, (0.0, 0.0, np.pi), payload=0.0, mobile=0.0))

        self.assertEqual(
            states,
            [
                LoaderOperationState.ALIGN,
                LoaderOperationState.APPROACH,
                LoaderOperationState.PENETRATE,
                LoaderOperationState.FILL,
                LoaderOperationState.CURL_LIFT,
                LoaderOperationState.BREAKOUT,
                LoaderOperationState.REVERSE,
                LoaderOperationState.TRANSPORT,
                LoaderOperationState.DUMP,
                LoaderOperationState.RETURN,
                LoaderOperationState.SETTLE,
                LoaderOperationState.DONE,
            ],
        )

    def test_commands_preserve_steering_channel(self) -> None:
        self.machine.start(0.0)
        decision = self.machine.step(observation(0.1, (0.0, 0.0, np.pi / 2)))
        self.assertNotEqual(decision.command.steering, 0.0)
        self.assertEqual(decision.command.brake, 0.0)

    def test_source_contains_no_runtime_pose_mutation(self) -> None:
        path = ROOT / "src/isaac_bulk_pipeline/operation/state_machine.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        forbidden = {"set_world_pose", "set_local_pose", "SetTranslate", "SetRotate"}
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden.isdisjoint(calls))
        self.assertNotIn("six_scoop", source.lower())

    def test_timestamp_must_be_monotonic(self) -> None:
        self.machine.start(1.0)
        with self.assertRaises(ValueError):
            self.machine.step(observation(0.9))


if __name__ == "__main__":
    unittest.main()
