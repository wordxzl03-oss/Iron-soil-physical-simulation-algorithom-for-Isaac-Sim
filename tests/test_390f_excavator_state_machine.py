import unittest
from dataclasses import replace

import numpy as np

from isaac_bulk_pipeline.operation import (
    ExcavatorCycleConfig,
    ExcavatorCycleObservation,
    ExcavatorCycleState,
    ExcavatorCycleStateMachine,
)


TARGET_KEYS = (
    "initial_pose", "approach_pile", "penetrate", "coordinated_cut",
    "breakout", "lift", "upper_body_swing", "dump_spill", "swing_back",
    "next_dig_ready",
)


def config(timeout=10.0):
    targets = {name: np.full(4, index * 0.1) for index, name in enumerate(TARGET_KEYS)}
    return ExcavatorCycleConfig(
        targets,
        state_timeout_s=timeout,
        joint_tolerance_rad=1.0e-4,
        penetration_depth_m=0.2,
        minimum_intersection_m3=0.01,
        minimum_cut_distance_m=0.25,
        minimum_payload_gain_m3=0.02,
        breakout_lip_clearance_m=0.05,
        transport_lip_clearance_m=1.0,
        reverse_distance_m=2.0,
        minimum_dump_release_m3=0.01,
        minimum_deposition_gain_m3=0.005,
    )


def observation(cfg, state, time, *, base=(0.0, 0.0, 0.0), depth=0.0,
                intersection=0.0, cut_distance=0.0, clearance=0.0,
                payload=0.0, deposited=0.0, mobile=0.0, airborne=0.0):
    target_key = ExcavatorCycleStateMachine._TARGET_BY_STATE.get(state, "initial_pose")
    return ExcavatorCycleObservation(
        time, cfg.phase_targets_rad[target_key], np.asarray(base), depth,
        intersection, cut_distance, clearance, payload, deposited, mobile,
        airborne,
    )


class ExcavatorCycleStateMachineTests(unittest.TestCase):
    def test_navigation_pivots_before_forward_travel_and_never_chases_in_reverse(self):
        cfg = config()
        machine = ExcavatorCycleStateMachine(
            cfg, np.zeros(3), np.array([5.0, 0.0, 0.0])
        )

        machine._last_base_pose = np.array([0.0, 0.0, 0.0])
        left, right = machine._navigation_commands(np.array([0.0, 5.0, 0.0]))
        self.assertAlmostEqual(left + right, 0.0)
        self.assertLess(left, 0.0)
        self.assertGreater(right, 0.0)

        left, right = machine._navigation_commands(np.array([-5.0, 0.0, 0.0]))
        self.assertAlmostEqual(left + right, 0.0)
        self.assertTrue(left * right < 0.0)

        left, right = machine._navigation_commands(np.array([5.0, 0.5, 0.0]))
        self.assertGreater(left + right, 0.0)

    def test_full_cycle_requires_observed_physical_completion(self):
        cfg = config()
        machine = ExcavatorCycleStateMachine(cfg, np.array([0.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]))
        first = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(first)
        self.assertEqual(machine.step(first).state, ExcavatorCycleState.APPROACH)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.APPROACH, 0.1)).state, ExcavatorCycleState.PENETRATE)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, 0.2, depth=0.2, intersection=0.01)).state, ExcavatorCycleState.CUT_AND_FILL)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, 0.3, cut_distance=0.3, payload=0.03)).state, ExcavatorCycleState.CURL_AND_BREAKOUT)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, 0.4, clearance=0.06, payload=0.03)).state, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, 0.5, clearance=1.1, payload=0.03)).state, ExcavatorCycleState.REVERSE_TRAVEL)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.REVERSE_TRAVEL, 0.6, base=(-2.1, 0.0, 0.0), payload=0.03)).state, ExcavatorCycleState.ALIGN_DUMP)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.ALIGN_DUMP, 0.7, base=(5.0, 0.0, 0.0), payload=0.03)).state, ExcavatorCycleState.DUMP)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.DUMP, 0.8, base=(5.0, 0.0, 0.0), payload=0.015)).state, ExcavatorCycleState.DEPOSITION)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.DEPOSITION, 0.9, base=(5.0, 0.0, 0.0), payload=0.015, deposited=0.006)).state, ExcavatorCycleState.BUCKET_RECOVERY)
        self.assertEqual(machine.step(observation(cfg, ExcavatorCycleState.BUCKET_RECOVERY, 1.0, payload=0.015, deposited=0.006)).state, ExcavatorCycleState.RETURN_TRAVEL)
        still_settling = replace(
            observation(
                cfg,
                ExcavatorCycleState.RETURN_TRAVEL,
                1.1,
                payload=0.015,
                deposited=0.006,
            ),
            terrain_settled=False,
        )
        self.assertEqual(
            machine.step(still_settling).state,
            ExcavatorCycleState.RETURN_TRAVEL,
        )
        decision = machine.step(
            replace(still_settling, timestamp_s=1.2, terrain_settled=True)
        )
        self.assertEqual(decision.state, ExcavatorCycleState.READY_NEXT_CYCLE)

    def test_return_has_separate_navigation_timeout_then_waits_for_settling(self):
        cfg = replace(config(timeout=1.0), return_travel_timeout_s=3.0)
        machine = ExcavatorCycleStateMachine(
            cfg, np.zeros(3), np.array([5.0, 0.0, 0.0])
        )
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial); machine.step(initial)
        machine.step(observation(cfg, ExcavatorCycleState.APPROACH, .1))
        machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, .2, depth=.2, intersection=.01))
        machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, .3, cut_distance=.3, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, .4, clearance=.06, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, .5, clearance=1.1, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.REVERSE_TRAVEL, .6, base=(-2.1, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.ALIGN_DUMP, .7, base=(5, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.DUMP, .8, base=(5, 0, 0), payload=.015))
        machine.step(observation(cfg, ExcavatorCycleState.DEPOSITION, .9, base=(5, 0, 0), payload=.015, deposited=.006))
        machine.step(observation(cfg, ExcavatorCycleState.BUCKET_RECOVERY, 1.0, payload=.015, deposited=.006))

        # Longer than the generic one-second phase timeout, but still inside
        # the explicit force-driven return navigation allowance.
        navigating = machine.step(
            observation(
                cfg,
                ExcavatorCycleState.RETURN_TRAVEL,
                2.2,
                base=(1.0, 0.0, 0.0),
                payload=.015,
                deposited=.006,
            )
        )
        self.assertEqual(navigating.state, ExcavatorCycleState.RETURN_TRAVEL)
        self.assertIsNone(navigating.failure)

        # At the exact stop target, dynamic terrain may finish asynchronously
        # without being mislabeled as a navigation timeout.
        settling = replace(
            observation(
                cfg,
                ExcavatorCycleState.RETURN_TRAVEL,
                5.0,
                payload=.015,
                deposited=.006,
            ),
            terrain_settled=False,
        )
        self.assertEqual(machine.step(settling).state, ExcavatorCycleState.RETURN_TRAVEL)
        self.assertEqual(
            machine.step(
                replace(settling, timestamp_s=5.1, terrain_settled=True)
            ).state,
            ExcavatorCycleState.READY_NEXT_CYCLE,
        )

    def test_timeout_is_failure_and_never_success_transition(self):
        cfg = config(timeout=1.0)
        machine = ExcavatorCycleStateMachine(cfg, np.zeros(3), np.array([5.0, 0.0, 0.0]))
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial)
        machine.step(initial)
        wrong_q = ExcavatorCycleObservation(1.1, np.full(4, 99.0), np.zeros(3), 0, 0, 0, 0, 0, 0, 0, 0)
        decision = machine.step(wrong_q)
        self.assertEqual(decision.state, ExcavatorCycleState.FAILED)
        self.assertEqual(decision.failure.code, "APPROACH_TIMEOUT")
        self.assertEqual(decision.transition_reason, "CYCLE_FAIL")

    def test_dump_cannot_complete_with_empty_or_unchanged_payload(self):
        cfg = config(timeout=2.0)
        machine = ExcavatorCycleStateMachine(cfg, np.zeros(3), np.array([5.0, 0.0, 0.0]))
        # Reach DUMP with payload, then keep it unchanged.
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial); machine.step(initial)
        machine.step(observation(cfg, ExcavatorCycleState.APPROACH, 0.1))
        machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, 0.2, depth=.2, intersection=.01))
        machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, 0.3, cut_distance=.3, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, 0.4, clearance=.06, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, 0.5, clearance=1.1, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.REVERSE_TRAVEL, 0.6, base=(-2.1, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.ALIGN_DUMP, 0.7, base=(5, 0, 0), payload=.03))
        decision = machine.step(observation(cfg, ExcavatorCycleState.DUMP, 0.8, base=(5, 0, 0), payload=.03))
        self.assertEqual(decision.state, ExcavatorCycleState.DUMP)
        self.assertFalse(decision.transitioned)

    def test_deposition_uses_dump_entry_baseline_when_landing_precedes_transition(self):
        cfg = config()
        machine = ExcavatorCycleStateMachine(
            cfg, np.zeros(3), np.array([5.0, 0.0, 0.0])
        )
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial); machine.step(initial)
        machine.step(observation(cfg, ExcavatorCycleState.APPROACH, 0.1))
        machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, 0.2, depth=.2, intersection=.01))
        machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, 0.3, cut_distance=.3, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, 0.4, clearance=.06, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, 0.5, clearance=1.1, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.REVERSE_TRAVEL, 0.6, base=(-2.1, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.ALIGN_DUMP, 0.7, base=(5, 0, 0), payload=.03, deposited=0.0))
        entered = machine.step(observation(cfg, ExcavatorCycleState.DUMP, 0.8, base=(5, 0, 0), payload=.015, deposited=.006))
        self.assertEqual(entered.state, ExcavatorCycleState.DEPOSITION)
        completed = machine.step(observation(cfg, ExcavatorCycleState.DEPOSITION, 0.9, base=(5, 0, 0), payload=.015, deposited=.006, airborne=0.0))
        self.assertEqual(completed.state, ExcavatorCycleState.BUCKET_RECOVERY)

    def test_deposition_waits_for_physical_terrain_settled_gate(self):
        cfg = config()
        machine = ExcavatorCycleStateMachine(
            cfg, np.zeros(3), np.array([5.0, 0.0, 0.0])
        )
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial); machine.step(initial)
        machine.step(observation(cfg, ExcavatorCycleState.APPROACH, 0.1))
        machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, 0.2, depth=.2, intersection=.01))
        machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, 0.3, cut_distance=.3, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, 0.4, clearance=.06, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, 0.5, clearance=1.1, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.REVERSE_TRAVEL, 0.6, base=(-2.1, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.ALIGN_DUMP, 0.7, base=(5, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.DUMP, 0.8, base=(5, 0, 0), payload=.015))
        moving = observation(
            cfg,
            ExcavatorCycleState.DEPOSITION,
            0.9,
            base=(5, 0, 0),
            payload=.015,
            deposited=.006,
            airborne=0.0,
        )
        moving = replace(moving, terrain_settled=False)
        self.assertEqual(machine.step(moving).state, ExcavatorCycleState.DEPOSITION)
        self.assertEqual(
            machine.step(replace(moving, timestamp_s=1.0, terrain_settled=True)).state,
            ExcavatorCycleState.BUCKET_RECOVERY,
        )

    def test_active_mobile_avalanche_is_not_misclassified_as_phase_timeout(self):
        cfg = config(timeout=1.0)
        machine = ExcavatorCycleStateMachine(
            cfg, np.zeros(3), np.array([5.0, 0.0, 0.0])
        )
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial); machine.step(initial)
        machine.step(observation(cfg, ExcavatorCycleState.APPROACH, 0.1))
        machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, 0.2, depth=.2, intersection=.01))
        machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, 0.3, cut_distance=.3, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, 0.4, clearance=.06, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, 0.5, clearance=1.1, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.REVERSE_TRAVEL, 0.6, base=(-2.1, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.ALIGN_DUMP, 0.7, base=(5, 0, 0), payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.DUMP, 0.8, base=(5, 0, 0), payload=.015))
        moving = replace(
            observation(
                cfg,
                ExcavatorCycleState.DEPOSITION,
                3.0,
                base=(5, 0, 0),
                payload=.015,
                deposited=.006,
                airborne=0.0,
            ),
            terrain_settled=False,
        )
        decision = machine.step(moving)
        self.assertEqual(decision.state, ExcavatorCycleState.DEPOSITION)
        self.assertIsNone(decision.failure)

    def test_track_commands_are_physical_effort_requests_not_pose_targets(self):
        cfg = config()
        machine = ExcavatorCycleStateMachine(cfg, np.zeros(3), np.array([5.0, 0.0, 0.0]))
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial)
        approach = machine.step(initial)
        self.assertGreater(approach.command.left_track_effort_fraction, 0.0)
        self.assertGreater(approach.command.right_track_effort_fraction, 0.0)
        self.assertFalse(hasattr(approach.command, "root_pose"))

    def test_dynamic_dump_target_uses_measured_reverse_completion_pose(self):
        cfg = replace(config(), dynamic_dump_from_reverse_entry=True)
        machine = ExcavatorCycleStateMachine(
            cfg, np.zeros(3), np.array([5.0, 0.0, 0.0])
        )
        initial = observation(cfg, ExcavatorCycleState.READY_AT_DIG_POSITION, 0.0)
        machine.start(initial); machine.step(initial)
        machine.step(observation(cfg, ExcavatorCycleState.APPROACH, .1))
        machine.step(observation(cfg, ExcavatorCycleState.PENETRATE, .2, depth=.2, intersection=.01))
        machine.step(observation(cfg, ExcavatorCycleState.CUT_AND_FILL, .3, cut_distance=.3, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.CURL_AND_BREAKOUT, .4, clearance=.06, payload=.03))
        machine.step(observation(cfg, ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, .5, clearance=1.1, payload=.03))
        completed_reverse_pose = np.array([-2.2, 0.3, 0.15])
        decision = machine.step(
            observation(
                cfg,
                ExcavatorCycleState.REVERSE_TRAVEL,
                .6,
                base=completed_reverse_pose,
                payload=.03,
            )
        )
        self.assertEqual(decision.state, ExcavatorCycleState.ALIGN_DUMP)
        np.testing.assert_allclose(machine.dump_pose, completed_reverse_pose)
        self.assertEqual(decision.command.left_track_effort_fraction, 0.0)
        self.assertEqual(decision.command.right_track_effort_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
