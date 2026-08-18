"""Observation-driven 390F task cycle with explicit failure semantics.

The state machine is independent of Isaac.  It emits articulation targets and
left/right track effort fractions; it never writes an articulation or root
pose.  A runtime timeout is a failed physical condition, never a successful
phase transition.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np


JOINT_NAMES_390F = ("swing_joint", "boom_joint", "stick_joint", "bucket_joint")


class ExcavatorCycleState(str, Enum):
    IDLE = "IDLE"
    READY_AT_DIG_POSITION = "READY_AT_DIG_POSITION"
    APPROACH = "APPROACH"
    PENETRATE = "PENETRATE"
    CUT_AND_FILL = "CUT_AND_FILL"
    CURL_AND_BREAKOUT = "CURL_AND_BREAKOUT"
    LIFT_TO_TRANSPORT_HEIGHT = "LIFT_TO_TRANSPORT_HEIGHT"
    REVERSE_TRAVEL = "REVERSE_TRAVEL"
    ALIGN_DUMP = "ALIGN_DUMP"
    DUMP = "DUMP"
    DEPOSITION = "DEPOSITION"
    BUCKET_RECOVERY = "BUCKET_RECOVERY"
    RETURN_TRAVEL = "RETURN_TRAVEL"
    READY_NEXT_CYCLE = "READY_NEXT_CYCLE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExcavatorCycleConfig:
    phase_targets_rad: Mapping[str, np.ndarray]
    state_timeout_s: float = 20.0
    return_travel_timeout_s: float | None = None
    joint_tolerance_rad: float = np.deg2rad(3.0)
    ready_position_tolerance_m: float = 0.40
    penetration_depth_m: float = 0.20
    # Full-width engagement gates.  The old controller advanced to CUT from
    # one deeply penetrated corner because it only used maximum edge depth.
    # These thresholds require a material fraction of the cutting edge to be
    # engaged before excavation can start.
    minimum_mean_penetration_depth_m: float = 0.10
    minimum_edge_engaged_fraction: float = 0.70
    edge_engagement_depth_m: float = 0.025
    minimum_intersection_m3: float = 1.0e-4
    minimum_cut_distance_m: float = 0.25
    minimum_payload_gain_m3: float = 0.02
    breakout_lip_clearance_m: float = 0.05
    transport_lip_clearance_m: float = 1.0
    reverse_distance_m: float = 2.0
    dump_position_tolerance_m: float = 0.75
    minimum_dump_release_m3: float = 0.01
    empty_payload_tolerance_m3: float = 1.0e-4
    minimum_deposition_gain_m3: float = 0.005
    deposition_mobile_tolerance_m3: float = 0.02
    approach_track_command: float = 0.22
    reverse_track_command: float = -0.65
    travel_track_command: float = 0.65
    dynamic_dump_from_reverse_entry: bool = False
    navigation_distance_gain: float = 0.55
    navigation_heading_gain: float = 0.75
    navigation_drive_heading_gate_rad: float = np.deg2rad(35.0)

    def __post_init__(self) -> None:
        targets: dict[str, np.ndarray] = {}
        for name, value in self.phase_targets_rad.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (4,) or not np.all(np.isfinite(array)):
                raise ValueError(f"[390FCycle] target {name} must be finite shape (4,)")
            copy = np.ascontiguousarray(array.copy())
            copy.setflags(write=False)
            targets[str(name)] = copy
        required = {
            "initial_pose", "approach_pile", "penetrate", "coordinated_cut",
            "breakout", "lift", "upper_body_swing", "dump_spill",
            "swing_back", "next_dig_ready",
        }
        missing = sorted(required - targets.keys())
        if missing:
            raise ValueError(f"[390FCycle] missing phase targets: {missing}")
        object.__setattr__(self, "phase_targets_rad", MappingProxyType(targets))
        positive = np.asarray(
            [
                self.state_timeout_s, self.joint_tolerance_rad,
                self.ready_position_tolerance_m, self.penetration_depth_m,
                self.minimum_mean_penetration_depth_m,
                self.edge_engagement_depth_m,
                self.minimum_intersection_m3, self.minimum_cut_distance_m,
                self.minimum_payload_gain_m3, self.breakout_lip_clearance_m,
                self.transport_lip_clearance_m, self.reverse_distance_m,
                self.dump_position_tolerance_m, self.minimum_dump_release_m3,
                self.empty_payload_tolerance_m3, self.minimum_deposition_gain_m3,
                self.deposition_mobile_tolerance_m3,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("[390FCycle] physical thresholds must be finite/positive")
        if not np.isfinite(self.minimum_edge_engaged_fraction) or not (
            0.0 < self.minimum_edge_engaged_fraction <= 1.0
        ):
            raise ValueError(
                "[390FCycle] minimum_edge_engaged_fraction must lie in (0,1]"
            )
        if self.return_travel_timeout_s is not None and (
            not np.isfinite(self.return_travel_timeout_s)
            or self.return_travel_timeout_s <= 0.0
        ):
            raise ValueError(
                "[390FCycle] return_travel_timeout_s must be finite/positive"
            )
        commands = np.asarray(
            [
                self.approach_track_command,
                self.reverse_track_command,
                self.travel_track_command,
                self.navigation_distance_gain,
                self.navigation_heading_gain,
                self.navigation_drive_heading_gate_rad,
            ]
        )
        if not np.all(np.isfinite(commands)) or np.any(commands[-2:] <= 0.0):
            raise ValueError("[390FCycle] track commands/gains must be finite and gains positive")
        if np.any(np.abs(commands[:3]) > 1.0):
            raise ValueError("[390FCycle] track commands must be in [-1,1]")
        if not 0.0 < self.navigation_drive_heading_gate_rad <= 0.5 * np.pi:
            raise ValueError(
                "[390FCycle] navigation_drive_heading_gate_rad must be in (0,pi/2]"
            )


@dataclass(frozen=True)
class ExcavatorCycleObservation:
    timestamp_s: float
    joint_position_rad: np.ndarray
    base_pose_xy_yaw: np.ndarray
    cutting_edge_depth_m: float
    tool_terrain_intersection_m3: float
    cutting_distance_m: float
    cutting_lip_clearance_m: float
    payload_volume_m3: float
    deposited_volume_m3: float
    mobile_volume_m3: float
    airborne_volume_m3: float
    terrain_settled: bool = True
    # Full-width cutting-edge diagnostics.  Defaults preserve compatibility
    # with non-production tests/callers that do not yet publish them.
    cutting_edge_mean_depth_m: float = 0.0
    cutting_edge_engaged_fraction: float = 0.0
    cutting_edge_left_depth_m: float = 0.0
    cutting_edge_center_depth_m: float = 0.0
    cutting_edge_right_depth_m: float = 0.0

    def __post_init__(self) -> None:
        for name, shape in (("joint_position_rad", (4,)), ("base_pose_xy_yaw", (3,))):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"[390FCycle] {name} must be finite shape {shape}")
            copy = np.ascontiguousarray(array.copy())
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        scalars = np.asarray(
            [
                self.timestamp_s, self.cutting_edge_depth_m,
                self.tool_terrain_intersection_m3, self.cutting_distance_m,
                self.cutting_lip_clearance_m, self.payload_volume_m3,
                self.deposited_volume_m3, self.mobile_volume_m3,
                self.airborne_volume_m3,
                self.cutting_edge_mean_depth_m,
                self.cutting_edge_engaged_fraction,
                self.cutting_edge_left_depth_m,
                self.cutting_edge_center_depth_m,
                self.cutting_edge_right_depth_m,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or self.timestamp_s < 0.0:
            raise ValueError("[390FCycle] observation contains non-finite/negative time")
        nonnegative = scalars[[2, 3, 5, 6, 7, 8]]
        if np.any(nonnegative < 0.0):
            raise ValueError("[390FCycle] observed volumes/distances must be non-negative")
        if not 0.0 <= self.cutting_edge_engaged_fraction <= 1.0:
            raise ValueError(
                "[390FCycle] cutting_edge_engaged_fraction must lie in [0,1]"
            )
        if not isinstance(self.terrain_settled, (bool, np.bool_)):
            raise ValueError("[390FCycle] terrain_settled must be boolean")
        object.__setattr__(self, "terrain_settled", bool(self.terrain_settled))


@dataclass(frozen=True)
class ExcavatorCycleCommand:
    joint_target_rad: np.ndarray
    left_track_effort_fraction: float = 0.0
    right_track_effort_fraction: float = 0.0
    hold_brake: bool = False

    def __post_init__(self) -> None:
        target = np.asarray(self.joint_target_rad, dtype=np.float64)
        if target.shape != (4,) or not np.all(np.isfinite(target)):
            raise ValueError("[390FCycle] joint target must be finite shape (4,)")
        if not np.all(np.isfinite([self.left_track_effort_fraction, self.right_track_effort_fraction])):
            raise ValueError("[390FCycle] track command is non-finite")
        if max(abs(self.left_track_effort_fraction), abs(self.right_track_effort_fraction)) > 1.0:
            raise ValueError("[390FCycle] track effort fractions must be in [-1,1]")
        copy = np.ascontiguousarray(target.copy()); copy.setflags(write=False)
        object.__setattr__(self, "joint_target_rad", copy)


@dataclass(frozen=True)
class ExcavatorCycleFailure:
    code: str
    state: ExcavatorCycleState
    elapsed_s: float
    message: str


@dataclass(frozen=True)
class ExcavatorCycleDecision:
    state: ExcavatorCycleState
    command: ExcavatorCycleCommand
    transitioned: bool
    transition_reason: str | None
    failure: ExcavatorCycleFailure | None
    state_elapsed_s: float
    completion_condition: str


class ExcavatorCycleStateMachine:
    """Formal 390F state machine with observation-based completion gates."""

    _TARGET_BY_STATE = {
        ExcavatorCycleState.READY_AT_DIG_POSITION: "initial_pose",
        ExcavatorCycleState.APPROACH: "approach_pile",
        ExcavatorCycleState.PENETRATE: "penetrate",
        ExcavatorCycleState.CUT_AND_FILL: "coordinated_cut",
        ExcavatorCycleState.CURL_AND_BREAKOUT: "breakout",
        ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT: "lift",
        ExcavatorCycleState.REVERSE_TRAVEL: "lift",
        ExcavatorCycleState.ALIGN_DUMP: "upper_body_swing",
        ExcavatorCycleState.DUMP: "dump_spill",
        ExcavatorCycleState.DEPOSITION: "dump_spill",
        ExcavatorCycleState.BUCKET_RECOVERY: "swing_back",
        ExcavatorCycleState.RETURN_TRAVEL: "next_dig_ready",
        ExcavatorCycleState.READY_NEXT_CYCLE: "next_dig_ready",
    }

    _FAILURE_BY_STATE = {
        state: f"{state.value}_TIMEOUT"
        for state in ExcavatorCycleState
        if state not in {ExcavatorCycleState.IDLE, ExcavatorCycleState.FAILED, ExcavatorCycleState.READY_NEXT_CYCLE}
    }

    _COMPLETION_CONDITION_BY_STATE = {
        ExcavatorCycleState.IDLE: "explicit user start command",
        ExcavatorCycleState.READY_AT_DIG_POSITION: "joint target and base at dig pose",
        ExcavatorCycleState.APPROACH: "approach joint target reached",
        ExcavatorCycleState.PENETRATE: (
            "full-width cutting-edge engagement, mean/max depth and tool-terrain intersection"
        ),
        ExcavatorCycleState.CUT_AND_FILL: "cut distance and measured payload gain",
        ExcavatorCycleState.CURL_AND_BREAKOUT: "curl target and cutting lip clear of active pile",
        ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT: "lift target and safe lip clearance",
        ExcavatorCycleState.REVERSE_TRAVEL: "measured base reverse displacement",
        ExcavatorCycleState.ALIGN_DUMP: "swing target and measured dump position",
        ExcavatorCycleState.DUMP: "dump orientation and measured payload release",
        ExcavatorCycleState.DEPOSITION: (
            "dump-origin airborne parcels settled and terrain below physical "
            "mobile/instability stop criteria"
        ),
        ExcavatorCycleState.BUCKET_RECOVERY: "recovery joint target reached",
        ExcavatorCycleState.RETURN_TRAVEL: "joint target and measured return to dig pose",
        ExcavatorCycleState.READY_NEXT_CYCLE: "cycle physically complete",
        ExcavatorCycleState.FAILED: "reset required; scene retained",
    }

    def __init__(self, config: ExcavatorCycleConfig, dig_pose_xy_yaw: np.ndarray, dump_pose_xy_yaw: np.ndarray) -> None:
        self.config = config
        self.dig_pose = self._pose(dig_pose_xy_yaw, "dig_pose_xy_yaw")
        self.dump_pose = self._pose(dump_pose_xy_yaw, "dump_pose_xy_yaw")
        self._state = ExcavatorCycleState.IDLE
        self._state_start_s = 0.0
        self._last_timestamp_s: float | None = None
        self._entry_payload_m3 = 0.0
        self._dump_entry_payload_m3 = 0.0
        self._entry_base_xy = np.zeros(2)
        self._failure: ExcavatorCycleFailure | None = None
        self._last_base_pose = np.array(self.dig_pose, copy=True)

    @property
    def state(self) -> ExcavatorCycleState:
        return self._state

    @property
    def failure(self) -> ExcavatorCycleFailure | None:
        return self._failure

    def start(self, observation: ExcavatorCycleObservation) -> None:
        if self._state not in {ExcavatorCycleState.IDLE, ExcavatorCycleState.READY_NEXT_CYCLE}:
            raise RuntimeError("[390FCycle] start requires IDLE or READY_NEXT_CYCLE")
        self._last_timestamp_s = observation.timestamp_s
        self._failure = None
        self._dump_entry_payload_m3 = 0.0
        self._last_base_pose = np.array(observation.base_pose_xy_yaw, copy=True)
        self._enter(ExcavatorCycleState.READY_AT_DIG_POSITION, observation)

    def reset(self) -> None:
        self._state = ExcavatorCycleState.IDLE
        self._state_start_s = 0.0
        self._last_timestamp_s = None
        self._failure = None
        self._last_base_pose = np.array(self.dig_pose, copy=True)

    def step(self, observation: ExcavatorCycleObservation) -> ExcavatorCycleDecision:
        if self._last_timestamp_s is not None and observation.timestamp_s < self._last_timestamp_s:
            raise ValueError("[390FCycle] observation timestamp moved backwards")
        self._last_timestamp_s = observation.timestamp_s
        self._last_base_pose = np.array(observation.base_pose_xy_yaw, copy=True)
        elapsed = max(0.0, observation.timestamp_s - self._state_start_s)
        if self._state in {ExcavatorCycleState.IDLE, ExcavatorCycleState.FAILED, ExcavatorCycleState.READY_NEXT_CYCLE}:
            return self._decision(elapsed, False, None)
        next_state, reason = self._completion(observation)
        if next_state is not None:
            self._enter(next_state, observation)
            return self._decision(0.0, True, reason)
        # A sustained avalanche advances in physical simulation time through
        # the Mobile Layer.  It is not a stalled actuator/task condition and
        # must not be converted into a phase timeout merely because settling
        # takes longer than an arm motion.  Once the terrain reports the
        # physical stop criteria, the normal timeout remains applicable to a
        # missing deposition/completion observation.
        return_stopped_at_target = bool(
            self._state is ExcavatorCycleState.RETURN_TRAVEL
            and self._joint_target_reached(observation)
            and np.linalg.norm(
                observation.base_pose_xy_yaw[:2] - self.dig_pose[:2]
            )
            <= 0.05
        )
        terrain_dynamically_advancing = bool(
            not observation.terrain_settled
            and (
                self._state is ExcavatorCycleState.DEPOSITION
                or return_stopped_at_target
            )
        )
        timeout_s = (
            self.config.return_travel_timeout_s
            if self._state is ExcavatorCycleState.RETURN_TRAVEL
            and self.config.return_travel_timeout_s is not None
            else self.config.state_timeout_s
        )
        if elapsed >= timeout_s and not terrain_dynamically_advancing:
            code = self._FAILURE_BY_STATE[self._state]
            failed_state = self._state
            self._failure = ExcavatorCycleFailure(
                code=code,
                state=failed_state,
                elapsed_s=elapsed,
                message=f"physical completion condition not met in {elapsed:.3f} s",
            )
            self._state = ExcavatorCycleState.FAILED
            return self._decision(elapsed, True, "CYCLE_FAIL")
        return self._decision(elapsed, False, None)

    def _completion(self, obs: ExcavatorCycleObservation) -> tuple[ExcavatorCycleState | None, str | None]:
        c = self.config
        state = self._state
        at_target = self._joint_target_reached(obs)
        if state is ExcavatorCycleState.READY_AT_DIG_POSITION:
            if at_target and self._position_near(obs.base_pose_xy_yaw, self.dig_pose, c.ready_position_tolerance_m):
                return ExcavatorCycleState.APPROACH, "dig_ready_pose_verified"
        elif state is ExcavatorCycleState.APPROACH:
            if at_target:
                return ExcavatorCycleState.PENETRATE, "approach_configuration_reached"
        elif state is ExcavatorCycleState.PENETRATE:
            # A single deeply buried tooth/corner is not a valid excavation
            # engagement.  Require a substantial fraction of the full cutting
            # edge plus a meaningful mean depth before the cut controller owns
            # the motion.  This is resolution-aware through the configured
            # edge_engagement_depth_m published by the runner.
            if (
                obs.cutting_edge_depth_m >= c.penetration_depth_m
                and obs.cutting_edge_mean_depth_m
                >= c.minimum_mean_penetration_depth_m
                and obs.cutting_edge_engaged_fraction
                >= c.minimum_edge_engaged_fraction
                and obs.tool_terrain_intersection_m3 >= c.minimum_intersection_m3
            ):
                return (
                    ExcavatorCycleState.CUT_AND_FILL,
                    "full_width_depth_and_intersection_verified",
                )
        elif state is ExcavatorCycleState.CUT_AND_FILL:
            if obs.cutting_distance_m >= c.minimum_cut_distance_m and obs.payload_volume_m3 - self._entry_payload_m3 >= c.minimum_payload_gain_m3:
                return ExcavatorCycleState.CURL_AND_BREAKOUT, "cut_distance_and_payload_gain_verified"
        elif state is ExcavatorCycleState.CURL_AND_BREAKOUT:
            if at_target and obs.cutting_lip_clearance_m >= c.breakout_lip_clearance_m:
                return ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT, "bucket_curled_and_lip_clear"
        elif state is ExcavatorCycleState.LIFT_TO_TRANSPORT_HEIGHT:
            if at_target and obs.cutting_lip_clearance_m >= c.transport_lip_clearance_m:
                return ExcavatorCycleState.REVERSE_TRAVEL, "safe_transport_clearance_verified"
        elif state is ExcavatorCycleState.REVERSE_TRAVEL:
            if np.linalg.norm(obs.base_pose_xy_yaw[:2] - self._entry_base_xy) >= c.reverse_distance_m:
                return ExcavatorCycleState.ALIGN_DUMP, "measured_reverse_displacement_verified"
        elif state is ExcavatorCycleState.ALIGN_DUMP:
            if at_target and self._position_near(obs.base_pose_xy_yaw, self.dump_pose, c.dump_position_tolerance_m):
                return ExcavatorCycleState.DUMP, "dump_pose_and_swing_verified"
        elif state is ExcavatorCycleState.DUMP:
            released = self._entry_payload_m3 - obs.payload_volume_m3
            if (
                at_target
                and self._entry_payload_m3 > c.empty_payload_tolerance_m3
                and released >= min(c.minimum_dump_release_m3, self._entry_payload_m3)
            ):
                return ExcavatorCycleState.DEPOSITION, "dump_orientation_and_payload_release_verified"
        elif state is ExcavatorCycleState.DEPOSITION:
            # Whole-domain Resting gain is not dump provenance: unrelated
            # Mobile->Resting deposition elsewhere in the terrain can change
            # it.  DUMP entry already requires a measured payload release;
            # deposition completion therefore waits for the released airborne
            # material to clear and for the complete terrain dynamics to meet
            # their physical stop criteria.  MassLedger closes the material
            # balance without inventing a global-Resting proxy for provenance.
            if (
                obs.airborne_volume_m3 <= c.empty_payload_tolerance_m3
                and obs.terrain_settled
            ):
                return ExcavatorCycleState.BUCKET_RECOVERY, "dump_airborne_cleared_and_terrain_settled"
        elif state is ExcavatorCycleState.BUCKET_RECOVERY:
            if at_target:
                return ExcavatorCycleState.RETURN_TRAVEL, "bucket_recovery_verified"
        elif state is ExcavatorCycleState.RETURN_TRAVEL:
            if (
                at_target
                and self._position_near(
                    obs.base_pose_xy_yaw,
                    self.dig_pose,
                    c.ready_position_tolerance_m,
                )
                and obs.terrain_settled
            ):
                return ExcavatorCycleState.READY_NEXT_CYCLE, "return_pose_verified"
        return None, None

    def _enter(self, state: ExcavatorCycleState, obs: ExcavatorCycleObservation) -> None:
        if state is ExcavatorCycleState.DUMP:
            # Preserve the released-payload baseline for audit.  Whole-domain
            # Resting volume is intentionally not used as dump provenance.
            self._dump_entry_payload_m3 = obs.payload_volume_m3
        if (
            state is ExcavatorCycleState.REVERSE_TRAVEL
            and self.config.dynamic_dump_from_reverse_entry
        ):
            travel = self.dump_pose[:2] - self.dig_pose[:2]
            norm = float(np.linalg.norm(travel))
            if norm <= 1.0e-12:
                heading = obs.base_pose_xy_yaw[2]
                travel = -np.asarray([np.cos(heading), np.sin(heading)])
            else:
                travel = travel / norm
            dynamic_dump = np.asarray(
                [
                    obs.base_pose_xy_yaw[0] + travel[0] * self.config.reverse_distance_m,
                    obs.base_pose_xy_yaw[1] + travel[1] * self.config.reverse_distance_m,
                    obs.base_pose_xy_yaw[2],
                ],
                dtype=np.float64,
            )
            dynamic_dump.setflags(write=False)
            self.dump_pose = dynamic_dump
        if (
            state is ExcavatorCycleState.ALIGN_DUMP
            and self.config.dynamic_dump_from_reverse_entry
        ):
            # The formal reverse gate is measured displacement, so the dump
            # base target must be the physical pose at that completion event.
            # Retaining an entry-time prediction causes ALIGN to chase a stale
            # point while the upper body swings.
            measured_dump = np.asarray(obs.base_pose_xy_yaw, dtype=np.float64).copy()
            measured_dump.setflags(write=False)
            self.dump_pose = measured_dump
        self._state = state
        self._state_start_s = obs.timestamp_s
        self._entry_payload_m3 = obs.payload_volume_m3
        self._entry_base_xy = np.array(obs.base_pose_xy_yaw[:2], copy=True)

    def _joint_target_reached(self, obs: ExcavatorCycleObservation) -> bool:
        key = self._TARGET_BY_STATE[self._state]
        target = self.config.phase_targets_rad[key]
        return bool(np.max(np.abs(obs.joint_position_rad - target)) <= self.config.joint_tolerance_rad)

    def _command(self) -> ExcavatorCycleCommand:
        if self._state in {ExcavatorCycleState.IDLE, ExcavatorCycleState.FAILED}:
            target = self.config.phase_targets_rad["initial_pose"]
            return ExcavatorCycleCommand(target, hold_brake=True)
        key = self._TARGET_BY_STATE[self._state]
        left = right = 0.0
        if self._state is ExcavatorCycleState.APPROACH:
            left = right = self.config.approach_track_command
        elif self._state is ExcavatorCycleState.REVERSE_TRAVEL:
            left = right = self.config.reverse_track_command
        elif self._state is ExcavatorCycleState.ALIGN_DUMP:
            left, right = self._navigation_commands(self.dump_pose)
        elif self._state is ExcavatorCycleState.RETURN_TRAVEL:
            left, right = self._navigation_commands(self.dig_pose)
        return ExcavatorCycleCommand(self.config.phase_targets_rad[key], left, right, hold_brake=(left == 0.0 and right == 0.0))

    def _navigation_commands(self, target_pose: np.ndarray) -> tuple[float, float]:
        delta = np.asarray(target_pose[:2] - self._last_base_pose[:2], dtype=np.float64)
        distance = float(np.linalg.norm(delta))
        if distance <= 0.05:
            return 0.0, 0.0
        desired_heading = float(np.arctan2(delta[1], delta[0]))
        heading_error = float(
            np.arctan2(
                np.sin(desired_heading - self._last_base_pose[2]),
                np.cos(desired_heading - self._last_base_pose[2]),
            )
        )
        # A force-driven heavy tracked machine cannot instantaneously follow
        # a point target.  Driving in reverse when the target is outside the
        # forward hemisphere made the former proportional controller chase a
        # moving bearing around a large loop.  Pivot first, then use forward
        # travel only after the chassis is sufficiently aligned.  This is
        # still differential-track actuation; no root pose is written.
        aligned_to_drive = bool(
            abs(heading_error) < self.config.navigation_drive_heading_gate_rad
        )
        drive = 0.0
        if aligned_to_drive:
            drive = float(
                np.clip(
                    self.config.navigation_distance_gain
                    * distance
                    * max(0.0, float(np.cos(heading_error))),
                    0.0,
                    abs(self.config.travel_track_command),
                )
            )
        turn = float(
            np.clip(
                self.config.navigation_heading_gain * np.sin(heading_error),
                -0.45,
                0.45,
            )
        )
        return (
            float(np.clip(drive - turn, -1.0, 1.0)),
            float(np.clip(drive + turn, -1.0, 1.0)),
        )

    def _decision(self, elapsed: float, transitioned: bool, reason: str | None) -> ExcavatorCycleDecision:
        return ExcavatorCycleDecision(
            self._state,
            self._command(),
            transitioned,
            reason,
            self._failure,
            elapsed,
            self._COMPLETION_CONDITION_BY_STATE[self._state],
        )

    @staticmethod
    def _pose(value: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError(f"[390FCycle] {name} must be finite shape (3,)")
        result = np.ascontiguousarray(array.copy()); result.setflags(write=False)
        return result

    @staticmethod
    def _position_near(observed: np.ndarray, target: np.ndarray, tolerance: float) -> bool:
        return bool(np.linalg.norm(observed[:2] - target[:2]) <= tolerance)
