"""Observation-driven continuous loading cycle using VehicleCommand only."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from ..vehicle import VehicleCommand


class LoaderOperationState(str, Enum):
    IDLE = "IDLE"
    DRIVE_TO_PRE_DIG = "DRIVE_TO_PRE_DIG"
    ALIGN = "ALIGN"
    APPROACH = "APPROACH"
    PENETRATE = "PENETRATE"
    FILL = "FILL"
    CURL_LIFT = "CURL_LIFT"
    BREAKOUT = "BREAKOUT"
    REVERSE = "REVERSE"
    TRANSPORT = "TRANSPORT"
    DUMP = "DUMP"
    RETURN = "RETURN"
    SETTLE = "SETTLE"
    DONE = "DONE"


def _pose(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"[Operation] {name} must be finite [x,y,yaw]")
    copy = np.ascontiguousarray(result.copy())
    copy.setflags(write=False)
    return copy


@dataclass(frozen=True)
class OperationTargets:
    pre_dig_pose_xy_yaw: np.ndarray
    dig_pose_xy_yaw: np.ndarray
    dump_pose_xy_yaw: np.ndarray
    return_pose_xy_yaw: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "pre_dig_pose_xy_yaw",
            "dig_pose_xy_yaw",
            "dump_pose_xy_yaw",
            "return_pose_xy_yaw",
        ):
            object.__setattr__(self, name, _pose(getattr(self, name), name))


@dataclass(frozen=True)
class LoaderOperationConfig:
    position_tolerance_m: float = 0.60
    heading_tolerance_rad: float = math.radians(8.0)
    penetration_target_m: float = 0.45
    payload_fill_target: float = 0.80
    reverse_distance_m: float = 3.0
    settle_mobile_volume_m3: float = 0.01
    maximum_state_duration_s: float = 12.0
    drive_throttle: float = 0.45
    approach_throttle: float = 0.25
    penetration_throttle: float = 0.22
    reverse_throttle: float = -0.35
    steering_gain: float = 1.4

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.position_tolerance_m,
                self.heading_tolerance_rad,
                self.penetration_target_m,
                self.reverse_distance_m,
                self.maximum_state_duration_s,
                self.steering_gain,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[Operation] positive configuration invalid")
        if not 0.0 < self.payload_fill_target <= 1.0:
            raise ValueError("[Operation] payload_fill_target must be in (0,1]")
        if self.settle_mobile_volume_m3 < 0.0:
            raise ValueError("[Operation] settle volume must be non-negative")
        for name in (
            "drive_throttle",
            "approach_throttle",
            "penetration_throttle",
            "reverse_throttle",
        ):
            if not -1.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"[Operation] {name} outside [-1,1]")


@dataclass(frozen=True)
class OperationObservation:
    timestamp_s: float
    vehicle_pose_xy_yaw: np.ndarray
    longitudinal_speed_m_s: float
    lift_angle_rad: float
    bucket_angle_rad: float
    penetration_depth_m: float
    payload_fill_ratio: float
    payload_volume_m3: float
    mobile_volume_m3: float
    airborne_parcel_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "vehicle_pose_xy_yaw", _pose(self.vehicle_pose_xy_yaw, "vehicle_pose"))
        scalars = np.asarray(
            [
                self.timestamp_s,
                self.longitudinal_speed_m_s,
                self.lift_angle_rad,
                self.bucket_angle_rad,
                self.penetration_depth_m,
                self.payload_fill_ratio,
                self.payload_volume_m3,
                self.mobile_volume_m3,
            ]
        )
        if not np.all(np.isfinite(scalars)) or self.timestamp_s < 0.0:
            raise ValueError("[Operation] observation contains invalid values")
        if self.penetration_depth_m < 0.0 or not 0.0 <= self.payload_fill_ratio <= 1.0:
            raise ValueError("[Operation] penetration/fill outside valid range")
        if self.payload_volume_m3 < 0.0 or self.mobile_volume_m3 < 0.0:
            raise ValueError("[Operation] reservoir volumes must be non-negative")
        if not isinstance(self.airborne_parcel_count, int) or self.airborne_parcel_count < 0:
            raise ValueError("[Operation] airborne count must be non-negative integer")


@dataclass(frozen=True)
class OperationDecision:
    state: LoaderOperationState
    command: VehicleCommand
    transitioned: bool
    transition_reason: str | None
    state_elapsed_s: float
    target_pose_xy_yaw: np.ndarray | None


class LoaderOperationStateMachine:
    """No robot methods exist here; output is exclusively VehicleCommand."""

    def __init__(
        self,
        targets: OperationTargets,
        config: LoaderOperationConfig | None = None,
    ) -> None:
        self.targets = targets
        self.config = config or LoaderOperationConfig()
        self._state = LoaderOperationState.IDLE
        self._state_start_time_s = 0.0
        self._last_timestamp_s: float | None = None
        self._dig_origin_xy: np.ndarray | None = None

    @property
    def state(self) -> LoaderOperationState:
        return self._state

    def start(self, timestamp_s: float = 0.0) -> None:
        timestamp = float(timestamp_s)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("[Operation] start timestamp invalid")
        if self._state not in {LoaderOperationState.IDLE, LoaderOperationState.DONE}:
            raise RuntimeError("[Operation] start requires IDLE or DONE")
        self._state = LoaderOperationState.DRIVE_TO_PRE_DIG
        self._state_start_time_s = timestamp
        self._last_timestamp_s = timestamp
        self._dig_origin_xy = None

    def reset(self) -> None:
        self._state = LoaderOperationState.IDLE
        self._state_start_time_s = 0.0
        self._last_timestamp_s = None
        self._dig_origin_xy = None

    def step(self, observation: OperationObservation) -> OperationDecision:
        if self._last_timestamp_s is not None and observation.timestamp_s < self._last_timestamp_s:
            raise ValueError("[Operation] observation timestamp moved backwards")
        self._last_timestamp_s = observation.timestamp_s
        if self._state is LoaderOperationState.IDLE:
            return self._decision(observation, VehicleCommand(brake=1.0), None)
        if self._state is LoaderOperationState.DONE:
            return self._decision(observation, VehicleCommand(brake=1.0), None)

        transitioned = False
        reason: str | None = None
        next_state, reason = self._transition(observation)
        if next_state is not self._state:
            self._state = next_state
            self._state_start_time_s = observation.timestamp_s
            transitioned = True
            if next_state is LoaderOperationState.PENETRATE:
                self._dig_origin_xy = np.array(observation.vehicle_pose_xy_yaw[:2], copy=True)
        command, target = self._command(observation)
        return OperationDecision(
            state=self._state,
            command=command.normalized(),
            transitioned=transitioned,
            transition_reason=reason if transitioned else None,
            state_elapsed_s=max(0.0, observation.timestamp_s - self._state_start_time_s),
            target_pose_xy_yaw=None if target is None else np.array(target, copy=True),
        )

    def _transition(self, obs: OperationObservation) -> tuple[LoaderOperationState, str | None]:
        elapsed = obs.timestamp_s - self._state_start_time_s
        timeout = elapsed >= self.config.maximum_state_duration_s
        if self._state is LoaderOperationState.DRIVE_TO_PRE_DIG and self._near(obs, self.targets.pre_dig_pose_xy_yaw):
            return LoaderOperationState.ALIGN, "pre_dig_position_reached"
        if self._state is LoaderOperationState.ALIGN and self._heading_aligned(obs, self.targets.dig_pose_xy_yaw):
            return LoaderOperationState.APPROACH, "dig_heading_aligned"
        if self._state is LoaderOperationState.APPROACH and self._near(obs, self.targets.dig_pose_xy_yaw):
            return LoaderOperationState.PENETRATE, "dig_face_reached"
        if self._state is LoaderOperationState.PENETRATE and (obs.penetration_depth_m >= self.config.penetration_target_m or timeout):
            return LoaderOperationState.FILL, "penetration_target_or_timeout"
        if self._state is LoaderOperationState.FILL and (obs.payload_fill_ratio >= self.config.payload_fill_target or timeout):
            return LoaderOperationState.CURL_LIFT, "payload_target_or_timeout"
        if self._state is LoaderOperationState.CURL_LIFT and (elapsed >= 2.0 or timeout):
            return LoaderOperationState.BREAKOUT, "curl_lift_completed"
        if self._state is LoaderOperationState.BREAKOUT and (elapsed >= 1.0 or timeout):
            return LoaderOperationState.REVERSE, "breakout_completed"
        if self._state is LoaderOperationState.REVERSE:
            origin = self._dig_origin_xy
            if origin is not None and np.linalg.norm(obs.vehicle_pose_xy_yaw[:2] - origin) >= self.config.reverse_distance_m:
                return LoaderOperationState.TRANSPORT, "reverse_clearance_reached"
        if self._state is LoaderOperationState.TRANSPORT and self._near(obs, self.targets.dump_pose_xy_yaw):
            return LoaderOperationState.DUMP, "dump_pose_reached"
        if self._state is LoaderOperationState.DUMP and (obs.payload_volume_m3 <= 1e-6 or timeout):
            return LoaderOperationState.RETURN, "payload_released_or_timeout"
        if self._state is LoaderOperationState.RETURN and self._near(obs, self.targets.return_pose_xy_yaw):
            return LoaderOperationState.SETTLE, "return_pose_reached"
        if self._state is LoaderOperationState.SETTLE and (
            obs.mobile_volume_m3 <= self.config.settle_mobile_volume_m3
            and obs.airborne_parcel_count == 0
            and elapsed >= 0.5
        ):
            return LoaderOperationState.DONE, "material_settled"
        return self._state, None

    def _command(self, obs: OperationObservation) -> tuple[VehicleCommand, np.ndarray | None]:
        state = self._state
        if state is LoaderOperationState.DRIVE_TO_PRE_DIG:
            return self._drive_command(obs, self.targets.pre_dig_pose_xy_yaw, self.config.drive_throttle), self.targets.pre_dig_pose_xy_yaw
        if state is LoaderOperationState.ALIGN:
            return self._drive_command(obs, self.targets.dig_pose_xy_yaw, 0.0), self.targets.dig_pose_xy_yaw
        if state is LoaderOperationState.APPROACH:
            command = self._drive_command(obs, self.targets.dig_pose_xy_yaw, self.config.approach_throttle)
            return VehicleCommand(
                throttle=command.throttle,
                steering=command.steering,
                lift=-0.35,
                bucket_curl=-0.15,
            ), self.targets.dig_pose_xy_yaw
        if state is LoaderOperationState.PENETRATE:
            return VehicleCommand(throttle=self.config.penetration_throttle, lift=-0.2, bucket_curl=-0.1), None
        if state is LoaderOperationState.FILL:
            return VehicleCommand(throttle=0.12, lift=0.18, bucket_curl=0.65), None
        if state in {LoaderOperationState.CURL_LIFT, LoaderOperationState.BREAKOUT}:
            return VehicleCommand(brake=1.0, lift=0.65, bucket_curl=0.85), None
        if state is LoaderOperationState.REVERSE:
            return VehicleCommand(throttle=self.config.reverse_throttle, lift=0.2, bucket_curl=0.3), None
        if state is LoaderOperationState.TRANSPORT:
            command = self._drive_command(obs, self.targets.dump_pose_xy_yaw, self.config.drive_throttle)
            return VehicleCommand(
                throttle=command.throttle,
                steering=command.steering,
                lift=0.15,
                bucket_curl=0.2,
            ), self.targets.dump_pose_xy_yaw
        if state is LoaderOperationState.DUMP:
            return VehicleCommand(brake=1.0, lift=0.15, bucket_curl=-1.0), None
        if state is LoaderOperationState.RETURN:
            return self._drive_command(obs, self.targets.return_pose_xy_yaw, self.config.drive_throttle), self.targets.return_pose_xy_yaw
        return VehicleCommand(brake=1.0), None

    def _drive_command(self, obs: OperationObservation, target: np.ndarray, throttle: float) -> VehicleCommand:
        delta = target[:2] - obs.vehicle_pose_xy_yaw[:2]
        distance = float(np.linalg.norm(delta))
        desired = target[2] if distance < self.config.position_tolerance_m else math.atan2(delta[1], delta[0])
        error = self._wrap(desired - obs.vehicle_pose_xy_yaw[2])
        steering = float(np.clip(self.config.steering_gain * error, -1.0, 1.0))
        if abs(throttle) <= 1e-12:
            # Low bounded creep lets articulated steering physically change yaw;
            # no heading or root pose is ever written.
            throttle_value = 0.08 if abs(error) > self.config.heading_tolerance_rad else 0.0
        else:
            throttle_value = float(np.sign(throttle) * min(abs(throttle), max(0.10, distance)))
        return VehicleCommand(throttle=throttle_value, steering=steering)

    def _near(self, obs: OperationObservation, target: np.ndarray) -> bool:
        return bool(np.linalg.norm(obs.vehicle_pose_xy_yaw[:2] - target[:2]) <= self.config.position_tolerance_m)

    def _heading_aligned(self, obs: OperationObservation, target: np.ndarray) -> bool:
        delta = target[:2] - obs.vehicle_pose_xy_yaw[:2]
        desired = target[2] if np.linalg.norm(delta) < 1e-9 else math.atan2(delta[1], delta[0])
        return abs(self._wrap(desired - obs.vehicle_pose_xy_yaw[2])) <= self.config.heading_tolerance_rad

    def _decision(self, obs: OperationObservation, command: VehicleCommand, target: np.ndarray | None) -> OperationDecision:
        return OperationDecision(
            state=self._state,
            command=command,
            transitioned=False,
            transition_reason=None,
            state_elapsed_s=max(0.0, obs.timestamp_s - self._state_start_time_s),
            target_pose_xy_yaw=target,
        )

    @staticmethod
    def _wrap(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
