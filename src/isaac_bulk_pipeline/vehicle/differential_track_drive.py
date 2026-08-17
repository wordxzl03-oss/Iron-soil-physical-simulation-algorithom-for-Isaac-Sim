"""Reduced-order differential track actuation without root-pose writes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DifferentialTrackDriveConfig:
    maximum_tractive_force_per_track_n: float = 120_000.0
    maximum_braking_force_per_track_n: float = 160_000.0
    command_slew_per_s: float = 1.5
    nominal_track_speed_m_s: float = 1.20
    full_force_speed_error_m_s: float = 1.20
    braking_full_force_speed_m_s: float = 0.50
    lateral_braking_full_force_speed_m_s: float = 0.25
    yaw_braking_full_rate_rad_s: float = 0.30
    braking_position_full_force_error_m: float = 0.20
    braking_yaw_full_force_error_rad: float = 0.20
    forward_axis_lower_body_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    provenance: str = "UNCALIBRATED_PARAMETER_REDUCED_ORDER_TRACK_FORCE_ACTUATOR"

    def __post_init__(self) -> None:
        values = np.asarray([
            self.maximum_tractive_force_per_track_n,
            self.maximum_braking_force_per_track_n,
            self.command_slew_per_s,
            self.nominal_track_speed_m_s,
            self.full_force_speed_error_m_s,
            self.braking_full_force_speed_m_s,
            self.lateral_braking_full_force_speed_m_s,
            self.yaw_braking_full_rate_rad_s,
            self.braking_position_full_force_error_m,
            self.braking_yaw_full_force_error_rad,
        ])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[TrackDrive] limits must be finite/positive")
        if self.provenance != "UNCALIBRATED_PARAMETER_REDUCED_ORDER_TRACK_FORCE_ACTUATOR":
            raise ValueError("[TrackDrive] uncalibrated provenance label is mandatory")
        axis = np.asarray(self.forward_axis_lower_body_local, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1.0e-12:
            raise ValueError("[TrackDrive] forward local axis must be finite/nonzero")
        object.__setattr__(self, "forward_axis_lower_body_local", tuple((axis / np.linalg.norm(axis)).tolist()))


@dataclass(frozen=True)
class DifferentialTrackDriveOutput:
    left_force_world_n: np.ndarray
    right_force_world_n: np.ndarray
    applied_left_command: float
    applied_right_command: float
    left_contact_active: bool
    right_contact_active: bool

    def __post_init__(self) -> None:
        for name in ("left_force_world_n", "right_force_world_n"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"[TrackDrive] {name} must be finite shape (3,)")
            copy = np.ascontiguousarray(value.copy()); copy.setflags(write=False)
            object.__setattr__(self, name, copy)


class DifferentialTrackDriveModel:
    """Map left/right commands to physical forces at the two track bodies.

    This is an actuator model, not a Track--Soil model.  Contact availability
    is supplied by the runtime/terramechanics layer; an airborne track cannot
    generate traction.  Differential forces create yaw through their real
    application points in PhysX.
    """

    def __init__(self, config: DifferentialTrackDriveConfig | None = None) -> None:
        self.config = config or DifferentialTrackDriveConfig()
        self._left = 0.0
        self._right = 0.0

    def reset(self) -> None:
        self._left = 0.0
        self._right = 0.0

    def step(
        self,
        left_command: float,
        right_command: float,
        lower_body_rotation_world: np.ndarray,
        dt_s: float,
        *,
        left_contact_active: bool,
        right_contact_active: bool,
        braking: bool = False,
        lower_body_linear_velocity_world: np.ndarray | None = None,
        lower_body_angular_velocity_world: np.ndarray | None = None,
        brake_position_error_world: np.ndarray | None = None,
        brake_yaw_error_rad: float = 0.0,
        positive_yaw_force_pair_torque_sign: float = -1.0,
    ) -> DifferentialTrackDriveOutput:
        command = np.asarray([left_command, right_command], dtype=np.float64)
        rotation = np.asarray(lower_body_rotation_world, dtype=np.float64)
        dt = float(dt_s)
        if not np.all(np.isfinite(command)) or np.any(np.abs(command) > 1.0):
            raise ValueError("[TrackDrive] commands must be finite in [-1,1]")
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("[TrackDrive] lower-body rotation must be finite (3,3)")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[TrackDrive] dt_s must be finite/positive")
        u, _, vh = np.linalg.svd(rotation)
        rigid_rotation = u @ vh
        forward = rigid_rotation @ np.asarray(self.config.forward_axis_lower_body_local)
        forward /= np.linalg.norm(forward)
        velocity = (
            np.zeros(3, dtype=np.float64)
            if lower_body_linear_velocity_world is None
            else np.asarray(lower_body_linear_velocity_world, dtype=np.float64)
        )
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("[TrackDrive] lower-body velocity must be finite shape (3,)")
        angular_velocity = (
            np.zeros(3, dtype=np.float64)
            if lower_body_angular_velocity_world is None
            else np.asarray(lower_body_angular_velocity_world, dtype=np.float64)
        )
        if angular_velocity.shape != (3,) or not np.all(np.isfinite(angular_velocity)):
            raise ValueError("[TrackDrive] lower-body angular velocity must be finite shape (3,)")
        position_error = (
            np.zeros(3, dtype=np.float64)
            if brake_position_error_world is None
            else np.asarray(brake_position_error_world, dtype=np.float64)
        )
        yaw_error = float(brake_yaw_error_rad)
        yaw_pair_sign = float(positive_yaw_force_pair_torque_sign)
        if position_error.shape != (3,) or not np.all(np.isfinite(position_error)):
            raise ValueError("[TrackDrive] brake position error must be finite shape (3,)")
        if not np.isfinite(yaw_error) or not np.isfinite(yaw_pair_sign) or abs(yaw_pair_sign) < 0.5:
            raise ValueError("[TrackDrive] brake yaw error/pair sign invalid")
        yaw_pair_sign = float(np.sign(yaw_pair_sign))
        maximum_delta = self.config.command_slew_per_s * dt
        self._left += float(np.clip(command[0] - self._left, -maximum_delta, maximum_delta))
        self._right += float(np.clip(command[1] - self._right, -maximum_delta, maximum_delta))
        limit = self.config.maximum_braking_force_per_track_n if braking else self.config.maximum_tractive_force_per_track_n
        brake_fraction = (
            float(
                np.clip(
                    -np.dot(velocity, forward)
                    / self.config.braking_full_force_speed_m_s
                    - np.dot(position_error, forward)
                    / self.config.braking_position_full_force_error_m,
                    -1.0,
                    1.0,
                )
            )
            if braking
            else 0.0
        )
        longitudinal_speed = float(np.dot(velocity, forward))
        if braking:
            left_fraction = right_fraction = brake_fraction
        else:
            # A joystick fraction requests belt speed, not perpetual force.
            # Saturated physical traction closes the measured longitudinal
            # speed error, so a constant command cannot accelerate the 390F
            # without bound on a low-friction normal-support collider.
            left_fraction = float(
                np.clip(
                    (
                        self._left * self.config.nominal_track_speed_m_s
                        - longitudinal_speed
                    )
                    / self.config.full_force_speed_error_m_s,
                    -1.0,
                    1.0,
                )
            )
            right_fraction = float(
                np.clip(
                    (
                        self._right * self.config.nominal_track_speed_m_s
                        - longitudinal_speed
                    )
                    / self.config.full_force_speed_error_m_s,
                    -1.0,
                    1.0,
                )
            )
        left_force = forward * (left_fraction * limit if left_contact_active else 0.0)
        right_force = forward * (right_fraction * limit if right_contact_active else 0.0)
        if braking:
            world_up = np.asarray([0.0, 0.0, 1.0])
            lateral = np.cross(world_up, forward)
            lateral_norm = float(np.linalg.norm(lateral))
            if lateral_norm > 1.0e-12:
                lateral /= lateral_norm
                lateral_fraction = float(
                    np.clip(
                        -np.dot(velocity, lateral)
                        / self.config.lateral_braking_full_force_speed_m_s,
                        -1.0,
                        1.0,
                    )
                    + np.clip(
                        -np.dot(position_error, lateral)
                        / self.config.braking_position_full_force_error_m,
                        -1.0,
                        1.0,
                    )
                )
                lateral_fraction = float(np.clip(lateral_fraction, -1.0, 1.0))
                lateral_force = lateral * lateral_fraction * limit
                if left_contact_active:
                    left_force += lateral_force
                if right_contact_active:
                    right_force += lateral_force
            yaw_fraction = float(
                np.clip(
                    angular_velocity[2]
                    / self.config.yaw_braking_full_rate_rad_s
                    + yaw_error
                    / self.config.braking_yaw_full_force_error_rad,
                    -1.0,
                    1.0,
                )
            )
            # Positive fraction means left +forward/right -forward.  Its yaw
            # moment sign depends on the CAD's actual left/right body layout.
            # Select the fraction sign that opposes the measured positive yaw
            # rate/error, rather than assuming labels imply lever-arm signs.
            yaw_fraction *= -yaw_pair_sign
            if left_contact_active:
                left_force += forward * yaw_fraction * limit
            if right_contact_active:
                right_force -= forward * yaw_fraction * limit
        for force in (left_force, right_force):
            norm = float(np.linalg.norm(force))
            if norm > limit:
                force *= limit / norm
        return DifferentialTrackDriveOutput(
            left_force, right_force, self._left, self._right,
            bool(left_contact_active), bool(right_contact_active),
        )
