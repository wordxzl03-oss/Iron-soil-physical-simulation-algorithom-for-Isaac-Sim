"""Force/power-limited 390F actuation with bounded target-velocity slew."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExcavatorJointActuatorLimit:
    joint_name: str
    effort_limit_nm: float
    velocity_limit_rad_s: float
    acceleration_limit_rad_s2: float
    provenance: str

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.effort_limit_nm,
                self.velocity_limit_rad_s,
                self.acceleration_limit_rad_s2,
            ],
            dtype=np.float64,
        )
        if not self.joint_name or not self.provenance:
            raise ValueError("[390FActuator] joint name/provenance must be non-empty")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[390FActuator] all joint limits must be finite/positive")


@dataclass(frozen=True)
class ExcavatorActuatorConfig:
    joints: tuple[ExcavatorJointActuatorLimit, ...]
    shared_positive_power_limit_w: float
    power_provenance: str

    def __post_init__(self) -> None:
        joints = tuple(self.joints)
        if not joints or len({item.joint_name for item in joints}) != len(joints):
            raise ValueError("[390FActuator] joint limits must be non-empty and unique")
        if not np.isfinite(self.shared_positive_power_limit_w) or self.shared_positive_power_limit_w <= 0.0:
            raise ValueError("[390FActuator] shared power limit must be finite/positive")
        if not self.power_provenance:
            raise ValueError("[390FActuator] power provenance must be non-empty")
        object.__setattr__(self, "joints", joints)

    @classmethod
    def cat_390f_l_mass_configuration(cls) -> "ExcavatorActuatorConfig":
        """Return documented bounds plus explicitly labelled engineering limits.

        Caterpillar publishes swing torque, engine power, pressure/flow,
        cylinder bore/stroke and endpoint digging forces, but not the cylinder
        pin coordinates in this USD.  Boom/stick generalized torque limits are
        therefore conservative pressure-times-area-times-stroke work-per-radian
        bounds, not claimed cylinder moment-arm models.
        """

        pressure_pa = 35_000_000.0
        boom_area_m2 = np.pi * 0.210**2 / 4.0
        stick_area_m2 = np.pi * 0.220**2 / 4.0
        bucket_tip_force_n = 470_400.0
        bucket_tip_radius_m = 2.505
        return cls(
            joints=(
                ExcavatorJointActuatorLimit(
                    "swing_joint",
                    260_000.0,
                    6.2 * 2.0 * np.pi / 60.0,
                    0.35,
                    "MANUFACTURER_SWING_TORQUE_AND_SPEED",
                ),
                ExcavatorJointActuatorLimit(
                    "boom_joint",
                    pressure_pa * 2.0 * boom_area_m2 * 1.967,
                    0.45,
                    0.35,
                    "ENGINEERING_UPPER_BOUND_FROM_MANUFACTURER_PRESSURE_BORE_STROKE_NO_PIN_GEOMETRY",
                ),
                ExcavatorJointActuatorLimit(
                    "stick_joint",
                    pressure_pa * stick_area_m2 * 2.262,
                    0.55,
                    0.45,
                    "ENGINEERING_UPPER_BOUND_FROM_MANUFACTURER_PRESSURE_BORE_STROKE_NO_PIN_GEOMETRY",
                ),
                ExcavatorJointActuatorLimit(
                    "bucket_joint",
                    bucket_tip_force_n * bucket_tip_radius_m,
                    0.70,
                    0.60,
                    "MANUFACTURER_ISO_BUCKET_FORCE_TIMES_PUBLISHED_TIP_RADIUS_BOUND",
                ),
            ),
            shared_positive_power_limit_w=391_000.0,
            power_provenance="MANUFACTURER_ISO_9249_NET_ENGINE_POWER_UPPER_BOUND",
        )


@dataclass(frozen=True)
class ExcavatorActuatorOutput:
    joint_names: tuple[str, ...]
    target_velocity_rad_s: np.ndarray
    effort_command_nm: np.ndarray
    positive_mechanical_power_w: float
    shared_power_scale: float
    effort_saturated: np.ndarray
    velocity_saturated: np.ndarray
    acceleration_saturated: np.ndarray


class ExcavatorActuatorModel:
    """Causal normalized velocity servo intended to drive PhysX by effort.

    Desired joint position is converted to a bounded velocity.  The *target
    velocity* is slew-limited using ``acceleration_limit_rad_s2``; this is not
    a hard bound on measured physical joint acceleration, which still follows
    multibody inertia, gravity, payload and external soil loads in PhysX.  Full
    effort is reached only when velocity error equals the configured speed
    bound, and a shared positive mechanical power cap is applied last.
    """

    def __init__(self, config: ExcavatorActuatorConfig, dof_names: Sequence[str]) -> None:
        self.config = config
        self.dof_names = tuple(str(item) for item in dof_names)
        if len(set(self.dof_names)) != len(self.dof_names):
            raise ValueError("[390FActuator] DOF names must be unique")
        mapping = {item.joint_name: item for item in config.joints}
        missing = [name for name in mapping if name not in self.dof_names]
        if missing:
            raise ValueError(f"[390FActuator] configured joints missing from articulation: {missing}")
        self._limits: Mapping[str, ExcavatorJointActuatorLimit] = mapping
        self._target_velocity = np.zeros(len(self.dof_names), dtype=np.float64)
        self._effort_bias = np.zeros(len(self.dof_names), dtype=np.float64)

    def reset(
        self,
        measured_velocity_rad_s: np.ndarray | None = None,
        measured_hold_effort_nm: np.ndarray | None = None,
    ) -> None:
        self._target_velocity.fill(0.0)
        self._effort_bias.fill(0.0)
        if measured_velocity_rad_s is not None:
            velocity = self._vector(measured_velocity_rad_s, "measured_velocity_rad_s")
            self._target_velocity[:] = velocity
        if measured_hold_effort_nm is not None:
            hold = self._vector(measured_hold_effort_nm, "measured_hold_effort_nm")
            for name, limit in self._limits.items():
                index = self.dof_names.index(name)
                self._effort_bias[index] = float(
                    np.clip(hold[index], -limit.effort_limit_nm, limit.effort_limit_nm)
                )

    def step(
        self,
        desired_position_rad: np.ndarray,
        measured_position_rad: np.ndarray,
        measured_velocity_rad_s: np.ndarray,
        dt_s: float,
    ) -> ExcavatorActuatorOutput:
        desired = self._vector(desired_position_rad, "desired_position_rad")
        position = self._vector(measured_position_rad, "measured_position_rad")
        velocity = self._vector(measured_velocity_rad_s, "measured_velocity_rad_s")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[390FActuator] dt_s must be finite/positive")
        effort = np.zeros(len(self.dof_names), dtype=np.float64)
        effort_saturated = np.zeros(len(self.dof_names), dtype=bool)
        velocity_saturated = np.zeros(len(self.dof_names), dtype=bool)
        acceleration_saturated = np.zeros(len(self.dof_names), dtype=bool)
        for name, limit in self._limits.items():
            index = self.dof_names.index(name)
            raw_velocity = (desired[index] - position[index]) / dt
            bounded_velocity = float(
                np.clip(raw_velocity, -limit.velocity_limit_rad_s, limit.velocity_limit_rad_s)
            )
            # Respect the acceleration-limited stopping distance as the joint
            # approaches its target.  Without this envelope a saturated slew
            # reaches the target at full speed and repeatedly reverses around
            # it, which is especially unsafe for payload release.
            position_error = float(desired[index] - position[index])
            stopping_speed = float(
                np.sqrt(2.0 * limit.acceleration_limit_rad_s2 * abs(position_error))
            )
            bounded_velocity = float(
                np.sign(bounded_velocity)
                * min(abs(bounded_velocity), stopping_speed)
            )
            velocity_saturated[index] = not np.isclose(raw_velocity, bounded_velocity)
            max_delta = limit.acceleration_limit_rad_s2 * dt
            delta = bounded_velocity - self._target_velocity[index]
            bounded_delta = float(np.clip(delta, -max_delta, max_delta))
            acceleration_saturated[index] = not np.isclose(delta, bounded_delta)
            self._target_velocity[index] += bounded_delta
            raw_effort = self._effort_bias[index] + (
                limit.effort_limit_nm
                * (self._target_velocity[index] - velocity[index])
                / limit.velocity_limit_rad_s
            )
            effort[index] = float(
                np.clip(raw_effort, -limit.effort_limit_nm, limit.effort_limit_nm)
            )
            effort_saturated[index] = not np.isclose(raw_effort, effort[index])
            # A static hold feed-forward measured at the initial pose can have
            # the wrong sign after a large linkage motion.  Once measured
            # speed exceeds the configured envelope, safety braking takes
            # precedence over both that bias and the trajectory request.
            if abs(velocity[index]) > limit.velocity_limit_rad_s:
                effort[index] = -float(np.sign(velocity[index])) * limit.effort_limit_nm
                effort_saturated[index] = True
                velocity_saturated[index] = True
        positive_power = float(np.sum(np.maximum(effort * velocity, 0.0)))
        scale = 1.0
        if positive_power > self.config.shared_positive_power_limit_w:
            scale = self.config.shared_positive_power_limit_w / positive_power
            effort *= scale
            positive_power = self.config.shared_positive_power_limit_w
        return ExcavatorActuatorOutput(
            joint_names=self.dof_names,
            target_velocity_rad_s=self._readonly(self._target_velocity),
            effort_command_nm=self._readonly(effort),
            positive_mechanical_power_w=positive_power,
            shared_power_scale=float(scale),
            effort_saturated=self._readonly(effort_saturated),
            velocity_saturated=self._readonly(velocity_saturated),
            acceleration_saturated=self._readonly(acceleration_saturated),
        )

    def _vector(self, value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (len(self.dof_names),) or not np.all(np.isfinite(result)):
            raise ValueError(f"[390FActuator] {name} shape/values invalid")
        return result

    @staticmethod
    def _readonly(value: np.ndarray) -> np.ndarray:
        result = np.ascontiguousarray(np.asarray(value).copy())
        result.setflags(write=False)
        return result
