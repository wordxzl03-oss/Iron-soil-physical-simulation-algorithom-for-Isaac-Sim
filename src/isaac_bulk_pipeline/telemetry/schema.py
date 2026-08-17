"""Isaac-independent Phase-D vehicle telemetry schemas.

The classes in this module deliberately contain no ``pxr`` or Isaac imports.
An Isaac adapter may populate these records, while validation, energy accounting,
logging and tests remain available in a normal CPython process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np


TELEMETRY_SCHEMA_VERSION = "phase_d.vehicle_telemetry.v1"


def _finite_scalar(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"[Telemetry] {name} must be finite; value={value!r}")
    return result


def _nonnegative_scalar(value: float, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result < 0.0:
        raise ValueError(f"[Telemetry] {name} must be non-negative; value={result}")
    return result


def _readonly_vector(
    value: Iterable[float] | np.ndarray,
    *,
    length: int,
    name: str,
) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(
            f"[Telemetry] {name} must be a finite vector of shape ({length},); "
            f"shape={result.shape}"
        )
    result.setflags(write=False)
    return result


class ActuatorCategory(str, Enum):
    """Mechanical-energy categories shared by logging and future objectives."""

    DRIVE = "drive"
    STEER = "steer"
    LIFT = "lift"
    BUCKET = "bucket"


class EffortSource(str, Enum):
    """Provenance of the torque used for mechanical-power integration."""

    MEASURED = "measured_effort"
    APPLIED_COMMAND = "applied_effort_command"
    ESTIMATED = "estimated_effort"


@dataclass(frozen=True)
class ActuatorPowerSample:
    """One rotary actuator sample using ``P = tau * omega`` in SI units."""

    actuator_name: str
    category: ActuatorCategory
    torque_nm: float
    angular_velocity_rad_s: float
    effort_source: EffortSource = EffortSource.APPLIED_COMMAND

    def __post_init__(self) -> None:
        if not isinstance(self.actuator_name, str) or not self.actuator_name.strip():
            raise ValueError("[Telemetry] actuator_name must be a non-empty string")
        try:
            category = ActuatorCategory(self.category)
        except ValueError as exc:
            raise ValueError(
                f"[Telemetry] unsupported actuator category={self.category!r}"
            ) from exc
        object.__setattr__(self, "category", category)
        try:
            effort_source = EffortSource(self.effort_source)
        except ValueError as exc:
            raise ValueError(
                f"[Telemetry] unsupported effort_source={self.effort_source!r}"
            ) from exc
        object.__setattr__(self, "effort_source", effort_source)
        object.__setattr__(
            self,
            "torque_nm",
            _finite_scalar(self.torque_nm, name="torque_nm"),
        )
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_scalar(
                self.angular_velocity_rad_s,
                name="angular_velocity_rad_s",
            ),
        )

    @property
    def power_w(self) -> float:
        """Instantaneous signed mechanical power in watts."""

        power = float(self.torque_nm * self.angular_velocity_rad_s)
        if not np.isfinite(power):
            raise ValueError(
                f"[Telemetry] actuator power overflow for {self.actuator_name!r}"
            )
        return power


@dataclass(frozen=True)
class JointTelemetrySample:
    """Raw Isaac joint observation with unambiguous effort provenance.

    Applied command, measured effort and estimated effort remain separate. The
    power sample uses measured effort when available, otherwise the applied
    command, and only then an explicitly labelled estimate.
    """

    joint_name: str
    category: ActuatorCategory
    angular_velocity_rad_s: float
    applied_effort_nm: float | None = None
    measured_effort_nm: float | None = None
    estimated_effort_nm: float | None = None
    target_velocity_rad_s: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_name, str) or not self.joint_name.strip():
            raise ValueError("[Telemetry] joint_name must be a non-empty string")
        try:
            category = ActuatorCategory(self.category)
        except ValueError as exc:
            raise ValueError(
                f"[Telemetry] unsupported joint category={self.category!r}"
            ) from exc
        object.__setattr__(self, "category", category)
        object.__setattr__(
            self,
            "angular_velocity_rad_s",
            _finite_scalar(
                self.angular_velocity_rad_s,
                name="joint.angular_velocity_rad_s",
            ),
        )
        for field_name in (
            "applied_effort_nm",
            "measured_effort_nm",
            "estimated_effort_nm",
            "target_velocity_rad_s",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _finite_scalar(value, name=f"joint.{field_name}"),
                )
        if (
            self.measured_effort_nm is None
            and self.applied_effort_nm is None
            and self.estimated_effort_nm is None
        ):
            raise ValueError(
                "[Telemetry] a joint sample requires measured, applied or "
                "explicitly estimated effort"
            )

    def to_power_sample(self) -> ActuatorPowerSample:
        if self.measured_effort_nm is not None:
            torque = self.measured_effort_nm
            source = EffortSource.MEASURED
        elif self.applied_effort_nm is not None:
            torque = self.applied_effort_nm
            source = EffortSource.APPLIED_COMMAND
        else:
            assert self.estimated_effort_nm is not None
            torque = self.estimated_effort_nm
            source = EffortSource.ESTIMATED
        return ActuatorPowerSample(
            actuator_name=self.joint_name,
            category=self.category,
            torque_nm=torque,
            angular_velocity_rad_s=self.angular_velocity_rad_s,
            effort_source=source,
        )


def compute_longitudinal_slip_ratio(
    *,
    wheel_radius_m: float,
    wheel_angular_velocity_rad_s: float,
    longitudinal_speed_m_s: float,
    low_speed_threshold_m_s: float = 0.10,
) -> float:
    """Compute signed longitudinal slip with a deterministic low-speed guard.

    For non-low-speed motion the formula is ``(r*omega-v) / max(|r*omega|,|v|)``.
    When both circumferential and longitudinal speeds are below the configured
    threshold, zero is returned to avoid a noise-dominated ratio near rest.
    """

    radius = _nonnegative_scalar(wheel_radius_m, name="wheel_radius_m")
    omega = _finite_scalar(
        wheel_angular_velocity_rad_s,
        name="wheel_angular_velocity_rad_s",
    )
    speed = _finite_scalar(longitudinal_speed_m_s, name="longitudinal_speed_m_s")
    threshold = _nonnegative_scalar(
        low_speed_threshold_m_s,
        name="low_speed_threshold_m_s",
    )
    if radius <= 0.0 or threshold <= 0.0:
        raise ValueError("[Telemetry] wheel radius and low-speed threshold must be > 0")
    circumferential_speed = radius * omega
    denominator = max(abs(circumferential_speed), abs(speed))
    if denominator < threshold:
        return 0.0
    return float((circumferential_speed - speed) / denominator)


@dataclass(frozen=True)
class WheelKinematicsSample:
    """Wheel target/actual kinematics and derived longitudinal slip."""

    timestamp_s: float
    wheel_name: str
    wheel_radius_m: float
    angular_velocity_rad_s: float
    target_angular_velocity_rad_s: float
    longitudinal_speed_m_s: float
    slip_ratio: float
    low_speed_threshold_m_s: float = 0.10

    @classmethod
    def from_kinematics(
        cls,
        *,
        timestamp_s: float,
        wheel_name: str,
        wheel_radius_m: float,
        angular_velocity_rad_s: float,
        target_angular_velocity_rad_s: float,
        longitudinal_speed_m_s: float,
        low_speed_threshold_m_s: float = 0.10,
    ) -> "WheelKinematicsSample":
        slip = compute_longitudinal_slip_ratio(
            wheel_radius_m=wheel_radius_m,
            wheel_angular_velocity_rad_s=angular_velocity_rad_s,
            longitudinal_speed_m_s=longitudinal_speed_m_s,
            low_speed_threshold_m_s=low_speed_threshold_m_s,
        )
        return cls(
            timestamp_s=timestamp_s,
            wheel_name=wheel_name,
            wheel_radius_m=wheel_radius_m,
            angular_velocity_rad_s=angular_velocity_rad_s,
            target_angular_velocity_rad_s=target_angular_velocity_rad_s,
            longitudinal_speed_m_s=longitudinal_speed_m_s,
            slip_ratio=slip,
            low_speed_threshold_m_s=low_speed_threshold_m_s,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.wheel_name, str) or not self.wheel_name.strip():
            raise ValueError("[Telemetry] wheel_name must be a non-empty string")
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_scalar(self.timestamp_s, name="timestamp_s"),
        )
        for field_name in (
            "wheel_radius_m",
            "angular_velocity_rad_s",
            "target_angular_velocity_rad_s",
            "longitudinal_speed_m_s",
            "slip_ratio",
            "low_speed_threshold_m_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_scalar(getattr(self, field_name), name=field_name),
            )
        if self.wheel_radius_m <= 0.0 or self.low_speed_threshold_m_s <= 0.0:
            raise ValueError(
                "[Telemetry] wheel radius and low-speed threshold must be > 0"
            )
        expected = compute_longitudinal_slip_ratio(
            wheel_radius_m=self.wheel_radius_m,
            wheel_angular_velocity_rad_s=self.angular_velocity_rad_s,
            longitudinal_speed_m_s=self.longitudinal_speed_m_s,
            low_speed_threshold_m_s=self.low_speed_threshold_m_s,
        )
        if not np.isclose(self.slip_ratio, expected, rtol=0.0, atol=1e-12):
            raise ValueError(
                "[Telemetry] slip_ratio does not match the documented formula; "
                f"expected={expected}, received={self.slip_ratio}"
            )


@dataclass(frozen=True)
class VehiclePoseSample:
    """Vehicle rigid-body pose, motion and local-slope observation in SI units."""

    timestamp_s: float
    position_world_m: np.ndarray
    orientation_xyzw: np.ndarray
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    linear_velocity_world_m_s: np.ndarray
    angular_velocity_world_rad_s: np.ndarray
    articulation_angle_rad: float
    local_terrain_slope_rad: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_scalar(self.timestamp_s, name="timestamp_s"),
        )
        object.__setattr__(
            self,
            "position_world_m",
            _readonly_vector(
                self.position_world_m,
                length=3,
                name="position_world_m",
            ),
        )
        orientation = _readonly_vector(
            self.orientation_xyzw,
            length=4,
            name="orientation_xyzw",
        )
        norm = float(np.linalg.norm(orientation))
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError(
                "[Telemetry] orientation_xyzw must be a unit quaternion; "
                f"norm={norm}"
            )
        object.__setattr__(self, "orientation_xyzw", orientation)
        object.__setattr__(
            self,
            "linear_velocity_world_m_s",
            _readonly_vector(
                self.linear_velocity_world_m_s,
                length=3,
                name="linear_velocity_world_m_s",
            ),
        )
        object.__setattr__(
            self,
            "angular_velocity_world_rad_s",
            _readonly_vector(
                self.angular_velocity_world_rad_s,
                length=3,
                name="angular_velocity_world_rad_s",
            ),
        )
        for field_name in (
            "roll_rad",
            "pitch_rad",
            "yaw_rad",
            "articulation_angle_rad",
            "local_terrain_slope_rad",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_scalar(getattr(self, field_name), name=field_name),
            )
        if not 0.0 <= self.local_terrain_slope_rad <= np.pi / 2.0:
            raise ValueError(
                "[Telemetry] local_terrain_slope_rad must be in [0, pi/2]; "
                f"value={self.local_terrain_slope_rad}"
            )


@dataclass(frozen=True)
class WheelTerrainContactSample:
    """A wheel/support-contact observation; it never modifies terrain state."""

    timestamp_s: float
    wheel_name: str
    position_world_m: np.ndarray
    normal_load_n: float
    tangential_load_n: float
    slip_ratio: float
    wheel_angular_velocity_rad_s: float
    contact_duration_s: float
    in_contact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.wheel_name, str) or not self.wheel_name.strip():
            raise ValueError("[Telemetry] wheel_name must be a non-empty string")
        object.__setattr__(
            self,
            "timestamp_s",
            _nonnegative_scalar(self.timestamp_s, name="timestamp_s"),
        )
        object.__setattr__(
            self,
            "position_world_m",
            _readonly_vector(
                self.position_world_m,
                length=3,
                name="contact.position_world_m",
            ),
        )
        object.__setattr__(
            self,
            "normal_load_n",
            _nonnegative_scalar(self.normal_load_n, name="normal_load_n"),
        )
        object.__setattr__(
            self,
            "tangential_load_n",
            _nonnegative_scalar(
                self.tangential_load_n,
                name="tangential_load_n",
            ),
        )
        object.__setattr__(
            self,
            "slip_ratio",
            _finite_scalar(self.slip_ratio, name="slip_ratio"),
        )
        object.__setattr__(
            self,
            "wheel_angular_velocity_rad_s",
            _finite_scalar(
                self.wheel_angular_velocity_rad_s,
                name="wheel_angular_velocity_rad_s",
            ),
        )
        object.__setattr__(
            self,
            "contact_duration_s",
            _nonnegative_scalar(
                self.contact_duration_s,
                name="contact_duration_s",
            ),
        )
        object.__setattr__(self, "in_contact", bool(self.in_contact))
        if not self.in_contact and (
            self.normal_load_n > 0.0
            or self.tangential_load_n > 0.0
            or self.contact_duration_s > 0.0
        ):
            raise ValueError(
                "[Telemetry] a non-contact sample cannot carry load or duration"
            )


@dataclass(frozen=True)
class VehicleTelemetryFrame:
    """One synchronized vehicle, actuator and wheel-contact observation."""

    pose: VehiclePoseSample
    actuators: tuple[ActuatorPowerSample, ...] = ()
    joints: tuple[JointTelemetrySample, ...] = ()
    wheel_kinematics: tuple[WheelKinematicsSample, ...] = ()
    wheel_contacts: tuple[WheelTerrainContactSample, ...] = ()
    payload_volume_m3: float = 0.0
    estimated_payload_mass_kg: float = 0.0
    schema_version: str = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError(
                "[Telemetry] unsupported schema_version; "
                f"expected={TELEMETRY_SCHEMA_VERSION!r}, "
                f"received={self.schema_version!r}"
            )
        if not isinstance(self.pose, VehiclePoseSample):
            raise TypeError("[Telemetry] pose must be VehiclePoseSample")
        actuators = tuple(self.actuators)
        joints = tuple(self.joints)
        wheel_kinematics = tuple(self.wheel_kinematics)
        contacts = tuple(self.wheel_contacts)
        if any(not isinstance(item, ActuatorPowerSample) for item in actuators):
            raise TypeError("[Telemetry] actuators must contain ActuatorPowerSample")
        if any(
            not isinstance(item, JointTelemetrySample) for item in joints
        ):
            raise TypeError("[Telemetry] joints must contain JointTelemetrySample")
        if any(
            not isinstance(item, WheelKinematicsSample) for item in wheel_kinematics
        ):
            raise TypeError(
                "[Telemetry] wheel_kinematics must contain WheelKinematicsSample"
            )
        if any(
            not isinstance(item, WheelTerrainContactSample) for item in contacts
        ):
            raise TypeError(
                "[Telemetry] wheel_contacts must contain WheelTerrainContactSample"
            )
        names = [item.actuator_name for item in actuators] + [
            item.joint_name for item in joints
        ]
        if len(names) != len(set(names)):
            raise ValueError("[Telemetry] actuator names must be unique within a frame")
        wheel_names = [item.wheel_name for item in contacts]
        if len(wheel_names) != len(set(wheel_names)):
            raise ValueError("[Telemetry] wheel names must be unique within a frame")
        for item in contacts:
            if not np.isclose(
                item.timestamp_s,
                self.pose.timestamp_s,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "[Telemetry] contact and pose timestamps must be synchronized"
                )
        kinematic_wheel_names = [item.wheel_name for item in wheel_kinematics]
        if len(kinematic_wheel_names) != len(set(kinematic_wheel_names)):
            raise ValueError(
                "[Telemetry] kinematic wheel names must be unique within a frame"
            )
        for item in wheel_kinematics:
            if not np.isclose(
                item.timestamp_s,
                self.pose.timestamp_s,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "[Telemetry] wheel kinematic and pose timestamps must be synchronized"
                )
        object.__setattr__(self, "actuators", actuators)
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "wheel_kinematics", wheel_kinematics)
        object.__setattr__(self, "wheel_contacts", contacts)
        object.__setattr__(
            self,
            "payload_volume_m3",
            _nonnegative_scalar(
                self.payload_volume_m3,
                name="payload_volume_m3",
            ),
        )
        object.__setattr__(
            self,
            "estimated_payload_mass_kg",
            _nonnegative_scalar(
                self.estimated_payload_mass_kg,
                name="estimated_payload_mass_kg",
            ),
        )

    def power_samples(self) -> tuple[ActuatorPowerSample, ...]:
        """Return power-ready samples while preserving selected effort source."""

        return self.actuators + tuple(item.to_power_sample() for item in self.joints)


def clone_frame(frame: VehicleTelemetryFrame) -> VehicleTelemetryFrame:
    """Return a deep, independently-owned immutable telemetry frame."""

    pose = frame.pose
    cloned_pose = VehiclePoseSample(
        timestamp_s=pose.timestamp_s,
        position_world_m=np.array(pose.position_world_m, copy=True),
        orientation_xyzw=np.array(pose.orientation_xyzw, copy=True),
        roll_rad=pose.roll_rad,
        pitch_rad=pose.pitch_rad,
        yaw_rad=pose.yaw_rad,
        linear_velocity_world_m_s=np.array(
            pose.linear_velocity_world_m_s,
            copy=True,
        ),
        angular_velocity_world_rad_s=np.array(
            pose.angular_velocity_world_rad_s,
            copy=True,
        ),
        articulation_angle_rad=pose.articulation_angle_rad,
        local_terrain_slope_rad=pose.local_terrain_slope_rad,
    )
    cloned_contacts = tuple(
        WheelTerrainContactSample(
            timestamp_s=item.timestamp_s,
            wheel_name=item.wheel_name,
            position_world_m=np.array(item.position_world_m, copy=True),
            normal_load_n=item.normal_load_n,
            tangential_load_n=item.tangential_load_n,
            slip_ratio=item.slip_ratio,
            wheel_angular_velocity_rad_s=item.wheel_angular_velocity_rad_s,
            contact_duration_s=item.contact_duration_s,
            in_contact=item.in_contact,
        )
        for item in frame.wheel_contacts
    )
    cloned_actuators = tuple(
        ActuatorPowerSample(
            actuator_name=item.actuator_name,
            category=item.category,
            torque_nm=item.torque_nm,
            angular_velocity_rad_s=item.angular_velocity_rad_s,
            effort_source=item.effort_source,
        )
        for item in frame.actuators
    )
    cloned_joints = tuple(
        JointTelemetrySample(
            joint_name=item.joint_name,
            category=item.category,
            angular_velocity_rad_s=item.angular_velocity_rad_s,
            applied_effort_nm=item.applied_effort_nm,
            measured_effort_nm=item.measured_effort_nm,
            estimated_effort_nm=item.estimated_effort_nm,
            target_velocity_rad_s=item.target_velocity_rad_s,
        )
        for item in frame.joints
    )
    cloned_wheel_kinematics = tuple(
        WheelKinematicsSample(
            timestamp_s=item.timestamp_s,
            wheel_name=item.wheel_name,
            wheel_radius_m=item.wheel_radius_m,
            angular_velocity_rad_s=item.angular_velocity_rad_s,
            target_angular_velocity_rad_s=item.target_angular_velocity_rad_s,
            longitudinal_speed_m_s=item.longitudinal_speed_m_s,
            slip_ratio=item.slip_ratio,
            low_speed_threshold_m_s=item.low_speed_threshold_m_s,
        )
        for item in frame.wheel_kinematics
    )
    return VehicleTelemetryFrame(
        pose=cloned_pose,
        actuators=cloned_actuators,
        joints=cloned_joints,
        wheel_kinematics=cloned_wheel_kinematics,
        wheel_contacts=cloned_contacts,
        payload_volume_m3=frame.payload_volume_m3,
        estimated_payload_mass_kg=frame.estimated_payload_mass_kg,
        schema_version=frame.schema_version,
    )
