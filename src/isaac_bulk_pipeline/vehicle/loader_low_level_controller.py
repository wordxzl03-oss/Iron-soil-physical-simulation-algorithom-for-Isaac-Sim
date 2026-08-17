"""Isaac-independent sparse target generation for the seven-DOF loader."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from .vehicle_command import VehicleCommand


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"[LoaderControl] {name} must be finite, got {value!r}")
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


@dataclass(frozen=True)
class PositionActuatorLimit:
    lower_rad: float
    upper_rad: float
    velocity_rad_s: float
    effort_nm: float

    def __post_init__(self) -> None:
        for name in ("lower_rad", "upper_rad", "velocity_rad_s", "effort_nm"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.lower_rad >= self.upper_rad:
            raise ValueError("[LoaderControl] actuator lower_rad must be < upper_rad")
        if self.velocity_rad_s <= 0.0 or self.effort_nm <= 0.0:
            raise ValueError("[LoaderControl] velocity and effort limits must be positive")


@dataclass(frozen=True)
class LoaderControlConfig:
    """Finite Phase-B limits, conservative with respect to the current USD."""

    wheel_joint_names: tuple[str, str, str, str] = (
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    )
    # All current wheel joint frames author +Y with no mirror rotation, so a
    # common sign is the audited default.  The tuple remains explicit and
    # configurable for replacement assets with mirrored joint frames.
    wheel_directions: tuple[int, int, int, int] = (1, 1, 1, 1)
    steering_joint_name: str = "articulation_joint"
    lift_joint_name: str = "lift_joint"
    bucket_joint_name: str = "bucket_joint"
    wheel_radius_m: float = 0.78
    wheel_velocity_rad_s: float = 10.0
    wheel_effort_nm: float = 20_000.0
    wheel_power_limit_w: float = 160_000.0
    wheel_power_reference_rad_s: float = 1.0
    wheel_power_min_guard_rad_s: float = 10.0
    wheel_power_discrete_safety_factor: float = 1.15
    steering: PositionActuatorLimit = field(
        default_factory=lambda: PositionActuatorLimit(
            math.radians(-35.0), math.radians(35.0), math.radians(18.0), 1_500_000.0
        )
    )
    lift: PositionActuatorLimit = field(
        default_factory=lambda: PositionActuatorLimit(
            math.radians(-10.0), math.radians(48.0), math.radians(12.0), 2_500_000.0
        )
    )
    bucket: PositionActuatorLimit = field(
        default_factory=lambda: PositionActuatorLimit(
            math.radians(-40.0), math.radians(60.0), math.radians(18.0), 1_500_000.0
        )
    )

    def __post_init__(self) -> None:
        if len(self.wheel_joint_names) != 4 or len(set(self.wheel_joint_names)) != 4:
            raise ValueError("[LoaderControl] exactly four unique wheel joints are required")
        if len(self.wheel_directions) != 4 or any(
            direction not in {-1, 1} for direction in self.wheel_directions
        ):
            raise ValueError("[LoaderControl] wheel_directions must contain four +/-1 values")
        if len(
            set(
                (*self.wheel_joint_names, self.steering_joint_name, self.lift_joint_name, self.bucket_joint_name)
            )
        ) != 7:
            raise ValueError("[LoaderControl] all seven configured DOF names must be unique")
        for name in (
            "wheel_radius_m",
            "wheel_velocity_rad_s",
            "wheel_effort_nm",
            "wheel_power_limit_w",
            "wheel_power_reference_rad_s",
            "wheel_power_min_guard_rad_s",
            "wheel_power_discrete_safety_factor",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"[LoaderControl] {name} must be positive")
            object.__setattr__(self, name, value)
        if self.wheel_power_discrete_safety_factor < 1.0:
            raise ValueError(
                "[LoaderControl] wheel_power_discrete_safety_factor must be >= 1"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LoaderControlConfig":
        names = values.get("dof_names", {})
        limits = values.get("limits", {})

        def position_limit(name: str, default: PositionActuatorLimit) -> PositionActuatorLimit:
            item = limits.get(name, {})
            return PositionActuatorLimit(
                lower_rad=math.radians(float(item.get("lower_deg", math.degrees(default.lower_rad)))),
                upper_rad=math.radians(float(item.get("upper_deg", math.degrees(default.upper_rad)))),
                velocity_rad_s=math.radians(
                    float(item.get("velocity_deg_s", math.degrees(default.velocity_rad_s)))
                ),
                effort_nm=float(item.get("effort_nm", default.effort_nm)),
            )

        default = cls()
        wheels = tuple(names.get("wheels", default.wheel_joint_names))
        directions = tuple(int(item) for item in values.get("wheel_directions", default.wheel_directions))
        return cls(
            wheel_joint_names=wheels,  # type: ignore[arg-type]
            wheel_directions=directions,  # type: ignore[arg-type]
            steering_joint_name=str(names.get("steering", default.steering_joint_name)),
            lift_joint_name=str(names.get("lift", default.lift_joint_name)),
            bucket_joint_name=str(names.get("bucket_curl", default.bucket_joint_name)),
            wheel_radius_m=float(values.get("wheel_radius_m", default.wheel_radius_m)),
            wheel_velocity_rad_s=float(limits.get("wheel_velocity_rad_s", default.wheel_velocity_rad_s)),
            wheel_effort_nm=float(limits.get("wheel_effort_nm", default.wheel_effort_nm)),
            wheel_power_limit_w=float(limits.get("wheel_power_limit_w", default.wheel_power_limit_w)),
            wheel_power_reference_rad_s=float(
                limits.get("wheel_power_reference_rad_s", default.wheel_power_reference_rad_s)
            ),
            wheel_power_min_guard_rad_s=float(
                limits.get(
                    "wheel_power_min_guard_rad_s",
                    default.wheel_power_min_guard_rad_s,
                )
            ),
            wheel_power_discrete_safety_factor=float(
                limits.get(
                    "wheel_power_discrete_safety_factor",
                    default.wheel_power_discrete_safety_factor,
                )
            ),
            steering=position_limit("steering", default.steering),
            lift=position_limit("lift", default.lift),
            bucket=position_limit("bucket_curl", default.bucket),
        )


@dataclass(frozen=True)
class DOFMapping:
    wheel_indices: tuple[int, int, int, int]
    steering_index: int
    lift_index: int
    bucket_index: int

    @classmethod
    def from_names(cls, dof_names: Sequence[str], config: LoaderControlConfig) -> "DOFMapping":
        names = tuple(str(name) for name in dof_names)
        if len(names) != len(set(names)):
            raise ValueError("[LoaderControl] articulation DOF names must be unique")
        index = {name: position for position, name in enumerate(names)}
        required = (
            *config.wheel_joint_names,
            config.steering_joint_name,
            config.lift_joint_name,
            config.bucket_joint_name,
        )
        missing = [name for name in required if name not in index]
        if missing:
            raise ValueError(f"[LoaderControl] missing expected DOFs: {missing}; got {list(names)}")
        return cls(
            wheel_indices=tuple(index[name] for name in config.wheel_joint_names),  # type: ignore[arg-type]
            steering_index=index[config.steering_joint_name],
            lift_index=index[config.lift_joint_name],
            bucket_index=index[config.bucket_joint_name],
        )


@dataclass(frozen=True)
class SparseJointAction:
    """Backend-neutral analogue of an Isaac ArticulationAction with joint_indices."""

    joint_indices: tuple[int, ...]
    joint_positions: tuple[float, ...] | None = None
    joint_velocities: tuple[float, ...] | None = None
    effort_limits: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        count = len(self.joint_indices)
        if count == 0 or len(set(self.joint_indices)) != count or any(index < 0 for index in self.joint_indices):
            raise ValueError("[SparseJointAction] indices must be non-empty, unique, and non-negative")
        if self.joint_positions is None and self.joint_velocities is None:
            raise ValueError("[SparseJointAction] a position or velocity channel is required")
        for channel_name in ("joint_positions", "joint_velocities", "effort_limits"):
            channel = getattr(self, channel_name)
            if channel is None:
                continue
            if len(channel) != count:
                raise ValueError(f"[SparseJointAction] {channel_name} length must match joint_indices")
            for value in channel:
                if not math.isfinite(float(value)):
                    raise ValueError(f"[SparseJointAction] {channel_name} must be finite")
            if channel_name == "effort_limits" and any(float(value) <= 0.0 for value in channel):
                raise ValueError("[SparseJointAction] effort limits must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint_indices": list(self.joint_indices),
            "joint_positions": None if self.joint_positions is None else list(self.joint_positions),
            "joint_velocities": None if self.joint_velocities is None else list(self.joint_velocities),
            "effort_limits": None if self.effort_limits is None else list(self.effort_limits),
        }


@dataclass(frozen=True)
class LowLevelControlOutput:
    wheel_action: SparseJointAction
    position_action: SparseJointAction
    normalized_command: VehicleCommand
    wheel_power_cap_basis: str
    wheel_power_cap_previous_measured_omega_rad_s: (
        tuple[float, float, float, float] | None
    )
    wheel_power_cap_target_omega_rad_s: tuple[float, float, float, float]
    wheel_power_cap_worst_case_omega_rad_s: tuple[float, float, float, float]
    wheel_power_cap_denominator_rad_s: tuple[float, float, float, float]
    wheel_power_cap_min_guard_rad_s: float
    wheel_power_cap_discrete_safety_factor: float
    wheel_command_envelope_power_bound_w: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        allowed_bases = {
            "MEASURED_OR_TARGET_WORST_CASE",
            "TARGET_JOINT_VELOCITY_FALLBACK",
        }
        if self.wheel_power_cap_basis not in allowed_bases:
            raise ValueError(
                f"[LoaderControl] invalid wheel power-cap basis: {self.wheel_power_cap_basis!r}"
            )
        optional_measured = self.wheel_power_cap_previous_measured_omega_rad_s
        if optional_measured is not None and (
            len(optional_measured) != 4
            or any(not math.isfinite(float(value)) for value in optional_measured)
        ):
            raise ValueError(
                "[LoaderControl] previous measured wheel omega must contain four finite values"
            )
        if (
            self.wheel_power_cap_basis == "MEASURED_OR_TARGET_WORST_CASE"
            and optional_measured is None
        ):
            raise ValueError("[LoaderControl] measured worst-case basis requires measured omega")
        for name, values in (
            ("wheel_power_cap_target_omega_rad_s", self.wheel_power_cap_target_omega_rad_s),
            ("wheel_power_cap_worst_case_omega_rad_s", self.wheel_power_cap_worst_case_omega_rad_s),
            ("wheel_power_cap_denominator_rad_s", self.wheel_power_cap_denominator_rad_s),
            ("wheel_command_envelope_power_bound_w", self.wheel_command_envelope_power_bound_w),
        ):
            if len(values) != 4 or any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"[LoaderControl] {name} must contain four finite values")
        if any(value <= 0.0 for value in self.wheel_power_cap_denominator_rad_s):
            raise ValueError("[LoaderControl] power-cap denominators must be positive")
        safety_factor = _finite(
            "wheel_power_cap_discrete_safety_factor",
            self.wheel_power_cap_discrete_safety_factor,
        )
        min_guard = _finite(
            "wheel_power_cap_min_guard_rad_s",
            self.wheel_power_cap_min_guard_rad_s,
        )
        if min_guard <= 0.0:
            raise ValueError("[LoaderControl] minimum power guard must be positive")
        if any(
            worst_case < min_guard
            for worst_case in self.wheel_power_cap_worst_case_omega_rad_s
        ):
            raise ValueError(
                "[LoaderControl] worst-case omega must include the minimum power guard"
            )
        if safety_factor < 1.0:
            raise ValueError("[LoaderControl] discrete power safety factor must be >= 1")
        if any(
            not math.isclose(
                denominator,
                worst_case * safety_factor,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for denominator, worst_case in zip(
                self.wheel_power_cap_denominator_rad_s,
                self.wheel_power_cap_worst_case_omega_rad_s,
            )
        ):
            raise ValueError(
                "[LoaderControl] protected denominator must equal worst-case omega times safety factor"
            )
        if any(value < 0.0 for value in self.wheel_command_envelope_power_bound_w):
            raise ValueError("[LoaderControl] wheel power-envelope bounds must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "normalized_command": self.normalized_command.as_dict(),
            "wheel_action": self.wheel_action.as_dict(),
            "position_action": self.position_action.as_dict(),
            "wheel_power_cap": {
                "basis": self.wheel_power_cap_basis,
                "previous_measured_omega_rad_s": (
                    None
                    if self.wheel_power_cap_previous_measured_omega_rad_s is None
                    else list(self.wheel_power_cap_previous_measured_omega_rad_s)
                ),
                "target_omega_rad_s": list(self.wheel_power_cap_target_omega_rad_s),
                "worst_case_omega_rad_s": list(
                    self.wheel_power_cap_worst_case_omega_rad_s
                ),
                "denominator_rad_s": list(self.wheel_power_cap_denominator_rad_s),
                "min_guard_rad_s": self.wheel_power_cap_min_guard_rad_s,
                "discrete_safety_factor": self.wheel_power_cap_discrete_safety_factor,
                "command_envelope_power_bound_w": list(
                    self.wheel_command_envelope_power_bound_w
                ),
            },
        }


class LoaderLowLevelController:
    """Map rate-limited normalized commands to bounded sparse DOF targets."""

    def __init__(self, dof_names: Sequence[str], config: LoaderControlConfig | None = None) -> None:
        self.config = config or LoaderControlConfig()
        self.dof_names = tuple(str(name) for name in dof_names)
        self.mapping = DOFMapping.from_names(self.dof_names, self.config)
        self._position_targets: dict[int, float] | None = None

    @staticmethod
    def _position_sequence(values: Sequence[float], expected: int) -> tuple[float, ...]:
        if len(values) != expected:
            raise ValueError(f"[LoaderControl] expected {expected} measured positions, got {len(values)}")
        result = tuple(_finite("measured_joint_position", value) for value in values)
        return result

    @staticmethod
    def _velocity_sequence(values: Sequence[float], expected: int) -> tuple[float, ...]:
        if len(values) != expected:
            raise ValueError(
                f"[LoaderControl] expected {expected} measured velocities, got {len(values)}"
            )
        return tuple(_finite("measured_joint_velocity", value) for value in values)

    def reset(self, measured_joint_positions: Sequence[float] | None = None) -> None:
        measured = (
            (0.0,) * len(self.dof_names)
            if measured_joint_positions is None
            else self._position_sequence(measured_joint_positions, len(self.dof_names))
        )
        limits_and_indices = (
            (self.config.steering, self.mapping.steering_index),
            (self.config.lift, self.mapping.lift_index),
            (self.config.bucket, self.mapping.bucket_index),
        )
        self._position_targets = {
            index: _clamp(measured[index], limit.lower_rad, limit.upper_rad)
            for limit, index in limits_and_indices
        }

    @staticmethod
    def _integrate(
        previous: float,
        command: float,
        dt_s: float,
        limit: PositionActuatorLimit,
    ) -> tuple[float, float]:
        requested = previous + command * limit.velocity_rad_s * dt_s
        target = _clamp(requested, limit.lower_rad, limit.upper_rad)
        target_velocity = (target - previous) / dt_s
        return target, target_velocity

    def step(
        self,
        command: VehicleCommand,
        dt_s: float,
        *,
        measured_joint_positions: Sequence[float] | None = None,
        measured_joint_velocities: Sequence[float] | None = None,
    ) -> LowLevelControlOutput:
        dt_s = _finite("dt_s", dt_s)
        if dt_s <= 0.0:
            raise ValueError("[LoaderControl] dt_s must be positive")
        if self._position_targets is None:
            self.reset(measured_joint_positions)
        assert self._position_targets is not None
        command = command.normalized()

        wheel_speed = 0.0 if command.brake > 0.0 else command.throttle * self.config.wheel_velocity_rad_s
        wheel_velocities = tuple(
            wheel_speed * direction for direction in self.config.wheel_directions
        )
        # A target-speed-only envelope cannot bound power when a wheel exceeds
        # its target downhill or during a transient. Runtime callers therefore
        # provide the previous physics step's measured full-DOF velocity vector.
        # The target-speed path remains an explicit pure-test fallback only.
        # A configured numerical floor also protects a contact/braking impulse
        # when both the previous measurement and current target are near zero.
        if measured_joint_velocities is None:
            power_cap_basis = "TARGET_JOINT_VELOCITY_FALLBACK"
            previous_measured_omegas = None
        else:
            measured_velocities = self._velocity_sequence(
                measured_joint_velocities, len(self.dof_names)
            )
            power_cap_basis = "MEASURED_OR_TARGET_WORST_CASE"
            previous_measured_omegas = tuple(
                measured_velocities[index] for index in self.mapping.wheel_indices
            )
        power_cap_worst_case_omegas = tuple(
            max(
                abs(target_omega),
                (
                    0.0
                    if previous_measured_omegas is None
                    else abs(previous_measured_omegas[index])
                ),
                self.config.wheel_power_reference_rad_s,
                self.config.wheel_power_min_guard_rad_s,
            )
            for index, target_omega in enumerate(wheel_velocities)
        )
        power_cap_denominators = tuple(
            omega * self.config.wheel_power_discrete_safety_factor
            for omega in power_cap_worst_case_omegas
        )
        wheel_effort_limits = tuple(
            min(
                self.config.wheel_effort_nm,
                self.config.wheel_power_limit_w
                / denominator,
            )
            for denominator in power_cap_denominators
        )
        wheel_command_envelope_power_bounds = tuple(
            effort_limit * worst_case_omega
            for effort_limit, worst_case_omega in zip(
                wheel_effort_limits, power_cap_worst_case_omegas
            )
        )
        wheel_action = SparseJointAction(
            joint_indices=self.mapping.wheel_indices,
            joint_positions=None,
            joint_velocities=wheel_velocities,
            effort_limits=wheel_effort_limits,
        )

        specs = (
            (self.mapping.steering_index, command.steering, self.config.steering),
            (self.mapping.lift_index, command.lift, self.config.lift),
            (self.mapping.bucket_index, command.bucket_curl, self.config.bucket),
        )
        position_values: list[float] = []
        velocity_values: list[float] = []
        effort_limits: list[float] = []
        for index, axis, limit in specs:
            target, target_velocity = self._integrate(
                self._position_targets[index], axis, dt_s, limit
            )
            self._position_targets[index] = target
            position_values.append(target)
            velocity_values.append(target_velocity)
            effort_limits.append(limit.effort_nm)
        position_action = SparseJointAction(
            joint_indices=(
                self.mapping.steering_index,
                self.mapping.lift_index,
                self.mapping.bucket_index,
            ),
            joint_positions=tuple(position_values),
            joint_velocities=tuple(velocity_values),
            effort_limits=tuple(effort_limits),
        )
        return LowLevelControlOutput(
            wheel_action=wheel_action,
            position_action=position_action,
            normalized_command=command,
            wheel_power_cap_basis=power_cap_basis,
            wheel_power_cap_previous_measured_omega_rad_s=previous_measured_omegas,
            wheel_power_cap_target_omega_rad_s=wheel_velocities,
            wheel_power_cap_worst_case_omega_rad_s=power_cap_worst_case_omegas,
            wheel_power_cap_denominator_rad_s=power_cap_denominators,
            wheel_power_cap_min_guard_rad_s=(
                self.config.wheel_power_min_guard_rad_s
            ),
            wheel_power_cap_discrete_safety_factor=(
                self.config.wheel_power_discrete_safety_factor
            ),
            wheel_command_envelope_power_bound_w=(
                wheel_command_envelope_power_bounds
            ),
        )


def audit_usd_inventory(
    inventory: Mapping[str, Any], config: LoaderControlConfig | None = None
) -> tuple[str, ...]:
    """Return deterministic compatibility errors for a Phase-A inventory JSON."""

    config = config or LoaderControlConfig()
    joints = {item.get("name"): item for item in inventory.get("joints", [])}
    errors: list[str] = []
    expected_efforts = {
        **{name: config.wheel_effort_nm for name in config.wheel_joint_names},
        config.steering_joint_name: config.steering.effort_nm,
        config.lift_joint_name: config.lift.effort_nm,
        config.bucket_joint_name: config.bucket.effort_nm,
    }
    for name, configured_effort in expected_efforts.items():
        joint = joints.get(name)
        if joint is None:
            errors.append(f"missing_joint:{name}")
            continue
        drives = joint.get("drives", [])
        if len(drives) != 1:
            errors.append(f"expected_one_drive:{name}")
            continue
        record = drives[0].get("max_force", {})
        value = record.get("value") if record.get("authored") else None
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            errors.append(f"invalid_usd_max_force:{name}")
        elif configured_effort > float(value):
            errors.append(f"configured_effort_exceeds_usd:{name}")

    limit_specs = {
        config.steering_joint_name: config.steering,
        config.lift_joint_name: config.lift,
        config.bucket_joint_name: config.bucket,
    }
    for name, configured in limit_specs.items():
        joint = joints.get(name)
        if joint is None:
            continue
        lower = joint.get("lower_limit", {}).get("value")
        upper = joint.get("upper_limit", {}).get("value")
        if lower is None or upper is None:
            errors.append(f"missing_usd_position_limits:{name}")
            continue
        if configured.lower_rad < math.radians(float(lower)) - 1e-12:
            errors.append(f"configured_lower_exceeds_usd:{name}")
        if configured.upper_rad > math.radians(float(upper)) + 1e-12:
            errors.append(f"configured_upper_exceeds_usd:{name}")
    return tuple(sorted(errors))
