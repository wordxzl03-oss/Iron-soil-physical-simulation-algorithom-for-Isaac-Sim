"""Normalized loader command and deterministic, dt-based slew limiting."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"[VehicleCommand] {name} must be finite, got {value!r}")
    return result


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


@dataclass(frozen=True)
class VehicleCommand:
    """Device-independent loader command.

    Signed channels use ``[-1, 1]`` and brake uses ``[0, 1]``.  Raw finite
    values are accepted so every input adapter can share the same explicit
    normalization step.
    """

    throttle: float = 0.0
    brake: float = 0.0
    steering: float = 0.0
    lift: float = 0.0
    bucket_curl: float = 0.0

    def __post_init__(self) -> None:
        for name in ("throttle", "brake", "steering", "lift", "bucket_curl"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))

    @classmethod
    def zero(cls) -> "VehicleCommand":
        return cls()

    def normalized(self, *, brake_priority: bool = True) -> "VehicleCommand":
        brake = _clamp(self.brake, 0.0, 1.0)
        throttle = _clamp(self.throttle, -1.0, 1.0)
        if brake_priority and brake > 0.0:
            throttle = 0.0
        return VehicleCommand(
            throttle=throttle,
            brake=brake,
            steering=_clamp(self.steering, -1.0, 1.0),
            lift=_clamp(self.lift, -1.0, 1.0),
            bucket_curl=_clamp(self.bucket_curl, -1.0, 1.0),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "throttle": self.throttle,
            "brake": self.brake,
            "steering": self.steering,
            "lift": self.lift,
            "bucket_curl": self.bucket_curl,
        }


@dataclass(frozen=True)
class CommandSlewLimits:
    """Maximum normalized command change per second for the four signed axes."""

    throttle_per_s: float = 1.5
    steering_per_s: float = 1.8
    lift_per_s: float = 1.25
    bucket_curl_per_s: float = 1.5

    def __post_init__(self) -> None:
        for name in (
            "throttle_per_s",
            "steering_per_s",
            "lift_per_s",
            "bucket_curl_per_s",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"[CommandSlewLimits] {name} must be positive")
            object.__setattr__(self, name, value)


class CommandSlewLimiter:
    """Stateful command ramp whose result depends on elapsed time, not frame count."""

    def __init__(self, limits: CommandSlewLimits | None = None) -> None:
        self.limits = limits or CommandSlewLimits()
        self._current = VehicleCommand.zero()

    @property
    def current(self) -> VehicleCommand:
        return self._current

    def reset(self, command: VehicleCommand | None = None) -> VehicleCommand:
        self._current = (command or VehicleCommand.zero()).normalized()
        return self._current

    @staticmethod
    def _approach(current: float, target: float, maximum_delta: float) -> float:
        delta = _clamp(target - current, -maximum_delta, maximum_delta)
        return current + delta

    def step(self, target: VehicleCommand, dt_s: float) -> VehicleCommand:
        dt_s = _finite("dt_s", dt_s)
        if dt_s < 0.0:
            raise ValueError("[CommandSlewLimiter] dt_s must be non-negative")
        target = target.normalized()
        current = self._current
        next_command = VehicleCommand(
            throttle=self._approach(
                current.throttle,
                target.throttle,
                self.limits.throttle_per_s * dt_s,
            ),
            # Braking is a safety channel and intentionally has immediate
            # priority; only the four signed motion channels are ramped.
            brake=target.brake,
            steering=self._approach(
                current.steering,
                target.steering,
                self.limits.steering_per_s * dt_s,
            ),
            lift=self._approach(
                current.lift,
                target.lift,
                self.limits.lift_per_s * dt_s,
            ),
            bucket_curl=self._approach(
                current.bucket_curl,
                target.bucket_curl,
                self.limits.bucket_curl_per_s * dt_s,
            ),
        ).normalized()
        self._current = next_command
        return next_command
