"""Forward-compatible left/right track input schemas."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _a(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite shape {shape}")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TrackGeometry:
    length_m: float
    width_m: float
    forward_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.length_m <= 0.0 or self.width_m <= 0.0:
            raise ValueError("track length/width must be positive metres")
        axis = np.asarray(self.forward_axis_local, dtype=np.float64)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-12:
            raise ValueError("forward_axis_local must be a finite nonzero 3-vector")


@dataclass(frozen=True)
class TrackState:
    """Observed track rigid-body state.

    ``belt_speed`` is supported through an integration adapter: it is resolved
    along ``TrackGeometry.forward_axis_local`` and passed to the existing
    TrackSoil effective surface-velocity input. Sprocket speed and explicit
    longitudinal/lateral slip are not separate constitutive inputs.
    """

    pose: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    belt_speed: float | None
    timestamp: float
    sprocket_angular_speed: float | None = None
    longitudinal_slip: float | None = None
    lateral_slip: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", _a(self.pose, (4, 4), "pose"))
        object.__setattr__(self, "linear_velocity", _a(self.linear_velocity, (3,), "linear_velocity"))
        object.__setattr__(self, "angular_velocity", _a(self.angular_velocity, (3,), "angular_velocity"))
        for name in ("timestamp", "belt_speed", "sprocket_angular_speed", "longitudinal_slip", "lateral_slip"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")

