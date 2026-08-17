"""Stable output schemas; unavailable physics is represented by ``None``."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import numpy as np


def _vec(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite shape (3,)")
    result = np.ascontiguousarray(result.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ReactionWrench:
    force_world: np.ndarray
    torque_world: np.ndarray
    application_point_world: np.ndarray | None = None
    residual_couple_world: np.ndarray | None = None
    status: str = "AVAILABLE_DIRECTLY"

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_world", _vec(self.force_world, "force_world"))
        object.__setattr__(self, "torque_world", _vec(self.torque_world, "torque_world"))
        if self.application_point_world is not None:
            object.__setattr__(self, "application_point_world", _vec(self.application_point_world, "application_point_world"))
        if self.residual_couple_world is not None:
            object.__setattr__(self, "residual_couple_world", _vec(self.residual_couple_world, "residual_couple_world"))


@dataclass(frozen=True)
class SoilStepResult:
    tool_wrench: Mapping[str, ReactionWrench | None]
    track_wrench: Mapping[str, ReactionWrench | None]
    payload_volume_m3: float
    payload_mass_kg: float
    captured_volume_delta_m3: float
    captured_mass_delta_kg: float
    displaced_volume_delta_m3: float | None
    spill_volume_delta_m3: float | None
    deposited_volume_delta_m3: float | None
    mobile_volume_m3: float
    moving_mobile_volume_m3: float | None
    mass_balance_error_m3: float
    terrain_state: str
    diagnostics: Any
    rl_feedback: Any
    production_result: Any = field(repr=False, compare=False)
    track_soil_result: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_wrench", MappingProxyType(dict(self.tool_wrench)))
        object.__setattr__(self, "track_wrench", MappingProxyType(dict(self.track_wrench)))
