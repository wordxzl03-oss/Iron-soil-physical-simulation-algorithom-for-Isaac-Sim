"""Loading-task goals independent of an assumed scoop count."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class TaskGoalKind(str, Enum):
    TARGET_MASS = "target_mass"
    TARGET_VOLUME = "target_volume"
    TARGET_REMAINING_RATIO = "target_remaining_ratio"
    TARGET_HEIGHTMAP = "target_heightmap"


@dataclass(frozen=True)
class GoalObservation:
    delivered_volume_m3: float
    assumed_bulk_density_kg_m3: float
    initial_resting_volume_m3: float
    current_resting_volume_m3: float
    H_resting_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.delivered_volume_m3,
                self.assumed_bulk_density_kg_m3,
                self.initial_resting_volume_m3,
                self.current_resting_volume_m3,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("[PlannerGoal] observation values must be finite/non-negative")
        if self.assumed_bulk_density_kg_m3 <= 0.0 or self.initial_resting_volume_m3 <= 0.0:
            raise ValueError("[PlannerGoal] density/initial volume must be positive")
        if self.H_resting_m is not None:
            field = np.asarray(self.H_resting_m, dtype=np.float64)
            if field.ndim != 2 or not np.all(np.isfinite(field)):
                raise ValueError("[PlannerGoal] H_resting_m must be a finite 2-D field")
            copy = np.ascontiguousarray(field.copy())
            copy.setflags(write=False)
            object.__setattr__(self, "H_resting_m", copy)


@dataclass(frozen=True)
class TaskGoalStatus:
    complete: bool
    progress_ratio: float
    achieved_value: float
    target_value: float
    kind: TaskGoalKind


@dataclass(frozen=True)
class TaskGoal:
    kind: TaskGoalKind
    target_value: float | None = None
    target_heightmap_m: np.ndarray | None = None
    excavation_mask: np.ndarray | None = None
    height_tolerance_m: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TaskGoalKind(self.kind))
        if self.kind is TaskGoalKind.TARGET_HEIGHTMAP:
            if self.target_heightmap_m is None:
                raise ValueError("[PlannerGoal] target heightmap is required")
            target = np.asarray(self.target_heightmap_m, dtype=np.float64)
            if target.ndim != 2 or not np.all(np.isfinite(target)) or np.any(target < 0.0):
                raise ValueError("[PlannerGoal] target heightmap must be finite/non-negative")
            mask = np.ones_like(target, dtype=bool) if self.excavation_mask is None else np.asarray(self.excavation_mask, dtype=bool)
            if mask.shape != target.shape or not np.any(mask):
                raise ValueError("[PlannerGoal] excavation mask must match and be non-empty")
            target = np.ascontiguousarray(target.copy()); target.setflags(write=False)
            mask = np.ascontiguousarray(mask.copy()); mask.setflags(write=False)
            object.__setattr__(self, "target_heightmap_m", target)
            object.__setattr__(self, "excavation_mask", mask)
        else:
            value = float(self.target_value) if self.target_value is not None else -1.0
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("[PlannerGoal] scalar target must be positive")
            if self.kind is TaskGoalKind.TARGET_REMAINING_RATIO and value > 1.0:
                raise ValueError("[PlannerGoal] remaining ratio must be in (0,1]")
            object.__setattr__(self, "target_value", value)
        if not np.isfinite(self.height_tolerance_m) or self.height_tolerance_m < 0.0:
            raise ValueError("[PlannerGoal] height tolerance must be non-negative")

    def evaluate(self, observation: GoalObservation) -> TaskGoalStatus:
        if self.kind is TaskGoalKind.TARGET_MASS:
            achieved = observation.delivered_volume_m3 * observation.assumed_bulk_density_kg_m3
            target = float(self.target_value)
            progress = achieved / target
        elif self.kind is TaskGoalKind.TARGET_VOLUME:
            achieved = observation.delivered_volume_m3
            target = float(self.target_value)
            progress = achieved / target
        elif self.kind is TaskGoalKind.TARGET_REMAINING_RATIO:
            current = observation.current_resting_volume_m3 / observation.initial_resting_volume_m3
            target = float(self.target_value)
            achieved = max(0.0, 1.0 - current)
            required = max(1e-12, 1.0 - target)
            progress = achieved / required
            return TaskGoalStatus(current <= target, float(np.clip(progress, 0.0, 1.0)), current, target, self.kind)
        else:
            if observation.H_resting_m is None or observation.H_resting_m.shape != self.target_heightmap_m.shape:
                raise ValueError("[PlannerGoal] current/target heightmap shape mismatch")
            error = np.maximum(observation.H_resting_m - self.target_heightmap_m, 0.0)
            selected = error[self.excavation_mask]
            achieved = float(np.mean(selected <= self.height_tolerance_m))
            target = 1.0
            progress = achieved
        return TaskGoalStatus(achieved >= target, float(np.clip(progress, 0.0, 1.0)), float(achieved), target, self.kind)
