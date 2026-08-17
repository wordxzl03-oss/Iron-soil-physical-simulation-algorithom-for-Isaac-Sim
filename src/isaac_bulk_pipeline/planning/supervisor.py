"""Convert each receding attack decision into one continuous operation cycle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..operation import LoaderOperationStateMachine, OperationTargets
from .goals import GoalObservation
from .lattice import ArticulatedState
from .receding import PlanDecision, RecedingLoadingPlanner
from .terrain_view import PlannerTerrainView


@dataclass(frozen=True)
class AutonomousCyclePlan:
    decision: PlanDecision
    operation: LoaderOperationStateMachine | None


class AutonomousLoadingSupervisor:
    def __init__(self, planner: RecedingLoadingPlanner, dump_pose_xy_yaw: np.ndarray, return_pose_xy_yaw: np.ndarray) -> None:
        self.planner = planner
        self.dump_pose = self._pose(dump_pose_xy_yaw)
        self.return_pose = self._pose(return_pose_xy_yaw)

    def plan_next_cycle(self, observation: GoalObservation, terrain_view: PlannerTerrainView, vehicle_state: ArticulatedState) -> AutonomousCyclePlan:
        decision = self.planner.replan(observation, terrain_view, vehicle_state)
        if decision.stop or decision.selected_attack is None:
            return AutonomousCyclePlan(decision, None)
        candidate = decision.selected_attack.candidate
        targets = OperationTargets(candidate.pre_dig_pose_xy_yaw, candidate.attack_pose_xy_yaw, self.dump_pose, self.return_pose)
        return AutonomousCyclePlan(decision, LoaderOperationStateMachine(targets))

    @staticmethod
    def _pose(value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=float)
        if result.shape != (3,) or not np.all(np.isfinite(result)):
            raise ValueError("[Autonomy] target pose must be finite [x,y,yaw]")
        return result.copy()
