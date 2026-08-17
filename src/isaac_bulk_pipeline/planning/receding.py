"""Action-level replanning and goal termination."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attack import AttackCandidateGenerator, AttackEvaluation, ClassicalAttackEvaluator
from .goals import GoalObservation, TaskGoal, TaskGoalStatus
from .lattice import ArticulatedState, LatticePath, StateLatticePlanner
from .terrain_view import PlannerTerrainView


@dataclass(frozen=True)
class PlanDecision:
    goal_status: TaskGoalStatus
    selected_attack: AttackEvaluation | None
    candidate_evaluations: tuple[AttackEvaluation, ...]
    path: LatticePath | None
    stop: bool
    replan_index: int


class RecedingLoadingPlanner:
    def __init__(self, goal: TaskGoal, generator: AttackCandidateGenerator, evaluator: ClassicalAttackEvaluator, path_planner: StateLatticePlanner) -> None:
        self.goal = goal; self.generator = generator; self.evaluator = evaluator; self.path_planner = path_planner
        self._replan_index = 0

    def replan(self, observation: GoalObservation, terrain_view: PlannerTerrainView, vehicle_state: ArticulatedState) -> PlanDecision:
        status = self.goal.evaluate(observation)
        index = self._replan_index; self._replan_index += 1
        if status.complete:
            return PlanDecision(status, None, (), None, True, index)
        candidates = self.generator.generate(terrain_view, observation.H_resting_m)
        evaluations = tuple(self.evaluator.evaluate(candidate, np.array([vehicle_state.x_m, vehicle_state.y_m, vehicle_state.yaw_rad])) for candidate in candidates)
        if not evaluations:
            return PlanDecision(status, None, (), None, True, index)
        selected = max(evaluations, key=lambda item: item.score)
        path = self.path_planner.plan(terrain_view, vehicle_state, selected.candidate.pre_dig_pose_xy_yaw)
        return PlanDecision(status, selected, evaluations, path, not path.reached_goal, index)

    def reset(self) -> None:
        self._replan_index = 0
