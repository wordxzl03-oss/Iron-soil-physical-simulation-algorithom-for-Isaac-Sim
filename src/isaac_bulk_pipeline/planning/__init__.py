"""Deterministic, terrain-state-driven classical autonomy baseline."""

from .attack import (
    AttackCandidate,
    AttackCandidateGenerator,
    AttackEvaluation,
    AttackObjectiveWeights,
    ClassicalAttackEvaluator,
)
from .goals import GoalObservation, TaskGoal, TaskGoalKind, TaskGoalStatus
from .lattice import ArticulatedLatticeConfig, ArticulatedState, LatticePath, StateLatticePlanner
from .receding import PlanDecision, RecedingLoadingPlanner
from .terrain_view import PlannerTerrainView, TerrainCostConfig
from .tracking import NonlinearPathTracker, PathTrackerConfig, smooth_path
from .supervisor import AutonomousCyclePlan, AutonomousLoadingSupervisor
from .safety import (
    SafetyMonitorConfig,
    VehicleSafetyMonitor,
    VehicleSafetyStatus,
    VehicleStabilityObservation,
)

__all__ = [
    "ArticulatedLatticeConfig",
    "ArticulatedState",
    "AttackCandidate",
    "AttackCandidateGenerator",
    "AttackEvaluation",
    "AttackObjectiveWeights",
    "ClassicalAttackEvaluator",
    "GoalObservation",
    "LatticePath",
    "NonlinearPathTracker",
    "PathTrackerConfig",
    "PlanDecision",
    "PlannerTerrainView",
    "RecedingLoadingPlanner",
    "StateLatticePlanner",
    "TaskGoal",
    "TaskGoalKind",
    "TaskGoalStatus",
    "TerrainCostConfig",
    "smooth_path",
    "AutonomousCyclePlan",
    "AutonomousLoadingSupervisor",
    "SafetyMonitorConfig",
    "VehicleSafetyMonitor",
    "VehicleSafetyStatus",
    "VehicleStabilityObservation",
]
