"""Compose robot-independent sweep, excavation, relaxation and state modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ..interaction import (
    ContinuousSweepBuilder,
    ExcavationOperator,
    ExcavationResult,
    SweepResult,
)
from ..solvers import RelaxationResult, TerrainRelaxationSolver
from ..terrain import MassLedger, TerrainGrid, TerrainState, TerrainStateManager
from ..tools import ToolDescriptor, ToolKinematicsAdapter, ToolState
from .action_recorder import ActionRecorder


@dataclass(frozen=True)
class StepResult:
    """Diagnostics for one observed ToolState transition."""

    cached_only: bool
    sweep: SweepResult | None
    excavation: ExcavationResult | None
    relaxation: RelaxationResult | None
    mesh_metrics: Any | None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSummary:
    """Committed action volumes and saved authoritative state."""

    action_index: int
    removed_volume_m3: float
    boundary_outflow_m3: float
    numerical_error_m3: float
    trajectory_pose_count: int
    state: TerrainState
    relaxation: RelaxationResult
    recorder_entry: Mapping[str, Any] | None = None


class SimulationController:
    """Run the complete height-field interaction chain without robot names."""

    def __init__(
        self,
        *,
        grid: TerrainGrid,
        descriptor: ToolDescriptor,
        sweep_builder: ContinuousSweepBuilder,
        excavation_operator: ExcavationOperator,
        solver: TerrainRelaxationSolver,
        state_manager: TerrainStateManager,
        mass_ledger: MassLedger,
        kinematics_adapter: ToolKinematicsAdapter | None = None,
        mesh_adapter: Any | None = None,
        recorder: ActionRecorder | None = None,
    ) -> None:
        if state_manager.grid is not grid:
            raise ValueError(
                "[SimulationController] state_manager must use the configured TerrainGrid"
            )
        self.grid = grid
        self.descriptor = descriptor
        self.sweep_builder = sweep_builder
        self.excavation_operator = excavation_operator
        self.solver = solver
        self.state_manager = state_manager
        self.mass_ledger = mass_ledger
        self.kinematics_adapter = kinematics_adapter
        self.mesh_adapter = mesh_adapter
        self.recorder = recorder
        self.previous_tool_state: ToolState | None = None
        self._action_active = False
        self._trajectory: list[np.ndarray] = []
        self._action_removed_start = 0.0
        self._action_outflow_start = 0.0
        self._step_index = 0
        self._last_relaxation: RelaxationResult | None = None

    def begin_action(self) -> int:
        """Start a new action from the previous action's stable H_current."""

        if self._action_active:
            raise RuntimeError("[SimulationController] an action is already active")
        action_index = self.state_manager.state.action_index
        self.state_manager.begin_action()
        self.previous_tool_state = None
        self._trajectory = []
        self._action_removed_start = self.mass_ledger.removed_volume_m3
        self._action_outflow_start = self.mass_ledger.boundary_outflow_m3
        self._step_index = 0
        self._last_relaxation = None
        self._action_active = True
        return action_index

    def process_tool_state(
        self,
        current: ToolState,
        *,
        cutting_enabled: bool = True,
        update_visualization: bool = True,
    ) -> StepResult:
        """Process one observed pose and explicitly gate excavation by phase."""

        self._require_action("process_tool_state")
        self._trajectory.append(np.array(current.pose_terrain, copy=True))
        if self.previous_tool_state is None:
            self.previous_tool_state = current
            return StepResult(
                cached_only=True,
                sweep=None,
                excavation=None,
                relaxation=None,
                mesh_metrics=None,
                diagnostics={
                    "reason": "first_tool_state_cached",
                    "cutting_enabled": bool(cutting_enabled),
                },
            )

        if not cutting_enabled:
            self.previous_tool_state = current
            self._step_index += 1
            return StepResult(
                cached_only=False,
                sweep=None,
                excavation=None,
                relaxation=None,
                mesh_metrics=None,
                diagnostics={
                    "reason": "cutting_disabled_for_action_phase",
                    "step_index": self._step_index,
                    "cutting_enabled": False,
                },
            )

        sweep = self.sweep_builder.build(
            self.previous_tool_state,
            current,
            self.grid,
            self.descriptor,
        )
        self.previous_tool_state = current
        self._step_index += 1
        if not np.any(sweep.affected_mask):
            return StepResult(
                cached_only=False,
                sweep=sweep,
                excavation=None,
                relaxation=None,
                mesh_metrics=None,
                diagnostics={"reason": "tool_sweep_outside_valid_terrain"},
            )

        excavation = self.excavation_operator.apply(
            self.state_manager.state.H_current,
            sweep,
            self.grid,
        )
        self.state_manager.apply_excavation(excavation)
        self.mass_ledger.record_excavation(excavation, self.grid)
        relaxation: RelaxationResult | None = None
        trigger = self.solver.config.solve_trigger
        should_relax = trigger == "every_step" or (
            trigger == "every_n_steps"
            and self._step_index % self.solver.config.solve_every_n_steps == 0
        )
        mesh_metrics = None
        if should_relax:
            relaxation = self.solver.solve(self.state_manager.state.H_current)
            self._accept_relaxation(relaxation)
            self._last_relaxation = relaxation
            if self.mesh_adapter is not None and update_visualization:
                mesh_metrics = self.mesh_adapter.update(
                    self.state_manager.state.H_current,
                    affected_bbox=None,
                )
        elif (
            self.mesh_adapter is not None
            and update_visualization
            and excavation.removed_volume_m3 > 0.0
        ):
            mesh_metrics = self.mesh_adapter.update(
                self.state_manager.state.H_current,
                affected_bbox=excavation.affected_bbox_grid,
            )
        return StepResult(
            cached_only=False,
            sweep=sweep,
            excavation=excavation,
            relaxation=relaxation,
            mesh_metrics=mesh_metrics,
            diagnostics={
                "step_index": self._step_index,
                "solver_trigger": trigger,
                "removed_volume_m3": excavation.removed_volume_m3,
                "affected_cell_count": int(excavation.affected_mask.sum()),
                "cutting_enabled": True,
            },
        )

    def step_from_robot(
        self,
        robot_adapter: Any,
        *,
        cutting_enabled: bool = True,
        update_visualization: bool = True,
    ) -> StepResult:
        """Read a real articulation/link pose and process its standardized state."""

        if self.kinematics_adapter is None:
            raise RuntimeError(
                "[SimulationController] step_from_robot requires ToolKinematicsAdapter"
            )
        joints = robot_adapter.get_joint_state()
        tool_state = self.kinematics_adapter.update(
            robot_adapter.get_tool_link_pose_world(), joints.timestamp
        )
        return self.process_tool_state(
            tool_state,
            cutting_enabled=cutting_enabled,
            update_visualization=update_visualization,
        )

    def end_action(self) -> ActionSummary:
        """Relax, record and commit the action so its stable map feeds the next."""

        self._require_action("end_action")
        action_index = self.state_manager.state.action_index
        if self.state_manager.state.H_stable is None:
            relaxation = self.solver.solve_sequence(
                self.state_manager.state.H_current
            )
            self._accept_relaxation(relaxation)
            self._last_relaxation = relaxation
            if self.mesh_adapter is not None:
                self.mesh_adapter.update(
                    self.state_manager.state.H_current,
                    affected_bbox=None,
                )
        assert self._last_relaxation is not None
        committed = self.state_manager.end_action()
        removed = self.mass_ledger.removed_volume_m3 - self._action_removed_start
        outflow = self.mass_ledger.boundary_outflow_m3 - self._action_outflow_start
        recorder_entry = None
        if self.recorder is not None:
            recorder_entry = self.recorder.record_action(
                action_index=action_index,
                state=committed,
                relaxation=self._last_relaxation,
                tool_trajectory=self._trajectory,
                removed_volume_m3=removed,
                ledger=self.mass_ledger,
                diagnostics={
                    "step_count": self._step_index,
                    "tool_type": self.descriptor.tool_type,
                    "tool_width_m": self.descriptor.nominal_width_m,
                },
            )
        summary = ActionSummary(
            action_index=action_index,
            removed_volume_m3=removed,
            boundary_outflow_m3=outflow,
            numerical_error_m3=self.mass_ledger.numerical_error_m3,
            trajectory_pose_count=len(self._trajectory),
            state=committed,
            relaxation=self._last_relaxation,
            recorder_entry=recorder_entry,
        )
        self.previous_tool_state = None
        self._action_active = False
        self._trajectory = []
        return summary

    def reset(self, robot_adapter: Any | None = None) -> None:
        """Reset terrain, ledger, pose history, solver, mesh and optional robot."""

        self.state_manager.reset()
        self.mass_ledger.reset(self.grid, self.state_manager.state.H_initial)
        self.solver.reset()
        if self.kinematics_adapter is not None:
            self.kinematics_adapter.reset()
        if self.mesh_adapter is not None:
            self.mesh_adapter.reset(self.state_manager.state.H_initial)
        if robot_adapter is not None:
            robot_adapter.reset()
        if self.recorder is not None:
            self.recorder.reset()
        self.previous_tool_state = None
        self._action_active = False
        self._trajectory = []
        self._step_index = 0
        self._last_relaxation = None

    def _accept_relaxation(self, relaxation: RelaxationResult) -> None:
        if not relaxation.converged:
            raise RuntimeError(
                "[SimulationController] refusing to commit a non-converged solver "
                f"result; iterations={relaxation.iteration_count}, "
                f"max_iterations={self.solver.config.max_iterations}, "
                f"boundary={self.solver.config.boundary_condition}, "
                f"legacy_max_slope={relaxation.diagnostics.get('legacy_max_slope')}, "
                f"critical_angle_deg={self.solver.config.critical_angle_deg}, "
                f"tolerance={self.solver.config.tolerance}"
            )
        self.state_manager.apply_relaxation(relaxation)
        self.mass_ledger.record_relaxation(relaxation)

    def _require_action(self, operation: str) -> None:
        if not self._action_active:
            raise RuntimeError(
                f"[SimulationController] {operation} requires begin_action"
            )
