"""One authoritative bridge across Phases F/G/H/I without Isaac imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..bulk_exchange import DumpAdvanceResult, DumpReleaseResult, DumpTarget, TerrainDumpOperator
from ..bulk_interaction import BulkInteractionResult, BulkMaterialInteractionModel, ToolTerrainIntersectionModel
from ..bulk_state import (
    BucketInternalFillModel,
    BulkStateManager,
    TerrainState,
    TerrainVolumeIntegrator,
)
from ..interaction import SweepResult
from ..soil_force import MobileMomentumBudget, SoilForceModel, SoilForceResult
from ..terrain import TerrainGrid
from ..tools import ToolDescriptor, ToolState
from .state_machine import LoaderOperationState, LoaderOperationStateMachine, OperationDecision, OperationObservation


@dataclass(frozen=True)
class LoadingCycleStepResult:
    decision: OperationDecision
    terrain_state: TerrainState
    interaction: BulkInteractionResult | None
    soil_force: SoilForceResult | None
    dump_release: DumpReleaseResult | None
    dump_advance: DumpAdvanceResult | None
    material_mode: str


class ContinuousLoadingCycleCoordinator:
    """Advance commands and material reservoirs from synchronized observations.

    Isaac owns rigid-body dynamics and supplies ``ToolState``/``SweepResult``.
    This coordinator owns no robot and has no pose-writing method. Its returned
    soil force is applied by ``IsaacSoilForceAdapter`` in the physics loop.
    """

    DIG_STATES = {LoaderOperationState.PENETRATE, LoaderOperationState.FILL}

    def __init__(
        self,
        *,
        operation: LoaderOperationStateMachine,
        state_manager: BulkStateManager,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        descriptor: ToolDescriptor,
        interaction: BulkMaterialInteractionModel | None = None,
        dump: TerrainDumpOperator | None = None,
        soil_force: SoilForceModel | None = None,
        intersection: ToolTerrainIntersectionModel | None = None,
        dump_target: DumpTarget = DumpTarget.TERRAIN,
        resting_relaxation: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None,
    ) -> None:
        self.operation = operation; self.state_manager = state_manager; self.grid = grid
        self.integrator = integrator; self.descriptor = descriptor
        self.interaction_model = interaction or BulkMaterialInteractionModel()
        self.dump_operator = dump or TerrainDumpOperator()
        self.soil_force_model = soil_force or SoilForceModel()
        self.intersection_model = intersection or ToolTerrainIntersectionModel()
        self.dump_target = DumpTarget(dump_target)
        self.resting_relaxation = resting_relaxation
        self._action_index = state_manager.snapshot().action_index

    def start(self, timestamp_s: float = 0.0) -> None:
        self.operation.start(timestamp_s)

    def step(
        self,
        observation: OperationObservation,
        tool_state: ToolState,
        dt_s: float,
        *,
        sweep: SweepResult | None = None,
        bucket_linear_acceleration_terrain_m_s2: np.ndarray | None = None,
    ) -> LoadingCycleStepResult:
        decision = self.operation.step(observation)
        interaction_result = None; force_result = None; release_result = None
        advance_result = None; mode = "settling"
        if decision.state in self.DIG_STATES:
            if sweep is None:
                raise ValueError("[LoadingCycle] dig states require a continuous sweep")
            before = self.state_manager.snapshot()
            coupled_tool_state = BucketInternalFillModel.apply_secondary_separation(
                tool_state, before.payload, self.descriptor
            )
            candidate = self.intersection_model.compute(
                before.H_resting_m + before.mobile_height_m,
                sweep,
                coupled_tool_state,
                self.grid,
                self.integrator,
            )
            interaction_result = self.interaction_model.advance(
                self.state_manager,
                candidate,
                coupled_tool_state,
                self.descriptor,
                self.grid,
                self.integrator,
                dt_s,
                action_index=self._action_index,
                resting_relaxation=self.resting_relaxation,
            )
            force_result = self.soil_force_model.compute(
                interaction_result.failure_zone,
                candidate,
                before.material,
                self.descriptor,
                coupled_tool_state,
                MobileMomentumBudget.from_mobile_result(
                    interaction_result.mobile_result,
                    interaction_result.activation_tool_impulse_on_mobile_terrain_ns,
                    dt_s,
                ),
            )
            mode = "dig_interaction"
        else:
            if decision.state is LoaderOperationState.DUMP:
                release_result = self.dump_operator.release(
                    self.state_manager,
                    tool_state,
                    self.descriptor,
                    target=self.dump_target,
                    bucket_linear_acceleration_terrain_m_s2=bucket_linear_acceleration_terrain_m_s2,
                )
                mode = "dump_release"
            advance_result = self.dump_operator.advance_airborne(
                self.state_manager,
                self.grid,
                self.integrator,
                dt_s,
                resting_relaxation=self.resting_relaxation,
            )
        return LoadingCycleStepResult(decision, self.state_manager.snapshot(), interaction_result, force_result, release_result, advance_result, mode)

    def reset(self) -> None:
        self.operation.reset(); self.state_manager.reset()
        self._action_index = self.state_manager.snapshot().action_index
