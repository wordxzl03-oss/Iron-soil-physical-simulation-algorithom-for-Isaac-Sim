"""Atomic state/ledger coordinator for Phase E explicit transfers."""

from __future__ import annotations

from collections.abc import Iterable

from .mass_ledger import (
    ConservativeTransfer,
    MassLedger,
    MassLedgerSnapshot,
)
from .models import TerrainState
from .volume_integrator import SurfaceTopology, TerrainVolumeIntegrator


class BulkStateManager:
    """Coordinate immutable TerrainState snapshots with a conservative ledger.

    This class intentionally implements no failure zone, flux, mobile transport,
    intake, deposition or spill logic. A future Phase-F/G algorithm must provide
    both the next spatial state and the explicit transfers that produced it.
    Commit succeeds atomically only when those two representations agree.
    """

    def __init__(
        self,
        initial_state: TerrainState,
        integrator: TerrainVolumeIntegrator,
        *,
        boundary_condition: str = "closed",
        absolute_tolerance_m3: float = 1e-10,
        relative_tolerance: float = 1e-10,
    ) -> None:
        if not isinstance(initial_state, TerrainState):
            raise TypeError("[BulkStateManager] initial_state must be TerrainState")
        if not isinstance(integrator, TerrainVolumeIntegrator):
            raise TypeError(
                "[BulkStateManager] integrator must be TerrainVolumeIntegrator"
            )
        if integrator.topology is not SurfaceTopology.TRIANGLE_A_C:
            raise ValueError(
                "[BulkStateManager] authoritative state requires triangle_a_c "
                "integration matching the visual mesh topology"
            )
        self._integrator = integrator
        self._initial_state = initial_state.snapshot()
        self._state = initial_state.snapshot()
        self._ledger = MassLedger.initialize(
            self._state,
            integrator,
            boundary_condition=boundary_condition,
            absolute_tolerance_m3=absolute_tolerance_m3,
            relative_tolerance=relative_tolerance,
        )
        self._ledger.assert_matches(self._state, integrator)

    def commit_transfers(
        self,
        next_state: TerrainState,
        transfers: Iterable[ConservativeTransfer],
    ) -> None:
        """Atomically accept a caller-computed state and its transfer ledger."""

        if not isinstance(next_state, TerrainState):
            raise TypeError("[BulkStateManager] next_state must be TerrainState")
        if next_state.H_resting_m.shape != self._state.H_resting_m.shape:
            raise ValueError("[BulkStateManager] terrain shape cannot change")
        if next_state.timestamp_s < self._state.timestamp_s:
            raise ValueError("[BulkStateManager] timestamp cannot move backwards")
        if next_state.action_index < self._state.action_index:
            raise ValueError("[BulkStateManager] action_index cannot move backwards")
        if next_state.material != self._initial_state.material:
            raise ValueError("[BulkStateManager] material scenario cannot change mid-run")
        if next_state.payload.capacity_m3 != self._initial_state.payload.capacity_m3:
            raise ValueError("[BulkStateManager] payload capacity cannot change mid-run")

        candidate = self._ledger.clone()
        provided = tuple(transfers)
        if any(not isinstance(item, ConservativeTransfer) for item in provided):
            raise TypeError(
                "[BulkStateManager] transfers must contain ConservativeTransfer"
            )
        for transfer in provided:
            candidate.transfer(transfer)
        candidate.assert_matches(next_state, self._integrator)

        # TerrainState is frozen and owns read-only arrays. Keeping the accepted
        # instance avoids three redundant full-grid copies per physics update;
        # the public snapshot() API still returns deep isolation for callers.
        self._state = next_state
        self._ledger = candidate

    @property
    def current_state(self) -> TerrainState:
        """Internal immutable zero-copy view for high-frequency solver code."""

        return self._state

    def snapshot(self) -> TerrainState:
        return self._state.snapshot()

    def ledger_snapshot(self) -> MassLedgerSnapshot:
        return self._ledger.snapshot()

    def reset(self) -> None:
        self._state = self._initial_state.snapshot()
        self._ledger.reset()
        self._ledger.assert_matches(self._state, self._integrator)
