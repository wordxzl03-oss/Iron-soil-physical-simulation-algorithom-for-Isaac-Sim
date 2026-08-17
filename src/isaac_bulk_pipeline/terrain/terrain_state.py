"""Authoritative multi-action height-map state transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interaction import ExcavationResult
from ..solvers import RelaxationResult
from .terrain_grid import TerrainGrid


@dataclass
class TerrainState:
    """Explicit copies for the current action and complete episode state."""

    H_initial: np.ndarray
    H_current: np.ndarray
    H_before_action: np.ndarray | None
    H_excavated: np.ndarray | None
    H_stable: np.ndarray | None
    action_index: int
    total_removed_volume_m3: float


class TerrainStateManager:
    """Enforce ``H_current`` continuity across excavation actions."""

    def __init__(self, grid: TerrainGrid, initial_heightmap: np.ndarray) -> None:
        self._grid = grid
        initial = self._copy(initial_heightmap, "initial_heightmap")
        self._state = TerrainState(
            H_initial=np.array(initial, copy=True),
            H_current=np.array(initial, copy=True),
            H_before_action=None,
            H_excavated=None,
            H_stable=None,
            action_index=0,
            total_removed_volume_m3=0.0,
        )
        self._action_active = False

    @property
    def state(self) -> TerrainState:
        return self._state

    @property
    def grid(self) -> TerrainGrid:
        return self._grid

    @property
    def action_active(self) -> bool:
        return self._action_active

    def begin_action(self) -> np.ndarray:
        """Snapshot the current stable state exactly once for a new action."""

        if self._action_active:
            raise RuntimeError(
                "[TerrainStateManager] begin_action called while an action is active; "
                f"action_index={self._state.action_index}"
            )
        self._state.H_before_action = np.array(self._state.H_current, copy=True)
        self._state.H_excavated = None
        self._state.H_stable = None
        self._action_active = True
        return np.array(self._state.H_before_action, copy=True)

    def apply_excavation(self, result: ExcavationResult) -> None:
        """Advance current state to the latest excavation result."""

        self._require_active("apply_excavation")
        height = self._copy(result.heightmap_excavated, "heightmap_excavated")
        self._state.H_excavated = np.array(height, copy=True)
        self._state.H_stable = None
        self._state.H_current = np.array(height, copy=True)
        self._state.total_removed_volume_m3 += float(result.removed_volume_m3)

    def apply_relaxation(self, result: RelaxationResult) -> None:
        """Make the solver's stable output authoritative for subsequent work."""

        self._require_active("apply_relaxation")
        stable = self._copy(result.heightmap_stable, "heightmap_stable")
        self._state.H_stable = np.array(stable, copy=True)
        self._state.H_current = np.array(stable, copy=True)

    def end_action(self) -> TerrainState:
        """Commit one action; the next begin starts from current stable terrain."""

        self._require_active("end_action")
        if self._state.H_excavated is None:
            self._state.H_excavated = np.array(self._state.H_current, copy=True)
        if self._state.H_stable is None:
            self._state.H_stable = np.array(self._state.H_current, copy=True)
        self._state.action_index += 1
        self._action_active = False
        return self.snapshot()

    def reset(self) -> None:
        """Restore the exact initial height map and zero all action state."""

        self._state.H_current = np.array(self._state.H_initial, copy=True)
        self._state.H_before_action = None
        self._state.H_excavated = None
        self._state.H_stable = None
        self._state.action_index = 0
        self._state.total_removed_volume_m3 = 0.0
        self._action_active = False

    def snapshot(self) -> TerrainState:
        """Return independent arrays suitable for recording or tests."""

        return TerrainState(
            H_initial=np.array(self._state.H_initial, copy=True),
            H_current=np.array(self._state.H_current, copy=True),
            H_before_action=None
            if self._state.H_before_action is None
            else np.array(self._state.H_before_action, copy=True),
            H_excavated=None
            if self._state.H_excavated is None
            else np.array(self._state.H_excavated, copy=True),
            H_stable=None
            if self._state.H_stable is None
            else np.array(self._state.H_stable, copy=True),
            action_index=self._state.action_index,
            total_removed_volume_m3=self._state.total_removed_volume_m3,
        )

    def _copy(self, heightmap: np.ndarray, name: str) -> np.ndarray:
        try:
            valid = self._grid.validate_heightmap(heightmap)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"[TerrainStateManager] invalid {name}; shape="
                f"{np.asarray(heightmap).shape}, prim_path={self._grid.terrain_prim_path}: {exc}"
            ) from exc
        return np.array(valid, dtype=np.float64, copy=True, order="C")

    def _require_active(self, operation: str) -> None:
        if not self._action_active:
            raise RuntimeError(
                f"[TerrainStateManager] {operation} requires begin_action; "
                f"action_index={self._state.action_index}"
            )
