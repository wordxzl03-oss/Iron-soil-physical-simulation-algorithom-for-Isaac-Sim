"""Isaac-independent control contract for the interactive 390F frontend."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping

import numpy as np


class InteractiveRuntimeState(str, Enum):
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    READY_NEXT_CYCLE = "READY_NEXT_CYCLE"
    EMERGENCY_STOPPED = "EMERGENCY_STOPPED"
    FAILED = "FAILED"
    CLOSED = "CLOSED"


class SoilForceMode(str, Enum):
    NO_SOIL_FORCE = "NO_SOIL_FORCE"
    QUASI_STATIC_ONLY = "QUASI_STATIC_ONLY"
    FULL_SOIL_FORCE = "FULL_SOIL_FORCE"


class ResetLevel(str, Enum):
    ROBOT = "ROBOT"
    PAYLOAD = "PAYLOAD"
    TERRAIN = "TERRAIN"
    ALL = "ALL"


@dataclass(frozen=True)
class RuntimeFailure:
    code: str
    message: str
    cycle: int
    phase: str
    timestamp_s: float
    physical_state: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code or not self.phase or not np.isfinite(self.timestamp_s):
            raise ValueError("[InteractiveControl] invalid failure record")


@dataclass(frozen=True)
class InteractiveControlSnapshot:
    state: InteractiveRuntimeState
    soil_force_mode: SoilForceMode
    track_soil_enabled: bool
    debug_overlay_enabled: bool
    kinematic_debug_enabled: bool
    requested_cycle_count: int
    completed_cycle_count: int
    current_cycle: int
    last_reset: ResetLevel | None
    failure: RuntimeFailure | None


class InteractiveControlModel:
    """Strict user-command state machine; it never advances physics itself."""

    def __init__(
        self,
        *,
        soil_force_mode: SoilForceMode = SoilForceMode.FULL_SOIL_FORCE,
        track_soil_enabled: bool = True,
        debug_overlay_enabled: bool = True,
    ) -> None:
        self._state = InteractiveRuntimeState.INITIALIZING
        self._soil_force_mode = SoilForceMode(soil_force_mode)
        self._track_soil_enabled = bool(track_soil_enabled)
        self._debug_overlay_enabled = bool(debug_overlay_enabled)
        self._kinematic_debug_enabled = False
        self._requested_cycles = 0
        self._completed_cycles = 0
        self._current_cycle = 0
        self._last_reset: ResetLevel | None = None
        self._failure: RuntimeFailure | None = None
        self._state_before_pause = InteractiveRuntimeState.READY

    @property
    def state(self) -> InteractiveRuntimeState:
        return self._state

    @property
    def physics_should_step(self) -> bool:
        return self._state is InteractiveRuntimeState.RUNNING

    def initialized(self) -> None:
        self._require_state(InteractiveRuntimeState.INITIALIZING)
        self._state = InteractiveRuntimeState.READY

    def run_cycles(self, count: int) -> None:
        if not isinstance(count, int) or count < 1:
            raise ValueError("[InteractiveControl] cycle count must be >= 1")
        self._require_state(
            InteractiveRuntimeState.READY,
            InteractiveRuntimeState.READY_NEXT_CYCLE,
        )
        self._requested_cycles = count
        self._completed_cycles = 0
        self._current_cycle = 1
        self._failure = None
        self._state = InteractiveRuntimeState.RUNNING

    def pause(self) -> None:
        self._require_state(InteractiveRuntimeState.RUNNING)
        self._state_before_pause = self._state
        self._state = InteractiveRuntimeState.PAUSED

    def resume(self) -> None:
        self._require_state(InteractiveRuntimeState.PAUSED)
        self._state = self._state_before_pause

    def toggle_start_pause(self) -> None:
        if self._state is InteractiveRuntimeState.RUNNING:
            self.pause()
        elif self._state is InteractiveRuntimeState.PAUSED:
            self.resume()
        elif self._state in {
            InteractiveRuntimeState.READY,
            InteractiveRuntimeState.READY_NEXT_CYCLE,
        }:
            self.run_cycles(1)
        else:
            raise RuntimeError(f"[InteractiveControl] Space invalid in {self._state.value}")

    def cycle_completed(self) -> None:
        self._require_state(InteractiveRuntimeState.RUNNING)
        self._completed_cycles += 1
        if self._completed_cycles >= self._requested_cycles:
            self._state = InteractiveRuntimeState.READY_NEXT_CYCLE
            self._current_cycle = self._completed_cycles
        else:
            self._current_cycle = self._completed_cycles + 1

    def emergency_stop(self) -> None:
        if self._state is InteractiveRuntimeState.CLOSED:
            return
        self._requested_cycles = 0
        self._state = InteractiveRuntimeState.EMERGENCY_STOPPED

    def fail(self, failure: RuntimeFailure) -> None:
        if self._state is InteractiveRuntimeState.CLOSED:
            raise RuntimeError("[InteractiveControl] cannot fail a closed runtime")
        self._failure = failure
        self._state = InteractiveRuntimeState.FAILED

    def reset_completed(self, level: ResetLevel) -> None:
        reset = ResetLevel(level)
        self._last_reset = reset
        self._failure = None
        self._requested_cycles = 0
        self._completed_cycles = 0
        self._current_cycle = 0
        self._state = InteractiveRuntimeState.READY

    def set_soil_force_mode(self, mode: SoilForceMode) -> None:
        self._require_configuration_state()
        self._soil_force_mode = SoilForceMode(mode)

    def cycle_soil_force_mode(self) -> SoilForceMode:
        self._require_configuration_state()
        modes = tuple(SoilForceMode)
        self._soil_force_mode = modes[(modes.index(self._soil_force_mode) + 1) % len(modes)]
        return self._soil_force_mode

    def toggle_track_soil(self) -> bool:
        self._require_configuration_state()
        self._track_soil_enabled = not self._track_soil_enabled
        return self._track_soil_enabled

    def toggle_debug_overlay(self) -> bool:
        self._debug_overlay_enabled = not self._debug_overlay_enabled
        return self._debug_overlay_enabled

    def toggle_kinematic_debug(self) -> bool:
        self._require_configuration_state()
        self._kinematic_debug_enabled = not self._kinematic_debug_enabled
        return self._kinematic_debug_enabled

    def close(self) -> None:
        self._state = InteractiveRuntimeState.CLOSED

    def snapshot(self) -> InteractiveControlSnapshot:
        return InteractiveControlSnapshot(
            state=self._state,
            soil_force_mode=self._soil_force_mode,
            track_soil_enabled=self._track_soil_enabled,
            debug_overlay_enabled=self._debug_overlay_enabled,
            kinematic_debug_enabled=self._kinematic_debug_enabled,
            requested_cycle_count=self._requested_cycles,
            completed_cycle_count=self._completed_cycles,
            current_cycle=self._current_cycle,
            last_reset=self._last_reset,
            failure=self._failure,
        )

    def _require_configuration_state(self) -> None:
        self._require_state(
            InteractiveRuntimeState.READY,
            InteractiveRuntimeState.READY_NEXT_CYCLE,
            InteractiveRuntimeState.EMERGENCY_STOPPED,
            InteractiveRuntimeState.FAILED,
        )

    def _require_state(self, *states: InteractiveRuntimeState) -> None:
        if self._state not in states:
            allowed = ", ".join(item.value for item in states)
            raise RuntimeError(
                f"[InteractiveControl] state={self._state.value}; required one of {allowed}"
            )
