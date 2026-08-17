"""Finite, resettable Phase-D telemetry recorder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .energy import MechanicalEnergyAccumulator, MechanicalEnergyReport
from .schema import VehicleTelemetryFrame, clone_frame


@dataclass(frozen=True)
class TelemetrySnapshot:
    """Deep snapshot suitable for episode logging and deterministic tests."""

    frames: tuple[VehicleTelemetryFrame, ...]
    energy: MechanicalEnergyReport
    elapsed_time_s: float
    distance_travelled_m: float


class VehicleTelemetryRecorder:
    """Record synchronized vehicle frames and integrate actuator energy."""

    def __init__(self) -> None:
        self._energy = MechanicalEnergyAccumulator()
        self._frames: list[VehicleTelemetryFrame] = []
        self._distance_travelled_m = 0.0

    def record(self, frame: VehicleTelemetryFrame, *, dt_s: float) -> None:
        if not isinstance(frame, VehicleTelemetryFrame):
            raise TypeError(
                "[VehicleTelemetryRecorder] frame must be VehicleTelemetryFrame"
            )
        if self._frames and frame.pose.timestamp_s <= self._frames[-1].pose.timestamp_s:
            raise ValueError(
                "[VehicleTelemetryRecorder] timestamps must increase strictly"
            )
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                f"[VehicleTelemetryRecorder] dt_s must be finite and > 0; value={dt_s!r}"
            )
        if self._frames:
            delta = (
                frame.pose.position_world_m
                - self._frames[-1].pose.position_world_m
            )
            distance = float(np.linalg.norm(delta))
            if not np.isfinite(distance):
                raise ValueError("[VehicleTelemetryRecorder] non-finite pose distance")
            self._distance_travelled_m += distance
        self._energy.add_step(frame.power_samples(), dt)
        self._frames.append(clone_frame(frame))

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            frames=tuple(clone_frame(frame) for frame in self._frames),
            energy=self._energy.snapshot(),
            elapsed_time_s=self._energy.elapsed_time_s,
            distance_travelled_m=float(self._distance_travelled_m),
        )

    def reset(self) -> None:
        self._frames.clear()
        self._energy.reset()
        self._distance_travelled_m = 0.0

    @property
    def frame_count(self) -> int:
        return len(self._frames)
