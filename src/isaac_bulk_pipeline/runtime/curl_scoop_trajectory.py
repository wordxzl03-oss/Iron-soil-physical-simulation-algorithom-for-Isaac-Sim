"""Continuous production CUT/CURL trajectory for the real 390F articulation.

The generator owns no robot or soil state.  It only supplies C1-continuous
joint targets to the existing force-, speed-, acceleration- and power-limited
production actuator.  Its waypoints are derived from the already audited
penetrate and breakout configurations; no physics or task gate is changed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurlScoopStage:
    name: str
    duration_s: float
    target_rad: np.ndarray


@dataclass(frozen=True)
class CurlScoopSample:
    stage: str
    elapsed_s: float
    stage_elapsed_s: float
    target_rad: np.ndarray
    complete: bool


class RealisticCurlScoopTrajectory:
    """Five-stage 390F backhoe scoop using the audited CAD joint polarity.

    Positive stick motion retracts the lip toward the machine and positive
    bucket motion closes/curls the bucket.  Negative boom motion raises the
    lip.  The coordinated stages therefore overlap stick retraction, positive
    curl and boom lift compensation instead of completing the cut with the
    bucket held open.
    """

    # Limits authored on the real USD revolute joints (degrees).
    LOWER_RAD = np.deg2rad(np.asarray([-np.inf, -25.0, -100.0, -120.0]))
    UPPER_RAD = np.deg2rad(np.asarray([np.inf, 45.0, 45.0, 60.0]))
    # Production actuator velocity limits, in articulation DOF order.
    VELOCITY_LIMIT_RAD_S = np.asarray(
        [6.2 * 2.0 * np.pi / 60.0, 0.45, 0.55, 0.70], dtype=np.float64
    )

    def __init__(
        self,
        penetrate_rad: np.ndarray,
        breakout_rad: np.ndarray,
    ) -> None:
        penetrate = self._vector(penetrate_rad, "penetrate_rad")
        breakout = self._vector(breakout_rad, "breakout_rad")
        self._penetrate = penetrate
        self._breakout = breakout
        self._entry = penetrate.copy()
        self._stages: tuple[CurlScoopStage, ...] = ()
        self.elapsed_s = 0.0
        self.reset(penetrate)

    @property
    def stages(self) -> tuple[CurlScoopStage, ...]:
        return self._stages

    @property
    def duration_s(self) -> float:
        return float(sum(item.duration_s for item in self._stages))

    def reset(self, entry_rad: np.ndarray) -> None:
        entry = self._vector(entry_rad, "entry_rad")
        self._entry = entry.copy()
        # A uses the audited penetrate pose to finish controlled engagement.
        # B--D are non-uniform fractions of the penetrate->breakout motion:
        # curl deliberately leads the closing process while negative boom
        # compensates the downward lip contribution of positive stick/curl.
        delta = self._breakout - self._penetrate
        stage_a = self._penetrate.copy()
        stage_b = self._penetrate + delta * np.asarray([0.0, 2.0 / 15.0, 0.25, 20.0 / 102.0])
        stage_c = self._penetrate + delta * np.asarray([0.0, 8.0 / 15.0, 1.0, 75.0 / 102.0])
        stage_d = self._penetrate + delta * np.asarray([0.0, 12.0 / 15.0, 1.0, 1.0])
        # A small, still in-limit over-retraction at closure preserves material
        # while E withdraws to the established reachable breakout pose.
        stage_d[2] = min(self.UPPER_RAD[2], self._breakout[2] + np.deg2rad(1.0))
        targets = (stage_a, stage_b, stage_c, stage_d, self._breakout.copy())
        definitions = (
            ("A_CONTROLLED_ENGAGEMENT", 0.75),
            ("B_INITIAL_CUT", 1.25),
            ("C_COORDINATED_SCOOP", 3.25),
            ("D_CLOSE_CAPTURE", 1.60),
            ("E_INITIAL_LIFT_WITHDRAWAL", 1.40),
        )
        stages: list[CurlScoopStage] = []
        previous = entry
        for (name, duration), target in zip(definitions, targets):
            self._validate_target(target, name)
            # Cubic smoothstep has peak speed 1.5*delta/duration.  Reject a
            # schedule that asks more than the accepted production envelope.
            peak = 1.5 * np.abs(target - previous) / duration
            if np.any(peak > self.VELOCITY_LIMIT_RAD_S + 1.0e-12):
                raise ValueError(
                    f"[390FCurlScoop] {name} exceeds production velocity limits: "
                    f"peak={peak.tolist()} limit={self.VELOCITY_LIMIT_RAD_S.tolist()}"
                )
            frozen = np.ascontiguousarray(target.copy())
            frozen.setflags(write=False)
            stages.append(CurlScoopStage(name, duration, frozen))
            previous = target
        self._stages = tuple(stages)
        self.elapsed_s = 0.0

    def sample(self, dt_s: float) -> CurlScoopSample:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[390FCurlScoop] dt_s must be finite/positive")
        self.elapsed_s = min(self.elapsed_s + dt, self.duration_s)
        stage_start_s = 0.0
        index = len(self._stages) - 1
        for candidate, stage in enumerate(self._stages):
            if self.elapsed_s <= stage_start_s + stage.duration_s:
                index = candidate
                break
            stage_start_s += stage.duration_s
        stage = self._stages[index]
        local = max(0.0, self.elapsed_s - stage_start_s)
        fraction = min(1.0, local / stage.duration_s)
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        start = self._entry if index == 0 else self._stages[index - 1].target_rad
        target = start + blend * (stage.target_rad - start)
        result = np.ascontiguousarray(target)
        result.setflags(write=False)
        return CurlScoopSample(
            stage=stage.name,
            elapsed_s=float(self.elapsed_s),
            stage_elapsed_s=float(local),
            target_rad=result,
            complete=bool(self.elapsed_s >= self.duration_s),
        )

    @classmethod
    def _validate_target(cls, value: np.ndarray, name: str) -> None:
        if np.any(value < cls.LOWER_RAD) or np.any(value > cls.UPPER_RAD):
            raise ValueError(f"[390FCurlScoop] {name} lies outside real USD joint limits")

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != (4,) or not np.all(np.isfinite(result)):
            raise ValueError(f"[390FCurlScoop] {name} must be finite shape (4,)")
        return np.ascontiguousarray(result.copy())
