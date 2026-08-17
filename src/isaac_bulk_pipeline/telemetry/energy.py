"""Mechanical positive-work and signed-energy integration for Phase D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schema import ActuatorCategory, ActuatorPowerSample


@dataclass(frozen=True)
class EnergyTotals:
    """Positive and signed mechanical energy in joules."""

    positive_j: float = 0.0
    signed_j: float = 0.0


@dataclass(frozen=True)
class MechanicalEnergyReport:
    """Energy totals separated into the four required actuator categories."""

    drive: EnergyTotals
    steer: EnergyTotals
    lift: EnergyTotals
    bucket: EnergyTotals
    total: EnergyTotals

    def for_category(self, category: ActuatorCategory | str) -> EnergyTotals:
        selected = ActuatorCategory(category)
        return {
            ActuatorCategory.DRIVE: self.drive,
            ActuatorCategory.STEER: self.steer,
            ActuatorCategory.LIFT: self.lift,
            ActuatorCategory.BUCKET: self.bucket,
        }[selected]


class MechanicalEnergyAccumulator:
    """Integrate per-actuator ``tau*omega`` using piecewise-constant steps.

    Positive work is accumulated per actuator before category summation. This
    avoids incorrectly cancelling simultaneous motoring and back-driven joints.
    Signed energy is also retained; no regenerative efficiency is assumed.
    """

    def __init__(self) -> None:
        self.reset()

    def add_step(
        self,
        samples: Iterable[ActuatorPowerSample],
        dt_s: float,
    ) -> None:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(
                f"[MechanicalEnergyAccumulator] dt_s must be finite and > 0; value={dt_s!r}"
            )
        values = tuple(samples)
        if any(not isinstance(item, ActuatorPowerSample) for item in values):
            raise TypeError(
                "[MechanicalEnergyAccumulator] samples must contain ActuatorPowerSample"
            )
        names = [item.actuator_name for item in values]
        if len(names) != len(set(names)):
            raise ValueError(
                "[MechanicalEnergyAccumulator] actuator names must be unique per step"
            )
        for item in values:
            power = item.power_w
            positive_increment = max(power, 0.0) * dt
            signed_increment = power * dt
            if not np.all(np.isfinite([positive_increment, signed_increment])):
                raise ValueError(
                    "[MechanicalEnergyAccumulator] non-finite energy increment"
                )
            category_values = self._energy[item.category]
            category_values[0] += positive_increment
            category_values[1] += signed_increment
        self._elapsed_time_s += dt
        self._step_count += 1

    @property
    def elapsed_time_s(self) -> float:
        return float(self._elapsed_time_s)

    @property
    def step_count(self) -> int:
        return self._step_count

    def snapshot(self) -> MechanicalEnergyReport:
        def totals(category: ActuatorCategory) -> EnergyTotals:
            values = self._energy[category]
            return EnergyTotals(
                positive_j=float(values[0]),
                signed_j=float(values[1]),
            )

        drive = totals(ActuatorCategory.DRIVE)
        steer = totals(ActuatorCategory.STEER)
        lift = totals(ActuatorCategory.LIFT)
        bucket = totals(ActuatorCategory.BUCKET)
        category_totals = (drive, steer, lift, bucket)
        return MechanicalEnergyReport(
            drive=drive,
            steer=steer,
            lift=lift,
            bucket=bucket,
            total=EnergyTotals(
                positive_j=float(sum(item.positive_j for item in category_totals)),
                signed_j=float(sum(item.signed_j for item in category_totals)),
            ),
        )

    def reset(self) -> None:
        self._energy = {
            category: np.zeros(2, dtype=np.float64)
            for category in ActuatorCategory
        }
        self._elapsed_time_s = 0.0
        self._step_count = 0
