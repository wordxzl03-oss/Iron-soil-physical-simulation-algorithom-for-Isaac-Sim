"""Comparable slope-trial metrics for the Phase-D 0/10/20-degree test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SlopeTrialMetrics:
    """Aggregate metrics from one constant-throttle slope-driving trial."""

    slope_deg: float
    throttle: float
    mean_speed_m_s: float
    mean_abs_drive_torque_nm: float
    mean_abs_slip_ratio: float
    positive_drive_energy_j: float
    travel_time_s: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.slope_deg,
                self.throttle,
                self.mean_speed_m_s,
                self.mean_abs_drive_torque_nm,
                self.mean_abs_slip_ratio,
                self.positive_drive_energy_j,
                self.travel_time_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("[SlopeTrialMetrics] all values must be finite")
        if not 0.0 <= self.slope_deg < 90.0:
            raise ValueError("[SlopeTrialMetrics] slope_deg must be in [0,90)")
        if not -1.0 <= self.throttle <= 1.0:
            raise ValueError("[SlopeTrialMetrics] throttle must be in [-1,1]")
        if np.any(values[2:] < 0.0):
            raise ValueError("[SlopeTrialMetrics] aggregate metrics must be non-negative")


def validate_comparable_slope_trials(
    trials: Iterable[SlopeTrialMetrics],
) -> tuple[SlopeTrialMetrics, ...]:
    """Sort trials and require a common throttle with unique slopes."""

    values = tuple(trials)
    if len(values) < 2 or any(not isinstance(item, SlopeTrialMetrics) for item in values):
        raise ValueError("[SlopeComparison] at least two valid trials are required")
    throttle = values[0].throttle
    if any(not np.isclose(item.throttle, throttle, atol=1e-12) for item in values):
        raise ValueError("[SlopeComparison] trials must use the same throttle")
    ordered = tuple(sorted(values, key=lambda item: item.slope_deg))
    slopes = [item.slope_deg for item in ordered]
    if len(slopes) != len(set(slopes)):
        raise ValueError("[SlopeComparison] slope values must be unique")
    return ordered
