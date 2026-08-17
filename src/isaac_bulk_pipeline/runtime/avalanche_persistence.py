"""Evidence-based classification of long-lived Resting/Mobile events.

The thresholds in this module classify diagnostics; they do not alter terrain
physics, stop an avalanche, or release material.  They are deliberately
serialized into every acceptance report so an engineering review can audit
the distinction between advancing material and a re-mobilization loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


PHYSICAL_LONG_RANGE_PROPAGATION = "PHYSICAL_LONG_RANGE_PROPAGATION"
REMOBILIZATION_LIMIT_CYCLE = "REMOBILIZATION_LIMIT_CYCLE"
PHYSICS_NOT_SETTLED = "PHYSICS_NOT_SETTLED"


@dataclass(frozen=True)
class PersistenceDiagnosticThresholds:
    mass_tolerance_m3: float = 1.0e-8
    epsilon_volume_m3: float = 1.0e-12
    # A cycle requires more repeated transfer than first-time transfer and a
    # tail interval in which throughput grows while unique reach is nearly
    # stationary. These are acceptance-only ratios, not constitutive inputs.
    reactivation_ratio_limit: float = 1.0
    circulation_ratio_limit: float = 4.0
    tail_fraction: float = 0.25
    tail_unique_area_growth_fraction_limit: float = 0.02
    tail_reactivated_share_limit: float = 0.50
    # Candidate retirement is a local central-gradient diagnostic only. It is
    # not a physics gate: transported material need not deposit on its source
    # cell, while the accepted MiniSlope criterion is edge-wise. Permanent
    # retirement is therefore assigned to the whole reached event only after
    # Mobile, connected instability and compact residual frontier all settle.
    retired_area_fraction_required: float = 0.95

    def __post_init__(self) -> None:
        values = np.asarray(list(asdict(self).values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[AvalanchePersistence] thresholds must be positive/finite")
        if not 0.0 < self.tail_fraction <= 0.5:
            raise ValueError("[AvalanchePersistence] tail_fraction must be in (0,0.5]")


@dataclass(frozen=True)
class PersistenceSample:
    simulation_time_s: float
    newly_activated_volume_m3: float
    reactivated_volume_m3: float
    cumulative_r2m_m3: float
    cumulative_m2r_m3: float
    net_terrain_volume_change_m3: float
    net_spatial_transfer_m3: float
    current_mobile_volume_m3: float
    maximum_mobile_speed_m_s: float
    newly_activated_area_m2: float
    reactivated_area_m2: float
    retired_candidate_area_m2: float
    active_tile_count: int
    connected_unstable_area_m2: float
    unique_activated_area_m2: float
    unique_cells_ever_activated: int
    cells_activated_more_than_once: int
    mass_error_m3: float
    terrain_settled: bool


@dataclass(frozen=True)
class PersistenceDecision:
    status: str
    classification: str
    reason: str
    metrics: dict[str, float | int | bool]
    thresholds: dict[str, float]


def classify_persistence(
    samples: Iterable[PersistenceSample],
    *,
    thresholds: PersistenceDiagnosticThresholds | None = None,
) -> PersistenceDecision:
    """Classify a completed/diagnostic run without modifying physical state."""

    rows = tuple(samples)
    if not rows:
        raise ValueError("[AvalanchePersistence] at least one sample is required")
    limits = thresholds or PersistenceDiagnosticThresholds()
    final = rows[-1]
    maximum_mobile = max(row.current_mobile_volume_m3 for row in rows)
    maximum_area = max(row.connected_unstable_area_m2 for row in rows)
    maximum_mass_error = max(abs(row.mass_error_m3) for row in rows)
    new_total = sum(row.newly_activated_volume_m3 for row in rows)
    reactivated_total = sum(row.reactivated_volume_m3 for row in rows)
    reactivation_ratio = reactivated_total / max(
        new_total, limits.epsilon_volume_m3
    )
    throughput = final.cumulative_r2m_m3 + final.cumulative_m2r_m3
    circulation_ratio = throughput / max(
        final.net_spatial_transfer_m3, limits.epsilon_volume_m3
    )
    tail_count = max(2, int(np.ceil(len(rows) * limits.tail_fraction)))
    tail = rows[-tail_count:]
    tail_unique_growth = max(
        0.0,
        tail[-1].unique_activated_area_m2 - tail[0].unique_activated_area_m2,
    )
    tail_unique_growth_fraction = tail_unique_growth / max(
        final.unique_activated_area_m2, limits.epsilon_volume_m3
    )
    tail_new = sum(row.newly_activated_volume_m3 for row in tail)
    tail_reactivated = sum(row.reactivated_volume_m3 for row in tail)
    tail_reactivated_share = tail_reactivated / max(
        tail_new + tail_reactivated, limits.epsilon_volume_m3
    )
    repeated_cell_fraction = final.cells_activated_more_than_once / max(
        final.unique_cells_ever_activated, 1
    )
    retired_fraction = final.retired_candidate_area_m2 / max(
        final.unique_activated_area_m2, limits.epsilon_volume_m3
    )
    conservative = maximum_mass_error <= limits.mass_tolerance_m3
    plateau_cycle_evidence = bool(
        reactivation_ratio > limits.reactivation_ratio_limit
        and circulation_ratio > limits.circulation_ratio_limit
        and tail_unique_growth_fraction
        <= limits.tail_unique_area_growth_fraction_limit
        and tail_reactivated_share >= limits.tail_reactivated_share_limit
        and repeated_cell_fraction > 0.0
        and not final.terrain_settled
    )
    physically_retired = bool(
        final.terrain_settled
        and final.active_tile_count == 0
        and final.current_mobile_volume_m3 <= limits.epsilon_volume_m3
        and final.connected_unstable_area_m2 == 0.0
        and final.unique_cells_ever_activated > 0
        and reactivation_ratio <= limits.reactivation_ratio_limit
    )
    metrics: dict[str, float | int | bool] = {
        "simulation_time_s": final.simulation_time_s,
        "max_mobile_m3": maximum_mobile,
        "final_mobile_m3": final.current_mobile_volume_m3,
        "max_connected_unstable_area_m2": maximum_area,
        "unique_activated_area_m2": final.unique_activated_area_m2,
        "reactivated_area_m2": final.reactivated_area_m2,
        "reactivation_ratio": reactivation_ratio,
        "circulation_ratio": circulation_ratio,
        "cumulative_r2m_m3": final.cumulative_r2m_m3,
        "cumulative_m2r_m3": final.cumulative_m2r_m3,
        "net_terrain_volume_change_m3": final.net_terrain_volume_change_m3,
        "net_spatial_transfer_m3": final.net_spatial_transfer_m3,
        "max_mass_error_m3": maximum_mass_error,
        "active_tiles_final": final.active_tile_count,
        "terrain_settled": final.terrain_settled,
        "unique_cells_ever_activated": final.unique_cells_ever_activated,
        "cells_activated_more_than_once": final.cells_activated_more_than_once,
        "repeated_cell_fraction": repeated_cell_fraction,
        "retired_area_fraction": retired_fraction,
        "tail_unique_area_growth_fraction": tail_unique_growth_fraction,
        "tail_reactivated_share": tail_reactivated_share,
        "mass_conservation_pass": conservative,
    }
    if not conservative:
        return PersistenceDecision(
            "FAIL",
            PHYSICS_NOT_SETTLED,
            "MATERIAL_CONSERVATION_EXCEEDED_FORMAL_1E_8_M3_TOLERANCE",
            metrics,
            asdict(limits),
        )
    if plateau_cycle_evidence:
        return PersistenceDecision(
            "FAIL",
            REMOBILIZATION_LIMIT_CYCLE,
            "REACTIVATED_THROUGHPUT_CONTINUED_WHILE_UNIQUE_REACH_PLATEAUED",
            metrics,
            asdict(limits),
        )
    if physically_retired:
        return PersistenceDecision(
            "PASS",
            PHYSICAL_LONG_RANGE_PROPAGATION,
            "FRONTIER_REACHED_NEW_AREA_THEN_MOBILE_CONNECTED_INSTABILITY_AND_RESIDUAL_FRONTIER_SETTLED",
            metrics,
            asdict(limits),
        )
    return PersistenceDecision(
        "FAIL",
        PHYSICS_NOT_SETTLED,
        "FINITE_REPLAY_ENDED_BEFORE_SETTLED_AND_LIMIT_CYCLE_EVIDENCE_WAS_NOT_DECISIVE",
        metrics,
        asdict(limits),
    )
