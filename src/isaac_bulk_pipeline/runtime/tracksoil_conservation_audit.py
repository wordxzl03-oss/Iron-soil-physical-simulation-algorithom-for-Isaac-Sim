"""Acceptance-only per-operator material ledger for the TrackSoil P0 audit.

The recorder consumes scalar DEVICE reductions only.  It never downloads or
mutates a terrain field and it is deliberately absent from the normal runtime
path.  Its purpose is to locate the first operator boundary whose authoritative
reservoir delta is not matched by a declared conservative transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_RESERVOIRS = ("resting", "mobile", "payload", "airborne", "outflow")


def _reservoirs(snapshot: Any) -> dict[str, float]:
    ledger = snapshot.material_ledger
    return {
        "resting": float(ledger.resting_m3),
        "mobile": float(ledger.mobile_m3),
        "payload": float(ledger.payload_m3),
        "airborne": float(ledger.airborne_m3),
        "outflow": float(ledger.outflow_m3),
        "total": float(
            ledger.resting_m3
            + ledger.mobile_m3
            + ledger.payload_m3
            + ledger.airborne_m3
            + ledger.outflow_m3
        ),
        "mass_error_m3": float(ledger.absolute_volume_error_m3),
        "initial_total_m3": float(ledger.initial_total_m3),
    }


def _delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    result = {name: float(after[name] - before[name]) for name in _RESERVOIRS}
    result["total"] = float(sum(result.values()))
    result["mass_error"] = float(
        after["mass_error_m3"] - before["mass_error_m3"]
    )
    return result


@dataclass
class _OpenStep:
    simulation_time_s: float
    phase: str
    phase_time_s: float
    physics_step: int
    before_track: dict[str, float]
    boundaries: list[dict[str, Any]]
    track: dict[str, Any]


class TrackSoilConservationAudit:
    """Record exact DEVICE scalar ledger boundaries without changing physics."""

    schema = "P0-2A_TRACKSOIL_FIRST_BAD_LEDGER/v1"

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self._open: _OpenStep | None = None
        self._phase = ""
        self._phase_start_s = 0.0
        self._previous_avalanche_r2m_m3 = 0.0
        self._previous_avalanche_m2r_m3 = 0.0

    @property
    def active(self) -> bool:
        return self._open is not None

    def begin_step(
        self,
        *,
        simulation_time_s: float,
        phase: str,
        physics_step: int,
        snapshot: Any,
    ) -> None:
        if self._open is not None:
            raise RuntimeError("[TrackSoilAudit] previous step was not completed")
        if phase != self._phase:
            self._phase = str(phase)
            self._phase_start_s = float(simulation_time_s)
        before = _reservoirs(snapshot)
        self._open = _OpenStep(
            simulation_time_s=float(simulation_time_s),
            phase=str(phase),
            phase_time_s=float(simulation_time_s - self._phase_start_s),
            physics_step=int(physics_step),
            before_track=before,
            boundaries=[{"label": "BEFORE_TRACKSOIL", "state": before}],
            track={
                "active": False,
                "affected_cell_count": 0,
                "footprint_area_m2": 0.0,
                "requested_r2m_m3": 0.0,
                "accepted_r2m_m3": 0.0,
                "rejected_r2m_m3": 0.0,
                "resting_volume_removed_m3": 0.0,
                "mobile_volume_added_m3": 0.0,
                "mobile_volume_removed_m3": 0.0,
                "resting_volume_added_m3": 0.0,
                "net_tracksoil_volume_change_m3": 0.0,
                "tracksoil_local_residual_m3": 0.0,
            },
        )

    def after_tracksoil(
        self,
        *,
        snapshot: Any,
        result: Any | None,
        affected_cell_count: int,
        footprint_area_m2: float,
        requested_r2m_m3: float = 0.0,
    ) -> None:
        if self._open is None:
            return
        state = _reservoirs(snapshot)
        change = _delta(self._open.before_track, state)
        accepted = (
            0.0
            if result is None
            else float(result.resting_to_mobile_volume_m3)
        )
        requested = float(requested_r2m_m3)
        self._open.track = {
            "active": bool(result is not None and accepted > 0.0),
            "affected_cell_count": int(affected_cell_count),
            "footprint_area_m2": float(footprint_area_m2),
            "requested_r2m_m3": requested,
            "accepted_r2m_m3": accepted,
            "rejected_r2m_m3": max(requested - accepted, 0.0),
            "resting_volume_removed_m3": float(-change["resting"]),
            "mobile_volume_added_m3": float(change["mobile"]),
            "mobile_volume_removed_m3": 0.0,
            "resting_volume_added_m3": 0.0,
            "net_tracksoil_volume_change_m3": float(
                change["resting"] + change["mobile"]
            ),
            "tracksoil_local_residual_m3": float(
                change["resting"] + change["mobile"]
            ),
            "declared_minus_observed_resting_removal_m3": float(
                accepted + change["resting"]
            ),
            "declared_minus_observed_mobile_addition_m3": float(
                accepted - change["mobile"]
            ),
        }
        self._open.boundaries.append(
            {
                "label": "AFTER_TRACKSOIL",
                "state": state,
                "delta_from_previous": change,
            }
        )

    def operator_boundary(self, label: str, snapshot: Any) -> None:
        if self._open is None:
            return
        state = _reservoirs(snapshot)
        previous = self._open.boundaries[-1]["state"]
        self._open.boundaries.append(
            {
                "label": str(label),
                "state": state,
                "delta_from_previous": _delta(previous, state),
            }
        )

    def finish_step(self, *, snapshot: Any, core_result: Any) -> None:
        if self._open is None:
            return
        final = _reservoirs(snapshot)
        previous = self._open.boundaries[-1]["state"]
        self._open.boundaries.append(
            {
                "label": "AFTER_PRODUCTION_CORE",
                "state": final,
                "delta_from_previous": _delta(previous, final),
            }
        )
        interaction = getattr(core_result, "interaction", None)
        dump = getattr(core_result, "dump_advance", None)
        avalanche = getattr(core_result, "avalanche_transition", None)
        mobile = (
            getattr(interaction, "mobile_result", None)
            if interaction is not None
            else getattr(dump, "mobile", None)
            if dump is not None
            else None
        )
        deposition = getattr(dump, "deposition", None) if dump is not None else None
        airborne = getattr(dump, "airborne", None) if dump is not None else None
        avalanche_r2m = (
            0.0 if avalanche is None else float(avalanche.cumulative_resting_to_mobile_m3)
        )
        avalanche_m2r = (
            0.0 if avalanche is None else float(avalanche.cumulative_mobile_to_resting_m3)
        )
        declared = {
            "failure_zone_r2m_m3": (
                0.0 if interaction is None else float(interaction.activated_volume_m3)
            ),
            "large_avalanche_r2m_m3": max(
                avalanche_r2m - self._previous_avalanche_r2m_m3, 0.0
            ),
            # DeviceLargeAvalanche owns R->M only. Its cumulative M->R counter
            # observes the shared deposition field; declaring that again here
            # would double-count the Deposition operator transaction below.
            "large_avalanche_m2r_m3": 0.0,
            "avalanche_persistence_observed_m2r_increment_m3": max(
                avalanche_m2r - self._previous_avalanche_m2r_m3, 0.0
            ),
            "tracksoil_r2m_m3": float(self._open.track["accepted_r2m_m3"]),
            "tracksoil_m2r_m3": 0.0,
            "mobile_m2p_m3": (
                0.0
                if interaction is None
                else float(interaction.intake_result.bucket_inflow_volume_m3)
            ),
            "payload_p2m_m3": 0.0,
            "payload_p2a_m3": (
                0.0
                if getattr(core_result, "dump_release", None) is None
                else float(core_result.dump_release.released_volume_m3)
            ),
            "airborne_a2m_m3": (
                0.0 if airborne is None else float(airborne.landed_volume_m3)
            ),
            "airborne_a2r_m3": 0.0,
            "mobile_m2r_deposition_m3": (
                0.0 if deposition is None else float(deposition.deposited_volume_m3)
            ),
            "mobile_transport_volume_before_m3": (
                0.0 if mobile is None else float(mobile.volume_before_m3)
            ),
            "mobile_transport_volume_after_m3": (
                0.0 if mobile is None else float(mobile.volume_after_m3)
            ),
            "mobile_transport_mass_residual_m3": (
                0.0 if mobile is None else float(mobile.transport_mass_residual_m3)
            ),
            "outflow_m3": float(
                final["outflow"] - self._open.before_track["outflow"]
            ),
            "other_m3": 0.0,
        }
        self._previous_avalanche_r2m_m3 = avalanche_r2m
        self._previous_avalanche_m2r_m3 = avalanche_m2r
        whole = _delta(self._open.before_track, final)
        self.records.append(
            {
                "simulation_time_s": self._open.simulation_time_s,
                "phase": self._open.phase,
                "phase_time_s": self._open.phase_time_s,
                "physics_step": self._open.physics_step,
                "reservoirs_m3": final,
                "step_delta_m3": whole,
                "declared_transfers_m3": declared,
                "tracksoil_control_volume": self._open.track,
                "operator_boundaries": self._open.boundaries,
            }
        )
        self._open = None

    def report(self) -> dict[str, Any]:
        if not self.records:
            return {
                "schema": self.schema,
                "first_bad_step_found": False,
                "records": [],
            }
        baseline_records: list[dict[str, Any]] = []
        for row in self.records:
            if row["tracksoil_control_volume"]["active"]:
                break
            baseline_records.append(row)
        if baseline_records:
            floor = max(
                abs(float(boundary["state"]["mass_error_m3"]))
                for row in baseline_records
                for boundary in row["operator_boundaries"]
            )
        else:
            floor = abs(
                float(
                    self.records[0]["operator_boundaries"][0]["state"][
                        "mass_error_m3"
                    ]
                )
            )
        threshold = max(100.0 * floor, 1.0e-10)
        boundaries = [
            {
                "physics_step": row["physics_step"],
                "simulation_time_s": row["simulation_time_s"],
                "phase": row["phase"],
                **boundary,
            }
            for row in self.records
            for boundary in row["operator_boundaries"][1:]
        ]
        first_index = next(
            (
                index
                for index, item in enumerate(boundaries)
                if abs(float(item["state"]["mass_error_m3"])) > threshold
            ),
            None,
        )
        first_bad = None if first_index is None else boundaries[first_index]
        last_good = (
            self.records[0]["operator_boundaries"][0]
            if first_index in {None, 0}
            else boundaries[first_index - 1]
        )
        return {
            "schema": self.schema,
            "location_threshold_m3": threshold,
            "cut_curl_numerical_floor_m3": floor,
            "first_bad_step_found": first_bad is not None,
            "last_good_boundary": last_good,
            "first_bad_boundary": first_bad,
            "maximum_abs_mass_error_m3": max(
                abs(float(row["reservoirs_m3"]["mass_error_m3"]))
                for row in self.records
            ),
            "cumulative_tracksoil_r2m_m3": sum(
                float(row["declared_transfers_m3"]["tracksoil_r2m_m3"])
                for row in self.records
            ),
            "maximum_abs_tracksoil_local_residual_m3": max(
                abs(float(row["tracksoil_control_volume"]["tracksoil_local_residual_m3"]))
                for row in self.records
            ),
            "records": self.records,
        }
