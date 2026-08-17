"""Strict multi-reservoir volume ledger for the Phase-E state model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .models import TerrainState, UNCALIBRATED_LABEL
from .volume_integrator import SurfaceTopology, TerrainVolumeIntegrator


class Reservoir(str, Enum):
    RESTING = "resting"
    MOBILE = "mobile"
    PAYLOAD = "payload"
    AIRBORNE = "airborne"
    OUTFLOW = "outflow"


@dataclass(frozen=True)
class ReservoirVolumes:
    resting_m3: float
    mobile_m3: float
    payload_m3: float
    airborne_m3: float
    outflow_m3: float

    @property
    def total_including_outflow_m3(self) -> float:
        return float(
            self.resting_m3
            + self.mobile_m3
            + self.payload_m3
            + self.airborne_m3
            + self.outflow_m3
        )

    def get(self, reservoir: Reservoir | str) -> float:
        selected = Reservoir(reservoir)
        return {
            Reservoir.RESTING: self.resting_m3,
            Reservoir.MOBILE: self.mobile_m3,
            Reservoir.PAYLOAD: self.payload_m3,
            Reservoir.AIRBORNE: self.airborne_m3,
            Reservoir.OUTFLOW: self.outflow_m3,
        }[selected]


@dataclass(frozen=True)
class MassBalanceReport:
    """Volume balance and density-derived mass-error estimate."""

    boundary_condition: str
    signed_volume_error_m3: float
    absolute_volume_error_m3: float
    relative_volume_error: float
    estimated_absolute_mass_error_kg: float
    assumed_bulk_density_kg_m3: float
    mass_value_kind: str = "density-based estimate"
    parameter_status: str = UNCALIBRATED_LABEL


@dataclass(frozen=True)
class MassLedgerSnapshot:
    initial_accounted_volume_m3: float
    reservoirs: ReservoirVolumes
    balance: MassBalanceReport
    transfer_count: int


@dataclass(frozen=True)
class ConservativeTransfer:
    """An explicit reservoir-to-reservoir volume movement, not a solver."""

    source: Reservoir
    destination: Reservoir
    volume_m3: float
    reason: str

    def __post_init__(self) -> None:
        try:
            source = Reservoir(self.source)
            destination = Reservoir(self.destination)
        except ValueError as exc:
            raise ValueError("[ConservativeTransfer] invalid reservoir") from exc
        volume = float(self.volume_m3)
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(
                "[ConservativeTransfer] volume_m3 must be finite and > 0"
            )
        if source is destination:
            raise ValueError(
                "[ConservativeTransfer] source and destination must differ"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("[ConservativeTransfer] reason must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "volume_m3", volume)


def reservoir_volumes_from_state(
    state: TerrainState,
    integrator: TerrainVolumeIntegrator,
) -> ReservoirVolumes:
    """Compute all five scalar reservoirs from one authoritative state."""

    if integrator.topology is not SurfaceTopology.TRIANGLE_A_C:
        raise ValueError(
            "[MassLedger] Phase-E authoritative state requires triangle_a_c "
            "integration matching DynamicMeshAdapter"
        )
    if state.H_resting_m.shape != integrator.shape:
        raise ValueError(
            "[MassLedger] state/integrator shape mismatch: "
            f"state={state.H_resting_m.shape}, integrator={integrator.shape}"
        )
    return ReservoirVolumes(
        resting_m3=integrator.integrate(state.h_resting_derived_m),
        mobile_m3=integrator.integrate(state.mobile_height_m),
        payload_m3=float(state.payload.volume_m3),
        airborne_m3=float(state.airborne_volume_m3),
        outflow_m3=float(state.outflow_volume_m3),
    )


class MassLedger:
    """Atomic, capacity-aware accounting across all Phase-E reservoirs."""

    def __init__(
        self,
        initial_volumes: ReservoirVolumes,
        *,
        payload_capacity_m3: float,
        assumed_bulk_density_kg_m3: float,
        boundary_condition: str,
        absolute_tolerance_m3: float = 1e-10,
        relative_tolerance: float = 1e-10,
    ) -> None:
        if boundary_condition not in {"closed", "open"}:
            raise ValueError("[MassLedger] boundary_condition must be closed or open")
        values = np.asarray(
            [
                initial_volumes.resting_m3,
                initial_volumes.mobile_m3,
                initial_volumes.payload_m3,
                initial_volumes.airborne_m3,
                initial_volumes.outflow_m3,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("[MassLedger] initial reservoirs must be finite/non-negative")
        capacity = float(payload_capacity_m3)
        density = float(assumed_bulk_density_kg_m3)
        absolute = float(absolute_tolerance_m3)
        relative = float(relative_tolerance)
        if (
            not np.all(np.isfinite([capacity, density, absolute, relative]))
            or capacity <= 0.0
            or density <= 0.0
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError("[MassLedger] invalid capacity/density/tolerance")
        if initial_volumes.payload_m3 > capacity + absolute:
            raise ValueError("[MassLedger] initial payload exceeds capacity")
        if boundary_condition == "closed" and initial_volumes.outflow_m3 > absolute:
            raise ValueError("[MassLedger] closed boundary cannot start with outflow")
        self._boundary_condition = boundary_condition
        self._payload_capacity_m3 = capacity
        self._density = density
        self._absolute_tolerance_m3 = absolute
        self._relative_tolerance = relative
        self._initial_volumes = initial_volumes
        self._initial_accounted_volume_m3 = (
            initial_volumes.total_including_outflow_m3
        )
        self._volumes = self._mapping(initial_volumes)
        self._transfer_count = 0

    @classmethod
    def initialize(
        cls,
        state: TerrainState,
        integrator: TerrainVolumeIntegrator,
        *,
        boundary_condition: str,
        absolute_tolerance_m3: float = 1e-10,
        relative_tolerance: float = 1e-10,
    ) -> "MassLedger":
        volumes = reservoir_volumes_from_state(state, integrator)
        return cls(
            volumes,
            payload_capacity_m3=state.payload.capacity_m3,
            assumed_bulk_density_kg_m3=(
                state.material.assumed_bulk_density_kg_m3
            ),
            boundary_condition=boundary_condition,
            absolute_tolerance_m3=absolute_tolerance_m3,
            relative_tolerance=relative_tolerance,
        )

    def transfer(self, transfer: ConservativeTransfer) -> None:
        if not isinstance(transfer, ConservativeTransfer):
            raise TypeError("[MassLedger] transfer must be ConservativeTransfer")
        if transfer.source is Reservoir.OUTFLOW:
            raise ValueError("[MassLedger] outflow is irreversible")
        if (
            transfer.destination is Reservoir.OUTFLOW
            and self._boundary_condition == "closed"
        ):
            raise ValueError("[MassLedger] closed boundary forbids outflow")
        source_volume = self._volumes[transfer.source]
        if transfer.volume_m3 > source_volume + self._absolute_tolerance_m3:
            raise ValueError(
                "[MassLedger] transfer exceeds source reservoir: "
                f"requested={transfer.volume_m3}, available={source_volume}, "
                f"source={transfer.source.value}"
            )
        destination_volume = self._volumes[transfer.destination] + transfer.volume_m3
        if (
            transfer.destination is Reservoir.PAYLOAD
            and destination_volume
            > self._payload_capacity_m3 + self._absolute_tolerance_m3
        ):
            raise ValueError(
                "[MassLedger] transfer would exceed payload capacity: "
                f"result={destination_volume}, capacity={self._payload_capacity_m3}"
            )
        updated_source = source_volume - transfer.volume_m3
        if updated_source < -self._absolute_tolerance_m3:
            raise ValueError("[MassLedger] transfer would create a negative reservoir")
        self._volumes[transfer.source] = max(0.0, updated_source)
        self._volumes[transfer.destination] = destination_volume
        self._transfer_count += 1
        self._assert_internal_balance()

    def assert_matches(
        self,
        state: TerrainState,
        integrator: TerrainVolumeIntegrator,
    ) -> None:
        actual = reservoir_volumes_from_state(state, integrator)
        for reservoir in Reservoir:
            expected_value = self._volumes[reservoir]
            actual_value = actual.get(reservoir)
            tolerance = self._comparison_tolerance(expected_value)
            if abs(actual_value - expected_value) > tolerance:
                raise ValueError(
                    "[MassLedger] state does not match explicit transfers: "
                    f"reservoir={reservoir.value}, expected={expected_value}, "
                    f"actual={actual_value}, tolerance={tolerance}"
                )
        report = self.balance_report()
        tolerance = self._comparison_tolerance(
            self._initial_accounted_volume_m3
        )
        if report.absolute_volume_error_m3 > tolerance:
            raise ValueError(
                "[MassLedger] mass/volume balance is not closed: "
                f"error_m3={report.signed_volume_error_m3}"
            )

    def balance_report(self) -> MassBalanceReport:
        volumes = self._as_volumes()
        if self._boundary_condition == "closed":
            accounted = (
                volumes.resting_m3
                + volumes.mobile_m3
                + volumes.payload_m3
                + volumes.airborne_m3
            )
        else:
            accounted = volumes.total_including_outflow_m3
        signed_error = float(self._initial_accounted_volume_m3 - accounted)
        absolute_error = abs(signed_error)
        denominator = max(abs(self._initial_accounted_volume_m3), 1e-15)
        return MassBalanceReport(
            boundary_condition=self._boundary_condition,
            signed_volume_error_m3=signed_error,
            absolute_volume_error_m3=absolute_error,
            relative_volume_error=float(absolute_error / denominator),
            estimated_absolute_mass_error_kg=float(absolute_error * self._density),
            assumed_bulk_density_kg_m3=self._density,
        )

    def snapshot(self) -> MassLedgerSnapshot:
        return MassLedgerSnapshot(
            initial_accounted_volume_m3=self._initial_accounted_volume_m3,
            reservoirs=self._as_volumes(),
            balance=self.balance_report(),
            transfer_count=self._transfer_count,
        )

    def clone(self) -> "MassLedger":
        clone = MassLedger(
            self._initial_volumes,
            payload_capacity_m3=self._payload_capacity_m3,
            assumed_bulk_density_kg_m3=self._density,
            boundary_condition=self._boundary_condition,
            absolute_tolerance_m3=self._absolute_tolerance_m3,
            relative_tolerance=self._relative_tolerance,
        )
        clone._volumes = dict(self._volumes)
        clone._transfer_count = self._transfer_count
        return clone

    def reset(self) -> None:
        self._volumes = self._mapping(self._initial_volumes)
        self._transfer_count = 0

    def _comparison_tolerance(self, reference: float) -> float:
        return max(
            self._absolute_tolerance_m3,
            self._relative_tolerance * max(abs(reference), 1.0),
        )

    def _assert_internal_balance(self) -> None:
        values = np.asarray(list(self._volumes.values()), dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("[MassLedger] invalid internal reservoir state")
        total = float(np.sum(values, dtype=np.float64))
        if abs(total - self._initial_accounted_volume_m3) > self._comparison_tolerance(
            self._initial_accounted_volume_m3
        ):
            raise ValueError("[MassLedger] internal transfer violated conservation")

    def _as_volumes(self) -> ReservoirVolumes:
        return ReservoirVolumes(
            resting_m3=float(self._volumes[Reservoir.RESTING]),
            mobile_m3=float(self._volumes[Reservoir.MOBILE]),
            payload_m3=float(self._volumes[Reservoir.PAYLOAD]),
            airborne_m3=float(self._volumes[Reservoir.AIRBORNE]),
            outflow_m3=float(self._volumes[Reservoir.OUTFLOW]),
        )

    @staticmethod
    def _mapping(volumes: ReservoirVolumes) -> dict[Reservoir, float]:
        return {
            Reservoir.RESTING: float(volumes.resting_m3),
            Reservoir.MOBILE: float(volumes.mobile_m3),
            Reservoir.PAYLOAD: float(volumes.payload_m3),
            Reservoir.AIRBORNE: float(volumes.airborne_m3),
            Reservoir.OUTFLOW: float(volumes.outflow_m3),
        }
