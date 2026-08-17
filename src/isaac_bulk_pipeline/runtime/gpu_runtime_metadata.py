"""Non-spatial reservoir metadata for the production device runtime.

Spatial material lives only in :class:`DeviceBulkState`; payload and sparse
airborne parcels are deliberately small host metadata.  This class is the
logical transaction boundary joining those scalar reservoirs to device-side
reductions without reconstructing a host TerrainState.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import MaterialParcel, PayloadState
from .bulk_state_authority import DeviceBulkState, DeviceMaterialLedger, DeviceMaterialLedgerSnapshot


@dataclass(frozen=True)
class GpuRuntimeScalarSnapshot:
    payload: PayloadState
    airborne_volume_m3: float
    airborne_parcel_count: int
    outflow_volume_m3: float
    deposited_volume_m3: float
    timestamp_s: float
    material_ledger: DeviceMaterialLedgerSnapshot


class GpuRuntimeMetadata:
    """Atomic logical reservoir metadata paired with DeviceBulkState fields."""

    def __init__(self, state: DeviceBulkState, payload: PayloadState) -> None:
        self._payload = payload
        self._airborne: tuple[MaterialParcel, ...] = ()
        self._outflow_m3 = 0.0
        self._deposited_m3 = 0.0
        self._timestamp_s = 0.0
        self._ledger = DeviceMaterialLedger(
            state,
            payload.assumed_bulk_density_kg_m3,
            payload_m3=payload.volume_m3,
        )

    @property
    def payload(self) -> PayloadState:
        return self._payload

    @property
    def airborne(self) -> tuple[MaterialParcel, ...]:
        return self._airborne

    @property
    def timestamp_s(self) -> float:
        return self._timestamp_s

    def advance_time(self, dt_s: float) -> None:
        dt = float(dt_s)
        if dt < 0.0:
            raise ValueError("[GpuRuntimeMetadata] dt_s must be non-negative")
        self._timestamp_s += dt

    def commit_mobile_to_payload(self, before: PayloadState, after: PayloadState, volume_m3: float) -> None:
        if abs(before.volume_m3 - self._payload.volume_m3) > 1e-12:
            raise RuntimeError("[GpuRuntimeMetadata] stale payload transaction")
        if abs((after.volume_m3 - before.volume_m3) - float(volume_m3)) > 1e-10:
            raise RuntimeError("[GpuRuntimeMetadata] Mobile→Payload transaction mismatch")
        self._payload = after

    def commit_payload_to_airborne(self, after_payload: PayloadState, parcels: tuple[MaterialParcel, ...], released_m3: float) -> None:
        before = self._payload.volume_m3
        if abs((before - after_payload.volume_m3) - float(released_m3)) > 1e-10:
            raise RuntimeError("[GpuRuntimeMetadata] Payload→Airborne transaction mismatch")
        if abs(sum(item.volume_m3 for item in parcels) - float(released_m3)) > 1e-10:
            raise RuntimeError("[GpuRuntimeMetadata] airborne parcel volume mismatch")
        self._payload = after_payload
        self._airborne = self._airborne + tuple(parcels)

    def commit_airborne_advance(self, remaining: tuple[MaterialParcel, ...], landed_m3: float, deposited_m3: float, dt_s: float) -> None:
        prior = sum(item.volume_m3 for item in self._airborne)
        next_volume = sum(item.volume_m3 for item in remaining)
        if abs(prior - next_volume - float(landed_m3)) > 1e-10:
            raise RuntimeError("[GpuRuntimeMetadata] Airborne→Mobile transaction mismatch")
        self._airborne = tuple(remaining)
        self._deposited_m3 += float(deposited_m3)
        self.advance_time(dt_s)

    def add_outflow(self, volume_m3: float) -> None:
        if volume_m3 < 0.0:
            raise ValueError("[GpuRuntimeMetadata] outflow must be non-negative")
        self._outflow_m3 += float(volume_m3)

    def reset_payload(self, state: DeviceBulkState, payload: PayloadState) -> None:
        """Replace payload at an explicit reset boundary and rebase the ledger."""

        self._payload = payload
        self._ledger = DeviceMaterialLedger(
            state,
            payload.assumed_bulk_density_kg_m3,
            payload_m3=payload.volume_m3,
            airborne_m3=sum(item.volume_m3 for item in self._airborne),
            outflow_m3=self._outflow_m3,
        )

    def snapshot(self, state: DeviceBulkState) -> GpuRuntimeScalarSnapshot:
        airborne_volume = sum(item.volume_m3 for item in self._airborne)
        ledger = self._ledger.snapshot(
            state,
            payload_m3=self._payload.volume_m3,
            airborne_m3=airborne_volume,
            outflow_m3=self._outflow_m3,
        )
        return GpuRuntimeScalarSnapshot(
            payload=self._payload,
            airborne_volume_m3=airborne_volume,
            airborne_parcel_count=len(self._airborne),
            outflow_volume_m3=self._outflow_m3,
            deposited_volume_m3=self._deposited_m3,
            timestamp_s=self._timestamp_s,
            material_ledger=ledger,
        )

    def checkpoint_record(self) -> dict[str, object]:
        """Serialize scalar/sparse reservoirs at an explicit checkpoint boundary."""

        payload = self._payload
        return {
            "payload": {
                "volume_m3": payload.volume_m3,
                "capacity_m3": payload.capacity_m3,
                "assumed_bulk_density_kg_m3": payload.assumed_bulk_density_kg_m3,
                "center_of_mass_bucket_frame_m": payload.center_of_mass_bucket_frame_m.tolist(),
            },
            "airborne": [
                {
                    "parcel_id": parcel.parcel_id,
                    "volume_m3": parcel.volume_m3,
                    "assumed_bulk_density_kg_m3": parcel.assumed_bulk_density_kg_m3,
                    "position_world_m": parcel.position_world_m.tolist(),
                    "velocity_world_m_s": parcel.velocity_world_m_s.tolist(),
                    "timestamp_s": parcel.timestamp_s,
                    "source": parcel.source,
                    "footprint": (
                        None
                        if parcel.footprint is None
                        else {
                            "lateral_axis_world_m": parcel.footprint.lateral_axis_world_m.tolist(),
                            "lateral_extent_m": parcel.footprint.lateral_extent_m,
                            "longitudinal_extent_m": parcel.footprint.longitudinal_extent_m,
                            "provenance": parcel.footprint.provenance,
                            "mass_profile": parcel.footprint.mass_profile,
                        }
                    ),
                }
                for parcel in self._airborne
            ],
            "outflow_m3": self._outflow_m3,
            "deposited_m3": self._deposited_m3,
            "timestamp_s": self._timestamp_s,
            "ledger_initial_total_m3": self._ledger.initial_total_m3,
        }

    @classmethod
    def from_checkpoint_record(
        cls,
        state: DeviceBulkState,
        record: dict[str, object],
        *,
        preserve_ledger_reference: bool = True,
    ) -> "GpuRuntimeMetadata":
        """Restore metadata without rebasing the material-conservation ledger."""

        payload_record = dict(record["payload"])
        payload = PayloadState(
            float(payload_record["volume_m3"]),
            float(payload_record["capacity_m3"]),
            float(payload_record["assumed_bulk_density_kg_m3"]),
            np.asarray(payload_record["center_of_mass_bucket_frame_m"], dtype=np.float64),
        )
        result = cls(state, payload)
        from ..bulk_state import AirborneParcelFootprint

        result._airborne = tuple(
            MaterialParcel(
                str(item["parcel_id"]),
                float(item["volume_m3"]),
                float(item["assumed_bulk_density_kg_m3"]),
                np.asarray(item["position_world_m"], dtype=np.float64),
                np.asarray(item["velocity_world_m_s"], dtype=np.float64),
                float(item["timestamp_s"]),
                str(item["source"]),
                (
                    None
                    if item.get("footprint") is None
                    else AirborneParcelFootprint(
                        np.asarray(
                            item["footprint"]["lateral_axis_world_m"],
                            dtype=np.float64,
                        ),
                        float(item["footprint"]["lateral_extent_m"]),
                        float(item["footprint"]["longitudinal_extent_m"]),
                        str(item["footprint"]["provenance"]),
                        str(item["footprint"].get("mass_profile", "UNIFORM_RECTANGLE")),
                    )
                ),
            )
            for item in record.get("airborne", [])
        )
        result._outflow_m3 = float(record.get("outflow_m3", 0.0))
        result._deposited_m3 = float(record.get("deposited_m3", 0.0))
        result._timestamp_s = float(record.get("timestamp_s", 0.0))
        result._ledger = DeviceMaterialLedger(
            state,
            payload.assumed_bulk_density_kg_m3,
            payload_m3=payload.volume_m3,
            airborne_m3=sum(parcel.volume_m3 for parcel in result._airborne),
            outflow_m3=result._outflow_m3,
            initial_total_m3=(
                float(record["ledger_initial_total_m3"])
                if preserve_ledger_reference
                else None
            ),
        )
        return result
