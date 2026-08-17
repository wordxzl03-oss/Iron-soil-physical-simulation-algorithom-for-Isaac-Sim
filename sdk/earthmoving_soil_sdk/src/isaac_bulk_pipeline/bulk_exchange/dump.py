"""Conservative payload release, ballistic landing and terrain deposition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np

from ..bulk_interaction import (
    DepositionOperator,
    DepositionResult,
    MobileLayerResult,
    MobileLayerSolver,
)
from ..bulk_state import (
    BucketFillPhase,
    BulkStateManager,
    ConservativeTransfer,
    Reservoir,
    TerrainState,
    TerrainVolumeIntegrator,
)
from ..terrain.terrain_grid import TerrainGrid
from ..tools import ToolDescriptor, ToolState
from .airborne import AirborneAdvanceResult, AirborneParcelModel
from .retention import BucketRetentionResult, BucketRetentionSpillModel


class DumpTarget(str, Enum):
    TERRAIN = "terrain"
    RECEIVER = "receiver"
    EXPORTED = "exported"


@dataclass(frozen=True)
class DumpReleaseResult:
    state: TerrainState
    target: DumpTarget
    retention: BucketRetentionResult
    released_volume_m3: float
    created_parcel_count: int
    exported_volume_m3: float
    transfers: tuple[ConservativeTransfer, ...]


@dataclass(frozen=True)
class DumpAdvanceResult:
    state: TerrainState
    airborne: AirborneAdvanceResult
    mobile: MobileLayerResult
    deposition: DepositionResult | None
    transfers: tuple[ConservativeTransfer, ...]
    minislope_volume_before_m3: float
    minislope_volume_after_m3: float
    minislope_boundary_outflow_m3: float


class TerrainDumpOperator:
    def __init__(
        self,
        *,
        retention: BucketRetentionSpillModel | None = None,
        airborne: AirborneParcelModel | None = None,
        mobile: MobileLayerSolver | None = None,
        deposition: DepositionOperator | None = None,
    ) -> None:
        self.retention_model = retention or BucketRetentionSpillModel()
        self.airborne_model = airborne or AirborneParcelModel()
        self.mobile_layer_solver = mobile or MobileLayerSolver()
        self.deposition_operator = deposition or DepositionOperator()

    def release(
        self,
        manager: BulkStateManager,
        tool_state: ToolState,
        descriptor: ToolDescriptor,
        *,
        target: DumpTarget | str = DumpTarget.TERRAIN,
        bucket_linear_acceleration_terrain_m_s2: np.ndarray | None = None,
    ) -> DumpReleaseResult:
        selected = DumpTarget(target)
        before = manager.current_state
        retention = self.retention_model.evaluate(
            before.payload,
            descriptor,
            tool_state,
            material_stop_angle_deg=before.material.stop_angle_deg,
            bucket_linear_acceleration_terrain_m_s2=(
                bucket_linear_acceleration_terrain_m_s2
            ),
        )
        transfers: list[ConservativeTransfer] = []
        parcels = before.airborne_parcels
        exported = 0.0
        if selected is DumpTarget.TERRAIN:
            released = retention.spill_volume_m3
            new_parcels = self.airborne_model.create_from_bucket_release(
                released,
                before.material,
                tool_state,
                descriptor,
                timestamp_s=before.timestamp_s,
                source="bucket_spill_or_dump",
                id_prefix=f"a{before.action_index:04d}_t{int(round(before.timestamp_s * 1000)):010d}",
                free_surface_normal_bucket_frame=(
                    retention.free_surface_normal_bucket_frame
                ),
                payload_center_of_mass_bucket_frame_m=(
                    before.payload.center_of_mass_bucket_frame_m
                ),
            )
            existing_ids = {item.parcel_id for item in parcels}
            if any(item.parcel_id in existing_ids for item in new_parcels):
                raise ValueError("[Dump] generated parcel ID collision")
            parcels = tuple(parcels) + tuple(new_parcels)
            payload = retention.payload
            if released > 0.0:
                transfers.append(
                    ConservativeTransfer(
                        Reservoir.PAYLOAD,
                        Reservoir.AIRBORNE,
                        released,
                        "bucket_retention_spill",
                    )
                )
        else:
            released = before.payload.volume_m3
            exported = released
            payload = self.retention_model.internal_fill_model.solve_payload(
                before.payload,
                descriptor,
                retention.free_surface_normal_bucket_frame,
                volume_m3=0.0,
                phase=BucketFillPhase.STATIC,
            )
            if exported > 0.0:
                transfers.append(
                    ConservativeTransfer(
                        Reservoir.PAYLOAD,
                        Reservoir.OUTFLOW,
                        exported,
                        f"dump_to_{selected.value}",
                    )
                )
        next_state = TerrainState(
            H_resting_m=before.H_resting_m,
            mobile_height_m=before.mobile_height_m,
            mobile_momentum_m2_s=before.mobile_momentum_m2_s,
            payload=payload,
            airborne_parcels=parcels,
            material=before.material,
            outflow_volume_m3=before.outflow_volume_m3 + exported,
            timestamp_s=before.timestamp_s,
            action_index=before.action_index,
            _trusted_arrays=True,
        )
        manager.commit_transfers(next_state, transfers)
        return DumpReleaseResult(
            state=manager.current_state,
            target=selected,
            retention=retention,
            released_volume_m3=released,
            created_parcel_count=len(parcels) - len(before.airborne_parcels),
            exported_volume_m3=exported,
            transfers=tuple(transfers),
        )

    def advance_airborne(
        self,
        manager: BulkStateManager,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        settle_landed_mobile: bool = True,
        resting_relaxation: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None,
    ) -> DumpAdvanceResult:
        before = manager.current_state
        airborne = self.airborne_model.advance(
            before.airborne_parcels,
            before.H_resting_m,
            before.mobile_height_m,
            before.mobile_momentum_m2_s,
            grid,
            integrator,
            dt_s,
        )
        transfers: list[ConservativeTransfer] = []
        if airborne.landed_volume_m3 > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.AIRBORNE,
                    Reservoir.MOBILE,
                    airborne.landed_volume_m3,
                    "ballistic_parcel_landing",
                )
            )
        resting = before.H_resting_m
        mobile_result = self.mobile_layer_solver.step(
            resting,
            airborne.mobile_height_m,
            airborne.mobile_momentum_m2_s,
            before.material,
            grid,
            integrator,
            dt_s,
        )
        if mobile_result.outflow_volume_m3 > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.MOBILE,
                    Reservoir.OUTFLOW,
                    mobile_result.outflow_volume_m3,
                    "post_landing_mobile_outflow",
                )
            )
        deposition: DepositionResult | None = None
        mobile = mobile_result.mobile_height_m
        momentum = mobile_result.mobile_momentum_m2_s
        if settle_landed_mobile:
            deposition = self.deposition_operator.apply(
                resting,
                mobile,
                momentum,
                before.material,
                grid,
                integrator,
                dt_s,
            )
            resting = deposition.H_resting_m
            mobile = deposition.mobile_height_m
            momentum = deposition.mobile_momentum_m2_s
            if deposition.deposited_volume_m3 > 0.0:
                transfers.append(
                    ConservativeTransfer(
                        Reservoir.MOBILE,
                        Reservoir.RESTING,
                        deposition.deposited_volume_m3,
                        "landed_mobile_deposition",
                    )
                )
        volume_before_relaxation = integrator.integrate(resting)
        relaxation_outflow = 0.0
        if resting_relaxation is not None:
            relaxed, reported_outflow = resting_relaxation(np.array(resting, copy=True))
            resting = np.asarray(grid.validate_heightmap(relaxed), dtype=np.float64)
            relaxation_outflow = float(reported_outflow)
            if not np.isfinite(relaxation_outflow) or relaxation_outflow < 0.0:
                raise ValueError("[Dump] MiniSlope boundary outflow must be non-negative")
        volume_after_relaxation = integrator.integrate(resting)
        measured_loss = volume_before_relaxation - volume_after_relaxation
        tolerance = max(1e-10, 1e-9 * max(volume_before_relaxation, 1.0))
        if abs(measured_loss - relaxation_outflow) > tolerance:
            raise ValueError(
                "[Dump] MiniSlope volume report mismatch: "
                f"measured={measured_loss}, reported={relaxation_outflow}"
            )
        if relaxation_outflow > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.OUTFLOW,
                    relaxation_outflow,
                    "post_dump_minislope_boundary_outflow",
                )
            )
        next_state = TerrainState(
            H_resting_m=resting,
            mobile_height_m=mobile,
            mobile_momentum_m2_s=momentum,
            payload=before.payload,
            airborne_parcels=airborne.remaining_parcels,
            material=before.material,
            outflow_volume_m3=(
                before.outflow_volume_m3
                + mobile_result.outflow_volume_m3
                + relaxation_outflow
            ),
            timestamp_s=before.timestamp_s + float(dt_s),
            action_index=before.action_index,
            _trusted_arrays=True,
        )
        manager.commit_transfers(next_state, transfers)
        return DumpAdvanceResult(
            state=manager.current_state,
            airborne=airborne,
            mobile=mobile_result,
            deposition=deposition,
            transfers=tuple(transfers),
            minislope_volume_before_m3=volume_before_relaxation,
            minislope_volume_after_m3=volume_after_relaxation,
            minislope_boundary_outflow_m3=relaxation_outflow,
        )
