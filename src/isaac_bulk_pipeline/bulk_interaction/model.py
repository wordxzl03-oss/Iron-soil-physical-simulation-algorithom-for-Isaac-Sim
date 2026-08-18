"""Phase-F conservative orchestration from intersection to stable state."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from ..bulk_state import (
    BulkStateManager,
    ConservativeTransfer,
    Reservoir,
    TerrainState,
    TerrainVolumeIntegrator,
)
from ..terrain.terrain_grid import TerrainGrid
from ..tools import ToolDescriptor, ToolState
from .bucket_intake import BucketIntakeModel, BucketIntakeResult
from .deposition import DepositionOperator, DepositionResult
from .failure_zone import FailureZone, FailureZoneModel
from .geometry import ToolTerrainIntersection
from .mobile_layer import MobileLayerResult, MobileLayerSolver
from .yield_criterion import physics_free_surface


@dataclass(frozen=True)
class BulkInteractionResult:
    state: TerrainState
    failure_zone: FailureZone
    mobile_result: MobileLayerResult
    intake_result: BucketIntakeResult
    deposition_result: DepositionResult
    transfers: tuple[ConservativeTransfer, ...]
    activated_volume_m3: float
    minislope_volume_before_m3: float
    minislope_volume_after_m3: float
    minislope_boundary_outflow_m3: float
    activation_tool_impulse_on_mobile_terrain_ns: np.ndarray
    activation_mode: str
    timings_ms: dict[str, float]


class BulkMaterialInteractionModel:
    """Execute one explicit conservative Phase-F material interaction step.

    ``resting_relaxation`` is intentionally a narrow event hook. It receives
    the post-deposition resting H and returns ``(H_stable, boundary_outflow)``.
    It cannot see or modify mobile material, payload, parcels or forces, keeping
    MiniSlope restricted to resting-terrain relaxation.
    """

    def __init__(
        self,
        *,
        failure_zone: FailureZoneModel | None = None,
        mobile_layer: MobileLayerSolver | None = None,
        bucket_intake: BucketIntakeModel | None = None,
        deposition: DepositionOperator | None = None,
    ) -> None:
        self.failure_zone_model = failure_zone or FailureZoneModel()
        self.mobile_layer_solver = mobile_layer or MobileLayerSolver()
        self.bucket_intake_model = bucket_intake or BucketIntakeModel()
        self.deposition_operator = deposition or DepositionOperator()

    def advance(
        self,
        manager: BulkStateManager,
        intersection: ToolTerrainIntersection,
        tool_state: ToolState,
        descriptor: ToolDescriptor,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        action_index: int | None = None,
        resting_relaxation: Callable[[np.ndarray], tuple[np.ndarray, float]] | None = None,
    ) -> BulkInteractionResult:
        before = manager.current_state
        timing_start = perf_counter()
        failure = self.failure_zone_model.compute(
            intersection,
            before.H_resting_m,
            before.material,
            grid,
            integrator,
            fallback_approach_direction_xy=tool_state.pose_terrain[:2, 1],
            H_free_m=physics_free_surface(before.H_resting_m, before.mobile_height_m),
        )
        failure_zone_ms = (perf_counter() - timing_start) * 1_000.0
        activation_mode = "FEE_FAILURE_ZONE"
        activated = np.minimum(failure.active_thickness_m, before.H_resting_m)
        if failure.applicability_status in {
            "OUTSIDE_FEE_DOMAIN",
            "PARTIAL_OUTSIDE_FEE_DOMAIN",
        }:
            # The CAD sweep is an independently measured geometric exclusion
            # volume.  It may conservatively displace material even when the
            # Luengo force closure has no finite wedge for the instantaneous
            # excavator rake.  This is explicitly *not* an invented force
            # correction: SoilForce still receives only the admissible FEE
            # strips, while the extra volume is labelled geometric sweep.
            activated = np.maximum(
                activated,
                np.minimum(intersection.penetration_depth_m, before.H_resting_m),
            )
            activation_mode = "GEOMETRIC_SWEEP_OUTSIDE_FEE_REQUIRES_UNRESOLVED_F_PEN"
        elif not np.any(activated > 0.0):
            activation_mode = "NONE"
        activated_volume = integrator.integrate(activated)
        resting = np.asarray(before.H_resting_m, dtype=np.float64) - activated
        mobile = np.asarray(before.mobile_height_m, dtype=np.float64) + activated
        momentum = np.asarray(before.mobile_momentum_m2_s, dtype=np.float64).copy()
        # V3 invariant: Resting -> Mobile activation transfers mass only.
        # Momentum is introduced later, and only through actual cutting-edge
        # contact cells.  The Mobile solver reports that impulse; its reaction
        # on the tool is exactly equal-and-opposite.
        activation_tool_impulse = np.zeros(3, dtype=np.float64)
        contact = np.asarray(intersection.affected_mask, dtype=bool) & (mobile > 0.0)
        contact_neighborhood = contact.copy()
        contact_neighborhood[1:, :] |= contact[:-1, :]
        contact_neighborhood[:-1, :] |= contact[1:, :]
        contact_neighborhood[:, 1:] |= contact[:, :-1]
        contact_neighborhood[:, :-1] |= contact[:, 1:]
        contact_neighborhood &= mobile > 0.0
        transfers: list[ConservativeTransfer] = []
        if activated_volume > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.MOBILE,
                    activated_volume,
                    (
                        "failure_zone_activation"
                        if activation_mode == "FEE_FAILURE_ZONE"
                        else "cad_geometric_sweep_outside_fee_requires_unresolved_f_pen"
                    ),
                )
            )

        timing_start = perf_counter()
        mobile_result = self.mobile_layer_solver.step(
            resting,
            mobile,
            momentum,
            before.material,
            grid,
            integrator,
            dt_s,
            tool_forcing_mask=contact_neighborhood,
            tool_velocity_xy_m_s=intersection.cutting_edge_velocity_terrain_m_s[:2],
        )
        mobile_layer_ms = (perf_counter() - timing_start) * 1_000.0
        if mobile_result.outflow_volume_m3 > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.MOBILE,
                    Reservoir.OUTFLOW,
                    mobile_result.outflow_volume_m3,
                    "mobile_boundary_outflow",
                )
            )
        timing_start = perf_counter()
        intake_result = self.bucket_intake_model.apply(
            mobile_result.mobile_height_m,
            mobile_result.mobile_momentum_m2_s,
            before.payload,
            tool_state,
            descriptor,
            grid,
            integrator,
            dt_s,
            terrain_surface_m=physics_free_surface(resting, mobile_result.mobile_height_m),
        )
        bucket_intake_ms = (perf_counter() - timing_start) * 1_000.0
        if intake_result.bucket_inflow_volume_m3 > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.MOBILE,
                    Reservoir.PAYLOAD,
                    intake_result.bucket_inflow_volume_m3,
                    "bucket_mouth_relative_flux",
                )
            )
        timing_start = perf_counter()
        deposition_result = self.deposition_operator.apply(
            resting,
            intake_result.mobile_height_m,
            intake_result.mobile_momentum_m2_s,
            before.material,
            grid,
            integrator,
            dt_s,
            active_tool_forcing_mask=contact_neighborhood,
        )
        deposition_ms = (perf_counter() - timing_start) * 1_000.0
        if deposition_result.deposited_volume_m3 > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.MOBILE,
                    Reservoir.RESTING,
                    deposition_result.deposited_volume_m3,
                    "mobile_settling_deposition",
                )
            )

        H_stable = np.asarray(deposition_result.H_resting_m, dtype=np.float64)
        volume_before_relaxation = integrator.integrate(H_stable)
        relaxation_outflow = 0.0
        if resting_relaxation is not None:
            relaxed, reported_outflow = resting_relaxation(np.array(H_stable, copy=True))
            H_stable = np.asarray(grid.validate_heightmap(relaxed), dtype=np.float64)
            relaxation_outflow = float(reported_outflow)
            if not np.isfinite(relaxation_outflow) or relaxation_outflow < 0.0:
                raise ValueError("[BulkInteraction] relaxation outflow invalid")
        volume_after_relaxation = integrator.integrate(H_stable)
        measured_relaxation_loss = volume_before_relaxation - volume_after_relaxation
        tolerance = max(1e-10, 1e-9 * max(volume_before_relaxation, 1.0))
        if abs(measured_relaxation_loss - relaxation_outflow) > tolerance:
            raise ValueError(
                "[BulkInteraction] MiniSlope volume report mismatch: "
                f"measured={measured_relaxation_loss}, reported={relaxation_outflow}"
            )
        if relaxation_outflow > 0.0:
            transfers.append(
                ConservativeTransfer(
                    Reservoir.RESTING,
                    Reservoir.OUTFLOW,
                    relaxation_outflow,
                    "minislope_boundary_outflow",
                )
            )
        next_state = TerrainState(
            H_resting_m=H_stable,
            mobile_height_m=deposition_result.mobile_height_m,
            mobile_momentum_m2_s=deposition_result.mobile_momentum_m2_s,
            payload=intake_result.payload,
            airborne_parcels=before.airborne_parcels,
            material=before.material,
            outflow_volume_m3=(
                before.outflow_volume_m3
                + mobile_result.outflow_volume_m3
                + relaxation_outflow
            ),
            timestamp_s=before.timestamp_s + float(dt_s),
            action_index=(
                before.action_index if action_index is None else int(action_index)
            ),
            _trusted_arrays=True,
        )
        manager.commit_transfers(next_state, transfers)
        return BulkInteractionResult(
            state=manager.current_state,
            failure_zone=failure,
            mobile_result=mobile_result,
            intake_result=intake_result,
            deposition_result=deposition_result,
            transfers=tuple(transfers),
            activated_volume_m3=activated_volume,
            minislope_volume_before_m3=volume_before_relaxation,
            minislope_volume_after_m3=volume_after_relaxation,
            minislope_boundary_outflow_m3=relaxation_outflow,
            activation_tool_impulse_on_mobile_terrain_ns=activation_tool_impulse,
            activation_mode=activation_mode,
            timings_ms={
                "failure_zone": failure_zone_ms,
                "mobile_layer": mobile_layer_ms,
                "bucket_intake": bucket_intake_ms,
                "deposition": deposition_ms,
            },
        )
