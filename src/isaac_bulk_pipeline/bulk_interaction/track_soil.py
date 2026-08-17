"""Local conservative reduced-order track/rut material transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import TerrainVolumeIntegrator
from ..performance import ActiveDomainManager, ActiveReason
from ..terrain import TerrainGrid


@dataclass(frozen=True)
class TrackSoilConfig:
    sinkage_rate_m_s: float = 0.012
    slip_gain: float = 0.8
    maximum_sinkage_per_step_m: float = 0.004
    maximum_total_rut_depth_m: float = 0.30
    shoulder_halo_cells: int = 3
    minimum_slip_speed_m_s: float = 0.02
    parameter_status: str = "ENGINEERING_REDUCED_ORDER_NOT_SITE_CALIBRATED"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.sinkage_rate_m_s,
                self.slip_gain,
                self.maximum_sinkage_per_step_m,
                self.maximum_total_rut_depth_m,
                self.minimum_slip_speed_m_s,
            ]
        )
        if np.any(values < 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("[TrackSoil] configuration must be finite/non-negative")
        if self.shoulder_halo_cells < 1:
            raise ValueError("[TrackSoil] shoulder_halo_cells must be >= 1")


@dataclass(frozen=True)
class TrackSoilResult:
    H_resting_m: np.ndarray
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    resting_to_mobile_volume_m3: float
    left_mean_sinkage_m: float
    right_mean_sinkage_m: float
    active_bbox_grid: tuple[int, int, int, int]
    parameter_status: str


class TrackSoilModel:
    """Convert local track sinkage into equal mobile shoulder volume.

    This is an explicitly engineering reduced-order rut model, not Bekker/Wong
    validation. It exists to close the conservative runtime path while site
    pressure-sinkage and shear parameters remain unavailable.
    """

    def __init__(self, config: TrackSoilConfig | None = None, *, tile_size: int = 64) -> None:
        self.config = config or TrackSoilConfig()
        self.tile_size = tile_size
        self._initial_resting: np.ndarray | None = None
        self._domain: ActiveDomainManager | None = None

    def initialize(self, resting_height_m: np.ndarray) -> None:
        height = np.asarray(resting_height_m, dtype=np.float64)
        if height.ndim != 2 or not np.all(np.isfinite(height)):
            raise ValueError("[TrackSoil] initial terrain invalid")
        self._initial_resting = np.array(height, copy=True)
        self._domain = ActiveDomainManager(height.shape, self.tile_size)

    def apply(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        *,
        left_footprint_mask: np.ndarray,
        right_footprint_mask: np.ndarray,
        left_track_velocity_xy_m_s: np.ndarray,
        right_track_velocity_xy_m_s: np.ndarray,
        base_velocity_xy_m_s: np.ndarray,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
    ) -> TrackSoilResult:
        if self._initial_resting is None or self._domain is None:
            self.initialize(H_resting_m)
        resting = np.asarray(H_resting_m, dtype=np.float64)
        mobile = np.asarray(mobile_height_m, dtype=np.float64)
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64)
        left = np.asarray(left_footprint_mask, dtype=bool)
        right = np.asarray(right_footprint_mask, dtype=bool)
        if any(item.shape != grid.shape for item in (resting, mobile, left, right)):
            raise ValueError("[TrackSoil] grid/mask shape mismatch")
        if momentum.shape != grid.shape + (2,):
            raise ValueError("[TrackSoil] momentum shape mismatch")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[TrackSoil] dt must be positive")
        velocities = [
            np.asarray(left_track_velocity_xy_m_s, dtype=np.float64),
            np.asarray(right_track_velocity_xy_m_s, dtype=np.float64),
        ]
        base = np.asarray(base_velocity_xy_m_s, dtype=np.float64)
        if base.shape != (2,) or any(item.shape != (2,) for item in velocities):
            raise ValueError("[TrackSoil] velocities must have shape (2,)")

        next_resting = np.array(resting, copy=True)
        next_mobile = np.array(mobile, copy=True)
        next_momentum = np.array(momentum, copy=True)
        weights = integrator.vertex_weights_m2
        total_volume = 0.0
        mean_sinkages: list[float] = []
        active = left | right
        for footprint, track_velocity in zip((left, right), velocities):
            if not np.any(footprint):
                mean_sinkages.append(0.0)
                continue
            slip = float(np.linalg.norm(track_velocity - base))
            slip_effect = max(slip - self.config.minimum_slip_speed_m_s, 0.0)
            requested = min(
                self.config.maximum_sinkage_per_step_m,
                self.config.sinkage_rate_m_s * dt * (1.0 + self.config.slip_gain * slip_effect),
            )
            existing_rut = np.maximum(
                self._initial_resting[footprint] - next_resting[footprint], 0.0
            )
            available_rut = np.maximum(self.config.maximum_total_rut_depth_m - existing_rut, 0.0)
            sinkage = np.minimum(requested, np.minimum(next_resting[footprint], available_rut))
            removed_volume = float(np.sum(sinkage * weights[footprint], dtype=np.float64))
            next_resting[footprint] -= sinkage
            shoulder = self._buffer(footprint, self.config.shoulder_halo_cells) & ~footprint
            shoulder_weight = float(np.sum(weights[shoulder], dtype=np.float64))
            if removed_volume > 0.0 and shoulder_weight <= 0.0:
                raise RuntimeError("[TrackSoil] no shoulder area for conservative rut transfer")
            if removed_volume > 0.0:
                added_height = removed_volume / shoulder_weight
                next_mobile[shoulder] += added_height
                next_momentum[shoulder] += added_height * track_velocity
            total_volume += removed_volume
            mean_sinkages.append(float(np.mean(sinkage)) if sinkage.size else 0.0)
            active |= shoulder
        self._domain.clear()
        if np.any(active):
            self._domain.mark_mask(active, ActiveReason.TRACK_LEFT | ActiveReason.TRACK_RIGHT)
        bbox = self._domain.combined_window()
        measured_resting_loss = -integrator.integrate_delta(resting, next_resting)
        measured_mobile_gain = integrator.integrate_delta(mobile, next_mobile)
        tolerance = max(1.0e-11, 1.0e-9 * max(total_volume, 1.0))
        if abs(measured_resting_loss - measured_mobile_gain) > tolerance:
            raise RuntimeError("[TrackSoil] resting/mobile transfer is not conservative")
        return TrackSoilResult(
            H_resting_m=next_resting,
            mobile_height_m=next_mobile,
            mobile_momentum_m2_s=next_momentum,
            resting_to_mobile_volume_m3=total_volume,
            left_mean_sinkage_m=mean_sinkages[0],
            right_mean_sinkage_m=mean_sinkages[1],
            active_bbox_grid=(bbox.row_start, bbox.col_start, bbox.row_stop, bbox.col_stop),
            parameter_status=self.config.parameter_status,
        )

    @staticmethod
    def _buffer(mask: np.ndarray, cells: int) -> np.ndarray:
        result = np.array(mask, dtype=bool, copy=True)
        for _ in range(cells):
            expanded = result.copy()
            expanded[1:] |= result[:-1]
            expanded[:-1] |= result[1:]
            expanded[:, 1:] |= result[:, :-1]
            expanded[:, :-1] |= result[:, 1:]
            result = expanded
        return result
