"""Unified mobile-to-resting deposition with exact vertex-volume transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain.terrain_grid import TerrainGrid
from .yield_criterion import evaluate_cohesive_yield


def _ro(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DepositionConfig:
    settling_rate_m_s: float = 0.30
    speed_threshold_m_s: float = 0.18
    require_below_stop_angle: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.settling_rate_m_s) or self.settling_rate_m_s <= 0.0:
            raise ValueError("[Deposition] settling_rate_m_s must be finite/positive")
        if not np.isfinite(self.speed_threshold_m_s) or self.speed_threshold_m_s < 0.0:
            raise ValueError("[Deposition] speed threshold must be finite/non-negative")


@dataclass(frozen=True)
class DepositionResult:
    H_resting_m: np.ndarray
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    deposited_height_m: np.ndarray
    deposited_volume_m3: float
    eligible_mask: np.ndarray

    def __post_init__(self) -> None:
        resting = np.asarray(self.H_resting_m, dtype=np.float64)
        mobile = np.asarray(self.mobile_height_m, dtype=np.float64)
        momentum = np.asarray(self.mobile_momentum_m2_s, dtype=np.float64)
        deposited = np.asarray(self.deposited_height_m, dtype=np.float64)
        mask = np.asarray(self.eligible_mask, dtype=bool)
        if resting.ndim != 2 or mobile.shape != resting.shape or deposited.shape != resting.shape or mask.shape != resting.shape:
            raise ValueError("[Deposition] field shapes invalid")
        if momentum.shape != resting.shape + (2,):
            raise ValueError("[Deposition] momentum shape invalid")
        if not np.all(np.isfinite(resting)) or not np.all(np.isfinite(mobile)) or not np.all(np.isfinite(momentum)) or not np.all(np.isfinite(deposited)):
            raise ValueError("[Deposition] output contains NaN/Inf")
        if np.any(resting < 0.0) or np.any(mobile < -1e-12) or np.any(deposited < 0.0):
            raise ValueError("[Deposition] output contains negative thickness")
        if not np.isfinite(self.deposited_volume_m3) or self.deposited_volume_m3 < 0.0:
            raise ValueError("[Deposition] deposited volume invalid")
        object.__setattr__(self, "H_resting_m", _ro(resting))
        object.__setattr__(self, "mobile_height_m", _ro(np.maximum(mobile, 0.0)))
        object.__setattr__(self, "mobile_momentum_m2_s", _ro(momentum))
        object.__setattr__(self, "deposited_height_m", _ro(deposited))
        frozen_mask = np.ascontiguousarray(mask.copy())
        frozen_mask.setflags(write=False)
        object.__setattr__(self, "eligible_mask", frozen_mask)


class DepositionOperator:
    def __init__(self, config: DepositionConfig | None = None) -> None:
        self.config = config or DepositionConfig()

    def apply(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        active_tool_forcing_mask: np.ndarray | None = None,
    ) -> DepositionResult:
        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64).copy()
        mobile = np.asarray(mobile_height_m, dtype=np.float64).copy()
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64).copy()
        if mobile.shape != grid.shape or momentum.shape != grid.shape + (2,):
            raise ValueError("[Deposition] state/grid shape mismatch")
        if not np.all(np.isfinite(mobile)) or np.any(mobile < 0.0) or not np.all(np.isfinite(momentum)):
            raise ValueError("[Deposition] input state invalid")
        if integrator.shape != grid.shape:
            raise ValueError("[Deposition] integrator/grid shape mismatch")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[Deposition] dt_s must be finite/positive")
        forcing = (
            np.zeros(grid.shape, dtype=bool)
            if active_tool_forcing_mask is None
            else np.asarray(active_tool_forcing_mask, dtype=bool)
        )
        if forcing.shape != grid.shape:
            raise ValueError("[Deposition] forcing mask shape mismatch")
        velocity = np.divide(
            momentum,
            mobile[..., None],
            out=np.zeros_like(momentum),
            where=mobile[..., None] > 1e-12,
        )
        speed = np.linalg.norm(velocity, axis=-1)
        # Deposition only changes the Resting/Mobile label; it leaves the
        # physical free surface unchanged.  Stability must therefore be
        # evaluated on that free surface, not on the buried Resting substrate.
        # Using the substrate creates a deadlock when a static mobile blanket
        # has already smoothed a formerly steep face.
        yield_state = evaluate_cohesive_yield(
            resting,
            mobile,
            material,
            grid,
            integrator,
            layer_depth_m=mobile,
            moving_mask=(mobile > 1e-12) & (speed > self.config.speed_threshold_m_s),
        )
        eligible = (
            (mobile > 0.0)
            & (speed <= self.config.speed_threshold_m_s)
            & ~forcing
        )
        before_mobile_volume = integrator.integrate(mobile)
        # No resolution-dependent "tail settling" is permitted here.  Even a
        # very small Mobile reservoir must satisfy the same constitutive
        # speed/Y_stop criteria as a larger one; otherwise grid resolution
        # silently changes material physics and can erase real moving mass.
        if self.config.require_below_stop_angle:
            eligible &= ~yield_state.continue_mask
        deposited = np.where(
            eligible,
            np.minimum(mobile, self.config.settling_rate_m_s * dt),
            0.0,
        )
        deposited_volume = integrator.integrate(deposited)
        old_mobile = mobile.copy()
        resting += deposited
        mobile -= deposited
        fraction = np.divide(
            mobile,
            old_mobile,
            out=np.zeros_like(mobile),
            where=old_mobile > 0.0,
        )
        momentum *= fraction[..., None]
        momentum[mobile <= 1e-12] = 0.0
        after_mobile_volume = integrator.integrate(mobile)
        if abs(before_mobile_volume - after_mobile_volume - deposited_volume) > max(
            1e-11, 1e-10 * max(before_mobile_volume, 1.0)
        ):
            raise RuntimeError("[Deposition] exact transfer balance failed")
        return DepositionResult(
            H_resting_m=resting,
            mobile_height_m=mobile,
            mobile_momentum_m2_s=momentum,
            deposited_height_m=deposited,
            deposited_volume_m3=deposited_volume,
            eligible_mask=eligible,
        )
