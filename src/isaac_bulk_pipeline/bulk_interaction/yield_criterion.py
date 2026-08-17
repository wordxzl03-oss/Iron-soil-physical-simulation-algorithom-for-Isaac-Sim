"""One free-surface and cohesive start/stop yield definition for V3.

The Mohr-Coulomb stress balance is literature-informed.  Applying it to a
single height-field layer and using the configured start/stop angles as
effective friction angles is a reduced-order closure, not a calibrated iron-
ore constitutive model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain.terrain_grid import TerrainGrid


YIELD_MODEL_CLASSIFICATION = (
    "LITERATURE_INFORMED_REDUCED_ORDER__NOT_YET_PHYSICALLY_CALIBRATED"
)


def physics_free_surface(
    H_resting_m: np.ndarray, mobile_height_m: np.ndarray
) -> np.ndarray:
    """Return the unique V3 physics surface ``H_resting + h_mobile``."""

    resting = np.asarray(H_resting_m, dtype=np.float64)
    mobile = np.asarray(mobile_height_m, dtype=np.float64)
    if resting.shape != mobile.shape or resting.ndim != 2:
        raise ValueError("[PhysicsFreeSurface] resting/mobile shapes must match H[y,x]")
    if not np.all(np.isfinite(resting)) or not np.all(np.isfinite(mobile)):
        raise ValueError("[PhysicsFreeSurface] surface fields must be finite")
    return resting + mobile


@dataclass(frozen=True)
class CohesiveYieldState:
    H_free_m: np.ndarray
    slope_rad: np.ndarray
    tau_drive_pa: np.ndarray
    tau_resist_start_pa: np.ndarray
    tau_resist_stop_pa: np.ndarray
    yield_start_margin_pa: np.ndarray
    yield_stop_margin_pa: np.ndarray
    start_mask: np.ndarray
    continue_mask: np.ndarray
    yielded_area_m2: float
    classification: str = YIELD_MODEL_CLASSIFICATION


def evaluate_cohesive_yield(
    H_resting_m: np.ndarray,
    mobile_height_m: np.ndarray,
    material: MaterialScenario,
    grid: TerrainGrid,
    integrator: TerrainVolumeIntegrator,
    *,
    layer_depth_m: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    gravity_m_s2: float = 9.81,
) -> CohesiveYieldState:
    """Evaluate cohesive start/stop hysteresis on the unique free surface.

    ``layer_depth_m`` is the depth of material eligible to yield.  For Mobile
    flow this is ``h_mobile``; a Resting avalanche detector supplies its
    configured reduced-order failure-layer depth.
    """

    free = physics_free_surface(H_resting_m, mobile_height_m)
    depth = np.asarray(
        mobile_height_m if layer_depth_m is None else layer_depth_m,
        dtype=np.float64,
    )
    if depth.shape != grid.shape or integrator.shape != grid.shape:
        raise ValueError("[CohesiveYield] grid/state/integrator shape mismatch")
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
        raise ValueError("[CohesiveYield] yielding depth must be finite/non-negative")
    gy, gx = np.gradient(free, grid.dy, grid.dx, edge_order=1)
    slope = np.arctan(np.hypot(gx, gy))
    rho = float(material.assumed_bulk_density_kg_m3)
    drive = rho * gravity_m_s2 * depth * np.sin(slope)
    cohesion = float(material.cohesion_proxy_pa)
    start_phi = np.deg2rad(material.start_angle_deg)
    stop_phi = np.deg2rad(material.stop_angle_deg)
    normal = rho * gravity_m_s2 * depth * np.cos(slope)
    resist_start = cohesion + normal * np.tan(start_phi)
    resist_stop = cohesion + normal * np.tan(stop_phi)
    start_margin = drive - resist_start
    stop_margin = drive - resist_stop
    valid = (depth > 0.0) & (integrator.vertex_weights_m2 > 0.0)
    if grid.valid_mask is not None:
        valid &= np.asarray(grid.valid_mask, dtype=bool)
    start = valid & (start_margin > 0.0)
    moving = np.zeros(grid.shape, dtype=bool) if moving_mask is None else np.asarray(moving_mask, dtype=bool)
    if moving.shape != grid.shape:
        raise ValueError("[CohesiveYield] moving_mask shape mismatch")
    continuing = start | (valid & moving & (stop_margin > 0.0))
    return CohesiveYieldState(
        H_free_m=free,
        slope_rad=slope,
        tau_drive_pa=drive,
        tau_resist_start_pa=resist_start,
        tau_resist_stop_pa=resist_stop,
        yield_start_margin_pa=start_margin,
        yield_stop_margin_pa=stop_margin,
        start_mask=start,
        continue_mask=continuing,
        yielded_area_m2=float(np.sum(integrator.vertex_weights_m2[start], dtype=np.float64)),
    )
