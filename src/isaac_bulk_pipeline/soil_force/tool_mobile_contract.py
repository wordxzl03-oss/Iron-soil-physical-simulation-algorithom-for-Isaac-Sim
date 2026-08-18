"""Narrow conservation contract for a future tool--Mobile contact model.

This module deliberately does *not* define a contact constitutive law.  It
only makes the units and action--reaction bookkeeping testable once a contact
operator has produced an accepted physical impulse.  Production terrain code
does not import this module, so adding the contract cannot create a hidden
force or alter the accepted Mobile V2 transport equations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _vector2(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"[ToolMobileContract] {name} must be finite shape (2,)")
    return vector


def external_acceleration_impulse_ns(
    mobile_depth_m: np.ndarray,
    dual_control_area_m2: np.ndarray,
    external_acceleration_xy_m_s2: np.ndarray,
    *,
    bulk_density_kg_m3: float,
    dt_s: float,
) -> np.ndarray:
    """Return the physical impulse implied by the Mobile V2 source semantics.

    Mobile V2 stores ``q = h*u`` and applies ``u += a_external*dt`` after
    transport and before basal friction.  Consequently the source-only
    momentum increment is ``rho * sum(A*h*a*dt)``.  ``A`` is the authoritative
    vertex dual-control area, not a uniform ``dx*dy`` approximation.
    """

    depth = np.asarray(mobile_depth_m, dtype=np.float64)
    area = np.asarray(dual_control_area_m2, dtype=np.float64)
    acceleration = np.asarray(external_acceleration_xy_m_s2, dtype=np.float64)
    if depth.shape != area.shape:
        raise ValueError("[ToolMobileContract] depth/area shape mismatch")
    if acceleration.shape != depth.shape + (2,):
        raise ValueError("[ToolMobileContract] acceleration must have depth.shape + (2,)")
    if not (
        np.all(np.isfinite(depth))
        and np.all(np.isfinite(area))
        and np.all(np.isfinite(acceleration))
    ):
        raise ValueError("[ToolMobileContract] source arrays must be finite")
    density = float(bulk_density_kg_m3)
    dt = float(dt_s)
    if density <= 0.0 or dt <= 0.0 or not np.isfinite(density + dt):
        raise ValueError("[ToolMobileContract] density and dt must be finite/positive")
    if np.any(depth < 0.0) or np.any(area < 0.0):
        raise ValueError("[ToolMobileContract] depth and area must be non-negative")
    return density * np.sum(
        area[..., None] * depth[..., None] * acceleration * dt,
        axis=tuple(range(depth.ndim)),
    )


@dataclass(frozen=True)
class ToolMobileSubstepContract:
    """One accepted contact impulse and its independently measured effect."""

    accepted_tool_to_mobile_impulse_xy_ns: np.ndarray
    measured_mobile_source_delta_xy_ns: np.ndarray
    contact_active: bool
    dt_sub_s: float

    def __post_init__(self) -> None:
        accepted = _vector2(
            self.accepted_tool_to_mobile_impulse_xy_ns,
            "accepted_tool_to_mobile_impulse_xy_ns",
        )
        measured = _vector2(
            self.measured_mobile_source_delta_xy_ns,
            "measured_mobile_source_delta_xy_ns",
        )
        dt = float(self.dt_sub_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[ToolMobileContract] dt_sub_s must be finite/positive")
        if not self.contact_active and np.any(accepted != 0.0):
            raise ValueError("[ToolMobileContract] nonzero accepted impulse without contact")
        object.__setattr__(self, "accepted_tool_to_mobile_impulse_xy_ns", accepted)
        object.__setattr__(self, "measured_mobile_source_delta_xy_ns", measured)
        object.__setattr__(self, "dt_sub_s", dt)

    @property
    def machine_reaction_impulse_xy_ns(self) -> np.ndarray:
        """The only admissible reaction command: exact negative of accepted J."""

        return -self.accepted_tool_to_mobile_impulse_xy_ns

    @property
    def source_residual_xy_ns(self) -> np.ndarray:
        return (
            self.measured_mobile_source_delta_xy_ns
            - self.accepted_tool_to_mobile_impulse_xy_ns
        )

    @property
    def action_reaction_residual_xy_ns(self) -> np.ndarray:
        return (
            self.accepted_tool_to_mobile_impulse_xy_ns
            + self.machine_reaction_impulse_xy_ns
        )


@dataclass
class ToolMobileFrameContractLedger:
    """Accumulate Mobile CFL substeps once into one machine-frame impulse.

    This is an audit/interface object, not a force applicator.  In particular,
    it cannot turn a requested or synthetic impulse into production physics.
    """

    substeps: list[ToolMobileSubstepContract] = field(default_factory=list)

    def append(self, record: ToolMobileSubstepContract) -> None:
        self.substeps.append(record)

    @property
    def accepted_tool_to_mobile_impulse_xy_ns(self) -> np.ndarray:
        return sum(
            (record.accepted_tool_to_mobile_impulse_xy_ns for record in self.substeps),
            start=np.zeros(2, dtype=np.float64),
        )

    @property
    def measured_mobile_source_delta_xy_ns(self) -> np.ndarray:
        return sum(
            (record.measured_mobile_source_delta_xy_ns for record in self.substeps),
            start=np.zeros(2, dtype=np.float64),
        )

    @property
    def machine_reaction_impulse_xy_ns(self) -> np.ndarray:
        return sum(
            (record.machine_reaction_impulse_xy_ns for record in self.substeps),
            start=np.zeros(2, dtype=np.float64),
        )

    @property
    def action_reaction_residual_xy_ns(self) -> np.ndarray:
        return (
            self.accepted_tool_to_mobile_impulse_xy_ns
            + self.machine_reaction_impulse_xy_ns
        )

