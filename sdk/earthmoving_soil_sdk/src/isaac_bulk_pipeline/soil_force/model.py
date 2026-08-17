"""Literature-constrained reduced-order soil force and momentum exchange."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ..bulk_interaction import FailureZone, MobileLayerResult, ToolTerrainIntersection
from ..bulk_state import MaterialScenario
from ..tools import ToolDescriptor, ToolState


_EPS = 1.0e-12


def _ro(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=np.float64).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SoilForceConfig:
    gravity_m_s2: float = 9.81
    surcharge_pa: float = 0.0
    maximum_resultant_force_n: float = 800_000.0
    minimum_direction_speed_m_s: float = 0.02

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.gravity_m_s2,
                self.surcharge_pa,
                self.maximum_resultant_force_n,
                self.minimum_direction_speed_m_s,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("[SoilForce] configuration must be finite/non-negative")
        if self.gravity_m_s2 <= 0.0 or self.maximum_resultant_force_n <= 0.0:
            raise ValueError("[SoilForce] gravity/maximum force must be positive")


@dataclass(frozen=True)
class MobileMomentumBudget:
    """One SI impulse balance over the explicit interaction window.

    ``tool_impulse_on_mobile`` is measured from the actual activation and
    Mobile-Layer velocity updates, not reconstructed from a fitted ``C v^2``
    term.  The paired soil-on-tool impulse is its exact negative.
    """

    momentum_before_terrain_kg_m_s: np.ndarray
    momentum_after_terrain_kg_m_s: np.ndarray
    gravity_pressure_impulse_terrain_ns: np.ndarray
    basal_friction_impulse_terrain_ns: np.ndarray
    numerical_dissipative_impulse_terrain_ns: np.ndarray
    tool_impulse_on_mobile_terrain_ns: np.ndarray
    integration_window_s: float
    classification: str = "CONSERVATION_BASED_ENGINEERING_MODEL"

    def __post_init__(self) -> None:
        for name in (
            "momentum_before_terrain_kg_m_s",
            "momentum_after_terrain_kg_m_s",
            "gravity_pressure_impulse_terrain_ns",
            "basal_friction_impulse_terrain_ns",
            "numerical_dissipative_impulse_terrain_ns",
            "tool_impulse_on_mobile_terrain_ns",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"[SoilForce] {name} must be finite shape (3,)")
            object.__setattr__(self, name, _ro(value))
        window = float(self.integration_window_s)
        if not np.isfinite(window) or window <= 0.0:
            raise ValueError("[SoilForce] momentum integration window must be positive")
        object.__setattr__(self, "integration_window_s", window)

    @classmethod
    def from_mobile_result(
        cls,
        mobile: MobileLayerResult,
        activation_tool_impulse_on_mobile_terrain_ns: np.ndarray,
        dt_s: float,
    ) -> "MobileMomentumBudget":
        activation = np.asarray(
            activation_tool_impulse_on_mobile_terrain_ns, dtype=np.float64
        )
        return cls(
            momentum_before_terrain_kg_m_s=(
                mobile.momentum_before_terrain_kg_m_s - activation
            ),
            momentum_after_terrain_kg_m_s=mobile.momentum_after_terrain_kg_m_s,
            gravity_pressure_impulse_terrain_ns=(
                mobile.gravity_pressure_impulse_terrain_ns
            ),
            basal_friction_impulse_terrain_ns=mobile.basal_friction_impulse_terrain_ns,
            numerical_dissipative_impulse_terrain_ns=(
                mobile.numerical_dissipative_impulse_terrain_ns
            ),
            tool_impulse_on_mobile_terrain_ns=(
                mobile.tool_impulse_on_mobile_terrain_ns + activation
            ),
            integration_window_s=dt_s,
        )

    @property
    def mobile_momentum_change_terrain_ns(self) -> np.ndarray:
        return _ro(
            self.momentum_after_terrain_kg_m_s
            - self.momentum_before_terrain_kg_m_s
        )

    @property
    def soil_impulse_on_tool_terrain_ns(self) -> np.ndarray:
        return _ro(-self.tool_impulse_on_mobile_terrain_ns)

    @property
    def soil_force_on_tool_terrain_n(self) -> np.ndarray:
        return _ro(self.soil_impulse_on_tool_terrain_ns / self.integration_window_s)

    @property
    def balance_residual_terrain_ns(self) -> np.ndarray:
        accounted = (
            self.gravity_pressure_impulse_terrain_ns
            + self.basal_friction_impulse_terrain_ns
            + self.numerical_dissipative_impulse_terrain_ns
            + self.tool_impulse_on_mobile_terrain_ns
        )
        return _ro(self.mobile_momentum_change_terrain_ns - accounted)

    @property
    def action_reaction_residual_terrain_ns(self) -> np.ndarray:
        return _ro(
            self.tool_impulse_on_mobile_terrain_ns
            + self.soil_impulse_on_tool_terrain_ns
        )


@dataclass(frozen=True)
class StripForceResult:
    strip_index: int
    failure_angle_deg: float
    depth_m: float
    width_m: float
    weight_component_n: float
    cohesion_component_n: float
    surcharge_component_n: float
    inertial_component_n: float
    cutting_resistance_n: float
    normal_resistance_n: float
    force_terrain_n: np.ndarray
    application_point_terrain_m: np.ndarray

    def __post_init__(self) -> None:
        scalars = np.asarray(
            [
                self.failure_angle_deg,
                self.depth_m,
                self.width_m,
                self.weight_component_n,
                self.cohesion_component_n,
                self.surcharge_component_n,
                self.inertial_component_n,
                self.cutting_resistance_n,
                self.normal_resistance_n,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)) or np.any(scalars[1:] < 0.0):
            raise ValueError("[SoilForce] strip scalar diagnostics are invalid")
        force = np.asarray(self.force_terrain_n, dtype=np.float64)
        point = np.asarray(self.application_point_terrain_m, dtype=np.float64)
        if force.shape != (3,) or point.shape != (3,) or not np.all(
            np.isfinite(np.r_[force, point])
        ):
            raise ValueError("[SoilForce] strip force/point must be finite shape (3,)")
        object.__setattr__(self, "force_terrain_n", _ro(force))
        object.__setattr__(
            self, "application_point_terrain_m", _ro(point)
        )


@dataclass(frozen=True)
class SoilForceResult:
    force_terrain_n: np.ndarray
    application_point_terrain_m: np.ndarray
    torque_about_tool_origin_terrain_nm: np.ndarray
    residual_couple_terrain_nm: np.ndarray
    quasi_static_force_terrain_n: np.ndarray
    active_momentum_force_terrain_n: np.ndarray
    cutting_tangent_terrain: np.ndarray
    tool_surface_normal_terrain: np.ndarray
    cutting_edge_lateral_terrain: np.ndarray
    cutting_resistance_n: float
    normal_resistance_n: float
    quasi_static_resultant_force_n: float
    active_momentum_resultant_force_n: float
    resultant_force_n: float
    unclipped_resultant_force_n: float
    force_was_limited: bool
    strip_results: tuple[StripForceResult, ...]
    momentum_budget: MobileMomentumBudget | None = None
    shared_failure_zone_centroid: bool = False
    application_point_model: str = "CUTTING_EDGE_STRIP_RESULTANT"
    term_classifications: Mapping[str, str] = field(
        default_factory=lambda: {
            "quasi_static_failure": "LITERATURE_BASED_REDUCED_ORDER",
            "active_momentum": "CONSERVATION_BASED_ENGINEERING_MODEL",
            "contact_resultant": "ENGINEERING_APPROXIMATION",
        },
        compare=False,
    )

    def __post_init__(self) -> None:
        for name in (
            "force_terrain_n",
            "application_point_terrain_m",
            "torque_about_tool_origin_terrain_nm",
            "residual_couple_terrain_nm",
            "quasi_static_force_terrain_n",
            "active_momentum_force_terrain_n",
            "cutting_tangent_terrain",
            "tool_surface_normal_terrain",
            "cutting_edge_lateral_terrain",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"[SoilForce] {name} must be finite shape (3,)")
            object.__setattr__(self, name, _ro(value))
        for name in (
            "cutting_resistance_n",
            "normal_resistance_n",
            "quasi_static_resultant_force_n",
            "active_momentum_resultant_force_n",
            "resultant_force_n",
            "unclipped_resultant_force_n",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"[SoilForce] {name} invalid")
        object.__setattr__(self, "strip_results", tuple(self.strip_results))
        object.__setattr__(
            self, "term_classifications", MappingProxyType(dict(self.term_classifications))
        )


class SoilForceModel:
    """Luengo/FEE strip resistance plus measured active-soil momentum exchange.

    Static strip magnitudes are the exact terms already solved on the shared
    ``FailureStripGeometry``.  Equation-level provenance and deviations are in
    ``docs/SOIL_FORCE_THEORY_TRACE.md``.  This is not a field calibration.
    """

    def __init__(self, config: SoilForceConfig | None = None) -> None:
        self.config = config or SoilForceConfig()

    def compute(
        self,
        failure_zone: FailureZone,
        intersection: ToolTerrainIntersection,
        material: MaterialScenario,
        descriptor: ToolDescriptor,
        tool_state: ToolState,
        momentum_budget: MobileMomentumBudget | None = None,
    ) -> SoilForceResult:
        self._validate_shared_geometry(failure_zone, material)
        tangent, surface_normal, lateral = self._contact_basis(
            failure_zone, intersection, descriptor, tool_state
        )
        origin = np.asarray(tool_state.pose_terrain[:3, 3], dtype=np.float64)
        geometry = descriptor.bucket_geometry
        if geometry is None:
            cutting_center = np.mean(tool_state.cutting_edge_terrain, axis=0)
        else:
            cutting_center = np.mean(tool_state.cutting_edge_terrain, axis=0)

        strips: list[StripForceResult] = []
        static_force = np.zeros(3, dtype=np.float64)
        static_torque = np.zeros(3, dtype=np.float64)
        total_cut = 0.0
        total_normal = 0.0
        weighted_point = np.zeros(3, dtype=np.float64)
        point_weight = 0.0
        for strip in failure_zone.strip_geometries:
            ratio = strip.activation_ratio
            weight = strip.weight_resistance_n * ratio
            cohesion = strip.cohesion_resistance_n * ratio
            surcharge = strip.surcharge_resistance_n * ratio
            # FailureStripGeometry stores the Luengo Eq. 5 resultant magnitude.
            # Eq. 2 resolves that resultant with rake+tool-friction angle.
            resultant = weight + cohesion + surcharge
            angle = np.deg2rad(
                strip.rake_angle_deg + strip.soil_tool_friction_angle_deg
            )
            tangential = resultant * abs(float(np.sin(angle)))
            signed_normal = resultant * float(np.cos(angle))
            strip_force = -tangential * tangent + signed_normal * surface_normal
            point = np.asarray(
                strip.effective_cutting_edge_terrain_m, dtype=np.float64
            )
            static_force += strip_force
            static_torque += np.cross(point - origin, strip_force)
            magnitude = float(np.linalg.norm(strip_force))
            weighted_point += magnitude * point
            point_weight += magnitude
            total_cut += tangential
            total_normal += abs(signed_normal)
            strips.append(
                StripForceResult(
                    strip_index=strip.strip_index,
                    failure_angle_deg=strip.failure_angle_deg,
                    depth_m=strip.penetration_depth_m,
                    width_m=strip.width_m,
                    weight_component_n=float(weight),
                    cohesion_component_n=float(cohesion),
                    surcharge_component_n=float(surcharge),
                    inertial_component_n=0.0,
                    cutting_resistance_n=float(tangential),
                    normal_resistance_n=float(abs(signed_normal)),
                    force_terrain_n=strip_force,
                    application_point_terrain_m=point,
                )
            )

        dynamic_force = np.array(
            np.zeros(3, dtype=np.float64)
            if momentum_budget is None
            else momentum_budget.soil_force_on_tool_terrain_n,
            dtype=np.float64,
            copy=True,
        )
        dynamic_torque = np.cross(cutting_center - origin, dynamic_force)
        if np.linalg.norm(dynamic_force) > 0.0:
            dynamic_weight = float(np.linalg.norm(dynamic_force))
            weighted_point += dynamic_weight * cutting_center
            point_weight += dynamic_weight

        force = static_force + dynamic_force
        torque = static_torque + dynamic_torque
        unclipped = float(np.linalg.norm(force))
        limited = unclipped > self.config.maximum_resultant_force_n
        scale = 1.0
        if limited:
            scale = self.config.maximum_resultant_force_n / unclipped
            force *= scale
            torque *= scale
            static_force *= scale
            dynamic_force *= scale
            total_cut *= scale
            total_normal *= scale
            strips = [
                replace(
                    strip,
                    weight_component_n=strip.weight_component_n * scale,
                    cohesion_component_n=strip.cohesion_component_n * scale,
                    surcharge_component_n=strip.surcharge_component_n * scale,
                    inertial_component_n=strip.inertial_component_n * scale,
                    cutting_resistance_n=strip.cutting_resistance_n * scale,
                    normal_resistance_n=strip.normal_resistance_n * scale,
                    force_terrain_n=strip.force_terrain_n * scale,
                )
                for strip in strips
            ]

        point = (
            cutting_center
            if point_weight <= _EPS
            else weighted_point / point_weight
        )
        residual_couple = torque - np.cross(point - origin, force)
        return SoilForceResult(
            force_terrain_n=force,
            application_point_terrain_m=point,
            torque_about_tool_origin_terrain_nm=torque,
            residual_couple_terrain_nm=residual_couple,
            quasi_static_force_terrain_n=static_force,
            active_momentum_force_terrain_n=dynamic_force,
            cutting_tangent_terrain=tangent,
            tool_surface_normal_terrain=surface_normal,
            cutting_edge_lateral_terrain=lateral,
            cutting_resistance_n=float(total_cut),
            normal_resistance_n=float(total_normal),
            quasi_static_resultant_force_n=float(np.linalg.norm(static_force)),
            active_momentum_resultant_force_n=float(np.linalg.norm(dynamic_force)),
            resultant_force_n=float(np.linalg.norm(force)),
            unclipped_resultant_force_n=unclipped,
            force_was_limited=limited,
            strip_results=tuple(strips),
            momentum_budget=momentum_budget,
        )

    def _contact_basis(
        self,
        failure_zone: FailureZone,
        intersection: ToolTerrainIntersection,
        descriptor: ToolDescriptor,
        tool_state: ToolState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        velocity = np.asarray(
            intersection.cutting_edge_velocity_terrain_m_s, dtype=np.float64
        )
        terrain_normal = np.asarray(intersection.local_terrain_normal, dtype=np.float64)
        projected = velocity - np.dot(velocity, terrain_normal) * terrain_normal
        if np.linalg.norm(projected) > self.config.minimum_direction_speed_m_s:
            tangent = projected / np.linalg.norm(projected)
        else:
            horizontal = np.r_[failure_zone.approach_direction_xy, 0.0]
            projected = horizontal - np.dot(horizontal, terrain_normal) * terrain_normal
            tangent = projected / max(float(np.linalg.norm(projected)), _EPS)

        edge = (
            np.asarray(tool_state.cutting_edge_terrain[-1])
            - np.asarray(tool_state.cutting_edge_terrain[0])
        )
        lateral = edge / max(float(np.linalg.norm(edge)), _EPS)
        geometry = descriptor.bucket_geometry
        if geometry is None:
            # ToolState arrays are intentionally read-only.  The legacy
            # no-geometry path still needs a mutable vector for normalization.
            normal = np.array(
                tool_state.pose_terrain[:3, 2], dtype=np.float64, copy=True
            )
        else:
            normal = (
                np.asarray(tool_state.pose_terrain[:3, :3], dtype=np.float64)
                @ geometry.bottom_plate_normal_local
            )
        normal /= max(float(np.linalg.norm(normal)), _EPS)
        if np.dot(normal, terrain_normal) < 0.0:
            normal = -normal
        # Gram-Schmidt removes tiny CAD/frame skew while retaining the real
        # cutting-edge and bottom-surface directions.
        lateral -= np.dot(lateral, normal) * normal
        lateral /= max(float(np.linalg.norm(lateral)), _EPS)
        tangent -= np.dot(tangent, lateral) * lateral
        tangent -= np.dot(tangent, normal) * normal
        tangent /= max(float(np.linalg.norm(tangent)), _EPS)
        return tangent, normal, lateral

    def _validate_shared_geometry(
        self,
        failure_zone: FailureZone,
        material: MaterialScenario,
    ) -> None:
        """Reject parameter drift instead of constructing a second wedge."""

        for strip in failure_zone.strip_geometries:
            expected = np.asarray(
                [
                    material.internal_friction_angle_deg,
                    np.rad2deg(np.arctan(material.tool_friction_coefficient)),
                    material.cohesion_proxy_pa,
                    material.assumed_bulk_density_kg_m3,
                    self.config.surcharge_pa,
                    self.config.gravity_m_s2,
                ],
                dtype=np.float64,
            )
            actual = np.asarray(
                [
                    strip.internal_friction_angle_deg,
                    strip.soil_tool_friction_angle_deg,
                    strip.cohesion_pa,
                    strip.assumed_bulk_density_kg_m3,
                    strip.surcharge_pa,
                    strip.gravity_m_s2,
                ],
                dtype=np.float64,
            )
            if not np.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10):
                raise ValueError(
                    "[SoilForce] material/surcharge/gravity differ from the shared "
                    "FailureStripGeometry; re-solve with those parameters"
                )
