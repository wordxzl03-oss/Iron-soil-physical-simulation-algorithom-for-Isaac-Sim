"""Physical-state transition between static terrain and sustained avalanches.

MiniSlope is a quasi-static residual relaxation operator.  This module owns the
distinct transition in which a connected, persistent static failure becomes a
finite-time Mobile Layer event.  It deliberately has no iteration-count input:
activation is determined from slope hysteresis, connected extent, mobilizable
volume, persistence in physical time, and the current mobile state.

The default parameters are reduced-order literature-style assumptions, not
measurements for a particular iron ore.  Their provenance is part of the public
configuration and sensitivity variants are provided so callers cannot silently
treat one arbitrary parameter set as calibrated truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from time import perf_counter

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain.terrain_grid import TerrainGrid
from .yield_criterion import evaluate_cohesive_yield, physics_free_surface


LITERATURE_REDUCED_ORDER_UNCALIBRATED = (
    "LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED"
)


def _readonly(value: np.ndarray, *, dtype=np.float64) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LargeAvalancheTransitionConfig:
    """Explicit assumptions for Resting-to-Mobile failure activation.

    ``mobilization_depth_m`` is the reduced-order failure-layer thickness.
    ``gravity_velocity_length_m`` is the characteristic distance over which
    gravity accelerates that layer at activation.  ``velocity_efficiency``
    accounts for unresolved internal deformation.  All three require material-
    and site-specific calibration before predictive use.
    """

    minimum_connected_cells: int = 32
    minimum_connected_area_m2: float = 0.08
    minimum_mobilizable_volume_m3: float = 0.004
    minimum_mean_excess_start_deg: float = 0.0
    persistence_time_s: float = 0.12
    mobilization_depth_m: float = 0.08
    gravity_velocity_length_m: float = 0.25
    velocity_efficiency: float = 0.45
    mobile_activity_speed_m_s: float = 0.05
    mobile_activity_volume_m3: float = 1.0e-5
    dry_tolerance_m: float = 1.0e-8
    settled_mobile_volume_m3: float = 1.0e-5
    settled_speed_m_s: float = 0.05
    connectivity: int = 8
    parameter_basis: str = LITERATURE_REDUCED_ORDER_UNCALIBRATED
    sensitivity_case: str = "nominal_uncalibrated"

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_connected_cells, int) or self.minimum_connected_cells < 1:
            raise ValueError("[LargeAvalanche] minimum_connected_cells must be >= 1")
        if self.connectivity not in (4, 8):
            raise ValueError("[LargeAvalanche] connectivity must be 4 or 8")
        numeric = np.asarray(
            [
                self.minimum_connected_area_m2,
                self.minimum_mobilizable_volume_m3,
                self.minimum_mean_excess_start_deg,
                self.persistence_time_s,
                self.mobilization_depth_m,
                self.gravity_velocity_length_m,
                self.velocity_efficiency,
                self.mobile_activity_speed_m_s,
                self.mobile_activity_volume_m3,
                self.dry_tolerance_m,
                self.settled_mobile_volume_m3,
                self.settled_speed_m_s,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            raise ValueError("[LargeAvalanche] numeric configuration must be finite/non-negative")
        if self.persistence_time_s <= 0.0:
            raise ValueError("[LargeAvalanche] persistence_time_s must be positive")
        if self.mobilization_depth_m <= 0.0 or self.gravity_velocity_length_m <= 0.0:
            raise ValueError("[LargeAvalanche] failure depth/velocity length must be positive")
        if not 0.0 < self.velocity_efficiency <= 1.0:
            raise ValueError("[LargeAvalanche] velocity_efficiency must lie in (0, 1]")
        if self.parameter_basis != LITERATURE_REDUCED_ORDER_UNCALIBRATED:
            raise ValueError(
                "[LargeAvalanche] transition parameters must remain explicitly "
                "literature-based reduced-order / uncalibrated"
            )
        if not self.sensitivity_case:
            raise ValueError("[LargeAvalanche] sensitivity_case must be non-empty")

    def sensitivity_ensemble(
        self, relative_fraction: float = 0.25
    ) -> tuple["LargeAvalancheTransitionConfig", ...]:
        """Return explicit low/nominal/high transition sensitivity cases."""

        fraction = float(relative_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("[LargeAvalanche] sensitivity fraction must lie in (0, 1)")
        return (
            replace(
                self,
                persistence_time_s=self.persistence_time_s * (1.0 - fraction),
                minimum_connected_area_m2=self.minimum_connected_area_m2 * (1.0 - fraction),
                minimum_mobilizable_volume_m3=self.minimum_mobilizable_volume_m3 * (1.0 - fraction),
                mobilization_depth_m=self.mobilization_depth_m * (1.0 + fraction),
                velocity_efficiency=min(1.0, self.velocity_efficiency * (1.0 + fraction)),
                sensitivity_case="more_mobile_uncalibrated",
            ),
            replace(self, sensitivity_case="nominal_uncalibrated"),
            replace(
                self,
                persistence_time_s=self.persistence_time_s * (1.0 + fraction),
                minimum_connected_area_m2=self.minimum_connected_area_m2 * (1.0 + fraction),
                minimum_mobilizable_volume_m3=self.minimum_mobilizable_volume_m3 * (1.0 + fraction),
                mobilization_depth_m=self.mobilization_depth_m * (1.0 - fraction),
                velocity_efficiency=self.velocity_efficiency * (1.0 - fraction),
                sensitivity_case="less_mobile_uncalibrated",
            ),
        )

    @classmethod
    def from_mapping(cls, value: dict[str, object]) -> "LargeAvalancheTransitionConfig":
        """Build from a runtime/YAML mapping while rejecting silent typos."""

        if not isinstance(value, dict):
            raise TypeError("[LargeAvalanche] configuration must be a mapping")
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise ValueError(f"[LargeAvalanche] unknown configuration fields: {unknown}")
        return cls(**value)


@dataclass(frozen=True)
class AvalanchePhysicalDiagnostics:
    classification: str
    start_angle_deg: float
    stop_angle_deg: float
    unstable_cell_count: int
    connected_region_count: int
    largest_connected_cell_count: int
    largest_connected_area_m2: float
    largest_connected_mobilizable_volume_m3: float
    mean_excess_start_deg: float
    maximum_excess_start_deg: float
    mean_excess_stop_deg: float
    maximum_excess_stop_deg: float
    persistence_s: float
    persistence_required_s: float
    current_mobile_volume_m3: float
    moving_mobile_volume_m3: float
    maximum_mobile_speed_m_s: float
    current_mobile_active: bool
    connected_extent_pass: bool
    unstable_volume_pass: bool
    excess_slope_pass: bool
    persistence_pass: bool
    transition_ready: bool
    parameter_basis: str
    sensitivity_case: str
    largest_connected_mask: np.ndarray
    slope_angle_deg: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "largest_connected_mask", _readonly(self.largest_connected_mask, dtype=bool)
        )
        object.__setattr__(self, "slope_angle_deg", _readonly(self.slope_angle_deg))


@dataclass(frozen=True)
class RestingToMobileTransitionResult:
    H_resting_m: np.ndarray
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    transferred_height_m: np.ndarray
    initial_velocity_xy_m_s: np.ndarray
    transferred_volume_m3: float
    momentum_before_kg_m_s: np.ndarray
    momentum_after_kg_m_s: np.ndarray
    gravity_initiation_impulse_kg_m_s: np.ndarray
    momentum_closure_error_kg_m_s: np.ndarray
    transitioned: bool
    diagnostics: AvalanchePhysicalDiagnostics

    def __post_init__(self) -> None:
        resting = np.asarray(self.H_resting_m, dtype=np.float64)
        mobile = np.asarray(self.mobile_height_m, dtype=np.float64)
        momentum = np.asarray(self.mobile_momentum_m2_s, dtype=np.float64)
        transfer = np.asarray(self.transferred_height_m, dtype=np.float64)
        velocity = np.asarray(self.initial_velocity_xy_m_s, dtype=np.float64)
        if resting.ndim != 2 or mobile.shape != resting.shape or transfer.shape != resting.shape:
            raise ValueError("[LargeAvalanche] transition field shapes invalid")
        if momentum.shape != resting.shape + (2,) or velocity.shape != momentum.shape:
            raise ValueError("[LargeAvalanche] transition vector field shapes invalid")
        if not all(np.all(np.isfinite(value)) for value in (resting, mobile, momentum, transfer, velocity)):
            raise ValueError("[LargeAvalanche] transition output contains NaN/Inf")
        if np.any(resting < -1e-12) or np.any(mobile < -1e-12) or np.any(transfer < -1e-12):
            raise ValueError("[LargeAvalanche] transition output contains negative thickness")
        object.__setattr__(self, "H_resting_m", _readonly(np.maximum(resting, 0.0)))
        object.__setattr__(self, "mobile_height_m", _readonly(np.maximum(mobile, 0.0)))
        object.__setattr__(self, "mobile_momentum_m2_s", _readonly(momentum))
        object.__setattr__(self, "transferred_height_m", _readonly(np.maximum(transfer, 0.0)))
        object.__setattr__(self, "initial_velocity_xy_m_s", _readonly(velocity))
        for name in (
            "momentum_before_kg_m_s",
            "momentum_after_kg_m_s",
            "gravity_initiation_impulse_kg_m_s",
            "momentum_closure_error_kg_m_s",
        ):
            vector = np.asarray(getattr(self, name), dtype=np.float64)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"[LargeAvalanche] {name} must be finite shape (2,)")
            object.__setattr__(self, name, _readonly(vector))


@dataclass(frozen=True)
class TerrainSettledDiagnostic:
    settled: bool
    ready_for_final_minislope: bool
    residual_static_relaxation_required: bool
    current_mobile_active: bool
    mobile_volume_m3: float
    moving_mobile_volume_m3: float
    maximum_mobile_speed_m_s: float
    above_start_cell_count: int
    largest_above_start_area_m2: float
    reason: str


class LargeAvalancheTransitionController:
    """Stateful physical-time detector and conservative activation operator."""

    NO_LARGE_EVENT = "NO_LARGE_EVENT"
    LOCAL_STATIC_INSTABILITY = "LOCAL_STATIC_INSTABILITY"
    PERSISTING_LARGE_UNSTABLE_REGION = "PERSISTING_LARGE_UNSTABLE_REGION"
    LARGE_AVALANCHE_MOBILE_PATH = "LARGE_AVALANCHE_MOBILE_PATH"

    def __init__(self, config: LargeAvalancheTransitionConfig | None = None) -> None:
        self.config = config or LargeAvalancheTransitionConfig()
        self._persistence_s = 0.0
        self._previous_connected_mask: np.ndarray | None = None
        self._mobilized_latch_mask: np.ndarray | None = None
        # Profiling is kept as host scalar telemetry.  It does not change the
        # accepted transition semantics or imply that this CPU diagnostic path
        # has already been ported to GPU.
        self._last_profile_ms: dict[str, float] = {
            "gradient": 0.0,
            "connected_components": 0.0,
            "diagnostics": 0.0,
            "transfer": 0.0,
            "total": 0.0,
        }

    @property
    def persistence_s(self) -> float:
        return self._persistence_s

    @property
    def last_profile_ms(self) -> dict[str, float]:
        """Measured host costs of the most recent transition observation."""

        return dict(self._last_profile_ms)

    def reset(self) -> None:
        self._persistence_s = 0.0
        self._previous_connected_mask = None
        self._mobilized_latch_mask = None
        for key in self._last_profile_ms:
            self._last_profile_ms[key] = 0.0

    def observe_and_maybe_mobilize(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        release_settled_latches: bool = True,
    ) -> RestingToMobileTransitionResult:
        total_start = perf_counter()
        self._last_profile_ms["transfer"] = 0.0
        resting, mobile, momentum, dt = self._validate_state(
            H_resting_m,
            mobile_height_m,
            mobile_momentum_m2_s,
            grid,
            integrator,
            dt_s,
        )
        diagnostics, grad_x, grad_y, slope_rad = self._diagnose(
            resting, mobile, momentum, material, grid, integrator,
            update_persistence=True, dt_s=dt,
            release_settled_latches=release_settled_latches,
        )
        transfer = np.zeros_like(resting)
        velocity = np.zeros(resting.shape + (2,), dtype=np.float64)
        before_momentum = self._integrated_momentum(
            momentum, integrator, material.assumed_bulk_density_kg_m3
        )
        if diagnostics.transition_ready:
            transfer_start = perf_counter()
            if self._mobilized_latch_mask is None:
                self._mobilized_latch_mask = np.zeros(grid.shape, dtype=bool)
            eligible = diagnostics.largest_connected_mask & ~self._mobilized_latch_mask
            # Stress-based start/stop hysteresis replaces the former fixed
            # slope trigger.  A cohesive steep face can therefore remain
            # static when its driving stress is below strength.
            failure_depth = np.minimum(resting, self.config.mobilization_depth_m)
            yield_state = evaluate_cohesive_yield(
                resting, mobile, material, grid, integrator,
                layer_depth_m=failure_depth,
            )
            hysteresis_width = np.maximum(
                yield_state.tau_resist_start_pa - yield_state.tau_resist_stop_pa,
                1.0e-12,
            )
            severity = np.clip(
                yield_state.yield_stop_margin_pa / hysteresis_width, 0.0, 1.0
            )
            transfer = np.where(
                eligible,
                np.minimum(resting, self.config.mobilization_depth_m * severity),
                0.0,
            )
            gradient_norm = np.hypot(grad_x, grad_y)
            direction_x = np.divide(-grad_x, gradient_norm, out=np.zeros_like(grad_x), where=gradient_norm > 1e-12)
            direction_y = np.divide(-grad_y, gradient_norm, out=np.zeros_like(grad_y), where=gradient_norm > 1e-12)
            driving_acceleration = 9.81 * np.maximum(
                np.sin(slope_rad)
                - material.mobile_friction_coefficient * np.cos(slope_rad),
                0.0,
            )
            speed = self.config.velocity_efficiency * np.sqrt(
                2.0 * driving_acceleration * self.config.gravity_velocity_length_m
            )
            velocity[..., 0] = direction_x * speed
            velocity[..., 1] = direction_y * speed
            velocity[transfer <= self.config.dry_tolerance_m] = 0.0
            resting = resting - transfer
            mobile = mobile + transfer
            momentum = momentum + transfer[..., None] * velocity
            self._mobilized_latch_mask |= transfer > self.config.dry_tolerance_m
            self._last_profile_ms["transfer"] = (
                perf_counter() - transfer_start
            ) * 1_000.0

        transferred_volume = integrator.integrate(transfer)
        initiation_impulse = self._integrated_momentum(
            transfer[..., None] * velocity,
            integrator,
            material.assumed_bulk_density_kg_m3,
        )
        after_momentum = self._integrated_momentum(
            momentum, integrator, material.assumed_bulk_density_kg_m3
        )
        closure = after_momentum - before_momentum - initiation_impulse
        total_before = integrator.integrate(np.asarray(H_resting_m) + np.asarray(mobile_height_m))
        total_after = integrator.integrate(resting + mobile)
        tolerance = max(1e-11, 1e-10 * max(total_before, 1.0))
        if abs(total_after - total_before) > tolerance:
            raise RuntimeError("[LargeAvalanche] Resting-to-Mobile volume conservation failed")
        if np.linalg.norm(closure) > max(1e-8, 1e-10 * max(np.linalg.norm(after_momentum), 1.0)):
            raise RuntimeError("[LargeAvalanche] initiation momentum accounting failed")

        result = RestingToMobileTransitionResult(
            H_resting_m=resting,
            mobile_height_m=mobile,
            mobile_momentum_m2_s=momentum,
            transferred_height_m=transfer,
            initial_velocity_xy_m_s=velocity,
            transferred_volume_m3=transferred_volume,
            momentum_before_kg_m_s=before_momentum,
            momentum_after_kg_m_s=after_momentum,
            gravity_initiation_impulse_kg_m_s=initiation_impulse,
            momentum_closure_error_kg_m_s=closure,
            transitioned=transferred_volume > 0.0,
            diagnostics=diagnostics,
        )
        self._last_profile_ms["total"] = (perf_counter() - total_start) * 1_000.0
        return result

    def diagnose(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> AvalanchePhysicalDiagnostics:
        resting, mobile, momentum, _ = self._validate_state(
            H_resting_m,
            mobile_height_m,
            mobile_momentum_m2_s,
            grid,
            integrator,
            1.0,
        )
        diagnostics, _, _, _ = self._diagnose(
            resting, mobile, momentum, material, grid, integrator,
            update_persistence=False, dt_s=0.0,
            release_settled_latches=False,
        )
        return diagnostics

    def settled_diagnostic(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> TerrainSettledDiagnostic:
        diagnostics = self.diagnose(
            H_resting_m,
            mobile_height_m,
            mobile_momentum_m2_s,
            material,
            grid,
            integrator,
        )
        mobile_quiet = (
            diagnostics.current_mobile_volume_m3 <= self.config.settled_mobile_volume_m3
            and (
                diagnostics.moving_mobile_volume_m3 <= self.config.mobile_activity_volume_m3
                and diagnostics.maximum_mobile_speed_m_s <= self.config.settled_speed_m_s
            )
        )
        large_static_event = diagnostics.connected_extent_pass and diagnostics.unstable_volume_pass
        residual = diagnostics.unstable_cell_count > 0
        ready = mobile_quiet and not large_static_event
        settled = ready and not residual
        if not mobile_quiet:
            reason = "MOBILE_LAYER_ACTIVE"
        elif large_static_event:
            reason = "LARGE_STATIC_FAILURE_REQUIRES_MOBILE_PATH"
        elif residual:
            reason = "LOCAL_RESIDUAL_READY_FOR_FINAL_MINISLOPE"
        else:
            reason = "TERRAIN_SETTLED"
        return TerrainSettledDiagnostic(
            settled=settled,
            ready_for_final_minislope=ready and residual,
            residual_static_relaxation_required=residual,
            current_mobile_active=diagnostics.current_mobile_active,
            mobile_volume_m3=diagnostics.current_mobile_volume_m3,
            moving_mobile_volume_m3=diagnostics.moving_mobile_volume_m3,
            maximum_mobile_speed_m_s=diagnostics.maximum_mobile_speed_m_s,
            above_start_cell_count=diagnostics.unstable_cell_count,
            largest_above_start_area_m2=diagnostics.largest_connected_area_m2,
            reason=reason,
        )

    def _diagnose(
        self,
        resting: np.ndarray,
        mobile: np.ndarray,
        momentum: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        update_persistence: bool,
        dt_s: float,
        release_settled_latches: bool,
    ) -> tuple[AvalanchePhysicalDiagnostics, np.ndarray, np.ndarray, np.ndarray]:
        diagnostic_start = perf_counter()
        surface = physics_free_surface(resting, mobile)
        gradient_start = perf_counter()
        grad_y, grad_x = np.gradient(surface, grid.dy, grid.dx, edge_order=1)
        slope_rad = np.arctan(np.hypot(grad_x, grad_y))
        slope_deg = np.rad2deg(slope_rad)
        self._last_profile_ms["gradient"] = (perf_counter() - gradient_start) * 1_000.0
        valid = integrator.vertex_weights_m2 > 0.0
        if grid.valid_mask is not None:
            valid &= np.asarray(grid.valid_mask, dtype=bool)
        failure_depth = np.minimum(resting, self.config.mobilization_depth_m)
        velocity = np.divide(
            momentum, mobile[..., None], out=np.zeros_like(momentum),
            where=mobile[..., None] > self.config.dry_tolerance_m,
        )
        speed = np.linalg.norm(velocity, axis=-1)
        moving = (mobile > self.config.dry_tolerance_m) & (
            speed > self.config.mobile_activity_speed_m_s
        )
        yield_state = evaluate_cohesive_yield(
            resting, mobile, material, grid, integrator,
            layer_depth_m=failure_depth, moving_mask=moving,
        )
        unstable = valid & yield_state.start_mask
        components_start = perf_counter()
        largest, region_count, largest_count = _largest_connected_region(
            unstable, self.config.connectivity
        )
        self._last_profile_ms["connected_components"] = (
            perf_counter() - components_start
        ) * 1_000.0
        weights = integrator.vertex_weights_m2
        area = float(np.sum(weights[largest], dtype=np.float64))
        hysteresis_width = np.maximum(
            yield_state.tau_resist_start_pa - yield_state.tau_resist_stop_pa,
            1.0e-12,
        )
        severity = np.clip(yield_state.yield_stop_margin_pa / hysteresis_width, 0.0, 1.0)
        if self._mobilized_latch_mask is not None and release_settled_latches:
            # A latch owns one mobilization tranche, not the complete lifetime
            # of a terrain cell.  Clear it before calculating eligibility so
            # newly exposed, still-yielded Resting enters this persistence
            # observation immediately.
            local_mobile_absent = mobile <= self.config.dry_tolerance_m
            clear = ~moving & (
                local_mobile_absent
                | (yield_state.yield_stop_margin_pa <= 0.0)
            )
            self._mobilized_latch_mask[clear] = False
        # Connected extent describes propagation reach; mobilizable volume is
        # only the new Resting material still eligible to enter Mobile.
        # Re-counting latched cells leaves an event permanently classified as
        # large even when its actual conservative transfer is exactly zero.
        eligible_failure = largest
        if self._mobilized_latch_mask is not None:
            eligible_failure = largest & ~self._mobilized_latch_mask
        mobilizable_height = np.where(
            eligible_failure,
            np.minimum(resting, self.config.mobilization_depth_m * severity),
            0.0,
        )
        mobilizable_volume = integrator.integrate(mobilizable_height)
        excess_start = np.maximum(slope_deg - material.start_angle_deg, 0.0)
        excess_stop = np.maximum(slope_deg - material.stop_angle_deg, 0.0)
        mean_start = float(np.mean(excess_start[largest])) if largest_count else 0.0
        max_start = float(np.max(excess_start[largest], initial=0.0))
        mean_stop = float(np.mean(excess_stop[largest])) if largest_count else 0.0
        max_stop = float(np.max(excess_stop[largest], initial=0.0))
        mobile_volume = integrator.integrate(mobile)
        moving_volume = integrator.integrate(np.where(moving, mobile, 0.0))
        mobile_active = (
            moving_volume >= self.config.mobile_activity_volume_m3
            and float(np.max(speed, initial=0.0)) >= self.config.mobile_activity_speed_m_s
        )
        extent_pass = (
            largest_count >= self.config.minimum_connected_cells
            and area >= self.config.minimum_connected_area_m2
        )
        volume_pass = mobilizable_volume >= self.config.minimum_mobilizable_volume_m3
        excess_pass = mean_start >= self.config.minimum_mean_excess_start_deg
        physical_candidate = extent_pass and volume_pass and excess_pass
        if update_persistence:
            overlap = (
                self._previous_connected_mask is not None
                and np.any(largest & self._previous_connected_mask)
            )
            self._persistence_s = (
                self._persistence_s + dt_s if physical_candidate and (overlap or self._previous_connected_mask is None)
                else dt_s if physical_candidate
                else 0.0
            )
            self._previous_connected_mask = np.array(largest, copy=True) if physical_candidate else None
        persistence_pass = self._persistence_s >= self.config.persistence_time_s
        ready = physical_candidate and persistence_pass
        if ready:
            classification = self.LARGE_AVALANCHE_MOBILE_PATH
        elif physical_candidate:
            classification = self.PERSISTING_LARGE_UNSTABLE_REGION
        elif np.any(unstable):
            classification = self.LOCAL_STATIC_INSTABILITY
        else:
            classification = self.NO_LARGE_EVENT
        diagnostics = AvalanchePhysicalDiagnostics(
            classification=classification,
            start_angle_deg=material.start_angle_deg,
            stop_angle_deg=material.stop_angle_deg,
            unstable_cell_count=int(np.count_nonzero(unstable)),
            connected_region_count=region_count,
            largest_connected_cell_count=largest_count,
            largest_connected_area_m2=area,
            largest_connected_mobilizable_volume_m3=mobilizable_volume,
            mean_excess_start_deg=mean_start,
            maximum_excess_start_deg=max_start,
            mean_excess_stop_deg=mean_stop,
            maximum_excess_stop_deg=max_stop,
            persistence_s=self._persistence_s,
            persistence_required_s=self.config.persistence_time_s,
            current_mobile_volume_m3=mobile_volume,
            moving_mobile_volume_m3=moving_volume,
            maximum_mobile_speed_m_s=float(np.max(speed, initial=0.0)),
            current_mobile_active=mobile_active,
            connected_extent_pass=extent_pass,
            unstable_volume_pass=volume_pass,
            excess_slope_pass=excess_pass,
            persistence_pass=persistence_pass,
            transition_ready=ready,
            parameter_basis=self.config.parameter_basis,
            sensitivity_case=self.config.sensitivity_case,
            largest_connected_mask=largest,
            slope_angle_deg=slope_deg,
        )
        self._last_profile_ms["diagnostics"] = max(
            0.0,
            (perf_counter() - diagnostic_start) * 1_000.0
            - self._last_profile_ms["gradient"]
            - self._last_profile_ms["connected_components"],
        )
        return diagnostics, grad_x, grad_y, slope_rad

    @staticmethod
    def _validate_state(
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64).copy()
        mobile = np.asarray(mobile_height_m, dtype=np.float64).copy()
        momentum = np.asarray(mobile_momentum_m2_s, dtype=np.float64).copy()
        if mobile.shape != grid.shape or momentum.shape != grid.shape + (2,):
            raise ValueError("[LargeAvalanche] state/grid shape mismatch")
        if integrator.shape != grid.shape:
            raise ValueError("[LargeAvalanche] integrator/grid shape mismatch")
        if np.any(resting < 0.0) or np.any(mobile < 0.0):
            raise ValueError("[LargeAvalanche] thickness fields must be non-negative")
        if not np.all(np.isfinite(mobile)) or not np.all(np.isfinite(momentum)):
            raise ValueError("[LargeAvalanche] state contains NaN/Inf")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[LargeAvalanche] dt_s must be finite/positive")
        return resting, mobile, momentum, dt

    @staticmethod
    def _integrated_momentum(
        momentum_m2_s: np.ndarray,
        integrator: TerrainVolumeIntegrator,
        density_kg_m3: float,
    ) -> np.ndarray:
        return density_kg_m3 * np.einsum(
            "ijk,ij->k",
            np.asarray(momentum_m2_s),
            integrator.vertex_weights_m2,
            dtype=np.float64,
            optimize=True,
        )


def _largest_connected_region(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, int, int]:
    """Return the largest component using row runs and a compact union-find.

    Physical slope masks normally contain long coherent runs.  The run-length
    representation avoids a Python cell queue and scales with boundary
    complexity rather than total active-cell count.  It is also a natural CPU
    reference for a future compact active-edge GPU component pass.
    """

    source = np.asarray(mask, dtype=bool)
    if source.ndim != 2:
        raise ValueError("[LargeAvalanche] connected mask must be 2-D")
    if not np.any(source):
        return np.zeros_like(source), 0, 0
    runs: list[tuple[int, int, int, int]] = []
    parent: list[int] = []
    sizes: list[int] = []
    previous: list[tuple[int, int, int]] = []
    diagonal = 1 if connectivity == 8 else 0

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if sizes[root_left] < sizes[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        sizes[root_left] += sizes[root_right]

    for row in range(source.shape[0]):
        padded = np.pad(source[row].astype(np.int8, copy=False), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1) - 1
        current: list[tuple[int, int, int]] = []
        previous_index = 0
        for start, stop in zip(starts.tolist(), stops.tolist()):
            label = len(parent)
            parent.append(label)
            sizes.append(stop - start + 1)
            runs.append((row, start, stop, label))
            current.append((start, stop, label))
            while previous_index < len(previous) and previous[previous_index][1] < start - diagonal:
                previous_index += 1
            scan = previous_index
            while scan < len(previous) and previous[scan][0] <= stop + diagonal:
                prev_start, prev_stop, prev_label = previous[scan]
                if prev_stop >= start - diagonal and prev_start <= stop + diagonal:
                    union(label, prev_label)
                scan += 1
        previous = current

    component_sizes: dict[int, int] = {}
    for _, start, stop, label in runs:
        root = find(label)
        component_sizes[root] = component_sizes.get(root, 0) + stop - start + 1
    largest_root, largest_size = max(component_sizes.items(), key=lambda item: item[1])
    result = np.zeros_like(source)
    for row, start, stop, label in runs:
        if find(label) == largest_root:
            result[row, start : stop + 1] = True
    return result, len(component_sizes), largest_size
