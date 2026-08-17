"""Standalone energy-auditable Mobile V2 finite-volume reference.

This module is deliberately excluded from the production operator chain.  It
implements an experimental barotropic depth-averaged layer with an independent
effective support ``b_eff``.  The pressure closure ``P=K*g*h^2/2`` is an
uncalibrated engineering closure, not a complete granular constitutive law.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MobileV2Config:
    dx_m: float = 0.05
    dy_m: float = 0.05
    gravity_m_s2: float = 9.81
    earth_pressure_coefficient: float = 0.45
    cfl: float = 0.18
    dry_tolerance_m: float = 1.0e-10
    start_angle_deg: float = 34.0
    stop_angle_deg: float = 30.0
    basal_friction_coefficient: float = 0.55
    maximum_substeps: int = 4096

    def __post_init__(self) -> None:
        values = np.asarray([
            self.dx_m, self.dy_m, self.gravity_m_s2,
            self.earth_pressure_coefficient, self.cfl,
            self.dry_tolerance_m, self.basal_friction_coefficient,
        ])
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("MobileV2 configuration must be finite/positive")
        if self.cfl > 0.25:
            raise ValueError("MobileV2 two-dimensional CFL must be <= 0.25")


@dataclass(frozen=True)
class MobileV2State:
    b_eff_m: np.ndarray
    h_m: np.ndarray
    q_m2_s: np.ndarray

    def __post_init__(self) -> None:
        b = np.asarray(self.b_eff_m, dtype=np.float64)
        h = np.asarray(self.h_m, dtype=np.float64)
        q = np.asarray(self.q_m2_s, dtype=np.float64)
        if b.ndim != 2 or h.shape != b.shape or q.shape != b.shape + (2,):
            raise ValueError("MobileV2 state shape mismatch")
        if not np.all(np.isfinite(b)) or not np.all(np.isfinite(h)) or not np.all(np.isfinite(q)):
            raise ValueError("MobileV2 state must be finite")
        if np.any(h < -1.0e-14):
            raise ValueError("MobileV2 h must be nonnegative")

    @property
    def H_free_m(self) -> np.ndarray:
        return np.asarray(self.b_eff_m) + np.asarray(self.h_m)


@dataclass(frozen=True)
class MobileV2Step:
    state: MobileV2State
    dt_s: float
    substeps: int
    maximum_cfl: float
    mass_before_m3: float
    mass_after_m3: float
    mass_residual_m3: float
    momentum_before_m4_s: np.ndarray
    momentum_after_m4_s: np.ndarray
    conservative_topography_wall_impulse_m4_s: np.ndarray
    friction_impulse_m4_s: np.ndarray
    external_impulse_m4_s: np.ndarray
    momentum_accounting_residual_m4_s: np.ndarray
    kinetic_before_j_per_density: float
    kinetic_after_j_per_density: float
    gravitational_before_j_per_density: float
    gravitational_after_j_per_density: float
    internal_before_j_per_density: float
    internal_after_j_per_density: float
    energy_before_j_per_density: float
    energy_after_j_per_density: float
    friction_dissipation_j_per_density: float
    external_work_j_per_density: float
    entrainment_deposition_energy_transfer_j_per_density: float
    numerical_energy_residual_j_per_density: float
    minimum_h_m: float


class MobileV2ReferenceSolver:
    """First-order face-shared hydrostatic-reconstruction FV reference."""

    classification = (
        "LITERATURE_INFORMED_REDUCED_ORDER+"
        "ENGINEERING_NUMERICAL_CLOSURE__NOT_PRODUCTION"
    )

    def __init__(self, config: MobileV2Config | None = None) -> None:
        self.config = config or MobileV2Config()

    def mass(self, state: MobileV2State) -> float:
        return float(np.sum(state.h_m, dtype=np.float64) * self.config.dx_m * self.config.dy_m)

    def momentum(self, state: MobileV2State) -> np.ndarray:
        return np.sum(state.q_m2_s, axis=(0, 1), dtype=np.float64) * self.config.dx_m * self.config.dy_m

    def energy_terms(self, state: MobileV2State) -> tuple[float, float, float, float]:
        h = np.asarray(state.h_m)
        q = np.asarray(state.q_m2_s)
        area = self.config.dx_m * self.config.dy_m
        speed2 = np.divide(
            np.sum(q * q, axis=-1), h * h,
            out=np.zeros_like(h), where=h > self.config.dry_tolerance_m,
        )
        kinetic = float(0.5 * np.sum(h * speed2, dtype=np.float64) * area)
        gravitational = float(
            self.config.gravity_m_s2
            * np.sum(h * state.b_eff_m, dtype=np.float64) * area
        )
        internal = float(
            0.5 * self.config.earth_pressure_coefficient
            * self.config.gravity_m_s2 * np.sum(h * h, dtype=np.float64) * area
        )
        return kinetic, gravitational, internal, kinetic + gravitational + internal

    def bookkeeping_relabel(
        self, state: MobileV2State, resting_label_delta_m: np.ndarray
    ) -> MobileV2State:
        """A label-only operation: physical ``b_eff/h/q/H_free`` are invariant."""
        delta = np.asarray(resting_label_delta_m, dtype=np.float64)
        if delta.shape != state.h_m.shape or not np.all(np.isfinite(delta)):
            raise ValueError("bookkeeping label delta shape/value invalid")
        return MobileV2State(
            np.array(state.b_eff_m, copy=True),
            np.array(state.h_m, copy=True),
            np.array(state.q_m2_s, copy=True),
        )

    def physical_entrainment(
        self,
        state: MobileV2State,
        depth_m: np.ndarray,
        entrained_velocity_m_s: np.ndarray | None = None,
    ) -> tuple[MobileV2State, float]:
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != state.h_m.shape or np.any(depth < 0.0):
            raise ValueError("entrainment depth invalid")
        if np.any(depth > state.b_eff_m + 1.0e-14):
            raise ValueError("entrainment exceeds available support")
        velocity = (
            np.zeros(state.q_m2_s.shape, dtype=np.float64)
            if entrained_velocity_m_s is None
            else np.broadcast_to(
                np.asarray(entrained_velocity_m_s, dtype=np.float64),
                state.q_m2_s.shape,
            )
        )
        before = self.energy_terms(state)[-1]
        result = MobileV2State(
            np.asarray(state.b_eff_m) - depth,
            np.asarray(state.h_m) + depth,
            np.asarray(state.q_m2_s) + depth[..., None] * velocity,
        )
        return result, self.energy_terms(result)[-1] - before

    def physical_deposition(
        self, state: MobileV2State, depth_m: np.ndarray
    ) -> tuple[MobileV2State, float]:
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != state.h_m.shape or np.any(depth < 0.0):
            raise ValueError("deposition depth invalid")
        if np.any(depth > state.h_m + 1.0e-14):
            raise ValueError("deposition exceeds Mobile thickness")
        before = self.energy_terms(state)[-1]
        retained = np.divide(
            np.asarray(state.h_m) - depth,
            state.h_m,
            out=np.zeros_like(depth),
            where=state.h_m > self.config.dry_tolerance_m,
        )
        result = MobileV2State(
            np.asarray(state.b_eff_m) + depth,
            np.maximum(np.asarray(state.h_m) - depth, 0.0),
            np.asarray(state.q_m2_s) * retained[..., None],
        )
        return result, self.energy_terms(result)[-1] - before

    def step(
        self,
        state: MobileV2State,
        dt_s: float,
        *,
        external_acceleration_m_s2: np.ndarray | None = None,
    ) -> MobileV2Step:
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite/positive")
        b = np.asarray(state.b_eff_m, dtype=np.float64).copy()
        h = np.asarray(state.h_m, dtype=np.float64).copy()
        q = np.asarray(state.q_m2_s, dtype=np.float64).copy()
        external = (
            np.zeros(q.shape, dtype=np.float64)
            if external_acceleration_m_s2 is None
            else np.broadcast_to(
                np.asarray(external_acceleration_m_s2, dtype=np.float64), q.shape
            ).copy()
        )
        initial = MobileV2State(b.copy(), h.copy(), q.copy())
        k0, g0, i0, e0 = self.energy_terms(initial)
        m0 = self.mass(initial)
        p0 = self.momentum(initial)
        conservative_impulse = np.zeros(2)
        friction_impulse = np.zeros(2)
        external_impulse = np.zeros(2)
        friction_dissipation = 0.0
        external_work = 0.0
        remaining = dt
        substeps = 0
        max_cfl = 0.0

        # Granular equilibrium: a fully static layer below Y_start is an exact
        # admissible equilibrium, not a shallow-water lake-at-rest restriction.
        if (
            np.max(np.abs(external), initial=0.0) <= 1.0e-14
            and self._globally_static_below_start(b, h, q)
        ):
            return self._result(
                initial, initial, dt, 0, 0.0, m0, p0,
                conservative_impulse, friction_impulse, external_impulse,
                k0, g0, i0, e0, 0.0, 0.0,
            )

        while remaining > 1.0e-14:
            if substeps >= self.config.maximum_substeps:
                raise RuntimeError("MobileV2 maximum_substeps exceeded")
            velocity = np.divide(
                q, h[..., None], out=np.zeros_like(q),
                where=h[..., None] > self.config.dry_tolerance_m,
            )
            wave = np.linalg.norm(velocity, axis=-1) + np.sqrt(
                self.config.earth_pressure_coefficient
                * self.config.gravity_m_s2 * np.maximum(h, 0.0)
            )
            maximum_wave = float(np.max(wave, initial=0.0))
            stable = (
                remaining if maximum_wave <= 1.0e-14
                else self.config.cfl * min(self.config.dx_m, self.config.dy_m)
                / maximum_wave
            )
            sub_dt = min(remaining, stable)
            max_cfl = max(
                max_cfl,
                maximum_wave * sub_dt / min(self.config.dx_m, self.config.dy_m),
            )
            before_conservative = np.sum(q, axis=(0, 1), dtype=np.float64)
            dh = np.zeros_like(h)
            dq = np.zeros_like(q)
            self._face_updates(b, h, q, sub_dt, dh, dq, axis=1)
            self._face_updates(b, h, q, sub_dt, dh, dq, axis=0)
            h_new = h + dh
            q_new = q + dq
            if np.min(h_new, initial=0.0) < -2.0e-13:
                raise RuntimeError("MobileV2 positivity failure")
            h_new = np.maximum(h_new, 0.0)
            q_new[h_new <= self.config.dry_tolerance_m] = 0.0
            after_conservative = np.sum(q_new, axis=(0, 1), dtype=np.float64)
            conservative_impulse += (
                after_conservative - before_conservative
            ) * self.config.dx_m * self.config.dy_m

            # External work/impulse uses midpoint velocity for the explicit
            # acceleration source.
            velocity_before = np.divide(
                q_new, h_new[..., None], out=np.zeros_like(q_new),
                where=h_new[..., None] > self.config.dry_tolerance_m,
            )
            q_external = q_new + h_new[..., None] * external * sub_dt
            velocity_external = np.divide(
                q_external, h_new[..., None], out=np.zeros_like(q_external),
                where=h_new[..., None] > self.config.dry_tolerance_m,
            )
            area = self.config.dx_m * self.config.dy_m
            external_impulse += np.sum(q_external - q_new, axis=(0, 1)) * area
            external_work += float(np.sum(
                h_new * np.sum(
                    0.5 * (velocity_before + velocity_external) * external,
                    axis=-1,
                ),
                dtype=np.float64,
            ) * area * sub_dt)

            speed = np.linalg.norm(velocity_external, axis=-1)
            decrement = self.config.basal_friction_coefficient * self.config.gravity_m_s2 * sub_dt
            factor = np.maximum(0.0, 1.0 - decrement / np.maximum(speed, 1.0e-15))
            velocity_after = velocity_external * factor[..., None]
            q_after = h_new[..., None] * velocity_after
            friction_impulse += np.sum(q_after - q_external, axis=(0, 1)) * area
            friction_dissipation += float(0.5 * np.sum(
                h_new * (speed * speed - np.sum(velocity_after * velocity_after, axis=-1)),
                dtype=np.float64,
            ) * area)
            h, q = h_new, q_after
            remaining -= sub_dt
            substeps += 1

        final = MobileV2State(b, h, q)
        return self._result(
            initial, final, dt, substeps, max_cfl, m0, p0,
            conservative_impulse, friction_impulse, external_impulse,
            k0, g0, i0, e0, friction_dissipation, external_work,
        )

    def _result(
        self,
        initial: MobileV2State,
        final: MobileV2State,
        dt: float,
        substeps: int,
        max_cfl: float,
        mass_before: float,
        momentum_before: np.ndarray,
        conservative_impulse: np.ndarray,
        friction_impulse: np.ndarray,
        external_impulse: np.ndarray,
        k0: float,
        g0: float,
        i0: float,
        e0: float,
        friction_dissipation: float,
        external_work: float,
    ) -> MobileV2Step:
        k1, g1, i1, e1 = self.energy_terms(final)
        mass_after = self.mass(final)
        momentum_after = self.momentum(final)
        expected_delta = conservative_impulse + friction_impulse + external_impulse
        energy_residual = (
            e1 - e0 - external_work + friction_dissipation
        )
        return MobileV2Step(
            final, dt, substeps, max_cfl,
            mass_before, mass_after, mass_after - mass_before,
            momentum_before, momentum_after, conservative_impulse,
            friction_impulse, external_impulse,
            momentum_after - momentum_before - expected_delta,
            k0, k1, g0, g1, i0, i1, e0, e1,
            friction_dissipation, external_work, 0.0, energy_residual,
            float(np.min(final.h_m, initial=0.0)),
        )

    def _globally_static_below_start(
        self, b: np.ndarray, h: np.ndarray, q: np.ndarray
    ) -> bool:
        if np.max(np.abs(q), initial=0.0) > 1.0e-14:
            return False
        free = b + h
        gy, gx = np.gradient(free, self.config.dy_m, self.config.dx_m)
        return bool(
            np.max(np.hypot(gx, gy), initial=0.0)
            <= np.tan(np.deg2rad(self.config.start_angle_deg)) + 1.0e-14
        )

    def _face_updates(
        self,
        b: np.ndarray,
        h: np.ndarray,
        q: np.ndarray,
        dt: float,
        dh: np.ndarray,
        dq: np.ndarray,
        *,
        axis: int,
    ) -> None:
        K = self.config.earth_pressure_coefficient
        g = self.config.gravity_m_s2
        if axis == 1:
            b_l, b_r = b[:, :-1], b[:, 1:]
            h_l, h_r = h[:, :-1], h[:, 1:]
            q_l, q_r = q[:, :-1], q[:, 1:]
            spacing = self.config.dx_m
            normal = 0
        else:
            b_l, b_r = b[:-1, :], b[1:, :]
            h_l, h_r = h[:-1, :], h[1:, :]
            q_l, q_r = q[:-1, :], q[1:, :]
            spacing = self.config.dy_m
            normal = 1

        # Generalized hydrostatic reconstruction for p'(h)=K*g*h and
        # topographic source -g*h*grad(b): steady head is b+K*h.
        head_l = b_l + K * h_l
        head_r = b_r + K * h_r
        b_face = np.maximum(b_l, b_r)
        hs_l = np.maximum((head_l - b_face) / K, 0.0)
        hs_r = np.maximum((head_r - b_face) / K, 0.0)
        scale_l = np.divide(hs_l, h_l, out=np.zeros_like(h_l), where=h_l > self.config.dry_tolerance_m)
        scale_r = np.divide(hs_r, h_r, out=np.zeros_like(h_r), where=h_r > self.config.dry_tolerance_m)
        qs_l = q_l * scale_l[..., None]
        qs_r = q_r * scale_r[..., None]
        flux_l = self._physical_flux(hs_l, qs_l, normal)
        flux_r = self._physical_flux(hs_r, qs_r, normal)
        ul = np.divide(qs_l[..., normal], hs_l, out=np.zeros_like(hs_l), where=hs_l > self.config.dry_tolerance_m)
        ur = np.divide(qs_r[..., normal], hs_r, out=np.zeros_like(hs_r), where=hs_r > self.config.dry_tolerance_m)
        speed = np.maximum(
            np.abs(ul) + np.sqrt(K * g * hs_l),
            np.abs(ur) + np.sqrt(K * g * hs_r),
        )
        U_l = np.concatenate((hs_l[..., None], qs_l), axis=-1)
        U_r = np.concatenate((hs_r[..., None], qs_r), axis=-1)
        flux = 0.5 * (flux_l + flux_r) - 0.5 * speed[..., None] * (U_r - U_l)
        p_l = 0.5 * K * g * h_l * h_l
        p_r = 0.5 * K * g * h_r * h_r
        ps_l = 0.5 * K * g * hs_l * hs_l
        ps_r = 0.5 * K * g * hs_r * hs_r
        flux_minus = flux.copy()
        flux_plus = flux.copy()
        flux_minus[..., 1 + normal] += p_l - ps_l
        flux_plus[..., 1 + normal] += p_r - ps_r
        factor = dt / spacing
        if axis == 1:
            dh[:, :-1] -= factor * flux_minus[..., 0]
            dh[:, 1:] += factor * flux_plus[..., 0]
            dq[:, :-1] -= factor * flux_minus[..., 1:]
            dq[:, 1:] += factor * flux_plus[..., 1:]
        else:
            dh[:-1, :] -= factor * flux_minus[..., 0]
            dh[1:, :] += factor * flux_plus[..., 0]
            dq[:-1, :] -= factor * flux_minus[..., 1:]
            dq[1:, :] += factor * flux_plus[..., 1:]

    def _physical_flux(
        self, h: np.ndarray, q: np.ndarray, normal: int
    ) -> np.ndarray:
        velocity = np.divide(
            q, h[..., None], out=np.zeros_like(q),
            where=h[..., None] > self.config.dry_tolerance_m,
        )
        normal_discharge = q[..., normal]
        result = np.empty(h.shape + (3,), dtype=np.float64)
        result[..., 0] = normal_discharge
        result[..., 1] = normal_discharge * velocity[..., 0]
        result[..., 2] = normal_discharge * velocity[..., 1]
        result[..., 1 + normal] += (
            0.5 * self.config.earth_pressure_coefficient
            * self.config.gravity_m_s2 * h * h
        )
        return result
