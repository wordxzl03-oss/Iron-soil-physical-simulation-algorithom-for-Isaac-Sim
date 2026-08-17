"""Local, positivity-preserving finite-volume mobile granular layer."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain.terrain_grid import TerrainGrid
from .yield_criterion import CohesiveYieldState, evaluate_cohesive_yield


def _readonly(value: np.ndarray) -> np.ndarray:
    candidate = np.asarray(value)
    if (
        candidate.dtype == np.float64
        and candidate.flags.c_contiguous
        and candidate.flags.owndata
    ):
        # Solver outputs are newly owned arrays. Freezing them in place avoids
        # a second 701x701 copy while preserving the public immutable contract.
        result = candidate
    else:
        result = np.ascontiguousarray(np.asarray(value, dtype=np.float64).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MobileLayerConfig:
    gravity_m_s2: float = 9.81
    cfl: float = 0.35
    maximum_substeps: int = 64
    dry_tolerance_m: float = 1e-8
    velocity_cap_m_s: float = 8.0
    pressure_coefficient: float = 0.45
    tool_forcing_relaxation_s: float = 0.12
    active_buffer_cells: int = 4
    boundary_condition: str = "closed"

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.gravity_m_s2,
                self.cfl,
                self.dry_tolerance_m,
                self.velocity_cap_m_s,
                self.pressure_coefficient,
                self.tool_forcing_relaxation_s,
            ]
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("[MobileLayer] numeric configuration must be finite/positive")
        if self.cfl > 0.9:
            raise ValueError("[MobileLayer] cfl must be <= 0.9")
        if not isinstance(self.maximum_substeps, int) or self.maximum_substeps < 1:
            raise ValueError("[MobileLayer] maximum_substeps must be >= 1")
        if not isinstance(self.active_buffer_cells, int) or self.active_buffer_cells < 0:
            raise ValueError("[MobileLayer] active_buffer_cells must be non-negative")
        if self.boundary_condition not in {"closed", "open"}:
            raise ValueError("[MobileLayer] boundary_condition must be closed/open")


@dataclass(frozen=True)
class MobileLayerResult:
    mobile_height_m: np.ndarray
    mobile_momentum_m2_s: np.ndarray
    outflow_volume_m3: float
    substeps: int
    cfl_limited: bool
    active_bbox_grid: tuple[int, int, int, int]
    volume_before_m3: float
    volume_after_m3: float
    maximum_speed_m_s: float
    minimum_height_m: float
    momentum_before_terrain_kg_m_s: np.ndarray
    momentum_after_terrain_kg_m_s: np.ndarray
    gravity_pressure_impulse_terrain_ns: np.ndarray
    basal_friction_impulse_terrain_ns: np.ndarray
    tool_impulse_on_mobile_terrain_ns: np.ndarray
    numerical_dissipative_impulse_terrain_ns: np.ndarray
    yielded_area_m2: float = 0.0
    moving_mobile_volume_m3: float = 0.0
    mobile_velocity_p95_m_s: float = 0.0
    yield_model_classification: str = "LITERATURE_INFORMED_REDUCED_ORDER__NOT_YET_PHYSICALLY_CALIBRATED"
    _trusted_solver_output: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        height = np.asarray(self.mobile_height_m, dtype=np.float64)
        momentum = np.asarray(self.mobile_momentum_m2_s, dtype=np.float64)
        if height.ndim != 2 or momentum.shape != height.shape + (2,):
            raise ValueError("[MobileLayer] output shapes invalid")
        if not self._trusted_solver_output:
            if not np.all(np.isfinite(height)) or np.any(height < -1e-12):
                raise ValueError("[MobileLayer] output height must be finite/non-negative")
            if not np.all(np.isfinite(momentum)):
                raise ValueError("[MobileLayer] output momentum must be finite")
            dry = height <= 1e-12
            if np.any(np.linalg.norm(momentum[dry], axis=-1) > 1e-10):
                raise ValueError("[MobileLayer] dry vertices must have zero momentum")
        for value in (
            self.outflow_volume_m3,
            self.volume_before_m3,
            self.volume_after_m3,
            self.maximum_speed_m_s,
            self.yielded_area_m2,
            self.moving_mobile_volume_m3,
            self.mobile_velocity_p95_m_s,
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("[MobileLayer] scalar output invalid")
        safe_height = height if self._trusted_solver_output else np.maximum(height, 0.0)
        object.__setattr__(self, "mobile_height_m", _readonly(safe_height))
        object.__setattr__(self, "mobile_momentum_m2_s", _readonly(momentum))
        for name in (
            "momentum_before_terrain_kg_m_s",
            "momentum_after_terrain_kg_m_s",
            "gravity_pressure_impulse_terrain_ns",
            "basal_friction_impulse_terrain_ns",
            "tool_impulse_on_mobile_terrain_ns",
            "numerical_dissipative_impulse_terrain_ns",
        ):
            vector = np.asarray(getattr(self, name), dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"[MobileLayer] {name} must be finite shape (3,)")
            object.__setattr__(self, name, _readonly(vector))


class MobileLayerSolver:
    """Graph finite-volume shallow granular transport on vertex control areas.

    The authoritative surface uses vertices. ``TerrainVolumeIntegrator`` area
    weights are therefore the control volumes. Every edge flux is represented
    as an explicit donor-to-receiver volume transfer; a donor limiter prevents
    negative thickness and guarantees closed-boundary volume conservation to
    roundoff. Momentum is advected with donor velocity, while gravity, pressure,
    Coulomb basal friction and tool forcing are explicit source terms.
    """

    def __init__(self, config: MobileLayerConfig | None = None) -> None:
        self.config = config or MobileLayerConfig()
        self._last_yield_state: CohesiveYieldState | None = None
        self._last_conservative_export_m3 = np.empty((0, 0), dtype=np.float64)

    @property
    def last_yield_state(self) -> CohesiveYieldState | None:
        return self._last_yield_state

    @property
    def last_conservative_export_m3(self) -> np.ndarray:
        """Actual donor-limited control-volume exports from the last step."""

        value = np.array(self._last_conservative_export_m3, copy=True)
        value.setflags(write=False)
        return value

    def step(
        self,
        H_resting_m: np.ndarray,
        mobile_height_m: np.ndarray,
        mobile_momentum_m2_s: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        dt_s: float,
        *,
        tool_forcing_mask: np.ndarray | None = None,
        tool_velocity_xy_m_s: np.ndarray | None = None,
    ) -> MobileLayerResult:
        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        height = np.asarray(mobile_height_m, dtype=np.float64).copy()
        momentum_field = np.asarray(mobile_momentum_m2_s, dtype=np.float64).copy()
        if height.shape != grid.shape or momentum_field.shape != grid.shape + (2,):
            raise ValueError("[MobileLayer] state/grid shape mismatch")
        if not np.all(np.isfinite(height)) or np.any(height < 0.0) or not np.all(np.isfinite(momentum_field)):
            raise ValueError("[MobileLayer] input fields must be finite and height non-negative")
        if integrator.shape != grid.shape:
            raise ValueError("[MobileLayer] integrator/grid shape mismatch")
        dt = float(dt_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("[MobileLayer] dt_s must be finite/positive")
        if tool_forcing_mask is None:
            forcing_mask = np.zeros(grid.shape, dtype=bool)
        else:
            forcing_mask = np.asarray(tool_forcing_mask, dtype=bool)
            if forcing_mask.shape != grid.shape:
                raise ValueError("[MobileLayer] tool forcing mask shape mismatch")
        if tool_velocity_xy_m_s is None:
            tool_velocity = np.zeros(2, dtype=np.float64)
        else:
            tool_velocity = np.asarray(tool_velocity_xy_m_s, dtype=np.float64)
            if tool_velocity.shape != (2,) or not np.all(np.isfinite(tool_velocity)):
                raise ValueError("[MobileLayer] tool velocity must be finite shape (2,)")

        weights = integrator.vertex_weights_m2
        density = float(material.assumed_bulk_density_kg_m3)
        momentum_before = self._integrated_momentum(momentum_field, weights, density)
        gravity_pressure_impulse = np.zeros(3, dtype=np.float64)
        basal_friction_impulse = np.zeros(3, dtype=np.float64)
        tool_impulse = np.zeros(3, dtype=np.float64)
        initial_volume = integrator.integrate(height)
        outflow = 0.0
        conservative_export = np.zeros(grid.shape, dtype=np.float64)
        remaining = dt
        substeps = 0
        cfl_limited = False
        active_mask = self._buffered_active_mask(
            (height > self.config.dry_tolerance_m) | forcing_mask,
            self.config.active_buffer_cells,
        )
        if not np.any(active_mask):
            self._last_conservative_export_m3 = conservative_export
            self._last_yield_state = evaluate_cohesive_yield(
                resting, height, material, grid, integrator, layer_depth_m=height
            )
            momentum_field.fill(0.0)
            return MobileLayerResult(
                mobile_height_m=height,
                mobile_momentum_m2_s=momentum_field,
                outflow_volume_m3=0.0,
                substeps=0,
                cfl_limited=False,
                active_bbox_grid=(0, 0, 0, 0),
                volume_before_m3=initial_volume,
                volume_after_m3=initial_volume,
                maximum_speed_m_s=0.0,
                minimum_height_m=float(height.min(initial=0.0)),
                momentum_before_terrain_kg_m_s=momentum_before,
                momentum_after_terrain_kg_m_s=np.zeros(3),
                gravity_pressure_impulse_terrain_ns=np.zeros(3),
                basal_friction_impulse_terrain_ns=np.zeros(3),
                tool_impulse_on_mobile_terrain_ns=np.zeros(3),
                numerical_dissipative_impulse_terrain_ns=-momentum_before,
                yielded_area_m2=self._last_yield_state.yielded_area_m2,
            )

        while remaining > 1e-14:
            if substeps >= self.config.maximum_substeps:
                raise RuntimeError(
                    "[MobileLayer] maximum_substeps exceeded; reduce dt or increase limit"
                )
            velocity = self._velocity(height, momentum_field)
            wave_speed = np.linalg.norm(velocity, axis=-1) + np.sqrt(
                self.config.pressure_coefficient
                * self.config.gravity_m_s2
                * np.maximum(height, 0.0)
            )
            maximum_wave = float(np.max(wave_speed[active_mask], initial=0.0))
            stable_dt = (
                remaining
                if maximum_wave <= 1e-12
                else self.config.cfl * min(grid.dx, grid.dy) / maximum_wave
            )
            sub_dt = min(remaining, stable_dt)
            if sub_dt < remaining - 1e-14:
                cfl_limited = True

            velocity, gravity_impulse, friction_impulse, forcing_impulse = self._apply_sources(
                resting,
                height,
                velocity,
                material,
                grid,
                sub_dt,
                forcing_mask,
                tool_velocity,
                weights,
                density,
                integrator,
            )
            gravity_pressure_impulse += gravity_impulse
            basal_friction_impulse += friction_impulse
            tool_impulse += forcing_impulse
            volume = height * weights
            momentum_volume = volume[..., None] * velocity
            transfers = self._edge_transfers(height, velocity, grid, sub_dt, active_mask)
            conservative_export += self._apply_transfers(
                volume, momentum_volume, velocity, transfers
            )
            if self.config.boundary_condition == "open":
                escaped = self._apply_open_boundary_outflow(
                    volume, momentum_volume, velocity, weights, grid, sub_dt
                )
                outflow += escaped
            height = np.divide(volume, weights, out=np.zeros_like(volume), where=weights > 0.0)
            # Never erase a thin control-volume remainder: dry_tolerance is a
            # velocity/flux guard, not a mass sink. Deposition owns the only
            # mobile-to-resting conversion.
            height = np.maximum(height, 0.0)
            velocity = np.divide(
                momentum_volume,
                volume[..., None],
                out=np.zeros_like(momentum_volume),
                where=volume[..., None] > self.config.dry_tolerance_m * weights[..., None],
            )
            speed = np.linalg.norm(velocity, axis=-1)
            excessive = speed > self.config.velocity_cap_m_s
            if np.any(excessive):
                velocity[excessive] *= (
                    self.config.velocity_cap_m_s / speed[excessive]
                )[:, None]
            momentum_field = height[..., None] * velocity
            momentum_field[height <= self.config.dry_tolerance_m] = 0.0
            remaining -= sub_dt
            substeps += 1

        final_volume = integrator.integrate(height)
        self._last_conservative_export_m3 = conservative_export
        balance_error = initial_volume - final_volume - outflow
        tolerance = max(1e-11, 1e-10 * max(initial_volume, 1.0))
        if abs(balance_error) > tolerance:
            raise RuntimeError(
                "[MobileLayer] finite-volume balance failed: "
                f"before={initial_volume}, after={final_volume}, outflow={outflow}, "
                f"error={balance_error}"
            )
        final_velocity = self._velocity(height, momentum_field)
        final_speed = np.linalg.norm(final_velocity, axis=-1)
        moving = (height > self.config.dry_tolerance_m) & (final_speed > 0.01)
        self._last_yield_state = evaluate_cohesive_yield(
            resting, height, material, grid, integrator,
            layer_depth_m=height, moving_mask=moving,
            gravity_m_s2=self.config.gravity_m_s2,
        )
        wet_speed = final_speed[height > self.config.dry_tolerance_m]
        momentum_after = self._integrated_momentum(momentum_field, weights, density)
        numerical_dissipative_impulse = (
            momentum_after
            - momentum_before
            - gravity_pressure_impulse
            - basal_friction_impulse
            - tool_impulse
        )
        return MobileLayerResult(
            mobile_height_m=height,
            mobile_momentum_m2_s=momentum_field,
            outflow_volume_m3=outflow,
            substeps=substeps,
            cfl_limited=cfl_limited,
            active_bbox_grid=self._bbox(active_mask),
            volume_before_m3=initial_volume,
            volume_after_m3=final_volume,
            maximum_speed_m_s=float(np.max(np.linalg.norm(final_velocity, axis=-1), initial=0.0)),
            minimum_height_m=float(np.min(height, initial=0.0)),
            momentum_before_terrain_kg_m_s=momentum_before,
            momentum_after_terrain_kg_m_s=momentum_after,
            gravity_pressure_impulse_terrain_ns=gravity_pressure_impulse,
            basal_friction_impulse_terrain_ns=basal_friction_impulse,
            tool_impulse_on_mobile_terrain_ns=tool_impulse,
            numerical_dissipative_impulse_terrain_ns=numerical_dissipative_impulse,
            yielded_area_m2=self._last_yield_state.yielded_area_m2,
            moving_mobile_volume_m3=integrator.integrate(np.where(moving, height, 0.0)),
            mobile_velocity_p95_m_s=float(np.percentile(wet_speed, 95.0)) if len(wet_speed) else 0.0,
        )

    def _apply_sources(
        self,
        resting: np.ndarray,
        height: np.ndarray,
        velocity: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        dt: float,
        forcing_mask: np.ndarray,
        tool_velocity: np.ndarray,
        weights: np.ndarray,
        density: float,
        integrator: TerrainVolumeIntegrator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        result = velocity.copy()
        # H_free is the shared physical surface.  H_resting alone contains
        # conservative Resting/Mobile ownership interfaces that are not
        # physical bed steps and must not drive Mobile acceleration.
        surface = resting + height
        grad_y, grad_x = np.gradient(surface, grid.dy, grid.dx, edge_order=1)
        pressure_y, pressure_x = np.gradient(height, grid.dy, grid.dx, edge_order=1)
        moving = np.linalg.norm(velocity, axis=-1) > 0.01
        yield_state = evaluate_cohesive_yield(
            resting, height, material, grid, integrator,
            layer_depth_m=height, moving_mask=moving,
            gravity_m_s2=self.config.gravity_m_s2,
        )
        dynamic = yield_state.continue_mask | forcing_mask
        result[..., 0] += np.where(dynamic, -self.config.gravity_m_s2 * (
            grad_x + self.config.pressure_coefficient * pressure_x
        ) * dt, 0.0)
        result[..., 1] += np.where(dynamic, -self.config.gravity_m_s2 * (
            grad_y + self.config.pressure_coefficient * pressure_y
        ) * dt, 0.0)
        gravity_impulse = self._velocity_change_impulse(
            height, velocity, result, weights, density
        )
        before_friction = result.copy()
        speed = np.linalg.norm(result, axis=-1)
        friction_delta = material.mobile_friction_coefficient * self.config.gravity_m_s2 * dt
        factor = np.maximum(0.0, 1.0 - friction_delta / np.maximum(speed, 1e-12))
        result *= factor[..., None]
        friction_impulse = self._velocity_change_impulse(
            height, before_friction, result, weights, density
        )
        before_tool = result.copy()
        if np.any(forcing_mask):
            blend = min(1.0, dt / self.config.tool_forcing_relaxation_s)
            result[forcing_mask] += blend * (tool_velocity - result[forcing_mask])
        result[height <= self.config.dry_tolerance_m] = 0.0
        tool_impulse = self._velocity_change_impulse(
            height, before_tool, result, weights, density
        )
        return result, gravity_impulse, friction_impulse, tool_impulse

    @staticmethod
    def _integrated_momentum(
        momentum_m2_s: np.ndarray,
        weights_m2: np.ndarray,
        density_kg_m3: float,
    ) -> np.ndarray:
        horizontal = density_kg_m3 * np.einsum(
            "ijk,ij->k",
            np.asarray(momentum_m2_s),
            np.asarray(weights_m2),
            dtype=np.float64,
            optimize=True,
        )
        return np.asarray([horizontal[0], horizontal[1], 0.0], dtype=np.float64)

    @staticmethod
    def _velocity_change_impulse(
        height_m: np.ndarray,
        before_m_s: np.ndarray,
        after_m_s: np.ndarray,
        weights_m2: np.ndarray,
        density_kg_m3: float,
    ) -> np.ndarray:
        horizontal = density_kg_m3 * np.sum(
            height_m[..., None]
            * (after_m_s - before_m_s)
            * weights_m2[..., None],
            axis=(0, 1),
            dtype=np.float64,
        )
        return np.asarray([horizontal[0], horizontal[1], 0.0], dtype=np.float64)

    @staticmethod
    def _velocity(height: np.ndarray, momentum: np.ndarray) -> np.ndarray:
        return np.divide(
            momentum,
            height[..., None],
            out=np.zeros_like(momentum),
            where=height[..., None] > 1e-12,
        )

    @staticmethod
    def _edge_transfers(
        height: np.ndarray,
        velocity: np.ndarray,
        grid: TerrainGrid,
        dt: float,
        active: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        transfers: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        # Each record holds flat donor indices, receiver indices and candidate m3.
        for axis, spacing, face_length in ((1, grid.dx, grid.dy), (0, grid.dy, grid.dx)):
            if axis == 1:
                left_h, right_h = height[:, :-1], height[:, 1:]
                face_velocity = 0.5 * (velocity[:, :-1, 0] + velocity[:, 1:, 0])
                edge_active = active[:, :-1] | active[:, 1:]
                rows, cols = np.indices(left_h.shape)
                first = rows * grid.nx + cols
                second = first + 1
            else:
                left_h, right_h = height[:-1, :], height[1:, :]
                face_velocity = 0.5 * (velocity[:-1, :, 1] + velocity[1:, :, 1])
                edge_active = active[:-1, :] | active[1:, :]
                rows, cols = np.indices(left_h.shape)
                first = rows * grid.nx + cols
                second = first + grid.nx
            upwind_h = np.where(face_velocity >= 0.0, left_h, right_h)
            signed_volume = upwind_h * face_velocity * face_length * dt
            valid = edge_active & (np.abs(signed_volume) > 0.0)
            donor = np.where(signed_volume >= 0.0, first, second)[valid]
            receiver = np.where(signed_volume >= 0.0, second, first)[valid]
            amount = np.abs(signed_volume[valid])
            transfers.append((donor.ravel(), receiver.ravel(), amount.ravel()))
        return transfers

    @staticmethod
    def _apply_transfers(
        volume: np.ndarray,
        momentum_volume: np.ndarray,
        velocity: np.ndarray,
        transfers: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        flat_volume = volume.ravel()
        flat_momentum = momentum_volume.reshape(-1, 2)
        flat_velocity = velocity.reshape(-1, 2)
        actual_outgoing = np.zeros_like(flat_volume)
        if not transfers:
            return actual_outgoing.reshape(volume.shape)
        donors = np.concatenate([item[0] for item in transfers])
        receivers = np.concatenate([item[1] for item in transfers])
        amounts = np.concatenate([item[2] for item in transfers])
        outgoing = np.zeros_like(flat_volume)
        np.add.at(outgoing, donors, amounts)
        factors = np.minimum(
            1.0,
            np.divide(flat_volume, outgoing, out=np.ones_like(flat_volume), where=outgoing > 0.0),
        )
        amounts *= factors[donors]
        np.add.at(actual_outgoing, donors, amounts)
        carried = amounts[:, None] * flat_velocity[donors]
        np.add.at(flat_volume, donors, -amounts)
        np.add.at(flat_volume, receivers, amounts)
        np.add.at(flat_momentum, donors, -carried)
        np.add.at(flat_momentum, receivers, carried)
        flat_volume[flat_volume < 0.0] = np.maximum(flat_volume[flat_volume < 0.0], 0.0)
        return actual_outgoing.reshape(volume.shape)

    @staticmethod
    def _apply_open_boundary_outflow(
        volume: np.ndarray,
        momentum_volume: np.ndarray,
        velocity: np.ndarray,
        control_area_m2: np.ndarray,
        grid: TerrainGrid,
        dt: float,
    ) -> float:
        escaped = 0.0
        boundaries = (
            (0, slice(None), np.maximum(-velocity[0, :, 1], 0.0), grid.dx),
            (-1, slice(None), np.maximum(velocity[-1, :, 1], 0.0), grid.dx),
            (slice(None), 0, np.maximum(-velocity[:, 0, 0], 0.0), grid.dy),
            (slice(None), -1, np.maximum(velocity[:, -1, 0], 0.0), grid.dy),
        )
        for row, col, outward_speed, face_length in boundaries:
            available = volume[row, col]
            local_area = control_area_m2[row, col]
            local_height = np.divide(
                available,
                local_area,
                out=np.zeros_like(available),
                where=local_area > 0.0,
            )
            candidate = np.minimum(
                available,
                local_height * np.maximum(outward_speed, 0.0) * face_length * dt,
            )
            fraction = np.divide(candidate, available, out=np.zeros_like(candidate), where=available > 0.0)
            volume[row, col] -= candidate
            momentum_volume[row, col] *= (1.0 - fraction)[..., None]
            escaped += float(np.sum(candidate))
        return escaped

    @staticmethod
    def _buffered_active_mask(mask: np.ndarray, cells: int) -> np.ndarray:
        result = np.asarray(mask, dtype=bool).copy()
        for _ in range(cells):
            expanded = result.copy()
            expanded[1:] |= result[:-1]
            expanded[:-1] |= result[1:]
            expanded[:, 1:] |= result[:, :-1]
            expanded[:, :-1] |= result[:, 1:]
            result = expanded
        return result

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        indices = np.argwhere(mask)
        if not len(indices):
            return (0, 0, 0, 0)
        low = indices.min(axis=0)
        high = indices.max(axis=0) + 1
        return int(low[0]), int(low[1]), int(high[0]), int(high[1])
