"""Small public facade over the frozen production EarthmovingPhysicsCore."""

from __future__ import annotations

from dataclasses import replace
from typing import Any
import numpy as np

from isaac_bulk_pipeline.runtime import EarthmovingPhysicsCore, ResetLevel, SoilForceMode
from isaac_bulk_pipeline.tools import ToolKinematicsAdapter

from .._version import PACKAGE_VERSION, PHYSICS_CORE_VERSION
from ..diagnostics.physics_diagnostics import PhysicsDiagnostics
from ..interaction.tool_state import ToolGeometry, ToolState
from ..rl.feedback import RLSoilFeedback
from ..running_gear.track_state import TrackGeometry, TrackState
from .config import SoilConfig
from .state import ReactionWrench, SoilStepResult


class SoilPhysics:
    """Vehicle-independent owner of the existing production soil core.

    The external application owns time, vehicle motion and control. One bucket
    is supported by the current core. Left/right track schemas are registered
    independently and update the same authoritative terrain as the bucket.
    """

    def __init__(self, config: SoilConfig, initial_heightmap_m: np.ndarray) -> None:
        self.config = config
        self.initial_heightmap_m = np.array(
            config.grid.validate_heightmap(initial_heightmap_m), dtype=np.float64, copy=True
        )
        self._tool_geometry: dict[str, ToolGeometry] = {}
        self._tool_state: dict[str, ToolState] = {}
        self._track_geometry: dict[str, TrackGeometry] = {}
        self._track_state: dict[str, TrackState] = {}
        self._core: EarthmovingPhysicsCore | None = None
        self._kinematics: ToolKinematicsAdapter | None = None
        self._last_patch: np.ndarray | None = None
        self._last_payload_m3 = 0.0
        self._last_deposited_total_m3 = 0.0

    @property
    def production_core(self) -> EarthmovingPhysicsCore:
        if self._core is None:
            raise RuntimeError("register a bucket before accessing the production core")
        return self._core

    @property
    def state_authority(self) -> str:
        return self.config.state_authority

    def register_tool(self, tool_id: str, geometry: ToolGeometry) -> None:
        if tool_id in self._tool_geometry:
            raise ValueError(f"tool already registered: {tool_id}")
        if self._tool_geometry:
            raise NotImplementedError("current alpha supports one bucket tool")
        self._tool_geometry[tool_id] = geometry
        self._kinematics = ToolKinematicsAdapter(geometry.descriptor, self.config.grid)
        self._core = EarthmovingPhysicsCore(
            grid=self.config.grid,
            descriptor=geometry.descriptor,
            material=self.config.material.to_scenario(),
            initial_heightmap_m=self.initial_heightmap_m,
            runtime_backend=self.config.runtime_backend,
            gpu_device=self.config.gpu_device,
            solver_backend=self.config.solver_backend,
            slope_backend=self.config.slope_backend,
            tile_size=self.config.tile_size,
            numerical_safety_max_iterations=self.config.numerical_safety_max_iterations,
            minislope_round_budget_per_step=self.config.minislope_round_budget_per_step,
            minislope_tolerance_m=self.config.minislope_tolerance_m,
            large_avalanche_config=self.config.large_avalanche,
            mass_tolerance_m3=self.config.mass_tolerance_m3,
            track_soil_config=self.config.track_soil,
        )

    def set_tool_state(self, tool_id: str, state: ToolState) -> None:
        if tool_id not in self._tool_geometry:
            raise KeyError(f"unregistered tool: {tool_id}")
        self._tool_state[tool_id] = state

    def register_track(self, track_id: str, geometry: TrackGeometry) -> None:
        if track_id not in {"left_track", "right_track"}:
            raise ValueError("alpha track IDs are left_track and right_track")
        if track_id in self._track_geometry:
            raise ValueError(f"track already registered: {track_id}")
        self._track_geometry[track_id] = geometry

    def set_track_state(self, track_id: str, state: TrackState) -> None:
        if track_id not in self._track_geometry:
            raise KeyError(f"unregistered track: {track_id}")
        self._track_state[track_id] = state

    def _internal_tool_state(self, public: ToolState) -> Any:
        assert self._kinematics is not None
        generated = self._kinematics.update(public.pose, public.timestamp)
        rotation_world_to_terrain = np.linalg.inv(self.config.grid.terrain_to_world_matrix)[:3, :3]
        return replace(
            generated,
            linear_velocity=rotation_world_to_terrain @ public.linear_velocity,
            angular_velocity=rotation_world_to_terrain @ public.angular_velocity,
        )

    def initialize(self) -> None:
        if len(self._tool_geometry) != 1 or len(self._tool_state) != 1:
            raise RuntimeError("register and set exactly one bucket before initialize")
        public = next(iter(self._tool_state.values()))
        self.production_core.initialize_tool(self._internal_tool_state(public))

    def step(
        self,
        dt_s: float,
        *,
        phase: str = "coordinated_cut",
        cycle: int = 1,
        soil_force_mode: str = "FULL_SOIL_FORCE",
        phase_ending: bool = False,
    ) -> SoilStepResult:
        if len(self._tool_state) != 1:
            raise RuntimeError("set one registered bucket state before step")
        public_tool_id, public_tool = next(iter(self._tool_state.items()))
        internal = self._internal_tool_state(public_tool)
        if self.production_core.previous_tool_state is None:
            self.production_core.initialize_tool(internal)
        before_payload = self.production_core.payload.volume_m3
        production = self.production_core.step(
            internal, phase=phase, cycle=cycle, dt_s=float(dt_s),
            soil_force_mode=SoilForceMode(soil_force_mode), phase_ending=phase_ending,
        )
        track_result = self._apply_tracks(float(dt_s))
        reservoir = self.production_core.reservoir_observation()
        payload_m3 = self.production_core.payload.volume_m3
        captured = payload_m3 - before_payload
        density = self.config.material.bulk_density_kg_m3
        force_result = production.force_result
        wrench = self._world_wrench(force_result)
        interaction = production.interaction
        displaced = None if interaction is None else float(interaction.activated_volume_m3)
        released = None if production.dump_release is None else float(production.dump_release.released_volume_m3)
        deposited = None
        if production.dump_advance is not None:
            deposited = float(production.dump_advance.deposition.deposited_volume_m3)
        patch = self._local_patch(internal.pose_terrain[:3, 3])
        delta_patch = None if self._last_patch is None or self._last_patch.shape != patch.shape else patch - self._last_patch
        self._last_patch = patch.copy()
        diag = PhysicsDiagnostics.from_step(self.config, production.physics_diagnostics)
        pdiag = production.physics_diagnostics
        penetration = contact_area = None
        if interaction is not None:
            source = getattr(interaction, "failure_bridge", None)
            intersection = source.intersection if source is not None else None
            if intersection is not None:
                penetration = float(np.max(intersection.penetration_depth_m))
                contact_area = float(
                    np.count_nonzero(intersection.affected_mask)
                    * self.config.grid.cell_area
                )
            else:
                depths = [
                    item.penetration_depth_m
                    for item in interaction.failure_zone.strip_geometries
                ]
                penetration = max(depths, default=0.0)
                contact_area = float(
                    np.count_nonzero(interaction.failure_zone.active_thickness_m > 0.0)
                    * self.config.grid.cell_area
                )
        moving = float(pdiag.mobile_moving_volume_m3)
        max_speed = None
        mobile_result = None if interaction is None else interaction.mobile_result
        if mobile_result is not None:
            max_speed = float(getattr(mobile_result, "maximum_speed_m_s", 0.0))
        feedback = RLSoilFeedback(
            bucket_force_world=None if wrench is None else wrench.force_world,
            bucket_torque_world=None if wrench is None else wrench.torque_world,
            bucket_force_norm_n=None if wrench is None else float(np.linalg.norm(wrench.force_world)),
            bucket_force_peak_since_last_rl_step_n=None,
            bucket_force_mean_since_last_rl_step_n=None,
            bucket_impulse_since_last_rl_step_ns=None,
            bucket_torque_peak_since_last_rl_step_nm=None,
            bucket_torque_mean_since_last_rl_step_nm=None,
            left_track_force_world=None, left_track_torque_world=None,
            right_track_force_world=None, right_track_torque_world=None,
            penetration_depth_m=penetration, contact_area_m2=contact_area,
            yielded_area_m2=float(pdiag.yielded_area_m2),
            failure_active_volume_m3=float(pdiag.failure_active_volume_m3),
            payload_volume_m3=payload_m3, payload_mass_kg=payload_m3 * density,
            captured_volume_delta_m3=captured, captured_mass_delta_kg=captured * density,
            displaced_volume_delta_m3=displaced,
            displaced_mass_delta_kg=None if displaced is None else displaced * density,
            spill_volume_delta_m3=released,
            spill_mass_delta_kg=None if released is None else released * density,
            deposited_volume_delta_m3=deposited,
            deposited_mass_delta_kg=None if deposited is None else deposited * density,
            mobile_volume_m3=float(reservoir["mobile_volume_m3"]),
            mobile_mass_kg=float(reservoir["mobile_volume_m3"]) * density,
            moving_mobile_volume_m3=moving,
            mobile_speed_summary_m_s={"p95": float(pdiag.mobile_velocity_p95_m_s), "maximum": max_speed} if max_speed is not None else {"p95": float(pdiag.mobile_velocity_p95_m_s)},
            local_heightmap_patch_m=patch, terrain_delta_patch_m=delta_patch,
            physical_yield_area_m2=float(pdiag.yielded_area_m2),
            terrain_state=pdiag.terrain_state,
            mass_error_m3=float(reservoir["mass_balance_error_m3"]),
            soil_power_w=None if wrench is None else float(np.dot(wrench.force_world, public_tool.linear_velocity) + np.dot(wrench.torque_world, public_tool.angular_velocity)),
            soil_work_delta_j=None if wrench is None else float((np.dot(wrench.force_world, public_tool.linear_velocity) + np.dot(wrench.torque_world, public_tool.angular_velocity)) * dt_s),
            physics_core_version=PHYSICS_CORE_VERSION, package_version=PACKAGE_VERSION,
            material_profile=self.config.material.profile,
            config_hash=self.config.solver_config_hash,
            availability=self.feedback_availability(),
        )
        return SoilStepResult(
            tool_wrench={public_tool_id: wrench},
            track_wrench={key: None for key in self._track_geometry},
            payload_volume_m3=payload_m3, payload_mass_kg=payload_m3 * density,
            captured_volume_delta_m3=captured, captured_mass_delta_kg=captured * density,
            displaced_volume_delta_m3=displaced, spill_volume_delta_m3=released,
            deposited_volume_delta_m3=deposited,
            mobile_volume_m3=float(reservoir["mobile_volume_m3"]),
            moving_mobile_volume_m3=moving,
            mass_balance_error_m3=float(reservoir["mass_balance_error_m3"]),
            terrain_state=pdiag.terrain_state, diagnostics=diag, rl_feedback=feedback,
            production_result=production, track_soil_result=track_result,
        )

    def _world_wrench(self, result: Any) -> ReactionWrench | None:
        if result is None:
            return None
        transform = self.config.grid.terrain_to_world_matrix
        rotation = transform[:3, :3]
        point_h = transform @ np.r_[result.application_point_terrain_m, 1.0]
        return ReactionWrench(
            rotation @ result.force_terrain_n,
            rotation @ result.torque_about_tool_origin_terrain_nm,
            point_h[:3] / point_h[3],
            rotation @ result.residual_couple_terrain_nm,
        )

    def _apply_tracks(self, dt_s: float) -> Any:
        if not self._track_geometry:
            return None
        masks = {key: self._track_mask(key) for key in ("left_track", "right_track")}
        velocities = {key: self._track_surface_velocity(key) for key in ("left_track", "right_track")}
        body = [state.linear_velocity[:2] for state in self._track_state.values()]
        base = np.mean(body, axis=0) if body else np.zeros(2)
        return self.production_core.apply_track_soil(
            left_footprint_mask=masks["left_track"], right_footprint_mask=masks["right_track"],
            left_track_velocity_xy_m_s=velocities["left_track"],
            right_track_velocity_xy_m_s=velocities["right_track"],
            base_velocity_xy_m_s=base, dt_s=dt_s,
        )

    def _track_surface_velocity(self, track_id: str) -> np.ndarray:
        state = self._track_state.get(track_id)
        geometry = self._track_geometry.get(track_id)
        if state is None or geometry is None:
            return np.zeros(2)
        axis_world = state.pose[:3, :3] @ np.asarray(geometry.forward_axis_local)
        axis_xy = axis_world[:2]
        norm = np.linalg.norm(axis_xy)
        if norm > 1e-12:
            axis_xy /= norm
        belt = 0.0 if state.belt_speed is None else float(state.belt_speed)
        return state.linear_velocity[:2] + belt * axis_xy

    def _track_mask(self, track_id: str) -> np.ndarray:
        mask = np.zeros(self.config.grid.shape, dtype=bool)
        state = self._track_state.get(track_id)
        geometry = self._track_geometry.get(track_id)
        if state is None or geometry is None:
            return mask
        center = self.config.grid.world_to_terrain(state.pose[:3, 3])
        rotation = np.linalg.inv(self.config.grid.terrain_to_world_matrix)[:3, :3] @ state.pose[:3, :3]
        forward = (rotation @ np.asarray(geometry.forward_axis_local))[:2]
        forward /= max(np.linalg.norm(forward), 1e-12)
        lateral = np.asarray([-forward[1], forward[0]])
        rows, cols = np.indices(self.config.grid.shape)
        x = self.config.grid.origin_x + cols * self.config.grid.dx - center[0]
        y = self.config.grid.origin_y + rows * self.config.grid.dy - center[1]
        return (np.abs(x * forward[0] + y * forward[1]) <= geometry.length_m / 2) & (np.abs(x * lateral[0] + y * lateral[1]) <= geometry.width_m / 2)

    def _local_patch(self, center_terrain: np.ndarray) -> np.ndarray:
        row_col = self.config.grid.terrain_to_grid(center_terrain)
        radius = self.config.terrain_patch_radius_cells
        row = int(round(row_col[0])); col = int(round(row_col[1]))
        rows = np.arange(max(0, row-radius), min(self.config.grid.ny, row+radius+1))
        cols = np.arange(max(0, col-radius), min(self.config.grid.nx, col+radius+1))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        values = self.production_core.surface_at_grid_coordinates(np.column_stack((rr.ravel(), cc.ravel())))
        return values.reshape(rr.shape)

    def reset(self) -> None:
        self.production_core.reset(ResetLevel.ALL)
        if self._kinematics is not None:
            self._kinematics.reset()
        self._last_patch = None

    @staticmethod
    def feedback_availability() -> dict[str, str]:
        return {
            "bucket_force_world": "AVAILABLE_DIRECTLY",
            "bucket_torque_world": "AVAILABLE_DIRECTLY",
            "force_aggregation": "DERIVABLE_WITHOUT_NEW_PHYSICS",
            "track_wrench": "NOT_CURRENTLY_AVAILABLE",
            "penetration_depth": "AVAILABLE_DIRECTLY",
            "contact_area": "AVAILABLE_DIRECTLY",
            "yielded_area": "AVAILABLE_DIRECTLY",
            "failure_active_volume": "AVAILABLE_DIRECTLY",
            "payload_volume": "AVAILABLE_DIRECTLY",
            "payload_mass": "DERIVABLE_WITHOUT_NEW_PHYSICS",
            "captured_delta": "DERIVABLE_WITHOUT_NEW_PHYSICS",
            "displaced_delta": "AVAILABLE_DIRECTLY",
            "spill_delta": "AVAILABLE_DIRECTLY_WHEN_DUMPING",
            "deposited_delta": "AVAILABLE_DIRECTLY_WHEN_DEPOSITING",
            "mobile_state": "AVAILABLE_DIRECTLY",
            "local_heightmap_patch": "DERIVABLE_WITHOUT_NEW_PHYSICS",
            "terrain_delta_patch": "DERIVABLE_WITHOUT_NEW_PHYSICS_AFTER_FIRST_SAMPLE",
            "mass_error": "AVAILABLE_DIRECTLY",
            "soil_power_work": "DERIVABLE_WITHOUT_NEW_PHYSICS",
            "compaction_density_evolution": "NOT_CURRENTLY_AVAILABLE",
        }
