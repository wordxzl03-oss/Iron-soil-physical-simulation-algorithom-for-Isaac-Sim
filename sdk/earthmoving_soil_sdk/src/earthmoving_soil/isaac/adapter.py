"""Passive Isaac bridge: observes bodies, applies wrench, publishes H_free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import numpy as np

from isaac_bulk_pipeline.config import MeshConfig
from isaac_bulk_pipeline.contact import IsaacChunkedContactMeshAdapter
from isaac_bulk_pipeline.visualization import ChunkedDynamicMeshAdapter

from ..core.soil_physics import SoilPhysics
from ..interaction.tool_state import ToolState
from ..running_gear.track_state import TrackState


@dataclass(frozen=True)
class IsaacTerrainSyncStatus:
    visual_source: str
    physics_and_visual_shared_authority: bool
    visual_dirty_tiles: int
    contact_dirty_tiles: int
    contact_policy: str


class IsaacSoilAdapter:
    """Thin adapter that never creates SimulationApp or owns the physics loop.

    Body bindings are existing Isaac RigidPrim-like objects. The caller owns
    articulation/controller actions and calls :meth:`step` from its own loop.
    Visual and optional contact meshes are derived from the same production
    ``H_free = H_resting + h_mobile`` authority. GPU mode publishes compact
    dirty tiles and performs no normal full-field terrain download.
    """

    def __init__(
        self,
        stage: Any,
        soil: SoilPhysics,
        *,
        sync_visual_mesh: bool = True,
        sync_contact_surface: bool = False,
        apply_reaction_wrench: bool = True,
        visual_root_prim_path: str = "/World/Terrain/SoilVisualChunks",
        contact_root_prim_path: str = "/World/Terrain/SoilContactChunks",
    ) -> None:
        self.stage = stage
        self.soil = soil
        self.sync_visual_mesh = bool(sync_visual_mesh)
        self.sync_contact_surface = bool(sync_contact_surface)
        self.apply_reaction_wrench = bool(apply_reaction_wrench)
        self.visual_root_prim_path = visual_root_prim_path
        self.contact_root_prim_path = contact_root_prim_path
        self._bodies: dict[str, Any] = {}
        self._visual: ChunkedDynamicMeshAdapter | None = None
        self._contact: IsaacChunkedContactMeshAdapter | None = None
        self._initialized = False

    def bind_tool_prim(self, tool_id: str, body: Any) -> None:
        self._bodies[tool_id] = body

    def bind_track_prim(self, track_id: str, body: Any) -> None:
        if track_id not in {"left_track", "right_track"}:
            raise ValueError("track_id must be left_track/right_track")
        self._bodies[track_id] = body

    def initialize(self, *, timestamp_s: float = 0.0) -> None:
        if self._initialized:
            raise RuntimeError("IsaacSoilAdapter already initialized")
        if self.sync_visual_mesh:
            self._visual = ChunkedDynamicMeshAdapter(self.soil.config.tile_size)
            self._visual.initialize(
                self.stage,
                self.soil.config.grid,
                MeshConfig(collision_enabled=False),
                self.soil.initial_heightmap_m,
                root_prim_path=self.visual_root_prim_path,
            )
        if self.sync_contact_surface:
            self._contact = IsaacChunkedContactMeshAdapter(self.soil.config.tile_size)
            self._contact.initialize(
                self.stage, self.soil.config.grid, self.soil.initial_heightmap_m,
                root_prim_path=self.contact_root_prim_path,
                timestamp_s=timestamp_s,
            )
        self._initialized = True

    def observe(self, timestamp_s: float, *, belt_speeds: dict[str, float | None] | None = None) -> None:
        belts = belt_speeds or {}
        for tool_id in self.soil._tool_geometry:
            body = self._required_body(tool_id)
            pose, linear, angular = self._body_state(body)
            self.soil.set_tool_state(tool_id, ToolState(pose, linear, angular, timestamp_s))
        for track_id in self.soil._track_geometry:
            body = self._required_body(track_id)
            pose, linear, angular = self._body_state(body)
            self.soil.set_track_state(
                track_id,
                TrackState(pose, linear, angular, belts.get(track_id), timestamp_s),
            )

    def step(
        self,
        dt_s: float,
        *,
        timestamp_s: float,
        phase: str = "coordinated_cut",
        cycle: int = 1,
        belt_speeds: dict[str, float | None] | None = None,
    ) -> Any:
        if not self._initialized:
            raise RuntimeError("initialize IsaacSoilAdapter before step")
        self.observe(timestamp_s, belt_speeds=belt_speeds)
        result = self.soil.step(dt_s, phase=phase, cycle=cycle)
        if self.apply_reaction_wrench:
            for tool_id, wrench in result.tool_wrench.items():
                if wrench is not None:
                    self._apply_wrench(self._required_body(tool_id), wrench)
        self.synchronize_terrain(timestamp_s)
        return result

    def synchronize_terrain(self, timestamp_s: float) -> IsaacTerrainSyncStatus:
        visual_count = contact_count = 0
        if self.soil.config.runtime_backend == "GPU_RUNTIME":
            samples = self.soil.production_core.consume_dirty_surface_tiles()
            if self._visual is not None and samples:
                visual_count = self._visual.update_tile_samples(
                    samples, tile_size_cells=self.soil.config.tile_size
                ).dirty_chunk_count
            if self._contact is not None and samples:
                contact_count = self._contact.update_tile_samples(
                    samples, tile_size_cells=self.soil.config.tile_size,
                    terrain_timestamp_s=timestamp_s,
                ).dirty_chunk_count
        else:
            state = self.soil.production_core.state
            surface = state.H_resting_m + state.mobile_height_m
            if self._visual is not None:
                visual_count = self._visual.update(surface).dirty_chunk_count
            if self._contact is not None:
                contact_count = self._contact.update(
                    surface, terrain_timestamp_s=timestamp_s
                ).dirty_chunk_count
        return IsaacTerrainSyncStatus(
            visual_source="H_FREE",
            physics_and_visual_shared_authority=True,
            visual_dirty_tiles=visual_count,
            contact_dirty_tiles=contact_count,
            contact_policy=(
                "CALLER_SCHEDULED_DIRTY_CHUNK_RECOOK"
                if self._contact is not None else "NOT_ENABLED"
            ),
        )

    def reset(self, *, timestamp_s: float = 0.0) -> None:
        self.soil.reset()
        if self._visual is not None:
            self._visual.reset(self.soil.initial_heightmap_m)
        if self._contact is not None:
            self._contact.update(
                self.soil.initial_heightmap_m, terrain_timestamp_s=timestamp_s
            )

    def _required_body(self, body_id: str) -> Any:
        if body_id not in self._bodies:
            raise RuntimeError(f"Isaac body is not bound: {body_id}")
        return self._bodies[body_id]

    @staticmethod
    def _body_state(body: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions, quaternions = body.get_world_poses()
        position = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[0]
        quaternion = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4)[0]
        rotation = IsaacSoilAdapter._rotation_wxyz(quaternion)
        pose = np.eye(4); pose[:3, :3] = rotation; pose[:3, 3] = position
        if hasattr(body, "get_velocities"):
            velocities = np.asarray(body.get_velocities(), dtype=np.float64).reshape(-1, 6)[0]
            linear, angular = velocities[:3], velocities[3:]
        else:
            linear = np.asarray(body.get_linear_velocities(), dtype=np.float64).reshape(-1, 3)[0]
            angular = np.asarray(body.get_angular_velocities(), dtype=np.float64).reshape(-1, 3)[0]
        return pose, linear, angular

    @staticmethod
    def _rotation_wxyz(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q / np.linalg.norm(q)
        return np.asarray([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _apply_wrench(body: Any, wrench: Any) -> None:
        method = getattr(body, "apply_forces_and_torques_at_pos", None)
        if not callable(method):
            raise RuntimeError("bound tool lacks apply_forces_and_torques_at_pos")
        method(
            forces=np.asarray(wrench.force_world, dtype=np.float32).reshape(1, 3),
            torques=np.asarray(
                wrench.residual_couple_world
                if wrench.residual_couple_world is not None
                else np.zeros(3), dtype=np.float32
            ).reshape(1, 3),
            positions=np.asarray(wrench.application_point_world, dtype=np.float32).reshape(1, 3),
            is_global=True,
        )
