"""Transform robot tool-link observations into terrain-local tool state."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
import warnings

import numpy as np

from ..terrain.terrain_grid import TerrainGrid
from .tool_descriptor import ToolDescriptor


def _immutable_array(value: np.ndarray, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(
            f"[ToolKinematicsAdapter] {name} must be finite shape {shape}; "
            f"received={array.shape}"
        )
    result = np.ascontiguousarray(array.copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ToolState:
    """One terrain-local tool observation in SI units.

    Poses use column-vector transforms. All proxy arrays are expressed in the
    terrain frame, translations are metres, and velocities are m/s and rad/s.
    """

    timestamp: float
    pose_world: np.ndarray
    pose_terrain: np.ndarray
    cutting_edge_terrain: np.ndarray
    bottom_profile_terrain: np.ndarray
    left_boundary_terrain: np.ndarray
    right_boundary_terrain: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    mouth_polygon_terrain: np.ndarray | None = None
    top_edge_terrain: np.ndarray | None = None
    separation_plane_direction_terrain: np.ndarray | None = None
    separation_plane_source: str = "TOOL_PLUS_Z_LEGACY_FALLBACK"
    tool_plus_z_separation_fallback_used: bool = True
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.timestamp):
            raise ValueError(
                f"[ToolKinematicsAdapter] timestamp must be finite; value={self.timestamp}"
            )
        for name in ("pose_world", "pose_terrain"):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape=(4, 4), name=name),
            )
        for name in (
            "cutting_edge_terrain",
            "bottom_profile_terrain",
            "left_boundary_terrain",
            "right_boundary_terrain",
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"[ToolKinematicsAdapter] {name} must be finite shape (N,3); "
                    f"received={array.shape}"
                )
            result = np.ascontiguousarray(array.copy())
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        for name in ("linear_velocity", "angular_velocity"):
            object.__setattr__(
                self,
                name,
                _immutable_array(getattr(self, name), shape=(3,), name=name),
            )
        for name in ("mouth_polygon_terrain", "top_edge_terrain"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
                raise ValueError(f"[ToolKinematicsAdapter] {name} must be finite shape (N,3)")
            result = np.ascontiguousarray(array.copy())
            result.setflags(write=False)
            object.__setattr__(self, name, result)
        separation = self.separation_plane_direction_terrain
        if separation is not None:
            vector = np.asarray(separation, dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1.0e-12:
                raise ValueError("[ToolKinematicsAdapter] separation direction must be finite/nonzero")
            object.__setattr__(
                self,
                "separation_plane_direction_terrain",
                _immutable_array(vector / np.linalg.norm(vector), shape=(3,), name="separation_plane_direction_terrain"),
            )
        if not isinstance(self.separation_plane_source, str) or not self.separation_plane_source:
            raise ValueError("[ToolKinematicsAdapter] separation_plane_source must be non-empty")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class ToolKinematicsAdapter:
    """Compute Tool Frame geometry and motion from an Isaac tool-link pose.

    ``update`` accepts ``T_world_from_link`` from :class:`RobotAdapter` and
    applies the configured ``T_link_from_tool``. The terrain transform and all
    descriptor distances are already in metres, so USD stage-unit conversion
    remains isolated inside RobotAdapter.
    """

    def __init__(self, descriptor: ToolDescriptor, terrain_grid: TerrainGrid) -> None:
        self._descriptor = descriptor
        self._terrain_grid = terrain_grid
        self._previous_timestamp: float | None = None
        self._previous_position: np.ndarray | None = None
        self._previous_rotation: np.ndarray | None = None

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    @property
    def terrain_grid(self) -> TerrainGrid:
        return self._terrain_grid

    def update(self, tool_link_pose_world: np.ndarray, timestamp: float) -> ToolState:
        """Return the current proxy state from a world-space tool-link pose."""

        link_pose = self._validate_affine(tool_link_pose_world, name="tool_link_pose_world")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError(
                f"[ToolKinematicsAdapter] timestamp must be finite; value={timestamp}"
            )
        if self._previous_timestamp is not None and timestamp < self._previous_timestamp:
            raise ValueError(
                "[ToolKinematicsAdapter] timestamp moved backwards; "
                f"previous={self._previous_timestamp}, current={timestamp}"
            )

        world_from_tool = link_pose @ self._descriptor.tool_to_link_matrix
        terrain_from_world = np.linalg.inv(
            self._terrain_grid.terrain_to_world_matrix
        )
        terrain_from_tool = terrain_from_world @ world_from_tool

        singular_values = np.linalg.svd(
            terrain_from_tool[:3, :3], compute_uv=False
        )
        mean_scale = float(np.mean(singular_values))
        non_uniform = not np.allclose(
            singular_values,
            mean_scale,
            rtol=1e-4,
            atol=1e-7,
        )
        if non_uniform:
            warnings.warn(
                "[ToolKinematicsAdapter] non-uniform scale detected in "
                "T_terrain_from_tool; proxy geometry is transformed exactly, "
                f"singular_values={singular_values.tolist()}",
                RuntimeWarning,
                stacklevel=2,
            )

        position = np.asarray(terrain_from_tool[:3, 3], dtype=np.float64)
        rotation = self._closest_rotation(terrain_from_tool[:3, :3])
        linear_velocity = np.zeros(3, dtype=np.float64)
        angular_velocity = np.zeros(3, dtype=np.float64)
        dt: float | None = None
        if self._previous_timestamp is not None:
            dt = timestamp - self._previous_timestamp
            if dt > 0.0:
                linear_velocity = (position - self._previous_position) / dt
                relative_rotation = rotation @ self._previous_rotation.T
                angular_velocity = self._rotation_vector(relative_rotation) / dt

        geometry = self._descriptor.bucket_geometry
        if geometry is None or geometry.legacy_fallback_used:
            mouth_polygon = None
            top_edge = None
            separation_direction = terrain_from_tool[:3, 2]
            separation_source = "TOOL_PLUS_Z_LEGACY_FALLBACK"
            separation_fallback = True
        else:
            mouth_polygon = self._transform_points(
                terrain_from_tool,
                geometry.mouth_polygon_local,
            )
            top_edge = self._transform_points(
                terrain_from_tool,
                geometry.top_edge_local,
            )
            separation_direction = (
                terrain_from_tool[:3, :3]
                @ geometry.separation_plane_direction_local
            )
            separation_source = "BUCKET_GEOMETRY_BOTTOM_PLATE"
            separation_fallback = False
        separation_direction = separation_direction / np.linalg.norm(separation_direction)
        state = ToolState(
            timestamp=timestamp,
            pose_world=world_from_tool,
            pose_terrain=terrain_from_tool,
            cutting_edge_terrain=self._transform_points(
                terrain_from_tool, self._descriptor.cutting_edge_local
            ),
            bottom_profile_terrain=self._transform_points(
                terrain_from_tool, self._descriptor.bottom_profile_local
            ),
            left_boundary_terrain=self._transform_points(
                terrain_from_tool, self._descriptor.left_boundary_local
            ),
            right_boundary_terrain=self._transform_points(
                terrain_from_tool, self._descriptor.right_boundary_local
            ),
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            mouth_polygon_terrain=mouth_polygon,
            top_edge_terrain=top_edge,
            separation_plane_direction_terrain=separation_direction,
            separation_plane_source=separation_source,
            tool_plus_z_separation_fallback_used=separation_fallback,
            diagnostics={
                "scale_singular_values": singular_values.tolist(),
                "uniform_scale": mean_scale,
                "non_uniform_scale": non_uniform,
                "delta_time_s": dt,
                "bucket_geometry_source": self._descriptor.geometry_source,
                "bucket_geometry_quality": self._descriptor.geometry_quality,
                "tool_plus_z_separation_fallback_used": separation_fallback,
            },
        )
        self._previous_timestamp = timestamp
        self._previous_position = np.array(position, copy=True)
        self._previous_rotation = np.array(rotation, copy=True)
        return state

    def reset(self) -> None:
        """Clear velocity history without changing geometry or configuration."""

        self._previous_timestamp = None
        self._previous_position = None
        self._previous_rotation = None

    @staticmethod
    def _validate_affine(value: np.ndarray, *, name: str) -> np.ndarray:
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError(
                f"[ToolKinematicsAdapter] {name} must be finite shape (4,4); "
                f"received={matrix.shape}"
            )
        if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError(
                f"[ToolKinematicsAdapter] {name} must be affine column-vector form"
            )
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-12:
            raise ValueError(f"[ToolKinematicsAdapter] {name} is singular")
        return np.ascontiguousarray(matrix)

    @staticmethod
    def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack(
            (np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64))
        )
        transformed = (matrix @ homogeneous.T).T
        if np.any(np.abs(transformed[:, 3]) < 1e-12):
            raise ValueError(
                "[ToolKinematicsAdapter] point transform produced zero homogeneous w"
            )
        return np.ascontiguousarray(transformed[:, :3] / transformed[:, 3, None])

    @staticmethod
    def _closest_rotation(linear: np.ndarray) -> np.ndarray:
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            rotation = u @ vt
        return rotation

    @staticmethod
    def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
        cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
        angle = float(np.arccos(cosine))
        skew = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        )
        if angle < 1e-8:
            return 0.5 * skew
        if np.pi - angle < 1e-5:
            eigenvalues, eigenvectors = np.linalg.eig(rotation)
            index = int(np.argmin(np.abs(eigenvalues - 1.0)))
            axis = np.real(eigenvectors[:, index])
            axis /= np.linalg.norm(axis)
            return angle * axis
        return angle * skew / (2.0 * np.sin(angle))
