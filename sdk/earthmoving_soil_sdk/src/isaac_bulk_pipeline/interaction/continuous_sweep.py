"""Adaptive continuous rasterization of a computational tool proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..config import SweepConfig
from ..terrain.terrain_grid import TerrainGrid
from ..tools import ToolDescriptor, ToolState


@dataclass(frozen=True)
class SweepResult:
    """Rasterized tool sweep over the complete terrain grid.

    ``affected_bbox_grid`` is ``(row_min, column_min, row_max, column_max)``
    with exclusive maxima. ``cut_surface`` is metres in terrain-local Z and is
    ``+inf`` outside ``affected_mask``.
    """

    affected_bbox_grid: tuple[int, int, int, int]
    affected_mask: np.ndarray
    cut_surface: np.ndarray
    sampled_poses: tuple[np.ndarray, ...]
    diagnostics: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        mask = np.asarray(self.affected_mask, dtype=bool)
        surface = np.asarray(self.cut_surface, dtype=np.float64)
        if mask.ndim != 2 or surface.shape != mask.shape:
            raise ValueError(
                "[ContinuousSweepBuilder] mask/cut surface shape mismatch; "
                f"mask={mask.shape}, cut_surface={surface.shape}"
            )
        if np.any(mask) and not np.all(np.isfinite(surface[mask])):
            raise ValueError(
                "[ContinuousSweepBuilder] affected cut surface contains NaN or Inf"
            )
        if len(self.affected_bbox_grid) != 4:
            raise ValueError(
                "[ContinuousSweepBuilder] affected_bbox_grid must contain four indices"
            )
        mask = np.ascontiguousarray(mask.copy())
        surface = np.ascontiguousarray(surface.copy())
        mask.setflags(write=False)
        surface.setflags(write=False)
        poses: list[np.ndarray] = []
        for pose in self.sampled_poses:
            matrix = np.asarray(pose, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "[ContinuousSweepBuilder] sampled pose must be finite (4,4); "
                    f"shape={matrix.shape}"
                )
            matrix = np.ascontiguousarray(matrix.copy())
            matrix.setflags(write=False)
            poses.append(matrix)
        object.__setattr__(self, "affected_mask", mask)
        object.__setattr__(self, "cut_surface", surface)
        object.__setattr__(self, "sampled_poses", tuple(poses))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class ContinuousSweepBuilder:
    """Build a gap-free 2.5-D cut surface between two :class:`ToolState`s."""

    def __init__(self, config: SweepConfig | None = None) -> None:
        self._config = config or SweepConfig()

    @property
    def config(self) -> SweepConfig:
        return self._config

    def build(
        self,
        previous: ToolState,
        current: ToolState,
        grid: TerrainGrid,
        descriptor: ToolDescriptor,
    ) -> SweepResult:
        """Rasterize adaptively sampled tool-bottom poses in terrain coordinates."""

        previous_pose = self._validate_pose(previous.pose_terrain, "previous.pose_terrain")
        current_pose = self._validate_pose(current.pose_terrain, "current.pose_terrain")
        translation_distance = float(
            np.linalg.norm(current_pose[:3, 3] - previous_pose[:3, 3])
        )
        previous_rotation, previous_stretch = self._polar(previous_pose[:3, :3])
        current_rotation, current_stretch = self._polar(current_pose[:3, :3])
        relative = current_rotation @ previous_rotation.T
        angular_displacement = self._rotation_angle(relative)
        cell_size = min(grid.dx, grid.dy)
        translation_limit = (
            self._config.max_translation_step_grid_fraction * cell_size
        )
        rotation_limit = np.deg2rad(self._config.max_rotation_step_deg)
        radius = max(descriptor.proxy_radius_m, cell_size)
        arc_limit = translation_limit / radius
        effective_rotation_limit = min(rotation_limit, arc_limit)
        translation_intervals = int(
            np.ceil(translation_distance / translation_limit)
        )
        rotation_intervals = int(
            np.ceil(angular_displacement / effective_rotation_limit)
        )
        interval_count = max(1, translation_intervals, rotation_intervals)

        mask = np.zeros(grid.shape, dtype=bool)
        cut_surface = np.full(grid.shape, np.inf, dtype=np.float64)
        previous_quaternion = self._rotation_to_quaternion(previous_rotation)
        current_quaternion = self._rotation_to_quaternion(current_rotation)
        poses: list[np.ndarray] = []
        local_quad = self._bottom_quad(descriptor)
        for fraction in np.linspace(0.0, 1.0, interval_count + 1):
            quaternion = self._slerp(
                previous_quaternion, current_quaternion, float(fraction)
            )
            rotation = self._quaternion_to_rotation(quaternion)
            stretch = (
                (1.0 - fraction) * previous_stretch
                + fraction * current_stretch
            )
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation @ stretch
            pose[:3, 3] = (
                (1.0 - fraction) * previous_pose[:3, 3]
                + fraction * current_pose[:3, 3]
            )
            poses.append(pose)
            quad = self._transform_points(pose, local_quad)
            self._rasterize_triangle(
                quad[[0, 1, 2]], grid, mask, cut_surface
            )
            self._rasterize_triangle(
                quad[[0, 2, 3]], grid, mask, cut_surface
            )

        if grid.valid_mask is not None:
            mask &= grid.valid_mask
            cut_surface[~mask] = np.inf
        bbox = self._bbox(mask)
        return SweepResult(
            affected_bbox_grid=bbox,
            affected_mask=mask,
            cut_surface=cut_surface,
            sampled_poses=tuple(poses),
            diagnostics={
                "translation_distance_m": translation_distance,
                "angular_displacement_rad": angular_displacement,
                "angular_displacement_deg": float(
                    np.rad2deg(angular_displacement)
                ),
                "tool_radius_m": radius,
                "translation_step_limit_m": translation_limit,
                "rotation_step_limit_rad": effective_rotation_limit,
                "translation_interval_count": translation_intervals,
                "rotation_interval_count": rotation_intervals,
                "interval_count": interval_count,
                "sampled_pose_count": len(poses),
                "affected_cell_count": int(mask.sum()),
            },
        )

    @staticmethod
    def _bottom_quad(descriptor: ToolDescriptor) -> np.ndarray:
        # Proxy contract: boundary first point is the rear bottom and last point
        # is the cutting edge. This is true for parameter, marker and file modes.
        rear_left = descriptor.left_boundary_local[0]
        rear_right = descriptor.right_boundary_local[0]
        edge_left = descriptor.cutting_edge_local[0]
        edge_right = descriptor.cutting_edge_local[-1]
        return np.ascontiguousarray(
            np.stack((rear_left, rear_right, edge_right, edge_left))
        )

    @staticmethod
    def _validate_pose(value: np.ndarray, name: str) -> np.ndarray:
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError(
                f"[ContinuousSweepBuilder] {name} must be finite (4,4); "
                f"shape={pose.shape}"
            )
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10):
            raise ValueError(
                f"[ContinuousSweepBuilder] {name} must be affine column-vector form"
            )
        if abs(float(np.linalg.det(pose[:3, :3]))) < 1e-12:
            raise ValueError(f"[ContinuousSweepBuilder] {name} is singular")
        return pose

    @staticmethod
    def _transform_points(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
        homogeneous = np.column_stack(
            (np.asarray(points, dtype=np.float64), np.ones(len(points)))
        )
        transformed = (pose @ homogeneous.T).T
        return np.ascontiguousarray(transformed[:, :3] / transformed[:, 3, None])

    @classmethod
    def _rasterize_triangle(
        cls,
        triangle: np.ndarray,
        grid: TerrainGrid,
        mask: np.ndarray,
        cut_surface: np.ndarray,
    ) -> None:
        x1, y1, z1 = triangle[0]
        x2, y2, z2 = triangle[1]
        x3, y3, z3 = triangle[2]
        denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(float(denominator)) < 1e-12:
            cls._rasterize_segment(triangle[0], triangle[1], grid, mask, cut_surface)
            cls._rasterize_segment(triangle[1], triangle[2], grid, mask, cut_surface)
            cls._rasterize_segment(triangle[2], triangle[0], grid, mask, cut_surface)
            return

        row_low = max(
            0, int(np.floor((min(y1, y2, y3) - grid.origin_y) / grid.dy)) - 1
        )
        row_high = min(
            grid.ny,
            int(np.ceil((max(y1, y2, y3) - grid.origin_y) / grid.dy)) + 2,
        )
        column_low = max(
            0, int(np.floor((min(x1, x2, x3) - grid.origin_x) / grid.dx)) - 1
        )
        column_high = min(
            grid.nx,
            int(np.ceil((max(x1, x2, x3) - grid.origin_x) / grid.dx)) + 2,
        )
        if row_low >= row_high or column_low >= column_high:
            return
        rows = np.arange(row_low, row_high, dtype=np.float64)
        columns = np.arange(column_low, column_high, dtype=np.float64)
        yy = grid.origin_y + rows[:, None] * grid.dy
        xx = grid.origin_x + columns[None, :] * grid.dx
        weight1 = ((y2 - y3) * (xx - x3) + (x3 - x2) * (yy - y3)) / denominator
        weight2 = ((y3 - y1) * (xx - x3) + (x1 - x3) * (yy - y3)) / denominator
        weight3 = 1.0 - weight1 - weight2
        inside = (
            (weight1 >= -1e-9)
            & (weight2 >= -1e-9)
            & (weight3 >= -1e-9)
        )
        if not np.any(inside):
            return
        z_values = weight1 * z1 + weight2 * z2 + weight3 * z3
        region_mask = mask[row_low:row_high, column_low:column_high]
        region_surface = cut_surface[row_low:row_high, column_low:column_high]
        region_mask |= inside
        np.minimum(region_surface, np.where(inside, z_values, np.inf), out=region_surface)

    @staticmethod
    def _rasterize_segment(
        start: np.ndarray,
        end: np.ndarray,
        grid: TerrainGrid,
        mask: np.ndarray,
        cut_surface: np.ndarray,
    ) -> None:
        delta = np.asarray(end) - np.asarray(start)
        projected_length = float(np.hypot(delta[0], delta[1]))
        sample_count = max(
            2, int(np.ceil(projected_length / (0.25 * min(grid.dx, grid.dy)))) + 1
        )
        fractions = np.linspace(0.0, 1.0, sample_count)
        points = start[None, :] + fractions[:, None] * delta[None, :]
        columns = np.rint((points[:, 0] - grid.origin_x) / grid.dx).astype(int)
        rows = np.rint((points[:, 1] - grid.origin_y) / grid.dy).astype(int)
        inside = (
            (rows >= 0)
            & (rows < grid.ny)
            & (columns >= 0)
            & (columns < grid.nx)
        )
        for row, column, height in zip(rows[inside], columns[inside], points[inside, 2]):
            mask[row, column] = True
            cut_surface[row, column] = min(cut_surface[row, column], height)

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        indices = np.argwhere(mask)
        if not len(indices):
            return (0, 0, 0, 0)
        low = indices.min(axis=0)
        high = indices.max(axis=0) + 1
        return int(low[0]), int(low[1]), int(high[0]), int(high[1])

    @staticmethod
    def _polar(linear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u, singular, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0.0:
            u[:, -1] *= -1.0
            singular[-1] *= -1.0
            rotation = u @ vt
        stretch = vt.T @ np.diag(singular) @ vt
        return rotation, stretch

    @staticmethod
    def _rotation_angle(rotation: np.ndarray) -> float:
        cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
        return float(np.arccos(cosine))

    @staticmethod
    def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[2, 1] - rotation[1, 2]) / scale,
                        0.25 * scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                    ]
                )
            elif index == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[0, 2] - rotation[2, 0]) / scale,
                        (rotation[0, 1] + rotation[1, 0]) / scale,
                        0.25 * scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                    ]
                )
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                quaternion = np.asarray(
                    [
                        (rotation[1, 0] - rotation[0, 1]) / scale,
                        (rotation[0, 2] + rotation[2, 0]) / scale,
                        (rotation[1, 2] + rotation[2, 1]) / scale,
                        0.25 * scale,
                    ]
                )
        return quaternion / np.linalg.norm(quaternion)

    @staticmethod
    def _quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion / np.linalg.norm(quaternion)
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
        finish = np.asarray(end, dtype=np.float64)
        initial = np.asarray(start, dtype=np.float64)
        dot = float(np.dot(initial, finish))
        if dot < 0.0:
            finish = -finish
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            result = initial + fraction * (finish - initial)
            return result / np.linalg.norm(result)
        angle = float(np.arccos(dot))
        sine = float(np.sin(angle))
        result = (
            np.sin((1.0 - fraction) * angle) / sine * initial
            + np.sin(fraction * angle) / sine * finish
        )
        return result / np.linalg.norm(result)
