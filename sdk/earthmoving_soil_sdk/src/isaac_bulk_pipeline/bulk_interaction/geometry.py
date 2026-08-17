"""Convert a legacy geometric sweep into a non-mutating interaction candidate."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..bulk_state import TerrainVolumeIntegrator
from ..interaction.continuous_sweep import SweepResult
from ..terrain.terrain_grid import TerrainGrid
from ..tools.tool_kinematics_adapter import ToolState


def _readonly(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ToolTerrainIntersection:
    """Candidate disturbed geometry; this object never changes terrain state."""

    affected_mask: np.ndarray
    penetration_depth_m: np.ndarray
    cutting_surface_m: np.ndarray
    bucket_velocity_terrain_m_s: np.ndarray
    cutting_edge_velocity_terrain_m_s: np.ndarray
    local_terrain_normal: np.ndarray
    local_slope_rad: float
    candidate_intersection_volume_m3: float
    affected_bbox_grid: tuple[int, int, int, int]
    cutting_edge_points_terrain_m: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float64)
    )
    separation_plane_direction_terrain: np.ndarray = field(
        default_factory=lambda: np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    )
    separation_plane_source: str = "TOOL_PLUS_Z_LEGACY_FALLBACK"
    tool_plus_z_separation_fallback_used: bool = True

    def __post_init__(self) -> None:
        mask = np.asarray(self.affected_mask, dtype=bool)
        depth = np.asarray(self.penetration_depth_m, dtype=np.float64)
        surface = np.asarray(self.cutting_surface_m, dtype=np.float64)
        if mask.ndim != 2 or depth.shape != mask.shape or surface.shape != mask.shape:
            raise ValueError("[ToolTerrainIntersection] field shapes must match H[y,x]")
        if not np.all(np.isfinite(depth)) or np.any(depth < 0.0):
            raise ValueError("[ToolTerrainIntersection] penetration must be finite/non-negative")
        if np.any(mask) and not np.all(np.isfinite(surface[mask])):
            raise ValueError("[ToolTerrainIntersection] affected surface must be finite")
        for name in (
            "bucket_velocity_terrain_m_s",
            "cutting_edge_velocity_terrain_m_s",
            "local_terrain_normal",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"[ToolTerrainIntersection] {name} must be finite shape (3,)")
            object.__setattr__(self, name, _readonly(value, dtype=np.float64))
        edge = np.asarray(self.cutting_edge_points_terrain_m, dtype=np.float64)
        if edge.ndim != 2 or edge.shape[1] != 3 or not np.all(np.isfinite(edge)):
            raise ValueError(
                "[ToolTerrainIntersection] cutting edge must be finite shape (N,3)"
            )
        separation = np.asarray(
            self.separation_plane_direction_terrain,
            dtype=np.float64,
        )
        if separation.shape != (3,) or not np.all(np.isfinite(separation)):
            raise ValueError(
                "[ToolTerrainIntersection] separation plane direction must be finite (3,)"
            )
        separation_norm = float(np.linalg.norm(separation))
        if separation_norm <= 1e-12:
            raise ValueError(
                "[ToolTerrainIntersection] separation plane direction must be nonzero"
            )
        if not np.isclose(np.linalg.norm(self.local_terrain_normal), 1.0, atol=1e-6):
            raise ValueError("[ToolTerrainIntersection] local normal must be unit length")
        if not np.isfinite(self.local_slope_rad) or self.local_slope_rad < 0.0:
            raise ValueError("[ToolTerrainIntersection] local slope must be finite/non-negative")
        if not np.isfinite(self.candidate_intersection_volume_m3) or self.candidate_intersection_volume_m3 < 0.0:
            raise ValueError("[ToolTerrainIntersection] candidate volume must be finite/non-negative")
        if not isinstance(self.separation_plane_source, str) or not self.separation_plane_source:
            raise ValueError("[ToolTerrainIntersection] separation source must be non-empty")
        if self.tool_plus_z_separation_fallback_used and self.separation_plane_source != "TOOL_PLUS_Z_LEGACY_FALLBACK":
            raise ValueError("[ToolTerrainIntersection] fallback flag/source disagree")
        object.__setattr__(self, "affected_mask", _readonly(mask, dtype=bool))
        object.__setattr__(self, "penetration_depth_m", _readonly(depth, dtype=np.float64))
        object.__setattr__(self, "cutting_surface_m", _readonly(surface, dtype=np.float64))
        object.__setattr__(
            self,
            "cutting_edge_points_terrain_m",
            _readonly(edge, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "separation_plane_direction_terrain",
            _readonly(separation / separation_norm, dtype=np.float64),
        )


class ToolTerrainIntersectionModel:
    """Interpret sweep geometry without treating it as removal or payload."""

    def __init__(self, minimum_penetration_m: float = 0.002) -> None:
        value = float(minimum_penetration_m)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("minimum_penetration_m must be finite/non-negative")
        self.minimum_penetration_m = value

    def compute(
        self,
        H_resting_m: np.ndarray,
        sweep: SweepResult,
        tool_state: ToolState,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> ToolTerrainIntersection:
        height = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        if integrator.shape != grid.shape:
            raise ValueError("[ToolTerrainIntersection] integrator/grid shape mismatch")
        candidate = sweep.affected_mask & np.isfinite(sweep.cut_surface)
        raw_depth = np.where(candidate, height - sweep.cut_surface, 0.0)
        depth = np.where(raw_depth >= self.minimum_penetration_m, raw_depth, 0.0)
        mask = depth > 0.0
        if grid.valid_mask is not None:
            mask &= grid.valid_mask
            depth = np.where(mask, depth, 0.0)

        gy, gx = np.gradient(height, grid.dy, grid.dx, edge_order=1)
        if np.any(mask):
            weights = integrator.vertex_weights_m2 * mask
            total = float(weights.sum())
            mean_gx = float(np.sum(gx * weights) / total)
            mean_gy = float(np.sum(gy * weights) / total)
        else:
            mean_gx = mean_gy = 0.0
        normal = np.asarray([-mean_gx, -mean_gy, 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        slope = float(np.arctan(np.hypot(mean_gx, mean_gy)))
        velocity = np.asarray(tool_state.linear_velocity, dtype=np.float64)
        edge_center = np.mean(tool_state.cutting_edge_terrain, axis=0)
        omega = np.asarray(tool_state.angular_velocity, dtype=np.float64)
        radius = edge_center - tool_state.pose_terrain[:3, 3]
        edge_velocity = velocity + np.cross(omega, radius)
        bbox = self._bbox(mask)
        return ToolTerrainIntersection(
            affected_mask=mask,
            penetration_depth_m=depth,
            cutting_surface_m=sweep.cut_surface,
            bucket_velocity_terrain_m_s=velocity,
            cutting_edge_velocity_terrain_m_s=edge_velocity,
            local_terrain_normal=normal,
            local_slope_rad=slope,
            candidate_intersection_volume_m3=integrator.integrate(depth),
            affected_bbox_grid=bbox,
            cutting_edge_points_terrain_m=tool_state.cutting_edge_terrain,
            separation_plane_direction_terrain=(
                tool_state.pose_terrain[:3, 2]
                if tool_state.separation_plane_direction_terrain is None
                else tool_state.separation_plane_direction_terrain
            ),
            separation_plane_source=(
                "TOOL_PLUS_Z_LEGACY_FALLBACK"
                if tool_state.separation_plane_direction_terrain is None
                else tool_state.separation_plane_source
            ),
            tool_plus_z_separation_fallback_used=(
                True
                if tool_state.separation_plane_direction_terrain is None
                else tool_state.tool_plus_z_separation_fallback_used
            ),
        )

    @staticmethod
    def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
        indices = np.argwhere(mask)
        if not len(indices):
            return (0, 0, 0, 0)
        low = indices.min(axis=0)
        high = indices.max(axis=0) + 1
        return int(low[0]), int(low[1]), int(high[0]), int(high[1])
