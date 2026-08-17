"""Uncalibrated mechanism-trend sensitivity checks using real failure geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_interaction import FailureZoneModel, ToolTerrainIntersection
from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..soil_force import SoilForceModel
from ..terrain import TerrainGrid
from ..tools import ToolDescriptor, ToolState


@dataclass(frozen=True)
class SensitivitySample:
    variable: str
    value: float
    resultant_force_n: float


class SoilForceSensitivityStudy:
    def run(self) -> dict[str, tuple[SensitivitySample, ...]]:
        result = {}
        result["penetration_depth_m"] = tuple(self._sample("penetration_depth_m", value) for value in (0.2, 0.4, 0.6))
        result["cohesion_proxy_pa"] = tuple(self._sample("cohesion_proxy_pa", value) for value in (0.0, 2000.0, 5000.0))
        result["internal_friction_angle_deg"] = tuple(self._sample("internal_friction_angle_deg", value) for value in (28.0, 34.0, 40.0))
        return result

    def _sample(self, variable: str, value: float) -> SensitivitySample:
        depth = value if variable == "penetration_depth_m" else 0.4
        material = MaterialScenario(
            name="sensitivity_UNCALIBRATED",
            assumed_bulk_density_kg_m3=2100.0,
            internal_friction_angle_deg=value if variable == "internal_friction_angle_deg" else 34.0,
            cohesion_proxy_pa=value if variable == "cohesion_proxy_pa" else 1500.0,
            tool_friction_coefficient=0.45,
            start_angle_deg=38.0,
            stop_angle_deg=30.0,
            mobile_friction_coefficient=0.55,
        )
        grid = TerrainGrid(81, 81, 0.05, 0.05, -2.0, -2.0, "/World/Terrain")
        integrator = TerrainVolumeIntegrator.from_grid(grid)
        rows, columns = np.indices(grid.shape)
        x = grid.origin_x + columns * grid.dx
        y = grid.origin_y + rows * grid.dy
        mask = (np.abs(x) <= 0.026) & (np.abs(y) <= 1.30)
        penetration = np.where(mask, depth, 0.0)
        surface = np.where(mask, 1.0 - depth, np.inf)
        intersection = ToolTerrainIntersection(
            affected_mask=mask,
            penetration_depth_m=penetration,
            cutting_surface_m=surface,
            bucket_velocity_terrain_m_s=np.asarray([1.0, 0.0, 0.0]),
            cutting_edge_velocity_terrain_m_s=np.asarray([1.0, 0.0, 0.0]),
            local_terrain_normal=np.asarray([0.0, 0.0, 1.0]),
            local_slope_rad=0.0,
            candidate_intersection_volume_m3=integrator.integrate(penetration),
            affected_bbox_grid=(39, 14, 42, 67),
            cutting_edge_points_terrain_m=np.asarray([[0.0, -1.35, 1.0 - depth], [0.0, 1.35, 1.0 - depth]]),
            separation_plane_direction_terrain=np.asarray([0.0, 0.0, 1.0]),
        )
        failure = FailureZoneModel().compute(
            intersection,
            np.ones(grid.shape),
            material,
            grid,
            integrator,
        )
        descriptor = ToolDescriptor(
            "bucket",
            "/World/Bucket/ToolFrame",
            np.asarray([[-1.35, 0, 0], [1.35, 0, 0]]),
            np.asarray([[0, -1, 0], [0, 0, 0]]),
            np.asarray([[-1.35, -1, 0], [-1.35, 0, 0]]),
            np.asarray([[1.35, -1, 0], [1.35, 0, 0]]),
            2.7,
        )
        pose = np.eye(4)
        tool = ToolState(
            0.0,
            pose,
            pose,
            descriptor.cutting_edge_local,
            descriptor.bottom_profile_local,
            descriptor.left_boundary_local,
            descriptor.right_boundary_local,
            np.asarray([1.0, 0.0, 0.0]),
            np.zeros(3),
        )
        force = SoilForceModel().compute(failure, intersection, material, descriptor, tool)
        return SensitivitySample(variable, float(value), force.resultant_force_n)

    @staticmethod
    def monotonic(samples: tuple[SensitivitySample, ...]) -> bool:
        return all(right.resultant_force_n > left.resultant_force_n for left, right in zip(samples, samples[1:]))
