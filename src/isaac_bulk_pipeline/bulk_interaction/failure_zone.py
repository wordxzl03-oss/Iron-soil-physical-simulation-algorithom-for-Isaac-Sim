"""Unified quasi-static failure-wedge geometry and conservative activation.

The analytical strip triangle, its Heightmap rasterization and the geometry
consumed by :mod:`soil_force` are intentionally represented by the same
``FailureStripGeometry`` object.  The mechanics and deviations from the cited
models are traced in ``docs/FAILURE_ZONE_THEORY_TRACE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..terrain.terrain_grid import TerrainGrid
from .geometry import ToolTerrainIntersection


_EPS = 1.0e-12


def _ro(value: np.ndarray, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FailureZoneConfig:
    """Numerical controls for the reduced-order critical-wedge solve."""

    strip_width_m: float = 0.20
    minimum_failure_angle_deg: float = 2.0
    maximum_failure_angle_deg: float = 88.0
    failure_angle_samples: int = 181
    solver_tolerance_deg: float = 1.0e-4
    maximum_solver_iterations: int = 80
    maximum_wedge_length_m: float = 4.0
    mobilization_depth_gain: float = 1.0
    surcharge_pa: float = 0.0
    gravity_m_s2: float = 9.81
    surface_version: str = "V3_CONTINUOUS_2P5D"
    surface_sample_spacing_m: float = 0.05
    lateral_transition_width_m: float = 0.20
    lateral_smoothing_passes: int = 2
    lateral_continuity_weight: float = 0.35

    def __post_init__(self) -> None:
        positive = np.asarray(
            [
                self.strip_width_m,
                self.maximum_wedge_length_m,
                self.solver_tolerance_deg,
                self.gravity_m_s2,
                self.surface_sample_spacing_m,
                self.lateral_transition_width_m,
            ],
            dtype=np.float64,
        )
        bounds = np.asarray(
            [self.minimum_failure_angle_deg, self.maximum_failure_angle_deg],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError("[FailureZone] positive configuration values are invalid")
        if not np.all(np.isfinite(bounds)) or not (
            0.0 < bounds[0] < bounds[1] < 90.0
        ):
            raise ValueError("[FailureZone] angle bounds must satisfy 0 < min < max < 90")
        if not np.isfinite(self.surcharge_pa) or self.surcharge_pa < 0.0:
            raise ValueError("[FailureZone] surcharge_pa must be finite/non-negative")
        if not isinstance(self.failure_angle_samples, int) or self.failure_angle_samples < 5:
            raise ValueError("[FailureZone] failure_angle_samples must be integer >= 5")
        if not isinstance(self.maximum_solver_iterations, int) or self.maximum_solver_iterations < 1:
            raise ValueError("[FailureZone] maximum_solver_iterations must be positive")
        if self.surface_version not in {"V3_CONTINUOUS_2P5D", "LEGACY_STRIPS"}:
            raise ValueError("[FailureZone] surface_version must be V3_CONTINUOUS_2P5D/LEGACY_STRIPS")
        if not isinstance(self.lateral_smoothing_passes, int) or self.lateral_smoothing_passes < 0:
            raise ValueError("[FailureZone] lateral_smoothing_passes must be non-negative")
        if not np.isfinite(self.lateral_continuity_weight) or not 0.0 <= self.lateral_continuity_weight <= 1.0:
            raise ValueError("[FailureZone] lateral_continuity_weight must lie in [0,1]")
        if not np.isclose(self.mobilization_depth_gain, 1.0, atol=0.0, rtol=0.0):
            raise ValueError(
                "[FailureZone] mobilization_depth_gain is retained for config compatibility "
                "but must equal 1.0; scaling activation would create a second wedge"
            )


@dataclass(frozen=True)
class FailureStripGeometry:
    """One authoritative analytical wedge plus its conservative raster map."""

    strip_index: int
    lateral_bounds_m: np.ndarray
    lateral_center_m: float
    width_m: float
    cutting_edge_terrain_m: np.ndarray
    effective_cutting_edge_terrain_m: np.ndarray
    local_terrain_elevation_m: float
    local_terrain_slope_rad: float
    penetration_depth_m: float
    rake_angle_deg: float
    failure_angle_deg: float
    longitudinal_extent_m: float
    failure_plane_length_m: float
    cross_section_vertices_terrain_m: np.ndarray
    cross_section_area_m2: float
    wedge_volume_m3: float
    centroid_terrain_m: np.ndarray
    internal_friction_angle_deg: float
    soil_tool_friction_angle_deg: float
    cohesion_pa: float
    assumed_bulk_density_kg_m3: float
    surcharge_pa: float
    gravity_m_s2: float
    luengo_Nw: float
    luengo_Nc: float
    luengo_Nq: float
    weight_resistance_n: float
    cohesion_resistance_n: float
    surcharge_resistance_n: float
    estimated_resistance_n: float
    solver_status: str
    solver_iterations: int
    failure_angle_boundary_hit: bool
    raster_indices_yx: np.ndarray
    raster_coverage_fractions: np.ndarray
    raster_requested_volume_m3: np.ndarray
    raster_activated_volume_m3: np.ndarray
    boundary_clipped: bool
    availability_clipped: bool

    def __post_init__(self) -> None:
        scalars = np.asarray(
            [
                self.lateral_center_m,
                self.width_m,
                self.local_terrain_elevation_m,
                self.local_terrain_slope_rad,
                self.penetration_depth_m,
                self.rake_angle_deg,
                self.failure_angle_deg,
                self.longitudinal_extent_m,
                self.failure_plane_length_m,
                self.cross_section_area_m2,
                self.wedge_volume_m3,
                self.internal_friction_angle_deg,
                self.soil_tool_friction_angle_deg,
                self.cohesion_pa,
                self.assumed_bulk_density_kg_m3,
                self.surcharge_pa,
                self.gravity_m_s2,
                self.luengo_Nw,
                self.luengo_Nc,
                self.luengo_Nq,
                self.weight_resistance_n,
                self.cohesion_resistance_n,
                self.surcharge_resistance_n,
                self.estimated_resistance_n,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("[FailureZone] strip geometry contains NaN/Inf")
        nonnegative = np.asarray(
            [
                self.width_m,
                self.penetration_depth_m,
                self.longitudinal_extent_m,
                self.failure_plane_length_m,
                self.cross_section_area_m2,
                self.wedge_volume_m3,
                self.cohesion_pa,
                self.assumed_bulk_density_kg_m3,
                self.surcharge_pa,
                self.gravity_m_s2,
                self.weight_resistance_n,
                self.cohesion_resistance_n,
                self.surcharge_resistance_n,
                self.estimated_resistance_n,
            ]
        )
        if np.any(nonnegative < 0.0):
            raise ValueError("[FailureZone] strip geometry has a negative physical measure")
        if self.width_m <= 0.0 or self.penetration_depth_m <= 0.0:
            raise ValueError("[FailureZone] non-empty strip width/depth must be positive")
        bounds = np.asarray(self.lateral_bounds_m, dtype=np.float64)
        cutting = np.asarray(self.cutting_edge_terrain_m, dtype=np.float64)
        effective = np.asarray(self.effective_cutting_edge_terrain_m, dtype=np.float64)
        vertices = np.asarray(self.cross_section_vertices_terrain_m, dtype=np.float64)
        centroid = np.asarray(self.centroid_terrain_m, dtype=np.float64)
        indices = np.asarray(self.raster_indices_yx, dtype=np.int64)
        coverage = np.asarray(self.raster_coverage_fractions, dtype=np.float64)
        requested = np.asarray(self.raster_requested_volume_m3, dtype=np.float64)
        activated = np.asarray(self.raster_activated_volume_m3, dtype=np.float64)
        if bounds.shape != (2,) or bounds[1] <= bounds[0]:
            raise ValueError("[FailureZone] lateral bounds are invalid")
        if cutting.shape != (3,) or effective.shape != (3,) or centroid.shape != (3,):
            raise ValueError("[FailureZone] strip points must have shape (3,)")
        if vertices.shape != (3, 3) or not np.all(np.isfinite(vertices)):
            raise ValueError("[FailureZone] cross section must be a finite triangle")
        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("[FailureZone] raster indices must have shape (N,2)")
        if coverage.shape != (len(indices),) or requested.shape != coverage.shape or activated.shape != coverage.shape:
            raise ValueError("[FailureZone] raster arrays must share length N")
        if (
            not np.all(np.isfinite(np.r_[bounds, cutting, effective, centroid, coverage, requested, activated]))
            or np.any(coverage < -_EPS)
            or np.any(coverage > 1.0 + _EPS)
            or np.any(requested < -_EPS)
            or np.any(activated < -_EPS)
            or np.any(activated > requested + 1.0e-10)
        ):
            raise ValueError("[FailureZone] invalid raster coverage/volume")
        if not isinstance(self.solver_status, str) or not self.solver_status:
            raise ValueError("[FailureZone] solver status must be non-empty")
        if self.solver_iterations < 0:
            raise ValueError("[FailureZone] solver iterations must be non-negative")
        object.__setattr__(self, "lateral_bounds_m", _ro(bounds))
        object.__setattr__(self, "cutting_edge_terrain_m", _ro(cutting))
        object.__setattr__(self, "effective_cutting_edge_terrain_m", _ro(effective))
        object.__setattr__(self, "cross_section_vertices_terrain_m", _ro(vertices))
        object.__setattr__(self, "centroid_terrain_m", _ro(centroid))
        object.__setattr__(self, "raster_indices_yx", _ro(indices, np.int64))
        object.__setattr__(self, "raster_coverage_fractions", _ro(np.clip(coverage, 0.0, 1.0)))
        object.__setattr__(self, "raster_requested_volume_m3", _ro(np.maximum(requested, 0.0)))
        object.__setattr__(self, "raster_activated_volume_m3", _ro(np.maximum(activated, 0.0)))

    @property
    def wedge_length_m(self) -> float:
        """Backward-compatible name for longitudinal extent."""

        return self.longitudinal_extent_m

    @property
    def rasterized_requested_volume_m3(self) -> float:
        return float(np.sum(self.raster_requested_volume_m3, dtype=np.float64))

    @property
    def rasterized_activated_volume_m3(self) -> float:
        return float(np.sum(self.raster_activated_volume_m3, dtype=np.float64))

    @property
    def activation_ratio(self) -> float:
        if self.wedge_volume_m3 <= _EPS:
            return 0.0
        return float(np.clip(self.rasterized_activated_volume_m3 / self.wedge_volume_m3, 0.0, 1.0))


# Compatibility import for downstream code written against the Phase-F name.
FailureStripResult = FailureStripGeometry


@dataclass(frozen=True)
class FailureZone:
    active_mask: np.ndarray
    active_thickness_m: np.ndarray
    active_volume_m3: float
    centroid_terrain_m: np.ndarray
    estimated_failure_angle_deg: float
    approach_direction_xy: np.ndarray
    strip_geometries: tuple[FailureStripGeometry, ...]
    estimated_total_resistance_n: float
    analytical_wedge_volume_m3: float
    rasterized_requested_volume_m3: float
    relative_volume_error: float
    failure_angle_boundary_hit_rate: float
    candidate_strip_count: int = 0
    excluded_strip_count: int = 0
    applicability_status: str = "NO_INTERSECTION"
    exclusion_diagnostics: tuple[str, ...] = ()
    failure_surface_profile: "FailureSurfaceProfile | None" = None
    model_classification: str = "PAPER_DIRECT_FEE_PLUS_ENGINEERING_CLOSURE"

    def __post_init__(self) -> None:
        mask = np.asarray(self.active_mask, dtype=bool)
        thickness = np.asarray(self.active_thickness_m, dtype=np.float64)
        if mask.ndim != 2 or thickness.shape != mask.shape:
            raise ValueError("[FailureZone] active fields must have matching H[y,x] shape")
        if not np.all(np.isfinite(thickness)) or np.any(thickness < 0.0):
            raise ValueError("[FailureZone] active thickness must be finite/non-negative")
        if not np.array_equal(mask, thickness > 0.0):
            raise ValueError("[FailureZone] mask must exactly match positive thickness")
        centroid = np.asarray(self.centroid_terrain_m, dtype=np.float64)
        direction = np.asarray(self.approach_direction_xy, dtype=np.float64)
        if centroid.shape != (3,) or direction.shape != (2,) or not np.all(np.isfinite(np.r_[centroid, direction])):
            raise ValueError("[FailureZone] centroid/direction shape invalid")
        if not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6):
            raise ValueError("[FailureZone] approach direction must be unit length")
        for name in (
            "active_volume_m3",
            "estimated_failure_angle_deg",
            "estimated_total_resistance_n",
            "analytical_wedge_volume_m3",
            "rasterized_requested_volume_m3",
            "relative_volume_error",
            "failure_angle_boundary_hit_rate",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"[FailureZone] {name} must be finite/non-negative")
        if self.failure_angle_boundary_hit_rate > 1.0 + _EPS:
            raise ValueError("[FailureZone] boundary-hit rate must be <= 1")
        if (
            self.candidate_strip_count < 0
            or self.excluded_strip_count < 0
            or self.excluded_strip_count > self.candidate_strip_count
        ):
            raise ValueError("[FailureZone] candidate/excluded strip counts are invalid")
        if not self.applicability_status:
            raise ValueError("[FailureZone] applicability status must be non-empty")
        object.__setattr__(self, "active_mask", _ro(mask, bool))
        object.__setattr__(self, "active_thickness_m", _ro(thickness))
        object.__setattr__(self, "centroid_terrain_m", _ro(centroid))
        object.__setattr__(self, "approach_direction_xy", _ro(direction))
        object.__setattr__(self, "strip_geometries", tuple(self.strip_geometries))
        object.__setattr__(self, "exclusion_diagnostics", tuple(self.exclusion_diagnostics))

    @property
    def strip_results(self) -> tuple[FailureStripGeometry, ...]:
        """Compatibility view of the geometry shared with SoilForce."""

        return self.strip_geometries


@dataclass(frozen=True)
class FailureSurfaceProfile:
    """Continuous lateral state sampled along cutting-edge arc length.

    ``beta0`` is the pointwise minimum-FEE plane.  ``beta`` is the mild
    continuity-closed plane used by rasterization.  The closure is explicitly
    an engineering reduction and is not attributed to Luengo/Reece.
    """

    arc_length_m: np.ndarray
    penetration_depth_m: np.ndarray
    terrain_slope_rad: np.ndarray
    rake_angle_rad: np.ndarray
    beta0_rad: np.ndarray
    beta_rad: np.ndarray
    runout_length_m: np.ndarray
    lateral_transition_weight: np.ndarray
    classification: str = "REDUCED_ORDER_ENGINEERING_CLOSURE"

    def __post_init__(self) -> None:
        arrays = [
            np.asarray(getattr(self, name), dtype=np.float64)
            for name in (
                "arc_length_m", "penetration_depth_m", "terrain_slope_rad",
                "rake_angle_rad", "beta0_rad", "beta_rad", "runout_length_m",
                "lateral_transition_weight",
            )
        ]
        if not arrays or arrays[0].ndim != 1 or any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("[FailureSurfaceV3] profile arrays must share shape (N,)")
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("[FailureSurfaceV3] profile contains NaN/Inf")
        if np.any(arrays[1] < 0.0) or np.any(arrays[6] < 0.0):
            raise ValueError("[FailureSurfaceV3] depth/runout must be non-negative")
        if np.any(arrays[7] < 0.0) or np.any(arrays[7] > 1.0):
            raise ValueError("[FailureSurfaceV3] transition weights must lie in [0,1]")
        for name, value in zip(
            (
                "arc_length_m", "penetration_depth_m", "terrain_slope_rad",
                "rake_angle_rad", "beta0_rad", "beta_rad", "runout_length_m",
                "lateral_transition_weight",
            ),
            arrays,
        ):
            object.__setattr__(self, name, _ro(value))

@dataclass(frozen=True)
class _SolvedWedge:
    beta_rad: float
    length_m: float
    plane_length_m: float
    area_m2: float
    volume_m3: float
    Nw: float
    Nc: float
    Nq: float
    weight_n: float
    cohesion_n: float
    surcharge_n: float
    resistance_n: float
    status: str
    iterations: int
    boundary_hit: bool


@dataclass(frozen=True)
class _RasterContribution:
    indices_yx: np.ndarray
    coverage: np.ndarray
    requested_m3: np.ndarray


class _NoAdmissibleWedge(ValueError):
    """The published FEE closure has no finite solution for this strip."""


class FailureZoneModel:
    """Solve one FEE-equilibrium wedge and rasterize that exact wedge per strip."""

    def __init__(self, config: FailureZoneConfig | None = None) -> None:
        self.config = config or FailureZoneConfig()

    def compute(
        self,
        intersection: ToolTerrainIntersection,
        H_resting_m: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        fallback_approach_direction_xy: np.ndarray = np.asarray([1.0, 0.0]),
        H_free_m: np.ndarray | None = None,
    ) -> FailureZone:
        if self.config.surface_version == "LEGACY_STRIPS":
            return self._compute_legacy(
                intersection, H_resting_m, material, grid, integrator,
                fallback_approach_direction_xy=fallback_approach_direction_xy,
            )
        return self._compute_v3(
            intersection, H_resting_m, material, grid, integrator,
            fallback_approach_direction_xy=fallback_approach_direction_xy,
            H_free_m=H_free_m,
        )

    def _compute_legacy(
        self,
        intersection: ToolTerrainIntersection,
        H_resting_m: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        fallback_approach_direction_xy: np.ndarray,
    ) -> FailureZone:
        height = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        if integrator.shape != grid.shape:
            raise ValueError("[FailureZone] integrator/grid shape mismatch")
        direction = self._approach_direction(intersection, fallback_approach_direction_xy)
        separation = np.asarray(
            intersection.separation_plane_direction_terrain, dtype=np.float64
        )
        # The bucket bottom/mouth separation plate is one-sided.  A cutting
        # edge velocity opposite its local penetration direction is a retreat
        # or back-drag state, not an admissible forward Luengo blade cut.  In
        # that state this primary plate contributes no failure wedge; treating
        # its directed 170-degree angle as a rake would make every Luengo
        # denominator non-positive.
        if float(np.dot(separation[:2], direction)) < -1.0e-8:
            return self._empty_zone(
                grid.shape, direction, applicability_status="REVERSE_SEPARATION_INACTIVE"
            )
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        indices = np.argwhere(intersection.affected_mask)
        if not len(indices):
            return self._empty_zone(grid.shape, direction)

        xy = np.column_stack(
            [
                grid.origin_x + indices[:, 1] * grid.dx,
                grid.origin_y + indices[:, 0] * grid.dy,
            ]
        )
        lateral_coordinate = xy @ lateral
        edge = np.asarray(intersection.cutting_edge_points_terrain_m, dtype=np.float64)
        if len(edge):
            edge_lateral = edge[:, :2] @ lateral
            low = float(np.min(edge_lateral))
            high = float(np.max(edge_lateral))
        else:
            half_cell = 0.5 * min(grid.dx, grid.dy)
            low = float(np.min(lateral_coordinate) - half_cell)
            high = float(np.max(lateral_coordinate) + half_cell)
        if high - low <= _EPS:
            low -= 0.5 * self.config.strip_width_m
            high += 0.5 * self.config.strip_width_m
        strip_count = max(1, int(np.ceil((high - low) / self.config.strip_width_m)))

        gy, gx = np.gradient(height, grid.dy, grid.dx, edge_order=1)
        terrain_grade = gx * direction[0] + gy * direction[1]
        pending: list[tuple[dict[str, object], _RasterContribution]] = []
        candidate_strip_count = 0
        excluded_diagnostics: list[str] = []
        requested_total = np.zeros(grid.shape, dtype=np.float64)
        for strip_id in range(strip_count):
            lower = low + strip_id * self.config.strip_width_m
            upper = min(high, lower + self.config.strip_width_m)
            if upper - lower <= _EPS:
                continue
            if strip_id == strip_count - 1:
                selected_mask = (lateral_coordinate >= lower - _EPS) & (lateral_coordinate <= upper + _EPS)
            else:
                selected_mask = (lateral_coordinate >= lower - _EPS) & (lateral_coordinate < upper - _EPS)
            selected = indices[selected_mask]
            if not len(selected):
                continue
            candidate_strip_count += 1
            depths = intersection.penetration_depth_m[selected[:, 0], selected[:, 1]]
            positive = depths > 0.0
            selected = selected[positive]
            depths = depths[positive]
            if not len(selected):
                continue
            max_depth = float(np.max(depths))
            deepest = selected[np.isclose(depths, max_depth, rtol=1.0e-8, atol=1.0e-10)]
            s_values = (
                (grid.origin_x + deepest[:, 1] * grid.dx) * direction[0]
                + (grid.origin_y + deepest[:, 0] * grid.dy) * direction[1]
            )
            s_cut = float(np.mean(s_values))
            center = 0.5 * (lower + upper)
            cut_xy = direction * s_cut + lateral * center
            terrain_z = float(np.mean(height[deepest[:, 0], deepest[:, 1]]))
            alpha = float(np.arctan(np.mean(terrain_grade[deepest[:, 0], deepest[:, 1]])))
            actual_edge = self._edge_point(edge, center, lateral, cut_xy, terrain_z - max_depth)
            rake = self._rake_angle(intersection.separation_plane_direction_terrain, direction)
            try:
                solved = self._solve_strip(
                    depth=max_depth,
                    width=upper - lower,
                    material=material,
                    local_slope_rad=alpha,
                    rake_angle_rad=rake,
                )
            except _NoAdmissibleWedge as error:
                excluded_diagnostics.append(f"strip={strip_id}: {error}")
                continue
            effective_edge = np.asarray([cut_xy[0], cut_xy[1], terrain_z - max_depth])
            forward_xy = cut_xy + direction * solved.length_m
            forward_z = terrain_z + solved.length_m * np.tan(alpha)
            vertices = np.asarray(
                [
                    effective_edge,
                    [cut_xy[0], cut_xy[1], terrain_z],
                    [forward_xy[0], forward_xy[1], forward_z],
                ],
                dtype=np.float64,
            )
            centroid = np.mean(vertices, axis=0)
            raster = self._rasterize_strip_wedge(
                cut_xy=cut_xy,
                lateral_bounds=(lower, upper),
                direction=direction,
                lateral=lateral,
                depth=max_depth,
                extent=solved.length_m,
                grid=grid,
                integrator=integrator,
            )
            for (row, column), volume in zip(raster.indices_yx, raster.requested_m3):
                requested_total[row, column] += volume
            pending.append(
                (
                    {
                        "strip_index": strip_id,
                        "lateral_bounds_m": np.asarray([lower, upper]),
                        "lateral_center_m": center,
                        "width_m": upper - lower,
                        "cutting_edge_terrain_m": actual_edge,
                        "effective_cutting_edge_terrain_m": effective_edge,
                        "local_terrain_elevation_m": terrain_z,
                        "local_terrain_slope_rad": alpha,
                        "penetration_depth_m": max_depth,
                        "rake_angle_deg": float(np.rad2deg(rake)),
                        "failure_angle_deg": float(np.rad2deg(solved.beta_rad)),
                        "longitudinal_extent_m": solved.length_m,
                        "failure_plane_length_m": solved.plane_length_m,
                        "cross_section_vertices_terrain_m": vertices,
                        "cross_section_area_m2": solved.area_m2,
                        "wedge_volume_m3": solved.volume_m3,
                        "centroid_terrain_m": centroid,
                        "internal_friction_angle_deg": material.internal_friction_angle_deg,
                        "soil_tool_friction_angle_deg": float(np.rad2deg(np.arctan(material.tool_friction_coefficient))),
                        "cohesion_pa": material.cohesion_proxy_pa,
                        "assumed_bulk_density_kg_m3": material.assumed_bulk_density_kg_m3,
                        "surcharge_pa": self.config.surcharge_pa,
                        "gravity_m_s2": self.config.gravity_m_s2,
                        "luengo_Nw": solved.Nw,
                        "luengo_Nc": solved.Nc,
                        "luengo_Nq": solved.Nq,
                        "weight_resistance_n": solved.weight_n,
                        "cohesion_resistance_n": solved.cohesion_n,
                        "surcharge_resistance_n": solved.surcharge_n,
                        "estimated_resistance_n": solved.resistance_n,
                        "solver_status": solved.status,
                        "solver_iterations": solved.iterations,
                        "failure_angle_boundary_hit": solved.boundary_hit,
                    },
                    raster,
                )
            )

        if not pending:
            return self._empty_zone(
                grid.shape,
                direction,
                candidate_strip_count=candidate_strip_count,
                excluded_strip_count=len(excluded_diagnostics),
                applicability_status=(
                    "OUTSIDE_FEE_DOMAIN"
                    if excluded_diagnostics
                    else "NO_POSITIVE_INTERSECTION"
                ),
                exclusion_diagnostics=tuple(excluded_diagnostics),
            )

        weights = np.asarray(integrator.vertex_weights_m2, dtype=np.float64)
        capacity = np.maximum(height, 0.0) * weights
        scale = np.ones(grid.shape, dtype=np.float64)
        occupied = requested_total > _EPS
        scale[occupied] = np.minimum(1.0, capacity[occupied] / requested_total[occupied])
        activated_total = requested_total * scale
        active = np.zeros(grid.shape, dtype=np.float64)
        usable = weights > _EPS
        active[usable] = activated_total[usable] / weights[usable]
        active = np.minimum(active, height)
        active[active < _EPS] = 0.0

        strips: list[FailureStripGeometry] = []
        for values, raster in pending:
            actual = np.asarray(
                [
                    requested * scale[row, column]
                    for (row, column), requested in zip(raster.indices_yx, raster.requested_m3)
                ],
                dtype=np.float64,
            )
            requested_sum = float(np.sum(raster.requested_m3, dtype=np.float64))
            actual_sum = float(np.sum(actual, dtype=np.float64))
            analytical = float(values["wedge_volume_m3"])
            values.update(
                {
                    "raster_indices_yx": raster.indices_yx,
                    "raster_coverage_fractions": raster.coverage,
                    "raster_requested_volume_m3": raster.requested_m3,
                    "raster_activated_volume_m3": actual,
                    "boundary_clipped": requested_sum < analytical - max(1.0e-10, 1.0e-8 * analytical),
                    "availability_clipped": actual_sum < requested_sum - max(1.0e-10, 1.0e-8 * requested_sum),
                }
            )
            strips.append(FailureStripGeometry(**values))

        active_volume = integrator.integrate(active)
        analytical_volume = float(sum(strip.wedge_volume_m3 for strip in strips))
        raster_requested = float(sum(strip.rasterized_requested_volume_m3 for strip in strips))
        denominator = max(analytical_volume, _EPS)
        relative_error = abs(active_volume - analytical_volume) / denominator
        centroid_denominator = sum(strip.wedge_volume_m3 for strip in strips)
        centroid = (
            sum(
                (strip.wedge_volume_m3 * strip.centroid_terrain_m for strip in strips),
                start=np.zeros(3, dtype=np.float64),
            )
            / centroid_denominator
            if centroid_denominator > _EPS
            else np.zeros(3, dtype=np.float64)
        )
        mean_angle = (
            sum(strip.failure_angle_deg * strip.wedge_volume_m3 for strip in strips)
            / centroid_denominator
            if centroid_denominator > _EPS
            else 0.0
        )
        return FailureZone(
            active_mask=active > 0.0,
            active_thickness_m=active,
            active_volume_m3=active_volume,
            centroid_terrain_m=np.asarray(centroid),
            estimated_failure_angle_deg=float(mean_angle),
            approach_direction_xy=direction,
            strip_geometries=tuple(strips),
            estimated_total_resistance_n=float(sum(strip.estimated_resistance_n for strip in strips)),
            analytical_wedge_volume_m3=analytical_volume,
            rasterized_requested_volume_m3=raster_requested,
            relative_volume_error=float(relative_error),
            failure_angle_boundary_hit_rate=float(np.mean([strip.failure_angle_boundary_hit for strip in strips])),
            candidate_strip_count=candidate_strip_count,
            excluded_strip_count=len(excluded_diagnostics),
            applicability_status=(
                "PARTIAL_OUTSIDE_FEE_DOMAIN"
                if excluded_diagnostics
                else "APPLICABLE"
            ),
            exclusion_diagnostics=tuple(excluded_diagnostics),
            model_classification="PAPER_DIRECT_FEE_LEGACY_STRIPS",
        )

    def _compute_v3(
        self,
        intersection: ToolTerrainIntersection,
        H_resting_m: np.ndarray,
        material: MaterialScenario,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
        *,
        fallback_approach_direction_xy: np.ndarray,
        H_free_m: np.ndarray | None,
    ) -> FailureZone:
        """Build a laterally continuous 2.5-D failure surface.

        The pointwise FEE minimisation is unchanged.  Only the old sampling
        closure (independent 0.20 m maximum-depth strips) is replaced by a
        cutting-edge arc-length field and a labelled continuity closure.
        """

        resting = np.asarray(grid.validate_heightmap(H_resting_m), dtype=np.float64)
        free = resting if H_free_m is None else np.asarray(grid.validate_heightmap(H_free_m), dtype=np.float64)
        if integrator.shape != grid.shape:
            raise ValueError("[FailureSurfaceV3] integrator/grid shape mismatch")
        direction = self._approach_direction(intersection, fallback_approach_direction_xy)
        separation = np.asarray(intersection.separation_plane_direction_terrain, dtype=np.float64)
        if float(np.dot(separation[:2], direction)) < -1.0e-8:
            return self._empty_zone(grid.shape, direction, applicability_status="REVERSE_SEPARATION_INACTIVE")
        affected_yx = np.argwhere(intersection.affected_mask)
        if not len(affected_yx):
            return self._empty_zone(grid.shape, direction)

        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        xy = np.column_stack((
            grid.origin_x + affected_yx[:, 1] * grid.dx,
            grid.origin_y + affected_yx[:, 0] * grid.dy,
        ))
        lateral_cell = xy @ lateral
        edge = np.asarray(intersection.cutting_edge_points_terrain_m, dtype=np.float64)
        edge_lateral = edge[:, :2] @ lateral if len(edge) else lateral_cell
        low, high = float(np.min(edge_lateral)), float(np.max(edge_lateral))
        if high - low <= _EPS:
            half = 0.5 * max(grid.dx, grid.dy)
            low, high = low - half, high + half
        spacing = min(self.config.surface_sample_spacing_m, high - low)
        count = max(1, int(np.ceil((high - low) / spacing)))
        bounds = np.linspace(low, high, count + 1)
        centers = 0.5 * (bounds[:-1] + bounds[1:])
        widths = np.diff(bounds)

        gy, gx = np.gradient(free, grid.dy, grid.dx, edge_order=1)
        grade = gx * direction[0] + gy * direction[1]
        penetration = np.asarray(intersection.penetration_depth_m, dtype=np.float64)
        rake = self._rake_angle(separation, direction)
        raw_depth = np.zeros(count)
        raw_alpha = np.zeros(count)
        s_cut = np.zeros(count)
        terrain_z = np.zeros(count)
        contact_support = np.zeros(count, dtype=bool)
        for i, center in enumerate(centers):
            distance = np.abs(lateral_cell - center)
            selected = affected_yx[distance <= max(0.75 * widths[i], 0.75 * max(grid.dx, grid.dy))]
            # No nearest-contact fallback: a lateral sample with no actual
            # positive-depth tool intersection has zero physical support and
            # must remain exactly zero through all continuity operations.
            # Otherwise a narrow corner/tooth contact can be spread across
            # the bucket width and overstate both failure volume and FEE force.
            if not len(selected):
                continue
            depth_values = penetration[selected[:, 0], selected[:, 1]]
            positive = depth_values > 0.0
            selected, depth_values = selected[positive], depth_values[positive]
            if not len(selected):
                continue
            contact_support[i] = True
            # Robust local average: deep contact receives more weight but one
            # raster vertex can no longer dictate the complete lateral slice.
            weights_local = np.maximum(depth_values, 0.1 * float(np.mean(depth_values)))
            weights_local /= np.sum(weights_local)
            raw_depth[i] = float(np.sum(depth_values * weights_local))
            raw_alpha[i] = float(np.arctan(np.sum(grade[selected[:, 0], selected[:, 1]] * weights_local)))
            local_xy = np.column_stack((
                grid.origin_x + selected[:, 1] * grid.dx,
                grid.origin_y + selected[:, 0] * grid.dy,
            ))
            s_cut[i] = float(np.sum((local_xy @ direction) * weights_local))
            terrain_z[i] = float(np.sum(free[selected[:, 0], selected[:, 1]] * weights_local))

        depth = self._smooth_lateral(raw_depth, support=contact_support)
        alpha = self._smooth_lateral(raw_alpha, support=contact_support)
        distance_to_end = np.minimum(centers - low, high - centers)
        u = np.clip(distance_to_end / self.config.lateral_transition_width_m, 0.0, 1.0)
        transition = u * u * (3.0 - 2.0 * u)
        depth *= transition
        depth[~contact_support] = 0.0
        alpha[~contact_support] = 0.0

        beta0 = np.zeros(count)
        solved0: list[_SolvedWedge | None] = [None] * count
        excluded: list[str] = []
        for i in range(count):
            if depth[i] <= _EPS:
                continue
            try:
                solved0[i] = self._solve_strip(
                    depth=float(depth[i]), width=float(widths[i]), material=material,
                    local_slope_rad=float(alpha[i]), rake_angle_rad=rake,
                )
                beta0[i] = solved0[i].beta_rad
            except _NoAdmissibleWedge as error:
                excluded.append(f"sample={i}: {error}")

        valid = beta0 > 0.0
        if not np.any(valid):
            return self._empty_zone(
                grid.shape, direction, candidate_strip_count=count,
                excluded_strip_count=len(excluded), applicability_status="OUTSIDE_FEE_DOMAIN",
                exclusion_diagnostics=tuple(excluded),
            )
        beta_filled = self._fill_lateral_within_support(
            beta0, valid=valid, support=contact_support
        )
        beta_supported = contact_support & (beta_filled > 0.0)
        beta_smooth = self._smooth_lateral(
            beta_filled, support=beta_supported
        )
        beta = (
            (1.0 - self.config.lateral_continuity_weight) * beta_filled
            + self.config.lateral_continuity_weight * beta_smooth
        )
        beta[~beta_supported] = 0.0

        requested_total = np.zeros(grid.shape, dtype=np.float64)
        pending: list[tuple[dict[str, object], _RasterContribution]] = []
        runout = np.zeros(count)
        for i in np.flatnonzero(beta_supported):
            try:
                solved = self._evaluate_beta(
                    beta_rad=float(beta[i]), depth=float(depth[i]), width=float(widths[i]),
                    material=material, local_slope_rad=float(alpha[i]), rake_angle_rad=rake,
                    iterations=solved0[i].iterations if solved0[i] is not None else 0,
                )
            except _NoAdmissibleWedge as error:
                excluded.append(f"sample={i}: {error}")
                continue
            runout[i] = solved.length_m
            center = centers[i]
            cut_xy = direction * s_cut[i] + lateral * center
            actual_edge = self._edge_point(edge, center, lateral, cut_xy, terrain_z[i] - depth[i])
            effective_edge = np.asarray([cut_xy[0], cut_xy[1], terrain_z[i] - depth[i]])
            forward_xy = cut_xy + direction * solved.length_m
            forward_z = terrain_z[i] + solved.length_m * np.tan(alpha[i])
            vertices = np.asarray([effective_edge, [cut_xy[0], cut_xy[1], terrain_z[i]], [forward_xy[0], forward_xy[1], forward_z]])
            raster = self._rasterize_strip_wedge(
                cut_xy=cut_xy, lateral_bounds=(float(bounds[i]), float(bounds[i + 1])),
                direction=direction, lateral=lateral, depth=float(depth[i]), extent=solved.length_m,
                grid=grid, integrator=integrator,
            )
            for (row, column), volume in zip(raster.indices_yx, raster.requested_m3):
                requested_total[row, column] += volume
            pending.append(({
                "strip_index": int(i), "lateral_bounds_m": bounds[i:i + 2],
                "lateral_center_m": float(center), "width_m": float(widths[i]),
                "cutting_edge_terrain_m": actual_edge, "effective_cutting_edge_terrain_m": effective_edge,
                "local_terrain_elevation_m": float(terrain_z[i]), "local_terrain_slope_rad": float(alpha[i]),
                "penetration_depth_m": float(depth[i]), "rake_angle_deg": float(np.rad2deg(rake)),
                "failure_angle_deg": float(np.rad2deg(solved.beta_rad)), "longitudinal_extent_m": solved.length_m,
                "failure_plane_length_m": solved.plane_length_m, "cross_section_vertices_terrain_m": vertices,
                "cross_section_area_m2": solved.area_m2, "wedge_volume_m3": solved.volume_m3,
                "centroid_terrain_m": np.mean(vertices, axis=0),
                "internal_friction_angle_deg": material.internal_friction_angle_deg,
                "soil_tool_friction_angle_deg": float(np.rad2deg(np.arctan(material.tool_friction_coefficient))),
                "cohesion_pa": material.cohesion_proxy_pa, "assumed_bulk_density_kg_m3": material.assumed_bulk_density_kg_m3,
                "surcharge_pa": self.config.surcharge_pa, "gravity_m_s2": self.config.gravity_m_s2,
                "luengo_Nw": solved.Nw, "luengo_Nc": solved.Nc, "luengo_Nq": solved.Nq,
                "weight_resistance_n": solved.weight_n, "cohesion_resistance_n": solved.cohesion_n,
                "surcharge_resistance_n": solved.surcharge_n, "estimated_resistance_n": solved.resistance_n,
                "solver_status": "continuity_closed_" + solved.status, "solver_iterations": solved.iterations,
                "failure_angle_boundary_hit": solved.boundary_hit,
            }, raster))

        if not pending:
            return self._empty_zone(grid.shape, direction, candidate_strip_count=count, excluded_strip_count=len(excluded), applicability_status="OUTSIDE_FEE_DOMAIN", exclusion_diagnostics=tuple(excluded))
        weights_area = np.asarray(integrator.vertex_weights_m2, dtype=np.float64)
        capacity = np.maximum(resting, 0.0) * weights_area
        scale = np.ones(grid.shape)
        occupied = requested_total > _EPS
        scale[occupied] = np.minimum(1.0, capacity[occupied] / requested_total[occupied])
        active = np.zeros(grid.shape)
        usable = weights_area > _EPS
        active[usable] = requested_total[usable] * scale[usable] / weights_area[usable]
        active = np.minimum(active, resting)
        active[active < _EPS] = 0.0
        strips: list[FailureStripGeometry] = []
        for values, raster in pending:
            actual = np.asarray([requested * scale[row, column] for (row, column), requested in zip(raster.indices_yx, raster.requested_m3)])
            requested_sum, actual_sum = float(np.sum(raster.requested_m3)), float(np.sum(actual))
            analytical = float(values["wedge_volume_m3"])
            values.update({
                "raster_indices_yx": raster.indices_yx, "raster_coverage_fractions": raster.coverage,
                "raster_requested_volume_m3": raster.requested_m3, "raster_activated_volume_m3": actual,
                "boundary_clipped": requested_sum < analytical - max(1e-10, 1e-8 * analytical),
                "availability_clipped": actual_sum < requested_sum - max(1e-10, 1e-8 * requested_sum),
            })
            strips.append(FailureStripGeometry(**values))
        active_volume = integrator.integrate(active)
        analytical = float(sum(item.wedge_volume_m3 for item in strips))
        requested = float(sum(item.rasterized_requested_volume_m3 for item in strips))
        centroid_den = max(analytical, _EPS)
        centroid = sum((item.wedge_volume_m3 * item.centroid_terrain_m for item in strips), start=np.zeros(3)) / centroid_den
        mean_angle = sum(item.failure_angle_deg * item.wedge_volume_m3 for item in strips) / centroid_den
        profile = FailureSurfaceProfile(
            arc_length_m=centers - low, penetration_depth_m=depth, terrain_slope_rad=alpha,
            rake_angle_rad=np.full(count, rake), beta0_rad=beta0,
            beta_rad=np.where(beta_supported, beta, 0.0),
            runout_length_m=runout, lateral_transition_weight=transition,
        )
        return FailureZone(
            active_mask=active > 0.0, active_thickness_m=active, active_volume_m3=active_volume,
            centroid_terrain_m=centroid, estimated_failure_angle_deg=float(mean_angle), approach_direction_xy=direction,
            strip_geometries=tuple(strips), estimated_total_resistance_n=float(sum(item.estimated_resistance_n for item in strips)),
            analytical_wedge_volume_m3=analytical, rasterized_requested_volume_m3=requested,
            relative_volume_error=abs(active_volume - analytical) / max(analytical, _EPS),
            failure_angle_boundary_hit_rate=float(np.mean([item.failure_angle_boundary_hit for item in strips])),
            candidate_strip_count=count, excluded_strip_count=len(excluded),
            applicability_status="PARTIAL_OUTSIDE_FEE_DOMAIN" if excluded else "APPLICABLE",
            exclusion_diagnostics=tuple(excluded), failure_surface_profile=profile,
            model_classification="PAPER_DIRECT_FEE_PLUS_REDUCED_ORDER_ENGINEERING_CLOSURE",
        )

    def _smooth_lateral(
        self,
        values: np.ndarray,
        *,
        support: np.ndarray | None = None,
    ) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        active = (
            np.ones(result.shape, dtype=bool)
            if support is None
            else np.asarray(support, dtype=bool)
        )
        if active.shape != result.shape:
            raise ValueError("[FailureSurfaceV3] lateral support shape mismatch")
        result[~active] = 0.0
        for _ in range(self.config.lateral_smoothing_passes):
            previous = result.copy()
            # Smooth only inside each contiguous physically supported run.
            # Unsupported samples remain exact zeros and cannot receive mass,
            # penetration depth or force support from a neighbour.
            indices = np.flatnonzero(active)
            if indices.size == 0:
                break
            breaks = np.flatnonzero(np.diff(indices) > 1) + 1
            for segment in np.split(indices, breaks):
                if segment.size == 1:
                    result[segment[0]] = previous[segment[0]]
                    continue
                lo, hi = int(segment[0]), int(segment[-1])
                if segment.size > 2:
                    result[lo + 1 : hi] = (
                        0.25 * previous[lo : hi - 1]
                        + 0.5 * previous[lo + 1 : hi]
                        + 0.25 * previous[lo + 2 : hi + 1]
                    )
                result[lo] = 0.75 * previous[lo] + 0.25 * previous[lo + 1]
                result[hi] = 0.75 * previous[hi] + 0.25 * previous[hi - 1]
            result[~active] = 0.0
        return result

    @staticmethod
    def _fill_lateral_within_support(
        values: np.ndarray,
        *,
        valid: np.ndarray,
        support: np.ndarray,
    ) -> np.ndarray:
        """Interpolate only within contiguous physical-contact support runs."""

        source = np.asarray(values, dtype=np.float64)
        valid_mask = np.asarray(valid, dtype=bool)
        support_mask = np.asarray(support, dtype=bool)
        if source.shape != valid_mask.shape or source.shape != support_mask.shape:
            raise ValueError("[FailureSurfaceV3] lateral interpolation shape mismatch")
        result = np.zeros_like(source)
        supported = np.flatnonzero(support_mask)
        if supported.size == 0:
            return result
        breaks = np.flatnonzero(np.diff(supported) > 1) + 1
        for segment in np.split(supported, breaks):
            anchors = segment[valid_mask[segment]]
            if anchors.size == 0:
                continue
            if anchors.size == 1:
                result[segment] = source[anchors[0]]
            else:
                result[segment] = np.interp(segment, anchors, source[anchors])
        return result

    def _evaluate_beta(
        self, *, beta_rad: float, depth: float, width: float,
        material: MaterialScenario, local_slope_rad: float, rake_angle_rad: float,
        iterations: int,
    ) -> _SolvedWedge:
        phi = np.deg2rad(material.internal_friction_angle_deg)
        delta = np.arctan(material.tool_friction_coefficient)
        alpha, beta = float(local_slope_rad), float(beta_rad)
        tangent = np.tan(beta) - np.tan(alpha)
        denominator = np.cos(rake_angle_rad + delta) + np.sin(rake_angle_rad + delta) / np.tan(beta + phi)
        if tangent <= _EPS or denominator <= 1e-8:
            raise _NoAdmissibleWedge("continuity-closed beta left the admissible FEE domain")
        length = depth / tangent
        if length > self.config.maximum_wedge_length_m + 1e-9:
            raise _NoAdmissibleWedge("continuity-closed runout exceeds configured admissibility")
        Nw = ((1.0 / np.tan(beta) - np.tan(alpha)) * (np.cos(alpha) + np.sin(alpha) / np.tan(beta + phi))) / (2.0 * denominator)
        Nc = (1.0 + 1.0 / (np.tan(beta) * np.tan(beta + phi))) / denominator
        Nq = (np.cos(alpha) + np.sin(alpha) / np.tan(beta + phi)) / denominator
        if not np.all(np.isfinite([Nw, Nc, Nq])) or min(Nw, Nc, Nq) < 0.0:
            raise _NoAdmissibleWedge("continuity-closed beta produced invalid FEE coefficients")
        area, volume = 0.5 * depth * length, 0.5 * depth * length * width
        weight = depth**2 * width * material.assumed_bulk_density_kg_m3 * self.config.gravity_m_s2 * Nw
        cohesion = material.cohesion_proxy_pa * width * depth * Nc
        surcharge = self.config.surcharge_pa * width * depth * Nq
        lower = max(np.deg2rad(self.config.minimum_failure_angle_deg), alpha + 1e-5, np.arctan(np.tan(alpha) + depth / self.config.maximum_wedge_length_m))
        upper = np.deg2rad(self.config.maximum_failure_angle_deg)
        hit = abs(beta - lower) < np.deg2rad(self.config.solver_tolerance_deg) * 2 or abs(beta - upper) < np.deg2rad(self.config.solver_tolerance_deg) * 2
        return _SolvedWedge(beta, float(length), float(np.hypot(length, depth + length * np.tan(alpha))), float(area), float(volume), float(Nw), float(Nc), float(Nq), float(weight), float(cohesion), float(surcharge), float(weight + cohesion + surcharge), "boundary_minimum" if hit else "converged", iterations, bool(hit))

    def _solve_strip(
        self,
        *,
        depth: float,
        width: float,
        material: MaterialScenario,
        local_slope_rad: float,
        rake_angle_rad: float,
    ) -> _SolvedWedge:
        """Minimize Luengo's quasi-static sloped-ground FEE over beta.

        Luengo et al. identify beta as a model parameter.  Selecting the
        critical plane by minimum predicted cutting force is therefore a
        documented literature-based reduced-order closure, not a claimed
        equation from that paper.
        """

        phi = np.deg2rad(material.internal_friction_angle_deg)
        delta = np.arctan(material.tool_friction_coefficient)
        alpha = float(local_slope_rad)
        length_bound = np.arctan(np.tan(alpha) + depth / self.config.maximum_wedge_length_m)
        lower = max(np.deg2rad(self.config.minimum_failure_angle_deg), alpha + 1.0e-5, length_bound)
        upper = np.deg2rad(self.config.maximum_failure_angle_deg)
        if lower >= upper:
            raise _NoAdmissibleWedge(
                "[FailureZone] no admissible failure angle: terrain slope/length constraint exceeds upper bound"
            )

        def evaluate(beta: float) -> tuple[float, tuple[float, ...]]:
            tan_difference = np.tan(beta) - np.tan(alpha)
            if tan_difference <= _EPS:
                return np.inf, ()
            length = depth / tan_difference
            denominator = (
                np.cos(rake_angle_rad + delta)
                + np.sin(rake_angle_rad + delta) / np.tan(beta + phi)
            )
            if denominator <= 1.0e-8 or not np.isfinite(denominator):
                return np.inf, ()
            slope_numerator = (
                (1.0 / np.tan(beta) - np.tan(alpha))
                * (np.cos(alpha) + np.sin(alpha) / np.tan(beta + phi))
            )
            Nw = slope_numerator / (2.0 * denominator)
            Nc = (1.0 + 1.0 / (np.tan(beta) * np.tan(beta + phi))) / denominator
            Nq = (np.cos(alpha) + np.sin(alpha) / np.tan(beta + phi)) / denominator
            if not np.all(np.isfinite([Nw, Nc, Nq])) or Nw < 0.0 or Nc < 0.0 or Nq < 0.0:
                return np.inf, ()
            area = 0.5 * depth * length
            volume = area * width
            weight = depth**2 * width * material.assumed_bulk_density_kg_m3 * self.config.gravity_m_s2 * Nw
            cohesion = material.cohesion_proxy_pa * width * depth * Nc
            surcharge = self.config.surcharge_pa * width * depth * Nq
            resistance = weight + cohesion + surcharge
            plane_length = float(np.hypot(length, depth + length * np.tan(alpha)))
            return float(resistance), (
                length,
                plane_length,
                area,
                volume,
                Nw,
                Nc,
                Nq,
                weight,
                cohesion,
                surcharge,
            )

        sample_angles = np.linspace(lower, upper, self.config.failure_angle_samples)
        sample_force = np.asarray([evaluate(float(beta))[0] for beta in sample_angles])
        finite = np.isfinite(sample_force)
        if not np.any(finite):
            raise _NoAdmissibleWedge(
                "[FailureZone] Luengo FEE has no admissible finite wedge: "
                f"depth={depth:.6g} m, width={width:.6g} m, "
                f"terrain_slope={np.rad2deg(alpha):.6g} deg, "
                f"rake={np.rad2deg(rake_angle_rad):.6g} deg, "
                f"phi={np.rad2deg(phi):.6g} deg, "
                f"delta={np.rad2deg(delta):.6g} deg, "
                f"beta_range=[{np.rad2deg(lower):.6g}, {np.rad2deg(upper):.6g}] deg"
            )
        best_index = int(np.nanargmin(np.where(finite, sample_force, np.nan)))
        left_index = max(0, best_index - 1)
        right_index = min(len(sample_angles) - 1, best_index + 1)
        a = float(sample_angles[left_index])
        b = float(sample_angles[right_index])
        iterations = 0
        if b > a and 0 < best_index < len(sample_angles) - 1:
            golden = 0.5 * (np.sqrt(5.0) - 1.0)
            c = b - golden * (b - a)
            d = a + golden * (b - a)
            fc = evaluate(c)[0]
            fd = evaluate(d)[0]
            tolerance = np.deg2rad(self.config.solver_tolerance_deg)
            while b - a > tolerance and iterations < self.config.maximum_solver_iterations:
                if fc <= fd:
                    b, d, fd = d, c, fc
                    c = b - golden * (b - a)
                    fc = evaluate(c)[0]
                else:
                    a, c, fc = c, d, fd
                    d = a + golden * (b - a)
                    fd = evaluate(d)[0]
                iterations += 1
            beta = 0.5 * (a + b)
            status = "converged"
        else:
            beta = float(sample_angles[best_index])
            status = "boundary_minimum"
        resistance, terms = evaluate(beta)
        if not terms:
            beta = float(sample_angles[best_index])
            resistance, terms = evaluate(beta)
        boundary_tolerance = max(np.deg2rad(self.config.solver_tolerance_deg) * 2.0, 1.0e-7)
        boundary_hit = bool(abs(beta - lower) <= boundary_tolerance or abs(beta - upper) <= boundary_tolerance)
        if boundary_hit:
            status = "boundary_minimum"
        (
            length,
            plane_length,
            area,
            volume,
            Nw,
            Nc,
            Nq,
            weight,
            cohesion,
            surcharge,
        ) = terms
        return _SolvedWedge(
            beta_rad=float(beta),
            length_m=float(length),
            plane_length_m=float(plane_length),
            area_m2=float(area),
            volume_m3=float(volume),
            Nw=float(Nw),
            Nc=float(Nc),
            Nq=float(Nq),
            weight_n=float(weight),
            cohesion_n=float(cohesion),
            surcharge_n=float(surcharge),
            resistance_n=float(resistance),
            status=status,
            iterations=iterations,
            boundary_hit=boundary_hit,
        )

    @staticmethod
    def _approach_direction(
        intersection: ToolTerrainIntersection,
        fallback: np.ndarray,
    ) -> np.ndarray:
        velocity = np.asarray(intersection.cutting_edge_velocity_terrain_m_s[:2], dtype=np.float64)
        direction = velocity if np.linalg.norm(velocity) > 1.0e-6 else np.asarray(fallback, dtype=np.float64)
        if direction.shape != (2,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= _EPS:
            raise ValueError("[FailureZone] fallback approach direction must be finite/nonzero")
        return direction / np.linalg.norm(direction)

    @staticmethod
    def _rake_angle(separation_direction: np.ndarray, approach: np.ndarray) -> float:
        direction = np.asarray(separation_direction, dtype=np.float64)
        horizontal = float(np.dot(direction[:2], approach))
        vertical = float(direction[2])
        if vertical < 0.0:
            horizontal = -horizontal
            vertical = -vertical
        rake = float(np.arctan2(max(vertical, 0.0), horizontal))
        return float(np.clip(rake, np.deg2rad(5.0), np.deg2rad(175.0)))

    @staticmethod
    def _edge_point(
        edge: np.ndarray,
        lateral_center: float,
        lateral: np.ndarray,
        fallback_xy: np.ndarray,
        fallback_z: float,
    ) -> np.ndarray:
        if len(edge) < 2:
            return np.asarray([fallback_xy[0], fallback_xy[1], fallback_z], dtype=np.float64)
        coordinates = edge[:, :2] @ lateral
        order = np.argsort(coordinates)
        coordinates = coordinates[order]
        ordered = edge[order]
        point = np.asarray(
            [np.interp(lateral_center, coordinates, ordered[:, axis]) for axis in range(3)],
            dtype=np.float64,
        )
        return point

    @classmethod
    def _rasterize_strip_wedge(
        cls,
        *,
        cut_xy: np.ndarray,
        lateral_bounds: tuple[float, float],
        direction: np.ndarray,
        lateral: np.ndarray,
        depth: float,
        extent: float,
        grid: TerrainGrid,
        integrator: TerrainVolumeIntegrator,
    ) -> _RasterContribution:
        """Integrate the analytical prism over vertex control polygons exactly."""

        if extent <= _EPS:
            return _RasterContribution(np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0))
        corners_xy = np.asarray(
            [
                cut_xy + lateral * (lateral_bounds[0] - np.dot(cut_xy, lateral)),
                cut_xy + lateral * (lateral_bounds[1] - np.dot(cut_xy, lateral)),
                cut_xy + direction * extent + lateral * (lateral_bounds[0] - np.dot(cut_xy, lateral)),
                cut_xy + direction * extent + lateral * (lateral_bounds[1] - np.dot(cut_xy, lateral)),
            ]
        )
        x_min, y_min = np.min(corners_xy, axis=0)
        x_max, y_max = np.max(corners_xy, axis=0)
        column_low = max(0, int(np.floor((x_min - grid.origin_x) / grid.dx)) - 1)
        column_high = min(grid.nx - 1, int(np.ceil((x_max - grid.origin_x) / grid.dx)) + 1)
        row_low = max(0, int(np.floor((y_min - grid.origin_y) / grid.dy)) - 1)
        row_high = min(grid.ny - 1, int(np.ceil((y_max - grid.origin_y) / grid.dy)) + 1)
        weights = np.asarray(integrator.vertex_weights_m2, dtype=np.float64)
        indices: list[tuple[int, int]] = []
        coverage: list[float] = []
        volumes: list[float] = []
        s_origin = float(np.dot(cut_xy, direction))
        for row in range(row_low, row_high + 1):
            if row == 0:
                y0 = grid.origin_y
            else:
                y0 = grid.origin_y + (row - 0.5) * grid.dy
            if row == grid.ny - 1:
                y1 = grid.origin_y + (grid.ny - 1) * grid.dy
            else:
                y1 = grid.origin_y + (row + 0.5) * grid.dy
            for column in range(column_low, column_high + 1):
                if weights[row, column] <= _EPS:
                    continue
                if grid.valid_mask is not None and not grid.valid_mask[row, column]:
                    continue
                if column == 0:
                    x0 = grid.origin_x
                else:
                    x0 = grid.origin_x + (column - 0.5) * grid.dx
                if column == grid.nx - 1:
                    x1 = grid.origin_x + (grid.nx - 1) * grid.dx
                else:
                    x1 = grid.origin_x + (column + 0.5) * grid.dx
                polygon_xy = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
                polygon_sl = np.column_stack(
                    [polygon_xy @ direction - s_origin, polygon_xy @ lateral]
                )
                clipped = cls._clip_rectangle(
                    polygon_sl,
                    0.0,
                    extent,
                    lateral_bounds[0],
                    lateral_bounds[1],
                )
                area, centroid_sl = cls._polygon_area_centroid(clipped)
                if area <= _EPS:
                    continue
                thickness = depth * (1.0 - centroid_sl[0] / extent)
                volume = area * max(thickness, 0.0)
                if volume <= _EPS:
                    continue
                control_area = max((x1 - x0) * (y1 - y0), _EPS)
                indices.append((row, column))
                coverage.append(float(np.clip(area / control_area, 0.0, 1.0)))
                volumes.append(float(volume))
        return _RasterContribution(
            indices_yx=np.asarray(indices, dtype=np.int64).reshape((-1, 2)),
            coverage=np.asarray(coverage, dtype=np.float64),
            requested_m3=np.asarray(volumes, dtype=np.float64),
        )

    @staticmethod
    def _clip_rectangle(
        polygon: np.ndarray,
        s_min: float,
        s_max: float,
        l_min: float,
        l_max: float,
    ) -> np.ndarray:
        result = np.asarray(polygon, dtype=np.float64)
        boundaries = ((0, s_min, True), (0, s_max, False), (1, l_min, True), (1, l_max, False))
        for axis, bound, keep_greater in boundaries:
            if not len(result):
                break
            output: list[np.ndarray] = []
            previous = result[-1]
            previous_inside = previous[axis] >= bound - _EPS if keep_greater else previous[axis] <= bound + _EPS
            for current in result:
                current_inside = current[axis] >= bound - _EPS if keep_greater else current[axis] <= bound + _EPS
                if current_inside != previous_inside:
                    denominator = current[axis] - previous[axis]
                    fraction = 0.0 if abs(denominator) <= _EPS else (bound - previous[axis]) / denominator
                    output.append(previous + fraction * (current - previous))
                if current_inside:
                    output.append(current)
                previous = current
                previous_inside = current_inside
            result = np.asarray(output, dtype=np.float64).reshape((-1, 2))
        return result

    @staticmethod
    def _polygon_area_centroid(polygon: np.ndarray) -> tuple[float, np.ndarray]:
        if len(polygon) < 3:
            return 0.0, np.zeros(2, dtype=np.float64)
        following = np.roll(polygon, -1, axis=0)
        cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
        signed_area = 0.5 * float(np.sum(cross))
        if abs(signed_area) <= _EPS:
            return 0.0, np.zeros(2, dtype=np.float64)
        centroid = np.sum((polygon + following) * cross[:, None], axis=0) / (6.0 * signed_area)
        return abs(signed_area), np.asarray(centroid, dtype=np.float64)

    @staticmethod
    def _empty_zone(
        shape: tuple[int, int],
        direction: np.ndarray,
        *,
        candidate_strip_count: int = 0,
        excluded_strip_count: int = 0,
        applicability_status: str = "NO_INTERSECTION",
        exclusion_diagnostics: tuple[str, ...] = (),
    ) -> FailureZone:
        return FailureZone(
            active_mask=np.zeros(shape, dtype=bool),
            active_thickness_m=np.zeros(shape, dtype=np.float64),
            active_volume_m3=0.0,
            centroid_terrain_m=np.zeros(3, dtype=np.float64),
            estimated_failure_angle_deg=0.0,
            approach_direction_xy=direction,
            strip_geometries=(),
            estimated_total_resistance_n=0.0,
            analytical_wedge_volume_m3=0.0,
            rasterized_requested_volume_m3=0.0,
            relative_volume_error=0.0,
            failure_angle_boundary_hit_rate=0.0,
            candidate_strip_count=candidate_strip_count,
            excluded_strip_count=excluded_strip_count,
            applicability_status=applicability_status,
            exclusion_diagnostics=exclusion_diagnostics,
        )
