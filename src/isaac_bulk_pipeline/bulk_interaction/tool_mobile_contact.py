"""Geometry and parameter-free impulse law for already-Mobile material.

The normal law is the minimum infinite-mass-wall impulse that removes closing
relative velocity.  It has no stiffness, relaxation time, force multiplier or
capture coefficient.  Tangential response is maximum-dissipation Coulomb
friction using the existing soil--tool interface coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..bulk_state import TerrainVolumeIntegrator
from ..terrain import TerrainGrid
from ..tools import ToolDescriptor, ToolState, assert_open_bucket_physical_contact


_EPS = 1.0e-12


def _ro(value: np.ndarray, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FrictionalWallImpulse:
    normal_impulse_xy_ns: np.ndarray
    tangential_impulse_xy_ns: np.ndarray
    total_impulse_xy_ns: np.ndarray
    mobile_kinetic_energy_change_j: float
    tool_to_mobile_work_j: float
    normal_dissipation_j: float
    frictional_dissipation_j: float

    @property
    def machine_reaction_impulse_xy_ns(self) -> np.ndarray:
        return -self.total_impulse_xy_ns


@dataclass(frozen=True)
class SurfaceConstrainedWallImpulse:
    requested_impulse_xyz_ns: np.ndarray
    represented_impulse_xyz_ns: np.ndarray
    unresolved_impulse_xyz_ns: np.ndarray
    represented_mobile_velocity_xyz_m_s: np.ndarray
    closing_speed_m_s: float
    normal_impulse_ns: float
    tangential_impulse_ns: float

    @property
    def machine_reaction_impulse_xyz_ns(self) -> np.ndarray:
        return -self.represented_impulse_xyz_ns


def resolve_surface_constrained_frictional_wall_impulse(
    *,
    mobile_mass_kg: float,
    mobile_velocity_xyz_m_s: np.ndarray,
    tool_velocity_xyz_m_s: np.ndarray,
    bucket_outward_normal_xyz: np.ndarray,
    terrain_normal_xyz: np.ndarray,
    tool_mobile_friction_coefficient: float,
) -> SurfaceConstrainedWallImpulse:
    """Resolve true-3D closing for a state that stores only horizontal momentum.

    ``mobile_velocity_xyz_m_s`` may include a kinematically reconstructed world-Z
    component (for example motion tangent to the local terrain), so contact
    detection uses the real 3-D bucket normal.  The authoritative Mobile V2
    state, however, stores only ``q_x,q_y`` on the horizontal grid.  Therefore
    the *accepted* reduced impulse is restricted to the world-XY subspace.

    The projected bucket normal is deliberately **not** renormalized.  A bucket
    face whose normal is almost vertical consequently has only a small
    representable horizontal effect instead of becoming an artificial unit
    horizontal bulldozing normal.  A full 3-D inelastic/Coulomb impulse is kept
    only as a dimensionality oracle; its unresolved part is never injected into
    the 2.5-D state or returned as machine reaction.

    The reduced correction is scaled analytically, only toward zero, so the
    state-consistent moving-wall inequality ``W_tool - DeltaK_xy >= 0`` holds.
    This is a conservation/passivity constraint, not an empirical damping term.
    """

    mass = float(mobile_mass_kg)
    mu = float(tool_mobile_friction_coefficient)
    mobile = np.asarray(mobile_velocity_xyz_m_s, dtype=np.float64)
    tool = np.asarray(tool_velocity_xyz_m_s, dtype=np.float64)
    bucket_n = np.asarray(bucket_outward_normal_xyz, dtype=np.float64)
    terrain_n = np.asarray(terrain_normal_xyz, dtype=np.float64)
    if mass < 0.0 or mu < 0.0 or not np.isfinite(mass + mu):
        raise ValueError("[ToolMobileContact] mass/mu must be finite/non-negative")
    if any(value.shape != (3,) for value in (mobile, tool, bucket_n, terrain_n)):
        raise ValueError("[ToolMobileContact] 3-D vectors must have shape (3,)")
    if not np.all(np.isfinite(np.r_[mobile, tool, bucket_n, terrain_n])):
        raise ValueError("[ToolMobileContact] 3-D vectors must be finite")
    bucket_norm = float(np.linalg.norm(bucket_n))
    terrain_norm = float(np.linalg.norm(terrain_n))
    if bucket_norm <= _EPS or terrain_norm <= _EPS:
        raise ValueError("[ToolMobileContact] normals must be nonzero")
    bucket_n = bucket_n / bucket_norm
    # Validate/normalize the support normal because callers use this helper as
    # the CPU oracle for the GPU terrain-tangent velocity reconstruction.  It is
    # intentionally not used as an impulse subspace: q_x,q_y, not a finite-area
    # surface momentum, remain authoritative in Mobile V2.
    terrain_n = terrain_n / terrain_norm
    relative = mobile - tool
    closing = float(relative @ bucket_n)
    zero = np.zeros(3, dtype=np.float64)
    if mass == 0.0 or closing >= 0.0:
        return SurfaceConstrainedWallImpulse(
            _ro(zero), _ro(zero), _ro(zero), _ro(mobile), closing, 0.0, 0.0
        )
    normal_delta = -closing

    # Full 3-D contact request retained only for dimensionality/error audit.
    slip3 = relative - closing * bucket_n
    slip3_speed = float(np.linalg.norm(slip3))
    requested_tangent_delta = zero.copy()
    if slip3_speed > _EPS:
        requested_tangent_delta = (
            -min(slip3_speed, mu * normal_delta) * slip3 / slip3_speed
        )
    requested_delta = normal_delta * bucket_n + requested_tangent_delta

    # Representable normal response in authoritative (x,y) momentum space.
    # No normalization after dropping z.
    bucket_normal_xy = np.asarray([bucket_n[0], bucket_n[1], 0.0])
    normal_represented_delta = normal_delta * bucket_normal_xy

    # Coulomb slip inside the representable XY contact tangent.
    relative_xy = np.asarray([relative[0], relative[1], 0.0])
    normal_xy_norm2 = float(bucket_normal_xy @ bucket_normal_xy)
    slip_represented = relative_xy.copy()
    if normal_xy_norm2 > _EPS * _EPS:
        slip_represented -= bucket_normal_xy * float(
            (slip_represented @ bucket_normal_xy) / normal_xy_norm2
        )
    slip_represented_speed = float(np.linalg.norm(slip_represented))
    tangent_represented_delta = zero.copy()
    represented_normal_magnitude = float(np.linalg.norm(normal_represented_delta))
    if slip_represented_speed > _EPS and represented_normal_magnitude > 0.0:
        tangent_represented_delta = (
            -min(slip_represented_speed, mu * represented_normal_magnitude)
            * slip_represented
            / slip_represented_speed
        )

    raw_represented_delta = normal_represented_delta + tangent_represented_delta

    # State-consistent passivity for the q_x,q_y kinetic-energy measure.
    represented_norm2 = float(raw_represented_delta @ raw_represented_delta)
    passivity_scale = 0.0
    if represented_norm2 > _EPS * _EPS:
        available_work_per_mass = float(
            raw_represented_delta @ (tool - mobile)
        )
        if available_work_per_mass > 0.0:
            passivity_scale = min(
                1.0, 2.0 * available_work_per_mass / represented_norm2
            )
    normal_accepted_delta = passivity_scale * normal_represented_delta
    tangent_accepted_delta = passivity_scale * tangent_represented_delta
    represented_delta = normal_accepted_delta + tangent_accepted_delta

    requested = mass * requested_delta
    represented = mass * represented_delta
    unresolved = requested - represented
    represented_velocity = mobile.copy()
    represented_velocity[0:2] += represented_delta[0:2]
    # z is kinematic/unresolved in Mobile V2 and is not impulsively changed by
    # a q_x,q_y update in this reduced contact oracle.
    return SurfaceConstrainedWallImpulse(
        _ro(requested),
        _ro(represented),
        _ro(unresolved),
        _ro(represented_velocity),
        closing,
        float(mass * np.linalg.norm(normal_accepted_delta)),
        float(mass * np.linalg.norm(tangent_accepted_delta)),
    )


def resolve_frictional_wall_impulse(
    *,
    mobile_mass_kg: float,
    mobile_velocity_xy_m_s: np.ndarray,
    tool_velocity_xy_m_s: np.ndarray,
    outward_normal_xy: np.ndarray,
    tool_mobile_friction_coefficient: float,
) -> FrictionalWallImpulse:
    """Resolve one 2-D depth-averaged Mobile/control-surface collision.

    ``outward_normal_xy`` points from the tool surface toward the Mobile
    control volume.  Closing therefore means ``dot(u-v_tool, n) < 0``.
    The wall is kinematic over the substep, so the minimum no-penetration
    impulse is ``J_n = -m*v_n*n``.  Coulomb friction then removes as much
    tangential slip as allowed by ``|J_t| <= mu*|J_n|``.
    """

    mass = float(mobile_mass_kg)
    mu = float(tool_mobile_friction_coefficient)
    mobile = np.asarray(mobile_velocity_xy_m_s, dtype=np.float64)
    tool = np.asarray(tool_velocity_xy_m_s, dtype=np.float64)
    normal = np.asarray(outward_normal_xy, dtype=np.float64)
    if mass < 0.0 or mu < 0.0 or not np.isfinite(mass + mu):
        raise ValueError("[ToolMobileContact] mass/mu must be finite/non-negative")
    if mobile.shape != (2,) or tool.shape != (2,) or normal.shape != (2,):
        raise ValueError("[ToolMobileContact] velocities/normal must have shape (2,)")
    if not np.all(np.isfinite(np.r_[mobile, tool, normal])):
        raise ValueError("[ToolMobileContact] velocities/normal must be finite")
    norm = float(np.linalg.norm(normal))
    if norm <= _EPS:
        raise ValueError("[ToolMobileContact] horizontal normal must be nonzero")
    normal = normal / norm
    zero = np.zeros(2, dtype=np.float64)
    relative = mobile - tool
    closing = float(np.dot(relative, normal))
    if mass == 0.0 or closing >= 0.0:
        return FrictionalWallImpulse(zero, zero, zero, 0.0, 0.0, 0.0, 0.0)

    normal_impulse = -mass * closing * normal
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    slip = float(np.dot(relative, tangent))
    required_tangent_magnitude = mass * abs(slip)
    tangent_magnitude = min(required_tangent_magnitude, mu * np.linalg.norm(normal_impulse))
    tangential_impulse = (
        zero
        if abs(slip) <= _EPS or tangent_magnitude <= 0.0
        else -np.sign(slip) * tangent_magnitude * tangent
    )
    total = normal_impulse + tangential_impulse
    after = mobile + total / mass
    kinetic_change = 0.5 * mass * (
        float(np.dot(after, after)) - float(np.dot(mobile, mobile))
    )
    tool_work = float(np.dot(total, tool))
    normal_after = float(np.dot(after - tool, normal))
    tangent_after = float(np.dot(after - tool, tangent))
    normal_dissipation = 0.5 * mass * max(0.0, closing * closing - normal_after * normal_after)
    frictional_dissipation = 0.5 * mass * max(0.0, slip * slip - tangent_after * tangent_after)
    return FrictionalWallImpulse(
        _ro(normal_impulse),
        _ro(tangential_impulse),
        _ro(total),
        float(kinetic_change),
        tool_work,
        float(normal_dissipation),
        float(frictional_dissipation),
    )


@dataclass(frozen=True)
class ToolMobileContactSupport:
    """Compact, geometry-confirmed candidate support for one machine frame."""

    flat_indices: np.ndarray
    closest_points_terrain_m: np.ndarray
    outward_normals_terrain: np.ndarray
    outward_normals_xy: np.ndarray
    tool_surface_velocity_terrain_m_s: np.ndarray
    cavity_signed_distance_m: np.ndarray
    mobile_volume_m3: float
    tool_reference_position_terrain_m: np.ndarray
    geometry_classification: str = "CAD_DERIVED_L1_TRIANGLE_PRISM_INTERSECTION"
    performance_diagnostics: Mapping[str, Any] = field(
        default_factory=dict, compare=False
    )

    def __post_init__(self) -> None:
        indices = np.asarray(self.flat_indices, dtype=np.int32).reshape(-1)
        count = indices.size
        values = {
            "closest_points_terrain_m": (self.closest_points_terrain_m, (count, 3)),
            "outward_normals_terrain": (self.outward_normals_terrain, (count, 3)),
            "outward_normals_xy": (self.outward_normals_xy, (count, 2)),
            "tool_surface_velocity_terrain_m_s": (
                self.tool_surface_velocity_terrain_m_s,
                (count, 3),
            ),
            "cavity_signed_distance_m": (self.cavity_signed_distance_m, (count,)),
            "tool_reference_position_terrain_m": (
                self.tool_reference_position_terrain_m,
                (3,),
            ),
        }
        object.__setattr__(self, "flat_indices", _ro(indices, np.int32))
        for name, (value, shape) in values.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"[ToolMobileContact] {name} must be finite {shape}")
            object.__setattr__(self, name, _ro(array))
        if count:
            xy_norms = np.linalg.norm(self.outward_normals_xy, axis=1)
            # A true 3-D contact may have no representable horizontal normal
            # (e.g. bucket floor). Legacy XY diagnostics are therefore either
            # unit vectors or exactly zero; production contact consumes the
            # retained 3-D normal instead.
            valid_xy = np.isclose(xy_norms, 1.0, atol=1.0e-10) | np.isclose(
                xy_norms, 0.0, atol=1.0e-10
            )
            if not np.all(valid_xy):
                raise ValueError(
                    "[ToolMobileContact] projected normals must be unit or zero"
                )
        volume = float(self.mobile_volume_m3)
        if not np.isfinite(volume) or volume < 0.0:
            raise ValueError("[ToolMobileContact] mobile volume must be finite/non-negative")
        object.__setattr__(self, "mobile_volume_m3", volume)
        object.__setattr__(
            self,
            "performance_diagnostics",
            MappingProxyType(dict(self.performance_diagnostics)),
        )

    @classmethod
    def empty(cls) -> "ToolMobileContactSupport":
        return cls(
            np.empty(0, dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
            np.zeros(3, dtype=np.float64),
        )

    @property
    def cell_count(self) -> int:
        return int(self.flat_indices.size)


def build_tool_mobile_contact_support(
    *,
    candidate_mask: np.ndarray,
    b_eff_after_activation_m: np.ndarray,
    mobile_after_activation_m: np.ndarray,
    bbox_yx: tuple[int, int, int, int],
    tool_state: ToolState,
    descriptor: ToolDescriptor,
    grid: TerrainGrid,
    integrator: TerrainVolumeIntegrator,
) -> ToolMobileContactSupport:
    """Intersect candidate Mobile prisms with the CAD-derived bucket surface."""

    total_start = perf_counter()
    geometry = descriptor.bucket_geometry
    if geometry is None:
        return ToolMobileContactSupport.empty()
    contact_geometry = assert_open_bucket_physical_contact(geometry)
    candidate = np.asarray(candidate_mask, dtype=bool)
    bed = np.asarray(b_eff_after_activation_m, dtype=np.float64)
    mobile = np.asarray(mobile_after_activation_m, dtype=np.float64)
    if candidate.shape != bed.shape or mobile.shape != bed.shape:
        raise ValueError("[ToolMobileContact] compact field shape mismatch")
    row0, row1, col0, col1 = bbox_yx
    if bed.shape != (row1 - row0, col1 - col0):
        raise ValueError("[ToolMobileContact] bbox/field shape mismatch")

    geometry_start = perf_counter()
    vertices = geometry.transform_points(
        tool_state.pose_terrain, geometry.interior_vertices_local
    )
    all_faces = np.asarray(geometry.interior_faces, dtype=np.int64)
    physical_face_mask = np.asarray(
        contact_geometry.containment_face_mask, dtype=bool
    )
    triangles = vertices[all_faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    raw_norms = np.linalg.norm(raw_normals, axis=1)
    valid_faces = raw_norms > _EPS
    physical_face_mask = physical_face_mask[valid_faces]
    triangles = triangles[valid_faces]
    face_normals = raw_normals[valid_faces] / raw_norms[valid_faces, None]
    geometry_prepare_ms = (perf_counter() - geometry_start) * 1_000.0
    if not len(triangles):
        return ToolMobileContactSupport.empty()

    candidate_start = perf_counter()
    candidate_indices = np.argwhere(candidate & (mobile > 0.0))
    candidate_build_ms = (perf_counter() - candidate_start) * 1_000.0
    weights = np.asarray(integrator.vertex_weights_m2, dtype=np.float64)
    selected_indices: list[int] = []
    closest_points: list[np.ndarray] = []
    normals_3d: list[np.ndarray] = []
    normals_xy: list[np.ndarray] = []
    tool_velocities: list[np.ndarray] = []
    signed_distances: list[float] = []
    selected_volume = 0.0
    reference = np.asarray(tool_state.pose_terrain[:3, 3], dtype=np.float64)
    triangle_aabb_test_count = 0
    ray_triangle_test_count = 0
    closest_point_query_count = 0
    broadphase_ms = 0.0
    narrowphase_ms = 0.0
    closest_point_ms = 0.0

    for local_row, local_col in candidate_indices:
        global_row = int(local_row + row0)
        global_col = int(local_col + col0)
        x = grid.origin_x + global_col * grid.dx
        y = grid.origin_y + global_row * grid.dy
        x_low = x if global_col == 0 else x - 0.5 * grid.dx
        x_high = x if global_col == grid.nx - 1 else x + 0.5 * grid.dx
        y_low = y if global_row == 0 else y - 0.5 * grid.dy
        y_high = y if global_row == grid.ny - 1 else y + 0.5 * grid.dy
        z_low = float(bed[local_row, local_col])
        z_high = z_low + float(mobile[local_row, local_col])
        box_low = np.asarray([x_low, y_low, z_low], dtype=np.float64)
        box_high = np.asarray([x_high, y_high, z_high], dtype=np.float64)
        center = 0.5 * (box_low + box_high)
        half = 0.5 * (box_high - box_low)
        phase_start = perf_counter()
        intersects = False
        physical_intersects = False
        for face_index, triangle in enumerate(triangles):
            triangle_aabb_test_count += 1
            if _triangle_box_overlap(triangle, center, half):
                intersects = True
                physical_intersects = bool(
                    physical_intersects or physical_face_mask[face_index]
                )
                if physical_intersects:
                    break
        broadphase_ms += (perf_counter() - phase_start) * 1_000.0
        phase_start = perf_counter()
        inside = _point_inside_closed_mesh(center, triangles)
        ray_triangle_test_count += len(triangles)
        narrowphase_ms += (perf_counter() - phase_start) * 1_000.0
        if not physical_intersects and not inside:
            continue

        phase_start = perf_counter()
        distances = []
        candidates = []
        physical_face_indices: list[int] = []
        for face_index, triangle in enumerate(triangles):
            if not physical_face_mask[face_index]:
                continue
            closest_point_query_count += 1
            closest = _closest_point_on_triangle(center, *triangle)
            candidates.append(closest)
            distances.append(float(np.linalg.norm(center - closest)))
            physical_face_indices.append(face_index)
        physical_position = int(np.argmin(distances))
        face_index = physical_face_indices[physical_position]
        closest_point_ms += (perf_counter() - phase_start) * 1_000.0
        closest = candidates[physical_position]
        delta = center - closest
        distance = distances[physical_position]
        if distance > _EPS:
            normal = delta / distance
        else:
            # Mesh winding encloses the bucket cavity.  For material inside
            # that cavity the tool-to-material normal is opposite the cavity
            # outward face normal; outside it follows the face normal.
            normal = (-1.0 if inside else 1.0) * face_normals[face_index]
        horizontal = normal[:2].copy()
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm <= _EPS:
            # V1.3 contact detection is a true 3-D oracle even though the
            # authoritative Mobile state stores only q_x/q_y. Retain a
            # vertical-only CAD contact for closing/dimensionality diagnostics;
            # its legacy XY projection is exactly zero and therefore contributes
            # no representable 2-D wall impulse.
            horizontal[:] = 0.0
        else:
            horizontal /= horizontal_norm
        surface_velocity = (
            np.asarray(tool_state.linear_velocity, dtype=np.float64)
            + np.cross(
                np.asarray(tool_state.angular_velocity, dtype=np.float64),
                closest - reference,
            )
        )
        selected_indices.append(global_row * grid.nx + global_col)
        closest_points.append(closest)
        normals_3d.append(normal)
        normals_xy.append(horizontal)
        tool_velocities.append(surface_velocity)
        signed_distances.append(-distance if inside else distance)
        selected_volume += float(weights[global_row, global_col] * mobile[local_row, local_col])

    diagnostics = {
        "execution_device": "CPU_NUMPY_PYTHON",
        "mobile_candidate_count": int(len(candidate_indices)),
        "cad_triangle_count": int(len(triangles)),
        "containment_triangle_count": int(len(triangles)),
        "physical_contact_triangle_count": int(np.count_nonzero(physical_face_mask)),
        "mouth_cap_physical_contact_triangle_count": int(
            contact_geometry.mouth_physical_closure_face_count
        ),
        "bucket_contact_geometry_contract": contact_geometry.metadata[
            "geometry_contract"
        ],
        "broadphase_pair_count": int(len(candidate_indices) * len(triangles)),
        "triangle_aabb_test_count": int(triangle_aabb_test_count),
        "ray_triangle_test_count": int(ray_triangle_test_count),
        "exact_test_count": int(
            triangle_aabb_test_count + ray_triangle_test_count
        ),
        "closest_point_query_count": int(closest_point_query_count),
        "accepted_contact_count": int(len(selected_indices)),
        "geometry_prepare_ms": float(geometry_prepare_ms),
        "candidate_build_ms": float(candidate_build_ms),
        "broadphase_ms": float(broadphase_ms),
        "narrowphase_ms": float(narrowphase_ms),
        "closest_point_ms": float(closest_point_ms),
        "tool_mobile_total_ms": float((perf_counter() - total_start) * 1_000.0),
    }
    if not selected_indices:
        return ToolMobileContactSupport(
            np.empty(0, dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
            reference,
            performance_diagnostics=diagnostics,
        )
    return ToolMobileContactSupport(
        np.asarray(selected_indices, dtype=np.int32),
        np.asarray(closest_points, dtype=np.float64),
        np.asarray(normals_3d, dtype=np.float64),
        np.asarray(normals_xy, dtype=np.float64),
        np.asarray(tool_velocities, dtype=np.float64),
        np.asarray(signed_distances, dtype=np.float64),
        selected_volume,
        reference,
        performance_diagnostics=diagnostics,
    )


def _closest_point_on_triangle(
    point: np.ndarray, first: np.ndarray, second: np.ndarray, third: np.ndarray
) -> np.ndarray:
    """Ericson's closest-point regions for one triangle."""

    ab = second - first
    ac = third - first
    ap = point - first
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return first
    bp = point - second
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return second
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return first + (d1 / (d1 - d3)) * ab
    cp = point - third
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return third
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return first + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return second + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (third - second)
    denominator = 1.0 / (va + vb + vc)
    return first + (vb * denominator) * ab + (vc * denominator) * ac


def _triangle_box_overlap(triangle: np.ndarray, center: np.ndarray, half: np.ndarray) -> bool:
    """Exact triangle/AABB SAT test for one Mobile dual-control prism."""

    vertices = np.asarray(triangle, dtype=np.float64) - center
    edges = (vertices[1] - vertices[0], vertices[2] - vertices[1], vertices[0] - vertices[2])
    axes = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.cross(edges[0], edges[1]),
    ]
    box_axes = axes[:3]
    axes.extend(np.cross(edge, axis) for edge in edges for axis in box_axes)
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= _EPS:
            continue
        unit = axis / norm
        projected = vertices @ unit
        radius = float(np.dot(half, np.abs(unit)))
        if float(np.min(projected)) > radius + _EPS or float(np.max(projected)) < -radius - _EPS:
            return False
    return True


def _point_inside_closed_mesh(point: np.ndarray, triangles: np.ndarray) -> bool:
    """Odd/even +X ray test with a non-axis-aligned perturbation."""

    direction = np.asarray([1.0, 0.371390676, 0.217831], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    hits: list[float] = []
    for triangle in triangles:
        first, second, third = triangle
        edge1 = second - first
        edge2 = third - first
        pvec = np.cross(direction, edge2)
        determinant = float(np.dot(edge1, pvec))
        if abs(determinant) <= _EPS:
            continue
        inverse = 1.0 / determinant
        tvec = point - first
        u = float(np.dot(tvec, pvec) * inverse)
        if u < -_EPS or u > 1.0 + _EPS:
            continue
        qvec = np.cross(tvec, edge1)
        v = float(np.dot(direction, qvec) * inverse)
        if v < -_EPS or u + v > 1.0 + _EPS:
            continue
        distance = float(np.dot(edge2, qvec) * inverse)
        if distance > _EPS:
            hits.append(distance)
    # Co-planar triangle pairs can report the same surface crossing twice.
    unique = []
    for value in sorted(hits):
        if not unique or abs(value - unique[-1]) > 1.0e-9:
            unique.append(value)
    return bool(len(unique) % 2)
