"""GPU-resident exact CAD contact support for already-Mobile material.

This module is an architectural port of :mod:`tool_mobile_contact`; it does
not define a second contact law.  Static bucket topology is uploaded once,
the rigid vertices are transformed once per frame, and one Warp thread tests
one wet Mobile dual-control-volume prism against the complete closed CAD mesh.
The accepted support is written directly into the persistent arrays consumed
by ``WarpProductionMobileV2Solver``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ..tools import assert_open_bucket_physical_contact


_KERNELS: dict[int, tuple[Any, Any]] = {}
_EPS = 1.0e-12


@dataclass(frozen=True)
class DeviceToolMobileContactSupport:
    """Scalar host handle to contact fields that remain authoritative on DEVICE."""

    runtime_identity: int
    cell_count: int
    mobile_volume_m3: float
    tool_reference_position_terrain_m: np.ndarray
    performance_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    bbox_yx: tuple[int, int, int, int] = (0, 0, 0, 0)
    geometry_classification: str = (
        "DEVICE_CAD_DERIVED_EXACT_TRIANGLE_PRISM_INTERSECTION_OR_CONTAINMENT"
    )
    device_resident: bool = True

    def __post_init__(self) -> None:
        reference = np.ascontiguousarray(
            np.asarray(self.tool_reference_position_terrain_m, dtype=np.float64)
        )
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError("[WarpToolMobileContact] tool reference must be finite (3,)")
        count = int(self.cell_count)
        volume = float(self.mobile_volume_m3)
        if count < 0 or not np.isfinite(volume) or volume < 0.0:
            raise ValueError("[WarpToolMobileContact] invalid scalar support summary")
        reference.setflags(write=False)
        object.__setattr__(self, "cell_count", count)
        object.__setattr__(self, "mobile_volume_m3", volume)
        object.__setattr__(self, "tool_reference_position_terrain_m", reference)
        object.__setattr__(
            self, "performance_diagnostics", MappingProxyType(dict(self.performance_diagnostics))
        )
        bbox = tuple(int(value) for value in self.bbox_yx)
        if len(bbox) != 4:
            raise ValueError("[WarpToolMobileContact] bbox_yx must contain four integers")
        object.__setattr__(self, "bbox_yx", bbox)


def _kernels(wp: Any) -> tuple[Any, Any]:
    cached = _KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.func
    def axis_separates(
        v0: wp.vec3d, v1: wp.vec3d, v2: wp.vec3d, half: wp.vec3d, axis: wp.vec3d
    ):
        length = wp.length(axis)
        if length <= wp.float64(_EPS):
            return False
        unit = axis / length
        p0 = wp.dot(v0, unit)
        p1 = wp.dot(v1, unit)
        p2 = wp.dot(v2, unit)
        minimum = wp.min(p0, wp.min(p1, p2))
        maximum = wp.max(p0, wp.max(p1, p2))
        radius = (
            half[0] * wp.abs(unit[0])
            + half[1] * wp.abs(unit[1])
            + half[2] * wp.abs(unit[2])
        )
        return minimum > radius + wp.float64(_EPS) or maximum < -radius - wp.float64(_EPS)

    @wp.func
    def triangle_box_overlap(
        a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, center: wp.vec3d, half: wp.vec3d
    ):
        v0 = a - center
        v1 = b - center
        v2 = c - center
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2
        xaxis = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
        yaxis = wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))
        zaxis = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
        if axis_separates(v0, v1, v2, half, xaxis):
            return False
        if axis_separates(v0, v1, v2, half, yaxis):
            return False
        if axis_separates(v0, v1, v2, half, zaxis):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e0, e1)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e0, xaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e0, yaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e0, zaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e1, xaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e1, yaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e1, zaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e2, xaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e2, yaxis)):
            return False
        if axis_separates(v0, v1, v2, half, wp.cross(e2, zaxis)):
            return False
        return True

    @wp.func
    def ray_hit_distance(point: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
        direction = wp.normalize(
            wp.vec3d(wp.float64(1.0), wp.float64(0.371390676), wp.float64(0.217831))
        )
        edge1 = b - a
        edge2 = c - a
        pvec = wp.cross(direction, edge2)
        determinant = wp.dot(edge1, pvec)
        if wp.abs(determinant) <= wp.float64(_EPS):
            return wp.float64(-1.0)
        inverse = wp.float64(1.0) / determinant
        tvec = point - a
        u = wp.dot(tvec, pvec) * inverse
        if u < -wp.float64(_EPS) or u > wp.float64(1.0) + wp.float64(_EPS):
            return wp.float64(-1.0)
        qvec = wp.cross(tvec, edge1)
        v = wp.dot(direction, qvec) * inverse
        if v < -wp.float64(_EPS) or u + v > wp.float64(1.0) + wp.float64(_EPS):
            return wp.float64(-1.0)
        distance = wp.dot(edge2, qvec) * inverse
        if distance <= wp.float64(_EPS):
            return wp.float64(-1.0)
        return distance

    @wp.func
    def closest_point_triangle(
        point: wp.vec3d, first: wp.vec3d, second: wp.vec3d, third: wp.vec3d
    ):
        ab = second - first
        ac = third - first
        ap = point - first
        d1 = wp.dot(ab, ap)
        d2 = wp.dot(ac, ap)
        if d1 <= wp.float64(0.0) and d2 <= wp.float64(0.0):
            return first
        bp = point - second
        d3 = wp.dot(ab, bp)
        d4 = wp.dot(ac, bp)
        if d3 >= wp.float64(0.0) and d4 <= d3:
            return second
        vc = d1 * d4 - d3 * d2
        if vc <= wp.float64(0.0) and d1 >= wp.float64(0.0) and d3 <= wp.float64(0.0):
            return first + (d1 / (d1 - d3)) * ab
        cp = point - third
        d5 = wp.dot(ab, cp)
        d6 = wp.dot(ac, cp)
        if d6 >= wp.float64(0.0) and d5 <= d6:
            return third
        vb = d5 * d2 - d1 * d6
        if vb <= wp.float64(0.0) and d2 >= wp.float64(0.0) and d6 <= wp.float64(0.0):
            return first + (d2 / (d2 - d6)) * ac
        va = d3 * d6 - d5 * d4
        if va <= wp.float64(0.0) and d4 - d3 >= wp.float64(0.0) and d5 - d6 >= wp.float64(0.0):
            return second + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (third - second)
        denominator = wp.float64(1.0) / (va + vb + vc)
        return first + (vb * denominator) * ab + (vc * denominator) * ac

    @wp.kernel
    def transform_vertices(
        local_x: wp.array(dtype=wp.float64),
        local_y: wp.array(dtype=wp.float64),
        local_z: wp.array(dtype=wp.float64),
        world_x: wp.array(dtype=wp.float64),
        world_y: wp.array(dtype=wp.float64),
        world_z: wp.array(dtype=wp.float64),
        r00: wp.float64, r01: wp.float64, r02: wp.float64, tx: wp.float64,
        r10: wp.float64, r11: wp.float64, r12: wp.float64, ty: wp.float64,
        r20: wp.float64, r21: wp.float64, r22: wp.float64, tz: wp.float64,
    ):
        i = wp.tid()
        x = local_x[i]
        y = local_y[i]
        z = local_z[i]
        world_x[i] = r00 * x + r01 * y + r02 * z + tx
        world_y[i] = r10 * x + r11 * y + r12 * z + ty
        world_z[i] = r20 * x + r21 * y + r22 * z + tz

    @wp.kernel
    def exact_contact(
        b_eff: wp.array(dtype=wp.float64),
        mobile: wp.array(dtype=wp.float64),
        weights: wp.array(dtype=wp.float64),
        world_x: wp.array(dtype=wp.float64),
        world_y: wp.array(dtype=wp.float64),
        world_z: wp.array(dtype=wp.float64),
        faces: wp.array(dtype=wp.int32),
        physical_face_mask: wp.array(dtype=wp.int32),
        face_count: int,
        row0: int, row1: int, col0: int, col1: int,
        rows: int, cols: int,
        dx: wp.float64, dy: wp.float64,
        origin_x: wp.float64, origin_y: wp.float64,
        reference_x: wp.float64, reference_y: wp.float64, reference_z: wp.float64,
        linear_x: wp.float64, linear_y: wp.float64, linear_z: wp.float64,
        omega_x: wp.float64, omega_y: wp.float64, omega_z: wp.float64,
        contact_mask: wp.array(dtype=wp.int32),
        normal_x: wp.array(dtype=wp.float64),
        normal_y: wp.array(dtype=wp.float64),
        normal3_x: wp.array(dtype=wp.float64),
        normal3_y: wp.array(dtype=wp.float64),
        normal3_z: wp.array(dtype=wp.float64),
        normal_terrain_x: wp.array(dtype=wp.float64),
        normal_terrain_y: wp.array(dtype=wp.float64),
        normal_terrain_z: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        velocity_z: wp.array(dtype=wp.float64),
        point_x: wp.array(dtype=wp.float64),
        point_y: wp.array(dtype=wp.float64),
        point_z: wp.array(dtype=wp.float64),
        signed_distance: wp.array(dtype=wp.float64),
        closest_face_index: wp.array(dtype=wp.int32),
        material_mask: wp.array(dtype=wp.int32),
        diagnostics_i: wp.array(dtype=wp.int32),
        diagnostics_f: wp.array(dtype=wp.float64),
    ):
        local = wp.tid()
        width = col1 - col0
        row = row0 + local // width
        col = col0 + local - (local // width) * width
        if row >= row1 or col >= col1:
            return
        index = row * cols + col
        depth = mobile[index]
        if depth <= wp.float64(0.0):
            return
        wp.atomic_add(diagnostics_i, 0, 1)  # wet / geometry candidates

        x = origin_x + wp.float64(col) * dx
        y = origin_y + wp.float64(row) * dy
        x_low = x - wp.float64(0.5) * dx
        x_high = x + wp.float64(0.5) * dx
        y_low = y - wp.float64(0.5) * dy
        y_high = y + wp.float64(0.5) * dy
        if col == 0:
            x_low = x
        if col == cols - 1:
            x_high = x
        if row == 0:
            y_low = y
        if row == rows - 1:
            y_high = y
        z_low = b_eff[index]
        z_high = z_low + depth
        center = wp.vec3d(
            wp.float64(0.5) * (x_low + x_high),
            wp.float64(0.5) * (y_low + y_high),
            wp.float64(0.5) * (z_low + z_high),
        )
        half = wp.vec3d(
            wp.float64(0.5) * (x_high - x_low),
            wp.float64(0.5) * (y_high - y_low),
            wp.float64(0.5) * (z_high - z_low),
        )

        intersects = bool(False)
        physical_intersects = bool(False)
        face = int(0)
        while face < face_count:
            ia = faces[3 * face]
            ib = faces[3 * face + 1]
            ic = faces[3 * face + 2]
            a = wp.vec3d(world_x[ia], world_y[ia], world_z[ia])
            b = wp.vec3d(world_x[ib], world_y[ib], world_z[ib])
            c = wp.vec3d(world_x[ic], world_y[ic], world_z[ic])
            wp.atomic_add(diagnostics_i, 1, 1)
            if triangle_box_overlap(a, b, c, center, half):
                intersects = bool(True)
                if physical_face_mask[face] != 0:
                    physical_intersects = bool(True)
            face = face + 1

        # Match the CPU oracle's unique positive ray-crossing semantics.  A
        # prior-face scan removes duplicate co-planar triangle hits without a
        # per-thread dynamic list or an approximate cavity primitive.
        unique_hits = int(0)
        face = int(0)
        while face < face_count:
            ia = faces[3 * face]
            ib = faces[3 * face + 1]
            ic = faces[3 * face + 2]
            a = wp.vec3d(world_x[ia], world_y[ia], world_z[ia])
            b = wp.vec3d(world_x[ib], world_y[ib], world_z[ib])
            c = wp.vec3d(world_x[ic], world_y[ic], world_z[ic])
            distance = ray_hit_distance(center, a, b, c)
            wp.atomic_add(diagnostics_i, 2, 1)
            if distance > wp.float64(0.0):
                duplicate = bool(False)
                prior = int(0)
                while prior < face:
                    ja = faces[3 * prior]
                    jb = faces[3 * prior + 1]
                    jc = faces[3 * prior + 2]
                    pa = wp.vec3d(world_x[ja], world_y[ja], world_z[ja])
                    pb = wp.vec3d(world_x[jb], world_y[jb], world_z[jb])
                    pc = wp.vec3d(world_x[jc], world_y[jc], world_z[jc])
                    previous_distance = ray_hit_distance(center, pa, pb, pc)
                    if previous_distance > wp.float64(0.0) and wp.abs(distance - previous_distance) <= wp.float64(1.0e-9):
                        duplicate = bool(True)
                    prior = prior + 1
                if not duplicate:
                    unique_hits = unique_hits + 1
            face = face + 1
        inside = unique_hits % 2 == 1
        wp.atomic_add(diagnostics_i, 3, 1)
        if not physical_intersects and not inside:
            return

        minimum_distance = wp.float64(1.0e300)
        closest = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        closest_face = int(0)
        face = int(0)
        while face < face_count:
            if physical_face_mask[face] == 0:
                face = face + 1
                continue
            ia = faces[3 * face]
            ib = faces[3 * face + 1]
            ic = faces[3 * face + 2]
            a = wp.vec3d(world_x[ia], world_y[ia], world_z[ia])
            b = wp.vec3d(world_x[ib], world_y[ib], world_z[ib])
            c = wp.vec3d(world_x[ic], world_y[ic], world_z[ic])
            candidate = closest_point_triangle(center, a, b, c)
            candidate_distance = wp.length(center - candidate)
            wp.atomic_add(diagnostics_i, 4, 1)
            if candidate_distance < minimum_distance:
                minimum_distance = candidate_distance
                closest = candidate
                closest_face = face
            face = face + 1

        normal = wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        if minimum_distance > wp.float64(_EPS):
            normal = (center - closest) / minimum_distance
        else:
            ia = faces[3 * closest_face]
            ib = faces[3 * closest_face + 1]
            ic = faces[3 * closest_face + 2]
            a = wp.vec3d(world_x[ia], world_y[ia], world_z[ia])
            b = wp.vec3d(world_x[ib], world_y[ib], world_z[ib])
            c = wp.vec3d(world_x[ic], world_y[ic], world_z[ic])
            face_normal = wp.normalize(wp.cross(b - a, c - a))
            normal = face_normal
            if inside:
                normal = -face_normal
        horizontal_length = wp.sqrt(normal[0] * normal[0] + normal[1] * normal[1])

        reference = wp.vec3d(reference_x, reference_y, reference_z)
        linear = wp.vec3d(linear_x, linear_y, linear_z)
        omega = wp.vec3d(omega_x, omega_y, omega_z)
        surface_velocity = linear + wp.cross(omega, closest - reference)
        contact_mask[index] = 1
        material_mask[index] = 1
        # Keep the historical normalized XY projection strictly for the CPU
        # oracle/legacy diagnostics. Production contact consumes ``normal3``.
        # Thus a nearly vertical face can no longer become a unit horizontal
        # bulldozing direction while old support-format tests remain valid.
        if horizontal_length > wp.float64(_EPS):
            normal_x[index] = normal[0] / horizontal_length
            normal_y[index] = normal[1] / horizontal_length
        else:
            normal_x[index] = wp.float64(0.0)
            normal_y[index] = wp.float64(0.0)
        normal3_x[index] = normal[0]
        normal3_y[index] = normal[1]
        normal3_z[index] = normal[2]
        normal_terrain_x[index] = normal[0]
        normal_terrain_y[index] = normal[1]
        normal_terrain_z[index] = normal[2]
        velocity_x[index] = surface_velocity[0]
        velocity_y[index] = surface_velocity[1]
        velocity_z[index] = surface_velocity[2]
        point_x[index] = closest[0]
        point_y[index] = closest[1]
        point_z[index] = closest[2]
        signed_distance[index] = minimum_distance
        if inside:
            signed_distance[index] = -minimum_distance
        # Acceptance-only surface attribution.  The production impulse kernel
        # does not consume this field; retaining the exact closest triangle
        # avoids inferring an inner/outer surface from a later projection.
        closest_face_index[index] = closest_face
        wp.atomic_add(diagnostics_i, 5, 1)
        wp.atomic_add(diagnostics_f, 0, weights[index] * depth)

    _KERNELS[id(wp)] = (transform_vertices, exact_contact)
    return transform_vertices, exact_contact


class WarpExactToolMobileContactGeometry:
    """Exact closed-containment/contact builder with canonical open-mouth mask."""

    backend_name = "GPU_WARP_EXACT_TOOL_MOBILE_CONTACT_SHARED_STATE"

    def __init__(self, *, state: Any, descriptor: Any) -> None:
        geometry = None if descriptor is None else descriptor.bucket_geometry
        if geometry is None:
            raise ValueError("[WarpToolMobileContact] authoritative bucket geometry required")
        contact_geometry = assert_open_bucket_physical_contact(geometry)
        self.state = state
        self.runtime = state.runtime
        self.grid = state.grid
        self.descriptor = descriptor
        vertices = np.asarray(geometry.interior_vertices_local, dtype=np.float64)
        faces = np.asarray(geometry.interior_faces, dtype=np.int32)
        raw = np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        )
        valid = np.linalg.norm(raw, axis=1) > _EPS
        self.faces = np.ascontiguousarray(faces[valid], dtype=np.int32)
        self.physical_contact_face_mask = np.ascontiguousarray(
            np.asarray(contact_geometry.containment_face_mask, dtype=np.int32)[valid],
            dtype=np.int32,
        )
        self.contact_geometry_contract = str(
            contact_geometry.metadata["geometry_contract"]
        )
        self.vertex_count = int(vertices.shape[0])
        self.face_count = int(self.faces.shape[0])
        self._prefix = f"tool_contact_{id(self):x}"
        wp = self.runtime.wp
        for suffix, value in (
            ("local_x", vertices[:, 0]),
            ("local_y", vertices[:, 1]),
            ("local_z", vertices[:, 2]),
            ("faces", self.faces.ravel()),
            ("physical_face_mask", self.physical_contact_face_mask),
        ):
            dtype = wp.int32 if suffix in {"faces", "physical_face_mask"} else wp.float64
            self.runtime.upload(f"{self._prefix}_{suffix}", value, dtype=dtype)
        for suffix in ("world_x", "world_y", "world_z"):
            self.runtime.zeros(f"{self._prefix}_{suffix}", self.vertex_count, dtype=wp.float64)
        self.runtime.zeros(f"{self._prefix}_diag_i", 8, dtype=wp.int32)
        self.runtime.zeros(f"{self._prefix}_diag_f", 4, dtype=wp.float64)
        if "v2_tool_contact_mask" not in self.runtime.arrays:
            self.runtime.zeros("v2_tool_contact_mask", state.size, dtype=wp.int32)
        if "v2_tool_closest_face_index" not in self.runtime.arrays:
            self.runtime.zeros("v2_tool_closest_face_index", state.size, dtype=wp.int32)
        for name in (
            "v2_tool_normal_x", "v2_tool_normal_y",
            "v2_tool_normal3_x", "v2_tool_normal3_y", "v2_tool_normal3_z",
            "v2_tool_normal_terrain_x", "v2_tool_normal_terrain_y",
            "v2_tool_normal_terrain_z",
            "v2_tool_velocity_x", "v2_tool_velocity_y", "v2_tool_velocity_z",
            "v2_tool_contact_point_x", "v2_tool_contact_point_y",
            "v2_tool_contact_point_z", "v2_tool_signed_distance",
        ):
            if name not in self.runtime.arrays:
                self.runtime.zeros(name, state.size, dtype=wp.float64)

    def clear(self) -> None:
        """Clear device evidence after the fused Mobile source consumed it."""

        for name in ("v2_tool_contact_mask", "material_mask"):
            self.runtime.arrays[name].zero_()

    def compute(
        self,
        *,
        bbox_yx: tuple[int, int, int, int],
        tool_state: Any,
    ) -> DeviceToolMobileContactSupport:
        row0, row1, col0, col1 = (int(value) for value in bbox_yx)
        if row1 <= row0 or col1 <= col0:
            row0 = row1 = col0 = col1 = 0
        runtime = self.runtime
        wp = runtime.wp
        static_h2d_before = runtime.telemetry.h2d_bytes
        for name in (
            "v2_tool_contact_mask", "v2_tool_normal_x", "v2_tool_normal_y",
            "v2_tool_normal3_x", "v2_tool_normal3_y", "v2_tool_normal3_z",
            "v2_tool_normal_terrain_x", "v2_tool_normal_terrain_y",
            "v2_tool_normal_terrain_z", "v2_tool_velocity_x", "v2_tool_velocity_y",
            "v2_tool_velocity_z",
            "v2_tool_contact_point_x", "v2_tool_contact_point_y",
            "v2_tool_contact_point_z", "v2_tool_signed_distance", "material_mask",
            "v2_tool_closest_face_index",
            f"{self._prefix}_diag_i", f"{self._prefix}_diag_f",
        ):
            runtime.arrays[name].zero_()
        transform, exact = _kernels(wp)
        pose = np.asarray(tool_state.pose_terrain, dtype=np.float64)
        reference = np.asarray(pose[:3, 3], dtype=np.float64)
        linear = np.asarray(tool_state.linear_velocity, dtype=np.float64)
        omega = np.asarray(tool_state.angular_velocity, dtype=np.float64)

        start = perf_counter()
        runtime.launch(transform, dim=self.vertex_count, inputs=[
            runtime.arrays[f"{self._prefix}_local_x"],
            runtime.arrays[f"{self._prefix}_local_y"],
            runtime.arrays[f"{self._prefix}_local_z"],
            runtime.arrays[f"{self._prefix}_world_x"],
            runtime.arrays[f"{self._prefix}_world_y"],
            runtime.arrays[f"{self._prefix}_world_z"],
            *[float(pose[i, j]) for i, j in (
                (0, 0), (0, 1), (0, 2), (0, 3),
                (1, 0), (1, 1), (1, 2), (1, 3),
                (2, 0), (2, 1), (2, 2), (2, 3),
            )],
        ])
        runtime.synchronize()
        transform_ms = (perf_counter() - start) * 1_000.0

        geometry_start = perf_counter()
        dim = max(0, (row1 - row0) * (col1 - col0))
        if dim:
            runtime.launch(exact, dim=dim, inputs=[
                runtime.arrays["b_eff"], runtime.arrays["mobile"], runtime.arrays["weights"],
                runtime.arrays[f"{self._prefix}_world_x"],
                runtime.arrays[f"{self._prefix}_world_y"],
                runtime.arrays[f"{self._prefix}_world_z"],
                runtime.arrays[f"{self._prefix}_faces"],
                runtime.arrays[f"{self._prefix}_physical_face_mask"], self.face_count,
                row0, row1, col0, col1, self.grid.ny, self.grid.nx,
                self.grid.dx, self.grid.dy, self.grid.origin_x, self.grid.origin_y,
                *reference.tolist(), *linear.tolist(), *omega.tolist(),
                runtime.arrays["v2_tool_contact_mask"],
                runtime.arrays["v2_tool_normal_x"], runtime.arrays["v2_tool_normal_y"],
                runtime.arrays["v2_tool_normal3_x"],
                runtime.arrays["v2_tool_normal3_y"],
                runtime.arrays["v2_tool_normal3_z"],
                runtime.arrays["v2_tool_normal_terrain_x"],
                runtime.arrays["v2_tool_normal_terrain_y"],
                runtime.arrays["v2_tool_normal_terrain_z"],
                runtime.arrays["v2_tool_velocity_x"],
                runtime.arrays["v2_tool_velocity_y"], runtime.arrays["v2_tool_velocity_z"],
                runtime.arrays["v2_tool_contact_point_x"],
                runtime.arrays["v2_tool_contact_point_y"],
                runtime.arrays["v2_tool_contact_point_z"],
                runtime.arrays["v2_tool_signed_distance"],
                runtime.arrays["v2_tool_closest_face_index"],
                runtime.arrays["material_mask"],
                runtime.arrays[f"{self._prefix}_diag_i"],
                runtime.arrays[f"{self._prefix}_diag_f"],
            ])
        runtime.synchronize()
        geometry_ms = (perf_counter() - geometry_start) * 1_000.0

        readback_start = perf_counter()
        diag_i = np.asarray(runtime.arrays[f"{self._prefix}_diag_i"].numpy(), dtype=np.int64)
        diag_f = np.asarray(runtime.arrays[f"{self._prefix}_diag_f"].numpy(), dtype=np.float64)
        runtime.telemetry.record_d2h(diag_i)
        runtime.telemetry.record_d2h(diag_f)
        scalar_d2h_ms = (perf_counter() - readback_start) * 1_000.0
        diagnostics = {
            "execution_device": "WARP_DEVICE_RESIDENT",
            "mobile_candidate_count": int(diag_i[0]),
            "geometry_candidate_count": int(diag_i[0]),
            "cad_triangle_count": self.face_count,
            "containment_triangle_count": self.face_count,
            "physical_contact_triangle_count": int(
                np.count_nonzero(self.physical_contact_face_mask)
            ),
            "mouth_cap_physical_contact_triangle_count": 0,
            "bucket_contact_geometry_contract": self.contact_geometry_contract,
            "broadphase_pair_count": int(diag_i[0]) * self.face_count,
            "triangle_aabb_test_count": int(diag_i[1]),
            "ray_triangle_test_count": int(diag_i[2]),
            "exact_test_count": int(diag_i[1] + diag_i[2]),
            "containment_test_count": int(diag_i[3]),
            "closest_point_query_count": int(diag_i[4]),
            "accepted_contact_count": int(diag_i[5]),
            "candidate_build_ms": 0.0,
            "candidate_build_timing_scope": "FUSED_IN_GPU_GEOMETRY_KERNEL",
            "cad_transform_ms": float(transform_ms),
            "gpu_geometry_ms": float(geometry_ms),
            "contact_support_compaction_ms": 0.0,
            "contact_support_compaction_scope": "DIRECT_RESIDENT_MASK_NO_COMPACTION",
            "contact_h2d_ms": 0.0,
            "contact_h2d_bytes": int(runtime.telemetry.h2d_bytes - static_h2d_before),
            "contact_h2d_transfer_count": 0,
            "scalar_d2h_ms": float(scalar_d2h_ms),
            "tool_mobile_total_ms": float(transform_ms + geometry_ms + scalar_d2h_ms),
            "static_vertex_count": self.vertex_count,
            "static_topology_persistent": True,
            "direct_device_mobile_consumption": True,
        }
        return DeviceToolMobileContactSupport(
            runtime_identity=id(runtime),
            cell_count=int(diag_i[5]),
            mobile_volume_m3=float(diag_f[0]),
            tool_reference_position_terrain_m=reference,
            performance_diagnostics=diagnostics,
            bbox_yx=(row0, row1, col0, col1),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "execution_device": self.runtime.device,
            "vertex_count": self.vertex_count,
            "triangle_count": self.face_count,
            "bucket_contact_geometry_contract": self.contact_geometry_contract,
            "topology_persistent_on_device": True,
            "contact_fields_resident": True,
        }

    def download_support_for_oracle(self) -> dict[str, np.ndarray]:
        """Download support fields solely for deterministic CPU-oracle tests.

        Production code must pass :class:`DeviceToolMobileContactSupport`
        directly to Mobile V2 and must not call this acceptance-only method.
        """

        mask = self.runtime.download("v2_tool_contact_mask").astype(bool)
        indices = np.flatnonzero(mask).astype(np.int32)
        def selected(name: str) -> np.ndarray:
            return np.asarray(self.runtime.download(name), dtype=np.float64)[indices]
        return {
            "flat_indices": indices,
            "closest_points_terrain_m": np.column_stack((
                selected("v2_tool_contact_point_x"),
                selected("v2_tool_contact_point_y"),
                selected("v2_tool_contact_point_z"),
            )),
            "outward_normals_xy": np.column_stack((
                selected("v2_tool_normal_x"), selected("v2_tool_normal_y")
            )),
            "outward_normals_terrain": np.column_stack((
                selected("v2_tool_normal_terrain_x"),
                selected("v2_tool_normal_terrain_y"),
                selected("v2_tool_normal_terrain_z"),
            )),
            "tool_surface_velocity_terrain_m_s": np.column_stack((
                selected("v2_tool_velocity_x"),
                selected("v2_tool_velocity_y"),
                selected("v2_tool_velocity_z"),
            )),
            "cavity_signed_distance_m": selected("v2_tool_signed_distance"),
            "closest_face_index": np.asarray(
                self.runtime.download("v2_tool_closest_face_index"), dtype=np.int32
            )[indices],
        }
