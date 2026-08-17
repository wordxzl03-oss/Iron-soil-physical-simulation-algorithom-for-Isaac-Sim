"""Authoritative bucket mouth, separation plate and closed interior geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_EPS = 1.0e-12


class GeometrySource(str, Enum):
    USD_MESH = "USD_MESH"
    USD_MESH_MARKERS = "USD_MESH_MARKERS"
    USD_MESH_GEOMETRY_INFERENCE = "USD_MESH_GEOMETRY_INFERENCE"
    MARKERS = "MARKERS"
    EXPLICIT_PROFILE = "EXPLICIT_PROFILE"
    LEGACY_FALLBACK = "LEGACY_FALLBACK"


class GeometryQuality(str, Enum):
    EXPLICIT = "EXPLICIT"
    REDUCED_ORDER = "REDUCED_ORDER"
    APPROXIMATE = "APPROXIMATE"


class CapacityMethod(str, Enum):
    CLOSED_MESH_VOLUME = "CLOSED_MESH_VOLUME"
    EXTRUDED_PROFILE = "EXTRUDED_PROFILE"
    CALIBRATED_VALUE = "CALIBRATED_VALUE"
    LEGACY_NOMINAL = "LEGACY_NOMINAL"


def _ro(value: np.ndarray, *, dtype: np.dtype | type = np.float64) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())
    result.setflags(write=False)
    return result


def _points(value: np.ndarray, *, name: str, minimum: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"[BucketGeometry] {name} must have shape (N,3), N>={minimum}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"[BucketGeometry] {name} contains NaN/Inf")
    return points


def _polygon_area_2d(polygon: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        first = q - p
        second = r - p
        return float(first[0] * second[1] - first[1] * second[0])

    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 < -_EPS and o3 * o4 < -_EPS


def _simple_polygon(polygon: np.ndarray) -> bool:
    count = len(polygon)
    for first in range(count):
        a, b = polygon[first], polygon[(first + 1) % count]
        for second in range(first + 1, count):
            if second in {first, (first + 1) % count} or (second + 1) % count == first:
                continue
            c, d = polygon[second], polygon[(second + 1) % count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    error = float(np.max(np.abs((points - centroid) @ normal)))
    return centroid, normal, error


def _polygon_area_3d(points: np.ndarray, normal: np.ndarray) -> float:
    cross_sum = np.zeros(3, dtype=np.float64)
    for current, following in zip(points, np.roll(points, -1, axis=0)):
        cross_sum += np.cross(current, following)
    return float(0.5 * abs(np.dot(cross_sum, normal)))


@dataclass(frozen=True)
class BucketGeometryDescriptor:
    """One validated geometry used by failure, intake and payload capacity.

    The implemented closed interior is a lateral extrusion of a simple profile.
    A future USD mesh extractor can populate the same contract with
    ``CLOSED_MESH_VOLUME`` after proving watertightness and winding.
    """

    cutting_edge_local: np.ndarray
    lip_local: np.ndarray
    bottom_plate_polygon_local: np.ndarray
    top_edge_local: np.ndarray
    mouth_polygon_local: np.ndarray
    interior_profile_local: np.ndarray
    back_wall_polygon_local: np.ndarray
    left_side_wall_local: np.ndarray
    right_side_wall_local: np.ndarray
    interior_vertices_local: np.ndarray
    interior_faces: np.ndarray
    geometric_capacity_m3: float
    rated_capacity_m3: float | None
    interior_width_m: float | None
    geometry_source: GeometrySource | str
    geometry_quality: GeometryQuality | str
    capacity_method: CapacityMethod | str
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        cutting = _points(self.cutting_edge_local, name="cutting_edge_local", minimum=2)
        lip = _points(self.lip_local, name="lip_local", minimum=2)
        bottom = _points(self.bottom_plate_polygon_local, name="bottom_plate_polygon_local", minimum=4)
        top = _points(self.top_edge_local, name="top_edge_local", minimum=2)
        mouth = _points(self.mouth_polygon_local, name="mouth_polygon_local", minimum=4)
        profile = _points(self.interior_profile_local, name="interior_profile_local", minimum=3)
        back = _points(self.back_wall_polygon_local, name="back_wall_polygon_local", minimum=4)
        left = _points(self.left_side_wall_local, name="left_side_wall_local", minimum=3)
        right = _points(self.right_side_wall_local, name="right_side_wall_local", minimum=3)
        vertices = _points(self.interior_vertices_local, name="interior_vertices_local", minimum=6)
        faces = np.asarray(self.interior_faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
            raise ValueError("[BucketGeometry] interior_faces must have shape (M,3)")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("[BucketGeometry] interior face index is out of range")
        source = GeometrySource(self.geometry_source)
        quality = GeometryQuality(self.geometry_quality)
        method = CapacityMethod(self.capacity_method)
        capacity = float(self.geometric_capacity_m3)
        rated = None if self.rated_capacity_m3 is None else float(self.rated_capacity_m3)
        if not np.isfinite(capacity) or capacity <= 0.0:
            raise ValueError("[BucketGeometry] geometric capacity must be finite/positive")
        if rated is not None and (not np.isfinite(rated) or rated <= 0.0):
            raise ValueError("[BucketGeometry] rated capacity must be finite/positive")

        edge_vector = cutting[-1] - cutting[0]
        edge_length = float(np.linalg.norm(edge_vector))
        top_vector = top[-1] - top[0]
        if edge_length <= 1.0e-8 or np.linalg.norm(top_vector) <= 1.0e-8:
            raise ValueError("[BucketGeometry] cutting/top edges must have nonzero length")
        if abs(float(np.dot(edge_vector, top_vector))) / (
            np.linalg.norm(edge_vector) * np.linalg.norm(top_vector)
        ) < 0.98:
            raise ValueError("[BucketGeometry] top edge must be parallel to cutting edge")
        if np.dot(edge_vector, top_vector) <= 0.0:
            raise ValueError("[BucketGeometry] cutting/top edges must share left-to-right ordering")
        if not (
            np.allclose(lip[0], cutting[0], atol=1.0e-9)
            and np.allclose(lip[-1], cutting[-1], atol=1.0e-9)
        ):
            raise ValueError("[BucketGeometry] lip must coincide with cutting-edge endpoints")
        if not any(np.allclose(lip[0], point, atol=1.0e-9) for point in mouth) or not any(
            np.allclose(lip[-1], point, atol=1.0e-9) for point in mouth
        ):
            raise ValueError("[BucketGeometry] lip must lie on the mouth boundary")

        mouth_centroid, mouth_plane_normal, mouth_error = _plane(mouth)
        if mouth_error > 1.0e-7:
            raise ValueError(f"[BucketGeometry] mouth polygon is non-planar; error={mouth_error}")
        mouth_area = _polygon_area_3d(mouth, mouth_plane_normal)
        if mouth_area <= 1.0e-10:
            raise ValueError("[BucketGeometry] mouth area must be positive")
        mouth_axis = edge_vector / edge_length
        mouth_up = np.cross(mouth_plane_normal, mouth_axis)
        if np.linalg.norm(mouth_up) <= _EPS:
            raise ValueError("[BucketGeometry] mouth local frame is degenerate")
        mouth_up /= np.linalg.norm(mouth_up)
        mouth_2d = np.column_stack(((mouth - mouth_centroid) @ mouth_axis, (mouth - mouth_centroid) @ mouth_up))
        if not _simple_polygon(mouth_2d):
            raise ValueError("[BucketGeometry] mouth polygon is self-intersecting")

        profile_yz = profile[:, 1:3]
        if np.ptp(profile[:, 0]) > 1.0e-7:
            raise ValueError("[BucketGeometry] extruded interior profile must lie in one x-plane")
        if not _simple_polygon(profile_yz):
            raise ValueError("[BucketGeometry] interior profile is self-intersecting")
        profile_area = abs(_polygon_area_2d(profile_yz))
        width = edge_length if self.interior_width_m is None else float(self.interior_width_m)
        if not np.isfinite(width) or width <= 0.0 or width > edge_length + 1.0e-9:
            raise ValueError(
                "[BucketGeometry] interior_width_m must be finite/positive and no "
                "larger than the cutting edge"
            )
        expected_capacity = profile_area * width
        if expected_capacity <= 1.0e-10:
            raise ValueError("[BucketGeometry] interior profile has zero area")
        if method is CapacityMethod.EXTRUDED_PROFILE and not np.isclose(
            capacity, expected_capacity, rtol=1.0e-9, atol=1.0e-10
        ):
            raise ValueError(
                "[BucketGeometry] EXTRUDED_PROFILE capacity disagrees with profile*width"
            )
        if not self._watertight(faces):
            raise ValueError("[BucketGeometry] closed interior faces are not watertight")
        if not self._consistently_wound(faces):
            raise ValueError("[BucketGeometry] closed interior faces have inconsistent winding")
        closed_volume = self._signed_volume(vertices, faces)
        if closed_volume <= 1.0e-12:
            raise ValueError("[BucketGeometry] closed interior winding must enclose positive volume")
        if method is CapacityMethod.EXTRUDED_PROFILE and not np.isclose(
            closed_volume, expected_capacity, rtol=1.0e-9, atol=1.0e-10
        ):
            raise ValueError("[BucketGeometry] closed interior volume disagrees with profile*width")
        if method is CapacityMethod.CLOSED_MESH_VOLUME and not np.isclose(
            capacity, closed_volume, rtol=1.0e-9, atol=1.0e-10
        ):
            raise ValueError("[BucketGeometry] CLOSED_MESH_VOLUME capacity disagrees with mesh")

        bottom_centroid, bottom_normal, bottom_error = _plane(bottom)
        if bottom_error > 1.0e-7 or _polygon_area_3d(bottom, bottom_normal) <= 1.0e-10:
            raise ValueError("[BucketGeometry] bottom plate must be planar with positive area")
        if np.dot(bottom_normal, np.asarray([0.0, 0.0, 1.0])) < 0.0:
            bottom_normal = -bottom_normal
        interior_centroid = np.mean(vertices, axis=0)
        if np.dot(mouth_plane_normal, interior_centroid - mouth_centroid) < 0.0:
            mouth_plane_normal = -mouth_plane_normal
        mouth_up = np.cross(mouth_plane_normal, mouth_axis)
        mouth_up /= np.linalg.norm(mouth_up)
        penetration = np.cross(bottom_normal, mouth_axis)
        penetration /= np.linalg.norm(penetration)
        rear_center = 0.5 * (bottom[0] + bottom[1])
        lip_center = 0.5 * (lip[0] + lip[-1])
        if np.dot(penetration, lip_center - rear_center) < 0.0:
            penetration = -penetration
        if np.dot(cutting[-1] - cutting[0], right[0] - left[0]) <= 0.0:
            raise ValueError("[BucketGeometry] left/right wall orientation is inconsistent")
        _, back_normal, back_error = _plane(back)
        if back_error > 1.0e-7 or _polygon_area_3d(back, back_normal) <= 1.0e-10:
            raise ValueError("[BucketGeometry] back wall must be planar with positive area")
        if not all(
            any(np.allclose(point, vertex, atol=1.0e-9) for vertex in vertices)
            for point in back
        ):
            raise ValueError("[BucketGeometry] back wall must belong to the closed interior")

        object.__setattr__(self, "cutting_edge_local", _ro(cutting))
        object.__setattr__(self, "lip_local", _ro(lip))
        object.__setattr__(self, "bottom_plate_polygon_local", _ro(bottom))
        object.__setattr__(self, "top_edge_local", _ro(top))
        object.__setattr__(self, "mouth_polygon_local", _ro(mouth))
        object.__setattr__(self, "interior_profile_local", _ro(profile))
        object.__setattr__(self, "back_wall_polygon_local", _ro(back))
        object.__setattr__(self, "left_side_wall_local", _ro(left))
        object.__setattr__(self, "right_side_wall_local", _ro(right))
        object.__setattr__(self, "interior_vertices_local", _ro(vertices))
        object.__setattr__(self, "interior_faces", _ro(faces, dtype=np.int64))
        object.__setattr__(self, "geometric_capacity_m3", capacity)
        object.__setattr__(self, "rated_capacity_m3", rated)
        object.__setattr__(self, "interior_width_m", width)
        object.__setattr__(self, "geometry_source", source)
        object.__setattr__(self, "geometry_quality", quality)
        object.__setattr__(self, "capacity_method", method)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "_mouth_centroid_local", _ro(mouth_centroid))
        object.__setattr__(self, "_mouth_normal_local", _ro(mouth_plane_normal))
        object.__setattr__(self, "_mouth_planarity_error_m", mouth_error)
        object.__setattr__(self, "_mouth_area_m2", mouth_area)
        mouth_frame = np.eye(4, dtype=np.float64)
        mouth_frame[:3, 0] = mouth_axis
        mouth_frame[:3, 1] = mouth_up
        mouth_frame[:3, 2] = mouth_plane_normal
        mouth_frame[:3, 3] = mouth_centroid
        object.__setattr__(self, "_mouth_local_frame", _ro(mouth_frame))
        object.__setattr__(self, "_bottom_normal_local", _ro(bottom_normal))
        object.__setattr__(self, "_penetration_direction_local", _ro(penetration))
        object.__setattr__(self, "_bottom_centroid_local", _ro(bottom_centroid))
        object.__setattr__(self, "_closed_mesh_volume_m3", closed_volume)

    @classmethod
    def from_extruded_profile(
        cls,
        *,
        cutting_edge_local: np.ndarray,
        bottom_profile_local: np.ndarray,
        interior_profile_local: np.ndarray,
        top_edge_local: np.ndarray,
        rated_capacity_m3: float | None,
        interior_width_m: float | None = None,
        geometry_source: GeometrySource | str,
        geometry_quality: GeometryQuality | str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BucketGeometryDescriptor":
        cutting = _points(cutting_edge_local, name="cutting_edge_local", minimum=2)
        bottom_profile = _points(bottom_profile_local, name="bottom_profile_local", minimum=2)
        profile = _points(interior_profile_local, name="interior_profile_local", minimum=3)
        top = _points(top_edge_local, name="top_edge_local", minimum=2)
        cutting_left_x = float(cutting[0, 0])
        cutting_right_x = float(cutting[-1, 0])
        if cutting_right_x <= cutting_left_x:
            raise ValueError("[BucketGeometry] cutting edge must be ordered left-to-right in +X")
        interior_width = (
            cutting_right_x - cutting_left_x
            if interior_width_m is None
            else float(interior_width_m)
        )
        if not np.isfinite(interior_width) or interior_width <= 0.0:
            raise ValueError("[BucketGeometry] interior width must be finite/positive")
        if interior_width > cutting_right_x - cutting_left_x + 1.0e-9:
            raise ValueError("[BucketGeometry] interior width cannot exceed cutting width")
        center_x = 0.5 * (cutting_left_x + cutting_right_x)
        left_x = center_x - 0.5 * interior_width
        right_x = center_x + 0.5 * interior_width
        left = np.column_stack((np.full(len(profile), left_x), profile[:, 1], profile[:, 2]))
        right = np.column_stack((np.full(len(profile), right_x), profile[:, 1], profile[:, 2]))
        rear = np.asarray(bottom_profile[0], dtype=np.float64)
        rear_left = rear.copy(); rear_left[0] = left_x
        rear_right = rear.copy(); rear_right[0] = right_x
        bottom = np.stack((rear_left, rear_right, cutting[-1], cutting[0]))
        mouth = np.stack((cutting[0], cutting[-1], top[-1], top[0]))
        back = np.stack((left[0], right[0], right[1], left[1]))
        vertices = np.vstack((left, right))
        faces = cls._extruded_faces(len(profile))
        if cls._signed_volume(vertices, faces) < 0.0:
            faces = faces[:, [0, 2, 1]]
        capacity = abs(_polygon_area_2d(profile[:, 1:3])) * interior_width
        return cls(
            cutting_edge_local=cutting,
            lip_local=np.stack((cutting[0], cutting[-1])),
            bottom_plate_polygon_local=bottom,
            top_edge_local=top,
            mouth_polygon_local=mouth,
            interior_profile_local=profile,
            back_wall_polygon_local=back,
            left_side_wall_local=left,
            right_side_wall_local=right,
            interior_vertices_local=vertices,
            interior_faces=faces,
            geometric_capacity_m3=capacity,
            rated_capacity_m3=rated_capacity_m3,
            interior_width_m=interior_width,
            geometry_source=geometry_source,
            geometry_quality=geometry_quality,
            capacity_method=CapacityMethod.EXTRUDED_PROFILE,
            metadata={} if metadata is None else metadata,
        )

    @staticmethod
    def _extruded_faces(count: int) -> np.ndarray:
        faces: list[tuple[int, int, int]] = []
        for index in range(1, count - 1):
            faces.append((0, index + 1, index))
            faces.append((count, count + index, count + index + 1))
        for index in range(count):
            following = (index + 1) % count
            faces.append((index, following, count + following))
            faces.append((index, count + following, count + index))
        return np.asarray(faces, dtype=np.int64)

    @staticmethod
    def _watertight(faces: np.ndarray) -> bool:
        edges: dict[tuple[int, int], int] = {}
        for face in faces:
            for start, end in zip(face, np.roll(face, -1)):
                edge = tuple(sorted((int(start), int(end))))
                edges[edge] = edges.get(edge, 0) + 1
        return bool(edges) and all(count == 2 for count in edges.values())

    @staticmethod
    def _consistently_wound(faces: np.ndarray) -> bool:
        directed: dict[tuple[int, int], int] = {}
        for face in faces:
            for start, end in zip(face, np.roll(face, -1)):
                edge = (int(start), int(end))
                directed[edge] = directed.get(edge, 0) + 1
        return bool(directed) and all(
            count == 1 and directed.get((end, start), 0) == 1
            for (start, end), count in directed.items()
        )

    @staticmethod
    def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
        volume = 0.0
        for first, second, third in faces:
            volume += float(
                np.dot(vertices[first], np.cross(vertices[second], vertices[third]))
            ) / 6.0
        return volume

    @property
    def cutting_edge_length_m(self) -> float:
        return float(np.linalg.norm(self.cutting_edge_local[-1] - self.cutting_edge_local[0]))

    @property
    def mouth_area_m2(self) -> float:
        return float(self._mouth_area_m2)

    @property
    def mouth_planarity_error_m(self) -> float:
        return float(self._mouth_planarity_error_m)

    @property
    def mouth_centroid_local(self) -> np.ndarray:
        return _ro(self._mouth_centroid_local)

    @property
    def mouth_normal_local(self) -> np.ndarray:
        return _ro(self._mouth_normal_local)

    @property
    def mouth_local_frame(self) -> np.ndarray:
        """``T_bucket_from_mouth`` with x along lip and z inward."""

        return _ro(self._mouth_local_frame)

    @property
    def mouth_plane_offset_local_m(self) -> float:
        return -float(np.dot(self.mouth_normal_local, self.mouth_centroid_local))

    @property
    def bottom_plate_normal_local(self) -> np.ndarray:
        return _ro(self._bottom_normal_local)

    @property
    def penetration_direction_local(self) -> np.ndarray:
        return _ro(self._penetration_direction_local)

    @property
    def separation_plane_direction_local(self) -> np.ndarray:
        """Primary separation-plate tangent used as Luengo rake direction."""

        return self.penetration_direction_local

    @property
    def effective_capacity_m3(self) -> float:
        if self.rated_capacity_m3 is None:
            return self.geometric_capacity_m3
        return float(min(self.geometric_capacity_m3, self.rated_capacity_m3))

    @property
    def closed_mesh_volume_m3(self) -> float:
        return float(self._closed_mesh_volume_m3)

    @property
    def legacy_fallback_used(self) -> bool:
        return self.geometry_source is GeometrySource.LEGACY_FALLBACK

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cutting_edge_local": self.cutting_edge_local.tolist(),
            "lip_local": self.lip_local.tolist(),
            "bottom_plate_polygon_local": self.bottom_plate_polygon_local.tolist(),
            "top_edge_local": self.top_edge_local.tolist(),
            "mouth_polygon_local": self.mouth_polygon_local.tolist(),
            "interior_profile_local": self.interior_profile_local.tolist(),
            "back_wall_polygon_local": self.back_wall_polygon_local.tolist(),
            "left_side_wall_local": self.left_side_wall_local.tolist(),
            "right_side_wall_local": self.right_side_wall_local.tolist(),
            "interior_vertices_local": self.interior_vertices_local.tolist(),
            "interior_faces": self.interior_faces.tolist(),
            "geometric_capacity_m3": self.geometric_capacity_m3,
            "rated_capacity_m3": self.rated_capacity_m3,
            "interior_width_m": self.interior_width_m,
            "geometry_source": self.geometry_source.value,
            "geometry_quality": self.geometry_quality.value,
            "capacity_method": self.capacity_method.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BucketGeometryDescriptor":
        values = dict(mapping)
        values.setdefault("interior_width_m", None)
        if "back_wall_polygon_local" not in values:
            left = np.asarray(values["left_side_wall_local"], dtype=np.float64)
            right = np.asarray(values["right_side_wall_local"], dtype=np.float64)
            values["back_wall_polygon_local"] = np.stack(
                (left[0], right[0], right[1], left[1])
            )
        return cls(**values)

    @staticmethod
    def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
        matrix = np.asarray(transform, dtype=np.float64)
        values = np.asarray(points, dtype=np.float64)
        homogeneous = np.column_stack((values, np.ones(len(values))))
        transformed = (matrix @ homogeneous.T).T
        return np.asarray(transformed[:, :3] / transformed[:, 3, None], dtype=np.float64)

    @staticmethod
    def transform_direction(transform: np.ndarray, direction: np.ndarray) -> np.ndarray:
        vector = np.asarray(transform, dtype=np.float64)[:3, :3] @ np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= _EPS:
            raise ValueError("[BucketGeometry] transformed direction is degenerate")
        return vector / norm
