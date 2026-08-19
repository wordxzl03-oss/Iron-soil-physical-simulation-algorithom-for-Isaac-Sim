"""Open physical-contact geometry for excavator buckets.

Containment and physical contact are intentionally different contracts:

* containment geometry is watertight so capacity and point-inside tests are
  well-defined;
* physical-contact geometry must leave the bucket mouth open so no solver can
  generate force on a fictitious mouth cap.

This module converts the authoritative :class:`BucketGeometryDescriptor`
containment mesh into an immutable open contact mesh and validates that the
only boundary introduced by removing the topological cap is the four-edge
bucket mouth.  Future Tool--Mobile, local-3D, DEM or MPM solvers should consume
this contract rather than reconstructing their own mouth-cap logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .bucket_geometry import BucketGeometryDescriptor


_EPS = 1.0e-12
_MOUTH_PLANE_TOL_M = 1.0e-9


def _readonly(value: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value, dtype=dtype).copy())
    result.setflags(write=False)
    return result


def physical_bucket_contact_face_mask(
    geometry: BucketGeometryDescriptor,
    *,
    mouth_plane_tolerance_m: float = _MOUTH_PLANE_TOL_M,
) -> np.ndarray:
    """Return the containment-face mask that excludes only the mouth cap.

    ``BucketGeometryDescriptor`` intentionally stores a watertight cavity.  For
    the current extruded-profile contract that watertightness is achieved with
    exactly two triangles lying in the mouth plane.  They are topology for
    capacity/containment only and are never physical steel.
    """

    tolerance = float(mouth_plane_tolerance_m)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(
            "[BucketPhysicalContact] mouth_plane_tolerance_m must be finite/positive"
        )

    vertices = np.asarray(geometry.interior_vertices_local, dtype=np.float64)
    faces = np.asarray(geometry.interior_faces, dtype=np.int64)
    mouth_centroid = np.asarray(geometry.mouth_centroid_local, dtype=np.float64)
    mouth_normal = np.asarray(geometry.mouth_normal_local, dtype=np.float64)
    mouth_norm = float(np.linalg.norm(mouth_normal))
    if mouth_norm <= _EPS:
        raise ValueError("[BucketPhysicalContact] mouth normal is degenerate")
    mouth_normal = mouth_normal / mouth_norm

    vertex_on_mouth = (
        np.abs((vertices - mouth_centroid) @ mouth_normal) <= tolerance
    )
    mouth_cap = np.all(vertex_on_mouth[faces], axis=1)
    cap_count = int(np.count_nonzero(mouth_cap))
    if cap_count != 2:
        raise ValueError(
            "[BucketPhysicalContact] closed containment mesh must contain exactly "
            f"two mouth-cap triangles; found={cap_count}"
        )
    return _readonly(~mouth_cap, dtype=bool)


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (first, second) if first < second else (second, first)
            counts[edge] = counts.get(edge, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    return np.asarray(boundary, dtype=np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class BucketPhysicalContactGeometry:
    """Immutable open bucket surface used by material-contact solvers.

    The face array references the same local vertices as the authoritative
    containment geometry but excludes the two topological mouth-cap triangles.
    A valid contract has one and only one open boundary: the four perimeter
    edges of the bucket mouth.
    """

    vertices_local: np.ndarray
    faces: np.ndarray
    containment_face_mask: np.ndarray
    removed_mouth_cap_face_indices: np.ndarray
    mouth_boundary_edges: np.ndarray
    mouth_centroid_local: np.ndarray
    mouth_normal_local: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_local, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        mask = np.asarray(self.containment_face_mask, dtype=bool).reshape(-1)
        removed = np.asarray(
            self.removed_mouth_cap_face_indices, dtype=np.int64
        ).reshape(-1)
        edges = np.asarray(self.mouth_boundary_edges, dtype=np.int64).reshape(-1, 2)
        centroid = np.asarray(self.mouth_centroid_local, dtype=np.float64)
        normal = np.asarray(self.mouth_normal_local, dtype=np.float64)

        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
            raise ValueError(
                "[BucketPhysicalContact] vertices_local must have shape (N,3)"
            )
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 1:
            raise ValueError("[BucketPhysicalContact] faces must have shape (M,3)")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("[BucketPhysicalContact] face index out of range")
        if centroid.shape != (3,) or normal.shape != (3,):
            raise ValueError(
                "[BucketPhysicalContact] mouth centroid/normal must have shape (3,)"
            )
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(centroid)):
            raise ValueError("[BucketPhysicalContact] geometry contains NaN/Inf")
        normal_norm = float(np.linalg.norm(normal))
        if not np.isfinite(normal_norm) or normal_norm <= _EPS:
            raise ValueError("[BucketPhysicalContact] mouth normal is degenerate")
        normal = normal / normal_norm

        if len(removed) != 2:
            raise ValueError(
                "[BucketPhysicalContact] exactly two containment mouth-cap faces "
                "must be removed"
            )
        if len(edges) != 4:
            raise ValueError(
                "[BucketPhysicalContact] open physical mesh must expose exactly "
                f"four mouth-boundary edges; found={len(edges)}"
            )
        if np.any(edges < 0) or np.any(edges >= len(vertices)):
            raise ValueError("[BucketPhysicalContact] boundary edge index out of range")

        distances = np.abs((vertices - centroid) @ normal)
        if not np.all(distances[np.unique(edges)] <= _MOUTH_PLANE_TOL_M):
            raise ValueError(
                "[BucketPhysicalContact] every open boundary edge must lie in the "
                "authoritative mouth plane"
            )
        face_on_mouth = np.all(
            distances[faces] <= _MOUTH_PLANE_TOL_M,
            axis=1,
        )
        if np.any(face_on_mouth):
            raise ValueError(
                "[BucketPhysicalContact] physical contact mesh contains a mouth "
                "closure face"
            )

        object.__setattr__(self, "vertices_local", _readonly(vertices, dtype=np.float64))
        object.__setattr__(self, "faces", _readonly(faces, dtype=np.int64))
        object.__setattr__(self, "containment_face_mask", _readonly(mask, dtype=bool))
        object.__setattr__(
            self,
            "removed_mouth_cap_face_indices",
            _readonly(removed, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "mouth_boundary_edges",
            _readonly(edges, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "mouth_centroid_local",
            _readonly(centroid, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "mouth_normal_local",
            _readonly(normal, dtype=np.float64),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_bucket_geometry(
        cls,
        geometry: BucketGeometryDescriptor,
    ) -> "BucketPhysicalContactGeometry":
        """Build and validate the unique open-contact view of one bucket."""

        vertices = np.asarray(geometry.interior_vertices_local, dtype=np.float64)
        containment_faces = np.asarray(geometry.interior_faces, dtype=np.int64)
        mask = physical_bucket_contact_face_mask(geometry)
        physical_faces = containment_faces[mask]
        removed = np.flatnonzero(~mask)
        boundary = _boundary_edges(physical_faces)

        return cls(
            vertices_local=vertices,
            faces=physical_faces,
            containment_face_mask=mask,
            removed_mouth_cap_face_indices=removed,
            mouth_boundary_edges=boundary,
            mouth_centroid_local=np.asarray(
                geometry.mouth_centroid_local, dtype=np.float64
            ),
            mouth_normal_local=np.asarray(geometry.mouth_normal_local, dtype=np.float64),
            metadata={
                "geometry_contract": "OPEN_PHYSICAL_CONTACT_V1",
                "source": "CLOSED_CONTAINMENT_MINUS_MOUTH_CAP",
                "containment_face_count": int(len(containment_faces)),
                "physical_face_count": int(len(physical_faces)),
                "removed_topological_mouth_cap_face_count": int(len(removed)),
                "mouth_physical_closure_face_count": 0,
                "physical_contact_mouth_open": True,
                "single_convex_bucket_enclosure_allowed": False,
            },
        )

    @property
    def mouth_physical_closure_face_count(self) -> int:
        return 0

    @property
    def physical_contact_mouth_open(self) -> bool:
        return True

    def transform_vertices(self, transform: np.ndarray) -> np.ndarray:
        """Return contact vertices transformed by a 4x4 column-vector pose."""

        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("[BucketPhysicalContact] transform must be finite 4x4")
        homogeneous = np.column_stack(
            (self.vertices_local, np.ones(len(self.vertices_local), dtype=np.float64))
        )
        transformed = (matrix @ homogeneous.T).T
        if np.any(np.abs(transformed[:, 3]) <= _EPS):
            raise ValueError("[BucketPhysicalContact] transform produced zero w")
        return np.ascontiguousarray(
            transformed[:, :3] / transformed[:, 3, None], dtype=np.float64
        )

    def assert_backend_representation(
        self,
        representation: str,
        *,
        piece_count: int | None = None,
    ) -> None:
        """Fail fast on collision approximations that necessarily seal the mouth.

        A single convex hull, box or bounding-volume approximation of the whole
        bucket cannot preserve a concave open cavity.  Multi-piece convex or
        solver-native triangle representations are not certified here; they
        still require the backend-specific mouth-aperture regression.
        """

        normalized = "".join(ch for ch in str(representation).lower() if ch.isalnum())
        whole_bucket_convex = normalized in {
            "convexhull",
            "convexhullsimplification",
            "boundingbox",
            "boundingcube",
            "box",
        }
        if whole_bucket_convex and (piece_count is None or int(piece_count) <= 1):
            raise ValueError(
                "[BucketPhysicalContact] backend representation seals the bucket "
                f"mouth: representation={representation!r}, piece_count={piece_count}"
            )


def assert_open_bucket_physical_contact(
    geometry: BucketGeometryDescriptor,
) -> BucketPhysicalContactGeometry:
    """Construct the contract and fail if any physical mouth closure exists."""

    return BucketPhysicalContactGeometry.from_bucket_geometry(geometry)
