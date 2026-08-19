from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaac_bulk_pipeline.tools import (
    BucketPhysicalContactGeometry,
    ToolDescriptorLoader,
    assert_open_bucket_physical_contact,
    physical_bucket_contact_face_mask,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_390F_TOOL_CONFIG = ROOT / "configs" / "excavator_390f_real_bucket.yaml"


def _real_390f_geometry():
    config = ToolDescriptorLoader.load_config(REAL_390F_TOOL_CONFIG)
    descriptor = ToolDescriptorLoader.load(config)
    assert descriptor.bucket_geometry is not None
    return descriptor.bucket_geometry


def _boundary_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for face in np.asarray(faces, dtype=np.int64):
        for first, second in (
            (int(face[0]), int(face[1])),
            (int(face[1]), int(face[2])),
            (int(face[2]), int(face[0])),
        ):
            edge = (first, second) if first < second else (second, first)
            counts[edge] = counts.get(edge, 0) + 1
    return {edge for edge, count in counts.items() if count == 1}


def test_real_390f_containment_is_closed_but_physical_contact_is_open() -> None:
    geometry = _real_390f_geometry()
    contact = assert_open_bucket_physical_contact(geometry)

    # Capacity/inside-outside geometry remains watertight.
    assert _boundary_edges(geometry.interior_faces) == set()

    # Physical contact removes exactly the two topological mouth-cap triangles.
    assert contact.removed_mouth_cap_face_indices.shape == (2,)
    assert int(np.count_nonzero(~contact.containment_face_mask)) == 2
    assert len(contact.faces) == len(geometry.interior_faces) - 2

    # The only physical-mesh boundary is the four-sided open bucket mouth.
    physical_boundary = _boundary_edges(contact.faces)
    assert len(physical_boundary) == 4
    assert physical_boundary == {
        tuple(map(int, edge)) for edge in contact.mouth_boundary_edges
    }
    assert contact.physical_contact_mouth_open is True
    assert contact.mouth_physical_closure_face_count == 0
    assert contact.metadata["geometry_contract"] == "OPEN_PHYSICAL_CONTACT_V1"
    assert contact.metadata["single_convex_bucket_enclosure_allowed"] is False


def test_real_390f_physical_faces_do_not_lie_in_mouth_plane() -> None:
    geometry = _real_390f_geometry()
    contact = BucketPhysicalContactGeometry.from_bucket_geometry(geometry)

    distances = np.abs(
        (contact.vertices_local - contact.mouth_centroid_local)
        @ contact.mouth_normal_local
    )
    physical_face_on_mouth = np.all(distances[contact.faces] <= 1.0e-9, axis=1)
    assert not np.any(physical_face_on_mouth)

    boundary_vertices = np.unique(contact.mouth_boundary_edges)
    assert np.all(distances[boundary_vertices] <= 1.0e-9)


def test_face_mask_keeps_containment_and_contact_semantics_separate() -> None:
    geometry = _real_390f_geometry()
    mask = physical_bucket_contact_face_mask(geometry)

    assert mask.dtype == np.bool_
    assert mask.shape == (len(geometry.interior_faces),)
    assert int(np.count_nonzero(mask)) == len(geometry.interior_faces) - 2
    assert int(np.count_nonzero(~mask)) == 2


def test_single_convex_hull_backend_is_rejected_fail_fast() -> None:
    contact = assert_open_bucket_physical_contact(_real_390f_geometry())

    with pytest.raises(ValueError, match="seals the bucket mouth"):
        contact.assert_backend_representation("convexHull")

    with pytest.raises(ValueError, match="seals the bucket mouth"):
        contact.assert_backend_representation("boundingBox", piece_count=1)

    # A solver-native triangle representation preserves the authored opening.
    contact.assert_backend_representation("openTriangleSurface")


def test_reintroducing_one_mouth_cap_face_is_rejected() -> None:
    geometry = _real_390f_geometry()
    contact = assert_open_bucket_physical_contact(geometry)
    containment_faces = np.asarray(geometry.interior_faces, dtype=np.int64)
    one_cap = containment_faces[contact.removed_mouth_cap_face_indices[:1]]
    illegally_closed_faces = np.vstack((contact.faces, one_cap))

    with pytest.raises(ValueError, match="mouth closure face"):
        BucketPhysicalContactGeometry(
            vertices_local=contact.vertices_local,
            faces=illegally_closed_faces,
            containment_face_mask=contact.containment_face_mask,
            removed_mouth_cap_face_indices=contact.removed_mouth_cap_face_indices,
            mouth_boundary_edges=contact.mouth_boundary_edges,
            mouth_centroid_local=contact.mouth_centroid_local,
            mouth_normal_local=contact.mouth_normal_local,
            metadata=contact.metadata,
        )


def test_identity_transform_preserves_real_390f_contact_vertices() -> None:
    contact = assert_open_bucket_physical_contact(_real_390f_geometry())
    transformed = contact.transform_vertices(np.eye(4, dtype=np.float64))
    np.testing.assert_allclose(transformed, contact.vertices_local, rtol=0.0, atol=0.0)


def test_cpu_tool_mobile_has_no_solver_local_mouth_cap_classifier() -> None:
    source = (
        ROOT
        / "src"
        / "isaac_bulk_pipeline"
        / "bulk_interaction"
        / "tool_mobile_contact.py"
    ).read_text(encoding="utf-8")

    assert "assert_open_bucket_physical_contact(geometry)" in source
    assert "def physical_bucket_contact_face_mask(" not in source
    assert "contact_geometry.containment_face_mask" in source


def test_warp_tool_mobile_has_no_solver_local_mouth_cap_classifier() -> None:
    source = (
        ROOT
        / "src"
        / "isaac_bulk_pipeline"
        / "bulk_interaction"
        / "warp_tool_mobile_contact.py"
    ).read_text(encoding="utf-8")

    assert "from ..tools import assert_open_bucket_physical_contact" in source
    assert "assert_open_bucket_physical_contact(geometry)" in source
    assert "contact_geometry.containment_face_mask" in source
    assert "from .tool_mobile_contact import physical_bucket_contact_face_mask" not in source
