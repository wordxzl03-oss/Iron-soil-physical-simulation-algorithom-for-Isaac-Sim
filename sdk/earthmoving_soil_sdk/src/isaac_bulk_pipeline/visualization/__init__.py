"""Production visual publication derived from authoritative H_free."""

from .dynamic_mesh_adapter import (
    DynamicMeshAdapter, MeshUpdateMetrics, build_mesh_arrays, compute_vertex_normals,
)
from .chunked_mesh_adapter import ChunkedDynamicMeshAdapter, ChunkedMeshUpdateMetrics

__all__ = [
    "DynamicMeshAdapter", "MeshUpdateMetrics", "ChunkedDynamicMeshAdapter",
    "ChunkedMeshUpdateMetrics", "build_mesh_arrays", "compute_vertex_normals",
]
