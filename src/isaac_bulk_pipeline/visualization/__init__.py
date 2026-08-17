"""Derived visual representations of authoritative terrain state."""

from .dynamic_mesh_adapter import (
    DynamicMeshAdapter,
    MeshUpdateMetrics,
    build_mesh_arrays,
    compute_vertex_normals,
)
from .lighting_manager import LightingManager, LightingPreset, get_lighting_preset
from .chunked_mesh_adapter import ChunkedDynamicMeshAdapter, ChunkedMeshUpdateMetrics
from .terrain_material_adapter import (
    TerrainMaterialAdapter,
    TerrainMaterialPreset,
    get_terrain_material_preset,
)

__all__ = [
    "DynamicMeshAdapter",
    "ChunkedDynamicMeshAdapter",
    "ChunkedMeshUpdateMetrics",
    "MeshUpdateMetrics",
    "LightingManager",
    "LightingPreset",
    "TerrainMaterialAdapter",
    "TerrainMaterialPreset",
    "build_mesh_arrays",
    "compute_vertex_normals",
    "get_lighting_preset",
    "get_terrain_material_preset",
]
