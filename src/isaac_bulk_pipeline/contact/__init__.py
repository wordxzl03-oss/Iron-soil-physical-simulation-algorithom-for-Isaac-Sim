"""Separate terrain ContactView for Isaac/PhysX vehicle support."""

from .backend import ContactBackendConfig, ContactCommitResult, TerrainContactBackend
from .collision_filter import (
    ALL_COLLISION_GROUPS,
    CollisionDecision,
    CollisionGroup,
    CollisionMatrix,
    CollisionValidation,
    validate_collision_matrix,
    validate_collision_memberships,
)
from .config import PhaseCContactConfig, WheelTerrainConfig, load_phase_c_contact_config
from .heightfield import (
    HeightFieldCapabilityStatus,
    PhysXHeightFieldContactBackend,
    assess_heightfield_capability,
)
from .slope import SlopePatchSpec, build_planar_slope_heightmap, estimate_slope_deg
from .triangle_mesh import ContactMeshData, TriangleMeshContactBackend, build_contact_mesh
from .chunked_contact import (
    ChunkedContactMeshBackend,
    ChunkedContactUpdate,
    ContactChunkSnapshot,
)
from .isaac_chunked_contact import (
    ContactSynchronizationStatus,
    IsaacChunkedContactMeshAdapter,
)

__all__ = [
    "ALL_COLLISION_GROUPS",
    "CollisionDecision",
    "CollisionGroup",
    "CollisionMatrix",
    "CollisionValidation",
    "ContactBackendConfig",
    "ContactCommitResult",
    "ContactMeshData",
    "ChunkedContactMeshBackend",
    "ChunkedContactUpdate",
    "ContactChunkSnapshot",
    "ContactSynchronizationStatus",
    "HeightFieldCapabilityStatus",
    "IsaacChunkedContactMeshAdapter",
    "PhaseCContactConfig",
    "PhysXHeightFieldContactBackend",
    "SlopePatchSpec",
    "TerrainContactBackend",
    "TriangleMeshContactBackend",
    "WheelTerrainConfig",
    "assess_heightfield_capability",
    "build_contact_mesh",
    "build_planar_slope_heightmap",
    "estimate_slope_deg",
    "load_phase_c_contact_config",
    "validate_collision_matrix",
    "validate_collision_memberships",
]
