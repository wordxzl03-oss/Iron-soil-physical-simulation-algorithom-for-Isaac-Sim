"""Isaac/PhysX publisher for dirty, fixed-topology terrain contact chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..terrain import TerrainGrid
from .chunked_contact import ChunkedContactMeshBackend, ChunkedContactUpdate


@dataclass(frozen=True)
class ContactSynchronizationStatus:
    terrain_timestamp_s: float
    contact_timestamp_s: float
    contact_lag_s: float
    contact_revision: int
    dirty_chunk_count: int
    updated_vertex_count: int
    elapsed_ms: float


class IsaacChunkedContactMeshAdapter:
    """Author hidden static triangle meshes and recook only dirty chunks.

    The authoritative state remains ``TerrainState.H_resting_m``.  This class
    publishes a delayed PhysX contact view at explicit event/low-frequency
    boundaries and reports that delay rather than pretending it is zero.
    """

    def __init__(self, chunk_size_cells: int = 64) -> None:
        self.backend = ChunkedContactMeshBackend(chunk_size_cells)
        self._stage: Any = None
        self._root_prim_path = ""
        self._meshes: dict[tuple[int, int], Any] = {}
        self._Vt: Any = None
        self._terrain_timestamp_s = 0.0
        self._contact_timestamp_s = 0.0
        self._active_keys: frozenset[tuple[int, int]] = frozenset()

    @property
    def collider_paths(self) -> tuple[str, ...]:
        return tuple(str(mesh.GetPath()) for _, mesh in sorted(self._meshes.items()))

    def initialize(
        self,
        stage: Any,
        grid: TerrainGrid,
        heightmap: np.ndarray,
        *,
        root_prim_path: str = "/World/Terrain/ContactChunks",
        timestamp_s: float = 0.0,
    ) -> ContactSynchronizationStatus:
        if self._stage is not None:
            raise RuntimeError("[IsaacChunkedContact] already initialized")
        from pxr import Gf, UsdGeom, UsdPhysics, Vt

        self.backend.initialize(grid, heightmap)
        self._stage = stage
        self._root_prim_path = root_prim_path
        self._Vt = Vt
        root = UsdGeom.Xform.Define(stage, root_prim_path)
        transform = grid.terrain_to_world_matrix
        if not np.allclose(transform, np.eye(4), atol=1.0e-12):
            root.AddTransformOp().Set(Gf.Matrix4d(*transform.T.ravel().tolist()))
        for snapshot in self.backend.snapshots():
            row, col = snapshot.key
            mesh = UsdGeom.Mesh.Define(stage, f"{root_prim_path}/chunk_{row:02d}_{col:02d}")
            mesh.CreatePointsAttr(self._vec3(snapshot.points_m))
            mesh.CreateFaceVertexCountsAttr(self._ints(snapshot.face_counts))
            mesh.CreateFaceVertexIndicesAttr(self._ints(snapshot.face_indices))
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateDoubleSidedAttr(False)
            UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
                UsdPhysics.Tokens.none
            )
            if mesh.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError("[IsaacChunkedContact] contact chunk must remain static")
            self._meshes[snapshot.key] = mesh
        stamp = float(timestamp_s)
        self._terrain_timestamp_s = stamp
        self._contact_timestamp_s = stamp
        return ContactSynchronizationStatus(stamp, stamp, 0.0, 0, len(self._meshes), sum(s.points_m.shape[0] for s in self.backend.snapshots()), 0.0)

    def set_active_mask(self, footprint_mask: np.ndarray, *, tile_halo: int = 1) -> int:
        """Enable PhysX collision only near the current real track support."""

        if self._stage is None:
            raise RuntimeError("[IsaacChunkedContact] initialize must be called first")
        selected = set(self.backend.dirty_keys(changed_mask=footprint_mask))
        if selected and tile_halo > 0:
            expanded = set(selected)
            for row, col in selected:
                for dr in range(-tile_halo, tile_halo + 1):
                    for dc in range(-tile_halo, tile_halo + 1):
                        key = (row + dr, col + dc)
                        if key in self._meshes:
                            expanded.add(key)
            selected = expanded
        frozen = frozenset(selected)
        if frozen == self._active_keys:
            return len(frozen)
        from pxr import UsdPhysics

        for key, mesh in self._meshes.items():
            UsdPhysics.CollisionAPI(mesh.GetPrim()).GetCollisionEnabledAttr().Set(
                key in frozen
            )
        self._active_keys = frozen
        return len(frozen)

    def update(
        self,
        heightmap: np.ndarray,
        *,
        terrain_timestamp_s: float,
        changed_mask: np.ndarray | None = None,
        affected_bbox: tuple[int, int, int, int] | None = None,
    ) -> ContactSynchronizationStatus:
        if self._stage is None:
            raise RuntimeError("[IsaacChunkedContact] initialize must be called first")
        update: ChunkedContactUpdate = self.backend.update(
            heightmap, changed_mask=changed_mask, affected_bbox=affected_bbox
        )
        for key in update.updated_keys:
            snapshot = self.backend.snapshot(key)
            # Fixed topology: only vertex positions change. PhysX recooks these
            # static triangle meshes when USD points are dirtied.
            self._meshes[key].GetPointsAttr().Set(self._vec3(snapshot.points_m))
        stamp = float(terrain_timestamp_s)
        self._terrain_timestamp_s = stamp
        if update.updated_keys:
            self._contact_timestamp_s = stamp
        return ContactSynchronizationStatus(
            terrain_timestamp_s=stamp,
            contact_timestamp_s=self._contact_timestamp_s,
            contact_lag_s=max(0.0, stamp - self._contact_timestamp_s),
            contact_revision=update.revision,
            dirty_chunk_count=len(update.updated_keys),
            updated_vertex_count=update.updated_vertex_count,
            elapsed_ms=update.elapsed_ms,
        )

    def update_tile_samples(
        self,
        samples_by_tile_id: dict[int, np.ndarray],
        *,
        tile_size_cells: int,
        terrain_timestamp_s: float,
    ) -> ContactSynchronizationStatus:
        """Publish compact device-originated tiles to matching PhysX chunks."""

        if self._stage is None:
            raise RuntimeError("[IsaacChunkedContact] initialize must be called first")
        if int(tile_size_cells) != self.backend.chunk_size_cells:
            raise ValueError("[IsaacChunkedContact] device/contact tile size mismatch")
        tiles_x = (
            self.backend.grid.shape[1] - 1 + tile_size_cells - 1
        ) // tile_size_cells
        by_key = {
            divmod(int(tile_id), tiles_x): np.asarray(sample, dtype=np.float64)
            for tile_id, sample in samples_by_tile_id.items()
        }
        update = self.backend.update_tile_samples(by_key)
        for key in update.updated_keys:
            snapshot = self.backend.snapshot(key)
            self._meshes[key].GetPointsAttr().Set(self._vec3(snapshot.points_m))
        stamp = float(terrain_timestamp_s)
        self._terrain_timestamp_s = stamp
        if update.updated_keys:
            self._contact_timestamp_s = stamp
        return ContactSynchronizationStatus(
            terrain_timestamp_s=stamp,
            contact_timestamp_s=self._contact_timestamp_s,
            contact_lag_s=max(0.0, stamp - self._contact_timestamp_s),
            contact_revision=update.revision,
            dirty_chunk_count=len(update.updated_keys),
            updated_vertex_count=update.updated_vertex_count,
            elapsed_ms=update.elapsed_ms,
        )

    def lag_status(self, terrain_timestamp_s: float) -> ContactSynchronizationStatus:
        stamp = float(terrain_timestamp_s)
        self._terrain_timestamp_s = stamp
        return ContactSynchronizationStatus(
            stamp,
            self._contact_timestamp_s,
            max(0.0, stamp - self._contact_timestamp_s),
            self.backend.revision,
            0,
            0,
            0.0,
        )

    def _vec3(self, values: np.ndarray) -> Any:
        return self._Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(values, dtype=np.float32))

    def _ints(self, values: np.ndarray) -> Any:
        return self._Vt.IntArray.FromNumpy(np.ascontiguousarray(values, dtype=np.int32))
