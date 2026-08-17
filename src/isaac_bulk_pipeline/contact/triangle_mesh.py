"""Pure-NumPy data backend for a hidden static triangle-mesh collider."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np

from ..terrain.terrain_grid import TerrainGrid
from .backend import (
    CommitReason,
    ContactBackendConfig,
    ContactCommitResult,
    TerrainContactBackend,
)


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ContactMeshData:
    """Engine-neutral mesh buffers in terrain-local metre coordinates."""

    points_m: np.ndarray
    face_vertex_counts: np.ndarray
    face_vertex_indices: np.ndarray
    sampled_rows: np.ndarray
    sampled_columns: np.ndarray
    source_shape_yx: tuple[int, int]
    nominal_spacing_xy_m: tuple[float, float]
    terrain_to_world_matrix: np.ndarray

    @property
    def vertex_count(self) -> int:
        return int(self.points_m.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.face_vertex_counts.size)

    @property
    def sampled_shape_yx(self) -> tuple[int, int]:
        return int(self.sampled_rows.size), int(self.sampled_columns.size)


def sample_indices(count: int, source_spacing_m: float, target_spacing_m: float) -> np.ndarray:
    """Return deterministic indices while retaining both source boundaries."""

    if count < 2:
        raise ValueError("[TriangleMeshContact] count must be >= 2")
    if source_spacing_m <= 0.0 or target_spacing_m <= 0.0:
        raise ValueError("[TriangleMeshContact] spacings must be positive")
    stride = max(1, int(ceil(target_spacing_m / source_spacing_m - 1e-12)))
    indices = np.arange(0, count, stride, dtype=np.int32)
    if int(indices[-1]) != count - 1:
        indices = np.concatenate((indices, np.asarray([count - 1], dtype=np.int32)))
    return indices


def build_contact_mesh(
    heightmap: np.ndarray,
    grid: TerrainGrid,
    *,
    target_spacing_m: float,
) -> ContactMeshData:
    """Downsample ``H[y,x]`` and build a +Z-wound support surface.

    The contact mesh is a separate allocation and topology from the render
    mesh.  For a non-rectangular valid mask, a quad is emitted only when all
    four sampled corner vertices are valid; holes therefore remain non-solid.
    """

    height = grid.validate_heightmap(heightmap)
    rows = sample_indices(grid.ny, grid.dy, target_spacing_m)
    columns = sample_indices(grid.nx, grid.dx, target_spacing_m)
    sampled = height[np.ix_(rows, columns)]
    x = grid.origin_x + columns.astype(np.float64) * grid.dx
    y = grid.origin_y + rows.astype(np.float64) * grid.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), sampled.ravel()))

    sampled_ny, sampled_nx = sampled.shape
    sampled_valid = (
        np.ones(sampled.shape, dtype=bool)
        if grid.valid_mask is None
        else grid.valid_mask[np.ix_(rows, columns)]
    )
    faces: list[tuple[int, int, int]] = []
    for row in range(sampled_ny - 1):
        for column in range(sampled_nx - 1):
            if not bool(np.all(sampled_valid[row : row + 2, column : column + 2])):
                continue
            a = row * sampled_nx + column
            b = a + 1
            d = a + sampled_nx
            c = d + 1
            faces.extend(((a, b, c), (a, c, d)))
    face_indices = np.asarray(faces, dtype=np.int32).reshape(-1)
    face_counts = np.full(len(faces), 3, dtype=np.int32)

    # The requested spacing is nominal.  The final interval may be shorter so
    # the true source boundary is always represented.
    spacing_x = float(grid.dx * max(1, int(columns[1] - columns[0])))
    spacing_y = float(grid.dy * max(1, int(rows[1] - rows[0])))
    return ContactMeshData(
        points_m=_readonly(points, np.float32),
        face_vertex_counts=_readonly(face_counts, np.int32),
        face_vertex_indices=_readonly(face_indices, np.int32),
        sampled_rows=_readonly(rows, np.int32),
        sampled_columns=_readonly(columns, np.int32),
        source_shape_yx=grid.shape,
        nominal_spacing_xy_m=(spacing_x, spacing_y),
        terrain_to_world_matrix=_readonly(grid.terrain_to_world_matrix.copy(), np.float64),
    )


class TriangleMeshContactBackend(TerrainContactBackend):
    """Stage contact mesh updates and expose explicit recook boundaries.

    ``update_from_heightmap`` never changes ``committed_mesh``.  An Isaac
    adapter should update USD/PhysX only after :meth:`commit` reports
    ``recook_required=True``.
    """

    def __init__(self, config: ContactBackendConfig | None = None) -> None:
        self.config = config or ContactBackendConfig()
        self._grid: TerrainGrid | None = None
        self._initial_mesh: ContactMeshData | None = None
        self._committed_mesh: ContactMeshData | None = None
        self._pending_mesh: ContactMeshData | None = None
        self._generation = -1
        self._last_commit_time_s: float | None = None
        self._pending_action_index: int | None = None

    @property
    def initialized(self) -> bool:
        return self._committed_mesh is not None

    @property
    def committed_mesh(self) -> ContactMeshData:
        if self._committed_mesh is None:
            raise RuntimeError("[TriangleMeshContact] backend is not initialized")
        return self._committed_mesh

    @property
    def pending(self) -> bool:
        return self._pending_mesh is not None

    @property
    def generation(self) -> int:
        return self._generation

    def initialize(self, grid: TerrainGrid, heightmap: np.ndarray) -> None:
        if self.initialized:
            raise RuntimeError("[TriangleMeshContact] initialize may only be called once")
        mesh = build_contact_mesh(
            heightmap,
            grid,
            target_spacing_m=self.config.target_spacing_m,
        )
        self._grid = grid
        self._initial_mesh = mesh
        self._committed_mesh = mesh
        self._generation = 0

    def update_from_heightmap(
        self,
        heightmap: np.ndarray,
        *,
        action_index: int | None = None,
        sim_time_s: float | None = None,
    ) -> None:
        del sim_time_s  # Staging is deliberately independent of engine time.
        if self._grid is None:
            raise RuntimeError("[TriangleMeshContact] initialize before update")
        if action_index is not None and action_index < 0:
            raise ValueError("[TriangleMeshContact] action_index must be non-negative")
        self._pending_mesh = build_contact_mesh(
            heightmap,
            self._grid,
            target_spacing_m=self.config.target_spacing_m,
        )
        self._pending_action_index = action_index

    def commit(
        self,
        *,
        reason: CommitReason = "action_end",
        sim_time_s: float | None = None,
    ) -> ContactCommitResult:
        current = self.committed_mesh
        if reason not in {"action_end", "low_frequency"}:
            raise ValueError("[TriangleMeshContact] commit reason must be action_end or low_frequency")
        if reason == "low_frequency":
            if self.config.commit_policy != "low_frequency":
                raise ValueError(
                    "[TriangleMeshContact] low_frequency commit is disabled by policy"
                )
            if sim_time_s is None or not np.isfinite(sim_time_s) or sim_time_s < 0.0:
                raise ValueError(
                    "[TriangleMeshContact] low_frequency commit requires finite sim_time_s"
                )
            interval = 1.0 / self.config.low_frequency_hz
            if (
                self._last_commit_time_s is not None
                and sim_time_s - self._last_commit_time_s < interval - 1e-12
            ):
                return ContactCommitResult(
                    committed=False,
                    generation=self._generation,
                    reason=reason,
                    vertex_count=current.vertex_count,
                    triangle_count=current.triangle_count,
                    recook_required=False,
                    rate_limited=True,
                )
        if self._pending_mesh is None:
            return ContactCommitResult(
                committed=False,
                generation=self._generation,
                reason=reason,
                vertex_count=current.vertex_count,
                triangle_count=current.triangle_count,
                recook_required=False,
            )
        self._committed_mesh = self._pending_mesh
        self._pending_mesh = None
        self._pending_action_index = None
        self._generation += 1
        if sim_time_s is not None:
            self._last_commit_time_s = float(sim_time_s)
        committed = self.committed_mesh
        return ContactCommitResult(
            committed=True,
            generation=self._generation,
            reason=reason,
            vertex_count=committed.vertex_count,
            triangle_count=committed.triangle_count,
            recook_required=True,
        )

    def reset(self) -> ContactCommitResult:
        if self._initial_mesh is None:
            raise RuntimeError("[TriangleMeshContact] initialize before reset")
        changed = self._committed_mesh is not self._initial_mesh or self._pending_mesh is not None
        self._committed_mesh = self._initial_mesh
        self._pending_mesh = None
        self._pending_action_index = None
        self._last_commit_time_s = None
        if changed:
            self._generation += 1
        mesh = self.committed_mesh
        return ContactCommitResult(
            committed=changed,
            generation=self._generation,
            reason="reset",
            vertex_count=mesh.vertex_count,
            triangle_count=mesh.triangle_count,
            recook_required=changed,
        )
