"""Fixed-topology USD mesh derived from an authoritative ``H[y, x]`` state.

The module imports USD bindings lazily so its NumPy geometry helpers and unit
tests remain usable outside Isaac Sim. No collision API is authored here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from ..config.loader import MeshConfig
from ..terrain.heightmap_io import validate_heightmap
from ..terrain.terrain_grid import TerrainGrid


@dataclass(frozen=True)
class MeshUpdateMetrics:
    """Diagnostics for one USD points update."""

    elapsed_ms: float
    vertex_count: int
    normals_updated: bool
    update_index: int
    affected_bbox_grid: tuple[int, int, int, int] | None


def build_mesh_arrays(
    heightmap: np.ndarray, grid: TerrainGrid
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build local XYZ points and triangle topology for ``H[y, x]``.

    Returns ``(points, face_counts, face_indices)`` with dtypes ``float32``,
    ``int32`` and ``int32``. Vertices are row-major: index ``row * nx + col``.
    Triangle winding points toward terrain-local +Z.
    """

    height = validate_heightmap(heightmap, expected_shape=grid.shape, copy=False)
    x = grid.origin_x + np.arange(grid.nx, dtype=np.float64) * grid.dx
    y = grid.origin_y + np.arange(grid.ny, dtype=np.float64) * grid.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel(), height.ravel())).astype(
        np.float32,
        copy=False,
    )

    rows, columns = np.meshgrid(
        np.arange(grid.ny - 1, dtype=np.int32),
        np.arange(grid.nx - 1, dtype=np.int32),
        indexing="ij",
    )
    a = rows * grid.nx + columns
    b = a + 1
    d = a + grid.nx
    c = d + 1
    faces = np.stack((a, b, c, a, c, d), axis=-1).reshape(-1, 3)
    face_indices = np.ascontiguousarray(faces.ravel(), dtype=np.int32)
    face_counts = np.full(faces.shape[0], 3, dtype=np.int32)
    return np.ascontiguousarray(points), face_counts, face_indices


def compute_vertex_normals(heightmap: np.ndarray, grid: TerrainGrid) -> np.ndarray:
    """Compute vectorized terrain-local unit normals for ``H[y, x]``."""

    height = validate_heightmap(heightmap, expected_shape=grid.shape, copy=False)
    derivative_y, derivative_x = np.gradient(height, grid.dy, grid.dx)
    normals = np.stack(
        (-derivative_x, -derivative_y, np.ones_like(height)), axis=-1
    )
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return np.ascontiguousarray(normals.reshape(-1, 3), dtype=np.float32)


class DynamicMeshAdapter:
    """Initialize one USD Mesh and update only its point/normal attributes.

    The adapter owns no authoritative terrain state. Callers must retain and
    update ``H_current`` separately. ``initialize`` must be called exactly once.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._mesh: Any = None
        self._points_attr: Any = None
        self._normals_attr: Any = None
        self._grid: TerrainGrid | None = None
        self._config: MeshConfig | None = None
        self._points: np.ndarray | None = None
        self._update_index = 0
        self._normal_update_interval = 1
        self._Vt: Any = None

    @property
    def initialized(self) -> bool:
        """Whether the USD Prim and fixed topology have been authored."""

        return self._initialized

    @property
    def prim_path(self) -> str | None:
        """Configured USD terrain Prim path, if initialized."""

        return None if self._grid is None else self._grid.terrain_prim_path

    def initialize(
        self,
        stage: Any,
        grid: TerrainGrid,
        config: MeshConfig,
        heightmap: np.ndarray,
    ) -> None:
        """Author one fixed-topology mesh from the initial authoritative state."""

        if self._initialized:
            raise RuntimeError(
                f"[DynamicMeshAdapter] already initialized; prim_path={self.prim_path}"
            )
        if config.collision_enabled:
            raise ValueError(
                "[DynamicMeshAdapter] dynamic CollisionAPI is intentionally disabled; "
                f"prim_path={grid.terrain_prim_path}"
            )
        if config.subdivision_scheme != "none":
            raise ValueError(
                "[DynamicMeshAdapter] subdivision_scheme must be 'none'; "
                f"value={config.subdivision_scheme!r}"
            )
        height = validate_heightmap(
            heightmap, expected_shape=grid.shape, output_dtype=np.float64, copy=False
        )

        try:
            from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt
        except ImportError as exc:
            raise RuntimeError(
                "[DynamicMeshAdapter] pxr bindings are unavailable. Run this method "
                "with Isaac Sim's python.sh."
            ) from exc

        stage_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not np.isclose(stage_meters_per_unit, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "[DynamicMeshAdapter] Stage length units must be metres; "
                f"metersPerUnit={stage_meters_per_unit}, "
                f"prim_path={grid.terrain_prim_path}"
            )
        stage_up_axis = UsdGeom.GetStageUpAxis(stage)
        if stage_up_axis != UsdGeom.Tokens.z:
            raise ValueError(
                "[DynamicMeshAdapter] Stage up axis must be Z; "
                f"upAxis={stage_up_axis}, prim_path={grid.terrain_prim_path}"
            )

        points, face_counts, face_indices = build_mesh_arrays(height, grid)
        mesh = UsdGeom.Mesh.Define(stage, grid.terrain_prim_path)
        points_attr = mesh.CreatePointsAttr(self._to_vt_vec3(points, Vt))
        mesh.CreateFaceVertexCountsAttr(self._to_vt_int(face_counts, Vt))
        mesh.CreateFaceVertexIndicesAttr(self._to_vt_int(face_indices, Vt))
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(bool(config.double_sided))
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*config.display_color_rgb)])

        # DisplayColor alone inherits a glossy renderer default in some Kit
        # layouts.  Bind a matte PreviewSurface so soil keeps readable shading
        # without plastic-looking white streaks.
        material = UsdShade.Material.Define(
            stage,
            f"{grid.terrain_prim_path}/SoilMaterial",
        )
        shader = UsdShade.Shader.Define(
            stage,
            f"{grid.terrain_prim_path}/SoilMaterial/PreviewSurface",
        )
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput(
            "diffuseColor",
            Sdf.ValueTypeNames.Color3f,
        ).Set(Gf.Vec3f(*config.display_color_rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.92)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(),
            "surface",
        )
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

        # NumPy uses column vectors with translation in the last column. Gf/USD
        # matrices use row-vector storage, so transpose explicitly at this edge.
        transform = grid.terrain_to_world_matrix
        if not np.allclose(transform, np.eye(4), atol=1e-12):
            gf_matrix = Gf.Matrix4d(*transform.T.ravel().tolist())
            UsdGeom.Xformable(mesh.GetPrim()).AddTransformOp().Set(gf_matrix)

        normals_attr = None
        if config.update_normals:
            normals = compute_vertex_normals(height, grid)
            normals_attr = mesh.CreateNormalsAttr(self._to_vt_vec3(normals, Vt))
            mesh.SetNormalsInterpolation("vertex")

        self._mesh = mesh
        self._points_attr = points_attr
        self._normals_attr = normals_attr
        self._grid = grid
        self._config = config
        self._points = points
        self._Vt = Vt
        self._normal_update_interval = max(
            1,
            int(round(config.mesh_update_rate_hz / config.normal_update_rate_hz)),
        )
        self._initialized = True

    def update(
        self,
        heightmap: np.ndarray,
        affected_bbox: tuple[int, int, int, int] | None = None,
    ) -> MeshUpdateMetrics:
        """Update vertex Z values and, at the configured cadence, normals.

        ``affected_bbox`` uses half-open ``(row_min, col_min, row_max, col_max)``.
        USD points arrays are currently replaced as one vectorized Vt array;
        topology and Prim identity remain unchanged.
        """

        if not self._initialized or self._grid is None or self._config is None:
            raise RuntimeError("[DynamicMeshAdapter] initialize must be called before update")
        assert self._points is not None
        assert self._points_attr is not None
        height = validate_heightmap(
            heightmap,
            expected_shape=self._grid.shape,
            output_dtype=np.float64,
            copy=False,
        )
        bbox = self._validate_bbox(affected_bbox)
        start = perf_counter()
        self._points[:, 2] = height.ravel().astype(np.float32, copy=False)
        self._points_attr.Set(self._to_vt_vec3(self._points, self._Vt))
        self._update_index += 1
        normals_updated = False
        if (
            self._config.update_normals
            and self._normals_attr is not None
            and self._update_index % self._normal_update_interval == 0
        ):
            normals = compute_vertex_normals(height, self._grid)
            self._normals_attr.Set(self._to_vt_vec3(normals, self._Vt))
            normals_updated = True
        elapsed_ms = (perf_counter() - start) * 1_000.0
        return MeshUpdateMetrics(
            elapsed_ms=elapsed_ms,
            vertex_count=self._points.shape[0],
            normals_updated=normals_updated,
            update_index=self._update_index,
            affected_bbox_grid=bbox,
        )

    def reset(self, heightmap: np.ndarray) -> MeshUpdateMetrics:
        """Restore visual points from a caller-provided authoritative state."""

        return self.update(heightmap, affected_bbox=None)

    def _validate_bbox(
        self, bbox: tuple[int, int, int, int] | None
    ) -> tuple[int, int, int, int] | None:
        if bbox is None:
            return None
        if self._grid is None or len(bbox) != 4:
            raise ValueError(
                f"[DynamicMeshAdapter] invalid affected_bbox={bbox!r}; prim_path={self.prim_path}"
            )
        row_min, col_min, row_max, col_max = bbox
        if not (
            0 <= row_min < row_max <= self._grid.ny
            and 0 <= col_min < col_max <= self._grid.nx
        ):
            raise ValueError(
                "[DynamicMeshAdapter] affected_bbox must be a nonempty half-open "
                f"grid region inside shape={self._grid.shape}; bbox={bbox}"
            )
        return bbox

    @staticmethod
    def _to_vt_vec3(values: np.ndarray, vt_module: Any) -> Any:
        array = np.ascontiguousarray(values, dtype=np.float32)
        try:
            return vt_module.Vec3fArray.FromNumpy(array)
        except Exception as exc:
            raise RuntimeError(
                "[DynamicMeshAdapter] Vt.Vec3fArray.FromNumpy failed; "
                f"shape={array.shape}, dtype={array.dtype}"
            ) from exc

    @staticmethod
    def _to_vt_int(values: np.ndarray, vt_module: Any) -> Any:
        array = np.ascontiguousarray(values, dtype=np.int32)
        try:
            return vt_module.IntArray.FromNumpy(array)
        except Exception as exc:
            raise RuntimeError(
                "[DynamicMeshAdapter] Vt.IntArray.FromNumpy failed; "
                f"shape={array.shape}, dtype={array.dtype}"
            ) from exc
