"""Isaac Sim smoke test for the Phase-1 fixed-topology mesh adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "multi_gpu": False,
        "width": 640,
        "height": 480,
        "anti_aliasing": 2,
    }
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import omni.usd
from pxr import UsdGeom, UsdPhysics

from isaac_bulk_pipeline.config import MeshConfig
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.visualization import DynamicMeshAdapter


def main() -> None:
    grid = TerrainGrid(
        nx=256,
        ny=256,
        dx=0.05,
        dy=0.05,
        origin_x=-6.375,
        origin_y=-6.375,
        terrain_prim_path="/World/Terrain/DynamicSurface",
    )
    yy, xx = np.meshgrid(
        np.arange(grid.ny) * grid.dy,
        np.arange(grid.nx) * grid.dx,
        indexing="ij",
    )
    initial = (
        2.2 * np.exp(-((xx - 7.5) / 2.8) ** 2 - ((yy - 5.0) / 1.8) ** 2)
        + 0.35 * np.exp(-((xx - 3.0) / 0.5) ** 2 - ((yy - 9.0) / 0.8) ** 2)
    ).astype(np.float64)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    adapter = DynamicMeshAdapter()
    adapter.initialize(
        stage,
        grid,
        MeshConfig(
            collision_enabled=False,
            update_normals=True,
            mesh_update_rate_hz=10.0,
            normal_update_rate_hz=5.0,
        ),
        initial,
    )
    prim_before = stage.GetPrimAtPath(grid.terrain_prim_path)
    if not prim_before.IsValid():
        raise RuntimeError("dynamic terrain Prim was not created")
    if prim_before.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("dynamic visual terrain unexpectedly has CollisionAPI")

    metrics = []
    current = initial.copy()
    for index in range(6):
        current[90:130, 80 + index : 120 + index] *= 0.985
        metrics.append(
            adapter.update(current, affected_bbox=(90, 80, 130, 126))
        )
        simulation_app.update()

    prim_after = stage.GetPrimAtPath(grid.terrain_prim_path)
    if prim_after.GetPath() != prim_before.GetPath():
        raise RuntimeError("dynamic terrain Prim identity/path changed during update")
    points = prim_after.GetAttribute("points").Get()
    face_indices = prim_after.GetAttribute("faceVertexIndices").Get()
    expected_vertices = grid.nx * grid.ny
    expected_indices = 6 * (grid.nx - 1) * (grid.ny - 1)
    if len(points) != expected_vertices or len(face_indices) != expected_indices:
        raise RuntimeError(
            "dynamic mesh topology mismatch: "
            f"points={len(points)}/{expected_vertices}, "
            f"indices={len(face_indices)}/{expected_indices}"
        )

    resolution_records = {
        "256": {
            "vertex_count": len(points),
            "triangle_count": len(face_indices) // 3,
            "update_elapsed_ms": [round(item.elapsed_ms, 3) for item in metrics],
        }
    }
    physical_span_m = 12.75
    for size in (128, 512):
        spacing = physical_span_m / (size - 1)
        variant_grid = TerrainGrid(
            nx=size,
            ny=size,
            dx=spacing,
            dy=spacing,
            origin_x=-physical_span_m / 2,
            origin_y=-physical_span_m / 2,
            terrain_prim_path=f"/World/Terrain/DynamicSurface_{size}",
        )
        variant_y, variant_x = np.meshgrid(
            np.linspace(-physical_span_m / 2, physical_span_m / 2, size),
            np.linspace(-physical_span_m / 2, physical_span_m / 2, size),
            indexing="ij",
        )
        variant_height = (
            2.0
            * np.exp(-((variant_x - 1.1) / 2.5) ** 2 - ((variant_y + 0.7) / 1.7) ** 2)
        )
        variant_adapter = DynamicMeshAdapter()
        variant_adapter.initialize(
            stage,
            variant_grid,
            MeshConfig(
                collision_enabled=False,
                update_normals=True,
                mesh_update_rate_hz=10.0,
                normal_update_rate_hz=5.0,
            ),
            variant_height,
        )
        variant_height[size // 3 : size // 2, size // 3 : size // 2] *= 0.98
        first_update = variant_adapter.update(variant_height)
        variant_height[size // 3 : size // 2, size // 3 : size // 2] *= 0.98
        second_update = variant_adapter.update(variant_height)
        simulation_app.update()
        variant_prim = stage.GetPrimAtPath(variant_grid.terrain_prim_path)
        variant_points = variant_prim.GetAttribute("points").Get()
        variant_indices = variant_prim.GetAttribute("faceVertexIndices").Get()
        if variant_prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"{size} terrain unexpectedly has CollisionAPI")
        if len(variant_points) != size * size:
            raise RuntimeError(f"{size} terrain vertex count mismatch")
        resolution_records[str(size)] = {
            "vertex_count": len(variant_points),
            "triangle_count": len(variant_indices) // 3,
            "update_elapsed_ms": [
                round(first_update.elapsed_ms, 3),
                round(second_update.elapsed_ms, 3),
            ],
        }
    print(
        json.dumps(
            {
                "status": "ISAAC_DYNAMIC_MESH_SMOKE_OK",
                "driver_expected": "580.173.02",
                "shape_yx": list(grid.shape),
                "vertex_count": len(points),
                "triangle_count": len(face_indices) // 3,
                "collision_api": False,
                "update_elapsed_ms": [round(item.elapsed_ms, 3) for item in metrics],
                "normal_updates": [item.normals_updated for item in metrics],
                "resolutions": resolution_records,
            }
        ),
        flush=True,
    )


try:
    main()
finally:
    simulation_app.close()
