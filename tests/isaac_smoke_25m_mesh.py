"""Isaac smoke test for the real 25 m, 0.05 m closed-pile configuration."""

from __future__ import annotations

import json
import sys
import traceback
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

from isaac_bulk_pipeline.config import load_config
from isaac_bulk_pipeline.terrain import HeightmapIO
from isaac_bulk_pipeline.visualization import DynamicMeshAdapter


def main() -> None:
    config = load_config(REPOSITORY_ROOT / "configs" / "project_25m.yaml")
    grid = config.terrain.to_grid()
    height = HeightmapIO.load(
        config.terrain.heightmap_path,
        grid=grid,
        source_axis_order=config.terrain.source_axis_order,
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    adapter = DynamicMeshAdapter()
    adapter.initialize(stage, grid, config.mesh, height)
    prim_before = stage.GetPrimAtPath(grid.terrain_prim_path)
    if prim_before.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("25 m dynamic visual mesh unexpectedly has CollisionAPI")

    updated = np.array(height, copy=True)
    patch = updated[300:360, 260:340]
    patch[...] = np.maximum(patch - 0.01, 0.0)
    metrics = adapter.update(updated, affected_bbox=(300, 260, 360, 340))
    simulation_app.update()
    prim_after = stage.GetPrimAtPath(grid.terrain_prim_path)
    points = prim_after.GetAttribute("points").Get()
    indices = prim_after.GetAttribute("faceVertexIndices").Get()
    if prim_after.GetPath() != prim_before.GetPath():
        raise RuntimeError("25 m terrain Prim was replaced during update")
    if len(points) != 701 * 701 or len(indices) // 3 != 2 * 700 * 700:
        raise RuntimeError(
            f"25 m topology mismatch: vertices={len(points)}, triangles={len(indices)//3}"
        )
    boundary_max = float(
        max(
            height[0].max(),
            height[-1].max(),
            height[:, 0].max(),
            height[:, -1].max(),
        )
    )
    result = {
        "status": "ISAAC_25M_DYNAMIC_MESH_SMOKE_OK",
        "shape_yx": list(height.shape),
        "grid_spacing_m": [grid.dx, grid.dy],
        "carrier_span_m": [(grid.nx - 1) * grid.dx, (grid.ny - 1) * grid.dy],
        "nominal_pile_span_m": 25.0,
        "peak_height_m": float(height.max()),
        "boundary_max_height_m": boundary_max,
        "vertex_count": len(points),
        "triangle_count": len(indices) // 3,
        "update_elapsed_ms": metrics.elapsed_ms,
        "prim_reused": True,
        "collision_api": False,
        "source_axis_conversion": "explicit legacy xy -> canonical yx",
    }
    output = REPOSITORY_ROOT / "outputs" / "phase5_25m_mesh_smoke.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    failure = REPOSITORY_ROOT / "outputs" / "phase5_25m_mesh_failure.txt"
    failure.unlink(missing_ok=True)
    print(json.dumps(result), flush=True)


try:
    main()
except BaseException:
    failure = REPOSITORY_ROOT / "outputs" / "phase5_25m_mesh_failure.txt"
    failure.parent.mkdir(parents=True, exist_ok=True)
    failure.write_text(traceback.format_exc(), encoding="utf-8")
    raise
else:
    simulation_app.close()
