"""End-to-end Isaac smoke test for articulation-driven height-field digging."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np


PARSER = argparse.ArgumentParser(add_help=True)
PARSER.add_argument(
    "--gui",
    action="store_true",
    help="Show the articulated excavation and dynamic height-field updates.",
)
PARSER.add_argument(
    "--hold-frames",
    type=int,
    default=300,
    help="Rendered frames to hold the final stable terrain before Reset in GUI mode.",
)
ARGS, _UNKNOWN = PARSER.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": not ARGS.gui,
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
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics

try:
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.articulations import Articulation as SingleArticulation
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.types import ArticulationAction

from isaac_bulk_pipeline.config import (
    ExcavationConfig,
    MaterialConfig,
    MeshConfig,
    SolverConfig,
    SweepConfig,
    load_config,
)
from isaac_bulk_pipeline.interaction import ContinuousSweepBuilder, ExcavationOperator
from isaac_bulk_pipeline.robot import RobotAdapter
from isaac_bulk_pipeline.runtime import ActionRecorder, SimulationController
from isaac_bulk_pipeline.solvers import MinimumSlopeAdapter
from isaac_bulk_pipeline.terrain import MassLedger, TerrainGrid, TerrainStateManager
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolKinematicsAdapter
from isaac_bulk_pipeline.visualization import DynamicMeshAdapter


MARKER_ROOT = "/World/WheelLoader/bucket/Markers"
MARKERS_LINK_M = {
    "ToolOrigin": (1.30, 0.0, 0.0),
    "CuttingEdgeLeft": (1.30, 1.60, 0.0),
    "CuttingEdgeCenter": (1.30, 0.0, 0.0),
    "CuttingEdgeRight": (1.30, -1.60, 0.0),
    "BottomRearLeft": (-1.10, 1.60, 0.0),
    "BottomRearRight": (-1.10, -1.60, 0.0),
    "SideTopLeft": (-1.10, 1.60, 1.25),
    "SideTopRight": (-1.10, -1.60, 1.25),
}


def _author_markers(stage) -> None:
    UsdGeom.Xform.Define(stage, MARKER_ROOT)
    for name, position in MARKERS_LINK_M.items():
        marker = UsdGeom.Xform.Define(stage, f"{MARKER_ROOT}/{name}")
        marker.AddTranslateOp().Set(Gf.Vec3d(*position))


def _closed_irregular_pile(grid: TerrainGrid, centre_xy: np.ndarray) -> np.ndarray:
    x = grid.origin_x + np.arange(grid.nx) * grid.dx
    y = grid.origin_y + np.arange(grid.ny) * grid.dy
    xx, yy = np.meshgrid(x, y, indexing="xy")
    local_x = xx - centre_xy[0]
    local_y = yy - centre_xy[1]
    rho = np.sqrt((local_x / 5.1) ** 2 + (local_y / 4.6) ** 2)
    base = 3.8 * np.clip(1.0 - rho, 0.0, 1.0) ** 1.12
    shoulder = 0.75 * np.exp(
        -((local_x + 1.8) / 1.4) ** 2 - ((local_y - 1.2) / 1.1) ** 2
    )
    texture = (
        0.10 * np.sin(1.7 * local_x + 0.3 * local_y)
        + 0.07 * np.sin(2.4 * local_y - 0.2 * local_x)
    )
    height = np.maximum(base + shoulder + texture * np.clip(base / 1.0, 0.0, 1.0), 0.0)
    boundary = np.minimum.reduce(
        (
            xx - grid.origin_x,
            yy - grid.origin_y,
            grid.origin_x + (grid.nx - 1) * grid.dx - xx,
            grid.origin_y + (grid.ny - 1) * grid.dy - yy,
        )
    )
    taper = np.clip(boundary / 0.7, 0.0, 1.0)
    height *= taper * taper * (3.0 - 2.0 * taper)
    height[[0, -1], :] = 0.0
    height[:, [0, -1]] = 0.0
    return np.ascontiguousarray(height, dtype=np.float64)


def main() -> None:
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    add_reference_to_stage(
        str((REPOSITORY_ROOT / "isaac_loader" / "wheel_loader.usd").resolve()),
        "/World/WheelLoader",
    )
    _author_markers(stage)
    loader = world.scene.add(
        SingleArticulation(
            prim_path="/World/WheelLoader/rear_chassis",
            name="full_pipeline_loader",
        )
    )
    world.reset()

    project = load_config(REPOSITORY_ROOT / "configs" / "project.yaml")
    if project.robot is None:
        raise RuntimeError("project robot config is missing")
    robot = RobotAdapter(
        articulation=loader,
        time_source=lambda: float(world.current_time),
    )
    robot.initialize(stage, project.robot)
    marker_config = ToolDescriptorLoader.load_config(
        REPOSITORY_ROOT / "configs" / "bucket_markers.yaml"
    )
    descriptor = ToolDescriptorLoader.load(
        marker_config,
        stage=stage,
        tool_link_prim=project.robot.tool_link_prim,
    )

    initial_tool_link = robot.get_tool_link_pose_world()
    initial_tool_world = initial_tool_link @ descriptor.tool_to_link_matrix
    span = 12.75
    grid = TerrainGrid(
        nx=256,
        ny=256,
        dx=0.05,
        dy=0.05,
        origin_x=float(initial_tool_world[0, 3] - span / 2.0),
        origin_y=float(initial_tool_world[1, 3] - span / 2.0),
        terrain_prim_path="/World/Terrain/DynamicSurface",
    )
    initial_height = _closed_irregular_pile(
        grid, initial_tool_world[:2, 3]
    )
    initial_volume = grid.compute_volume(initial_height)
    terrain_mesh = DynamicMeshAdapter()
    terrain_mesh.initialize(
        stage,
        grid,
        MeshConfig(
            collision_enabled=False,
            update_normals=True,
            mesh_update_rate_hz=10.0,
            normal_update_rate_hz=5.0,
        ),
        initial_height,
    )
    mesh_prim_before = stage.GetPrimAtPath(grid.terrain_prim_path)
    if mesh_prim_before.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("dynamic terrain unexpectedly has CollisionAPI")

    state_manager = TerrainStateManager(grid, initial_height)
    ledger = MassLedger.initialize(
        grid,
        initial_height,
        MaterialConfig(
            bulk_density_kg_m3=2350.0,
            density_is_estimated=True,
        ),
    )
    solver = MinimumSlopeAdapter(
        grid,
        SolverConfig(
            solve_trigger="action_end",
            critical_angle_deg=34.0,
            max_iterations=4000,
            tolerance=1e-5,
            sequence_enabled=True,
            sequence_stride=200,
            sequence_max_frames=24,
            sequence_dtype="float32",
            sequence_memory_limit_mb=64.0,
            boundary_condition="closed",
            conservation_tolerance_m3=1e-7,
        ),
    )
    recorder = ActionRecorder(REPOSITORY_ROOT / "outputs" / "isaac_full_pipeline")
    episode_dir = recorder.initialize(
        initial_height,
        {
            "runtime": "Isaac Sim 4.5.0",
            "robot_prim": project.robot.robot_root_prim,
            "tool_link_prim": project.robot.tool_link_prim,
            "tool_descriptor_source": descriptor.metadata["descriptor_source"],
            "grid_shape_yx": list(grid.shape),
            "grid_spacing_m": [grid.dx, grid.dy],
            "initial_volume_m3": initial_volume,
        },
    )
    kinematics = ToolKinematicsAdapter(descriptor, grid)
    controller = SimulationController(
        grid=grid,
        descriptor=descriptor,
        sweep_builder=ContinuousSweepBuilder(SweepConfig()),
        excavation_operator=ExcavationOperator(ExcavationConfig()),
        solver=solver,
        state_manager=state_manager,
        mass_ledger=ledger,
        kinematics_adapter=kinematics,
        mesh_adapter=terrain_mesh,
        recorder=recorder,
    )

    controller.begin_action()
    controller.step_from_robot(robot)
    names = list(loader.dof_names)
    for required in ("lift_joint", "bucket_joint"):
        if required not in names:
            raise RuntimeError(f"missing joint {required}; available={names}")
    targets = np.zeros(len(names), dtype=np.float32)
    targets[names.index("lift_joint")] = np.deg2rad(20.0)
    targets[names.index("bucket_joint")] = np.deg2rad(30.0)
    articulation_controller = loader.get_articulation_controller()
    excavation_updates = 0
    sampled_pose_total = 0
    for frame in range(120):
        articulation_controller.apply_action(
            ArticulationAction(joint_positions=targets)
        )
        world.step(render=ARGS.gui)
        if (frame + 1) % 5 == 0:
            step_result = controller.step_from_robot(robot)
            if step_result.sweep is not None:
                sampled_pose_total += step_result.sweep.diagnostics[
                    "sampled_pose_count"
                ]
            if (
                step_result.excavation is not None
                and step_result.excavation.removed_volume_m3 > 0.0
            ):
                excavation_updates += 1
    summary = controller.end_action()
    simulation_app.update()
    if ARGS.gui:
        for _ in range(max(0, ARGS.hold_frames)):
            world.step(render=True)

    if summary.removed_volume_m3 <= 0.05:
        raise RuntimeError(
            f"real articulated sweep removed too little: {summary.removed_volume_m3} m3"
        )
    if summary.state.action_index != 1:
        raise RuntimeError(f"action state did not commit: {summary.state.action_index}")
    if abs(ledger.numerical_error_m3) > 1e-6:
        raise RuntimeError(f"volume ledger error too large: {ledger.numerical_error_m3}")
    if not summary.relaxation.converged:
        raise RuntimeError(
            "minimum-slope relaxation did not converge; "
            f"iterations={summary.relaxation.iteration_count}"
        )
    mesh_prim_after = stage.GetPrimAtPath(grid.terrain_prim_path)
    if mesh_prim_after.GetPath() != mesh_prim_before.GetPath():
        raise RuntimeError("terrain Prim identity changed")
    points = np.asarray(mesh_prim_after.GetAttribute("points").Get(), dtype=np.float64)
    if len(points) != grid.nx * grid.ny:
        raise RuntimeError(f"dynamic mesh vertex count mismatch: {len(points)}")
    np.testing.assert_allclose(
        points[:, 2], summary.state.H_current.ravel(), atol=2e-6
    )

    stable_before_reset = np.array(summary.state.H_current, copy=True)
    mass_estimate_before_reset = ledger.removed_mass_estimate_kg
    controller.reset()
    simulation_app.update()
    if not np.array_equal(state_manager.state.H_current, initial_height):
        raise RuntimeError("pipeline reset did not restore H_initial")
    reset_points = np.asarray(
        mesh_prim_after.GetAttribute("points").Get(), dtype=np.float64
    )
    np.testing.assert_allclose(reset_points[:, 2], initial_height.ravel(), atol=2e-6)

    result = {
        "status": "ISAAC_FULL_BULK_PIPELINE_SMOKE_OK",
        "episode_dir": str(episode_dir),
        "grid_shape_yx": list(grid.shape),
        "grid_spacing_m": [grid.dx, grid.dy],
        "dof_names": names,
        "tool_width_m": descriptor.nominal_width_m,
        "excavation_update_count": excavation_updates,
        "adaptive_sampled_pose_total": sampled_pose_total,
        "removed_volume_m3": summary.removed_volume_m3,
        "initial_terrain_volume_m3": initial_volume,
        "stable_terrain_volume_m3": grid.compute_volume(stable_before_reset),
        "boundary_outflow_m3": summary.boundary_outflow_m3,
        "numerical_error_m3": summary.numerical_error_m3,
        "relaxation_iterations": summary.relaxation.iteration_count,
        "relaxation_converged": summary.relaxation.converged,
        "relaxation_sequence_count": len(summary.relaxation.heightmap_sequence),
        "dynamic_mesh_vertex_count": len(points),
        "dynamic_mesh_prim_reused": True,
        "dynamic_mesh_collision_api": False,
        "reset_restored_initial": True,
        "mass_value_kind": "density-based estimate",
        "removed_mass_estimate_kg": mass_estimate_before_reset,
        "is_measured_bucket_payload": False,
    }
    output_path = REPOSITORY_ROOT / "outputs" / "phase5_isaac_pipeline_smoke.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    failure_path = REPOSITORY_ROOT / "outputs" / "phase5_isaac_pipeline_failure.txt"
    failure_path.unlink(missing_ok=True)
    print(json.dumps(result), flush=True)


try:
    main()
except BaseException:
    failure_path = REPOSITORY_ROOT / "outputs" / "phase5_isaac_pipeline_failure.txt"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
    raise
else:
    simulation_app.close()
