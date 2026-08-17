"""Isaac Sim smoke test for Phase-2 robot/tool adapters and semantic markers."""

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
from pxr import Gf, UsdGeom

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

from isaac_bulk_pipeline.config import load_config
from isaac_bulk_pipeline.robot import RobotAdapter
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolKinematicsAdapter


MARKER_ROOT = "/World/WheelLoader/bucket/Markers"
MARKER_POSITIONS_LINK_M = {
    "ToolOrigin": (1.30, 0.0, 0.0),
    "CuttingEdgeLeft": (1.30, 1.60, 0.0),
    "CuttingEdgeCenter": (1.30, 0.0, 0.0),
    "CuttingEdgeRight": (1.30, -1.60, 0.0),
    "BottomRearLeft": (-1.10, 1.60, 0.0),
    "BottomRearRight": (-1.10, -1.60, 0.0),
    "SideTopLeft": (-1.10, 1.60, 1.25),
    "SideTopRight": (-1.10, -1.60, 1.25),
}


def _author_semantic_markers(stage) -> None:
    """Author geometry-free semantic Xforms in the bucket link frame."""

    UsdGeom.Xform.Define(stage, MARKER_ROOT)
    for name, position in MARKER_POSITIONS_LINK_M.items():
        marker = UsdGeom.Xform.Define(stage, f"{MARKER_ROOT}/{name}")
        marker.AddTranslateOp().Set(Gf.Vec3d(*position))


def main() -> None:
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    loader_usd = REPOSITORY_ROOT / "isaac_loader" / "wheel_loader.usd"
    add_reference_to_stage(str(loader_usd.resolve()), "/World/WheelLoader")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    _author_semantic_markers(stage)

    loader = world.scene.add(
        SingleArticulation(
            prim_path="/World/WheelLoader/rear_chassis",
            name="phase2_wheel_loader",
        )
    )
    world.reset()

    project_config = load_config(REPOSITORY_ROOT / "configs" / "project.yaml")
    if project_config.robot is None:
        raise RuntimeError("project config did not load a robot section")
    robot = RobotAdapter(
        articulation=loader,
        time_source=lambda: float(world.current_time),
    )
    robot.initialize(stage, project_config.robot)

    marker_config = ToolDescriptorLoader.load_config(
        REPOSITORY_ROOT / "configs" / "bucket_markers.yaml"
    )
    descriptor = ToolDescriptorLoader.load(
        marker_config,
        stage=stage,
        tool_link_prim=project_config.robot.tool_link_prim,
    )
    expected_link_from_tool = np.asarray(
        [
            [0.0, 1.0, 0.0, 1.30],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if not np.allclose(
        descriptor.tool_to_link_matrix, expected_link_from_tool, atol=1e-6
    ):
        raise RuntimeError(
            "semantic markers produced an unexpected Tool Frame: "
            f"{descriptor.tool_to_link_matrix.tolist()}"
        )
    if not np.isclose(descriptor.nominal_width_m, 3.2, atol=1e-6):
        raise RuntimeError(f"marker width mismatch: {descriptor.nominal_width_m}")

    grid = TerrainGrid(
        nx=256,
        ny=256,
        dx=0.05,
        dy=0.05,
        origin_x=-6.375,
        origin_y=-6.375,
        terrain_prim_path="/World/Terrain/DynamicSurface",
    )
    kinematics = ToolKinematicsAdapter(descriptor, grid)
    initial_joints = robot.get_joint_state()
    initial_state = kinematics.update(
        robot.get_tool_link_pose_world(), initial_joints.timestamp
    )

    names = list(loader.dof_names)
    for required in ("lift_joint", "bucket_joint"):
        if required not in names:
            raise RuntimeError(f"missing {required}; available={names}")
    targets = np.zeros(len(names), dtype=np.float32)
    targets[names.index("lift_joint")] = np.deg2rad(20.0)
    targets[names.index("bucket_joint")] = np.deg2rad(30.0)
    controller = loader.get_articulation_controller()
    for _ in range(180):
        controller.apply_action(ArticulationAction(joint_positions=targets))
        world.step(render=False)

    final_joints = robot.get_joint_state()
    final_state = kinematics.update(
        robot.get_tool_link_pose_world(), final_joints.timestamp
    )
    origin_delta = final_state.pose_world[:3, 3] - initial_state.pose_world[:3, 3]
    edge_delta = (
        final_state.cutting_edge_terrain
        - initial_state.cutting_edge_terrain
    )
    if np.linalg.norm(origin_delta) < 0.05:
        raise RuntimeError(
            f"joint targets did not move Tool Frame enough; delta={origin_delta.tolist()}"
        )
    if np.linalg.norm(edge_delta) < 0.05:
        raise RuntimeError("cutting-edge proxy did not follow the articulated bucket")
    for name in ("lift_joint", "bucket_joint"):
        index = names.index(name)
        if abs(final_joints.positions_rad[index] - initial_joints.positions_rad[index]) < 0.05:
            raise RuntimeError(
                f"joint state did not change for {name}: "
                f"initial={initial_joints.positions_rad[index]}, "
                f"final={final_joints.positions_rad[index]}"
            )

    result = {
        "status": "ISAAC_ROBOT_TOOL_ADAPTERS_SMOKE_OK",
        "dof_names": names,
        "tool_width_m": descriptor.nominal_width_m,
        "descriptor_source": descriptor.metadata["descriptor_source"],
        "tool_origin_delta_m": origin_delta.tolist(),
        "tool_speed_m_s": float(np.linalg.norm(final_state.linear_velocity)),
        "tool_angular_speed_rad_s": float(
            np.linalg.norm(final_state.angular_velocity)
        ),
        "scale_singular_values": final_state.diagnostics[
            "scale_singular_values"
        ],
        "non_uniform_scale": final_state.diagnostics["non_uniform_scale"],
        "joint_positions_rad": {
            name: float(final_joints.positions_rad[index])
            for index, name in enumerate(names)
        },
    }
    output_path = REPOSITORY_ROOT / "outputs" / "phase2_isaac_smoke.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    failure_path = REPOSITORY_ROOT / "outputs" / "phase2_isaac_failure.txt"
    failure_path.unlink(missing_ok=True)
    print(json.dumps(result), flush=True)


try:
    main()
except BaseException:
    failure_path = REPOSITORY_ROOT / "outputs" / "phase2_isaac_failure.txt"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
    raise
else:
    simulation_app.close()
