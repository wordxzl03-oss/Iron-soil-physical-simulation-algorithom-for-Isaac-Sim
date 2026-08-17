"""Modular Isaac Sim demo: six articulation-driven 25 m pile excavations.

This is the production P0 GUI entrypoint. Every physics frame flows through
``isaac_bulk_pipeline``; the deprecated ``HeightfieldSoil`` prototype is not
used. The implemented computational tool is explicitly
``FlatBottomQuadProxy_L0``. Soil remains a reduced-order 2.5-D height field.

Linux example::

    /home/eric/isaacsim/python.sh \
      /home/eric/Desktop/mesh/isaac_loader/interactive_dig_demo.py
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import traceback
from pathlib import Path
from time import perf_counter


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--project-config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "project_25m.yaml",
    )
    parser.add_argument(
        "--tool-config",
        type=Path,
        help="Optional bucket_small/medium/large YAML override.",
    )
    parser.add_argument(
        "--loader-usd",
        type=Path,
        default=Path(__file__).with_name("wheel_loader.usd"),
    )
    parser.add_argument("--scoops", type=int, default=6)
    parser.add_argument("--mesh-update-stride", type=int, default=4)
    parser.add_argument("--hold-frames", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "modular_25m_six_scoop",
    )
    # Accepted only so old launchers fail with a precise migration message
    # instead of forwarding these arguments into Kit.
    parser.add_argument("--grid-spacing-m", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--workspace-size-m", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--visual-mesh-stride", type=int, help=argparse.SUPPRESS)
    args, remaining = parser.parse_known_args()
    if args.scoops != 6:
        raise ValueError("P0 acceptance entrypoint requires exactly --scoops 6")
    if args.mesh_update_stride < 1:
        raise ValueError("--mesh-update-stride must be >= 1")
    sys.argv = [sys.argv[0], *remaining]
    return args


ARGS = _parse_args()

try:
    from isaacsim import SimulationApp
except ImportError:  # Isaac 4.2 compatibility namespace.
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1280,
        "height": 720,
        "multi_gpu": False,
        "anti_aliasing": 2,
    }
)

# Isaac/Omniverse modules must be imported only after SimulationApp starts.
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

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

for source_path in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaac_bulk_pipeline.config import load_config  # noqa: E402
from isaac_bulk_pipeline.interaction import (  # noqa: E402
    ContinuousSweepBuilder,
    ExcavationOperator,
)
from isaac_bulk_pipeline.robot import RobotAdapter  # noqa: E402
from isaac_bulk_pipeline.runtime import ActionRecorder, SimulationController  # noqa: E402
from isaac_bulk_pipeline.solvers import MinimumSlopeAdapter  # noqa: E402
from isaac_bulk_pipeline.terrain import (  # noqa: E402
    HeightmapIO,
    MassLedger,
    TerrainStateManager,
)
from isaac_bulk_pipeline.tools import (  # noqa: E402
    ToolDescriptorLoader,
    ToolKinematicsAdapter,
)
from isaac_bulk_pipeline.visualization import DynamicMeshAdapter  # noqa: E402


ACTUAL_PROXY_TYPE = "FlatBottomQuadProxy_L0"


def _configure_scene_lighting(stage) -> None:
    """Author deterministic daylight so the GUI never depends on Kit defaults."""

    UsdGeom.Xform.Define(stage, "/World/Lighting")

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/Sky")
    dome.CreateIntensityAttr(250.0)
    dome.CreateExposureAttr(0.0)
    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.00))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lighting/Sun")
    sun.CreateIntensityAttr(1100.0)
    sun.CreateExposureAttr(0.0)
    sun.CreateAngleAttr(1.0)
    sun.CreateColorAttr(Gf.Vec3f(1.00, 0.91, 0.76))
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
        Gf.Vec3f(-48.0, 24.0, -32.0)
    )

    # A broad overhead fill keeps the bucket-facing cut and the loader's dark
    # chassis readable even when the sun is behind the current camera view.
    fill = UsdLux.SphereLight.Define(stage, "/World/Lighting/OverheadFill")
    fill.CreateIntensityAttr(8000.0)
    fill.CreateExposureAttr(0.0)
    fill.CreateRadiusAttr(6.0)
    fill.CreateColorAttr(Gf.Vec3f(0.88, 0.93, 1.00))
    UsdGeom.Xformable(fill.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(17.5, 11.0, 24.0)
    )


def _front_contact_y(
    height_yx: np.ndarray,
    x_m: float,
    *,
    origin_x_m: float,
    origin_y_m: float,
    dx_m: float,
    dy_m: float,
) -> float:
    column = int(
        np.clip(
            round((x_m - origin_x_m) / dx_m),
            0,
            height_yx.shape[1] - 1,
        )
    )
    active_rows = np.flatnonzero(height_yx[:, column] > 0.25)
    row = int(active_rows[0] if len(active_rows) else np.argmax(height_yx[:, column]))
    return float(origin_y_m + row * dy_m)


def _pile_centre_x(height_yx: np.ndarray, *, origin_x_m: float, dx_m: float) -> float:
    weights_x = height_yx.sum(axis=0, dtype=np.float64)
    x = origin_x_m + np.arange(height_yx.shape[1], dtype=np.float64) * dx_m
    return float(np.dot(x, weights_x) / max(float(weights_x.sum()), 1e-12))


def _set_loader_pose(loader, x_m: float, y_m: float, z_m: float) -> None:
    # Loader local +X faces world/terrain +Y.
    orientation_wxyz = np.asarray(
        [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)],
        dtype=np.float32,
    )
    position = np.asarray([x_m, y_m, z_m], dtype=np.float32)
    try:
        loader.set_world_pose(position=position, orientation=orientation_wxyz)
    except TypeError:
        loader.set_world_poses(
            positions=position[None, :],
            orientations=orientation_wxyz[None, :],
        )


def _apply_joint_targets(
    articulation_controller,
    dof_names: list[str],
    targets: dict[str, float],
) -> None:
    positions = np.zeros(len(dof_names), dtype=np.float32)
    for name, value in targets.items():
        positions[dof_names.index(name)] = np.float32(value)
    articulation_controller.apply_action(
        ArticulationAction(joint_positions=positions)
    )


def _observe_physics_frame(
    *,
    world,
    robot: RobotAdapter,
    kinematics: ToolKinematicsAdapter,
    controller: SimulationController,
    recorder: ActionRecorder,
    simulation_frame: int,
    action_index: int,
    action_phase: str,
    cutting_enabled: bool,
    update_visualization: bool,
):
    world.step(render=not ARGS.headless)
    joints = robot.get_joint_state()
    tool_link_pose_world = robot.get_tool_link_pose_world()
    tool_state = kinematics.update(tool_link_pose_world, joints.timestamp)
    step_result = controller.process_tool_state(
        tool_state,
        cutting_enabled=cutting_enabled,
        update_visualization=update_visualization,
    )
    excavation = step_result.excavation
    affected_cell_count = (
        0 if excavation is None else int(excavation.affected_mask.sum())
    )
    removed_volume_m3 = (
        0.0 if excavation is None else float(excavation.removed_volume_m3)
    )
    recorder.record_frame(
        timestamp=joints.timestamp,
        simulation_frame=simulation_frame,
        action_index=action_index,
        action_phase=action_phase,
        joint_names=joints.names,
        joint_positions=joints.positions_rad,
        joint_velocities=joints.velocities_rad_s,
        tool_link_pose_world=tool_link_pose_world,
        tool_pose_terrain=tool_state.pose_terrain,
        cutting_enabled=cutting_enabled,
        affected_cell_count=affected_cell_count,
        removed_volume_m3=removed_volume_m3,
    )
    return tool_state, step_result


def _run_phase(
    *,
    world,
    loader,
    articulation_controller,
    dof_names: list[str],
    robot: RobotAdapter,
    kinematics: ToolKinematicsAdapter,
    controller: SimulationController,
    recorder: ActionRecorder,
    simulation_frame_counter: list[int],
    action_index: int,
    action_phase: str,
    x_m: float,
    root_z_m: float,
    y_start_m: float,
    y_end_m: float,
    lift_start_deg: float,
    lift_end_deg: float,
    curl_start_deg: float,
    curl_end_deg: float,
    frame_count: int,
    cutting_enabled: bool,
):
    last_tool_state = None
    removed_volume_m3 = 0.0
    for frame_index in range(frame_count):
        fraction = (frame_index + 1) / frame_count
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        y_m = (1.0 - smooth) * y_start_m + smooth * y_end_m
        lift = np.deg2rad(
            (1.0 - smooth) * lift_start_deg + smooth * lift_end_deg
        )
        curl = np.deg2rad(
            (1.0 - smooth) * curl_start_deg + smooth * curl_end_deg
        )
        _set_loader_pose(loader, x_m, y_m, root_z_m)
        _apply_joint_targets(
            articulation_controller,
            dof_names,
            {"lift_joint": float(lift), "bucket_joint": float(curl)},
        )
        simulation_frame_counter[0] += 1
        last_tool_state, step_result = _observe_physics_frame(
            world=world,
            robot=robot,
            kinematics=kinematics,
            controller=controller,
            recorder=recorder,
            simulation_frame=simulation_frame_counter[0],
            action_index=action_index,
            action_phase=action_phase,
            cutting_enabled=cutting_enabled,
            update_visualization=(
                simulation_frame_counter[0] % ARGS.mesh_update_stride == 0
            ),
        )
        if step_result.excavation is not None:
            removed_volume_m3 += step_result.excavation.removed_volume_m3
    return last_tool_state, float(removed_volume_m3)


def _validate_completed_episode(
    *,
    episode_dir: Path,
    grid,
    initial_heightmap: np.ndarray,
    action_summaries,
    ledger: MassLedger,
    recorder: ActionRecorder,
    stage,
    terrain_prim_path: str,
) -> dict[str, object]:
    with np.load(episode_dir / "H0_H6.npz", allow_pickle=False) as archive:
        heightmaps = np.array(archive["heightmaps_m"], copy=True)
    if heightmaps.shape != (7, 701, 701) or heightmaps.dtype != np.float32:
        raise RuntimeError(
            f"H0_H6 contract failed: shape={heightmaps.shape}, dtype={heightmaps.dtype}"
        )
    if not np.all(np.isfinite(heightmaps)) or np.any(heightmaps < 0.0):
        raise RuntimeError("H0_H6 contains NaN, Inf or negative terrain")
    np.testing.assert_allclose(heightmaps[0], initial_heightmap, atol=2e-6)
    continuity_ok = True
    for action_index in range(6):
        before = np.load(
            episode_dir / f"H_before_action_{action_index:03d}.npy",
            allow_pickle=False,
        )
        expected = (
            initial_heightmap
            if action_index == 0
            else np.load(
                episode_dir / f"H_stable_{action_index - 1:03d}.npy",
                allow_pickle=False,
            )
        )
        if not np.array_equal(before, expected):
            continuity_ok = False
            break
    if not continuity_ok:
        raise RuntimeError("H(N+1) did not begin from the previous H_stable")
    if len(action_summaries) != 6 or recorder.frame_count != 1440:
        raise RuntimeError(
            "six-action/full-frame count failed; "
            f"actions={len(action_summaries)}, frames={recorder.frame_count}"
        )
    if abs(ledger.numerical_error_m3) > 1e-5:
        raise RuntimeError(
            f"volume ledger did not close: error={ledger.numerical_error_m3}"
        )
    mesh_prim = stage.GetPrimAtPath(terrain_prim_path)
    if not mesh_prim.IsValid() or mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("dynamic terrain Prim invalid or has forbidden CollisionAPI")
    heightmap_volumes = [grid.compute_volume(item) for item in heightmaps]
    return {
        "heightmap_shape": list(heightmaps.shape),
        "heightmap_dtype": str(heightmaps.dtype),
        "heightmap_volumes_m3": heightmap_volumes,
        "continuity_ok": continuity_ok,
        "all_finite": True,
        "full_frame_log_count": recorder.frame_count,
        "effective_excavation_event_count": (
            recorder.effective_excavation_event_count
        ),
        "dynamic_mesh_prim_reused": True,
        "dynamic_mesh_collision_api": False,
    }


def main() -> None:
    started = perf_counter()
    project = load_config(ARGS.project_config)
    if project.robot is None or project.tool is None:
        raise RuntimeError("project_25m.yaml must define robot and tool sections")
    grid = project.terrain.to_grid()
    if grid.shape != (701, 701) or not np.isclose(grid.dx, 0.05) or not np.isclose(grid.dy, 0.05):
        raise RuntimeError(
            f"P0 requires 701x701 at 0.05 m; shape={grid.shape}, spacing={(grid.dx, grid.dy)}"
        )
    if ARGS.grid_spacing_m is not None and not np.isclose(ARGS.grid_spacing_m, grid.dx):
        raise ValueError("deprecated --grid-spacing-m disagrees with project_25m.yaml")
    if ARGS.workspace_size_m is not None:
        configured_span = (grid.nx - 1) * grid.dx
        if not np.isclose(ARGS.workspace_size_m, configured_span):
            raise ValueError(
                "deprecated --workspace-size-m disagrees with project_25m.yaml"
            )

    initial_heightmap = HeightmapIO.load(
        project.terrain.heightmap_path,
        grid=grid,
        source_axis_order=project.terrain.source_axis_order,
        output_dtype=np.float64,
    )
    if not ARGS.loader_usd.is_file():
        raise FileNotFoundError(f"loader USD not found: {ARGS.loader_usd}")

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    _configure_scene_lighting(stage)
    add_reference_to_stage(str(ARGS.loader_usd.resolve()), project.robot.robot_root_prim)
    loader = world.scene.add(
        SingleArticulation(
            prim_path=project.robot.articulation_root_prim,
            name="modular_six_scoop_loader",
        )
    )
    world.reset()

    robot = RobotAdapter(
        articulation=loader,
        time_source=lambda: float(world.current_time),
    )
    robot.initialize(stage, project.robot)
    tool_config = (
        project.tool
        if ARGS.tool_config is None
        else ToolDescriptorLoader.load_config(ARGS.tool_config)
    )
    descriptor = ToolDescriptorLoader.load(
        tool_config,
        stage=stage,
        tool_link_prim=project.robot.tool_link_prim,
    )
    if descriptor.actual_proxy_type != ACTUAL_PROXY_TYPE:
        raise RuntimeError(
            f"unexpected computational proxy: {descriptor.actual_proxy_type}"
        )

    terrain_mesh = DynamicMeshAdapter()
    terrain_mesh.initialize(
        stage,
        grid,
        project.mesh,
        initial_heightmap,
    )
    terrain_prim_before = stage.GetPrimAtPath(grid.terrain_prim_path)
    if terrain_prim_before.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError("dynamic terrain unexpectedly has CollisionAPI")

    state_manager = TerrainStateManager(grid, initial_heightmap)
    ledger = MassLedger.initialize(grid, initial_heightmap, project.material)
    solver = MinimumSlopeAdapter(grid, project.solver)
    recorder = ActionRecorder(ARGS.output_dir)
    episode_dir = recorder.initialize(
        initial_heightmap,
        {
            "runtime": "Isaac Sim 4.5.0",
            "project_config": str(ARGS.project_config.resolve()),
            "seed": project.project.seed,
            "robot_prim": project.robot.robot_root_prim,
            "articulation_prim": project.robot.articulation_root_prim,
            "tool_link_prim": project.robot.tool_link_prim,
            "tool_descriptor": descriptor.to_mapping(),
            "actual_proxy_type": ACTUAL_PROXY_TYPE,
            "proxy_limitations": [
                "flat four-point bucket-bottom quadrilateral only",
                "no curved bottom",
                "no teeth",
                "no interior capacity or payload model",
            ],
            "grid_shape_yx": list(grid.shape),
            "grid_spacing_m": [grid.dx, grid.dy],
            "initial_volume_m3": grid.compute_volume(initial_heightmap),
            "solver_sequence": {
                "sequence_enabled": project.solver.sequence_enabled,
                "sequence_stride": project.solver.sequence_stride,
                "sequence_max_frames": project.solver.sequence_max_frames,
                "sequence_dtype": project.solver.sequence_dtype,
                "sequence_memory_limit_mb": project.solver.sequence_memory_limit_mb,
            },
        },
    )
    kinematics = ToolKinematicsAdapter(descriptor, grid)
    controller = SimulationController(
        grid=grid,
        descriptor=descriptor,
        sweep_builder=ContinuousSweepBuilder(project.sweep),
        excavation_operator=ExcavationOperator(project.excavation),
        solver=solver,
        state_manager=state_manager,
        mass_ledger=ledger,
        kinematics_adapter=kinematics,
        mesh_adapter=terrain_mesh,
        recorder=recorder,
    )

    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=np.asarray([45.0, -30.0, 24.0]),
            target=np.asarray([17.5, 18.0, 3.0]),
            camera_prim_path="/OmniverseKit_Persp",
        )
    except Exception:
        pass

    dof_names = list(loader.dof_names)
    for required_joint in ("lift_joint", "bucket_joint"):
        if required_joint not in dof_names:
            raise RuntimeError(
                f"loader missing {required_joint}; available={dof_names}"
            )
    articulation_controller = loader.get_articulation_controller()
    centre_x = _pile_centre_x(
        initial_heightmap,
        origin_x_m=grid.origin_x,
        dx_m=grid.dx,
    )
    lateral_offsets = [0.0, -3.0, 3.0, -1.5, 1.5, 0.6]
    simulation_frame_counter = [0]
    action_summaries = []

    for scoop_index in range(6):
        action_index = controller.begin_action()
        if action_index != scoop_index:
            raise RuntimeError(
                f"action lifecycle index mismatch: {action_index} != {scoop_index}"
            )
        current_height = state_manager.state.H_current
        entry_x = centre_x + lateral_offsets[scoop_index]
        contact_y = _front_contact_y(
            current_height,
            entry_x,
            origin_x_m=grid.origin_x,
            origin_y_m=grid.origin_y,
            dx_m=grid.dx,
            dy_m=grid.dy,
        )
        root_start_y = contact_y - 5.7
        root_z = 1.55

        calibration_state, _ = _run_phase(
            world=world,
            loader=loader,
            articulation_controller=articulation_controller,
            dof_names=dof_names,
            robot=robot,
            kinematics=kinematics,
            controller=controller,
            recorder=recorder,
            simulation_frame_counter=simulation_frame_counter,
            action_index=action_index,
            action_phase="calibration",
            x_m=entry_x,
            root_z_m=root_z,
            y_start_m=root_start_y,
            y_end_m=root_start_y,
            lift_start_deg=-8.0,
            lift_end_deg=-8.0,
            curl_start_deg=-5.0,
            curl_end_deg=-5.0,
            frame_count=12,
            cutting_enabled=False,
        )
        assert calibration_state is not None
        measured_edge_z = float(
            np.min(calibration_state.cutting_edge_terrain[:, 2])
        )
        root_z += 0.15 - measured_edge_z

        _run_phase(
            world=world,
            loader=loader,
            articulation_controller=articulation_controller,
            dof_names=dof_names,
            robot=robot,
            kinematics=kinematics,
            controller=controller,
            recorder=recorder,
            simulation_frame_counter=simulation_frame_counter,
            action_index=action_index,
            action_phase="approach",
            x_m=entry_x,
            root_z_m=root_z,
            y_start_m=root_start_y - 0.7,
            y_end_m=root_start_y,
            lift_start_deg=-8.0,
            lift_end_deg=-8.0,
            curl_start_deg=-5.0,
            curl_end_deg=-5.0,
            frame_count=24,
            cutting_enabled=False,
        )
        _run_phase(
            world=world,
            loader=loader,
            articulation_controller=articulation_controller,
            dof_names=dof_names,
            robot=robot,
            kinematics=kinematics,
            controller=controller,
            recorder=recorder,
            simulation_frame_counter=simulation_frame_counter,
            action_index=action_index,
            action_phase="penetration",
            x_m=entry_x,
            root_z_m=root_z,
            y_start_m=root_start_y,
            y_end_m=root_start_y + 3.2,
            lift_start_deg=-8.0,
            lift_end_deg=-8.0,
            curl_start_deg=-5.0,
            curl_end_deg=-5.0,
            frame_count=70,
            cutting_enabled=True,
        )
        _run_phase(
            world=world,
            loader=loader,
            articulation_controller=articulation_controller,
            dof_names=dof_names,
            robot=robot,
            kinematics=kinematics,
            controller=controller,
            recorder=recorder,
            simulation_frame_counter=simulation_frame_counter,
            action_index=action_index,
            action_phase="curl_and_lift",
            x_m=entry_x,
            root_z_m=root_z,
            y_start_m=root_start_y + 3.2,
            y_end_m=root_start_y + 3.2,
            lift_start_deg=-8.0,
            lift_end_deg=20.0,
            curl_start_deg=-5.0,
            curl_end_deg=46.0,
            frame_count=64,
            cutting_enabled=True,
        )
        _run_phase(
            world=world,
            loader=loader,
            articulation_controller=articulation_controller,
            dof_names=dof_names,
            robot=robot,
            kinematics=kinematics,
            controller=controller,
            recorder=recorder,
            simulation_frame_counter=simulation_frame_counter,
            action_index=action_index,
            action_phase="retreat",
            x_m=entry_x,
            root_z_m=root_z,
            y_start_m=root_start_y + 3.2,
            y_end_m=root_start_y - 1.0,
            lift_start_deg=20.0,
            lift_end_deg=20.0,
            curl_start_deg=46.0,
            curl_end_deg=46.0,
            frame_count=70,
            cutting_enabled=False,
        )
        try:
            summary = controller.end_action()
        except BaseException:
            # Keep the exact pre-relaxation state when a long, full-resolution
            # acceptance run fails.  This makes solver failures reproducible
            # without replaying several minutes of Isaac physics first.
            np.save(
                episode_dir / f"H_failed_pre_relax_{action_index:03d}.npy",
                state_manager.state.H_current,
            )
            raise
        action_summaries.append(summary)
        print(
            json.dumps(
                {
                    "status": "ACTION_COMMITTED",
                    "action_index": summary.action_index,
                    "removed_volume_m3": summary.removed_volume_m3,
                    "relaxation_iterations": summary.relaxation.iteration_count,
                    "trajectory_pose_count": summary.trajectory_pose_count,
                    "frame_count": recorder.frame_count,
                }
            ),
            flush=True,
        )

    h0_h6_path = recorder.finalize_episode(expected_action_count=6)
    validation = _validate_completed_episode(
        episode_dir=episode_dir,
        grid=grid,
        initial_heightmap=initial_heightmap,
        action_summaries=action_summaries,
        ledger=ledger,
        recorder=recorder,
        stage=stage,
        terrain_prim_path=grid.terrain_prim_path,
    )
    elapsed_s = perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    result = {
        "status": "MODULAR_25M_SIX_SCOOP_OK",
        "episode_dir": str(episode_dir),
        "h0_h6_path": str(h0_h6_path),
        "actual_proxy_type": ACTUAL_PROXY_TYPE,
        "grid_shape_yx": list(grid.shape),
        "grid_spacing_m": [grid.dx, grid.dy],
        "action_removed_volumes_m3": [
            summary.removed_volume_m3 for summary in action_summaries
        ],
        "action_relaxation_iterations": [
            summary.relaxation.iteration_count for summary in action_summaries
        ],
        "total_removed_volume_m3": ledger.removed_volume_m3,
        "boundary_outflow_m3": ledger.boundary_outflow_m3,
        "numerical_error_m3": ledger.numerical_error_m3,
        "runtime_seconds": elapsed_s,
        "peak_process_rss_mb": peak_rss_mb,
        **validation,
    }
    (episode_dir / "run_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_path = REPOSITORY_ROOT / "outputs" / "modular_25m_six_scoop_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)

    if not ARGS.headless:
        for _ in range(max(0, ARGS.hold_frames)):
            world.step(render=True)


try:
    main()
except BaseException:
    failure_path = REPOSITORY_ROOT / "outputs" / "modular_25m_six_scoop_failure.txt"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(traceback.format_exc(), encoding="utf-8")
    raise
else:
    failure_path = REPOSITORY_ROOT / "outputs" / "modular_25m_six_scoop_failure.txt"
    failure_path.unlink(missing_ok=True)
    simulation_app.close()
