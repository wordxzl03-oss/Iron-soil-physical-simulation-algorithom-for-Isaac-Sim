"""Three-cycle 390F + modular bulk-pipeline Isaac acceptance runtime."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
import traceback

import numpy as np


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--case", choices=("A", "B", "C"), required=True)
PARSER.add_argument("--steps-per-phase", type=int, default=18)
PARSER.add_argument("--render", action="store_true")
ARGS, _UNKNOWN = PARSER.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "multi_gpu": False,
        "width": 640,
        "height": 360,
        "anti_aliasing": 2,
    }
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for path in (SOURCE_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


PHASES = (
    ("initial_pose", (15.0, 20.0, -30.0, 0.0)),
    ("approach_pile", (15.0, 5.0, -60.0, -20.0)),
    ("penetrate", (15.0, -10.0, -70.0, -30.0)),
    ("coordinated_cut", (15.0, 0.0, -60.0, 10.0)),
    ("curl_filling", (15.0, 12.0, -45.0, 45.0)),
    ("breakout", (15.0, 25.0, -35.0, 45.0)),
    ("lift", (15.0, 35.0, -35.0, 40.0)),
    ("upper_body_swing", (65.0, 35.0, -35.0, 40.0)),
    ("dump_spill", (65.0, 30.0, -40.0, -80.0)),
    ("deposition", (65.0, 30.0, -40.0, -80.0)),
    ("swing_back", (15.0, 30.0, -35.0, 0.0)),
    ("next_dig_ready", (15.0, 20.0, -30.0, 0.0)),
)
DIG_PHASES = {"penetrate", "coordinated_cut", "curl_filling"}


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _profile_summary(samples: dict[str, list[float]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for name, values in sorted(samples.items()):
        data = np.asarray(values, dtype=np.float64)
        result[name] = {
            "sample_count": int(len(data)),
            "mean_ms": float(np.mean(data)),
            "p50_ms": float(np.percentile(data, 50.0)),
            "p95_ms": float(np.percentile(data, 95.0)),
            "max_ms": float(np.max(data)),
        }
    return result


def _prim_or_descendant_has_api(prim, api_type) -> bool:
    if not prim or not prim.IsValid():
        return False
    if prim.HasAPI(api_type):
        return True
    return any(_prim_or_descendant_has_api(child, api_type) for child in prim.GetChildren())


def main() -> None:
    import omni.usd
    from pxr import UsdGeom, UsdPhysics
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

    from isaac_bulk_pipeline.bulk_exchange import DumpTarget, TerrainDumpOperator
    from isaac_bulk_pipeline.bulk_interaction import (
        BulkMaterialInteractionModel,
        ToolTerrainIntersectionModel,
    )
    from isaac_bulk_pipeline.bulk_state import (
        BucketInternalFillModel,
        BulkStateManager,
        MaterialScenario,
        PayloadState,
        TerrainState,
        TerrainVolumeIntegrator,
    )
    from isaac_bulk_pipeline.config import MeshConfig, RobotConfig, SolverConfig, SweepConfig
    from isaac_bulk_pipeline.interaction import ContinuousSweepBuilder
    from isaac_bulk_pipeline.robot import RobotAdapter
    from isaac_bulk_pipeline.soil_force import (
        IsaacPayloadMassAdapter,
        IsaacSoilForceAdapter,
        MobileMomentumBudget,
        SoilForceModel,
    )
    from isaac_bulk_pipeline.solvers import MinimumSlopeAdapter
    from isaac_bulk_pipeline.terrain import HeightmapIO, TerrainGrid
    from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolKinematicsAdapter
    from isaac_bulk_pipeline.vehicle import (
        ExcavatorActuatorConfig,
        ExcavatorActuatorModel,
    )
    from isaac_bulk_pipeline.visualization import DynamicMeshAdapter, LightingManager

    deterministic_seed = 390
    np.random.seed(deterministic_seed)

    case_dir = REPOSITORY_ROOT / "outputs" / "integrated_390f_v1" / f"case_{ARGS.case}"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir.parent / f"case_{ARGS.case}_failure.json").unlink(missing_ok=True)
    crash_progress_path = case_dir / "last_physics_step_progress.json"
    dt = 1.0 / 60.0
    world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
    ground = world.scene.add_default_ground_plane(z_position=0.0)
    asset = Path("/home/eric/桌面/bulldozer_sim/bulldozer_main.usd")
    prefix = "/World/Excavator"
    add_reference_to_stage(str(asset), prefix)
    root = (
        f"{prefix}/_90F_LME_ISAAC_DETAIL_DELIVERY/"
        "tn__390F_LME_ISAAC_DETAIL_DELIVERY_/tn__aaa0ab9"
    )
    bucket_path = f"{root}/tn__390F_DETAIL_DELIVERY_Bucket1_zf0"
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    # The source file contains targetVelocity=800 on the bucket.  It is not a
    # valid 390F command, so all four drive feed-forward velocities are cleared.
    for joint_name in ("swing_joint", "boom_joint", "stick_joint", "bucket_joint"):
        drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{root}/Joints/{joint_name}"), "angular"
        )
        drive.GetTargetVelocityAttr().Set(0.0)

    excavator = world.scene.add(SingleArticulation(root, name=f"integrated_390f_{ARGS.case}"))
    bucket_view = world.scene.add(
        RigidPrim(bucket_path, name=f"bucket_force_{ARGS.case}", reset_xform_properties=False)
    )
    ground_path = getattr(ground, "prim_path", "/World/defaultGroundPlane")
    tracks = world.scene.add(
        RigidPrim(
            prim_paths_expr=f"{root}/tn__390F_DETAIL_DELIVERY_Track_.*",
            name=f"track_contact_{ARGS.case}",
            reset_xform_properties=False,
        )
    )

    terrain_transform = np.eye(4)
    terrain_transform[:3, 3] = [-0.25, -17.5, 0.0]
    grid = TerrainGrid(
        701,
        701,
        0.05,
        0.05,
        0.0,
        0.0,
        "/World/Terrain/DynamicBulkSurface",
        terrain_to_world_matrix=terrain_transform,
    )
    initial_height = HeightmapIO.load(
        REPOSITORY_ROOT
        / "continuous_heightmap_25m_closed_dataset"
        / "sequence_000_H0_initial_m.csv",
        grid=grid,
        source_axis_order="xy",
    )
    mesh = DynamicMeshAdapter()
    mesh.initialize(
        stage,
        grid,
        MeshConfig(
            collision_enabled=False,
            update_normals=True,
            mesh_update_rate_hz=10.0,
            normal_update_rate_hz=5.0,
            display_color_rgb=(0.33, 0.12, 0.035),
        ),
        initial_height,
    )
    LightingManager("outdoor_day").author_usd(stage, "/World/Integrated390FLighting")

    render_product = None
    if ARGS.render:
        import omni.replicator.core as rep

        camera = rep.create.camera(position=(15.0, -19.0, 12.0), look_at=(5.5, 0.0, 2.2))
        render_product = rep.create.render_product(camera, (640, 360))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=str(case_dir / "frames"), rgb=True)
        writer.attach([render_product])

    world.reset()
    names = tuple(str(item) for item in excavator.dof_names)
    required = ("swing_joint", "boom_joint", "stick_joint", "bucket_joint")
    if names != required:
        raise RuntimeError(f"unexpected 390F DOFs: {names}")
    controller = excavator.get_articulation_controller()
    initial_joint_position = np.asarray(
        excavator.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    actuator_config = ExcavatorActuatorConfig.cat_390f_l_mass_configuration()
    limit_by_name = {item.joint_name: item for item in actuator_config.joints}
    drive_by_name = {
        joint_name: UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{root}/Joints/{joint_name}"), "angular"
        )
        for joint_name in required
    }
    # Measure the source drive effort required to hold the authored initial
    # configuration.  It seeds gravity/load compensation before the position
    # drives are disabled; subsequent cycle motion is torque-controlled.
    controller.apply_action(ArticulationAction(joint_positions=initial_joint_position))
    world.step(render=False)
    initial_hold_effort = np.asarray(
        excavator.get_measured_joint_efforts(), dtype=np.float64
    ).reshape(-1)
    for joint_name in required:
        drive_by_name[joint_name].GetMaxForceAttr().Set(
            limit_by_name[joint_name].effort_limit_nm
        )
    actuator = ExcavatorActuatorModel(actuator_config, names)
    actuator.reset(
        np.asarray(excavator.get_joint_velocities(), dtype=np.float64).reshape(-1),
        initial_hold_effort,
    )
    runtime_time_origin_s = float(world.current_time)
    drive_target = np.asarray(
        excavator.get_joint_positions(), dtype=np.float64
    ).reshape(-1)
    robot = RobotAdapter(
        articulation=excavator,
        time_source=lambda: float(world.current_time) - runtime_time_origin_s,
    )
    robot.initialize(
        stage,
        RobotConfig(prefix, root, bucket_path),
    )
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(
            REPOSITORY_ROOT / "configs" / "excavator_390f_real_bucket.yaml"
        )
    )
    kinematics = ToolKinematicsAdapter(descriptor, grid)
    sweep_builder = ContinuousSweepBuilder(SweepConfig())
    intersection_model = ToolTerrainIntersectionModel()
    interaction_model = BulkMaterialInteractionModel()
    soil_force_model = SoilForceModel()
    dump_operator = TerrainDumpOperator()
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    material = MaterialScenario(
        "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED_FLOW_ANGLES_ENGINEERING",
        1370.0,
        29.8,
        800.0,
        float(np.tan(np.deg2rad(34.4))),
        38.0,
        30.0,
        0.35,
    )
    minislope = MinimumSlopeAdapter(
        grid,
        SolverConfig(
            critical_angle_deg=material.stop_angle_deg,
            max_iterations=1_000,
            tolerance=1.0e-8,
            boundary_condition="closed",
            conservation_tolerance_m3=1.0e-8,
        ),
    )
    payload = PayloadState(0.0, descriptor.effective_capacity_m3, material.assumed_bulk_density_kg_m3, np.zeros(3))
    initial_state = TerrainState(
        initial_height,
        np.zeros(grid.shape),
        np.zeros(grid.shape + (2,)),
        payload,
        (),
        material,
        0.0,
        0.0,
        0,
    )
    manager = BulkStateManager(initial_state, integrator, boundary_condition="closed")
    soil_adapter = IsaacSoilForceAdapter(bucket_view)

    com_world = np.asarray(bucket_view.get_coms()[0], dtype=np.float64).reshape(-1, 3)[0]
    link_world = robot.get_tool_link_pose_world()
    base_com_link = (np.linalg.inv(link_world) @ np.r_[com_world, 1.0])[:3]
    payload_mass_adapter = IsaacPayloadMassAdapter(
        stage, bucket_path, descriptor, base_com_link_m=base_com_link
    )
    last_authored_payload_mass_kg = 0.0

    def update_payload_mass_at_event(payload_state: PayloadState) -> bool:
        nonlocal last_authored_payload_mass_kg
        requested = payload_state.estimated_payload_mass_kg
        if np.isclose(requested, last_authored_payload_mass_kg, rtol=0.0, atol=1.0e-9):
            return False
        payload_mass_adapter.update_action_event(payload_state)
        last_authored_payload_mass_kg = requested
        return True

    dynamic_prim = stage.GetPrimAtPath(grid.terrain_prim_path)
    collision_evidence = {
        "support_ground_collision_api": bool(
            _prim_or_descendant_has_api(
                stage.GetPrimAtPath(ground_path), UsdPhysics.CollisionAPI
            )
        ),
        "track_collision_api": [
            bool(stage.GetPrimAtPath(path).GetChild("Mesh").HasAPI(UsdPhysics.CollisionAPI))
            for path in tracks.prim_paths
        ],
        "dynamic_bulk_collision_api": bool(dynamic_prim.HasAPI(UsdPhysics.CollisionAPI)),
        "bucket_collision_api": bool(
            stage.GetPrimAtPath(bucket_path).GetChild("Mesh").HasAPI(UsdPhysics.CollisionAPI)
        ),
        "bucket_bulk_rigid_pair_possible": False,
        "custom_bulk_solver_enabled": True,
    }

    rows: list[dict[str, object]] = []
    terrain_sequence = [np.array(initial_height, dtype=np.float32, copy=True)]
    cycle_summaries: list[dict[str, object]] = []
    timings: dict[str, list[float]] = {}
    previous_tool = kinematics.update(robot.get_tool_link_pose_world(), 0.0)
    previous_target = np.asarray(PHASES[-1][1], dtype=np.float64)
    initial_volume = manager.ledger_snapshot().initial_accounted_volume_m3
    total_soil_impulse = np.zeros(3)
    total_mechanical_work_j = 0.0
    physics_steps = 0
    custom_solver_calls = 0
    minislope_calls = 0
    maximum_lag_s = 0.0

    def relax_resting_height(heightmap: np.ndarray) -> tuple[np.ndarray, float]:
        nonlocal minislope_calls
        relax_start = perf_counter()
        result = minislope.solve(heightmap)
        timings.setdefault("minislope", []).append(
            (perf_counter() - relax_start) * 1_000.0
        )
        minislope_calls += 1
        return result.heightmap_stable, result.boundary_outflow_m3

    for cycle in range(1, 4):
        cycle_start_volume = manager.snapshot().payload.volume_m3
        cycle_start_terrain = integrator.integrate(manager.snapshot().H_resting_m)
        cycle_work = 0.0
        for phase_index, (phase, target_deg) in enumerate(PHASES):
            phase_start_target = previous_target.copy()
            phase_target = np.asarray(target_deg, dtype=np.float64)
            if phase == "dump_spill":
                update_payload_mass_at_event(manager.snapshot().payload)
            for local_step in range(ARGS.steps_per_phase):
                step_wall_start = perf_counter()
                fraction = _smoothstep((local_step + 1) / ARGS.steps_per_phase)
                target = np.deg2rad(
                    phase_start_target + fraction * (phase_target - phase_start_target)
                )
                robot_state_time = float(world.current_time) - runtime_time_origin_s
                current_tool = kinematics.update(
                    robot.get_tool_link_pose_world(), robot_state_time
                )
                applied_force = np.zeros(3)
                computed_force = np.zeros(3)
                quasi_force = np.zeros(3)
                momentum_force = np.zeros(3)
                soil_impulse = np.zeros(3)
                interaction = None
                force_result = None
                before = manager.snapshot()
                terrain_state_time = before.timestamp_s
                if phase in DIG_PHASES:
                    coupled_tool = BucketInternalFillModel.apply_secondary_separation(
                        current_tool, before.payload, descriptor
                    )
                    sweep = sweep_builder.build(previous_tool, current_tool, grid, descriptor)
                    intersection = intersection_model.compute(
                        before.H_resting_m, sweep, coupled_tool, grid, integrator
                    )
                    interaction_start = perf_counter()
                    interaction = interaction_model.advance(
                        manager,
                        intersection,
                        coupled_tool,
                        descriptor,
                        grid,
                        integrator,
                        dt,
                        action_index=(cycle - 1) * len(PHASES) + phase_index,
                    )
                    timings.setdefault("soil_solver", []).append(
                        (perf_counter() - interaction_start) * 1_000.0
                    )
                    for key, value in interaction.timings_ms.items():
                        timings.setdefault(key, []).append(value)
                    force_start = perf_counter()
                    force_result = soil_force_model.compute(
                        interaction.failure_zone,
                        intersection,
                        before.material,
                        descriptor,
                        coupled_tool,
                        MobileMomentumBudget.from_mobile_result(
                            interaction.mobile_result,
                            interaction.activation_tool_impulse_on_mobile_terrain_ns,
                            dt,
                        ),
                    )
                    timings.setdefault("soil_force", []).append(
                        (perf_counter() - force_start) * 1_000.0
                    )
                    computed_force = np.asarray(force_result.force_terrain_n)
                    quasi_force = np.asarray(force_result.quasi_static_force_terrain_n)
                    momentum_force = np.asarray(force_result.active_momentum_force_terrain_n)
                    if ARGS.case == "B":
                        applied_force = quasi_force
                        transform = grid.terrain_to_world_matrix
                        point_h = transform @ np.r_[force_result.application_point_terrain_m, 1.0]
                        bucket_view.apply_forces_and_torques_at_pos(
                            forces=(transform[:3, :3] @ applied_force).astype(np.float32).reshape(1, 3),
                            torques=np.zeros((1, 3), dtype=np.float32),
                            positions=(point_h[:3] / point_h[3]).astype(np.float32).reshape(1, 3),
                            is_global=True,
                        )
                    elif ARGS.case == "C":
                        applied_force = computed_force
                        soil_adapter.apply(force_result, grid.terrain_to_world_matrix)
                    soil_impulse = applied_force * dt
                    total_soil_impulse += soil_impulse
                    custom_solver_calls += 1
                elif phase == "dump_spill":
                    release_start = perf_counter()
                    dump_operator.release(
                        manager,
                        current_tool,
                        descriptor,
                        target=DumpTarget.TERRAIN,
                    )
                    timings.setdefault("retention_spill", []).append(
                        (perf_counter() - release_start) * 1_000.0
                    )

                pre_step_q = np.asarray(
                    excavator.get_joint_positions(), dtype=np.float64
                ).reshape(-1)
                pre_step_qd = np.asarray(
                    excavator.get_joint_velocities(), dtype=np.float64
                ).reshape(-1)
                actuator_output = actuator.step(target, pre_step_q, pre_step_qd, dt)
                crash_progress_path.write_text(
                    json.dumps(
                        {
                            "stage": "BEFORE_WORLD_STEP",
                            "case": ARGS.case,
                            "cycle": cycle,
                            "phase": phase,
                            "local_step": local_step,
                            "physics_step": physics_steps + 1,
                            "joint_position_rad": pre_step_q.tolist(),
                            "joint_velocity_rad_s": pre_step_qd.tolist(),
                            "effort_command_nm": actuator_output.effort_command_nm.tolist(),
                            "payload_volume_m3": manager.snapshot().payload.volume_m3,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                for joint_index, joint_name in enumerate(names):
                    drive_by_name[joint_name].GetMaxForceAttr().Set(
                        max(abs(float(actuator_output.effort_command_nm[joint_index])), 1.0)
                    )
                    limit = limit_by_name[joint_name]
                    if abs(pre_step_qd[joint_index]) > limit.velocity_limit_rad_s:
                        drive_target[joint_index] = (
                            pre_step_q[joint_index]
                            - np.sign(pre_step_qd[joint_index])
                            * limit.velocity_limit_rad_s
                            * dt
                        )
                    else:
                        drive_target[joint_index] += (
                            actuator_output.target_velocity_rad_s[joint_index] * dt
                        )
                controller.apply_action(
                    ArticulationAction(joint_positions=drive_target)
                )
                physics_start = perf_counter()
                render_this_step = bool(ARGS.render and physics_steps % 6 == 0)
                world.step(render=render_this_step)
                timings.setdefault("physics_step", []).append(
                    (perf_counter() - physics_start) * 1_000.0
                )
                physics_steps += 1
                crash_progress_path.write_text(
                    json.dumps(
                        {
                            "stage": "AFTER_WORLD_STEP",
                            "case": ARGS.case,
                            "cycle": cycle,
                            "phase": phase,
                            "local_step": local_step,
                            "physics_step": physics_steps,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

                if phase not in DIG_PHASES:
                    dump_start = perf_counter()
                    dump_operator.advance_airborne(
                        manager,
                        grid,
                        integrator,
                        dt,
                        resting_relaxation=(
                            relax_resting_height
                            if phase == "deposition"
                            and local_step == ARGS.steps_per_phase - 1
                            else None
                        ),
                    )
                    timings.setdefault("airborne_deposition", []).append(
                        (perf_counter() - dump_start) * 1_000.0
                    )
                state = manager.snapshot()
                if physics_steps % 6 == 0:
                    mesh_start = perf_counter()
                    mesh.update(state.H_resting_m)
                    timings.setdefault("terrain_update", []).append(
                        (perf_counter() - mesh_start) * 1_000.0
                    )
                q = np.asarray(excavator.get_joint_positions(), dtype=np.float64).reshape(-1)
                qd = np.asarray(excavator.get_joint_velocities(), dtype=np.float64).reshape(-1)
                measured_effort = np.asarray(
                    excavator.get_measured_joint_efforts(), dtype=np.float64
                ).reshape(-1)
                incremental_work = float(np.sum(np.abs(measured_effort * qd)) * dt)
                cycle_work += incremental_work
                total_mechanical_work_j += incremental_work
                tip = np.mean(current_tool.cutting_edge_terrain, axis=0)
                ledger = manager.ledger_snapshot()
                soil_force_time = robot_state_time if force_result is not None else None
                visual_time = state.timestamp_s if physics_steps % 6 == 0 else state.timestamp_s - (physics_steps % 6) * dt
                maximum_lag_s = max(maximum_lag_s, state.timestamp_s - visual_time)
                row: dict[str, object] = {
                    "case": ARGS.case,
                    "cycle": cycle,
                    "phase": phase,
                    "physics_step": physics_steps,
                    "robot_state_timestamp_s": robot_state_time,
                    "soil_force_timestamp_s": soil_force_time,
                    "terrain_state_timestamp_s": state.timestamp_s,
                    "visual_mesh_timestamp_s": visual_time,
                    "contact_surface_timestamp_s": 0.0,
                    "joint_target_rad": json.dumps(target.tolist()),
                    "joint_position_rad": json.dumps(q.tolist()),
                    "joint_velocity_rad_s": json.dumps(qd.tolist()),
                    "joint_effort_nm": json.dumps(measured_effort.tolist()),
                    "actuator_effort_command_nm": json.dumps(
                        actuator_output.effort_command_nm.tolist()
                    ),
                    "actuator_target_velocity_rad_s": json.dumps(
                        actuator_output.target_velocity_rad_s.tolist()
                    ),
                    "actuator_positive_power_w": actuator_output.positive_mechanical_power_w,
                    "actuator_shared_power_scale": actuator_output.shared_power_scale,
                    "actuator_effort_saturated": json.dumps(
                        actuator_output.effort_saturated.tolist()
                    ),
                    "actuator_velocity_saturated": json.dumps(
                        actuator_output.velocity_saturated.tolist()
                    ),
                    "actuator_acceleration_saturated": json.dumps(
                        actuator_output.acceleration_saturated.tolist()
                    ),
                    "bucket_tip_terrain_m": json.dumps(tip.tolist()),
                    "computed_soil_force_terrain_n": json.dumps(computed_force.tolist()),
                    "quasi_static_force_terrain_n": json.dumps(quasi_force.tolist()),
                    "momentum_force_terrain_n": json.dumps(momentum_force.tolist()),
                    "failure_zone_applicability": (
                        None
                        if interaction is None
                        else interaction.failure_zone.applicability_status
                    ),
                    "failure_zone_candidate_strips": (
                        0
                        if interaction is None
                        else interaction.failure_zone.candidate_strip_count
                    ),
                    "failure_zone_excluded_strips": (
                        0
                        if interaction is None
                        else interaction.failure_zone.excluded_strip_count
                    ),
                    "material_activation_mode": (
                        None if interaction is None else interaction.activation_mode
                    ),
                    "applied_soil_force_terrain_n": json.dumps(applied_force.tolist()),
                    "soil_impulse_terrain_ns": json.dumps(soil_impulse.tolist()),
                    "payload_volume_m3": state.payload.volume_m3,
                    "payload_mass_kg": state.payload.estimated_payload_mass_kg,
                    "payload_geometric_fill_ratio": state.payload.fill_ratio,
                    "payload_com_bucket_m": json.dumps(state.payload.center_of_mass_bucket_frame_m.tolist()),
                    "payload_phase": None if state.payload.internal_fill is None else state.payload.internal_fill.phase.value,
                    "separation_plane_source": (
                        current_tool.separation_plane_source
                        if interaction is None
                        else coupled_tool.separation_plane_source
                    ),
                    "resting_volume_m3": integrator.integrate(state.H_resting_m),
                    "mobile_volume_m3": integrator.integrate(state.mobile_height_m),
                    "airborne_volume_m3": state.airborne_volume_m3,
                    "outflow_volume_m3": state.outflow_volume_m3,
                    "mass_balance_error_m3": ledger.balance.absolute_volume_error_m3,
                    "incremental_abs_mechanical_work_j": incremental_work,
                }
                rows.append(row)
                timings.setdefault("full_step", []).append(
                    (perf_counter() - step_wall_start) * 1_000.0
                )
                previous_tool = current_tool
            previous_target = phase_target
            update_payload_mass_at_event(manager.snapshot().payload)
        state = manager.snapshot()
        terrain_sequence.append(np.array(state.H_resting_m, dtype=np.float32, copy=True))
        cycle_summaries.append(
            {
                "cycle": cycle,
                "payload_start_m3": cycle_start_volume,
                "payload_end_m3": state.payload.volume_m3,
                "terrain_start_m3": cycle_start_terrain,
                "terrain_end_m3": integrator.integrate(state.H_resting_m),
                "airborne_end_m3": state.airborne_volume_m3,
                "mobile_end_m3": integrator.integrate(state.mobile_height_m),
                "mechanical_work_abs_j": cycle_work,
                "mass_balance_error_m3": manager.ledger_snapshot().balance.absolute_volume_error_m3,
            }
        )

    final_state = manager.snapshot()
    collision_evidence["track_ground_contact_force_matrix_n"] = None
    collision_evidence["track_ground_contact_nonzero"] = None
    collision_evidence["track_contact_sensor_status"] = (
        "NOT_MEASURED: ISAAC_4_5_CONTACT_SENSOR_VIEW CAUSED REPRODUCIBLE_NATIVE_CRASH; "
        "TRACK_AND_SUPPORT_COLLISION_API_EVIDENCE_RETAINED"
    )
    collision_evidence["custom_solver_call_count"] = custom_solver_calls
    collision_evidence["minislope_call_count"] = minislope_calls

    log_path = case_dir / "three_cycle_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        case_dir / "terrain_state_sequence.npz",
        **{f"cycle_{index}": value for index, value in enumerate(terrain_sequence)},
    )
    if ARGS.case == "C":
        np.savetxt(case_dir / "before_heightmap_m.csv", initial_height, delimiter=",")
        np.savetxt(case_dir / "after_heightmap_m.csv", final_state.H_resting_m, delimiter=",")
    simulated_duration = physics_steps * dt
    profile = {
        "case": ARGS.case,
        "physics_dt_s": dt,
        "soil_grid_shape": list(grid.shape),
        "soil_grid_spacing_m": [grid.dx, grid.dy],
        "silent_resolution_downgrade": False,
        "module_timings": _profile_summary(timings),
        "simulated_duration_s": simulated_duration,
        "measured_wall_time_s": sum(timings["full_step"]) / 1000.0,
        "rtf": simulated_duration / max(sum(timings["full_step"]) / 1000.0, 1e-12),
        "max_resident_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    summary = {
        "case": ARGS.case,
        "soil_force_mode": {"A": "NO_SOIL_FORCE", "B": "QUASI_STATIC_FEE_ONLY", "C": "FULL_SOIL_MODEL"}[ARGS.case],
        "cycles": cycle_summaries,
        "physics_step_count": physics_steps,
        "custom_solver_call_count": custom_solver_calls,
        "minislope_call_count": minislope_calls,
        "initial_total_volume_m3": initial_volume,
        "final_total_volume_m3": manager.ledger_snapshot().reservoirs.total_including_outflow_m3,
        "final_mass_balance_error_m3": manager.ledger_snapshot().balance.absolute_volume_error_m3,
        "total_soil_impulse_terrain_ns": total_soil_impulse.tolist(),
        "total_abs_mechanical_work_j": total_mechanical_work_j,
        "maximum_visual_lag_s": maximum_lag_s,
        "collision_evidence": collision_evidence,
        "terrain_reset_between_cycles": False,
        "vehicle_reset_between_cycles": False,
        "payload_reset_between_cycles": False,
        "drive_target_velocity_800_removed": True,
        "position_drives_disabled_during_cycles": False,
        "cycle_actuation": "FORCE_CAPPED_VELOCITY_ACCELERATION_SHAPED_FORCE_TYPE_POSITION_DRIVE",
        "initial_measured_hold_effort_nm": initial_hold_effort.tolist(),
        "shared_positive_power_limit_w": actuator_config.shared_positive_power_limit_w,
        "teleport_used": False,
        "deterministic_seed": deterministic_seed,
        "stochastic_model_terms_enabled": False,
    }
    (case_dir / "cycle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (case_dir / "runtime_profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    if ARGS.render:
        import omni.replicator.core as rep

        rep.orchestrator.wait_until_complete()
    failure_path = (
        REPOSITORY_ROOT
        / "outputs"
        / "integrated_390f_v1"
        / f"case_{ARGS.case}_failure.json"
    )
    failure_path.unlink(missing_ok=True)
    print(json.dumps({"status": "PASS", "case": ARGS.case, "output": str(case_dir)}), flush=True)


failure_path = REPOSITORY_ROOT / "outputs" / "integrated_390f_v1" / f"case_{ARGS.case}_failure.json"
try:
    main()
except BaseException:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path.write_text(json.dumps({"status": "FAIL", "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
    raise
finally:
    simulation_app.close()
