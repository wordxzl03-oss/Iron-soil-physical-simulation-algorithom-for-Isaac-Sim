"""Audit the real 390F USD and run an observation-gated no-soil motion test.

This script intentionally does not create track joints, write a root pose, or
pretend that an arm-only asset completed travel.  Missing track actuation is a
hard machine-acceptance blocker recorded in the output JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--config", default="configs/390f_v2_interactive.yaml")
PARSER.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
PARSER.add_argument("--phase-timeout-s", type=float, default=20.0)
ARGS, _UNKNOWN = PARSER.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless, "multi_gpu": False, "width": 960, "height": 540})

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _json_value(value):
    if value is None:
        return None
    if hasattr(value, "pathString"):
        return value.pathString
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    try:
        array = np.asarray(value)
        if array.ndim:
            converted = array.tolist()
            return converted if not array.dtype.hasobject else str(value)
        converted = array.item()
        if isinstance(converted, (str, int, float, bool)) or converted is None:
            return converted
        return str(converted)
    except (TypeError, ValueError):
        return str(value)


def _attr(prim, name):
    attribute = prim.GetAttribute(name)
    return _json_value(attribute.Get()) if attribute and attribute.IsValid() else None


def _relationship(prim, name):
    relationship = prim.GetRelationship(name)
    if not relationship or not relationship.IsValid():
        return []
    return [target.pathString for target in relationship.GetTargets()]


def _world_pose_m(prim, meters_per_unit, UsdGeom):
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=np.float64).T
    matrix[:3, 3] *= meters_per_unit
    return matrix


def main() -> None:
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

    from isaac_bulk_pipeline.runtime import Interactive390FConfig
    from isaac_bulk_pipeline.tools import ToolDescriptorLoader
    from isaac_bulk_pipeline.vehicle import (
        DifferentialTrackDriveConfig,
        DifferentialTrackDriveModel,
        IsaacDifferentialTrackDriveAdapter,
    )
    from isaac_bulk_pipeline.visualization import LightingManager

    config = Interactive390FConfig.load(ROOT / ARGS.config)
    output_dir = ROOT / "outputs" / "390f_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "390f_kinematic_geometry_audit.json"
    acceptance_path = output_dir / "no_soil_machine_acceptance.json"

    world = World(stage_units_in_meters=1.0, physics_dt=config.physics_dt_s, rendering_dt=config.physics_dt_s)
    add_reference_to_stage(str(config.vehicle_asset), "/World/Excavator")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    ground_path = "/World/LocalSupportGround"
    ground = UsdGeom.Cube.Define(stage, ground_path)
    ground.CreateSizeAttr(2.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.10))
    ground.AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.10))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground.CreateDisplayColorAttr([(0.17, 0.19, 0.21)])
    normal_contact_material = UsdShade.Material.Define(
        stage, "/World/Materials/TrackActuatorNormalContactOnly"
    )
    normal_contact_physics = UsdPhysics.MaterialAPI.Apply(
        normal_contact_material.GetPrim()
    )
    normal_contact_physics.CreateStaticFrictionAttr(0.0)
    normal_contact_physics.CreateDynamicFrictionAttr(0.0)
    normal_contact_physics.CreateRestitutionAttr(0.0)
    for path in (ground_path, config.left_track_body, config.right_track_body):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
            normal_contact_material,
            UsdShade.Tokens.strongerThanDescendants,
            "physics",
        )
    LightingManager("outdoor_day").author_usd(stage, "/World/AuditLighting")
    support_group = UsdPhysics.CollisionGroup.Define(stage, "/World/Audit_390F_CollisionGroups/TerrainSupport")
    support_collection = Usd.CollectionAPI.Apply(support_group.GetPrim(), "colliders")
    support_collection.CreateIncludesRel().SetTargets([Sdf.Path(ground_path)])
    bucket_group = UsdPhysics.CollisionGroup.Define(stage, "/World/Audit_390F_CollisionGroups/BucketInteraction")
    bucket_collection = Usd.CollectionAPI.Apply(bucket_group.GetPrim(), "colliders")
    bucket_collection.CreateIncludesRel().SetTargets([Sdf.Path(config.bucket_link)])
    support_group.CreateFilteredGroupsRel().AddTarget(bucket_group.GetPath())
    bucket_group.CreateFilteredGroupsRel().AddTarget(support_group.GetPath())

    root = config.articulation_root
    track_bounds_before_placement = []
    bounds_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    for path in (config.left_track_body, config.right_track_body):
        aligned = bounds_cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        track_bounds_before_placement.append({
            "path": path,
            "minimum_world": list(aligned.GetMin()),
            "maximum_world": list(aligned.GetMax()),
        })
    minimum_track_z = min(item["minimum_world"][2] for item in track_bounds_before_placement)
    initial_placement_z_m = 0.0
    if config.auto_align_track_bottom_to_ground:
        initial_placement_z_m = float(config.track_ground_clearance_m - minimum_track_z)
        prefix_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Excavator"))
        prefix_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, initial_placement_z_m))
    joint_names = ("swing_joint", "boom_joint", "stick_joint", "bucket_joint")
    joints = []
    authored_drive_target_velocity_removed = []
    for name in joint_names:
        prim = stage.GetPrimAtPath(f"{root}/Joints/{name}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        drive_data = {
            "type": _attr(prim, "drive:angular:physics:type"),
            "stiffness": _attr(prim, "drive:angular:physics:stiffness"),
            "damping": _attr(prim, "drive:angular:physics:damping"),
            "max_force": _attr(prim, "drive:angular:physics:maxForce"),
            "target_position_deg": _attr(prim, "drive:angular:physics:targetPosition"),
            "target_velocity_deg_s": _attr(prim, "drive:angular:physics:targetVelocity"),
        }
        old_velocity = drive.GetTargetVelocityAttr().Get()
        drive.GetTargetVelocityAttr().Set(0.0)
        authored_drive_target_velocity_removed.append({"joint": name, "authored_deg_s": _json_value(old_velocity)})
        joints.append(
            {
                "name": name,
                "prim_path": prim.GetPath().pathString,
                "prim_type": prim.GetTypeName(),
                "axis": _attr(prim, "physics:axis"),
                "lower_limit_deg": _attr(prim, "physics:lowerLimit"),
                "upper_limit_deg": _attr(prim, "physics:upperLimit"),
                "body0": _relationship(prim, "physics:body0"),
                "body1": _relationship(prim, "physics:body1"),
                "local_pos0": _attr(prim, "physics:localPos0"),
                "local_pos1": _attr(prim, "physics:localPos1"),
                "local_rot0": _attr(prim, "physics:localRot0"),
                "local_rot1": _attr(prim, "physics:localRot1"),
                "drive": drive_data,
            }
        )

    track_prims = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if "track" in prim.GetName().lower()
    ]
    track_joint_prims = [
        {"path": prim.GetPath().pathString, "type": prim.GetTypeName()}
        for prim in stage.Traverse()
        if "track" in prim.GetPath().pathString.lower()
        and prim.IsA(UsdPhysics.Joint)
    ]
    track_actuator_prims = [
        item for item in track_joint_prims
        if item["type"] in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
    ]
    lower_body_path = f"{root}/tn__390F_DETAIL_DELIVERY_Lower_Body1_xl0"
    lower_prim = stage.GetPrimAtPath(lower_body_path)
    lower_pose_before_physics = _world_pose_m(lower_prim, 1.0, UsdGeom)
    axis_u, _, axis_vh = np.linalg.svd(lower_pose_before_physics[:3, :3])
    lower_rigid_before_physics = axis_u @ axis_vh
    forward_axis_index = int(np.argmax(np.abs(lower_rigid_before_physics[0, :])))
    forward_axis_sign = float(np.sign(lower_rigid_before_physics[0, forward_axis_index]))
    derived_forward_axis_local = np.zeros(3)
    derived_forward_axis_local[forward_axis_index] = forward_axis_sign
    lower_body_physics_attributes = {
        attribute.GetName(): _json_value(attribute.Get())
        for attribute in lower_prim.GetAttributes()
        if "physics" in attribute.GetName().lower() or "physx" in attribute.GetName().lower()
    }
    articulation_joint_inventory = []
    articulation_root_api_prims_before_override = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    articulation_root_physics_attributes = {
        attribute.GetName(): _json_value(attribute.Get())
        for attribute in stage.GetPrimAtPath(root).GetAttributes()
        if "physics" in attribute.GetName().lower() or "physx" in attribute.GetName().lower()
    }
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and prim.GetPath().HasPrefix(root):
            articulation_joint_inventory.append({
                "path": prim.GetPath().pathString,
                "type": prim.GetTypeName(),
                "body0": _relationship(prim, "physics:body0"),
                "body1": _relationship(prim, "physics:body1"),
                "enabled": _attr(prim, "physics:jointEnabled"),
            })
    world_anchor_runtime_disabled = False
    if config.mobile_base_enabled:
        anchor = UsdPhysics.Joint(stage.GetPrimAtPath(config.world_anchor_joint))
        if not anchor or not anchor.GetPrim().IsValid():
            raise RuntimeError(f"[390FAudit] missing configured world anchor: {config.world_anchor_joint}")
        anchor.GetJointEnabledAttr().Set(False)
        # Referenced fixed joints may still be instantiated by PhysX even with
        # jointEnabled=false.  Author an inactive opinion in the composed
        # runtime stage; the referenced source USD on disk is untouched.
        anchor.GetPrim().SetActive(False)
        world_anchor_runtime_disabled = True
    world_anchor_valid_after_override = stage.GetPrimAtPath(config.world_anchor_joint).IsValid()

    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(config.bucket_descriptor)
    )
    geometry = descriptor.bucket_geometry
    tool_geometry = {
        "descriptor_source": descriptor.geometry_source,
        "descriptor_quality": descriptor.geometry_quality,
        "tool_to_link_matrix": descriptor.tool_to_link_matrix.tolist(),
        "cutting_edge_local_m": descriptor.cutting_edge_local.tolist(),
        "bottom_profile_local_m": descriptor.bottom_profile_local.tolist(),
        "bucket_capacity_m3": descriptor.effective_capacity_m3,
        "mouth_polygon_local_m": None if geometry is None else geometry.mouth_polygon_local.tolist(),
        "bottom_plate_normal_local": None if geometry is None else geometry.bottom_plate_normal_local.tolist(),
        "separation_plane_direction_local": None if geometry is None else geometry.separation_plane_direction_local.tolist(),
    }

    excavator = world.scene.add(SingleArticulation(root, name="audit_390f"))
    left_track = world.scene.add(RigidPrim(config.left_track_body, name="audit_left_track", reset_xform_properties=False))
    right_track = world.scene.add(RigidPrim(config.right_track_body, name="audit_right_track", reset_xform_properties=False))
    lower_body = world.scene.add(RigidPrim(config.lower_body, name="audit_lower_body", reset_xform_properties=False))
    # The authored initial pose places the audited cutting edge below the
    # support plane.  Use the only already verified collision-clear pose as a
    # scene initial condition; this is set before runtime stepping and is not a
    # root-pose motion command.
    excavator.set_joints_default_state(positions=np.asarray(config.phase_targets_rad["penetrate"], dtype=np.float64))
    world.reset()
    dof_names = tuple(str(value) for value in excavator.dof_names)
    initial_q = np.asarray(excavator.get_joint_positions(), dtype=np.float64).reshape(-1)
    controller = excavator.get_articulation_controller()
    track_drive = IsaacDifferentialTrackDriveAdapter(
        left_track,
        right_track,
        lower_body,
        DifferentialTrackDriveModel(
            DifferentialTrackDriveConfig(
                maximum_tractive_force_per_track_n=float(config.controller["maximum_tractive_force_per_track_n"]),
                maximum_braking_force_per_track_n=float(config.controller["maximum_braking_force_per_track_n"]),
                nominal_track_speed_m_s=config.nominal_track_belt_speed_m_s,
                full_force_speed_error_m_s=config.nominal_track_belt_speed_m_s,
                forward_axis_lower_body_local=tuple(derived_forward_axis_local.tolist()),
            )
        ),
    )
    # Track force acceptance runs before deliberately probing the known-bad
    # historical arm targets.  Otherwise bucket/ground collisions from those
    # targets contaminate the mobile-base measurement.
    hold_q = np.asarray(config.phase_targets_rad["penetrate"], dtype=np.float64)
    for _ in range(int(np.ceil(3.0 / config.physics_dt_s))):
        controller.apply_action(ArticulationAction(joint_positions=hold_q))
        world.step(render=not ARGS.headless)
    track_drive.reset()
    start_position = np.asarray(lower_body.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
    start_quaternion = np.asarray(lower_body.get_world_poses()[1], dtype=np.float64).reshape(-1, 4)[0]
    from scipy.spatial.transform import Rotation
    start_rotation = Rotation.from_quat([start_quaternion[1], start_quaternion[2], start_quaternion[3], start_quaternion[0]]).as_matrix()
    start_forward = start_rotation[:, 0]
    start_forward = start_rotation @ derived_forward_axis_local
    start_forward[2] = 0.0
    start_forward /= np.linalg.norm(start_forward)

    def current_track_forward() -> np.ndarray:
        quaternion = np.asarray(lower_body.get_world_poses()[1], dtype=np.float64).reshape(-1, 4)[0]
        rotation = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
        forward = rotation @ derived_forward_axis_local
        forward[2] = 0.0
        return forward / np.linalg.norm(forward)

    def heading_correction() -> float:
        forward = current_track_forward()
        cross_z = float(forward[0] * start_forward[1] - forward[1] * start_forward[0])
        angular = np.asarray(lower_body.get_angular_velocities(), dtype=np.float64).reshape(-1, 3)[0]
        return float(np.clip(1.5 * cross_z - 0.35 * angular[2], -0.35, 0.35))
    reverse_steps = 0
    reverse_position = start_position.copy()
    reverse_speed = 0.0
    for _ in range(int(np.ceil(8.0 / config.physics_dt_s))):
        controller.apply_action(ArticulationAction(joint_positions=hold_q))
        position = np.asarray(lower_body.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
        velocity = np.asarray(lower_body.get_linear_velocities(), dtype=np.float64).reshape(-1, 3)[0]
        signed = float(np.dot(position[:2] - start_position[:2], start_forward[:2]))
        reverse_speed = float(np.dot(velocity[:2], start_forward[:2]))
        # Match the formal state machine: apply a bounded reverse request until
        # measured displacement reaches the target.  A position-PD command
        # previously self-braked before 2 m and made the actuator audit test a
        # different controller from the actual runtime.
        command = -0.65
        correction = heading_correction()
        track_drive.apply(
            float(np.clip(command - correction, -0.9, 0.9)),
            float(np.clip(command + correction, -0.9, 0.9)),
            config.physics_dt_s, left_contact_active=True, right_contact_active=True,
        )
        world.step(render=not ARGS.headless)
        reverse_steps += 1
        reverse_position = np.asarray(lower_body.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
        signed = float(np.dot(reverse_position[:2] - start_position[:2], start_forward[:2]))
        if signed <= -2.0:
            break
    reverse_displacement = abs(float(np.dot(reverse_position[:2] - start_position[:2], start_forward[:2])))
    return_steps = 0
    return_position = reverse_position.copy()
    return_speed = 0.0
    for _ in range(int(np.ceil(12.0 / config.physics_dt_s))):
        controller.apply_action(ArticulationAction(joint_positions=hold_q))
        position = np.asarray(lower_body.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
        velocity = np.asarray(lower_body.get_linear_velocities(), dtype=np.float64).reshape(-1, 3)[0]
        signed = float(np.dot(position[:2] - start_position[:2], start_forward[:2]))
        return_speed = float(np.dot(velocity[:2], start_forward[:2]))
        position_error = -signed
        command = float(np.clip(0.45 * position_error - 0.8 * return_speed, -0.75, 0.75))
        if abs(position_error) > 0.15 and abs(command) < 0.65:
            command = float(np.sign(position_error) * 0.65)
        correction = heading_correction()
        track_drive.apply(
            float(np.clip(command - correction, -0.9, 0.9)),
            float(np.clip(command + correction, -0.9, 0.9)),
            config.physics_dt_s, left_contact_active=True, right_contact_active=True,
        )
        world.step(render=not ARGS.headless)
        return_steps += 1
        return_position = np.asarray(lower_body.get_world_poses()[0], dtype=np.float64).reshape(-1, 3)[0]
        signed = float(np.dot(return_position[:2] - start_position[:2], start_forward[:2]))
        if return_steps > 30 and abs(signed) <= 0.15 and abs(return_speed) <= 0.15:
            break
    return_error = float(np.linalg.norm(return_position[:2] - start_position[:2]))
    vertical_excursion = float(max(abs(reverse_position[2] - start_position[2]), abs(return_position[2] - start_position[2])))
    reverse_travel_pass = bool(reverse_displacement >= 2.0)
    return_travel_pass = bool(return_error <= 0.5)
    track_force_pass = bool(
        reverse_travel_pass
        and return_travel_pass
        and vertical_excursion <= 0.5
    )
    track_force_acceptance = {
        "actuation": "BOUNDED_FORCE_AT_LEFT_RIGHT_TRACK_RIGID_BODIES",
        "provenance": track_drive.model.config.provenance,
        "root_pose_write_count": 0,
        "start_lower_body_world_m": start_position.tolist(),
        "start_forward_world": start_forward.tolist(),
        "reverse_lower_body_world_m": reverse_position.tolist(),
        "return_lower_body_world_m": return_position.tolist(),
        "reverse_displacement_m": reverse_displacement,
        "return_position_error_m": return_error,
        "maximum_endpoint_vertical_excursion_m": vertical_excursion,
        "reverse_steps": reverse_steps,
        "reverse_endpoint_speed_m_s": reverse_speed,
        "return_steps": return_steps,
        "status": "PASS" if track_force_pass else "FAIL",
    }

    phase_names = (
        "initial_pose", "penetrate", "breakout", "lift",
        "upper_body_swing", "dump_spill", "next_dig_ready",
    )
    dynamic_phases = []
    arm_pass = dof_names == joint_names
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    for phase_name in phase_names:
        target = np.asarray(config.phase_targets_rad[phase_name], dtype=np.float64)
        start_time = float(world.current_time)
        max_error = float("inf")
        finite = True
        steps = 0
        while float(world.current_time) - start_time < ARGS.phase_timeout_s:
            controller.apply_action(ArticulationAction(joint_positions=target))
            world.step(render=not ARGS.headless)
            q = np.asarray(excavator.get_joint_positions(), dtype=np.float64).reshape(-1)
            qd = np.asarray(excavator.get_joint_velocities(), dtype=np.float64).reshape(-1)
            finite = bool(np.all(np.isfinite(q)) and np.all(np.isfinite(qd)))
            steps += 1
            if not finite:
                break
            max_error = float(np.max(np.abs(q - target)))
            if max_error <= np.deg2rad(3.0) and float(np.max(np.abs(qd))) <= np.deg2rad(4.0):
                break
        final_q = np.asarray(excavator.get_joint_positions(), dtype=np.float64).reshape(-1)
        bucket_pose = _world_pose_m(stage.GetPrimAtPath(config.bucket_link), meters_per_unit, UsdGeom)
        passed = bool(finite and max_error <= np.deg2rad(3.0))
        arm_pass = arm_pass and passed
        dynamic_phases.append(
            {
                "phase": phase_name,
                "target_rad": target.tolist(),
                "final_rad": final_q.tolist(),
                "max_abs_error_rad": max_error,
                "elapsed_sim_s": float(world.current_time) - start_time,
                "physics_steps": steps,
                "bucket_link_pose_world_m": bucket_pose.tolist(),
                "status": "PASS" if passed else "FAIL",
            }
        )

    authored_limits = {item["name"]: (item["lower_limit_deg"], item["upper_limit_deg"]) for item in joints}
    target_limit_checks = {}
    for phase, target_rad in config.phase_targets_rad.items():
        checks = []
        for index, name in enumerate(joint_names):
            lower, upper = authored_limits[name]
            degrees = float(np.rad2deg(target_rad[index]))
            checks.append({
                "joint": name,
                "target_deg": degrees,
                "within_authored_limit": bool(lower is not None and upper is not None and lower <= degrees <= upper),
            })
        target_limit_checks[phase] = checks

    audit = {
        "schema": "390f-kinematic-geometry-audit/v2",
        "asset": str(config.vehicle_asset),
        "articulation_root": root,
        "stage_meters_per_unit": meters_per_unit,
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "support_ground_source": "LOCAL_USD_CUBE_NO_NUCLEUS_DEPENDENCY",
        "initial_scene_placement": {
            "method": "TRACK_CAD_BOUND_TO_GROUND_BEFORE_PHYSICS_INITIALIZATION",
            "track_bounds_before_placement": track_bounds_before_placement,
            "translation_world_z_m": initial_placement_z_m,
            "runtime_root_pose_writes": 0,
        },
        "bucket_support_collision_filter": {
            "ground": ground_path,
            "bucket": config.bucket_link,
            "filtered": True,
            "reason": "custom soil owns bucket/terrain interaction; ground remains track support",
        },
        "dof_names": list(dof_names),
        "expected_arm_dofs": list(joint_names),
        "initial_joint_position_rad": initial_q.tolist(),
        "joint_metadata": joints,
        "phase_target_limit_checks": target_limit_checks,
        "authored_drive_target_velocity_removed": authored_drive_target_velocity_removed,
        "track_named_prim_count": len(track_prims),
        "track_named_joint_prims": track_joint_prims,
        "track_actuator_prims": track_actuator_prims,
        "lower_body_physics_attributes": lower_body_physics_attributes,
        "derived_track_forward_axis_lower_body_local": derived_forward_axis_local.tolist(),
        "articulation_joint_inventory": articulation_joint_inventory,
        "articulation_root_api_prims_before_override": articulation_root_api_prims_before_override,
        "articulation_root_physics_attributes": articulation_root_physics_attributes,
        "mobile_base_runtime_override": {
            "configured": config.mobile_base_enabled,
            "world_anchor_joint": config.world_anchor_joint,
            "joint_disabled_for_runtime": world_anchor_runtime_disabled,
            "composed_runtime_prim_deactivated": world_anchor_runtime_disabled,
            "source_usd_modified": False,
            "world_anchor_prim_valid_after_override": world_anchor_valid_after_override,
            "root_pose_write_count": 0,
        },
        "track_actuator_audit": {
            "status": "FAIL" if not track_actuator_prims else "REQUIRES_DYNAMIC_VALIDATION",
            "reason": "USD track bodies are attached by fixed joints; no left/right track drive DOF exists" if not track_actuator_prims else "track joints require mapping",
            "root_pose_write_allowed": False,
            "permitted_remediation": "apply bounded left/right tractive forces at physical track contact bodies or author real drive joints",
            "contact_domain_ownership": {
                "normal_support": "PHYSX",
                "tangential_shear": "DIFFERENTIAL_TRACK_FORCE_ACTUATOR",
                "physx_coulomb_friction": 0.0,
                "reason": "prevent duplicate tangential resistance in the reduced-order drive acceptance",
            },
        },
        "track_force_actuator_dynamic_acceptance": track_force_acceptance,
        "tool_geometry": tool_geometry,
        "rake_definition_audit": {
            "runtime_definition": "angle of CAD-derived bottom-plate penetration/separation tangent above horizontal projected along instantaneous cutting direction",
            "literature_definition": "Luengo/Reece blade rake is tied to the effective cutting blade/separation face orientation",
            "equivalence_status": "NOT_PROVEN_EQUIVALENT_FOR_CURVED_390F_BUCKET",
            "force_model_gate": "OUT-OF-DOMAIN CASES MUST NOT BE FIXED BY EXPANDING ANGLE LIMITS; requires effective cutting-face validation or excavator-specific model",
        },
        "dynamic_arm_phases": dynamic_phases,
        "kinematic_status": "PASS" if arm_pass else "FAIL",
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    tracks_available = track_force_pass
    acceptance = {
        "schema": "390f-no-soil-machine-acceptance/v2",
        "soil_force_mode": "NO_SOIL_FORCE",
        "root_pose_writes": 0,
        "arm_sequence_status": "PASS" if arm_pass else "FAIL",
        "asset_track_drive_dof_status": "MISSING_FIXED_JOINTS_ONLY" if not track_actuator_prims else "PRESENT",
        "real_left_right_track_actuation_status": "PASS" if tracks_available else "FAIL",
        "reverse_travel_status": "PASS" if reverse_travel_pass else "FAIL",
        "return_travel_status": "PASS" if return_travel_pass else "FAIL",
        "full_physics_run_gate": "PASS" if arm_pass and tracks_available else "BLOCKED",
        "overall_status": "PASS" if arm_pass and tracks_available else "FAIL",
        "blocking_failures": ([] if tracks_available else ["TRACK_FORCE_ACTUATOR_DYNAMIC_ACCEPTANCE_FAILED"]) + ([] if arm_pass else ["ARM_TARGET_TRACKING_FAILED"]),
        "kinematic_audit": str(audit_path),
    }
    acceptance_path.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": acceptance["overall_status"], "audit": str(audit_path), "acceptance": str(acceptance_path)}), flush=True)


try:
    main()
except BaseException:
    failure = ROOT / "outputs" / "390f_v2" / "no_soil_machine_acceptance_runtime_failure.json"
    failure.parent.mkdir(parents=True, exist_ok=True)
    failure.write_text(json.dumps({"status": "FAIL", "traceback": traceback.format_exc()}, indent=2) + "\n", encoding="utf-8")
    raise
finally:
    simulation_app.close()
