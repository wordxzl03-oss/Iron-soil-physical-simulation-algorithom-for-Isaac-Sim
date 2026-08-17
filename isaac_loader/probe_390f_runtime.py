"""Headless public-API probe for the real 390F articulation."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

from isaacsim import SimulationApp


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ASSET = Path("/home/eric/桌面/bulldozer_sim/bulldozer_main.usd")
OUTPUT = REPOSITORY_ROOT / "outputs" / "real_390f_runtime_probe.json"


simulation_app = SimulationApp({"headless": True})


def _jsonify(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    import numpy as np
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction
    import omni.usd
    from pxr import UsdPhysics

    from isaac_bulk_pipeline.vehicle import ExcavatorActuatorConfig, ExcavatorActuatorModel

    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    add_reference_to_stage(str(ASSET), "/World/Excavator")
    world.scene.add_default_ground_plane(z_position=0.0)
    root = (
        "/World/Excavator/_90F_LME_ISAAC_DETAIL_DELIVERY/"
        "tn__390F_LME_ISAAC_DETAIL_DELIVERY_/tn__aaa0ab9"
    )
    stage = omni.usd.get_context().get_stage()
    actuator_config = ExcavatorActuatorConfig.cat_390f_l_mass_configuration()
    limit_by_name = {item.joint_name: item for item in actuator_config.joints}
    for joint_name in ("swing_joint", "boom_joint", "stick_joint", "bucket_joint"):
        joint_prim = stage.GetPrimAtPath(f"{root}/Joints/{joint_name}")
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        drive.GetTargetVelocityAttr().Set(0.0)
        drive.GetMaxForceAttr().Set(limit_by_name[joint_name].effort_limit_nm)
    excavator = world.scene.add(SingleArticulation(root, name="real_390f_probe"))
    bodies = world.scene.add(
        RigidPrim(
            prim_paths_expr=f"{root}/tn__390F_DETAIL_DELIVERY_.*",
            name="real_390f_body_probe",
            reset_xform_properties=False,
        )
    )
    world.reset()
    names = tuple(str(item) for item in excavator.dof_names)
    initial = np.asarray(excavator.get_joint_positions(), dtype=np.float64).reshape(-1)
    target = initial.copy()
    for name, delta_deg in {
        "swing_joint": 2.0,
        "boom_joint": 2.0,
        "stick_joint": -2.0,
        "bucket_joint": 3.0,
    }.items():
        if name in names:
            target[names.index(name)] += np.deg2rad(delta_deg)
    controller = excavator.get_articulation_controller()
    actuator = ExcavatorActuatorModel(actuator_config, names)
    actuator.reset(np.asarray(excavator.get_joint_velocities(), dtype=np.float64).reshape(-1))
    last_actuator = None
    drive_target = initial.copy()
    for _ in range(120):
        current_position = np.asarray(
            excavator.get_joint_positions(), dtype=np.float64
        ).reshape(-1)
        current_velocity = np.asarray(
            excavator.get_joint_velocities(), dtype=np.float64
        ).reshape(-1)
        last_actuator = actuator.step(
            target, current_position, current_velocity, 1.0 / 60.0
        )
        drive_target += last_actuator.target_velocity_rad_s * (1.0 / 60.0)
        remaining = target - drive_target
        crossed = np.sign(remaining) != np.sign(target - initial)
        drive_target[crossed] = target[crossed]
        controller.apply_action(
            ArticulationAction(joint_positions=drive_target)
        )
        world.step(render=False)
    final = np.asarray(excavator.get_joint_positions(), dtype=np.float64).reshape(-1)
    result = {
        "status": "PASS",
        "asset": str(ASSET),
        "articulation_root": root,
        "dof_names": list(names),
        "initial_joint_positions_rad": initial.tolist(),
        "target_joint_positions_rad": target.tolist(),
        "final_joint_positions_rad": final.tolist(),
        "joint_motion_norm_rad": float(np.linalg.norm(final - initial)),
        "joint_velocities_rad_s": np.asarray(
            excavator.get_joint_velocities(), dtype=np.float64
        ).reshape(-1).tolist(),
        "applied_joint_efforts": np.asarray(
            excavator.get_applied_joint_efforts(), dtype=np.float64
        ).reshape(-1).tolist(),
        "measured_joint_efforts": np.asarray(
            excavator.get_measured_joint_efforts(), dtype=np.float64
        ).reshape(-1).tolist(),
        "control_method": "ExcavatorActuatorModel command shaper + force-capped USD drives -> ArticulationController.apply_action(joint_positions)",
        "teleport_used": False,
        "drive_target_velocity_cleared": True,
        "last_effort_command_nm": last_actuator.effort_command_nm.tolist(),
        "last_shared_power_scale": last_actuator.shared_power_scale,
        "rigid_body_paths": list(bodies.prim_paths),
        "runtime_body_masses_kg": np.asarray(bodies.get_masses(), dtype=np.float64).reshape(-1).tolist(),
        "runtime_body_coms": _jsonify(bodies.get_coms()),
        "runtime_body_inertias": _jsonify(bodies.get_inertias()),
    }
    for key, method in (
        ("runtime_masses_kg", "get_masses"),
        ("runtime_coms", "get_coms"),
        ("runtime_inertias", "get_inertias"),
    ):
        try:
            result[key] = np.asarray(getattr(excavator, method)(), dtype=np.float64).tolist()
        except Exception as error:
            result[key] = None
            result[f"{key}_error"] = f"{type(error).__name__}: {error}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


try:
    main()
except BaseException:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps({"status": "FAIL", "traceback": traceback.format_exc()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    raise
finally:
    simulation_app.close()
