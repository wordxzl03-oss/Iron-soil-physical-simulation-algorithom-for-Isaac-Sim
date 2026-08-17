"""Offline kinematic-debug scan of the real 390F articulation workspace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--config", default="configs/390f_v2_interactive.yaml")
PARSER.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
ARGS, _ = PARSER.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless, "multi_gpu": False, "width": 640, "height": 360})
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from scipy.spatial.transform import Rotation

    from isaac_bulk_pipeline.runtime import Interactive390FConfig
    from isaac_bulk_pipeline.tools import ToolDescriptorLoader

    config = Interactive390FConfig.load(ROOT / ARGS.config)
    world = World(stage_units_in_meters=1.0, physics_dt=config.physics_dt_s, rendering_dt=config.physics_dt_s)
    world.get_physics_context().set_gravity(0.0)
    add_reference_to_stage(str(config.vehicle_asset), "/World/Excavator")
    articulation = world.scene.add(SingleArticulation(config.articulation_root, name="kinematic_workspace_390f"))
    bucket = world.scene.add(RigidPrim(config.bucket_link, name="kinematic_workspace_bucket", reset_xform_properties=False))
    world.reset()
    descriptor = ToolDescriptorLoader.load(ToolDescriptorLoader.load_config(config.bucket_descriptor))
    geometry = descriptor.bucket_geometry
    if geometry is None:
        raise RuntimeError("[390FWorkspace] real bucket geometry is required")

    boom_values = np.deg2rad(np.linspace(-25.0, 45.0, 15))
    stick_values = np.deg2rad(np.linspace(-100.0, 45.0, 16))
    bucket_values = np.deg2rad(np.linspace(-120.0, 60.0, 10))
    records = []
    for boom in boom_values:
        for stick in stick_values:
            for bucket_angle in bucket_values:
                q = np.asarray([0.0, boom, stick, bucket_angle], dtype=np.float64)
                articulation.set_joint_positions(q)
                articulation.set_joint_velocities(np.zeros(4))
                position, quaternion_wxyz = bucket.get_world_poses()
                position = np.asarray(position, dtype=np.float64).reshape(-1, 3)[0]
                quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1, 4)[0]
                rigid = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
                link = np.eye(4)
                link[:3, :3] = rigid * 0.001
                link[:3, 3] = position
                world_from_tool = link @ descriptor.tool_to_link_matrix
                lip = world_from_tool[:3, 3]
                separation = world_from_tool[:3, :3] @ geometry.separation_plane_direction_local
                separation /= np.linalg.norm(separation)
                mouth_normal = world_from_tool[:3, :3] @ geometry.mouth_normal_local
                mouth_normal /= np.linalg.norm(mouth_normal)
                records.append(np.r_[q, lip, separation, mouth_normal])
    data = np.asarray(records, dtype=np.float64)
    output = ROOT / "outputs" / "390f_v2"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "390f_kinematic_workspace.npz",
        samples=data,
        columns=np.asarray([
            "swing_rad", "boom_rad", "stick_rad", "bucket_rad",
            "lip_x_m", "lip_y_m", "lip_z_m", "separation_x", "separation_y",
            "separation_z", "mouth_normal_x", "mouth_normal_y", "mouth_normal_z",
        ]),
    )

    desired = {
        "dig_ready": np.asarray([5.25, 0.0, 1.8]),
        "penetration": np.asarray([6.0, 0.0, 0.55]),
        "breakout": np.asarray([5.5, 0.0, 2.3]),
        "transport": np.asarray([4.8, 0.0, 4.5]),
        "dump_height": np.asarray([4.8, 0.0, 5.2]),
    }
    candidates = {}
    for name, target in desired.items():
        position_error = np.linalg.norm(data[:, 4:7] - target, axis=1)
        # Dig/carry states prefer the bottom-plate separation tangent pointing
        # forward (+X); dump prefers an open/downward mouth.  These are ranking
        # terms only, not literature validation.
        if name == "dump_height":
            orientation_penalty = np.maximum(data[:, 12], 0.0)
        else:
            orientation_penalty = 0.5 * (1.0 - np.clip(data[:, 7], -1.0, 1.0))
        score = position_error + orientation_penalty
        indices = np.argsort(score)[:12]
        candidates[name] = [
            {
                "joint_deg": np.rad2deg(data[index, :4]).tolist(),
                "lip_world_m": data[index, 4:7].tolist(),
                "separation_world": data[index, 7:10].tolist(),
                "mouth_normal_world": data[index, 10:13].tolist(),
                "position_error_m": float(position_error[index]),
                "ranking_score": float(score[index]),
            }
            for index in indices
        ]

    safe = np.asarray(config.phase_targets_rad["penetrate"], dtype=np.float64)
    articulation.set_joint_positions(safe)
    articulation.set_joint_velocities(np.zeros(4))
    base_position, base_quaternion = bucket.get_world_poses()
    sign_probe = []
    for joint_index, joint_name in enumerate(("swing_joint", "boom_joint", "stick_joint", "bucket_joint")):
        positions = []
        for delta_deg in (-5.0, 0.0, 5.0):
            q = safe.copy(); q[joint_index] += np.deg2rad(delta_deg)
            articulation.set_joint_positions(q); articulation.set_joint_velocities(np.zeros(4))
            position, quaternion_wxyz = bucket.get_world_poses()
            position = np.asarray(position, dtype=np.float64).reshape(-1, 3)[0]
            quaternion = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(-1, 4)[0]
            rigid = Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()
            link = np.eye(4); link[:3, :3] = rigid * .001; link[:3, 3] = position
            lip = (link @ descriptor.tool_to_link_matrix)[:3, 3]
            positions.append({"delta_deg": delta_deg, "lip_world_m": lip.tolist()})
        sign_probe.append({"joint": joint_name, "samples": positions})

    report = {
        "schema": "390f-kinematic-workspace/v1",
        "mode": "KINEMATIC_DEBUG_FIXED_BASE_ZERO_GRAVITY",
        "formal_runtime_teleport_allowed": False,
        "sample_count": int(len(data)),
        "joint_grid_deg": {
            "swing": [0.0], "boom": [-25.0, 45.0, 15],
            "stick": [-100.0, 45.0, 16], "bucket": [-120.0, 60.0, 10],
        },
        "joint_sign_probe": sign_probe,
        "ranked_pose_candidates": candidates,
        "selection_status": "CANDIDATES_ONLY_REQUIRES_DYNAMIC_COLLISION_AND_CLEARANCE_ACCEPTANCE",
    }
    report_path = output / "390f_kinematic_workspace_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "samples": len(data), "report": str(report_path)}), flush=True)


try:
    main()
finally:
    simulation_app.close()
