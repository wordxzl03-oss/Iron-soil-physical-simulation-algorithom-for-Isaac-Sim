"""Render the fixed dynamic policy with grounded simple_wheel_loader.obj."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from stable_baselines3 import SAC

from dynamic_obj_scooping_env import DynamicObjScoopingTrajectoryEnv
from render_obj_loader_scooping import (
    load_obj_parts,
    rotate_yz,
    transform_point,
)
from simulate_inertial_obj_scooping import rotate_xz
from wheel_loader_dynamics import grid_height_function


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "rl_runs/sac_dynamic_simple_loader_fixed/"
            "dynamic_simple_loader_sac.zip"
        ),
    )
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_scoop_demo/dynamic_simple_loader_fixed.gif"),
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=82)
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = "Microsoft YaHei"
        plt.rcParams["axes.unicode_minus"] = False

    env = DynamicObjScoopingTrajectoryEnv(seed=args.seed)
    observation, reset_info = env.reset(seed=args.seed)
    model = SAC.load(args.model, device="auto")
    action, _ = model.predict(observation, deterministic=True)
    _, reward, _, _, info = env.step(action)
    trajectory = env.action_to_trajectory(action)
    entry = np.asarray(info["entry_xy_m"])
    heading = np.deg2rad(trajectory.heading_deg)
    forward = np.array([np.sin(heading), np.cos(heading)])
    right = np.array([np.cos(heading), -np.sin(heading)])

    parts = load_obj_parts(Path("simple_wheel_loader.obj"))
    fixed_names = [
        "simple_chassis", "simple_cab",
        "wheel_front_left", "wheel_front_right",
        "wheel_rear_left", "wheel_rear_right",
    ]
    wheel_names = [name for name in fixed_names if name.startswith("wheel_")]
    wheel_centers = {
        name: parts[name].vertices.mean(axis=0) for name in wheel_names
    }
    boom = parts["loader_boom"]
    bucket = parts["loader_bucket"]
    root = env.obj_points["root_pin"]
    pin = env.obj_points["bucket_pin"]
    low_angle = env.low_boom_angle_deg
    pin_low = transform_point(pin, root, low_angle)
    axle_center_local_y = -2.20
    chassis_pivot = np.array([0.0, axle_center_local_y, 0.72])

    dynamic_indices = np.unique(
        np.linspace(
            0, len(env.last_vehicle_states) - 1,
            min(48, len(env.last_vehicle_states)),
        ).round().astype(int)
    )
    render_records: list[tuple] = []
    for index in dynamic_indices:
        render_records.append(
            (env.last_vehicle_states[index], low_angle, 0.0, index, "动力学接近与贯入")
        )
    last_state = env.last_vehicle_states[-1]
    last_index = len(env.last_vehicle_states) - 1
    for fraction in np.linspace(0.0, 1.0, 12)[1:]:
        render_records.append(
            (
                last_state, low_angle,
                trajectory.curl_angle_deg * fraction,
                last_index, "停车后卷斗",
            )
        )
    for fraction in np.linspace(0.0, 1.0, 14)[1:]:
        render_records.append(
            (
                last_state, low_angle + 34.0 * fraction,
                trajectory.curl_angle_deg,
                last_index, "卷斗完成后举升",
            )
        )

    colors = {
        "simple_chassis": "#f5a000", "simple_cab": "#f9c74f",
        "wheel_front_left": "#202020", "wheel_front_right": "#202020",
        "wheel_rear_left": "#202020", "wheel_rear_right": "#202020",
        "loader_boom": "#f5a000", "loader_bucket": "#e67e00",
    }

    def pose_parts(vehicle_state, boom_angle: float, curl_angle: float):
        posed: dict[str, np.ndarray] = {}
        for name in fixed_names:
            vertices = parts[name].vertices
            if name in wheel_names:
                vertices = rotate_yz(
                    vertices,
                    wheel_centers[name],
                    -np.rad2deg(vehicle_state.distance_m / 0.72),
                )
            posed[name] = vertices
        posed["loader_boom"] = rotate_yz(boom.vertices, root, boom_angle)
        pin_now = transform_point(pin, root, boom_angle)
        bucket_low = rotate_yz(bucket.vertices, root, low_angle)
        posed["loader_bucket"] = (
            rotate_yz(bucket_low, pin_low, curl_angle) + (pin_now - pin_low)
        )
        translation = np.array(
            [
                vehicle_state.x_m,
                vehicle_state.y_m - axle_center_local_y,
                vehicle_state.z_m - 0.72,
            ]
        )
        for name, vertices in posed.items():
            vertices = rotate_yz(
                vertices, chassis_pivot, np.rad2deg(vehicle_state.pitch_rad)
            )
            vertices = rotate_xz(
                vertices, chassis_pivot, np.rad2deg(vehicle_state.roll_rad)
            )
            posed[name] = vertices + translation
        return posed

    def sample_profile(terrain: np.ndarray):
        local_y = np.linspace(-10.0, 7.0, 320)
        points = entry[None, :] + local_y[:, None] * forward
        ii = np.clip(
            np.rint(points[:, 0] / env.spacing).astype(int), 0, env.grid_size - 1
        )
        jj = np.clip(
            np.rint(points[:, 1] / env.spacing).astype(int), 0, env.grid_size - 1
        )
        return local_y, terrain[ii, jj]

    reached = max(info["dynamic_penetration_m"], 1e-9)
    poses: list[dict[str, np.ndarray]] = []
    terrains: list[np.ndarray] = []
    min_clearances: list[float] = []
    for vehicle_state, boom_angle, curl_angle, row, _phase in render_records:
        penetration = float(env.last_dynamics[row, 4])
        fraction = np.clip(penetration / reached, 0.0, 1.0)
        terrain = env._initial - env.last_removed * fraction
        pose = pose_parts(vehicle_state, boom_angle, curl_angle)
        poses.append(pose)
        terrains.append(terrain)
        # Check actual OBJ wheel bottoms against the same local terrain field.
        clearances = []
        for name in wheel_names:
            vertices = pose[name]
            bottom_index = int(np.argmin(vertices[:, 2]))
            point = vertices[bottom_index]
            global_xy = entry + point[0] * right + point[1] * forward
            sampler = grid_height_function(
                terrain, (0.0, 0.0), (env.spacing, env.spacing)
            )
            ground = sampler(float(global_xy[0]), float(global_xy[1]))
            clearances.append(float(point[2] - ground))
        min_clearances.append(min(clearances))

    fig, (ax_top, ax_side) = plt.subplots(
        1, 2, figsize=(12.8, 5.6), constrained_layout=True
    )
    coordinates = np.arange(env.grid_size) * env.spacing

    def draw(frame: int) -> None:
        ax_top.clear()
        ax_side.clear()
        vehicle_state, _boom, _curl, row, phase = render_records[frame]
        terrain = terrains[frame]
        pose = poses[frame]
        ax_top.imshow(
            terrain,
            origin="lower",
            extent=(0, env.workspace_size_m, 0, env.workspace_size_m),
            cmap="terrain",
            vmin=0,
            vmax=max(7.0, float(env._initial.max())),
        )
        centers = np.asarray(
            [
                entry + state.x_m * right + state.y_m * forward
                for state in env.last_vehicle_states
            ]
        )
        ax_top.plot(centers[:, 1], centers[:, 0], "--", color="#d32f2f")
        current = entry + vehicle_state.x_m * right + vehicle_state.y_m * forward
        ax_top.scatter([current[1]], [current[0]], color="#f5a000", s=60)
        ax_top.set(
            xlim=(0, env.workspace_size_m), ylim=(0, env.workspace_size_m),
            xlabel="y [m]", ylabel="x [m]", aspect="equal",
            title="四轮接地动力学车辆中心轨迹",
        )

        local_y, profile = sample_profile(terrain)
        ax_side.fill_between(local_y, 0, profile, color="#9b7653", alpha=0.88)
        ax_side.plot(local_y, profile, color="#5d4037", linewidth=1.4)
        for name, vertices in pose.items():
            part = parts[name]
            stride = 1 if len(part.faces) < 180 else 4
            ax_side.add_collection(
                PolyCollection(
                    vertices[:, 1:3][part.faces[::stride]],
                    facecolor=colors[name],
                    edgecolor="#202020",
                    linewidth=0.15,
                    alpha=0.96,
                )
            )
        ax_side.set(
            xlim=(-10, 7),
            ylim=(0, max(7.0, float(env._initial.max()) + 1.5)),
            xlabel="局部进铲方向 [m]", ylabel="高度 z [m]", aspect="equal",
            title=(
                f"{phase} | v={vehicle_state.speed_m_s:.2f} m/s | "
                f"pitch={np.rad2deg(vehicle_state.pitch_rad):.1f}° | "
                f"最小轮地间隙={min_clearances[frame] * 1000:.1f} mm"
            ),
        )
        ax_side.grid(alpha=0.2)
        fig.suptitle(
            f"修正后的 simple_wheel_loader 动力学 SAC | "
            f"装载={info['loaded_volume_m3']:.2f} m³ | "
            f"实际贯入={info['dynamic_penetration_m']:.2f} m"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, draw, frames=len(render_records), blit=False)
    animation.save(args.output, PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)
    summary = {
        "model": str(args.model),
        "obj_path": str(Path("simple_wheel_loader.obj").resolve()),
        "seed": args.seed,
        "animation": str(args.output),
        "reward": reward,
        "peak_height_m": reset_info["peak_height_m"],
        **info,
        "minimum_rendered_wheel_clearance_m": float(min(min_clearances)),
        "wheel_penetration_detected": bool(min(min_clearances) < -1e-5),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
