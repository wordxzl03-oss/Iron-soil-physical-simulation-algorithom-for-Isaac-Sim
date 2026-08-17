"""Render a trained policy with the exact frame/boom/bucket meshes from OBJ."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stable_baselines3 import SAC

from generate_loader_dataset import trajectory_geometry
from obj_scooping_env import ObjScoopingTrajectoryEnv
from physics_aware_trajectory import apply_planned_cut, plan_resistance_aware_dig
from render_obj_loader_scooping import rotate_yz, transform_point
from slope_model import relax_critical_slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("rl_runs/sac_obj_scooping/scooping_sac_obj_final.zip"),
    )
    parser.add_argument("--obj", type=Path, default=Path("wheel_buck.obj"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--output", type=Path, default=Path("rl_scoop_demo/obj_policy_demo.gif")
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=72)
    args = parser.parse_args()
    if not args.model.exists() or not args.obj.exists():
        raise FileNotFoundError("model or OBJ file does not exist")

    env = ObjScoopingTrajectoryEnv(obj_path=args.obj, seed=args.seed)
    observation, reset_info = env.reset(seed=args.seed)
    model = SAC.load(args.model, device="auto")
    action, _ = model.predict(observation, deterministic=True)
    trajectory = env.action_to_trajectory(action)
    entry, _start, _end, forward_plot, _toe = trajectory_geometry(
        env._initial,
        (env.spacing, env.spacing),
        trajectory,
        env.workspace_size_m,
    )
    plan = plan_resistance_aware_dig(
        env._initial,
        (env.spacing, env.spacing),
        np.array([entry[1], entry[0]]),
        forward_plot,
        trajectory,
        env._material,
        env._limits,
        count=61,
    )
    raw_scooped, raw_volume, raw_removed = apply_planned_cut(
        env._initial,
        (env.spacing, env.spacing),
        entry,
        trajectory.heading_deg,
        trajectory,
        plan,
    )
    del raw_scooped
    loaded_volume = min(raw_volume, env.bucket_capacity_m3)
    capacity_scale = loaded_volume / max(raw_volume, 1e-9)
    removed = raw_removed * capacity_scale
    scooped = env._initial - removed
    final, _ = relax_critical_slope(
        scooped,
        (env.spacing, env.spacing),
        env._material.internal_friction_deg,
    )

    chinese_font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams["font.family"] = "Microsoft YaHei"
        plt.rcParams["axes.unicode_minus"] = False

    frame_part = env.obj_parts["loader_frame_world.stl"]
    boom_part = env.obj_parts["loader_boom_world.stl"]
    bucket_part = env.obj_parts["loader_bucket_world.stl"]
    root = env.obj_points["root_pin"]
    pin = env.obj_points["bucket_pin"]
    edge = env.obj_points["cutting_edge"]
    low_angle = env.low_boom_angle_deg
    pin_low = transform_point(pin, root, low_angle)
    heading = np.deg2rad(trajectory.heading_deg)
    rotation_xy = np.array(
        [[np.cos(heading), np.sin(heading)],
         [-np.sin(heading), np.cos(heading)]]
    )
    frame_indices = np.linspace(0, len(plan.path) - 1, 30).round().astype(int)
    poses: list[dict[str, np.ndarray]] = []
    edge_trace: list[np.ndarray] = []

    def orient_and_place(
        vertices: np.ndarray, local_edge: np.ndarray, target_edge: np.ndarray
    ) -> np.ndarray:
        result = vertices.copy()
        result[:, :2] = result[:, :2] @ rotation_xy.T
        rotated_edge = local_edge.copy()
        rotated_edge[:2] = rotated_edge[:2] @ rotation_xy.T
        return result + (target_edge - rotated_edge)

    for index in frame_indices:
        lift_fraction = np.clip(
            (plan.cutting_edge_z[index] - env._limits.ground_clearance_m)
            / max(env.obj_max_edge_lift_m, 1e-6),
            0.0,
            1.0,
        )
        boom_angle = low_angle + 34.0 * lift_fraction
        curl_angle = float(plan.pitch_deg[index])
        boom = rotate_yz(boom_part.vertices, root, boom_angle)
        pin_now = transform_point(pin, root, boom_angle)
        bucket_low = rotate_yz(bucket_part.vertices, root, low_angle)
        bucket = (
            rotate_yz(bucket_low, pin_low, curl_angle)
            + (pin_now - pin_low)
        )
        edge_low = transform_point(edge, root, low_angle)
        local_edge = (
            transform_point(edge_low, pin_low, curl_angle)
            + (pin_now - pin_low)
        )
        # Planner path is [plot y, plot x, z]; world OBJ uses [soil x, soil y, z].
        target_edge = np.array(
            [
                plan.path[index, 1]
                + trajectory.travel_length * 0.44
                * np.cos(np.deg2rad(plan.pitch_deg[index]))
                * np.sin(heading),
                plan.path[index, 0]
                + trajectory.travel_length * 0.44
                * np.cos(np.deg2rad(plan.pitch_deg[index]))
                * np.cos(heading),
                plan.cutting_edge_z[index],
            ]
        )
        poses.append(
            {
                "frame": orient_and_place(frame_part.vertices, local_edge, target_edge),
                "boom": orient_and_place(boom, local_edge, target_edge),
                "bucket": orient_and_place(bucket, local_edge, target_edge),
            }
        )
        edge_trace.append(target_edge)
    edge_trace_array = np.asarray(edge_trace)

    coordinates = np.arange(env.grid_size) * env.spacing
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    fig = plt.figure(figsize=(12.4, 5.7), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    colors = {"frame": "#263238", "boom": "#f5a000", "bucket": "#e67e00"}
    parts = {"frame": frame_part, "boom": boom_part, "bucket": bucket_part}

    def terrain_at(frame: int, index: int) -> np.ndarray:
        fraction = float(plan.cut_fraction[index])
        if fraction < 1.0:
            return env._initial - removed * fraction
        slump = max(0.0, (frame / max(len(frame_indices) - 1, 1) - 0.82) / 0.18)
        return scooped * (1 - slump) + final * slump

    def draw(frame: int) -> None:
        ax3d.clear()
        ax2d.clear()
        index = int(frame_indices[frame])
        terrain = terrain_at(frame, index)
        pose = poses[frame]
        ax3d.plot_surface(
            xx, yy, terrain, cmap="terrain",
            vmin=0, vmax=max(7.0, float(env._initial.max())),
            linewidth=0, antialiased=False, alpha=0.90,
            rstride=2, cstride=2,
        )
        for key in ("frame", "boom", "bucket"):
            part = parts[key]
            stride = max(1, len(part.faces) // 700)
            ax3d.add_collection3d(
                Poly3DCollection(
                    pose[key][part.faces[::stride]],
                    facecolor=colors[key],
                    edgecolor="#202020",
                    linewidth=0.12,
                    alpha=0.97,
                )
            )
        ax3d.plot(
            edge_trace_array[:, 0], edge_trace_array[:, 1],
            edge_trace_array[:, 2], "--", color="#d32f2f", linewidth=1.5,
        )
        ax3d.set(
            xlim=(0, env.workspace_size_m),
            ylim=(0, env.workspace_size_m),
            zlim=(0, max(7.0, float(env._initial.max()) + 1.5)),
            xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
            title=f"{args.obj.name} 三维网格与强化学习轨迹",
        )
        ax3d.view_init(elev=25, azim=-58)

        target = edge_trace_array[frame, :2]
        sample_s = np.linspace(-7.0, 7.0, 260)
        forward_soil = np.array([np.sin(heading), np.cos(heading)])
        points = target[None, :] + sample_s[:, None] * forward_soil
        ii = np.clip(np.rint(points[:, 0] / env.spacing).astype(int), 0, env.grid_size - 1)
        jj = np.clip(np.rint(points[:, 1] / env.spacing).astype(int), 0, env.grid_size - 1)
        profile = terrain[ii, jj]
        ax2d.fill_between(sample_s, 0, profile, color="#9b7653", alpha=0.88)
        ax2d.plot(sample_s, profile, color="#5d4037", linewidth=1.4)
        for key in ("frame", "boom", "bucket"):
            vertices = pose[key]
            relative = vertices[:, :2] - target[None, :]
            longitudinal = relative @ forward_soil
            yz = np.column_stack((longitudinal, vertices[:, 2]))
            part = parts[key]
            stride = max(1, len(part.faces) // 500)
            ax2d.add_collection(
                PolyCollection(
                    yz[part.faces[::stride]],
                    facecolor=colors[key],
                    edgecolor="#202020",
                    linewidth=0.15,
                    alpha=0.97,
                )
            )
        ax2d.plot(
            edge_trace_array[:, :2] @ forward_soil,
            edge_trace_array[:, 2],
            "--", color="#d32f2f", linewidth=1.2,
        )
        ax2d.set(
            xlim=(-7, 7),
            ylim=(0, max(7.0, float(env._initial.max()) + 1.5)),
            xlabel="相对刃口的进铲方向 [m]", ylabel="高度 z [m]",
            title=(
                f"OBJ 侧视 | 进度={100 * plan.cut_fraction[index]:.0f}% | "
                f"阻力={plan.resistance_n[index] / 1000:.0f} kN"
            ),
            aspect="equal",
        )
        ax2d.grid(alpha=0.2)
        fig.suptitle(
            f"OBJ 约束 SAC 策略 | 装载={loaded_volume:.2f}/{env.bucket_capacity_m3:.2f} m³ "
            f"| heading={trajectory.heading_deg:+.1f}°"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, draw, frames=len(frame_indices), blit=False)
    animation.save(args.output, PillowWriter(fps=args.fps), dpi=args.dpi)
    plt.close(fig)

    summary = {
        "model": str(args.model),
        "obj_path": str(args.obj.resolve()),
        "obj_source_parts": list(env.obj_source_part_names),
        "seed": args.seed,
        "device": str(model.device),
        "animation": str(args.output),
        "peak_height_m": reset_info["peak_height_m"],
        "loaded_volume_m3": float(loaded_volume),
        "raw_swept_volume_m3": float(raw_volume),
        "bucket_capacity_m3": env.bucket_capacity_m3,
        "obj_bucket_width_m": env.obj_bucket_width_m,
        "entry_xy_m": [float(value) for value in entry],
        "trajectory": asdict(trajectory),
        "action": np.asarray(action, dtype=float).tolist(),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
