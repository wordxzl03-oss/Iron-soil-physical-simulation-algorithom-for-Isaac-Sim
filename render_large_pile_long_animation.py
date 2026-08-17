"""Replay a validated excavation and render it with simple_wheel_loader.obj."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

from large_pile_excavation_env import LargePileExcavationEnv
from render_obj_loader_scooping import load_obj_parts, rotate_yz, transform_point
from simulate_inertial_obj_scooping import rotate_xz
from wheel_loader_dynamics import grid_height_function


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pile-dir", type=Path,
        default=Path("large_pile_validation_100/pile_0055"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("large_pile_validation_100/pile_0055/excavation_long.gif"),
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=76)
    parser.add_argument(
        "--frames-per-scoop",
        type=int,
        default=3,
        help="sample this many continuous vehicle states for each scoop",
    )
    args = parser.parse_args()

    summary = json.loads(
        (args.pile_dir / "summary.json").read_text(encoding="utf-8")
    )
    seed = int(summary["requested_seed"])
    target_remaining_fraction = float(
        summary.get("target_remaining_fraction", 0.20)
    )
    peak_height_range_m = tuple(
        summary.get("peak_height_range_m", [20.5, 24.0])
    )
    actions = np.asarray(
        [record["action"] for record in summary["actions"]], dtype=np.float32
    )
    env = LargePileExcavationEnv(
        seed=seed,
        target_remaining_fraction=target_remaining_fraction,
        peak_height_range_m=peak_height_range_m,
    )
    env.reset(seed=seed)
    parts = load_obj_parts(Path("simple_wheel_loader.obj"))
    fixed_names = [
        "simple_chassis", "simple_cab", "wheel_front_left",
        "wheel_front_right", "wheel_rear_left", "wheel_rear_right",
    ]
    wheel_names = [name for name in fixed_names if name.startswith("wheel_")]
    wheel_centers = {
        name: parts[name].vertices.mean(axis=0) for name in wheel_names
    }
    root, pin = env.obj_points["root_pin"], env.obj_points["bucket_pin"]
    low_angle = env.low_boom_angle_deg
    pin_low = transform_point(pin, root, low_angle)
    chassis_pivot = np.array([0.0, -2.20, 0.72])

    def pose_parts(state, boom_angle: float, curl_angle: float):
        posed = {}
        for name in fixed_names:
            vertices = parts[name].vertices
            if name in wheel_names:
                vertices = rotate_yz(
                    vertices, wheel_centers[name],
                    -np.rad2deg(state.distance_m / 0.72),
                )
            posed[name] = vertices
        posed["loader_boom"] = rotate_yz(
            parts["loader_boom"].vertices, root, boom_angle
        )
        pin_now = transform_point(pin, root, boom_angle)
        bucket_low = rotate_yz(
            parts["loader_bucket"].vertices, root, low_angle
        )
        posed["loader_bucket"] = (
            rotate_yz(bucket_low, pin_low, curl_angle) + pin_now - pin_low
        )
        translation = np.array([state.x_m, state.y_m + 2.20, state.z_m - 0.72])
        for name, vertices in posed.items():
            vertices = rotate_yz(
                vertices, chassis_pivot, np.rad2deg(state.pitch_rad)
            )
            vertices = rotate_xz(
                vertices, chassis_pivot, np.rad2deg(state.roll_rad)
            )
            posed[name] = vertices + translation
        return posed

    frames = []
    initial_volume = env.initial_volume_m3
    for scoop_index, action in enumerate(actions, start=1):
        terrain_before = env._initial.copy()
        trajectory = env.action_to_trajectory(action)
        _, _, terminated, _, info = env.step(action)
        terrain_after = env._initial.copy()
        entry = np.asarray(info["entry_xy_m"])
        angle = np.deg2rad(trajectory.heading_deg)
        forward = np.array([np.sin(angle), np.cos(angle)])
        right = np.array([np.cos(angle), -np.sin(angle)])
        state_indices = np.linspace(
            0,
            len(env.last_vehicle_states) - 1,
            args.frames_per_scoop,
        ).round().astype(int)
        for phase_index, state_index in enumerate(state_indices):
            state = env.last_vehicle_states[state_index]
            progress = phase_index / max(len(state_indices) - 1, 1)
            fraction = float(np.clip((progress - 0.58) / 0.34, 0.0, 1.0))
            fraction = fraction * fraction * (3.0 - 2.0 * fraction)
            terrain = terrain_before * (1 - fraction) + terrain_after * fraction
            curl = trajectory.curl_angle_deg * fraction
            lift = float(np.clip((progress - 0.78) / 0.22, 0.0, 1.0))
            lift = lift * lift * (3.0 - 2.0 * lift)
            boom_angle = low_angle + 24.0 * lift
            pose = pose_parts(state, boom_angle, curl)
            global_center = entry + state.x_m * right + state.y_m * forward
            sampler = grid_height_function(
                terrain, (0.0, 0.0), (env.spacing, env.spacing)
            )
            # The dynamics uses wheel-centre point contacts. On a sharp grid
            # ridge, a finite OBJ tire can otherwise intersect terrain away
            # from its centre. Lift the rigid rendered assembly to the highest
            # support required by every tire vertex.
            mesh_clearance = float("inf")
            for name in wheel_names:
                vertices = pose[name]
                for point in vertices:
                    xy = entry + point[0] * right + point[1] * forward
                    mesh_clearance = min(
                        mesh_clearance,
                        float(point[2] - sampler(float(xy[0]), float(xy[1]))),
                    )
            render_lift = max(0.0, 0.02 - mesh_clearance)
            if render_lift:
                for name in pose:
                    pose[name] = pose[name] + np.array([0.0, 0.0, render_lift])
            clearances = []
            for name in wheel_names:
                for point in pose[name]:
                    xy = entry + point[0] * right + point[1] * forward
                    clearances.append(
                        float(point[2] - sampler(float(xy[0]), float(xy[1])))
                    )
            frames.append({
                "terrain": terrain, "pose": pose, "entry": entry,
                "forward": forward, "right": right, "center": global_center,
                "scoop": scoop_index, "load": info["loaded_volume_m3"],
                "remaining": float(terrain.sum() * env.spacing**2 / initial_volume),
                "clearance": min(clearances),
                "phase": (
                    "Approach" if progress < 0.58
                    else "Cut and fill" if progress < 0.78
                    else "Curl and lift"
                ),
            })
        if terminated:
            break

    colors = {
        "simple_chassis": "#f5a000", "simple_cab": "#f9c74f",
        "wheel_front_left": "#202020", "wheel_front_right": "#202020",
        "wheel_rear_left": "#202020", "wheel_rear_right": "#202020",
        "loader_boom": "#f5a000", "loader_bucket": "#e67e00",
    }
    fig, (ax_top, ax_side) = plt.subplots(
        1, 2, figsize=(12.8, 5.7), constrained_layout=True
    )

    def draw(index: int) -> None:
        frame = frames[index]
        terrain = frame["terrain"]
        ax_top.clear()
        ax_side.clear()
        ax_top.imshow(
            terrain, origin="lower",
            extent=(0, env.workspace_size_m, 0, env.workspace_size_m),
            cmap="terrain", vmin=0, vmax=float(frames[0]["terrain"].max()),
        )
        center = frame["center"]
        ax_top.scatter(center[1], center[0], color="#ff9800", s=75)
        ax_top.arrow(
            center[1], center[0], frame["forward"][1] * 2,
            frame["forward"][0] * 2, color="#d32f2f", width=0.08,
        )
        ax_top.set(
            xlim=(0, env.workspace_size_m), ylim=(0, env.workspace_size_m),
            xlabel="y [m]", ylabel="x [m]", aspect="equal",
            title="Excavation overview and current approach",
        )

        local_y = np.linspace(-10, 6, 280)
        points = frame["entry"][None, :] + local_y[:, None] * frame["forward"]
        ii = np.clip(
            np.rint(points[:, 0] / env.spacing).astype(int), 0, env.grid_size - 1
        )
        jj = np.clip(
            np.rint(points[:, 1] / env.spacing).astype(int), 0, env.grid_size - 1
        )
        profile = terrain[ii, jj]
        ax_side.fill_between(local_y, 0, profile, color="#9b7653", alpha=0.9)
        ax_side.plot(local_y, profile, color="#5d4037", linewidth=1.2)
        for name, vertices in frame["pose"].items():
            faces = parts[name].faces
            stride = 1 if len(faces) < 180 else 4
            ax_side.add_collection(PolyCollection(
                vertices[:, 1:3][faces[::stride]], facecolor=colors[name],
                edgecolor="#202020", linewidth=0.15, alpha=0.97,
            ))
        ax_side.set(
            xlim=(-10, 6), ylim=(0, 8), xlabel="approach direction [m]",
            ylabel="height [m]", aspect="equal",
            title=(
                f"simple_wheel_loader.obj | wheel clearance "
                f"{frame['clearance'] * 1000:.0f} mm"
            ),
        )
        ax_side.grid(alpha=0.2)
        fig.suptitle(
            f"Scoop {frame['scoop']}/{len(actions)} | "
            f"{frame['phase']} | "
            f"load {frame['load']:.2f} m³ | "
            f"remaining {100 * frame['remaining']:.2f}%"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    if args.output.suffix.lower() == ".mp4":
        import imageio_ffmpeg
        import matplotlib

        matplotlib.rcParams["animation.ffmpeg_path"] = (
            imageio_ffmpeg.get_ffmpeg_exe()
        )
        writer = FFMpegWriter(
            fps=args.fps,
            codec="libx264",
            bitrate=1800,
            extra_args=[
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ],
        )
    else:
        writer = PillowWriter(fps=args.fps)
    animation.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(fig)
    result = {
        "source": str(args.pile_dir),
        "animation": str(args.output),
        "obj": str(Path("simple_wheel_loader.obj").resolve()),
        "scoops": len(actions),
        "frames": len(frames),
        "frames_per_scoop": args.frames_per_scoop,
        "fps": args.fps,
        "duration_seconds": len(frames) / args.fps,
        "final_remaining_fraction": frames[-1]["remaining"],
        "minimum_rendered_wheel_clearance_m": min(
            frame["clearance"] for frame in frames
        ),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
