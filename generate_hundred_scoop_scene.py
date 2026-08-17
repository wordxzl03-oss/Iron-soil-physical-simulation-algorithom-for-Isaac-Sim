"""Continuously scoop one persistent randomized pile one hundred times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

from generate_five_scoop_groups import (
    loader_mechanism_points,
    simulate_group,
)
from render_obj_loader_scooping import (
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
    rotate_yz,
    transform_point,
)
from wheel_loader_dynamics import (
    VehicleParameters,
    VehicleState,
    fit_vehicle_to_terrain,
    grid_height_function,
)

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def render(group: dict, output_dir: Path, fps: int = 5) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = group["frames"][::2]  # initial plus each post-scoop stable state
    coordinates = group["x"]
    resolution = float(group["resolution_m"])
    passes = group["passes"]
    initial_volume = float(group["initial"].sum() * resolution**2)
    volumes = np.asarray(
        [frame.sum() * resolution**2 for frame in frames]
    )
    loads = np.asarray([item["loaded_volume_m3"] for item in passes])
    cumulative_load = np.r_[0.0, np.cumsum(loads)]
    force_kn = np.asarray(
        [item["peak_resistance_n"] for item in passes]
    ) / 1000.0

    raw = load_obj_parts(Path("wheel_buck.obj"))
    prepared, points = _prepare_model(raw, infer_landmarks(raw))
    bucket_part = prepared["loader_bucket_world.stl"]
    root, pin, edge = (
        points["root_pin"], points["bucket_pin"], points["cutting_edge"]
    )
    angles = np.linspace(-18.0, 8.0, 1001)
    heights = np.asarray(
        [transform_point(edge, root, angle)[2] for angle in angles]
    )
    low_angle = float(angles[np.argmin(np.abs(heights - 0.12))])
    pin_low = transform_point(pin, root, low_angle)
    chassis_pivot = np.array([0.0, -2.275, 0.72])
    parameters = VehicleParameters()

    fig, ((ax_top, ax_side), (ax_change, ax_stats)) = plt.subplots(
        2, 2, figsize=(12.4, 8.4), constrained_layout=True
    )

    def draw(frame_index: int) -> None:
        for axis in (ax_top, ax_side, ax_change, ax_stats):
            axis.clear()
        terrain = frames[frame_index]
        completed = frame_index
        active_index = min(max(completed - 1, 0), len(passes) - 1)
        record = passes[active_index]
        vehicle_x = float(record["lateral_offset_m"])
        edge_target_y = (
            float(record["entry_y_m"])
            + float(record["capacity_limited_penetration_m"])
        )

        ax_top.imshow(
            terrain, origin="lower", cmap="terrain", vmin=0.0, vmax=10.0,
            extent=(coordinates[0], coordinates[-1],
                    coordinates[0], coordinates[-1]),
        )
        if completed:
            recent = passes[max(0, completed - 12):completed]
            for old in recent:
                ax_top.plot(
                    [old["entry_y_m"],
                     old["entry_y_m"] + old["capacity_limited_penetration_m"]],
                    [old["lateral_offset_m"]] * 2,
                    color="#d32f2f", alpha=0.30, linewidth=0.8,
                )
        vehicle_shift = edge_target_y - transform_point(
            edge, root, low_angle
        )[1] - 3.0
        ax_top.add_patch(plt.Rectangle(
            (vehicle_shift - 4.20, vehicle_x - 1.30),
            3.85, 2.60, facecolor="#f5a000",
            edgecolor="#202020", linewidth=0.8,
        ))
        ax_top.set(
            xlim=(-15, 15), ylim=(-15, 15), aspect="equal",
            xlabel="y [m]", ylabel="x [m]",
            title=f"同一料堆持续作业 | 已完成 {completed}/100 铲",
        )

        row = int(np.clip(
            round((vehicle_x - coordinates[0]) / resolution),
            0, terrain.shape[0] - 1,
        ))
        sampler = grid_height_function(
            terrain, (coordinates[0], coordinates[0]),
            (resolution, resolution),
        )
        state = VehicleState(x_m=vehicle_x, y_m=vehicle_shift)
        pose = fit_vehicle_to_terrain(state, sampler, parameters)
        translation = np.array([
            vehicle_x, vehicle_shift, pose.z_m - 0.72
        ])
        boom_angle = low_angle + float(record.get("boom_lift_deg", 34.0))
        curl_angle = float(record.get("curl_angle_deg", 47.0))
        root_local, pin_now, _ = loader_mechanism_points(
            root, pin, edge, low_angle, boom_angle, curl_angle
        )
        bucket_low = rotate_yz(bucket_part.vertices, root, low_angle)
        bucket_vertices = (
            rotate_yz(bucket_low, pin_low, curl_angle)
            + (pin_now - pin_low)
        )
        bucket_world = (
            rotate_yz(
                bucket_vertices, chassis_pivot,
                np.rad2deg(pose.pitch_rad),
            )
            + translation
        )
        chassis = np.array([
            [0.0, -4.20, 0.68], [0.0, -0.35, 0.68],
            [0.0, -0.35, 2.25], [0.0, -4.20, 2.25],
        ])
        chassis_world = (
            rotate_yz(chassis, chassis_pivot, np.rad2deg(pose.pitch_rad))
            + translation
        )
        root_world = (
            rotate_yz(
                root_local[None, :], chassis_pivot,
                np.rad2deg(pose.pitch_rad),
            )[0] + translation
        )
        pin_world = (
            rotate_yz(
                pin_now[None, :], chassis_pivot,
                np.rad2deg(pose.pitch_rad),
            )[0] + translation
        )
        ax_side.fill_between(
            coordinates, 0.0, terrain[row],
            color="#9b7653", alpha=0.86,
        )
        ax_side.plot(
            coordinates, group["initial"][row],
            "--", color="#6d4c41", linewidth=1.0, label="初始剖面",
        )
        ax_side.fill(
            chassis_world[:, 1], chassis_world[:, 2],
            color="#f5a000", edgecolor="#202020",
        )
        for local_y in (-1.05, -3.35):
            center = rotate_yz(
                np.array([[0.0, local_y, 0.74]]),
                chassis_pivot, np.rad2deg(pose.pitch_rad),
            )[0] + translation
            ax_side.add_patch(plt.Circle(
                (center[1], center[2]), 0.72, color="#202020"
            ))
        ax_side.plot(
            [root_world[1], pin_world[1]],
            [root_world[2], pin_world[2]],
            color="#f5a000", linewidth=5.0,
        )
        yz = bucket_world[:, 1:3]
        stride = max(1, len(bucket_part.faces) // 260)
        ax_side.add_collection(PolyCollection(
            yz[bucket_part.faces[::stride]],
            facecolor="#e67e00", edgecolor="none", alpha=0.92,
        ))
        ax_side.set(
            xlim=(vehicle_shift - 6.0, vehicle_shift + 8.0),
            ylim=(0.0, 7.0), aspect="equal",
            xlabel="y [m]", ylabel="z [m]",
            title=(
                f"当前轨迹侧视 | pitch={np.rad2deg(pose.pitch_rad):.1f}°，"
                f"roll={np.rad2deg(pose.roll_rad):.1f}°"
            ),
        )
        ax_side.legend(fontsize=7)

        change = group["initial"] - terrain
        ax_change.imshow(
            change, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
            extent=(coordinates[0], coordinates[-1],
                    coordinates[0], coordinates[-1]),
        )
        ax_change.set(
            aspect="equal", xlabel="y [m]", ylabel="x [m]",
            title="相对初始料堆的累计高程变化 [m]",
        )

        scoop_axis = np.arange(1, len(passes) + 1)
        ax_stats.plot(
            scoop_axis, cumulative_load[1:],
            color="#d32f2f", label="累计装载 [m³]",
        )
        ax_stats.axvline(completed, color="#202020", linestyle="--")
        ax_stats.set(
            xlim=(0, 100), ylim=(0, max(cumulative_load) * 1.05),
            xlabel="铲次", ylabel="累计装载 [m³]",
            title=(
                f"剩余料堆体积 {volumes[frame_index]:.1f}/"
                f"{initial_volume:.1f} m³"
            ),
        )
        force_axis = ax_stats.twinx()
        force_axis.plot(
            scoop_axis, force_kn, color="#1565c0",
            alpha=0.45, label="峰值阻力 [kN]",
        )
        force_axis.set_ylabel("峰值阻力 [kN]", color="#1565c0")
        ax_stats.grid(alpha=0.2)
        fig.suptitle(
            f"固定场景连续100铲 | 当前累计装载 "
            f"{cumulative_load[completed]:.1f} m³"
        )

    animation = FuncAnimation(fig, draw, frames=len(frames))
    gif_path = output_dir / "hundred_scoops.gif"
    animation.save(gif_path, PillowWriter(fps=fps), dpi=58)
    plt.close(fig)

    fig_s, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, field, title, cmap in (
        (axes[0, 0], group["initial"], "初始料堆", "terrain"),
        (axes[0, 1], group["final"], "100铲后料堆", "terrain"),
        (axes[1, 0], group["initial"] - group["final"], "累计高程变化", "RdBu_r"),
    ):
        image = axis.imshow(field, origin="lower", cmap=cmap)
        axis.set_title(title)
        fig_s.colorbar(image, ax=axis, shrink=0.78)
    axes[1, 1].plot(np.arange(101), cumulative_load, label="累计装载")
    axes[1, 1].plot(
        np.arange(101), initial_volume - volumes,
        "--", label="高度场净减少体积",
    )
    axes[1, 1].set(xlabel="铲次", ylabel="体积 [m³]", title="体积与装载")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)
    fig_s.savefig(output_dir / "hundred_scoops_summary.png", dpi=155)
    plt.close(fig_s)

    metadata = {
        key: value for key, value in group.items()
        if key not in {"initial", "final", "frames", "x", "y", "dynamics"}
    }
    metadata["initial_pile_volume_m3"] = initial_volume
    metadata["final_pile_volume_m3"] = float(volumes[-1])
    metadata["cumulative_loaded_volume_m3"] = float(cumulative_load[-1])
    (output_dir / "hundred_scoops_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "hundred_scoops_data.npz",
        initial_height=group["initial"],
        final_height=group["final"],
        post_scoop_heights=frames,
        x=coordinates, y=coordinates,
        load_m3=loads, cumulative_load_m3=cumulative_load,
        force_kn=force_kn, pile_volume_m3=volumes,
    )
    print(f"saved: {gif_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1731118847)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("hundred_scoop_scene"),
    )
    parser.add_argument("--resolution", type=float, default=0.25)
    args = parser.parse_args()
    group = simulate_group(
        1, args.seed, resolution=args.resolution, scoop_count=100
    )
    render(group, args.output_dir)


if __name__ == "__main__":
    main()
