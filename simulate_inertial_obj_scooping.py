"""Accelerate a simple OBJ wheel loader, coast into a pile, curl and lift."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection

from physics_aware_trajectory import MaterialParameters, excavation_resistance
from render_multi_obj_loader_scooping import cut_state_offset
from render_obj_loader_scooping import (
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
    make_flatter_stockpile,
    rotate_yz,
    transform_point,
)
from wheel_loader_dynamics import (
    VehicleParameters,
    VehicleState,
    fit_vehicle_to_terrain,
    grid_height_function,
    integrate_bicycle,
)

rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "DejaVu Sans"
]
rcParams["axes.unicode_minus"] = False


def simulate_inertial_entry(
    cutting_edge_offset_y: float,
    entry_y: float,
    *,
    target_impact_speed_m_s: float = 2.0,
    post_contact_drive_command: float = 0.40,
    bucket_width_m: float = 2.7,
    max_depth_m: float = 1.05,
    nominal_penetration_m: float = 1.5,
    dt: float = 0.05,
    vehicle_x_m: float = 0.0,
    terrain_height_function: Callable[[float, float], float] | None = None,
    vehicle_parameters: VehicleParameters | None = None,
    axle_center_offset_y: float = 0.0,
) -> tuple[np.ndarray, list]:
    """Return vehicle/force history until inertia and resistance stop the loader."""
    vehicle = vehicle_parameters or VehicleParameters()
    material = MaterialParameters(
        bulk_density_kg_m3=1900.0,
        cohesion_pa=12_000.0,
        internal_friction_deg=42.0,
        bucket_friction=0.45,
        velocity_drag=0.8,
    )
    edge_from_axle_center = cutting_edge_offset_y - axle_center_offset_y
    contact_center = entry_y - edge_from_axle_center
    state = VehicleState(x_m=vehicle_x_m, y_m=contact_center - 4.0)
    if terrain_height_function is not None:
        pose = fit_vehicle_to_terrain(
            state, terrain_height_function, vehicle
        )
        state.z_m = pose.z_m
        state.pitch_rad = pose.pitch_rad
        state.roll_rad = pose.roll_rad
    rows: list[list[float]] = []
    states: list[VehicleState] = []
    for _ in range(round(15.0 / dt)):
        edge_y = state.y_m + edge_from_axle_center
        penetration = max(0.0, edge_y - entry_y)
        depth = min(
            max_depth_m,
            max_depth_m * penetration / nominal_penetration_m,
        )
        accumulated = depth * bucket_width_m * penetration * 0.62
        resistance = (
            excavation_resistance(
                depth,
                bucket_width_m,
                abs(state.speed_m_s),
                accumulated,
                material,
            )
            if penetration > 0
            else 0.0
        )
        if penetration <= 0:
            drive = float(
                np.clip(
                    0.04 + 0.23 * (target_impact_speed_m_s - state.speed_m_s),
                    0.0,
                    0.70,
                )
            )
        else:
            drive = post_contact_drive_command
        next_state, forces = integrate_bicycle(
            state,
            drive,
            0.0,
            0.0,
            dt,
            vehicle,
            excavation_resistance_n=resistance,
            terrain_height_function=terrain_height_function,
        )
        rows.append([
            state.time_s, state.y_m, edge_y, state.speed_m_s,
            penetration, depth, accumulated, drive,
            forces.applied_drive_n, -forces.excavation_n,
            -(forces.rolling_n + forces.velocity_n + forces.grade_n),
            forces.net_n, state.z_m, state.pitch_rad, state.roll_rad,
        ])
        states.append(state)
        state = next_state
        if penetration > 0.15 and state.speed_m_s <= 0.03:
            break
    return np.asarray(rows), states


def rotate_xz(
    points: np.ndarray, pivot: np.ndarray, angle_deg: float
) -> np.ndarray:
    """Roll world points about an axis parallel to +Y."""
    angle = np.deg2rad(angle_deg)
    result = np.asarray(points, dtype=float).copy()
    rel_x = result[:, 0] - pivot[0]
    rel_z = result[:, 2] - pivot[2]
    result[:, 0] = pivot[0] + np.cos(angle) * rel_x + np.sin(angle) * rel_z
    result[:, 2] = pivot[2] - np.sin(angle) * rel_x + np.cos(angle) * rel_z
    return result


def render(
    simple_obj: Path,
    source_obj: Path,
    output_dir: Path,
    fps: int = 10,
) -> None:
    simple_parts = load_obj_parts(simple_obj)
    raw = load_obj_parts(source_obj)
    _, points = _prepare_model(raw, infer_landmarks(raw))
    root, pin, edge = points["root_pin"], points["bucket_pin"], points["cutting_edge"]
    candidates = np.linspace(-18.0, 8.0, 2001)
    heights = np.array(
        [transform_point(edge, root, angle)[2] for angle in candidates]
    )
    low_angle = float(candidates[np.argmin(np.abs(heights - 0.12))])
    low_edge = transform_point(edge, root, low_angle)
    bucket_width = float(
        np.ptp(simple_parts["loader_bucket"].vertices[:, 0])
    )
    entry_y = -11.15
    spacing = 0.25
    xx, yy, initial = make_flatter_stockpile(
        spacing, slope_angle_deg=42.0
    )
    terrain_sampler = grid_height_function(
        initial, (float(xx[0, 0]), float(yy[0, 0])),
        (spacing, spacing),
    )
    data, dynamic_states = simulate_inertial_entry(
        low_edge[1], entry_y, bucket_width_m=bucket_width,
        terrain_height_function=terrain_sampler,
    )
    final_penetration = float(data[-1, 4])
    final_shift = float(data[-1, 1])

    final_terrain, removed = cut_state_offset(
        initial, xx, yy, 1.0, 0.0, entry_y,
        final_penetration, bucket_width, 1.05,
    )
    loaded_volume = float(removed.sum() * spacing**2)

    # Downsample dynamics, then append fixed-pin curl and boom lift.
    dynamic_indices = np.unique(
        np.linspace(0, len(data) - 1, min(58, len(data))).astype(int)
    )
    render_states: list[tuple[float, float, float]] = []
    terrains: list[np.ndarray] = []
    phases: list[str] = []
    row_indices: list[int] = []
    for index in dynamic_indices:
        penetration = float(data[index, 4])
        fraction = np.clip(
            penetration / max(final_penetration, 1e-9), 0.0, 1.0
        )
        terrain, _ = cut_state_offset(
            initial, xx, yy, fraction, 0.0, entry_y,
            final_penetration, bucket_width, 1.05,
        )
        render_states.append((float(data[index, 1]), low_angle, 0.0))
        terrains.append(terrain)
        phases.append(
            "接触前加速" if penetration <= 0
            else "依靠惯性贯入，速度自然下降"
        )
        row_indices.append(int(index))
    for p in np.linspace(0, 1, 16):
        render_states.append((final_shift, low_angle, 48.0 * p))
        terrains.append(final_terrain)
        phases.append("停车后绕斗销收斗")
        row_indices.append(len(data) - 1)
    high_angle = low_angle + 34.0
    for p in np.linspace(0, 1, 16):
        render_states.append(
            (final_shift, low_angle + (high_angle - low_angle) * p, 48.0)
        )
        terrains.append(final_terrain)
        phases.append("收斗完成后举升动臂")
        row_indices.append(len(data) - 1)

    fixed_names = [
        "simple_chassis", "simple_cab",
        "wheel_front_left", "wheel_front_right",
        "wheel_rear_left", "wheel_rear_right",
    ]
    wheel_names = [name for name in fixed_names if name.startswith("wheel_")]
    wheel_centers = {
        name: simple_parts[name].vertices.mean(axis=0) for name in wheel_names
    }
    boom_base = simple_parts["loader_boom"]
    bucket_base = simple_parts["loader_bucket"]
    pin_low = transform_point(pin, root, low_angle)

    def pose_parts(
        state: tuple[float, float, float],
        distance_m: float,
        vehicle_state: VehicleState,
    ) -> dict[str, np.ndarray]:
        shift, boom_angle, curl_angle = state
        result = {}
        for name in fixed_names:
            vertices = simple_parts[name].vertices
            if name in wheel_names:
                vertices = rotate_yz(
                    vertices, wheel_centers[name],
                    -np.rad2deg(distance_m / 0.72),
                )
            result[name] = vertices
        result["loader_boom"] = (
            rotate_yz(boom_base.vertices, root, boom_angle)
        )
        pin_now = transform_point(pin, root, boom_angle)
        bucket_low = rotate_yz(bucket_base.vertices, root, low_angle)
        curled = rotate_yz(bucket_low, pin_low, curl_angle)
        result["loader_bucket"] = curled + (pin_now - pin_low)
        chassis_pivot = np.array([0.0, -2.275, 0.72])
        translation = np.array([
            vehicle_state.x_m,
            shift,
            vehicle_state.z_m - 0.72,
        ])
        for name, vertices in result.items():
            vertices = rotate_yz(
                vertices, chassis_pivot,
                np.rad2deg(vehicle_state.pitch_rad),
            )
            vertices = rotate_xz(
                vertices, chassis_pivot,
                np.rad2deg(vehicle_state.roll_rad),
            )
            result[name] = vertices + translation
        return result

    poses = [
        pose_parts(
            state,
            dynamic_states[row].distance_m,
            dynamic_states[row],
        )
        for state, row in zip(render_states, row_indices)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "inertial_scooping_data.npz",
        dynamics=data,
        columns=np.asarray([
            "time_s", "vehicle_y_m", "edge_y_m", "speed_m_s",
            "penetration_m", "depth_m", "estimated_load_m3", "drive_command",
            "drive_force_n", "excavation_force_n", "ground_resistance_n",
            "net_force_n", "vehicle_z_m", "pitch_rad", "roll_rad",
        ]),
        initial_height=initial,
        final_height=final_terrain,
    )

    colors = {
        "simple_chassis": "#f5a000", "simple_cab": "#f9c74f",
        "wheel_front_left": "#202020", "wheel_front_right": "#202020",
        "wheel_rear_left": "#202020", "wheel_rear_right": "#202020",
        "loader_boom": "#f5a000", "loader_bucket": "#e67e00",
    }
    center = initial.shape[0] // 2
    fig, (ax_scene, ax_force) = plt.subplots(
        1, 2, figsize=(12, 5.2), constrained_layout=True
    )
    ax_speed = ax_force.twinx()

    def draw(frame: int) -> None:
        ax_scene.clear()
        ax_force.clear()
        ax_speed.clear()
        terrain = terrains[frame]
        pose = poses[frame]
        ax_scene.fill_between(
            yy[center], 0, terrain[center], color="#9b7653", alpha=0.88
        )
        ax_scene.plot(yy[center], terrain[center], color="#5d4037", linewidth=1.8)
        for name, vertices in pose.items():
            part = simple_parts[name]
            yz = vertices[:, 1:3]
            stride = 1 if len(part.faces) < 150 else 4
            ax_scene.add_collection(
                PolyCollection(
                    yz[part.faces[::stride]],
                    facecolor=colors[name], edgecolor="#202020",
                    linewidth=0.15, alpha=0.94,
                )
            )
        row = row_indices[frame]
        ax_scene.set(
            xlim=(-20, 6), ylim=(0, 11), aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="高度 [m]",
            title=(
                f"{phases[frame]}\n"
                f"v={data[row,3]:.2f} m/s, 贯入={data[row,4]:.2f} m"
            ),
        )
        ax_scene.grid(alpha=0.2)

        ax_force.plot(data[:, 0], data[:, 8] / 1000, label="驱动力")
        ax_force.plot(data[:, 0], data[:, 9] / 1000, label="铲装阻力")
        ax_force.plot(data[:, 0], data[:, 10] / 1000, label="地面阻力")
        ax_force.axvline(data[row, 0], color="#d32f2f", linestyle="--")
        ax_speed.plot(data[:, 0], data[:, 3], color="#1565c0", label="速度")
        ax_speed.set_ylabel("速度 [m/s]", color="#1565c0")
        ax_force.set(
            xlabel="时间 [s]", ylabel="力 [kN]",
            title="驱动力—阻力—惯性减速",
        )
        ax_force.grid(alpha=0.2)
        ax_force.legend(loc="upper left")
        fig.suptitle(
            f"矩形车体OBJ惯性冲料 | 冲料速度 {np.max(data[:,3]):.2f} m/s | "
            f"最终贯入 {final_penetration:.2f} m | 装载 {loaded_volume:.2f} m³"
        )

    animation = FuncAnimation(fig, draw, frames=len(render_states))
    gif_path = output_dir / "inertial_obj_scooping.gif"
    animation.save(gif_path, PillowWriter(fps=fps), dpi=92)
    plt.close(fig)

    diagnostic = output_dir / "inertial_scooping_diagnostics.png"
    fig_d, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    axes[0].plot(data[:, 4], data[:, 3], linewidth=2)
    axes[0].set(
        xlabel="贯入深度 [m]", ylabel="速度 [m/s]",
        title="接触后速度随贯入自然下降",
    )
    axes[0].grid(alpha=0.25)
    axes[1].plot(data[:, 0], data[:, 8] / 1000, label="驱动力")
    axes[1].plot(data[:, 0], data[:, 9] / 1000, label="铲装阻力")
    axes[1].plot(data[:, 0], data[:, 11] / 1000, label="合力")
    axes[1].set(xlabel="时间 [s]", ylabel="力 [kN]", title="纵向力平衡")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig_d.savefig(diagnostic, dpi=160)
    plt.close(fig_d)
    print(f"saved: {gif_path}")
    print(f"saved: {diagnostic}")
    print(f"impact_speed_m_s={np.max(data[:,3]):.4f}")
    print(f"final_penetration_m={final_penetration:.4f}")
    print(f"loaded_volume_m3={loaded_volume:.4f}")
    print(f"peak_excavation_force_n={np.max(data[:,9]):.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obj", type=Path, default=Path("simple_wheel_loader.obj")
    )
    parser.add_argument(
        "--source-obj", type=Path, default=Path("wheel_buck.obj")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("inertial_scooping_demo")
    )
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    render(args.obj, args.source_obj, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
