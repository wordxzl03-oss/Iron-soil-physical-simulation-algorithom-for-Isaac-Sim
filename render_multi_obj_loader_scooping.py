"""Render several randomized OBJ wheel-loader scoops on one evolving pile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter

from physics_aware_trajectory import MaterialParameters, excavation_resistance
from render_obj_loader_scooping import (
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
    make_flatter_stockpile,
    make_payload_particles,
    rotate_yz,
    transform_point,
)
from slope_model import relax_localized_failure_wedge


def cut_state_offset(
    initial: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    fraction: float,
    entry_x: float,
    entry_y: float,
    travel: float,
    width: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove one laterally offset wheel-loader swept volume."""
    reached = entry_y + travel * fraction
    longitudinal = np.clip((yy - entry_y) / travel, 0.0, 1.0)
    transverse = np.abs(xx - entry_x)
    side_taper = np.clip(1.0 - (2.0 * transverse / width) ** 8, 0.0, 1.0)
    inside = (
        (transverse <= width / 2)
        & (yy >= entry_y)
        & (yy <= reached)
    )
    requested = max_depth * (0.30 + 0.70 * longitudinal) * side_taper
    removed = np.minimum(initial, requested) * inside
    return initial - removed, removed


def sample_trajectories(
    count: int, bucket_width: float, seed: int
) -> list[dict[str, float]]:
    """Generate reproducible, overlapping passes across the pile face."""
    rng = np.random.default_rng(seed)
    nominal = np.linspace(-1.25, 1.25, count) * bucket_width
    records = []
    for index, lateral in enumerate(nominal, start=1):
        records.append(
            {
                "pass": index,
                "lateral_offset_m": float(lateral + rng.uniform(-0.18, 0.18)),
                "penetration_m": float(rng.uniform(1.15, 1.50)),
                "max_depth_m": float(rng.uniform(0.92, 1.10)),
                "curl_angle_deg": float(rng.uniform(44.0, 50.0)),
                "boom_lift_deg": float(rng.uniform(30.0, 36.0)),
            }
        )
    return records


def render(
    obj_path: Path,
    output: Path,
    passes: int = 4,
    seed: int = 2042,
    fps: int = 12,
    pile_angle_deg: float = 42.0,
) -> None:
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    if not 2 <= passes <= 8:
        raise ValueError("passes must lie between 2 and 8")

    raw = load_obj_parts(obj_path)
    parts, points = _prepare_model(raw, infer_landmarks(raw))
    frame_part = parts["loader_frame_world.stl"]
    boom_part = parts["loader_boom_world.stl"]
    bucket_part = parts["loader_bucket_world.stl"]
    root, pin, edge = (
        points["root_pin"], points["bucket_pin"], points["cutting_edge"]
    )
    angles = np.linspace(-18.0, 8.0, 2001)
    heights = np.array([transform_point(edge, root, angle)[2] for angle in angles])
    low_angle = float(angles[np.argmin(np.abs(heights - 0.12))])
    low_edge = transform_point(edge, root, low_angle)
    bucket_width = float(np.ptp(bucket_part.vertices[:, 0]))

    spacing = 0.25
    xx, yy, initial = make_flatter_stockpile(
        spacing, slope_angle_deg=pile_angle_deg
    )
    current = initial.copy()
    entry_y = -11.15
    trajectories = sample_trajectories(passes, bucket_width, seed)
    material = MaterialParameters(
        bulk_density_kg_m3=1900.0,
        cohesion_pa=12_000.0,
        internal_friction_deg=42.0,
        bucket_friction=0.45,
        velocity_drag=0.8,
    )

    # state: x shift, y shift, boom angle, curl angle
    states: list[tuple[float, float, float, float]] = []
    terrains: list[np.ndarray] = []
    phases: list[str] = []
    pass_ids: list[int] = []
    fill_fractions: list[float] = []
    forces: list[float] = []
    summaries: list[dict[str, float | int | bool]] = []
    per_pass_final: list[np.ndarray] = []
    # Keep full-resolution physics but sample the expensive OBJ rendering.
    approach_n, cut_n, curl_n, lift_n, retreat_n, slump_n = 1, 3, 2, 2, 1, 3

    for trajectory in trajectories:
        pass_id = int(trajectory["pass"])
        offset = float(trajectory["lateral_offset_m"])
        penetration = float(trajectory["penetration_m"])
        max_depth = float(trajectory["max_depth_m"])
        curl_angle = float(trajectory["curl_angle_deg"])
        high_angle = low_angle + float(trajectory["boom_lift_deg"])
        base_y_shift = entry_y - low_edge[1]
        pass_initial = current.copy()
        pass_peak_force = 0.0
        removed = np.zeros_like(current)

        for p in np.linspace(0, 1, approach_n, endpoint=False):
            states.append((offset, base_y_shift - 2.8 * (1 - p), low_angle, 0.0))
            terrains.append(pass_initial)
            phases.append("接近坡脚")
            pass_ids.append(pass_id)
            fill_fractions.append(0.0)
            forces.append(0.0)

        for p in np.linspace(0, 1, cut_n):
            cut, removed = cut_state_offset(
                pass_initial, xx, yy, p, offset, entry_y,
                penetration, bucket_width, max_depth,
            )
            loaded = float(removed.sum() * spacing**2)
            force = excavation_resistance(
                max_depth * p, bucket_width, 0.62, loaded, material
            )
            pass_peak_force = max(pass_peak_force, force)
            states.append(
                (offset, base_y_shift + penetration * p, low_angle, 0.0)
            )
            terrains.append(cut)
            phases.append("低位贯入")
            pass_ids.append(pass_id)
            fill_fractions.append(float(p))
            forces.append(force)

        scooped = terrains[-1]
        collision_xy = (offset, entry_y + 0.80 * penetration)
        (
            failed,
            failure_stats,
            _,
            _,
            failure_mask,
            failure_history,
        ) = relax_localized_failure_wedge(
            scooped,
            (spacing, spacing),
            collision_xy,
            origin_xy=(float(xx[0, 0]), float(yy[0, 0])),
            collision_force_n=pass_peak_force,
            collision_displacement_m=0.45,
            bucket_width_m=bucket_width,
            disturbance_length_m=2.5,
            max_propagation_radius_m=5.5,
            active_layer_depth_m=0.65,
            wedge_half_angle_deg=58.0,
            stop_angle_deg=38.0,
            friction_angle_deg=40.0,
            cohesion_pa=5_000.0,
            bulk_density=1900.0,
            record_history=True,
        )
        final_y_shift = base_y_shift + penetration
        for p in np.linspace(0, 1, curl_n):
            states.append((offset, final_y_shift, low_angle, curl_angle * p))
            terrains.append(scooped)
            phases.append("绕斗销收斗")
            pass_ids.append(pass_id)
            fill_fractions.append(1.0)
            forces.append(pass_peak_force * (1 - p))
        for p in np.linspace(0, 1, lift_n):
            boom_angle = low_angle + (high_angle - low_angle) * p
            states.append((offset, final_y_shift, boom_angle, curl_angle))
            terrains.append(scooped)
            phases.append("动臂举升")
            pass_ids.append(pass_id)
            fill_fractions.append(1.0)
            forces.append(0.0)
        safe_y_shift = final_y_shift - 3.0
        for p in np.linspace(0, 1, retreat_n):
            states.append(
                (offset, final_y_shift - 3.0 * p, high_angle, curl_angle)
            )
            terrains.append(scooped)
            phases.append("带载退出危险区")
            pass_ids.append(pass_id)
            fill_fractions.append(1.0)
            forces.append(0.0)

        history_indices = np.linspace(
            0, len(failure_history) - 1, slump_n
        ).astype(int)
        for history_index in history_indices:
            states.append((offset, safe_y_shift, high_angle, curl_angle))
            terrains.append(failure_history[history_index])
            phases.append("局部滑裂")
            pass_ids.append(pass_id)
            fill_fractions.append(1.0)
            forces.append(0.0)

        current = failed
        per_pass_final.append(current.copy())
        loaded_volume = float(removed.sum() * spacing**2)
        summaries.append(
            {
                **trajectory,
                "loaded_volume_m3": loaded_volume,
                "peak_resistance_n": pass_peak_force,
                "failure_area_m2": failure_stats.active_area_m2,
                "failure_radius_m": failure_stats.max_propagation_distance_m,
                "failure_moved_volume_m3": failure_stats.moved_volume_m3,
                "failure_energy_used_j": failure_stats.energy_used_j,
                "max_active_depth_m": failure_stats.max_mobilized_depth_m,
                "triggered": failure_stats.triggered,
                "failure_cell_count": int(failure_mask.sum()),
            }
        )

    def pose_parts(
        state: tuple[float, float, float, float]
    ) -> dict[str, np.ndarray]:
        x_shift, y_shift, boom_angle, curl_angle = state
        translation = np.array([x_shift, y_shift, 0.0])
        frame = frame_part.vertices + translation
        boom = rotate_yz(boom_part.vertices, root, boom_angle) + translation
        pin_now = transform_point(pin, root, boom_angle)
        bucket_low = rotate_yz(bucket_part.vertices, root, low_angle)
        pin_low = transform_point(pin, root, low_angle)
        bucket = (
            rotate_yz(bucket_low, pin_low, curl_angle)
            + (pin_now - pin_low)
            + translation
        )
        return {"frame": frame, "boom": boom, "bucket": bucket}

    def pose_payload(
        base: np.ndarray, state: tuple[float, float, float, float]
    ) -> np.ndarray:
        x_shift, y_shift, boom_angle, curl_angle = state
        pin_now = transform_point(pin, root, boom_angle)
        pin_low = transform_point(pin, root, low_angle)
        low = rotate_yz(base, root, low_angle)
        return (
            rotate_yz(low, pin_low, curl_angle)
            + (pin_now - pin_low)
            + np.array([x_shift, y_shift, 0.0])
        )

    posed = [pose_parts(state) for state in states]
    payload_base = make_payload_particles(pin, bucket_width, count=110, seed=seed)
    maximum_penetration = 0.0
    coordinate_min = float(xx[0, 0])
    for terrain, pose in zip(terrains, posed):
        for key in ("frame", "boom"):
            vertices = pose[key]
            ix = np.rint((vertices[:, 0] - coordinate_min) / spacing).astype(int)
            iy = np.rint((vertices[:, 1] - coordinate_min) / spacing).astype(int)
            inside = (
                (ix >= 0) & (ix < terrain.shape[0])
                & (iy >= 0) & (iy < terrain.shape[1])
            )
            if np.any(inside):
                depth = terrain[ix[inside], iy[inside]] - vertices[inside, 2]
                maximum_penetration = max(
                    maximum_penetration, float(depth.max(initial=0.0))
                )
    if maximum_penetration > 0.03:
        raise RuntimeError(
            f"frame/boom penetrates terrain by {maximum_penetration:.3f} m"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "trajectories.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output.parent / "multi_scoop_terrain.npz",
        initial_height=initial,
        final_height=current,
        pass_final_height=np.asarray(per_pass_final),
        grid_spacing_m=np.asarray((spacing, spacing)),
        x=xx[:, 0],
        y=yy[0],
    )

    fig, (ax_side, ax_top) = plt.subplots(
        1, 2, figsize=(10.8, 4.8), constrained_layout=True
    )
    colors = {"frame": "#263238", "boom": "#f5a000", "bucket": "#e67e00"}

    def draw(index: int) -> None:
        ax_side.clear()
        ax_top.clear()
        terrain = terrains[index]
        pose = posed[index]
        track_x = float(states[index][0])
        track_i = int(
            np.clip(round((track_x - float(xx[0, 0])) / spacing), 0, terrain.shape[0] - 1)
        )
        profile = terrain[track_i]
        ax_side.fill_between(
            yy[track_i], 0, profile, color="#9b7653", alpha=0.88
        )
        ax_side.plot(yy[track_i], profile, color="#5d4037", linewidth=1.8)
        for key, part in (
            ("frame", frame_part), ("boom", boom_part), ("bucket", bucket_part)
        ):
            yz = pose[key][:, 1:3]
            stride = max(1, len(yz) // 260)
            ax_side.scatter(
                yz[::stride, 0], yz[::stride, 1],
                s=5, color=colors[key], edgecolor="#202020",
                linewidth=0.08, alpha=0.90,
            )
        fill_count = round(len(payload_base) * fill_fractions[index])
        if fill_count:
            payload = pose_payload(payload_base[:fill_count], states[index])
            ax_side.scatter(
                payload[:, 1], payload[:, 2],
                s=6, c="#5d4037", alpha=0.9,
            )
        ax_side.set(
            xlim=(-16, 13), ylim=(0, 11), aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="高度 [m]",
            title=f"轨迹剖面 | 第{pass_ids[index]}/{passes}铲：{phases[index]}",
        )
        ax_side.grid(alpha=0.2)

        change = initial - terrain
        ax_top.imshow(
            change, origin="lower",
            extent=(float(yy.min()), float(yy.max()), float(xx.min()), float(xx.max())),
            cmap="RdBu_r", vmin=-0.8, vmax=0.8,
        )
        for record in trajectories:
            color = "#d32f2f" if int(record["pass"]) == pass_ids[index] else "#555555"
            ax_top.plot(
                [entry_y - 2.8, entry_y + record["penetration_m"]],
                [record["lateral_offset_m"]] * 2,
                "--", color=color, linewidth=1.4,
            )
        # Show the loader footprint in the top view.
        frame_xy = pose["frame"][:, :2]
        ax_top.scatter(
            frame_xy[:: max(1, len(frame_xy) // 80), 1],
            frame_xy[:: max(1, len(frame_xy) // 80), 0],
            s=4, color="#263238", alpha=0.55,
        )
        ax_top.set(
            xlim=(-15, 2), ylim=(-9, 9), aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="横向 x [m]",
            title="累计高程变化与多轨迹",
        )
        fig.suptitle(
            f"连续多次铲装 | 当前第{pass_ids[index]}铲 | "
            f"累计装载 {sum(float(s['loaded_volume_m3']) for s in summaries[:pass_ids[index]]):.2f} m³ | "
            f"当前阻力 {forces[index] / 1000:.0f} kN",
            fontsize=12,
        )

    animation = FuncAnimation(fig, draw, frames=len(states), interval=1000 / fps)
    animation.save(output, PillowWriter(fps=fps), dpi=68)
    plt.close(fig)

    final_figure = output.parent / "multi_scoop_final_comparison.png"
    fig_f, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for axis, field, title, cmap in (
        (axes[0], initial, "初始高度 [m]", "terrain"),
        (axes[1], current, f"{passes}铲后高度 [m]", "terrain"),
        (axes[2], initial - current, "累计高程变化 [m]", "RdBu_r"),
    ):
        im = axis.imshow(
            field, origin="lower",
            extent=(float(yy.min()), float(yy.max()), float(xx.min()), float(xx.max())),
            cmap=cmap,
        )
        axis.set(title=title, xlabel="y [m]", ylabel="x [m]")
        fig_f.colorbar(im, ax=axis, shrink=0.78)
    fig_f.savefig(final_figure, dpi=160)
    plt.close(fig_f)
    print(f"saved: {output}")
    print(f"saved: {final_figure}")
    print(f"max_frame_boom_terrain_penetration_m={maximum_penetration:.6f}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=Path("wheel_buck.obj"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("multi_obj_scooping_demo/multi_scoop.gif"),
    )
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--pile-angle", type=float, default=42.0)
    args = parser.parse_args()
    render(
        args.obj, args.output, args.passes, args.seed, args.fps, args.pile_angle
    )


if __name__ == "__main__":
    main()
