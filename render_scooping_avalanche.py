"""Animate a wheel loader scooping a metastable pile and triggering collapse."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from physics_aware_trajectory import MaterialParameters, excavation_resistance
from render_hysteretic_avalanche import make_steep_stockpile
from slope_model import relax_hysteretic_avalanche


_CUTTING_EDGE_FROM_PIVOT = np.array([1.55, -0.80])


def _bucket_polygon(pivot_y: float, pivot_z: float, pitch_deg: float) -> np.ndarray:
    """Bucket outline rotated about its rear hinge, not its cutting edge."""
    # Rear hinge is the origin. Vertex 0 is the cutting edge; the remaining
    # vertices form the floor, back plate and upper lip.
    local = np.array([
        _CUTTING_EDGE_FROM_PIVOT,
        [0.02, -0.66],
        [-0.08, 0.45],
        [1.22, 0.28],
    ])
    angle = np.deg2rad(pitch_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    return local @ rotation.T + np.array([pivot_y, pivot_z])


def _pivot_for_edge(edge_y: float, edge_z: float, pitch_deg: float) -> tuple[float, float]:
    angle = np.deg2rad(pitch_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    offset = _CUTTING_EDGE_FROM_PIVOT @ rotation.T
    return edge_y - float(offset[0]), edge_z - float(offset[1])


def _cut_state(
    initial: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    fraction: float,
    entry_y: float,
    travel: float,
    width: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    reached = entry_y + travel * fraction
    longitudinal = np.clip((yy - entry_y) / travel, 0.0, 1.0)
    side_taper = np.clip(1.0 - (2.0 * np.abs(xx) / width) ** 8, 0.0, 1.0)
    inside = (
        (np.abs(xx) <= width / 2)
        & (yy >= entry_y)
        & (yy <= reached)
    )
    requested = max_depth * (0.30 + 0.70 * longitudinal) * side_taper
    removed = np.minimum(initial, requested) * inside
    return initial - removed, removed


def render(output: Path, fps: int = 12) -> None:
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    spacing = 0.25
    xx, yy, initial = make_steep_stockpile(spacing)
    entry_y, travel, bucket_width, max_depth = -7.15, 1.5, 3.2, 1.15

    approach_count, cut_count = 14, 26
    curl_count, lift_count, exit_count = 20, 30, 22
    terrain_frames: list[np.ndarray] = []
    bucket_states: list[tuple[float, float, float]] = []
    phases: list[str] = []
    forces: list[float] = []
    material = MaterialParameters(
        bulk_density_kg_m3=1900.0,
        cohesion_pa=12_000.0,
        internal_friction_deg=42.0,
        bucket_friction=0.45,
        velocity_drag=0.8,
    )

    for p in np.linspace(0, 1, approach_count, endpoint=False):
        terrain_frames.append(initial)
        edge_y = -10.0 + (entry_y + 10.0) * p
        pivot_y, pivot_z = _pivot_for_edge(edge_y, 0.15, 0.0)
        bucket_states.append((pivot_y, pivot_z, 0.0))
        phases.append("1. 地面接近")
        forces.append(0.0)

    cut_terrains: list[np.ndarray] = []
    removed = np.zeros_like(initial)
    for p in np.linspace(0, 1, cut_count):
        terrain, removed = _cut_state(
            initial, xx, yy, p, entry_y, travel, bucket_width, max_depth
        )
        cut_terrains.append(terrain)
        edge_y = entry_y + travel * p
        # Phase 1: rigid, level penetration. Curl is deliberately postponed
        # until penetration is complete.
        pitch = 0.0
        pivot_y, pivot_z = _pivot_for_edge(edge_y, 0.15, pitch)
        terrain_frames.append(terrain)
        bucket_states.append((pivot_y, pivot_z, pitch))
        phases.append("2. 铲斗保持水平，短程贯入")
        depth = max_depth * min(1.0, p / 0.72)
        load = float(removed.sum() * spacing**2)
        forces.append(
            excavation_resistance(depth, bucket_width, 0.75, load, material)
        )

    scooped = cut_terrains[-1]
    disturbance = np.exp(
        -((xx / 2.0) ** 2 + ((yy - (entry_y + travel * 0.72)) / 1.15) ** 2)
    )
    # The exposed front face may fail across its width. The rear half remains
    # cohesive; the mask boundary acts as the crown scarp of the slide.
    exposed_front = yy < 0.5
    collapsed, stats, _, collapse_history = relax_hysteretic_avalanche(
        scooped,
        (spacing, spacing),
        start_angle_deg=72.0,
        stop_angle_deg=38.0,
        friction_angle_deg=38.0,
        cohesion_pa=30_000.0,
        disturbance=0.98 * disturbance,
        failure_mask=exposed_front,
        stochasticity=0.012,
        seed=19,
        record_history=True,
    )
    event_count = curl_count + lift_count
    indices = np.unique(
        np.linspace(0, len(collapse_history) - 1, event_count).astype(int)
    )
    collapse_frames = [collapse_history[index] for index in indices]
    while len(collapse_frames) < event_count:
        collapse_frames.append(collapsed)

    # Phase 2: hold the rear hinge fixed and curl. The cutting edge therefore
    # follows a circular arc upward, as on a real Z-bar/parallel-bar linkage.
    final_edge_y = entry_y + travel
    fixed_pivot_y, fixed_pivot_z = _pivot_for_edge(final_edge_y, 0.15, 0.0)
    for p, terrain in zip(
        np.linspace(0, 1, curl_count), collapse_frames[:curl_count]
    ):
        terrain_frames.append(terrain)
        bucket_states.append((fixed_pivot_y, fixed_pivot_z, 50.0 * p))
        phases.append("3. 后铰点固定：刃口沿圆弧向上收斗")
        forces.append(forces[-1] * (1.0 - 0.75 * p))

    # Phase 3: after curling is complete, translate the whole bucket upward.
    for p, terrain in zip(
        np.linspace(0, 1, lift_count), collapse_frames[curl_count:]
    ):
        terrain_frames.append(terrain)
        bucket_states.append((fixed_pivot_y, fixed_pivot_z + 3.0 * p, 50.0))
        phases.append("4. 收斗完成：举升臂带动整斗上移")
        forces.append(forces[-1] * (1.0 - p))

    for p in np.linspace(0, 1, exit_count):
        terrain_frames.append(collapsed)
        bucket_states.append((fixed_pivot_y - 4.2 * p, fixed_pivot_z + 3.0, 50.0))
        phases.append("5. 保持收斗姿态，举斗倒车退出")
        forces.append(0.0)

    loaded_volume = float(removed.sum() * spacing**2)
    peak_force = max(forces)
    edge_trace = np.array(
        [_bucket_polygon(*state)[0] for state in bucket_states]
    )
    center = initial.shape[0] // 2
    fig = plt.figure(figsize=(12, 5.4), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    def draw(index: int) -> None:
        ax3d.clear()
        ax2d.clear()
        terrain = terrain_frames[index]
        by, bz, pitch = bucket_states[index]
        bucket = _bucket_polygon(by, bz, pitch)

        ax3d.plot_surface(
            xx, yy, terrain, cmap="terrain", vmin=0, vmax=10,
            linewidth=0, antialiased=True
        )
        # A visible loader bucket: two side plates plus cutting edge.
        for x_side in (-bucket_width / 2, bucket_width / 2):
            ax3d.plot(
                np.full(len(bucket) + 1, x_side),
                np.r_[bucket[:, 0], bucket[0, 0]],
                np.r_[bucket[:, 1], bucket[0, 1]],
                color="#f28e00", linewidth=3.0,
            )
        for vertex in (0, 1, 2, 3):
            ax3d.plot(
                [-bucket_width / 2, bucket_width / 2],
                [bucket[vertex, 0]] * 2,
                [bucket[vertex, 1]] * 2,
                color="#f28e00", linewidth=2.2,
            )
        ax3d.set(
            xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 11),
            xlabel="横向 x [m]", ylabel="进铲方向 y [m]", zlabel="高度 [m]",
            title=phases[index],
        )
        ax3d.view_init(elev=24, azim=-58)

        profile = terrain[center]
        ax2d.fill_between(yy[center], 0, profile, color="#9b7653", alpha=0.9)
        ax2d.plot(yy[center], initial[center], "--", color="#666", lw=1.2, label="初始坡面")
        ax2d.plot(yy[center], profile, color="#5d4037", lw=2.0, label="当前坡面")
        ax2d.add_patch(
            Polygon(bucket, closed=True, facecolor="#ff9f1c",
                    edgecolor="black", linewidth=2.0, alpha=0.94)
        )
        ax2d.plot(
            edge_trace[: index + 1, 0], edge_trace[: index + 1, 1],
            "--", color="#d62728", linewidth=1.5, alpha=0.85,
            label="切削刃轨迹",
        )
        ax2d.scatter(
            [by], [bz], s=34, color="#1565c0", edgecolor="white",
            linewidth=0.7, zorder=6, label="后铰点",
        )
        if index >= approach_count + int(0.55 * cut_count):
            payload = bucket.mean(axis=0) + (bucket - bucket.mean(axis=0)) * np.array([0.62, 0.48])
            payload[:, 1] += 0.18
            ax2d.add_patch(
                Polygon(payload, closed=True, facecolor="#6d4c41",
                        edgecolor="#3e2723", linewidth=1.0)
            )
        ax2d.set(
            xlim=(-10, 10), ylim=(0, 11), aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="高度 [m]",
            title=f"中心剖面 | 当前阻力 {forces[index] / 1000:.0f} kN",
        )
        ax2d.grid(alpha=0.22)
        ax2d.legend(loc="upper right")
        fig.suptitle(
            f"真实尺度铲装—滞回塌方耦合 | 装载 {loaded_volume:.2f} m³ | "
            f"峰值阻力 {peak_force / 1000:.0f} kN",
            fontsize=12,
        )

    animation = FuncAnimation(fig, draw, frames=len(terrain_frames), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"saved: {output}")
    print(f"loaded_volume_m3={loaded_volume:.6f}")
    print(f"peak_resistance_n={peak_force:.1f}")
    print(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scooping_avalanche_demo/loader_scooping_collapse.gif"),
    )
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    render(args.output, args.fps)


if __name__ == "__main__":
    main()
