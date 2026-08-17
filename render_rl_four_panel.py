"""Four-panel wheel-loader renderer matching the 100-group GIF viewpoint."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Polygon

from animate_loader_3d import LoaderTrajectory


def _oriented_box(
    center: np.ndarray, forward: np.ndarray, length: float, width: float
) -> np.ndarray:
    right = np.array([forward[1], -forward[0]])
    return np.asarray(
        [
            center - 0.5 * length * forward - 0.5 * width * right,
            center + 0.5 * length * forward - 0.5 * width * right,
            center + 0.5 * length * forward + 0.5 * width * right,
            center - 0.5 * length * forward + 0.5 * width * right,
        ]
    )


def render_rl_scoop_four_panel(
    output: Path,
    initial: np.ndarray,
    scooped: np.ndarray,
    final: np.ndarray,
    removed: np.ndarray,
    path: np.ndarray,
    pitch_deg: np.ndarray,
    cut_fraction: np.ndarray,
    trajectory: LoaderTrajectory,
    loaded_volume: float,
    spacing: float,
    workspace_size: float,
    resistance_n: np.ndarray,
    speed_m_s: np.ndarray,
    fps: int,
    dpi: int,
    *,
    scoop_number: int = 1,
    total_scoops: int = 1,
    cumulative_loaded_volume: float | None = None,
    change_reference: np.ndarray | None = None,
) -> None:
    chinese_font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        plt.rcParams["font.family"] = "Microsoft YaHei"
        plt.rcParams["axes.unicode_minus"] = False
    coordinates = np.arange(initial.shape[0]) * spacing
    heading = np.deg2rad(trajectory.heading_deg)
    # Soil coordinates are [x, y], plot/path coordinates are [y, x].
    forward_soil = np.array([np.sin(heading), np.cos(heading)])
    forward_plot = np.array([forward_soil[1], forward_soil[0]])
    right_plot = np.array([forward_plot[1], -forward_plot[0]])
    frame_indices = np.linspace(0, len(path) - 1, 24).round().astype(int)
    fig, ((ax_top, ax_side), (ax_contact, ax_change)) = plt.subplots(
        2, 2, figsize=(12.8, 8.6), constrained_layout=True
    )

    def soil_for(index: int) -> np.ndarray:
        fraction = float(cut_fraction[index])
        if fraction < 1.0:
            return initial - removed * fraction
        slump = max(0.0, (index / max(len(path) - 1, 1) - 0.82) / 0.18)
        return scooped * (1.0 - slump) + final * slump

    def sample_profile(
        terrain: np.ndarray, center_plot: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linspace(-8.0, 9.0, 260)
        points = center_plot[None, :] + distance[:, None] * forward_plot
        soil_x = points[:, 1]
        soil_y = points[:, 0]
        ii = np.clip(np.rint(soil_x / spacing).astype(int), 0, terrain.shape[0] - 1)
        jj = np.clip(np.rint(soil_y / spacing).astype(int), 0, terrain.shape[1] - 1)
        return distance, terrain[ii, jj]

    def draw(frame: int) -> None:
        for axis in (ax_top, ax_side, ax_contact, ax_change):
            axis.clear()
        index = int(frame_indices[frame])
        terrain = soil_for(index)
        bucket = path[index, :2]
        # Bucket path is the control reference. Chassis stays behind it.
        chassis_center = bucket - 3.25 * forward_plot
        progress = float(cut_fraction[index])
        phase_t = index / max(len(path) - 1, 1)

        ax_top.imshow(
            terrain,
            origin="lower",
            extent=(0.0, workspace_size, 0.0, workspace_size),
            cmap="terrain",
            vmin=0.0,
            vmax=max(7.0, float(initial.max())),
        )
        ax_top.plot(path[:, 0], path[:, 1], "--", color="#d32f2f", linewidth=1.4)
        ax_top.add_patch(
            Polygon(
                _oriented_box(chassis_center, forward_plot, 4.0, 2.6),
                closed=True,
                facecolor="#f5a000",
                edgecolor="#202020",
                linewidth=1.2,
                zorder=5,
            )
        )
        root = chassis_center + 1.75 * forward_plot
        ax_top.plot(
            [root[0], bucket[0]], [root[1], bucket[1]],
            color="#e67e00", linewidth=4.0, zorder=6,
        )
        bucket_left = bucket - 0.5 * trajectory.bucket_width * right_plot
        bucket_right = bucket + 0.5 * trajectory.bucket_width * right_plot
        ax_top.plot(
            [bucket_left[0], bucket_right[0]],
            [bucket_left[1], bucket_right[1]],
            color="#e67e00", linewidth=6.0, zorder=7,
        )
        for longitudinal in (-1.25, 1.25):
            wheel_center = chassis_center + longitudinal * forward_plot
            for lateral in (-1.38, 1.38):
                center = wheel_center + lateral * right_plot
                ax_top.add_patch(Circle(center, 0.24, color="#202020", zorder=6))
        ax_top.set(
            xlim=(0, workspace_size), ylim=(0, workspace_size),
            xlabel="y [m]", ylabel="x [m]", aspect="equal",
            title="俯视：训练策略轨迹与随机土堆",
        )

        distance, profile = sample_profile(terrain, bucket)
        bucket_s = 0.0
        chassis_s = -3.25
        wheel_ground = float(np.interp(chassis_s, distance, profile))
        chassis_z = wheel_ground + 0.72
        chassis_poly = np.array(
            [
                [chassis_s - 2.0, chassis_z],
                [chassis_s + 1.75, chassis_z],
                [chassis_s + 1.75, chassis_z + 1.55],
                [chassis_s - 2.0, chassis_z + 1.55],
            ]
        )
        bucket_z = float(path[index, 2])
        boom_root = np.array([chassis_s + 1.45, chassis_z + 1.30])
        bucket_center = np.array([bucket_s, bucket_z + 0.22])
        bucket_angle = np.deg2rad(float(pitch_deg[index]))
        bucket_shape = np.array(
            [[0.55, -0.18], [-0.55, -0.08], [-0.42, 0.48], [0.40, 0.38]]
        )
        rotation = np.array(
            [[np.cos(bucket_angle), -np.sin(bucket_angle)],
             [np.sin(bucket_angle), np.cos(bucket_angle)]]
        )
        bucket_shape = bucket_shape @ rotation.T + bucket_center

        ax_side.fill_between(distance, 0.0, profile, color="#9b7653", alpha=0.88)
        ax_side.plot(distance, profile, color="#5d4037", linewidth=1.2)
        ax_side.add_patch(
            Polygon(chassis_poly, closed=True, facecolor="#f5a000", edgecolor="#202020")
        )
        for wheel_s in (chassis_s - 1.15, chassis_s + 1.05):
            ax_side.add_patch(
                Circle((wheel_s, chassis_z), 0.72, facecolor="#202020", edgecolor="black")
            )
        elbow = 0.52 * boom_root + 0.48 * bucket_center + np.array([0.0, 0.35])
        ax_side.plot(
            [boom_root[0], elbow[0], bucket_center[0]],
            [boom_root[1], elbow[1], bucket_center[1]],
            color="#e67e00", linewidth=5.0,
        )
        ax_side.add_patch(
            Polygon(bucket_shape, closed=True, facecolor="#e67e00", edgecolor="#9a4d00")
        )
        ax_side.set(
            xlim=(-7.0, 7.0), ylim=(0.0, max(7.0, float(initial.max()) + 1.2)),
            xlabel="进铲方向 [m]", ylabel="高度 z [m]", aspect="equal",
            title=f"轨迹侧视 | pitch={pitch_deg[index]:.1f}°",
        )
        ax_side.grid(alpha=0.2)

        _, pre_profile = sample_profile(initial, bucket)
        ax_contact.fill_between(
            distance, 0.0, profile, color="#9b7653", alpha=0.88, label="当前土体"
        )
        ax_contact.plot(
            distance, pre_profile, "--", color="#6d4c41", linewidth=1.2,
            label="铲装前坡面",
        )
        ax_contact.fill_between(
            distance, profile, pre_profile,
            where=pre_profile > profile + 1e-8,
            color="#ef5350", alpha=0.60, label="已切除物料",
        )
        ax_contact.add_patch(
            Polygon(bucket_shape, closed=True, facecolor="#e67e00", edgecolor="#9a4d00")
        )
        force = float(resistance_n[index])
        if force > 1.0:
            ax_contact.annotate(
                "", xy=(-1.35, bucket_z + 0.45), xytext=(0.0, bucket_z + 0.45),
                arrowprops=dict(arrowstyle="-|>", color="#1565c0", linewidth=2.0),
            )
            ax_contact.text(
                -1.35, bucket_z + 0.62, f"土体反力 {force / 1000:.0f} kN",
                color="#1565c0", fontsize=8,
            )
        ax_contact.set(
            xlim=(-2.5, 2.5),
            ylim=(max(0.0, bucket_z - 1.2), max(2.5, bucket_z + 2.0)),
            xlabel="进铲方向 [m]", ylabel="高度 z [m]", aspect="equal",
            title=f"铲斗—土体接触 | 进度={100 * progress:.0f}% | v={speed_m_s[index]:.2f} m/s",
        )
        ax_contact.grid(alpha=0.2)
        ax_contact.legend(loc="upper right", fontsize=7)

        reference = initial if change_reference is None else change_reference
        change = reference - terrain
        ax_change.imshow(
            change,
            origin="lower",
            extent=(0.0, workspace_size, 0.0, workspace_size),
            cmap="RdBu_r", vmin=-1.25, vmax=1.25,
        )
        ax_change.set(
            xlim=(0, workspace_size), ylim=(0, workspace_size),
            xlabel="y [m]", ylabel="x [m]", aspect="equal",
            title="累计高程变化 [m]",
        )
        if phase_t < 0.20:
            phase = "接近坡脚"
        elif phase_t < 0.45:
            phase = "水平贯入"
        elif phase_t < 0.70:
            phase = "举升并卷斗"
        elif phase_t < 0.80:
            phase = "完成起铲"
        else:
            phase = "举斗倒车退出"
        fig.suptitle(
            f"训练模型连续铲装：第{scoop_number}/{total_scoops}铲 · {phase} | "
            f"本铲 {loaded_volume:.2f} m³ | "
            f"累计 {(loaded_volume if cumulative_loaded_volume is None else cumulative_loaded_volume):.2f} m³ | "
            f"heading={trajectory.heading_deg:+.1f}°"
        )

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
