"""Render a 10 m stockpile that holds a steep face, then suddenly collapses."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import font_manager

from slope_model import relax_hysteretic_avalanche


def make_steep_stockpile(
    resolution: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 20 x 20 m pile with a 64 degree locally cohesive front."""
    coords = np.arange(-10.0, 10.0 + 0.5 * resolution, resolution)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    front = np.maximum(0.0, (yy + 7.0) * np.tan(np.deg2rad(64.0)))
    rear = np.maximum(0.0, (9.0 - yy) * np.tan(np.deg2rad(45.0)))
    sides = np.maximum(0.0, (9.5 - np.abs(xx)) * np.tan(np.deg2rad(55.0)))
    height = np.minimum.reduce((front, rear, sides, np.full_like(xx, 10.0)))
    # Small-scale surface variation without creating grid-scale unstable spikes.
    height *= 1.0 + 0.012 * np.sin(0.8 * xx) * np.sin(0.65 * yy)
    return xx, yy, np.clip(height, 0.0, None)


def render(output: Path, fps: int = 12) -> None:
    for font_name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(font.name == font_name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    spacing = 0.25
    xx, yy, initial = make_steep_stockpile(spacing)
    disturbance = np.exp(-((xx / 2.3) ** 2 + ((yy + 4.5) / 0.9) ** 2))
    final, stats, _, collapse = relax_hysteretic_avalanche(
        initial,
        (spacing, spacing),
        start_angle_deg=72.0,
        stop_angle_deg=38.0,
        friction_angle_deg=38.0,
        cohesion_pa=30_000.0,
        disturbance=0.95 * disturbance,
        stochasticity=0.012,
        seed=7,
        record_history=True,
    )
    if not stats.triggered:
        raise RuntimeError("demo parameters did not trigger an avalanche")

    # Hold the intact face, show an impact pulse, animate collapse, then hold.
    sampled_indices = np.unique(
        np.linspace(0, len(collapse) - 1, min(55, len(collapse))).astype(int)
    )
    collapse_frames = [collapse[index] for index in sampled_indices]
    frames = [initial] * 18 + collapse_frames + [final] * 18
    stages = (
        ["亚稳陡坡：64°坡面保持"] * 12
        + ["铲斗冲击 / 振动累积"] * 6
        + ["连通破坏：快速塌方"] * len(collapse_frames)
        + ["停止：回落至约38°休止角"] * 18
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.5, 5.2), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    center_index = initial.shape[0] // 2

    def draw(index: int) -> None:
        height = frames[index]
        ax3d.clear()
        ax2d.clear()
        ax3d.plot_surface(
            xx, yy, height, cmap="terrain", vmin=0, vmax=10,
            linewidth=0, antialiased=True
        )
        ax3d.set(
            xlim=(-10, 10), ylim=(-10, 10), zlim=(0, 11),
            xlabel="x [m]", ylabel="y [m]", zlabel="height [m]"
        )
        ax3d.view_init(elev=24, azim=-58)
        ax3d.set_title(stages[index])

        ax2d.fill_between(
            yy[center_index], 0, height[center_index],
            color="#9b6b3e", alpha=0.82
        )
        ax2d.plot(yy[center_index], initial[center_index], "--",
                  color="#555555", lw=1.3, label="初始轮廓")
        ax2d.plot(yy[center_index], height[center_index],
                  color="#6b351d", lw=2.2, label="当前轮廓")
        if 12 <= index < 18:
            ax2d.annotate(
                "局部扰动", xy=(-4.5, 4.8), xytext=(-8.5, 7.0),
                arrowprops={"arrowstyle": "->", "color": "crimson"},
                color="crimson", fontsize=11
            )
        ax2d.set(
            xlim=(-10, 10), ylim=(0, 11), xlabel="前后方向 y [m]",
            ylabel="height [m]", title="中心剖面"
        )
        ax2d.grid(alpha=0.2)
        ax2d.legend(loc="upper right")
        fig.suptitle(
            f"滞回塌方模型 | 起滑角72°，停止角38° | "
            f"累计网格转移量 {stats.moved_volume:.1f} m³",
            fontsize=12
        )

    animation = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(fig)
    print(f"saved: {output}")
    print(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hysteretic_avalanche_demo/steep_face_collapse.gif"),
    )
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    render(args.output, args.fps)


if __name__ == "__main__":
    main()
