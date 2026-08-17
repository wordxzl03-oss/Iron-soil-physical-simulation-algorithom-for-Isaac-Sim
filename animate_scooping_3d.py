"""Create a 3-D animation of clamshell scooping and soil slumping."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from slope_model import (
    gaussian_pile,
    random_pile,
    relax_critical_slope,
    scoop_ellipsoid,
)


def smoothstep(value: float) -> float:
    value = np.clip(value, 0.0, 1.0)
    return float(value * value * (3.0 - 2.0 * value))


def bucket_lines(
    center: tuple[float, float, float],
    opening: float,
    radii: tuple[float, float, float] = (0.75, 0.55, 0.62),
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return orange wireframe curves for two clamshell bucket halves."""
    cx, cy, cz = center
    rx, ry, rz = radii
    curves = []
    # Each jaw is a quarter-ellipsoid shell rotating away from the centre.
    for side in (-1.0, 1.0):
        hinge_shift = side * opening * 0.32
        for fraction in np.linspace(-1.0, 1.0, 5):
            theta = np.linspace(0.0, np.pi / 2, 24)
            x = cx + side * (rx * np.sin(theta) + hinge_shift)
            y = cy + fraction * ry * np.cos(theta)
            z = cz + rz * np.cos(theta)
            curves.append((x, y, z))
        for theta_value in np.linspace(0.0, np.pi / 2, 5):
            fraction = np.linspace(-1.0, 1.0, 24)
            x = np.full_like(fraction, cx + side * (
                rx * np.sin(theta_value) + hinge_shift
            ))
            y = cy + fraction * ry * np.cos(theta_value)
            z = np.full_like(fraction, cz + rz * np.cos(theta_value))
            curves.append((x, y, z))
    return curves


def make_animation(
    output: Path,
    angle: float = 34.0,
    fps: int = 20,
    dpi: int = 110,
    random_shape: bool = False,
    seed: int = 7,
) -> None:
    dx = dy = 0.15
    shape = (41, 41)
    initial = (
        random_pile(*shape, dx, dy, seed=seed)
        if random_shape
        else gaussian_pile(*shape, dx, dy)
    )
    # Attack near the highest point, with a small offset toward the near slope.
    peak_index = np.unravel_index(np.argmax(initial), initial.shape)
    center = (
        float(np.clip(peak_index[0] * dx - 0.35, 0.75, 5.25)),
        float(np.clip(peak_index[1] * dy, 0.60, 5.40)),
    )
    scooped, removed_volume = scoop_ellipsoid(
        initial, center, (dx, dy), radii_xy=(0.75, 0.55), max_depth=0.65
    )
    relaxed, stats = relax_critical_slope(
        scooped, (dx, dy), critical_angle_deg=angle
    )
    removed = initial - scooped

    x = np.arange(shape[0]) * dx
    y = np.arange(shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    surface_x, surface_y = yy, xx  # conventional display: x horizontal

    # (phase name, number of frames)
    phases = [
        ("Approach", 20),
        ("Descend", 22),
        ("Close and scoop", 28),
        ("Lift", 22),
        ("Soil slump", 35),
        ("Stable", 15),
    ]
    frame_map: list[tuple[str, float]] = []
    for name, count in phases:
        frame_map.extend((name, i / max(count - 1, 1)) for i in range(count))

    fig = plt.figure(figsize=(9.2, 7.2))
    axis = fig.add_subplot(111, projection="3d")
    output.parent.mkdir(parents=True, exist_ok=True)

    def draw(frame_number: int):
        phase, raw_progress = frame_map[frame_number]
        progress = smoothstep(raw_progress)
        axis.clear()

        if phase in ("Approach", "Descend"):
            height = initial
            scoop_fraction = 0.0
        elif phase == "Close and scoop":
            scoop_fraction = progress
            height = initial - removed * scoop_fraction
        elif phase == "Lift":
            scoop_fraction = 1.0
            height = scooped
        elif phase == "Soil slump":
            scoop_fraction = 1.0
            # Smooth display interpolation between the two physically computed
            # stable states; the final state is produced by the CA solver.
            height = scooped * (1.0 - progress) + relaxed * progress
        else:
            scoop_fraction = 1.0
            height = relaxed

        if phase == "Approach":
            bucket_z = 3.05 - 0.45 * progress
            opening = progress
        elif phase == "Descend":
            bucket_z = 2.60 - 1.15 * progress
            opening = 1.0
        elif phase == "Close and scoop":
            bucket_z = 1.45 - 0.12 * progress
            opening = 1.0 - progress
        elif phase == "Lift":
            bucket_z = 1.33 + 1.45 * progress
            opening = 0.0
        else:
            bucket_z = 2.78
            opening = 0.0

        axis.plot_surface(
            surface_x,
            surface_y,
            height,
            cmap="terrain",
            vmin=0.0,
            vmax=2.0,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        for bx, by, bz in bucket_lines(
            (center[1], center[0], bucket_z), opening
        ):
            axis.plot(bx, by, bz, color="#f28e2b", linewidth=2.0)

        # Show the captured soil travelling with the closed bucket.
        if phase in ("Close and scoop", "Lift", "Soil slump", "Stable"):
            mask = removed > 0.015
            carried_z = bucket_z + 0.10 + removed[mask] * 0.35
            axis.scatter(
                surface_y[mask],
                surface_x[mask],
                carried_z,
                s=7,
                c=removed[mask],
                cmap="copper",
                vmin=0,
                vmax=max(float(removed.max()), 1e-6),
                alpha=0.75,
                depthshade=True,
            )

        axis.set(
            xlim=(0, 6),
            ylim=(0, 6),
            zlim=(0, 3.5),
            xlabel="y [m]",
            ylabel="x [m]",
            zlabel="height [m]",
            title=(
                f"3-D clamshell scooping — {phase}\n"
                f"removed={removed_volume * scoop_fraction:.3f} m^3, "
                f"critical angle={angle:g} deg"
                + (f", random seed={seed}" if random_shape else "")
            ),
        )
        axis.view_init(elev=28, azim=-55 + 8 * np.sin(frame_number / 35))
        axis.set_box_aspect((1, 1, 0.58))
        return ()

    animation = FuncAnimation(
        fig, draw, frames=len(frame_map), interval=1000 / fps, blit=False
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"saved: {output}")
    print(f"frames: {len(frame_map)}, removed volume: {removed_volume:.6f} m^3")
    print(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("scooping_3d.gif"))
    parser.add_argument("--angle", type=float, default=34.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument(
        "--random", action="store_true", help="use a random asymmetric pile"
    )
    parser.add_argument("--seed", type=int, default=7, help="random pile seed")
    args = parser.parse_args()
    make_animation(
        args.output, args.angle, args.fps, args.dpi, args.random, args.seed
    )


if __name__ == "__main__":
    main()
