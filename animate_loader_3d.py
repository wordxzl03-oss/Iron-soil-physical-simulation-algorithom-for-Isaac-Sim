"""Animate a wheel loader bucket penetrating, curling and lifting soil."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from slope_model import (
    gaussian_pile,
    random_pile,
    relax_critical_slope,
    scoop_loader_bucket,
)


@dataclass(frozen=True)
class LoaderTrajectory:
    heading_deg: float = 0.0
    lateral_offset: float = 0.0
    approach_distance: float = 0.85
    travel_length: float = 0.92
    bucket_width: float = 1.12
    max_depth: float = 0.42
    curl_angle_deg: float = 48.0
    lift_height: float = 1.45
    speed_scale: float = 1.0


def sample_trajectory(
    rng: np.random.Generator, linear_scale: float = 1.0
) -> LoaderTrajectory:
    """Sample one trajectory from the domain-randomization distribution."""
    if linear_scale <= 0:
        raise ValueError("linear_scale must be positive")
    return LoaderTrajectory(
        heading_deg=float(rng.uniform(-18, 18)),
        lateral_offset=float(rng.uniform(-0.38, 0.38) * linear_scale),
        approach_distance=float(rng.uniform(0.65, 1.05) * linear_scale),
        travel_length=float(rng.uniform(0.72, 1.12) * linear_scale),
        bucket_width=float(rng.uniform(0.95, 1.28) * linear_scale),
        max_depth=float(rng.uniform(0.30, 0.50) * linear_scale),
        curl_angle_deg=float(rng.uniform(36, 60)),
        lift_height=float(rng.uniform(1.10, 1.85) * linear_scale),
        speed_scale=float(rng.uniform(0.80, 1.25)),
    )


def smoothstep(value: float) -> float:
    value = np.clip(value, 0.0, 1.0)
    return float(value * value * (3 - 2 * value))


def loader_bucket_lines(
    center_xyz: tuple[float, float, float],
    pitch_deg: float,
    heading_deg: float = 0.0,
    length: float = 0.82,
    width: float = 1.12,
    height: float = 0.52,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Wireframe of an open-front loader bucket.

    Local u points forward, v spans bucket width and w points upward.
    Rotation about v depicts bucket curl.
    """
    # Side profile: cutting edge, floor, curved rear and open upper lip.
    profile = np.array(
        [
            [length / 2, 0.00],
            [-length / 2, 0.12],
            [-length / 2, height],
            [length * 0.28, height * 0.86],
        ]
    )
    angle = np.deg2rad(pitch_deg)
    heading = np.deg2rad(heading_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    cx, cy, cz = center_xyz

    def transform(u: np.ndarray, v: np.ndarray, w: np.ndarray):
        ur = u * cos_a - w * sin_a
        wr = u * sin_a + w * cos_a
        return (
            cx + ur * cos_h - v * sin_h,
            cy + ur * sin_h + v * cos_h,
            cz + wr,
        )

    lines = []
    # Two closed side plates.
    closed_profile = np.vstack([profile, profile[0]])
    for v_value in (-width / 2, width / 2):
        u, w = closed_profile[:, 0], closed_profile[:, 1]
        lines.append(transform(u, np.full_like(u, v_value), w))
    # Cross-members create the wide loader-bucket appearance.
    for u_value, w_value in profile:
        v = np.linspace(-width / 2, width / 2, 20)
        lines.append(
            transform(np.full_like(v, u_value), v, np.full_like(v, w_value))
        )
    # Floor ribs.
    for v_value in np.linspace(-width / 2, width / 2, 6):
        u = np.linspace(-length / 2, length / 2, 20)
        w = np.interp(u, [-length / 2, length / 2], [0.12, 0.0])
        lines.append(transform(u, np.full_like(u, v_value), w))
    return lines


def make_animation(
    output: Path,
    angle: float,
    fps: int,
    dpi: int,
    random_shape: bool,
    seed: int,
    trajectory: LoaderTrajectory | None = None,
) -> dict[str, float | int | bool]:
    trajectory = trajectory or LoaderTrajectory()
    dx = dy = 0.15
    shape = (41, 41)
    initial = (
        random_pile(*shape, dx, dy, seed=seed)
        if random_shape
        else gaussian_pile(*shape, dx, dy)
    )
    peak_i, peak_j = np.unravel_index(np.argmax(initial), initial.shape)
    heading = np.deg2rad(trajectory.heading_deg)
    transverse_x = np.cos(heading)
    transverse_y = -np.sin(heading)
    forward_x = np.sin(heading)
    forward_y = np.cos(heading)
    target_x = peak_i * dx + trajectory.lateral_offset * transverse_x
    target_y = peak_j * dy + trajectory.lateral_offset * transverse_y
    cut_start = (
        float(np.clip(target_x - trajectory.travel_length * 0.72 * forward_x, 0.8, 5.2)),
        float(np.clip(target_y - trajectory.travel_length * 0.72 * forward_y, 0.3, 5.0)),
    )
    scooped, removed_volume, removed = scoop_loader_bucket(
        initial,
        cut_start,
        (dx, dy),
        travel_length=trajectory.travel_length,
        bucket_width=trajectory.bucket_width,
        max_depth=trajectory.max_depth,
        heading_deg=trajectory.heading_deg,
    )
    relaxed, stats = relax_critical_slope(scooped, (dx, dy), angle)

    x = np.arange(shape[0]) * dx
    y = np.arange(shape[1]) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    plot_x, plot_y = yy, xx
    base_phases = [
        ("Approach", 24),
        ("Penetrate", 32),
        ("Curl bucket", 25),
        ("Lift", 24),
        ("Soil slump", 35),
        ("Stable", 15),
    ]
    phases = [
        (name, max(8, round(count / trajectory.speed_scale)))
        for name, count in base_phases
    ]
    frames = [
        (name, index / max(count - 1, 1))
        for name, count in phases
        for index in range(count)
    ]
    fig = plt.figure(figsize=(9.2, 7.2))
    axis = fig.add_subplot(111, projection="3d")
    output.parent.mkdir(parents=True, exist_ok=True)

    entry_plot = np.array([cut_start[1], cut_start[0]])
    forward_plot = np.array([forward_y, forward_x])
    start_plot = entry_plot - trajectory.approach_distance * forward_plot
    end_plot = entry_plot + trajectory.travel_length * 0.58 * forward_plot

    def draw(frame_index: int):
        phase, raw = frames[frame_index]
        p = smoothstep(raw)
        axis.clear()
        cut_fraction = 0.0
        pitch = 0.0
        bucket_z = 0.10
        bucket_xy = start_plot.copy()

        if phase == "Approach":
            bucket_xy = start_plot - 0.55 * (1 - p) * forward_plot
            soil = initial
        elif phase == "Penetrate":
            bucket_xy = start_plot + (end_plot - start_plot) * p
            cut_fraction = p
            soil = initial - removed * p
        elif phase == "Curl bucket":
            bucket_xy = end_plot
            cut_fraction = 1.0
            pitch = trajectory.curl_angle_deg * p
            bucket_z = 0.10 + 0.18 * p
            soil = scooped
        elif phase == "Lift":
            bucket_xy = end_plot - 0.20 * p * forward_plot
            cut_fraction = 1.0
            pitch = trajectory.curl_angle_deg
            bucket_z = 0.28 + trajectory.lift_height * p
            soil = scooped
        elif phase == "Soil slump":
            bucket_xy = end_plot - 0.20 * forward_plot
            cut_fraction = 1.0
            pitch = trajectory.curl_angle_deg
            bucket_z = 0.28 + trajectory.lift_height
            soil = scooped * (1 - p) + relaxed * p
        else:
            bucket_xy = end_plot - 0.20 * forward_plot
            cut_fraction = 1.0
            pitch = trajectory.curl_angle_deg
            bucket_z = 0.28 + trajectory.lift_height
            soil = relaxed

        axis.plot_surface(
            plot_x,
            plot_y,
            soil,
            cmap="terrain",
            vmin=0,
            vmax=2,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        for bx, by, bz in loader_bucket_lines(
            (bucket_xy[0], bucket_xy[1], bucket_z),
            pitch,
            trajectory.heading_deg,
            length=trajectory.travel_length * 0.88,
            width=trajectory.bucket_width,
            height=trajectory.bucket_width * 0.46,
        ):
            axis.plot(bx, by, bz, color="#f28e2b", linewidth=2.1)

        if phase in ("Curl bucket", "Lift", "Soil slump", "Stable"):
            mask = removed > 0.02
            axis.scatter(
                bucket_xy[0] + (plot_x[mask] - plot_x[mask].mean()) * 0.55,
                bucket_xy[1] + (plot_y[mask] - plot_y[mask].mean()) * 0.72,
                bucket_z + 0.20 + removed[mask] * 0.40,
                c=removed[mask],
                cmap="copper",
                s=7,
                alpha=0.78,
            )

        axis.set(
            xlim=(0, 6),
            ylim=(0, 6),
            zlim=(0, 3.5),
            xlabel="travel y [m]",
            ylabel="x [m]",
            zlabel="height [m]",
            title=(
                f"Wheel-loader bucket — {phase}\n"
                f"loaded={removed_volume * cut_fraction:.3f} m^3, "
                f"critical angle={angle:g} deg"
                + (f", random seed={seed}" if random_shape else "")
                + f", heading={trajectory.heading_deg:+.1f} deg"
            ),
        )
        axis.view_init(elev=27, azim=-61 + 7 * np.sin(frame_index / 38))
        axis.set_box_aspect((1, 1, 0.58))
        return ()

    animation = FuncAnimation(
        fig, draw, frames=len(frames), interval=1000 / fps, blit=False
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"saved: {output}")
    print(f"frames: {len(frames)}, loaded volume: {removed_volume:.6f} m^3")
    print(stats)
    result = asdict(trajectory)
    result.update(
        {
            "soil_seed": seed,
            "random_shape": random_shape,
            "loaded_volume_m3": removed_volume,
            "relaxation_iterations": stats.iterations,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("loader_scooping_3d.gif"))
    parser.add_argument("--angle", type=float, default=34.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument(
        "--domain-randomize",
        action="store_true",
        help="sample the loader trajectory parameters",
    )
    parser.add_argument("--trajectory-seed", type=int, default=0)
    parser.add_argument(
        "--save-parameters", type=Path, help="write sampled parameters to JSON"
    )
    args = parser.parse_args()
    trajectory = (
        sample_trajectory(np.random.default_rng(args.trajectory_seed))
        if args.domain_randomize
        else LoaderTrajectory()
    )
    result = make_animation(
        args.output,
        args.angle,
        args.fps,
        args.dpi,
        args.random,
        args.seed,
        trajectory,
    )
    if args.save_parameters:
        args.save_parameters.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
