"""Generate domain-randomized wheel-loader soil/trajectory data and GIFs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from animate_loader_3d import LoaderTrajectory, loader_bucket_lines, sample_trajectory
from slope_model import (
    max_neighbor_slope,
    random_pile,
    relax_critical_slope,
    scoop_loader_bucket,
)
from physics_aware_trajectory import (
    MachineLimits,
    MaterialParameters,
    apply_planned_cut,
    plan_resistance_aware_dig,
)


def trajectory_geometry(
    initial: np.ndarray,
    spacing: tuple[float, float],
    trajectory: LoaderTrajectory,
    workspace_size: float,
) -> tuple[
    tuple[float, float], np.ndarray, np.ndarray, np.ndarray, float
]:
    dx, dy = spacing
    peak_i, peak_j = np.unravel_index(np.argmax(initial), initial.shape)
    heading = np.deg2rad(trajectory.heading_deg)
    forward_soil = np.array([np.sin(heading), np.cos(heading)])
    transverse_soil = np.array([np.cos(heading), -np.sin(heading)])
    target = (
        np.array([peak_i * dx, peak_j * dy])
        + trajectory.lateral_offset * transverse_soil
    )
    target = np.clip(target, 0, workspace_size)

    # Trace a ray from the workspace boundary toward the pile peak and choose
    # the first contact with the toe. This prevents the loader from spawning
    # halfway up the slope.
    distances = []
    for coordinate, direction in zip(target, forward_soil):
        if direction > 1e-8:
            distances.append(coordinate / direction)
        elif direction < -1e-8:
            distances.append((workspace_size - coordinate) / (-direction))
    max_backtrack = max(0.0, min(distances) - max(dx, dy))
    ray_distance = np.linspace(max_backtrack, 0.0, 240)
    ray_points = target[None, :] - ray_distance[:, None] * forward_soil
    ray_i = np.clip(np.rint(ray_points[:, 0] / dx).astype(int), 0, initial.shape[0] - 1)
    ray_j = np.clip(np.rint(ray_points[:, 1] / dy).astype(int), 0, initial.shape[1] - 1)
    ray_height = initial[ray_i, ray_j]
    toe_threshold = max(0.35, 0.055 * float(initial.max()))
    contacts = np.flatnonzero(ray_height > toe_threshold)
    if len(contacts):
        contact_index = max(int(contacts[0]) - 1, 0)
        entry = ray_points[contact_index]
    else:
        entry = target - trajectory.travel_length * forward_soil

    # Keep the detected toe unchanged. Moving it inward to make the complete
    # bucket fit inside the height map would move the cutting edge up-slope.
    entry = np.clip(entry, 0.0, workspace_size)
    forward_plot = np.array([forward_soil[1], forward_soil[0]])
    entry_plot = np.array([entry[1], entry[0]])
    bucket_half_length = trajectory.travel_length * 0.44
    start_plot = entry_plot - bucket_half_length * forward_plot
    end_plot = (
        entry_plot
        + (trajectory.travel_length - bucket_half_length) * forward_plot
    )
    return (
        (float(entry[0]), float(entry[1])),
        start_plot,
        end_plot,
        forward_plot,
        toe_threshold,
    )


def make_path(
    start: np.ndarray,
    end: np.ndarray,
    forward: np.ndarray,
    trajectory: LoaderTrajectory,
    count: int = 61,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.zeros((count, 3))
    pitch = np.zeros(count)
    cut_fraction = np.zeros(count)
    for index, t in enumerate(np.linspace(0, 1, count)):
        if t < 0.25:
            p = t / 0.25
            xy = (
                start
                - trajectory.approach_distance * (1 - p) * forward
            )
            z = 0.10
        elif t < 0.55:
            p = (t - 0.25) / 0.30
            xy = start + (end - start) * p
            z = 0.10
            cut_fraction[index] = p
        elif t < 0.72:
            p = (t - 0.55) / 0.17
            xy = end
            z = 0.10 + 0.18 * p
            pitch[index] = trajectory.curl_angle_deg * p
            cut_fraction[index] = 1
        else:
            p = (t - 0.72) / 0.28
            xy = end - 0.20 * p * forward
            z = 0.28 + trajectory.lift_height * p
            pitch[index] = trajectory.curl_angle_deg
            cut_fraction[index] = 1
        path[index] = [xy[0], xy[1], z]
    return path, pitch, cut_fraction


def render_episode(
    output: Path,
    initial: np.ndarray,
    scooped: np.ndarray,
    final: np.ndarray,
    removed: np.ndarray,
    path: np.ndarray,
    pitch: np.ndarray,
    cut_fraction: np.ndarray,
    trajectory: LoaderTrajectory,
    angle: float,
    loaded_volume: float,
    fps: int,
    dpi: int,
    spacing: tuple[float, float],
    workspace_size: float,
    z_limit: float,
) -> None:
    dx, dy = spacing
    grid_x = np.arange(initial.shape[0]) * dx
    grid_y = np.arange(initial.shape[1]) * dy
    xx, yy = np.meshgrid(grid_x, grid_y, indexing="ij")
    frame_indices = np.linspace(0, len(path) - 1, 24).astype(int)
    view_margin = max(3.5, trajectory.bucket_width * 1.3)
    view_x_min = max(0.0, float(path[:, 0].min()) - view_margin)
    view_x_max = min(workspace_size, float(path[:, 0].max()) + view_margin)
    view_y_min = max(0.0, float(path[:, 1].min()) - view_margin)
    view_y_max = min(workspace_size, float(path[:, 1].max()) + view_margin)
    fig = plt.figure(figsize=(8.4, 6.2))
    axis = fig.add_subplot(111, projection="3d")

    def draw(frame: int):
        index = frame_indices[frame]
        fraction = cut_fraction[index]
        if fraction < 1:
            soil = initial - removed * fraction
        else:
            slump = max(0.0, (index / (len(path) - 1) - 0.82) / 0.18)
            soil = scooped * (1 - slump) + final * slump
        axis.clear()
        axis.plot_surface(
            yy,
            xx,
            soil,
            cmap="terrain",
            vmin=0,
            vmax=z_limit,
            rstride=3,
            cstride=3,
            linewidth=0,
            antialiased=False,
        )
        center = tuple(path[index])
        for bx, by, bz in loader_bucket_lines(
            center,
            float(pitch[index]),
            trajectory.heading_deg,
            length=trajectory.travel_length * 0.88,
            width=trajectory.bucket_width,
            height=trajectory.bucket_width * 0.46,
        ):
            axis.plot(bx, by, bz, color="#ff7f0e", linewidth=3.0)
        axis.scatter(
            [center[0]], [center[1]], [center[2] + 0.35],
            s=260, c="#ff7f0e", edgecolors="black", linewidths=1.4,
            depthshade=False,
        )
        axis.text(
            center[0], center[1], center[2] + 0.85,
            "BUCKET", color="black", fontsize=8,
            horizontalalignment="center",
        )
        axis.plot(
            path[:, 0], path[:, 1], path[:, 2] + 0.08,
            "--", color="#d62728", linewidth=1, alpha=0.65,
        )
        normalized_time = index / max(len(path) - 1, 1)
        if normalized_time < 0.20:
            phase_name = "ground approach"
        elif normalized_time < 0.45:
            phase_name = "level penetration"
        elif normalized_time < 0.70:
            phase_name = "lift and curl"
        elif normalized_time < 0.80:
            phase_name = "finish curl"
        else:
            phase_name = "raised reverse exit"

        # Visual payload proxy so the loading and carry phases are readable.
        if fraction > 0.55 and float(removed.max()) > 0:
            payload_mask = removed > max(0.03, 0.08 * float(removed.max()))
            payload_x = yy[payload_mask]
            payload_y = xx[payload_mask]
            if len(payload_x):
                center_x, center_y, center_z = path[index]
                axis.scatter(
                    center_x + (payload_x - payload_x.mean()) * 0.22,
                    center_y + (payload_y - payload_y.mean()) * 0.22,
                    center_z + 0.28 + removed[payload_mask] * 0.25,
                    c=removed[payload_mask],
                    cmap="copper",
                    s=8,
                    alpha=0.85,
                    depthshade=True,
                )
        axis.set(
            xlim=(view_x_min, view_x_max),
            ylim=(view_y_min, view_y_max),
            zlim=(0, min(z_limit, max(6.0, float(initial.max()) + 1.5))),
            xlabel="travel y [m]", ylabel="x [m]", zlabel="z [m]",
            title=(
                f"{phase_name}\n"
                f"load={loaded_volume:.3f} m^3, repose={angle:.1f} deg, "
                f"heading={trajectory.heading_deg:+.1f} deg"
            ),
        )
        # View from the loader/approach side toward the pile so the bucket is
        # not hidden behind the terrain surface.
        axis.view_init(elev=18, azim=trajectory.heading_deg + 180)
        axis.set_box_aspect(
            (view_x_max - view_x_min, view_y_max - view_y_min, 6.5)
        )
        return ()

    animation = FuncAnimation(fig, draw, frames=len(frame_indices), blit=False)
    animation.save(output, PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, default=Path("loader_dataset_100"))
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dpi", type=int, default=58)
    parser.add_argument("--no-animations", action="store_true")
    parser.add_argument("--workspace-size", type=float, default=30.0)
    parser.add_argument(
        "--pile-footprint",
        type=float,
        default=20.0,
        help="approximate pile footprint width and length [m]",
    )
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=0.25,
        help="height-map cell size [m]; 0.125 gives high-detail final renders",
    )
    parser.add_argument("--pile-height-min", type=float, default=8.0)
    parser.add_argument("--pile-height-max", type=float, default=10.0)
    parser.add_argument("--repose-angle-min", type=float, default=40.0)
    parser.add_argument("--repose-angle-max", type=float, default=48.0)
    parser.add_argument(
        "--machine-scale",
        type=float,
        default=2.6,
        help="linear scale applied to the compact-loader trajectory domain",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")

    data_dir = args.output_dir / "data"
    animation_dir = args.output_dir / "animations"
    data_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_animations:
        animation_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    if args.workspace_size <= 0 or args.grid_resolution <= 0:
        raise ValueError("workspace size and grid resolution must be positive")
    if args.pile_height_min <= 0 or args.pile_height_max < args.pile_height_min:
        raise ValueError("invalid pile height range")
    if not 0 < args.pile_footprint < args.workspace_size:
        raise ValueError("pile footprint must be smaller than workspace")
    dx = dy = args.grid_resolution
    grid_count = round(args.workspace_size / args.grid_resolution) + 1

    for episode in range(args.count):
        soil_seed = int(rng.integers(0, 2**31 - 1))
        peak_height = float(
            rng.uniform(args.pile_height_min, args.pile_height_max)
        )
        # random_pile's base footprint is about 5.5 m; scale it to the
        # requested pile footprint while leaving a ground approach zone.
        spatial_scale = float(
            rng.uniform(args.pile_footprint / 7.1, args.pile_footprint / 5.8)
        )
        repose_angle = float(
            rng.uniform(args.repose_angle_min, args.repose_angle_max)
        )
        # Calibrate material quantity so that the peak *after* critical-slope
        # stabilization, rather than the unconstrained raw peak, matches the
        # requested physical pile height.
        unit_shape = random_pile(
            grid_count, grid_count, dx, dy,
            soil_seed, 1.0, spatial_scale
        )
        raw_amplitude = peak_height
        for _ in range(12):
            initial, initial_stats = relax_critical_slope(
                unit_shape * raw_amplitude, (dx, dy), repose_angle
            )
            stabilized_peak = float(initial.max())
            raw_amplitude *= peak_height / max(stabilized_peak, 1e-9)
        initial, initial_stats = relax_critical_slope(
            unit_shape * raw_amplitude, (dx, dy), repose_angle
        )
        stabilized_peak = float(initial.max())
        trajectory = sample_trajectory(rng, args.machine_scale)
        entry, start, end, forward, toe_threshold = trajectory_geometry(
            initial, (dx, dy), trajectory, args.workspace_size
        )
        material = MaterialParameters(
            bulk_density_kg_m3=float(rng.uniform(1750, 2300)),
            cohesion_pa=float(rng.uniform(3000, 14000)),
            internal_friction_deg=repose_angle,
            bucket_friction=float(rng.uniform(0.35, 0.55)),
            velocity_drag=float(rng.uniform(0.8, 1.6)),
        )
        machine_limits = MachineLimits(
            max_resistance_n=float(rng.uniform(190_000, 290_000)),
            ground_clearance_m=float(rng.uniform(0.08, 0.16)),
            max_cutting_edge_lift_m=float(rng.uniform(1.0, 1.7)),
        )
        entry_plot = np.array([entry[1], entry[0]])
        plan = plan_resistance_aware_dig(
            initial,
            (dx, dy),
            entry_plot,
            forward,
            trajectory,
            material,
            machine_limits,
        )
        path = plan.path
        pitch = plan.pitch_deg
        cut_fraction = plan.cut_fraction
        scooped, loaded_volume, removed = apply_planned_cut(
            initial, (dx, dy), entry, trajectory.heading_deg,
            trajectory, plan
        )
        final, final_stats = relax_critical_slope(
            scooped, (dx, dy), repose_angle
        )
        approach_count = int(np.floor(0.20 * (len(path) - 1))) + 1
        cutting_edge = (
            path[:approach_count, :2]
            + trajectory.travel_length * 0.44 * forward[None, :]
        )
        approach_i = np.clip(
            np.rint(cutting_edge[:, 1] / dx).astype(int),
            0, initial.shape[0] - 1,
        )
        approach_j = np.clip(
            np.rint(cutting_edge[:, 0] / dy).astype(int),
            0, initial.shape[1] - 1,
        )
        max_approach_terrain = float(initial[approach_i, approach_j].max())
        approach_clear = bool(max_approach_terrain <= toe_threshold + 0.15)
        nominal_capacity = (
            trajectory.bucket_width
            * trajectory.travel_length
            * trajectory.max_depth
            * 0.65
        )
        fill_factor = loaded_volume / nominal_capacity
        conservation_error = abs(
            final_stats.volume_after - final_stats.volume_before
        )
        slope_limit = float(np.tan(np.deg2rad(repose_angle)))
        stable = max_neighbor_slope(final, (dx, dy)) <= slope_limit + 1e-6
        path_in_bounds = bool(
            np.all(
                (path[:, 0] >= -args.machine_scale)
                & (path[:, 0] <= args.workspace_size + args.machine_scale)
            )
            and np.all(
                (path[:, 1] >= -args.machine_scale)
                & (path[:, 1] <= args.workspace_size + args.machine_scale)
            )
            and np.all(
                (path[:, 2] >= 0)
                & (path[:, 2] <= args.pile_height_max + args.machine_scale)
            )
        )
        plausible_fill = bool(0.60 <= fill_factor <= 1.20)
        passed = bool(
            conservation_error < 1e-10
            and stable
            and path_in_bounds
            and approach_clear
            and plausible_fill
            and loaded_volume > 0
        )
        stem = f"episode_{episode + 1:03d}"
        np.savez_compressed(
            data_dir / f"{stem}.npz",
            initial_height=initial.astype(np.float32),
            after_scoop=scooped.astype(np.float32),
            final_height=final.astype(np.float32),
            removed_height=removed.astype(np.float32),
            bucket_path=path.astype(np.float32),
            bucket_pitch_deg=pitch.astype(np.float32),
            cut_fraction=cut_fraction.astype(np.float32),
            cutting_edge_z=plan.cutting_edge_z.astype(np.float32),
            cut_depth=plan.cut_depth.astype(np.float32),
            resistance_n=plan.resistance_n.astype(np.float32),
            speed_m_s=plan.speed_m_s.astype(np.float32),
            force_limited=plan.force_limited,
            grid_spacing_m=np.array([dx, dy], dtype=np.float32),
            workspace_size_m=np.float32(args.workspace_size),
        )
        if not args.no_animations:
            render_episode(
                animation_dir / f"{stem}.gif",
                initial, scooped, final, removed, path, pitch, cut_fraction,
                trajectory, repose_angle, loaded_volume, args.fps, args.dpi,
                (dx, dy), args.workspace_size, args.pile_height_max + 2.0,
            )
        record = {
            "episode": episode + 1,
            "soil_seed": soil_seed,
            "peak_height_m": peak_height,
            "stabilized_peak_height_m": stabilized_peak,
            "raw_amplitude_m": raw_amplitude,
            "spatial_scale": spatial_scale,
            "repose_angle_deg": repose_angle,
            **asdict(trajectory),
            "nominal_bucket_capacity_m3": nominal_capacity,
            "loaded_volume_m3": loaded_volume,
            "fill_factor": fill_factor,
            "initial_volume_m3": initial_stats.volume_after,
            "remaining_volume_m3": final_stats.volume_after,
            "volume_conservation_error_m3": conservation_error,
            "max_final_slope": final_stats.max_slope,
            "stable": stable,
            "path_in_bounds": path_in_bounds,
            "toe_height_threshold_m": toe_threshold,
            "max_approach_terrain_height_m": max_approach_terrain,
            "approach_clear": approach_clear,
            **asdict(material),
            "max_resistance_n": machine_limits.max_resistance_n,
            "ground_clearance_m": machine_limits.ground_clearance_m,
            "max_cutting_edge_lift_m": machine_limits.max_cutting_edge_lift_m,
            "peak_resistance_n": float(plan.resistance_n.max()),
            "force_limited_steps": int(plan.force_limited.sum()),
            "plausible_fill": plausible_fill,
            "validation_passed": passed,
            "data_file": f"data/{stem}.npz",
            "animation_file": (
                f"animations/{stem}.gif" if not args.no_animations else ""
            ),
        }
        records.append(record)
        print(
            f"[{episode + 1:03d}/{args.count:03d}] "
            f"load={loaded_volume:.3f} fill={fill_factor:.2f} pass={passed}",
            flush=True,
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    passed_count = sum(item["validation_passed"] for item in records)
    summary = {
        "episode_count": args.count,
        "master_seed": args.seed,
        "workspace_size_m": args.workspace_size,
        "pile_footprint_m": args.pile_footprint,
        "grid_resolution_m": args.grid_resolution,
        "pile_height_range_m": [
            args.pile_height_min, args.pile_height_max
        ],
        "repose_angle_range_deg": [
            args.repose_angle_min, args.repose_angle_max
        ],
        "peak_height_definition": "height after critical-slope stabilization",
        "machine_linear_scale": args.machine_scale,
        "validation_passed": passed_count,
        "validation_failed": args.count - passed_count,
        "loaded_volume_m3": {
            "min": min(item["loaded_volume_m3"] for item in records),
            "mean": float(np.mean([item["loaded_volume_m3"] for item in records])),
            "max": max(item["loaded_volume_m3"] for item in records),
        },
        "fill_factor": {
            "min": min(item["fill_factor"] for item in records),
            "mean": float(np.mean([item["fill_factor"] for item in records])),
            "max": max(item["fill_factor"] for item in records),
        },
        "notes": [
            "Kinematic height-map model; it does not model traction, hydraulic force, or particle dynamics.",
            "Fill-factor plausibility band is 0.60 to 1.20.",
            "Volume conservation tolerance is 1e-10 m^3.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
