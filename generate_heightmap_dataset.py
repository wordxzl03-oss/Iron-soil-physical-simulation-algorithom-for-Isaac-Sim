"""Generate compact Minslope height-map transitions for downstream testing.

Each sample contains a stable procedural pile, a randomized loader-bucket cut,
and the stable pile after critical-slope relaxation.  The resulting files are
intended as a simple, deterministic interchange format for another simulator
or a learned state-transition model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slope_model import (
    max_neighbor_slope,
    random_pile,
    relax_critical_slope,
    scoop_loader_bucket,
)


SCHEMA_VERSION = "minslope-heightmap-v1"
ACTION_FIELDS = [
    "entry_x_m",
    "entry_y_m",
    "travel_length_m",
    "bucket_width_m",
    "max_depth_m",
    "heading_deg",
]


def _stable_pile(
    grid_size: int,
    spacing_m: float,
    seed: int,
    target_peak_m: float,
    spatial_scale: float,
    repose_angle_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a randomized pile whose relaxed peak matches the target."""
    unit_shape = random_pile(
        grid_size,
        grid_size,
        spacing_m,
        spacing_m,
        seed=seed,
        peak_height=1.0,
        spatial_scale=spatial_scale,
    )
    raw_amplitude = target_peak_m
    stats = None
    stable = unit_shape
    for _ in range(12):
        stable, stats = relax_critical_slope(
            unit_shape * raw_amplitude,
            (spacing_m, spacing_m),
            critical_angle_deg=repose_angle_deg,
        )
        peak = float(stable.max())
        relative_error = abs(peak - target_peak_m) / target_peak_m
        if relative_error <= 1e-4:
            break
        raw_amplitude *= target_peak_m / max(peak, 1e-12)
    # If the final calibration iteration updated the amplitude, make sure the
    # returned terrain and diagnostics correspond to that updated value.
    if abs(float(stable.max()) - target_peak_m) / target_peak_m > 1e-4:
        stable, stats = relax_critical_slope(
            unit_shape * raw_amplitude,
            (spacing_m, spacing_m),
            critical_angle_deg=repose_angle_deg,
        )
    assert stats is not None
    return stable, {
        "raw_amplitude_m": raw_amplitude,
        "stabilized_peak_height_m": float(stable.max()),
        "initial_relaxation_iterations": stats.iterations,
        "initial_relaxation_converged": stats.converged,
    }


def _sample_bucket_action(
    rng: np.random.Generator,
    height: np.ndarray,
    spacing_m: float,
) -> dict[str, float]:
    """Aim a randomized bucket pass through the neighbourhood of the summit."""
    peak_i, peak_j = np.unravel_index(np.argmax(height), height.shape)
    peak_xy = np.array([peak_i * spacing_m, peak_j * spacing_m])
    heading_deg = float(rng.uniform(-180.0, 180.0))
    heading = np.deg2rad(heading_deg)
    forward = np.array([np.sin(heading), np.cos(heading)])
    lateral = np.array([np.cos(heading), -np.sin(heading)])
    travel_length_m = float(rng.uniform(0.75, 1.15))
    bucket_width_m = float(rng.uniform(0.65, 1.05))
    depth_cap_m = max(0.03, min(0.55, 0.42 * float(height.max())))
    depth_floor_m = min(0.22, 0.55 * depth_cap_m)
    max_depth_m = float(rng.uniform(depth_floor_m, depth_cap_m))
    lateral_offset_m = float(rng.uniform(-0.22, 0.22))
    entry_xy = (
        peak_xy
        - rng.uniform(0.55, 0.78) * travel_length_m * forward
        + lateral_offset_m * lateral
    )
    workspace_max = (np.asarray(height.shape) - 1) * spacing_m
    entry_xy = np.clip(entry_xy, 0.0, workspace_max)
    return {
        "entry_x_m": float(entry_xy[0]),
        "entry_y_m": float(entry_xy[1]),
        "travel_length_m": travel_length_m,
        "bucket_width_m": bucket_width_m,
        "max_depth_m": max_depth_m,
        "heading_deg": heading_deg,
    }


def _save_preview(
    output: Path,
    initial: np.ndarray,
    final: np.ndarray,
    delta: np.ndarray,
    spacing_m: float,
) -> None:
    extent = (
        0.0,
        (initial.shape[1] - 1) * spacing_m,
        0.0,
        (initial.shape[0] - 1) * spacing_m,
    )
    limit = float(max(initial.max(), final.max()))
    delta_limit = float(max(abs(delta.min()), abs(delta.max()), 1e-9))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    panels = (
        (initial, "Initial stable pile", "terrain", 0.0, limit, "height [m]"),
        (final, "Stable pile after scoop", "terrain", 0.0, limit, "height [m]"),
        (delta, "Height change", "RdBu_r", -delta_limit, delta_limit, "delta [m]"),
    )
    for axis, (data, title, cmap, vmin, vmax, label) in zip(axes, panels):
        image = axis.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.set(title=title, xlabel="y [m]", ylabel="x [m]")
        fig.colorbar(image, ax=axis, label=label, shrink=0.82)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def generate_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.grid_size < 9:
        raise ValueError("--grid-size must be at least 9")
    if args.grid_spacing <= 0:
        raise ValueError("--grid-spacing must be positive")
    if not 0 < args.repose_angle_min <= args.repose_angle_max < 90:
        raise ValueError("invalid repose-angle range")
    if not 0 < args.peak_height_min <= args.peak_height_max:
        raise ValueError("invalid peak-height range")
    if args.bulk_density <= 0:
        raise ValueError("--bulk-density must be positive")

    output_dir: Path = args.output_dir
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    grid = np.arange(args.grid_size, dtype=np.float64) * args.grid_spacing
    cell_area_m2 = args.grid_spacing**2

    initial_maps: list[np.ndarray] = []
    post_cut_maps: list[np.ndarray] = []
    final_maps: list[np.ndarray] = []
    removed_maps: list[np.ndarray] = []
    delta_maps: list[np.ndarray] = []
    actions: list[list[float]] = []
    episode_seeds: list[int] = []
    repose_angles: list[float] = []
    records: list[dict[str, Any]] = []

    for index in range(args.count):
        episode_seed = int(rng.integers(0, 2**31 - 1))
        target_peak_m = float(
            rng.uniform(args.peak_height_min, args.peak_height_max)
        )
        repose_angle_deg = float(
            rng.uniform(args.repose_angle_min, args.repose_angle_max)
        )
        spatial_scale = float(rng.uniform(0.82, 1.12))
        initial, pile_info = _stable_pile(
            args.grid_size,
            args.grid_spacing,
            episode_seed,
            target_peak_m,
            spatial_scale,
            repose_angle_deg,
        )
        action = _sample_bucket_action(rng, initial, args.grid_spacing)
        post_cut, removed_volume_m3, removed = scoop_loader_bucket(
            initial,
            (action["entry_x_m"], action["entry_y_m"]),
            (args.grid_spacing, args.grid_spacing),
            travel_length=action["travel_length_m"],
            bucket_width=action["bucket_width_m"],
            max_depth=action["max_depth_m"],
            heading_deg=action["heading_deg"],
        )
        final, final_stats = relax_critical_slope(
            post_cut,
            (args.grid_spacing, args.grid_spacing),
            critical_angle_deg=repose_angle_deg,
        )
        delta = final - initial

        initial_volume_m3 = float(initial.sum() * cell_area_m2)
        final_volume_m3 = float(final.sum() * cell_area_m2)
        expected_final_volume_m3 = initial_volume_m3 - removed_volume_m3
        conservation_error_m3 = abs(final_volume_m3 - expected_final_volume_m3)
        slope_limit = float(np.tan(np.deg2rad(repose_angle_deg)))
        max_final_slope = max_neighbor_slope(
            final, (args.grid_spacing, args.grid_spacing)
        )
        valid = bool(
            pile_info["initial_relaxation_converged"]
            and final_stats.converged
            and removed_volume_m3 > 0.0
            and conservation_error_m3 <= 1e-9
            and max_final_slope <= slope_limit + 1e-6
            and np.all(np.isfinite(final))
            and float(final.min()) >= -1e-10
        )
        if not valid:
            raise RuntimeError(f"sample {index:04d} failed validation")

        sample_name = f"sample_{index:04d}"
        sample_path = samples_dir / f"{sample_name}.npz"
        np.savez_compressed(
            sample_path,
            initial_heightmap_m=initial.astype(np.float32),
            post_cut_heightmap_m=post_cut.astype(np.float32),
            final_heightmap_m=final.astype(np.float32),
            removed_heightmap_m=removed.astype(np.float32),
            delta_heightmap_m=delta.astype(np.float32),
            grid_x_m=grid.astype(np.float32),
            grid_y_m=grid.astype(np.float32),
            grid_spacing_m=np.float32(args.grid_spacing),
            origin_xy_m=np.array([0.0, 0.0], dtype=np.float32),
            action=np.asarray([action[field] for field in ACTION_FIELDS], dtype=np.float32),
            repose_angle_deg=np.float32(repose_angle_deg),
            bulk_density_kg_m3=np.float32(args.bulk_density),
            removed_volume_m3=np.float32(removed_volume_m3),
            removed_mass_kg=np.float32(removed_volume_m3 * args.bulk_density),
        )
        if index == 0:
            np.savetxt(
                output_dir / "sample_0000_initial_heightmap_m.csv",
                initial,
                delimiter=",",
                fmt="%.7f",
            )
            np.savetxt(
                output_dir / "sample_0000_final_heightmap_m.csv",
                final,
                delimiter=",",
                fmt="%.7f",
            )
            _save_preview(
                output_dir / "sample_0000_preview.png",
                initial,
                final,
                delta,
                args.grid_spacing,
            )

        initial_maps.append(initial.astype(np.float32))
        post_cut_maps.append(post_cut.astype(np.float32))
        final_maps.append(final.astype(np.float32))
        removed_maps.append(removed.astype(np.float32))
        delta_maps.append(delta.astype(np.float32))
        actions.append([action[field] for field in ACTION_FIELDS])
        episode_seeds.append(episode_seed)
        repose_angles.append(repose_angle_deg)
        records.append(
            {
                "sample_id": index,
                "file": f"samples/{sample_name}.npz",
                "seed": episode_seed,
                "target_peak_height_m": target_peak_m,
                "spatial_scale": spatial_scale,
                "repose_angle_deg": repose_angle_deg,
                **pile_info,
                "action": action,
                "initial_volume_m3": initial_volume_m3,
                "removed_volume_m3": removed_volume_m3,
                "removed_mass_kg": removed_volume_m3 * args.bulk_density,
                "final_volume_m3": final_volume_m3,
                "conservation_error_m3": conservation_error_m3,
                "max_final_slope": max_final_slope,
                "validation_passed": valid,
            }
        )
        print(
            f"[{index + 1:04d}/{args.count:04d}] "
            f"peak={initial.max():.3f} m removed={removed_volume_m3:.4f} m^3",
            flush=True,
        )

    np.savez_compressed(
        output_dir / "heightmap_dataset.npz",
        initial_heightmaps_m=np.stack(initial_maps),
        post_cut_heightmaps_m=np.stack(post_cut_maps),
        final_heightmaps_m=np.stack(final_maps),
        removed_heightmaps_m=np.stack(removed_maps),
        delta_heightmaps_m=np.stack(delta_maps),
        actions=np.asarray(actions, dtype=np.float32),
        action_fields=np.asarray(ACTION_FIELDS),
        seeds=np.asarray(episode_seeds, dtype=np.int64),
        repose_angles_deg=np.asarray(repose_angles, dtype=np.float32),
        grid_x_m=grid.astype(np.float32),
        grid_y_m=grid.astype(np.float32),
        grid_spacing_m=np.float32(args.grid_spacing),
        origin_xy_m=np.array([0.0, 0.0], dtype=np.float32),
        bulk_density_kg_m3=np.float32(args.bulk_density),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": "generate_heightmap_dataset.py",
        "description": "Minslope reduced-order height-map scoop transitions",
        "sample_count": args.count,
        "master_seed": args.seed,
        "shape": [args.grid_size, args.grid_size],
        "dtype": "float32",
        "height_unit": "m",
        "horizontal_unit": "m",
        "grid_spacing_m": [args.grid_spacing, args.grid_spacing],
        "origin_xy_m": [0.0, 0.0],
        "axis_convention": "heightmap[i,j] is z at x=grid_x[i], y=grid_y[j]",
        "action_fields": ACTION_FIELDS,
        "bulk_density_kg_m3": args.bulk_density,
        "bulk_density_note": "Used only to convert removed volume to mass; it does not alter Minslope dynamics.",
        "all_samples_valid": all(item["validation_passed"] for item in records),
        "samples": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        """# Minslope height-map test dataset

`heightmap_dataset.npz` contains all samples. Individual files are under
`samples/`. Heights and horizontal coordinates are in metres.

For every sample:

- `initial_heightmap_m`: stable pile before the bucket action;
- `post_cut_heightmap_m`: pile immediately after the kinematic cut;
- `final_heightmap_m`: pile after critical-slope relaxation;
- `removed_heightmap_m`: material captured by the cut;
- `delta_heightmap_m`: `final - initial`;
- `action`: values ordered according to `action_fields`.

Coordinate convention: `heightmap[i, j]` is the z height at
`x = grid_x_m[i]`, `y = grid_y_m[j]`. Array axis 0 is x and axis 1 is y.
The cell volume represented by a height value is `height * dx * dy`.

The two CSV files and PNG are human-readable exports of sample 0. Refer to
`manifest.json` for parameters and validation results. This is a reduced-order
height-field model, not DEM or calibrated physical ground truth.
""",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("heightmap_test_dataset")
    )
    parser.add_argument("--grid-size", type=int, default=61)
    parser.add_argument("--grid-spacing", type=float, default=0.10)
    parser.add_argument("--peak-height-min", type=float, default=1.2)
    parser.add_argument("--peak-height-max", type=float, default=2.2)
    parser.add_argument("--repose-angle-min", type=float, default=30.0)
    parser.add_argument("--repose-angle-max", type=float, default=40.0)
    parser.add_argument("--bulk-density", type=float, default=1800.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = generate_dataset(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "sample_count": manifest["sample_count"],
                "shape": manifest["shape"],
                "all_samples_valid": manifest["all_samples_valid"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
