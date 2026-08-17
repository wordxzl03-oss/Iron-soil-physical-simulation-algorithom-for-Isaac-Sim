"""Render a gallery of reproducible Perlin/fractal-noise stockpiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from slope_model import random_pile, relax_critical_slope

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def render(
    output: Path,
    seed: int = 2028,
    count: int = 4,
    resolution: float = 0.125,
    workspace_size: float = 30.0,
) -> None:
    rng = np.random.default_rng(seed)
    if resolution <= 0.0 or workspace_size <= 0.0:
        raise ValueError("resolution and workspace size must be positive")
    size = round(workspace_size / resolution) + 1
    coords = (np.arange(size) - (size - 1) / 2) * resolution
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    cols = 3
    rows = int(np.ceil(count / cols))
    fig = plt.figure(figsize=(12, 3.65 * rows), constrained_layout=True)
    for index in range(count):
        pile_seed = int(rng.integers(0, 2**31 - 1))
        peak = float(rng.uniform(7.0, 10.0))
        scale = float(rng.uniform(3.0, 3.7))
        repose = float(rng.uniform(38.0, 45.0))
        raw = random_pile(
            size, size, resolution, resolution,
            seed=pile_seed, peak_height=peak, spatial_scale=scale,
        )
        stable, stats = relax_critical_slope(
            raw, (resolution, resolution), repose
        )
        ax = fig.add_subplot(rows, cols, index + 1, projection="3d")
        ax.plot_surface(
            xx, yy, stable, cmap="terrain", linewidth=0,
            antialiased=True, rcount=size, ccount=size,
        )
        ax.set(
            xlim=(-15, 15), ylim=(-15, 15), zlim=(0, 10.5),
            title=(
                f"seed={pile_seed}\n峰高={stable.max():.1f} m，"
                f"休止角={repose:.1f}°"
            ),
            xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
        )
        ax.view_init(elev=31, azim=-56)
        ax.set_box_aspect((1, 1, 0.55))
    fig.suptitle("Perlin/分形噪声随机料堆：峰顶、脊线、坡脚与粗糙度随机化")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=145)
    plt.close(fig)
    print(f"saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("perlin_pile_gallery/perlin_pile_gallery.png"),
    )
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--resolution", type=float, default=0.125,
        help="height-map cell size [m]; use 0.25 for faster previews",
    )
    parser.add_argument("--workspace-size", type=float, default=30.0)
    args = parser.parse_args()
    render(
        args.output, args.seed, args.count,
        args.resolution, args.workspace_size,
    )


if __name__ == "__main__":
    main()
