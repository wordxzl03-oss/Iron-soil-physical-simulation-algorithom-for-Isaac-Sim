"""Batch-generate domain-randomized wheel-loader trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from animate_loader_3d import make_animation, sample_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--trajectory-seed", type=int, default=2026)
    parser.add_argument("--soil-seed", type=int, default=27)
    parser.add_argument("--output-dir", type=Path, default=Path("loader_domains"))
    parser.add_argument("--angle", type=float, default=34.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=90)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.trajectory_seed)
    records = []
    for index in range(args.count):
        trajectory = sample_trajectory(rng)
        output = args.output_dir / f"trajectory_{index + 1:03d}.gif"
        record = make_animation(
            output,
            args.angle,
            args.fps,
            args.dpi,
            True,
            args.soil_seed,
            trajectory,
        )
        record["file"] = output.name
        records.append(record)
    manifest = args.output_dir / "trajectories.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
