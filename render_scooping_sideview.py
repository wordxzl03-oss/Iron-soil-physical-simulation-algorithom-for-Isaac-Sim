"""Render a clear side-view GIF from a generated loader dataset episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("scooping_sideview.gif"))
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    record = json.loads((args.dataset / "manifest.json").read_text())[args.episode - 1]
    stem = f"episode_{args.episode:03d}"
    data = np.load(args.dataset / "data" / f"{stem}.npz")
    initial = data["initial_height"]
    removed = data["removed_height"]
    final = data["final_height"]
    path = data["bucket_path"]
    pitch = data["bucket_pitch_deg"]
    fraction = data["cut_fraction"]
    dx, dy = data["grid_spacing_m"]
    heading = np.deg2rad(record["heading_deg"])
    forward = np.array([np.cos(heading), np.sin(heading)])
    origin = path[0, :2] - 2.0 * forward
    distance = np.linspace(0, 18, 260)
    sample_xy = origin[None, :] + distance[:, None] * forward[None, :]

    def profile(height):
        i = np.clip(np.rint(sample_xy[:, 1] / dx).astype(int), 0, height.shape[0] - 1)
        j = np.clip(np.rint(sample_xy[:, 0] / dy).astype(int), 0, height.shape[1] - 1)
        return height[i, j]

    initial_profile = profile(initial)
    removed_profile = profile(removed)
    final_profile = profile(final)
    path_s = (path[:, :2] - origin[None, :]) @ forward
    frames = np.linspace(0, len(path) - 1, 32).astype(int)
    fig, axis = plt.subplots(figsize=(10, 5.8))

    def draw(frame_number):
        index = frames[frame_number]
        axis.clear()
        if fraction[index] < 1:
            soil = initial_profile - removed_profile * fraction[index]
        else:
            slump = np.clip((index / (len(path) - 1) - 0.82) / 0.18, 0, 1)
            soil = (
                (initial_profile - removed_profile) * (1 - slump)
                + final_profile * slump
            )
        axis.fill_between(distance, 0, soil, color="#9b7653", alpha=0.92)
        axis.plot(distance, soil, color="#5d4037", linewidth=2)
        axis.plot(path_s, path[:, 2] + 0.35, "--", color="#d62728", alpha=0.7)

        length = record["travel_length"] * 0.88
        height = record["bucket_width"] * 0.46
        local = np.array([
            [-length / 2, 0.10],
            [length / 2, 0.00],
            [length * 0.30, height],
            [-length / 2, height],
        ])
        angle = np.deg2rad(pitch[index])
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        bucket = local @ rotation.T + np.array([path_s[index], path[index, 2]])
        axis.add_patch(
            Polygon(bucket, closed=True, facecolor="#ff9f1c",
                    edgecolor="black", linewidth=2.5, alpha=0.92)
        )
        if fraction[index] > 0.48:
            payload = bucket.copy()
            center = payload.mean(axis=0)
            payload = center + (payload - center) * np.array([0.68, 0.52])
            payload[:, 1] += 0.12
            axis.add_patch(
                Polygon(payload, closed=True, facecolor="#6d4c41",
                        edgecolor="#3e2723", linewidth=1.2)
            )

        t = index / max(len(path) - 1, 1)
        phase = (
            "1. LEVEL CUT-IN" if 0.20 <= t < 0.45
            else "2. LIFT + CURL" if 0.45 <= t < 0.80
            else "3. RAISED REVERSE EXIT" if t >= 0.80
            else "GROUND APPROACH"
        )
        axis.set(
            xlim=(0, 18), ylim=(0, 11),
            xlabel="distance along approach [m]", ylabel="height [m]",
            title=(
                f"{phase}\nload={record['loaded_volume_m3']:.2f} m^3, "
                f"peak resistance={record['peak_resistance_n']/1000:.0f} kN"
            ),
        )
        axis.grid(alpha=0.25)
        axis.set_aspect("equal", adjustable="box")
        return ()

    animation = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    animation.save(args.output, PillowWriter(fps=args.fps), dpi=110)
    plt.close(fig)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
