"""Plot force, cut depth and cutting-edge height for one dataset episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episode", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("trajectory_diagnostics.png"))
    args = parser.parse_args()
    stem = f"episode_{args.episode:03d}"
    data = np.load(args.dataset / "data" / f"{stem}.npz")
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    record = manifest[args.episode - 1]
    time = np.arange(len(data["cutting_edge_z"])) * 0.1 / record["speed_scale"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, data["cutting_edge_z"], color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("cutting edge z [m]")
    axes[1].plot(time, data["cut_depth"], color="#8c564b", linewidth=2)
    axes[1].set_ylabel("cut depth [m]")
    axes[2].plot(time, data["resistance_n"] / 1000, color="#d62728", linewidth=2)
    axes[2].axhline(
        record["max_resistance_n"] / 1000,
        color="black", linestyle="--", label="machine limit",
    )
    limited = data["force_limited"].astype(bool)
    axes[2].scatter(
        time[limited], data["resistance_n"][limited] / 1000,
        color="#ff7f0e", s=18, label="force limited",
    )
    axes[2].set(xlabel="time [s]", ylabel="resistance [kN]")
    axes[2].legend()
    fig.suptitle(
        f"Physics-aware digging trajectory — episode {args.episode:03d}\n"
        f"load={record['loaded_volume_m3']:.2f} m^3, "
        f"peak resistance={record['peak_resistance_n']/1000:.1f} kN"
    )
    fig.savefig(args.output, dpi=170)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
