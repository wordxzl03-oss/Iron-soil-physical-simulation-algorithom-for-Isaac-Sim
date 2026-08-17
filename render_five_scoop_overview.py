"""Render final-terrain examples and statistics for five-scoop groups."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def render(root: Path = Path("five_scoop_groups_100")) -> None:
    summary = json.loads(
        (root / "dataset_summary.json").read_text(encoding="utf-8")
    )
    selected = np.linspace(1, summary["groups"], 12).round().astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.4), constrained_layout=True)
    for axis, group_id in zip(axes.flat, selected):
        data = np.load(root / f"group_{group_id:03d}" / "five_scoop_data.npz")
        change = data["initial_height"] - data["final_height"]
        axis.imshow(
            change, origin="lower", cmap="RdBu_r", vmin=-0.8, vmax=0.8,
            extent=(data["y"][0], data["y"][-1], data["x"][0], data["x"][-1]),
        )
        axis.set(
            title=f"第{group_id:03d}组：5铲后高程变化",
            xlabel="y [m]", ylabel="x [m]", aspect="equal",
        )
    fig.suptitle("100组连续五铲随机工况抽样总览")
    fig.savefig(root / "five_scoop_overview.png", dpi=145)
    plt.close(fig)

    passes = [
        scoop for group in summary["records"] for scoop in group["passes"]
    ]
    load = np.asarray([item["loaded_volume_m3"] for item in passes])
    speed = np.asarray([item["impact_speed_m_s"] for item in passes])
    penetration = np.asarray(
        [item["capacity_limited_penetration_m"] for item in passes]
    )
    force = np.asarray([item["peak_resistance_n"] for item in passes]) / 1000
    fig_s, axes_s = plt.subplots(2, 2, figsize=(9, 6.8), constrained_layout=True)
    for axis, values, xlabel in (
        (axes_s[0, 0], load, "每铲装载量 [m³]"),
        (axes_s[0, 1], speed, "冲料接触速度 [m/s]"),
        (axes_s[1, 0], penetration, "斗容约束后贯入 [m]"),
        (axes_s[1, 1], force, "峰值铲装阻力 [kN]"),
    ):
        axis.hist(values, bins=18, color="#f5a000", edgecolor="#333333")
        axis.axvline(values.mean(), color="#d32f2f", linestyle="--")
        axis.set(xlabel=xlabel, ylabel="铲数", title=f"均值={values.mean():.2f}")
        axis.grid(alpha=0.2)
    fig_s.suptitle("500铲动力学与装载统计")
    fig_s.savefig(root / "five_scoop_statistics.png", dpi=155)
    plt.close(fig_s)
    print(f"saved: {root / 'five_scoop_overview.png'}")
    print(f"saved: {root / 'five_scoop_statistics.png'}")


if __name__ == "__main__":
    render()
