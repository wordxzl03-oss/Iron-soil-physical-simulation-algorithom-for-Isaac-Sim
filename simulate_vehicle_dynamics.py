"""Demonstrate wheel-loader bicycle dynamics, drive and ground resistance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon

from wheel_loader_dynamics import (
    VehicleParameters,
    WheelLoaderDynamicsEnv,
)


def control(
    time_s: float,
    speed_m_s: float,
    excavation_resistance_n: float,
    max_drive_force_n: float,
) -> np.ndarray:
    if time_s < 6.0:
        target = 1.5
        drive = np.clip(0.035 + 0.24 * (target - speed_m_s), 0.0, 0.65)
        return np.array([drive, 0.12 * np.sin(0.65 * time_s), 0.0])
    if time_s < 11.0:
        target = 0.6
        feedforward = (excavation_resistance_n + 8_000.0) / max_drive_force_n
        drive = np.clip(feedforward + 0.32 * (target - speed_m_s), 0.0, 1.0)
        return np.array([drive, 0.0, 0.0])
    if time_s < 12.5:
        return np.array([0.0, 0.0, 0.85])
    target = -1.5
    drive = np.clip(-0.035 + 0.24 * (target - speed_m_s), -0.65, 0.0)
    return np.array([drive, -0.06, 0.0])


def excavation_profile(state) -> float:
    if 6.0 <= state.time_s < 11.0 and state.speed_m_s >= 0:
        phase = np.clip((state.time_s - 6.0) / 3.0, 0.0, 1.0)
        return float(165_000.0 * phase)
    return 0.0


def run(output_dir: Path, duration_s: float = 18.0) -> None:
    parameters = VehicleParameters()
    env = WheelLoaderDynamicsEnv(
        parameters=parameters,
        dt=0.05,
        max_steps=round(duration_s / 0.05),
        excavation_function=excavation_profile,
    )
    env.reset(options={"x_m": 0.0, "y_m": -12.0})
    rows = []
    while env.state.time_s < duration_s - 1e-12:
        excavation = excavation_profile(env.state)
        action = control(
            env.state.time_s,
            env.state.speed_m_s,
            excavation,
            parameters.max_drive_force_n,
        )
        _, _, _, _, info = env.step(action)
        force = info["forces"]
        rows.append(
            [
                env.state.time_s,
                env.state.x_m,
                env.state.y_m,
                env.state.heading_rad,
                env.state.speed_m_s,
                env.state.steering_rad,
                force.applied_drive_n,
                force.rolling_n + force.grade_n + force.velocity_n,
                force.excavation_n,
                force.net_n,
                force.traction_limit_n,
                force.slip_ratio,
                env.state.energy_j,
            ]
        )
    data = np.asarray(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "vehicle_dynamics_data.npz",
        data=data,
        columns=np.asarray(
            [
                "time_s", "x_m", "y_m", "heading_rad", "speed_m_s",
                "steering_rad", "drive_n", "ground_n", "excavation_n",
                "net_n", "traction_limit_n", "slip_ratio", "energy_j",
            ]
        ),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(data[:, 1], data[:, 2], color="#1565c0", linewidth=2)
    axes[0, 0].scatter(data[0, 1], data[0, 2], label="start", color="green")
    axes[0, 0].scatter(data[-1, 1], data[-1, 2], label="end", color="red")
    axes[0, 0].set(
        xlabel="x [m]", ylabel="y [m]", title="Bicycle-model path"
    )
    axes[0, 0].axis("equal")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(data[:, 0], data[:, 4], label="speed")
    axes[0, 1].axvspan(6, 11, color="#8d6e63", alpha=0.15, label="digging")
    axes[0, 1].set(
        xlabel="time [s]", ylabel="speed [m/s]", title="Longitudinal speed"
    )
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(data[:, 0], data[:, 6] / 1000, label="drive")
    axes[1, 0].plot(data[:, 0], data[:, 7] / 1000, label="ground resistance")
    axes[1, 0].plot(data[:, 0], data[:, 8] / 1000, label="excavation")
    axes[1, 0].plot(data[:, 0], data[:, 9] / 1000, label="net")
    axes[1, 0].set(
        xlabel="time [s]", ylabel="force [kN]", title="Force balance"
    )
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    axes[1, 1].plot(data[:, 0], data[:, 11], label="slip ratio")
    axes[1, 1].plot(
        data[:, 0], data[:, 12] / 3.6e6, label="drive energy [kWh]"
    )
    axes[1, 1].set(
        xlabel="time [s]", title="Traction utilization and energy"
    )
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend()
    fig.suptitle(
        "25 t wheel loader: bicycle kinematics + longitudinal dynamics"
    )
    figure_path = output_dir / "vehicle_dynamics_diagnostics.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    sampled = np.linspace(0, len(data) - 1, 90).astype(int)
    fig_a, axis = plt.subplots(figsize=(7.2, 6.2))

    def draw(frame: int) -> None:
        axis.clear()
        index = sampled[frame]
        x, y, heading = data[index, 1], data[index, 2], data[index, 3]
        axis.plot(data[: index + 1, 1], data[: index + 1, 2], "--", color="#777")
        local = np.array(
            [[-1.25, -2.2], [1.25, -2.2], [1.25, 2.2], [-1.25, 2.2]]
        )
        rotation = np.array(
            [[np.cos(heading), np.sin(heading)],
             [-np.sin(heading), np.cos(heading)]]
        )
        body = local @ rotation.T + np.array([x, y])
        axis.add_patch(
            Polygon(body, closed=True, facecolor="#f5a000", edgecolor="black")
        )
        drive = data[index, 6] / 70_000
        resistance = -(data[index, 7] + data[index, 8]) / 70_000
        forward = np.array([np.sin(heading), np.cos(heading)])
        axis.arrow(
            x, y, *(drive * forward), color="green", width=0.05,
            length_includes_head=True,
        )
        axis.arrow(
            x, y, *(-resistance * forward), color="red", width=0.05,
            length_includes_head=True,
        )
        axis.axhspan(-1.0, 5.0, color="#8d6e63", alpha=0.10)
        axis.set(
            xlim=(-8, 8), ylim=(-14, 14), aspect="equal",
            xlabel="x [m]", ylabel="y [m]",
            title=(
                f"t={data[index, 0]:.1f}s  v={data[index, 4]:.2f}m/s  "
                f"drive={data[index, 6]/1000:.0f}kN  "
                f"resistance={-(data[index, 7]+data[index, 8])/1000:.0f}kN"
            ),
        )
        axis.grid(alpha=0.2)

    animation = FuncAnimation(fig_a, draw, frames=len(sampled))
    animation_path = output_dir / "vehicle_dynamics.gif"
    animation.save(animation_path, PillowWriter(fps=10), dpi=95)
    plt.close(fig_a)
    print(f"saved: {figure_path}")
    print(f"saved: {animation_path}")
    print(f"final_state={env.state}")
    print(f"max_speed_m_s={np.max(data[:, 4]):.3f}")
    print(f"min_speed_m_s={np.min(data[:, 4]):.3f}")
    print(f"max_slip_ratio={np.max(data[:, 11]):.4f}")
    print(f"drive_energy_kwh={data[-1, 12] / 3.6e6:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("vehicle_dynamics_demo")
    )
    parser.add_argument("--duration", type=float, default=18.0)
    args = parser.parse_args()
    run(args.output_dir, args.duration)


if __name__ == "__main__":
    main()
