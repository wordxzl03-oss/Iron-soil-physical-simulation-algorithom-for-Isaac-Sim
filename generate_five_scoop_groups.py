"""Generate randomized groups; each group continuously performs five scoops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from render_multi_obj_loader_scooping import cut_state_offset, sample_trajectories
from render_obj_loader_scooping import (
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
    rotate_yz,
    transform_point,
)
from simulate_inertial_obj_scooping import simulate_inertial_entry
from slope_model import (
    random_pile,
    relax_critical_slope,
    relax_localized_failure_wedge,
)
from wheel_loader_dynamics import (
    VehicleParameters,
    VehicleState,
    fit_vehicle_to_terrain,
    grid_height_function,
)

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


def _front_toe(profile: np.ndarray, coordinates: np.ndarray) -> float:
    active = np.flatnonzero(profile > 0.12)
    return float(coordinates[active[0]]) if len(active) else float(coordinates[len(coordinates) // 3])


def _capacity_limited_cut(
    terrain: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    entry_x: float,
    entry_y: float,
    proposed_travel: float,
    bucket_width: float,
    max_depth: float,
    spacing: float,
    capacity_m3: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shorten penetration until the swept volume fits the bucket."""
    cut, removed = cut_state_offset(
        terrain, xx, yy, 1.0, entry_x, entry_y,
        proposed_travel, bucket_width, max_depth,
    )
    volume = float(removed.sum() * spacing**2)
    if volume <= capacity_m3:
        return cut, removed, proposed_travel
    low, high = 0.04, proposed_travel
    for _ in range(20):
        travel = 0.5 * (low + high)
        trial, trial_removed = cut_state_offset(
            terrain, xx, yy, 1.0, entry_x, entry_y,
            travel, bucket_width, max_depth,
        )
        trial_volume = float(trial_removed.sum() * spacing**2)
        if trial_volume <= capacity_m3:
            low = travel
            cut, removed = trial, trial_removed
        else:
            high = travel
    return cut, removed, low


def loader_mechanism_points(
    root: np.ndarray,
    pin: np.ndarray,
    edge: np.ndarray,
    low_angle_deg: float,
    boom_angle_deg: float,
    curl_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-length root, bucket-pin and cutting-edge points."""
    pin_low = transform_point(pin, root, low_angle_deg)
    edge_low = transform_point(edge, root, low_angle_deg)
    pin_now = transform_point(pin, root, boom_angle_deg)
    curled_edge = transform_point(edge_low, pin_low, curl_angle_deg)
    return root.copy(), pin_now, curled_edge + (pin_now - pin_low)


def sample_repeated_trajectories(
    count: int, bucket_width: float, seed: int
) -> list[dict[str, float]]:
    """Cover the pile face in shuffled five-lane cycles."""
    if count <= 8:
        return sample_trajectories(count, bucket_width, seed)
    rng = np.random.default_rng(seed)
    lane_centers = np.linspace(-1.25, 1.25, 5) * bucket_width
    records: list[dict[str, float]] = []
    while len(records) < count:
        for lateral in rng.permutation(lane_centers):
            if len(records) >= count:
                break
            records.append({
                "pass": len(records) + 1,
                "lateral_offset_m": float(
                    lateral + rng.uniform(-0.24, 0.24)
                ),
                "penetration_m": float(rng.uniform(1.15, 1.55)),
                "max_depth_m": float(rng.uniform(0.88, 1.10)),
                "curl_angle_deg": float(rng.uniform(44.0, 50.0)),
                "boom_lift_deg": float(rng.uniform(30.0, 36.0)),
            })
    return records


def simulate_group(
    group_id: int,
    seed: int,
    *,
    resolution: float = 0.25,
    bucket_capacity_m3: float = 3.0,
    scoop_count: int = 5,
) -> dict:
    rng = np.random.default_rng(seed)
    workspace = 30.0
    size = round(workspace / resolution) + 1
    coordinates = np.linspace(-workspace / 2, workspace / 2, size)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    peak_target = float(rng.uniform(8.0, 10.0))
    repose = float(rng.uniform(39.0, 46.0))
    scale = float(rng.uniform(3.0, 3.65))
    raw = random_pile(
        size, size, resolution, resolution,
        seed=int(rng.integers(0, 2**31 - 1)),
        peak_height=peak_target,
        spatial_scale=scale,
    )
    terrain, initial_stats = relax_critical_slope(
        raw, (resolution, resolution), repose
    )
    initial = terrain.copy()

    source = load_obj_parts(Path("wheel_buck.obj"))
    prepared, points = _prepare_model(source, infer_landmarks(source))
    bucket_width = float(
        np.ptp(prepared["loader_bucket_world.stl"].vertices[:, 0])
    )
    root, edge = points["root_pin"], points["cutting_edge"]
    angles = np.linspace(-18.0, 8.0, 1001)
    edge_heights = np.array(
        [transform_point(edge, root, angle)[2] for angle in angles]
    )
    low_angle = float(angles[np.argmin(np.abs(edge_heights - 0.12))])
    edge_low = transform_point(edge, root, low_angle)
    if scoop_count <= 0:
        raise ValueError("scoop_count must be positive")
    trajectories = sample_repeated_trajectories(
        scoop_count, bucket_width, seed + 17
    )

    frames = [terrain.copy()]
    frame_labels = ["初始料堆"]
    summaries = []
    dynamics = []
    for pass_index, trajectory in enumerate(trajectories, start=1):
        offset = float(trajectory["lateral_offset_m"])
        row = int(np.clip(round((offset - coordinates[0]) / resolution), 0, size - 1))
        entry_y = _front_toe(terrain[row], coordinates)
        terrain_sampler = grid_height_function(
            terrain, (coordinates[0], coordinates[0]),
            (resolution, resolution),
        )
        dynamic, vehicle_states = simulate_inertial_entry(
            float(edge_low[1]),
            entry_y,
            target_impact_speed_m_s=float(rng.uniform(1.8, 2.4)),
            post_contact_drive_command=float(rng.uniform(0.34, 0.48)),
            bucket_width_m=bucket_width,
            max_depth_m=float(trajectory["max_depth_m"]),
            nominal_penetration_m=float(trajectory["penetration_m"]),
            vehicle_x_m=offset,
            terrain_height_function=terrain_sampler,
        )
        proposed = min(
            float(dynamic[-1, 4]),
            float(trajectory["penetration_m"]) * 1.35,
        )
        cut, removed, actual_travel = _capacity_limited_cut(
            terrain, xx, yy, offset, entry_y, proposed,
            bucket_width, float(trajectory["max_depth_m"]),
            resolution, bucket_capacity_m3,
        )
        loaded = float(removed.sum() * resolution**2)
        peak_force = float(np.max(dynamic[:, 9]))
        frames.append(cut.copy())
        frame_labels.append(f"第{pass_index}铲：贯入并装料")
        failed, failure_stats, _, _, _, _ = relax_localized_failure_wedge(
            cut,
            (resolution, resolution),
            (offset, entry_y + 0.80 * actual_travel),
            origin_xy=(coordinates[0], coordinates[0]),
            collision_force_n=peak_force,
            collision_displacement_m=min(0.45, 0.35 * actual_travel),
            bucket_width_m=bucket_width,
            disturbance_length_m=2.5,
            max_propagation_radius_m=5.5,
            active_layer_depth_m=0.65,
            wedge_half_angle_deg=58.0,
            stop_angle_deg=38.0,
            friction_angle_deg=40.0,
            cohesion_pa=5_000.0,
            bulk_density=1900.0,
            record_history=False,
        )
        terrain = failed
        frames.append(terrain.copy())
        frame_labels.append(f"第{pass_index}铲：退出后局部稳定")
        contact_rows = np.flatnonzero(dynamic[:, 4] > 0.0)
        impact_speed = (
            float(dynamic[contact_rows[0], 3]) if len(contact_rows) else 0.0
        )
        summaries.append({
            "pass": pass_index,
            "lateral_offset_m": offset,
            "entry_y_m": entry_y,
            "impact_speed_m_s": impact_speed,
            "dynamic_stop_penetration_m": float(dynamic[-1, 4]),
            "capacity_limited_penetration_m": actual_travel,
            "loaded_volume_m3": loaded,
            "bucket_capacity_m3": bucket_capacity_m3,
            "fill_factor": loaded / bucket_capacity_m3,
            "peak_resistance_n": peak_force,
            "curl_angle_deg": float(trajectory["curl_angle_deg"]),
            "boom_lift_deg": float(trajectory["boom_lift_deg"]),
            "max_chassis_z_m": float(max(state.z_m for state in vehicle_states)),
            "max_abs_pitch_deg": float(np.rad2deg(max(
                abs(state.pitch_rad) for state in vehicle_states
            ))),
            "max_abs_roll_deg": float(np.rad2deg(max(
                abs(state.roll_rad) for state in vehicle_states
            ))),
            "failure_triggered": failure_stats.triggered,
            "failure_moved_volume_m3": failure_stats.moved_volume_m3,
        })
        dynamics.append(dynamic)

    return {
        "group": group_id,
        "seed": seed,
        "resolution_m": resolution,
        "repose_angle_deg": repose,
        "requested_peak_height_m": peak_target,
        "stabilized_peak_height_m": float(initial.max()),
        "initial_relaxation_converged": initial_stats.converged,
        "initial": initial,
        "final": terrain,
        "x": coordinates,
        "y": coordinates,
        "frames": np.asarray(frames),
        "frame_labels": frame_labels,
        "passes": summaries,
        "dynamics": dynamics,
    }


def render_group(group: dict, output: Path, fps: int = 2) -> None:
    frames = group["frames"]
    coordinates = group["x"]
    passes = group["passes"]
    resolution = float(group["resolution_m"])
    raw = load_obj_parts(Path("wheel_buck.obj"))
    prepared, points = _prepare_model(raw, infer_landmarks(raw))
    bucket_part = prepared["loader_bucket_world.stl"]
    root, pin, edge = (
        points["root_pin"], points["bucket_pin"], points["cutting_edge"]
    )
    angles = np.linspace(-18.0, 8.0, 1001)
    edge_heights = np.asarray(
        [transform_point(edge, root, angle)[2] for angle in angles]
    )
    low_angle = float(angles[np.argmin(np.abs(edge_heights - 0.12))])
    pin_low = transform_point(pin, root, low_angle)
    edge_low = transform_point(edge, root, low_angle)
    vehicle_parameters = VehicleParameters()
    chassis_pivot = np.array([0.0, -2.275, 0.72])
    sequence: list[dict] = [{
        "terrain": frames[0],
        "pre_terrain": frames[0],
        "pass_index": 0,
        "progress": 0.0,
        "phase": "approach",
        "label": "第1铲：接近坡脚",
        "completed": 0,
    }]
    contact_fractions = np.linspace(0.0, 1.0, 5)
    for pass_index in range(5):
        pre_terrain = frames[2 * pass_index]
        cut_terrain = frames[2 * pass_index + 1]
        stable_terrain = frames[2 * pass_index + 2]
        removed = np.maximum(pre_terrain - cut_terrain, 0.0)
        record = passes[pass_index]
        entry_y = float(record["entry_y_m"])
        travel = float(record["capacity_limited_penetration_m"])
        for fraction in contact_fractions:
            reach = entry_y + travel * float(fraction)
            cell_fraction = np.clip(
                (reach - coordinates) / resolution + 1.0, 0.0, 1.0
            )
            progressive = pre_terrain - removed * cell_fraction[None, :]
            sequence.append({
                "terrain": progressive,
                "pre_terrain": pre_terrain,
                "pass_index": pass_index,
                "progress": float(fraction),
                "phase": "cut",
                "label": (
                    f"第{pass_index + 1}铲：刃口接触与切削 "
                    f"{100 * fraction:.0f}%"
                ),
                "completed": pass_index,
            })
        sequence.append({
            "terrain": stable_terrain,
            "pre_terrain": pre_terrain,
            "pass_index": pass_index,
            "progress": 1.0,
            "phase": "stable",
            "label": f"第{pass_index + 1}铲：退出后局部稳定",
            "completed": pass_index + 1,
        })

    fig, ((ax_top, ax_side), (ax_contact, ax_change)) = plt.subplots(
        2, 2, figsize=(12.8, 8.6), constrained_layout=True,
    )

    def mechanism(
        boom_angle: float, curl_angle: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return loader_mechanism_points(
            root, pin, edge, low_angle, boom_angle, curl_angle
        )

    def pitch_point(
        point: np.ndarray, pitch_rad: float, translation: np.ndarray
    ) -> np.ndarray:
        return (
            rotate_yz(
                point[None, :], chassis_pivot, np.rad2deg(pitch_rad)
            )[0]
            + translation
        )

    def draw(index: int) -> None:
        ax_top.clear()
        ax_side.clear()
        ax_contact.clear()
        ax_change.clear()
        item = sequence[index]
        terrain = item["terrain"]
        pre_terrain = item["pre_terrain"]
        image = ax_top.imshow(
            terrain,
            origin="lower",
            extent=(coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]),
            cmap="terrain", vmin=0.0, vmax=10.0,
        )
        completed = int(item["completed"])
        for record in passes:
            color = "#d32f2f" if int(record["pass"]) == min(completed + 1, 5) else "#333333"
            ax_top.plot(
                [record["entry_y_m"], record["entry_y_m"] + record["capacity_limited_penetration_m"]],
                [record["lateral_offset_m"]] * 2,
                "--", color=color, linewidth=1.4,
            )
        pass_index = int(item["pass_index"])
        active_pass = passes[pass_index]
        phase_is_cut = item["phase"] == "cut"
        phase_is_stable = item["phase"] == "stable"
        if phase_is_cut:
            edge_target_y = (
                active_pass["entry_y_m"]
                + active_pass["capacity_limited_penetration_m"]
                * float(item["progress"])
            )
            vehicle_shift = edge_target_y - edge_low[1]
        elif phase_is_stable:
            edge_target_y = (
                active_pass["entry_y_m"]
                + active_pass["capacity_limited_penetration_m"]
            )
            vehicle_shift = edge_target_y - edge_low[1] - 3.0
        else:
            vehicle_shift = active_pass["entry_y_m"] - edge_low[1] - 2.8
        vehicle_x = active_pass["lateral_offset_m"]
        boom_angle = low_angle + (
            float(active_pass.get("boom_lift_deg", 34.0))
            if phase_is_stable else 0.0
        )
        curl_angle = (
            float(active_pass.get("curl_angle_deg", 47.0))
            if phase_is_stable else 0.0
        )
        root_local, pin_local, edge_local = mechanism(
            boom_angle, curl_angle
        )
        # Fixed OBJ geometry: vehicle shift moves every mechanism point by
        # the same amount, so root-pin and pin-edge lengths cannot change.
        mechanism_y = (
            np.asarray([root_local[1], pin_local[1], edge_local[1]])
            + vehicle_shift
        )
        mechanism_x = np.full(3, vehicle_x)
        ax_top.add_patch(Rectangle(
            (vehicle_shift - 4.20, vehicle_x - 1.30),
            3.85, 2.60, facecolor="#f5a000",
            edgecolor="#202020", linewidth=1.0, alpha=0.92,
        ))
        for wheel_y in (vehicle_shift - 1.05, vehicle_shift - 3.35):
            for wheel_x in (vehicle_x - 1.48, vehicle_x + 1.30):
                ax_top.add_patch(Rectangle(
                    (wheel_y - 0.35, wheel_x), 0.70, 0.18,
                    facecolor="#202020", edgecolor="none",
                ))
        ax_top.plot(
            mechanism_y[:2], mechanism_x[:2],
            color="#f5a000", linewidth=3.2,
        )
        ax_top.plot(
            mechanism_y[1:], mechanism_x[1:],
            color="#e67e00", linewidth=2.4,
        )
        ax_top.plot(
            [mechanism_y[2], mechanism_y[2]],
            [vehicle_x - 1.35, vehicle_x + 1.35],
            color="#e67e00", linewidth=4.0,
        )

        row = int(np.clip(
            round((vehicle_x - coordinates[0]) / resolution),
            0, terrain.shape[0] - 1,
        ))
        terrain_sampler = grid_height_function(
            terrain, (coordinates[0], coordinates[0]),
            (resolution, resolution),
        )
        vehicle_state = VehicleState(
            x_m=vehicle_x, y_m=vehicle_shift
        )
        pose = fit_vehicle_to_terrain(
            vehicle_state, terrain_sampler, vehicle_parameters
        )
        translation = np.array([
            vehicle_x, vehicle_shift, pose.z_m - 0.72
        ])
        chassis = np.array([
            [0.0, -4.20, 0.68], [0.0, -0.35, 0.68],
            [0.0, -0.35, 2.25], [0.0, -4.20, 2.25],
        ])
        chassis_world = np.asarray([
            pitch_point(point, pose.pitch_rad, translation)
            for point in chassis
        ])
        mech_world = np.asarray([
            pitch_point(point, pose.pitch_rad, translation)
            for point in (root_local, pin_local, edge_local)
        ])
        bucket_low_vertices = rotate_yz(
            bucket_part.vertices, root, low_angle
        )
        bucket_vertices = (
            rotate_yz(bucket_low_vertices, pin_low, curl_angle)
            + (pin_local - pin_low)
        )
        bucket_world = (
            rotate_yz(
                bucket_vertices, chassis_pivot,
                np.rad2deg(pose.pitch_rad),
            )
            + translation
        )
        ax_side.fill_between(
            coordinates, 0.0, terrain[row],
            color="#9b7653", alpha=0.85,
        )
        ax_side.plot(coordinates, terrain[row], color="#5d4037")
        ax_side.fill(
            chassis_world[:, 1], chassis_world[:, 2],
            color="#f5a000", edgecolor="#202020",
        )
        for wheel_local_y in (-1.05, -3.35):
            center_local = np.array([0.0, wheel_local_y, 0.74])
            center_world = pitch_point(
                center_local, pose.pitch_rad, translation
            )
            wheel = plt.Circle(
                (center_world[1], center_world[2]), 0.72,
                facecolor="#202020", edgecolor="#000000",
            )
            ax_side.add_patch(wheel)
        ax_side.plot(
            mech_world[:2, 1], mech_world[:2, 2],
            color="#f5a000", linewidth=5.0,
        )
        ax_side.plot(
            mech_world[1:, 1], mech_world[1:, 2],
            color="#e67e00", linewidth=4.0,
        )
        face_stride = max(1, len(bucket_part.faces) // 300)
        bucket_yz = bucket_world[:, 1:3]
        ax_side.add_collection(PolyCollection(
            bucket_yz[bucket_part.faces[::face_stride]],
            facecolor="#e67e00",
            edgecolor="none",
            linewidth=0.0,
            alpha=0.92,
            zorder=4,
        ))
        ax_side.scatter(
            mech_world[:, 1], mech_world[:, 2],
            s=20, color="#202020", zorder=5,
        )
        ax_side.set(
            xlim=(vehicle_shift - 6.0, vehicle_shift + 7.0),
            ylim=(0.0, 7.0), aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="高度 z [m]",
            title=(
                f"轨迹侧视 | pitch={np.rad2deg(pose.pitch_rad):.1f}°，"
                f"roll={np.rad2deg(pose.roll_rad):.1f}°"
            ),
        )
        ax_side.grid(alpha=0.2)

        pre_profile = pre_terrain[row]
        current_profile = terrain[row]
        contact_y = float(mech_world[2, 1])
        contact_z = float(mech_world[2, 2])
        ax_contact.fill_between(
            coordinates, 0.0, current_profile,
            color="#9b7653", alpha=0.88, label="当前土体",
        )
        ax_contact.plot(
            coordinates, pre_profile, "--",
            color="#6d4c41", linewidth=1.2, label="接触前坡面",
        )
        ax_contact.fill_between(
            coordinates, current_profile, pre_profile,
            where=pre_profile > current_profile + 1e-9,
            color="#ef5350", alpha=0.60, label="本铲已切除",
        )
        ax_contact.add_collection(PolyCollection(
            bucket_yz[bucket_part.faces[::face_stride]],
            facecolor="#e67e00", edgecolor="none",
            linewidth=0.0, alpha=0.94, zorder=4,
        ))
        ax_contact.scatter(
            [contact_y], [contact_z], s=34,
            color="#d32f2f", edgecolor="white", linewidth=0.7,
            zorder=6, label="刃口",
        )
        dynamics = np.asarray(group["dynamics"][pass_index], dtype=float)
        target_penetration = (
            float(active_pass["capacity_limited_penetration_m"])
            * float(item["progress"])
        )
        dynamic_index = int(np.argmin(
            np.abs(dynamics[:, 4] - target_penetration)
        ))
        speed_now = float(dynamics[dynamic_index, 3])
        resistance_now = (
            float(dynamics[dynamic_index, 9]) if phase_is_cut else 0.0
        )
        if phase_is_cut and resistance_now > 1.0:
            arrow_length = 0.45 + 0.75 * min(
                resistance_now / 230_000.0, 1.0
            )
            ax_contact.annotate(
                "",
                xy=(contact_y - arrow_length, contact_z + 0.18),
                xytext=(contact_y, contact_z + 0.18),
                arrowprops=dict(
                    arrowstyle="-|>", color="#1565c0", linewidth=2.0
                ),
            )
            ax_contact.text(
                contact_y - arrow_length,
                contact_z + 0.32,
                f"土体反力 {resistance_now / 1000:.0f} kN",
                color="#1565c0", fontsize=8,
            )
        local_pre_height = float(np.interp(
            contact_y, coordinates, pre_profile
        ))
        ax_contact.set(
            xlim=(contact_y - 2.3, contact_y + 2.3),
            ylim=(max(0.0, min(contact_z, local_pre_height) - 1.0),
                  max(2.2, max(contact_z, local_pre_height) + 1.8)),
            aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="高度 z [m]",
            title=(
                "铲斗—土体接触放大 | "
                f"贯入={target_penetration:.2f} m，"
                f"v={speed_now:.2f} m/s"
            ),
        )
        ax_contact.grid(alpha=0.2)
        ax_contact.legend(loc="upper right", fontsize=7)

        change = group["initial"] - terrain
        ax_change.imshow(
            change,
            origin="lower",
            extent=(coordinates[0], coordinates[-1], coordinates[0], coordinates[-1]),
            cmap="RdBu_r", vmin=-0.8, vmax=0.8,
        )
        for axis in (ax_top, ax_change):
            axis.set(xlabel="y [m]", ylabel="x [m]", aspect="equal")
        ax_top.set_title(item["label"])
        ax_change.set_title("累计高程变化 [m]")
        loaded = sum(
            record["loaded_volume_m3"] for record in passes[:completed]
        )
        if phase_is_cut:
            loaded += (
                float(active_pass["loaded_volume_m3"])
                * float(item["progress"])
            )
        fig.suptitle(
            f"第{group['group']:03d}组：连续5铲 | 已完成 {completed}/5 | "
            f"累计装载 {loaded:.2f} m³"
        )
        image.set_clim(0.0, 10.0)

    animation = FuncAnimation(fig, draw, frames=len(sequence))
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, PillowWriter(fps=fps), dpi=54)
    plt.close(fig)


def generate(
    count: int,
    seed: int,
    output_dir: Path,
    resolution: float,
    render_animations: bool,
) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for group_index in range(1, count + 1):
        group_seed = int(rng.integers(0, 2**31 - 1))
        group = simulate_group(
            group_index, group_seed, resolution=resolution
        )
        group_dir = output_dir / f"group_{group_index:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            group_dir / "five_scoop_data.npz",
            initial_height=group["initial"],
            final_height=group["final"],
            frames=group["frames"],
            x=group["x"],
            y=group["y"],
            dynamics=np.asarray(group["dynamics"], dtype=object),
        )
        metadata = {
            key: value for key, value in group.items()
            if key not in {"initial", "final", "frames", "x", "y", "dynamics"}
        }
        (group_dir / "summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if render_animations:
            render_group(group, group_dir / "five_scoop.gif")
        all_records.append(metadata)
        print(
            f"[{group_index:03d}/{count:03d}] "
            f"load={sum(item['loaded_volume_m3'] for item in group['passes']):.2f} m3"
        )
    summary = {
        "groups": count,
        "scoops_per_group": 5,
        "total_scoops": count * 5,
        "seed": seed,
        "resolution_m": resolution,
        "animations": render_animations,
        "records": all_records,
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {output_dir / 'dataset_summary.json'}")


def rerender_existing(
    output_dir: Path,
    start_group: int = 1,
    end_group: int | None = None,
) -> None:
    """Redraw saved groups without rerunning soil or vehicle simulation."""
    dataset = json.loads(
        (output_dir / "dataset_summary.json").read_text(encoding="utf-8")
    )
    end_group = end_group or int(dataset["groups"])
    if not 1 <= start_group <= end_group <= int(dataset["groups"]):
        raise ValueError("invalid rerender group range")
    for index in range(start_group, end_group + 1):
        metadata = dataset["records"][index - 1]
        group_dir = output_dir / f"group_{index:03d}"
        data = np.load(
            group_dir / "five_scoop_data.npz", allow_pickle=True
        )
        group = {
            **metadata,
            "initial": data["initial_height"],
            "final": data["final_height"],
            "frames": data["frames"],
            "x": data["x"],
            "y": data["y"],
            "dynamics": list(data["dynamics"]),
        }
        render_group(group, group_dir / "five_scoop.gif")
        print(f"[{index:03d}/{dataset['groups']:03d}] redrawn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20280727)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("five_scoop_groups_100")
    )
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument("--no-animations", action="store_true")
    parser.add_argument(
        "--rerender-existing", action="store_true",
        help="redraw saved GIFs with fixed OBJ linkage and side profile",
    )
    parser.add_argument("--start-group", type=int, default=1)
    parser.add_argument("--end-group", type=int)
    args = parser.parse_args()
    if args.rerender_existing:
        rerender_existing(
            args.output_dir, args.start_group, args.end_group
        )
        return
    generate(
        args.count, args.seed, args.output_dir,
        args.resolution, not args.no_animations,
    )


if __name__ == "__main__":
    main()
