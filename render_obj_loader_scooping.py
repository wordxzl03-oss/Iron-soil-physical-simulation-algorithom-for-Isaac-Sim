"""Animate scooping with the three-part wheel_buck.obj loader geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from physics_aware_trajectory import MaterialParameters, excavation_resistance
from render_scooping_avalanche import _cut_state
from slope_model import relax_localized_failure_wedge


@dataclass(frozen=True)
class MeshPart:
    name: str
    vertices: np.ndarray
    faces: np.ndarray


def load_obj_parts(path: Path) -> dict[str, MeshPart]:
    """Read OBJ object groups while preserving their shared world coordinates."""
    vertices: list[list[float]] = []
    object_faces: dict[str, list[list[int]]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("o "):
            current = raw[2:].strip()
            object_faces.setdefault(current, [])
        elif raw.startswith("v "):
            vertices.append([float(value) for value in raw.split()[1:4]])
        elif raw.startswith("f ") and current is not None:
            face = [int(token.split("/")[0]) - 1 for token in raw.split()[1:]]
            if len(face) == 3:
                object_faces[current].append(face)
    all_vertices = np.asarray(vertices, dtype=float)
    result: dict[str, MeshPart] = {}
    for name, global_faces_list in object_faces.items():
        global_faces = np.asarray(global_faces_list, dtype=int)
        used, inverse = np.unique(global_faces.ravel(), return_inverse=True)
        result[name] = MeshPart(
            name=name,
            vertices=all_vertices[used],
            faces=inverse.reshape(global_faces.shape),
        )
    return result


def rotate_yz(points: np.ndarray, pivot: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate world points in the longitudinal/vertical YZ plane."""
    angle = np.deg2rad(angle_deg)
    result = np.asarray(points, dtype=float).copy()
    rel_y = result[:, 1] - pivot[1]
    rel_z = result[:, 2] - pivot[2]
    result[:, 1] = pivot[1] + np.cos(angle) * rel_y - np.sin(angle) * rel_z
    result[:, 2] = pivot[2] + np.sin(angle) * rel_y + np.cos(angle) * rel_z
    return result


def transform_point(point: np.ndarray, pivot: np.ndarray, angle_deg: float) -> np.ndarray:
    return rotate_yz(np.asarray(point, dtype=float)[None, :], pivot, angle_deg)[0]


def make_flatter_stockpile(
    resolution: float = 0.25,
    slope_angle_deg: float = 42.0,
    peak_height: float = 10.0,
    half_footprint: float = 11.2,
    half_workspace: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 10 m pile with roughly 42-degree faces and clear ground."""
    if not 30.0 <= slope_angle_deg <= 55.0:
        raise ValueError("slope_angle_deg must be between 30 and 55")
    coords = np.arange(
        -half_workspace, half_workspace + 0.5 * resolution, resolution
    )
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    slope = np.tan(np.deg2rad(slope_angle_deg))
    front = np.maximum(0.0, (yy + half_footprint) * slope)
    rear = np.maximum(0.0, (half_footprint - yy) * slope)
    sides = np.maximum(0.0, (half_footprint - np.abs(xx)) * slope)
    height = np.minimum.reduce(
        (front, rear, sides, np.full_like(xx, peak_height))
    )
    return xx, yy, np.clip(height, 0.0, None)


def make_payload_particles(
    bucket_pin: np.ndarray,
    bucket_width: float,
    count: int = 180,
    seed: int = 2027,
) -> np.ndarray:
    """Generate a reproducible wedge-shaped granular bed inside the bucket."""
    rng = np.random.default_rng(seed)
    u = rng.random(count)
    points = np.empty((count, 3), dtype=float)
    points[:, 0] = rng.uniform(-0.43 * bucket_width, 0.43 * bucket_width, count)
    points[:, 1] = bucket_pin[1] + 0.25 + 1.35 * u
    bed = 0.18 + 0.42 * (1.0 - u)
    points[:, 2] = bucket_pin[2] - 0.48 + rng.random(count) * bed
    return points


def ballistic_particle(
    release_position: np.ndarray,
    age_s: float,
    velocity: np.ndarray,
    ground_height: float,
) -> tuple[np.ndarray, bool]:
    """Advance a spilled particle and clamp it to the terrain surface."""
    position = np.asarray(release_position, dtype=float) + age_s * velocity
    position[2] -= 0.5 * 9.81 * age_s**2
    landed = bool(position[2] <= ground_height + 0.04)
    if landed:
        position[2] = ground_height + 0.04
    return position, landed


def partition_payload_volume(
    loaded_volume: float, particle_count: int, spilled_count: int
) -> tuple[float, float]:
    """Return retained and spilled volume represented by equal-volume points."""
    if loaded_volume < 0 or particle_count <= 0:
        raise ValueError("invalid payload volume or particle count")
    if not 0 <= spilled_count <= particle_count:
        raise ValueError("spilled_count must lie within particle_count")
    spilled = loaded_volume * spilled_count / particle_count
    return loaded_volume - spilled, spilled


def infer_landmarks(parts: dict[str, MeshPart]) -> dict[str, np.ndarray | float]:
    """Infer center plane, ground and the two pins from adjacent mesh bounds."""
    boom = parts["loader_boom_world.stl"].vertices
    bucket = parts["loader_bucket_world.stl"].vertices
    frame = parts["loader_frame_world.stl"].vertices
    center_x = float(np.median(np.r_[boom[:, 0], bucket[:, 0], frame[:, 0]]))
    ground_z = float(frame[:, 2].min())
    # Nearest component surfaces identify cylindrical pin neighborhoods.
    def nearest_midpoint(a: np.ndarray, b: np.ndarray, count: int = 40) -> np.ndarray:
        a = np.unique(np.round(a, decimals=9), axis=0)
        b = np.unique(np.round(b, decimals=9), axis=0)
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        index = np.argmin(squared, axis=1)
        distance = np.sqrt(squared[np.arange(len(a)), index])
        selected = np.argsort(distance)[:count]
        midpoint = 0.5 * (a[selected] + b[index[selected]])
        return np.array([center_x, np.median(midpoint[:, 1]), np.median(midpoint[:, 2])])

    root = nearest_midpoint(boom, frame)
    bucket_pin = nearest_midpoint(boom, bucket)
    front = bucket[bucket[:, 1] <= bucket[:, 1].min() + 0.02]
    cutting_edge = np.array(
        [center_x, float(np.median(front[:, 1])), float(np.median(front[:, 2]))]
    )
    return {
        "center_x": center_x,
        "ground_z": ground_z,
        "root_pin": root,
        "bucket_pin": bucket_pin,
        "cutting_edge": cutting_edge,
    }


def _prepare_model(
    parts: dict[str, MeshPart], landmarks: dict[str, np.ndarray | float]
) -> tuple[dict[str, MeshPart], dict[str, np.ndarray]]:
    """Mirror the model to face +Y, center X, and put ground at Z=0."""
    center_x = float(landmarks["center_x"])
    ground_z = float(landmarks["ground_z"])
    prepared = {}
    for name, part in parts.items():
        vertices = part.vertices.copy()
        vertices[:, 0] -= center_x
        vertices[:, 1] *= -1.0
        vertices[:, 2] -= ground_z
        prepared[name] = MeshPart(name, vertices, part.faces)

    points = {}
    for key in ("root_pin", "bucket_pin", "cutting_edge"):
        point = np.asarray(landmarks[key], dtype=float).copy()
        point[0] -= center_x
        point[1] *= -1.0
        point[2] -= ground_z
        points[key] = point
    return prepared, points


def render(
    obj_path: Path, output: Path, fps: int = 12, pile_angle_deg: float = 42.0
) -> None:
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(font.name == name for font in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    raw_parts = load_obj_parts(obj_path)
    required = {
        "loader_frame_world.stl",
        "loader_boom_world.stl",
        "loader_bucket_world.stl",
    }
    missing = required - raw_parts.keys()
    if missing:
        raise ValueError(f"OBJ is missing required objects: {sorted(missing)}")
    parts, points = _prepare_model(raw_parts, infer_landmarks(raw_parts))
    frame_base = parts["loader_frame_world.stl"]
    boom_base = parts["loader_boom_world.stl"]
    bucket_base = parts["loader_bucket_world.stl"]
    root = points["root_pin"]
    pin = points["bucket_pin"]
    edge = points["cutting_edge"]

    # Lower the boom until the real mesh cutting edge is 0.12 m above ground.
    candidates = np.linspace(-18.0, 8.0, 2001)
    edge_heights = np.array(
        [transform_point(edge, root, angle)[2] for angle in candidates]
    )
    low_angle = float(candidates[np.argmin(np.abs(edge_heights - 0.12))])
    low_pin = transform_point(pin, root, low_angle)
    low_edge = transform_point(edge, root, low_angle)

    spacing = 0.25
    xx, yy, initial = make_flatter_stockpile(
        spacing, slope_angle_deg=pile_angle_deg
    )
    entry_y, penetration = -11.15, 1.35
    bucket_width = float(np.ptp(bucket_base.vertices[:, 0]))
    max_depth = 1.05
    base_shift = entry_y - low_edge[1]

    states: list[tuple[float, float, float]] = []  # machine shift, boom angle, curl
    terrains: list[np.ndarray] = []
    phases: list[str] = []
    forces: list[float] = []
    removed = np.zeros_like(initial)
    approach_n, cut_n, curl_n, lift_n = 10, 18, 18, 20
    retreat_n, slump_n, exit_n = 12, 24, 10
    material = MaterialParameters(
        bulk_density_kg_m3=1900.0, cohesion_pa=12_000.0,
        internal_friction_deg=42.0, bucket_friction=0.45, velocity_drag=0.8,
    )

    for p in np.linspace(0, 1, approach_n, endpoint=False):
        states.append((base_shift - 3.0 * (1.0 - p), low_angle, 0.0))
        terrains.append(initial)
        phases.append("1. 实际整车模型沿地面接近")
        forces.append(0.0)

    for p in np.linspace(0, 1, cut_n):
        terrain, removed = _cut_state(
            initial, xx, yy, p, entry_y, penetration,
            bucket_width, max_depth,
        )
        states.append((base_shift + penetration * p, low_angle, 0.0))
        terrains.append(terrain)
        phases.append("2. 整车推进，真实铲斗保持低位")
        load = float(removed.sum() * spacing**2)
        forces.append(
            excavation_resistance(max_depth * p, bucket_width, 0.65, load, material)
        )

    scooped = terrains[-1]
    collision_xy = (0.0, entry_y + 0.80 * penetration)
    collapsed, avalanche, disturbance, safety_factor, failure_wedge, history = (
        relax_localized_failure_wedge(
            scooped,
            (spacing, spacing),
            collision_xy,
            origin_xy=(float(xx[0, 0]), float(yy[0, 0])),
            collision_force_n=max(forces),
            collision_displacement_m=0.45,
            bucket_width_m=bucket_width,
            disturbance_length_m=2.5,
            max_propagation_radius_m=5.5,
            active_layer_depth_m=0.65,
            wedge_half_angle_deg=58.0,
            stop_angle_deg=38.0,
            friction_angle_deg=40.0,
            cohesion_pa=5_000.0,
            bulk_density=1900.0,
            record_history=True,
        )
    )
    # Curl and lift clear of the cut face first. Main collapse is delayed until
    # the mechanism is above the falling-material zone; otherwise a height-map
    # avalanche can unrealistically refill the volume occupied by the low boom.
    slump_history_indices = np.linspace(0, len(history) - 1, slump_n).astype(int)
    slump_terrains = [history[index] for index in slump_history_indices]
    final_shift = base_shift + penetration

    # Curl about the actual boom-to-bucket pin. Boom and vehicle stay fixed.
    for p in np.linspace(0, 1, curl_n):
        states.append((final_shift, low_angle, 48.0 * p))
        terrains.append(scooped)
        phases.append("3. 动臂不动，铲斗绕真实斗销收斗")
        forces.append(forces[-1] * (1.0 - 0.8 * p))

    # Raise the actual boom about its root pin. The linkage keeps bucket attitude.
    high_angle = low_angle + 34.0
    for p in np.linspace(0, 1, lift_n):
        angle = low_angle + (high_angle - low_angle) * p
        states.append((final_shift, angle, 48.0))
        terrains.append(scooped)
        phases.append("4. 铲斗已收拢，动臂绕根销举升")
        forces.append(forces[-1] * (1.0 - p))

    safe_shift = final_shift - 3.0
    for p in np.linspace(0, 1, retreat_n):
        states.append((final_shift - 3.0 * p, high_angle, 48.0))
        terrains.append(scooped)
        phases.append("5. 举斗后先退出塌方危险区")
        forces.append(0.0)

    for terrain in slump_terrains:
        states.append((safe_shift, high_angle, 48.0))
        terrains.append(terrain)
        phases.append("6. 机构离开后，受扰前坡发生塌方")
        forces.append(0.0)

    for p in np.linspace(0, 1, exit_n):
        states.append((safe_shift - 0.8 * p, high_angle, 48.0))
        terrains.append(collapsed)
        phases.append("7. 保持举升姿态，继续倒车退出")
        forces.append(0.0)

    def posed_parts(state: tuple[float, float, float]) -> dict[str, np.ndarray]:
        shift, boom_angle, curl_angle = state
        root_now = root + np.array([0.0, shift, 0.0])
        frame = frame_base.vertices + np.array([0.0, shift, 0.0])
        boom_unshifted = rotate_yz(boom_base.vertices, root, boom_angle)
        boom = boom_unshifted + np.array([0.0, shift, 0.0])
        pin_now_unshifted = transform_point(pin, root, boom_angle)

        # The bucket follows the boom pin. Its attitude after curl is maintained
        # during lifting by compensating the boom rotation (simplified linkage).
        bucket_low = rotate_yz(bucket_base.vertices, root, low_angle)
        pin_low = transform_point(pin, root, low_angle)
        curled = rotate_yz(bucket_low, pin_low, curl_angle)
        bucket = curled + (pin_now_unshifted - pin_low) + np.array([0.0, shift, 0.0])
        return {"frame": frame, "boom": boom, "bucket": bucket, "root": root_now,
                "pin": pin_now_unshifted + np.array([0.0, shift, 0.0])}

    posed = [posed_parts(state) for state in states]

    def pose_bucket_points(
        base_points: np.ndarray, state: tuple[float, float, float]
    ) -> np.ndarray:
        shift, boom_angle, curl_angle = state
        pin_now = transform_point(pin, root, boom_angle)
        points_low = rotate_yz(base_points, root, low_angle)
        pin_low = transform_point(pin, root, low_angle)
        curled = rotate_yz(points_low, pin_low, curl_angle)
        return curled + (pin_now - pin_low) + np.array([0.0, shift, 0.0])

    edge_index = int(np.argmin(
        (bucket_base.vertices[:, 1] - edge[1]) ** 2
        + (bucket_base.vertices[:, 2] - edge[2]) ** 2
    ))
    edge_trace = np.array([pose["bucket"][edge_index, 1:3] for pose in posed])
    payload_base = make_payload_particles(pin, bucket_width)
    payload_count = len(payload_base)
    cut_start = approach_n
    cut_end = cut_start + cut_n
    curl_end = cut_end + curl_n
    spill_count = max(1, round(0.10 * payload_count))
    spill_ids = np.arange(payload_count - spill_count, payload_count)
    spill_release_frames = np.linspace(
        curl_end + 2, curl_end + lift_n + retreat_n - 2, spill_count
    ).astype(int)
    release_by_particle = dict(
        zip(spill_ids.astype(int), spill_release_frames.astype(int))
    )
    release_positions = {
        particle: pose_bucket_points(
            payload_base[particle : particle + 1], states[frame]
        )[0]
        for particle, frame in release_by_particle.items()
    }
    spill_rng = np.random.default_rng(91)
    spill_velocities = {
        particle: np.array([
            spill_rng.uniform(-0.12, 0.12),
            spill_rng.uniform(-0.35, -0.08),
            spill_rng.uniform(0.05, 0.35),
        ])
        for particle in release_by_particle
    }

    def sample_ground(terrain: np.ndarray, x: float, y: float) -> float:
        coordinate_min = float(xx[0, 0])
        ix = int(np.clip(round((x - coordinate_min) / spacing), 0, terrain.shape[0] - 1))
        iy = int(np.clip(round((y - coordinate_min) / spacing), 0, terrain.shape[1] - 1))
        return float(terrain[ix, iy])

    particle_frames: list[tuple[np.ndarray, np.ndarray]] = []
    for frame_index, (terrain, state) in enumerate(zip(terrains, states)):
        if frame_index < cut_start:
            filled_count = 0
        elif frame_index < cut_end:
            filled_count = round(
                payload_count * (frame_index - cut_start + 1) / cut_n
            )
        else:
            filled_count = payload_count
        attached = pose_bucket_points(payload_base, state)
        attached_mask = np.arange(payload_count) < filled_count
        spilled: list[np.ndarray] = []
        for particle, release_frame in release_by_particle.items():
            if frame_index < release_frame or particle >= filled_count:
                continue
            attached_mask[particle] = False
            age = (frame_index - release_frame) / fps
            release = release_positions[particle]
            trial = release + age * spill_velocities[particle]
            ground = sample_ground(terrain, float(trial[0]), float(trial[1]))
            position, _ = ballistic_particle(
                release, age, spill_velocities[particle], ground
            )
            spilled.append(position)
        particle_frames.append(
            (
                attached[attached_mask],
                np.asarray(spilled, dtype=float).reshape((-1, 3)),
            )
        )

    def maximum_nonbucket_penetration() -> float:
        maximum = 0.0
        coordinate_min = float(xx[0, 0])
        for terrain, pose in zip(terrains, posed):
            for key in ("frame", "boom"):
                vertices = pose[key]
                ix = np.rint((vertices[:, 0] - coordinate_min) / spacing).astype(int)
                iy = np.rint((vertices[:, 1] - coordinate_min) / spacing).astype(int)
                inside = (
                    (ix >= 0) & (ix < terrain.shape[0])
                    & (iy >= 0) & (iy < terrain.shape[1])
                )
                if np.any(inside):
                    depth = terrain[ix[inside], iy[inside]] - vertices[inside, 2]
                    maximum = max(maximum, float(depth.max(initial=0.0)))
        return maximum

    nonbucket_penetration = maximum_nonbucket_penetration()
    if nonbucket_penetration > 0.03:
        raise RuntimeError(
            f"frame/boom penetrates terrain by {nonbucket_penetration:.3f} m"
        )
    loaded_volume = float(removed.sum() * spacing**2)
    retained_volume, spilled_volume = partition_payload_volume(
        loaded_volume, payload_count, spill_count
    )
    peak_force = max(forces)

    fig = plt.figure(figsize=(12.4, 5.5), constrained_layout=True)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    colors = {"frame": "#263238", "boom": "#f5a000", "bucket": "#e67e00"}
    center = initial.shape[0] // 2
    failure_i, failure_j = np.where(failure_wedge)

    def draw(index: int) -> None:
        ax3d.clear()
        ax2d.clear()
        terrain = terrains[index]
        pose = posed[index]
        attached_particles, spilled_particles = particle_frames[index]
        ax3d.plot_surface(
            xx, yy, terrain, cmap="terrain", vmin=0, vmax=10,
            linewidth=0, antialiased=True, alpha=0.92,
        )
        if index >= cut_end and len(failure_i):
            ax3d.scatter(
                xx[failure_i, failure_j], yy[failure_i, failure_j],
                terrain[failure_i, failure_j] + 0.05,
                s=5, c="#d32f2f", alpha=0.42, depthshade=False,
            )
        for key, part in (
            ("frame", frame_base), ("boom", boom_base), ("bucket", bucket_base)
        ):
            vertices = pose[key]
            faces = part.faces[::2] if len(part.faces) > 500 else part.faces
            collection = Poly3DCollection(
                vertices[faces], facecolor=colors[key], edgecolor="#202020",
                linewidth=0.12, alpha=0.96,
            )
            ax3d.add_collection3d(collection)
        if len(attached_particles):
            ax3d.scatter(
                attached_particles[:, 0], attached_particles[:, 1],
                attached_particles[:, 2], s=7, c="#5d4037",
                depthshade=True, alpha=0.92,
            )
        if len(spilled_particles):
            ax3d.scatter(
                spilled_particles[:, 0], spilled_particles[:, 1],
                spilled_particles[:, 2], s=9, c="#8d6e63",
                depthshade=True, alpha=0.95,
            )
        ax3d.set(
            xlim=(-15, 15), ylim=(-16, 13), zlim=(0, 11),
            xlabel="横向 x [m]", ylabel="进铲方向 y [m]", zlabel="高度 [m]",
            title=phases[index],
        )
        ax3d.view_init(elev=23, azim=-60)

        profile = terrain[center]
        ax2d.fill_between(yy[center], 0, profile, color="#9b7653", alpha=0.88)
        ax2d.plot(yy[center], profile, color="#5d4037", linewidth=1.8)
        local_failure = failure_wedge[center]
        if index >= cut_end and np.any(local_failure):
            ax2d.scatter(
                yy[center, local_failure], profile[local_failure] + 0.05,
                s=11, color="#d32f2f", alpha=0.65,
                label="局部滑裂楔体",
            )
        for key, part in (
            ("frame", frame_base), ("boom", boom_base), ("bucket", bucket_base)
        ):
            yz = pose[key][:, 1:3]
            faces = part.faces[::3] if len(part.faces) > 500 else part.faces
            ax2d.add_collection(
                PolyCollection(
                    yz[faces], facecolor=colors[key], edgecolor="#202020",
                    linewidth=0.18, alpha=0.90,
                )
            )
        ax2d.plot(
            edge_trace[: index + 1, 0], edge_trace[: index + 1, 1],
            "--", color="#d62728", linewidth=1.5, label="真实刃口轨迹",
        )
        ax2d.scatter(
            [pose["root"][1], pose["pin"][1]],
            [pose["root"][2], pose["pin"][2]],
            c=["#1565c0", "#00acc1"], s=28, zorder=8,
            edgecolor="white", linewidth=0.6, label="根销 / 斗销",
        )
        if len(attached_particles):
            ax2d.scatter(
                attached_particles[:, 1], attached_particles[:, 2],
                s=7, color="#5d4037", alpha=0.86, zorder=7,
                label="斗内物料",
            )
        if len(spilled_particles):
            ax2d.scatter(
                spilled_particles[:, 1], spilled_particles[:, 2],
                s=9, color="#8d6e63", alpha=0.95, zorder=7,
                label="洒落物料",
            )
        ax2d.set(
            xlim=(-16, 13), ylim=(0, 11), aspect="equal",
            xlabel="进铲方向 y [m]", ylabel="高度 [m]",
            title=f"OBJ中心剖面 | 当前阻力 {forces[index] / 1000:.0f} kN",
        )
        ax2d.grid(alpha=0.2)
        ax2d.legend(loc="upper right")
        fig.suptitle(
            f"wheel_buck.obj 缓坡铲装（坡面约{pile_angle_deg:g}°） | "
            f"装入 {loaded_volume:.2f} / 保留 {retained_volume:.2f} / "
            f"洒落 {spilled_volume:.2f} m³ | "
            f"局部失稳半径 {avalanche.max_propagation_distance_m:.1f} m",
            fontsize=12,
        )

    animation = FuncAnimation(fig, draw, frames=len(states), interval=1000 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    diagnostics_path = output.with_name("local_failure_diagnostics.npz")
    np.savez_compressed(
        diagnostics_path,
        disturbance=disturbance,
        factor_of_safety=safety_factor,
        failure_wedge=failure_wedge,
        scooped_height=scooped,
        final_height=collapsed,
        collision_xy_m=np.asarray(collision_xy),
        grid_spacing_m=np.asarray((spacing, spacing)),
        origin_xy_m=np.asarray((xx[0, 0], yy[0, 0])),
    )
    diagnostics_figure = output.with_name("local_failure_diagnostics.png")
    fig_d, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    fields = (
        (disturbance, "局部扰动场", "magma"),
        (np.clip(safety_factor, 0, 3), "Mohr-Coulomb安全系数", "viridis"),
        (scooped - collapsed, "局部滑裂楔体高程变化 [m]", "coolwarm"),
    )
    extent = (float(yy.min()), float(yy.max()), float(xx.min()), float(xx.max()))
    for axis, (field, title, cmap) in zip(axes, fields):
        image = axis.imshow(field, origin="lower", extent=extent, cmap=cmap)
        axis.contour(
            yy, xx, failure_wedge.astype(float), levels=[0.5],
            colors="red", linewidths=1.0,
        )
        axis.scatter([collision_xy[1]], [collision_xy[0]], marker="x", c="white", s=35)
        axis.set(title=title, xlabel="y [m]", ylabel="x [m]")
        fig_d.colorbar(image, ax=axis, shrink=0.8)
    fig_d.savefig(diagnostics_figure, dpi=150)
    plt.close(fig_d)
    print(f"saved: {output}")
    print(f"saved: {diagnostics_path}")
    print(f"saved: {diagnostics_figure}")
    print(f"boom_root_pin={root}")
    print(f"bucket_pin={pin}")
    print(f"cutting_edge={edge}")
    print(f"low_boom_angle_deg={low_angle:.3f}")
    print(f"loaded_volume_m3={loaded_volume:.6f}")
    print(f"retained_volume_m3={retained_volume:.6f}")
    print(f"spilled_volume_m3={spilled_volume:.6f}")
    print(f"peak_resistance_n={peak_force:.1f}")
    print(f"max_frame_boom_terrain_penetration_m={nonbucket_penetration:.6f}")
    print(avalanche)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", type=Path, default=Path("wheel_buck.obj"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("obj_loader_scooping_demo/wheel_loader_scooping.gif"),
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--pile-angle", type=float, default=42.0, help="pile face angle [deg]"
    )
    args = parser.parse_args()
    render(args.obj, args.output, args.fps, args.pile_angle)


if __name__ == "__main__":
    main()
