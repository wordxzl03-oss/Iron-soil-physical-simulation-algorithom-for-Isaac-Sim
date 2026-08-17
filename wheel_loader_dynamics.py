"""Low-speed bicycle and longitudinal dynamics for a wheel loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class VehicleParameters:
    mass_kg: float = 25_000.0
    wheelbase_m: float = 3.25
    max_steer_deg: float = 32.0
    max_drive_force_n: float = 220_000.0
    max_brake_force_n: float = 260_000.0
    max_power_w: float = 310_000.0
    tire_ground_friction: float = 0.82
    rolling_resistance_coefficient: float = 0.025
    velocity_resistance_n_per_mps: float = 2_200.0
    aerodynamic_drag_n_per_mps2: float = 80.0
    gravity_m_s2: float = 9.81
    max_forward_speed_m_s: float = 4.2
    max_reverse_speed_m_s: float = 2.8
    track_width_m: float = 2.84
    wheel_radius_m: float = 0.72


@dataclass
class VehicleState:
    """Heading is zero along +Y; positive heading turns toward +X."""

    x_m: float = 0.0
    y_m: float = 0.0
    heading_rad: float = 0.0
    z_m: float = 0.72
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    speed_m_s: float = 0.0
    steering_rad: float = 0.0
    distance_m: float = 0.0
    energy_j: float = 0.0
    time_s: float = 0.0


@dataclass(frozen=True)
class ForceBreakdown:
    requested_drive_n: float
    applied_drive_n: float
    brake_n: float
    rolling_n: float
    grade_n: float
    velocity_n: float
    excavation_n: float
    net_n: float
    traction_limit_n: float
    slip_ratio: float


@dataclass(frozen=True)
class TerrainPose:
    """Quasi-static chassis pose fitted to four wheel-ground contacts."""

    z_m: float
    pitch_rad: float
    roll_rad: float
    wheel_heights_m: tuple[float, float, float, float]


def fit_vehicle_to_terrain(
    state: VehicleState,
    height_function: Callable[[float, float], float],
    parameters: VehicleParameters,
) -> TerrainPose:
    """Fit chassis height, pitch and roll to FL/FR/RL/RR terrain heights."""
    forward = np.array([
        np.sin(state.heading_rad), np.cos(state.heading_rad)
    ])
    right = np.array([
        np.cos(state.heading_rad), -np.sin(state.heading_rad)
    ])
    center = np.array([state.x_m, state.y_m])
    half_wheelbase = 0.5 * parameters.wheelbase_m
    half_track = 0.5 * parameters.track_width_m
    positions = (
        center + half_wheelbase * forward - half_track * right,
        center + half_wheelbase * forward + half_track * right,
        center - half_wheelbase * forward - half_track * right,
        center - half_wheelbase * forward + half_track * right,
    )
    heights = tuple(float(height_function(p[0], p[1])) for p in positions)
    front = 0.5 * (heights[0] + heights[1])
    rear = 0.5 * (heights[2] + heights[3])
    left = 0.5 * (heights[0] + heights[2])
    right_height = 0.5 * (heights[1] + heights[3])
    pitch = float(np.arctan2(front - rear, parameters.wheelbase_m))
    roll = float(np.arctan2(right_height - left, parameters.track_width_m))
    # A four-corner height field is generally not perfectly planar. Position
    # the rigid chassis on the highest required support plane so no tire can
    # pass below its local terrain contact; lower corners may retain a small
    # clearance, representing suspension articulation.
    half_wheelbase = 0.5 * parameters.wheelbase_m
    half_track = 0.5 * parameters.track_width_m
    offsets = (
        half_wheelbase * np.tan(pitch) - half_track * np.tan(roll),
        half_wheelbase * np.tan(pitch) + half_track * np.tan(roll),
        -half_wheelbase * np.tan(pitch) - half_track * np.tan(roll),
        -half_wheelbase * np.tan(pitch) + half_track * np.tan(roll),
    )
    ground_center = max(
        height - offset for height, offset in zip(heights, offsets)
    )
    return TerrainPose(
        z_m=float(ground_center + parameters.wheel_radius_m),
        pitch_rad=pitch,
        roll_rad=roll,
        wheel_heights_m=heights,
    )


def grid_height_function(
    height: np.ndarray,
    origin_xy: tuple[float, float],
    spacing_xy: tuple[float, float],
) -> Callable[[float, float], float]:
    """Return a clamped bilinear height sampler for a regular height map."""
    field = np.asarray(height, dtype=float)
    origin_x, origin_y = origin_xy
    dx, dy = spacing_xy
    if field.ndim != 2 or dx <= 0.0 or dy <= 0.0:
        raise ValueError("height must be 2-D and spacing must be positive")

    def sample(x_m: float, y_m: float) -> float:
        fx = np.clip((x_m - origin_x) / dx, 0.0, field.shape[0] - 1.0)
        fy = np.clip((y_m - origin_y) / dy, 0.0, field.shape[1] - 1.0)
        i0, j0 = int(np.floor(fx)), int(np.floor(fy))
        i1, j1 = min(i0 + 1, field.shape[0] - 1), min(j0 + 1, field.shape[1] - 1)
        tx, ty = fx - i0, fy - j0
        return float(
            (1.0 - tx) * (1.0 - ty) * field[i0, j0]
            + tx * (1.0 - ty) * field[i1, j0]
            + (1.0 - tx) * ty * field[i0, j1]
            + tx * ty * field[i1, j1]
        )

    return sample


def _motion_sign(speed: float, requested_drive: float) -> float:
    if abs(speed) > 1e-4:
        return float(np.sign(speed))
    if abs(requested_drive) > 1e-6:
        return float(np.sign(requested_drive))
    return 0.0


def vehicle_force_balance(
    state: VehicleState,
    drive_command: float,
    brake_command: float,
    grade_rad: float,
    excavation_resistance_n: float,
    parameters: VehicleParameters,
) -> ForceBreakdown:
    """Resolve drive and resistance forces along the vehicle longitudinal axis."""
    drive_command = float(np.clip(drive_command, -1.0, 1.0))
    brake_command = float(np.clip(brake_command, 0.0, 1.0))
    normal = (
        parameters.mass_kg
        * parameters.gravity_m_s2
        * np.cos(grade_rad)
    )
    traction_limit = parameters.tire_ground_friction * normal
    requested_drive = drive_command * parameters.max_drive_force_n
    if abs(state.speed_m_s) > 0.25:
        power_limit = parameters.max_power_w / abs(state.speed_m_s)
        requested_drive = float(
            np.clip(requested_drive, -power_limit, power_limit)
        )
    applied_drive = float(
        np.clip(requested_drive, -traction_limit, traction_limit)
    )
    slip_ratio = max(
        0.0,
        (abs(requested_drive) - traction_limit)
        / max(abs(requested_drive), 1.0),
    )
    direction = _motion_sign(state.speed_m_s, requested_drive)
    brake = -direction * brake_command * parameters.max_brake_force_n
    rolling = (
        -direction * parameters.rolling_resistance_coefficient * normal
    )
    grade = (
        -parameters.mass_kg
        * parameters.gravity_m_s2
        * np.sin(grade_rad)
    )
    velocity = (
        -parameters.velocity_resistance_n_per_mps * state.speed_m_s
        -parameters.aerodynamic_drag_n_per_mps2
        * state.speed_m_s
        * abs(state.speed_m_s)
    )
    excavation = -direction * max(0.0, float(excavation_resistance_n))
    net = applied_drive + brake + rolling + grade + velocity + excavation
    return ForceBreakdown(
        requested_drive_n=requested_drive,
        applied_drive_n=applied_drive,
        brake_n=brake,
        rolling_n=rolling,
        grade_n=grade,
        velocity_n=velocity,
        excavation_n=excavation,
        net_n=net,
        traction_limit_n=traction_limit,
        slip_ratio=slip_ratio,
    )


def integrate_bicycle(
    state: VehicleState,
    drive_command: float,
    steer_command: float,
    brake_command: float,
    dt: float,
    parameters: VehicleParameters,
    *,
    grade_rad: float = 0.0,
    excavation_resistance_n: float = 0.0,
    terrain_height_function: Callable[[float, float], float] | None = None,
) -> tuple[VehicleState, ForceBreakdown]:
    """Advance a low-speed bicycle model using semi-implicit Euler."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    current_pose = (
        fit_vehicle_to_terrain(state, terrain_height_function, parameters)
        if terrain_height_function is not None else None
    )
    effective_grade = (
        current_pose.pitch_rad if current_pose is not None else grade_rad
    )
    forces = vehicle_force_balance(
        state,
        drive_command,
        brake_command,
        effective_grade,
        excavation_resistance_n,
        parameters,
    )
    acceleration = forces.net_n / parameters.mass_kg
    speed = state.speed_m_s + acceleration * dt
    speed = float(
        np.clip(
            speed,
            -parameters.max_reverse_speed_m_s,
            parameters.max_forward_speed_m_s,
        )
    )
    # Do not let resistance reverse a stationary vehicle through zero.
    if (
        state.speed_m_s * speed < 0
        and abs(drive_command) < 0.05
    ):
        speed = 0.0
    steering = float(
        np.clip(steer_command, -1.0, 1.0)
        * np.deg2rad(parameters.max_steer_deg)
    )
    yaw_rate = speed / parameters.wheelbase_m * np.tan(steering)
    heading = state.heading_rad + yaw_rate * dt
    x = state.x_m + speed * np.sin(heading) * dt
    y = state.y_m + speed * np.cos(heading) * dt
    energy = state.energy_j + max(
        0.0, forces.applied_drive_n * speed
    ) * dt
    updated = VehicleState(
        x_m=float(x),
        y_m=float(y),
        heading_rad=float(heading),
        z_m=state.z_m,
        pitch_rad=state.pitch_rad,
        roll_rad=state.roll_rad,
        speed_m_s=speed,
        steering_rad=steering,
        distance_m=state.distance_m + abs(speed) * dt,
        energy_j=float(energy),
        time_s=state.time_s + dt,
    )
    if terrain_height_function is not None:
        pose = fit_vehicle_to_terrain(
            updated, terrain_height_function, parameters
        )
        updated.z_m = pose.z_m
        updated.pitch_rad = pose.pitch_rad
        updated.roll_rad = pose.roll_rad
    return updated, forces


class WheelLoaderDynamicsEnv:
    """Dependency-free Gymnasium-style core for later RL integration.

    Action is ``[drive, steering, brake]`` with each component normalized to
    [-1, 1], [-1, 1], and [0, 1]. Observation contains
    ``[x, y, heading, speed, steering, excavation_force, slip, energy]``.
    """

    def __init__(
        self,
        parameters: VehicleParameters | None = None,
        dt: float = 0.05,
        max_steps: int = 1200,
        grade_function: Callable[[float, float], float] | None = None,
        terrain_height_function: Callable[[float, float], float] | None = None,
        excavation_function: Callable[[VehicleState], float] | None = None,
    ) -> None:
        self.parameters = parameters or VehicleParameters()
        self.dt = dt
        self.max_steps = max_steps
        self.grade_function = grade_function or (lambda _x, _y: 0.0)
        self.terrain_height_function = terrain_height_function
        self.excavation_function = excavation_function or (lambda _state: 0.0)
        self.state = VehicleState()
        self.steps = 0
        self.last_forces = vehicle_force_balance(
            self.state, 0.0, 0.0, 0.0, 0.0, self.parameters
        )

    def _observation(self, excavation_force: float) -> np.ndarray:
        return np.array(
            [
                self.state.x_m,
                self.state.y_m,
                self.state.heading_rad,
                self.state.speed_m_s,
                self.state.steering_rad,
                excavation_force,
                self.last_forces.slip_ratio,
                self.state.energy_j,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del seed  # Reserved for Gymnasium-compatible stochastic extensions.
        options = options or {}
        self.state = VehicleState(
            x_m=float(options.get("x_m", 0.0)),
            y_m=float(options.get("y_m", 0.0)),
            heading_rad=float(options.get("heading_rad", 0.0)),
            speed_m_s=float(options.get("speed_m_s", 0.0)),
        )
        if self.terrain_height_function is not None:
            pose = fit_vehicle_to_terrain(
                self.state, self.terrain_height_function, self.parameters
            )
            self.state.z_m = pose.z_m
            self.state.pitch_rad = pose.pitch_rad
            self.state.roll_rad = pose.roll_rad
        self.steps = 0
        excavation = float(self.excavation_function(self.state))
        reset_grade = (
            self.state.pitch_rad
            if self.terrain_height_function is not None
            else float(self.grade_function(self.state.x_m, self.state.y_m))
        )
        self.last_forces = vehicle_force_balance(
            self.state, 0.0, 0.0,
            reset_grade,
            excavation, self.parameters,
        )
        return self._observation(excavation), {"forces": self.last_forces}

    def step(
        self, action: np.ndarray | list[float] | tuple[float, float, float]
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_array = np.asarray(action, dtype=float)
        if action_array.shape != (3,):
            raise ValueError("action must have shape (3,)")
        grade = (
            self.state.pitch_rad
            if self.terrain_height_function is not None
            else float(self.grade_function(self.state.x_m, self.state.y_m))
        )
        excavation = float(self.excavation_function(self.state))
        previous_y = self.state.y_m
        self.state, self.last_forces = integrate_bicycle(
            self.state,
            float(action_array[0]),
            float(action_array[1]),
            float(action_array[2]),
            self.dt,
            self.parameters,
            grade_rad=grade,
            excavation_resistance_n=excavation,
            terrain_height_function=self.terrain_height_function,
        )
        self.steps += 1
        progress = self.state.y_m - previous_y
        reward = (
            progress
            - 0.03 * self.last_forces.slip_ratio
            - 2e-8 * abs(self.last_forces.applied_drive_n * self.state.speed_m_s)
        )
        terminated = False
        truncated = self.steps >= self.max_steps
        observation = self._observation(excavation)
        info = {
            "forces": self.last_forces,
            "grade_rad": grade,
            "excavation_resistance_n": excavation,
        }
        return observation, float(reward), terminated, truncated, info
