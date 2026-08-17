"""High-level RL task evaluated through four-wheel vehicle dynamics."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from generate_loader_dataset import trajectory_geometry
from obj_scooping_env import ObjScoopingTrajectoryEnv
from simulate_inertial_obj_scooping import simulate_inertial_entry
from slope_model import scoop_loader_bucket
from wheel_loader_dynamics import VehicleParameters, grid_height_function


class DynamicObjScoopingTrajectoryEnv(ObjScoopingTrajectoryEnv):
    """Choose a scoop trajectory, then execute it with the vehicle model.

    The policy remains a high-level trajectory policy, but its reward uses the
    penetration actually reached by longitudinal dynamics under excavation
    resistance. Chassis Z, pitch and roll come from four wheel contacts.
    """

    def __init__(
        self,
        *,
        obj_path: str | Path = "simple_wheel_loader.obj",
        **kwargs,
    ) -> None:
        super().__init__(obj_path=obj_path, **kwargs)
        # Match the wheel centers in simple_wheel_loader.obj:
        # front y=-1.05, rear y=-3.35, left/right x=+-1.42.
        self.vehicle_parameters = VehicleParameters(
            wheelbase_m=2.30,
            track_width_m=2.84,
            wheel_radius_m=0.72,
        )
        self.last_dynamics = np.empty((0, 15), dtype=float)
        self.last_vehicle_states = []
        self.last_removed = np.zeros_like(self._initial)
        self.last_after_scoop = self._initial.copy()

    def step(self, action: np.ndarray):
        if self._episode_done:
            raise RuntimeError("episode is done; call reset() before step()")
        trajectory = self.action_to_trajectory(action)
        entry, _start, _end, _forward, _toe = trajectory_geometry(
            self._initial,
            (self.spacing, self.spacing),
            trajectory,
            self.workspace_size_m,
        )
        terrain_sampler = grid_height_function(
            self._initial,
            (0.0, 0.0),
            (self.spacing, self.spacing),
        )
        heading = np.deg2rad(trajectory.heading_deg)
        # simulate_inertial_entry operates along +Y. Rotate only the terrain
        # sampling frame; the resulting travel distance is heading invariant.
        forward = np.array([np.sin(heading), np.cos(heading)])
        right = np.array([np.cos(heading), -np.sin(heading)])

        def local_terrain(x_local: float, y_local: float) -> float:
            point = np.asarray(entry) + x_local * right + y_local * forward
            return terrain_sampler(float(point[0]), float(point[1]))

        target_speed = 1.25 + 1.05 * (
            (trajectory.speed_scale - 0.65) / (1.20 - 0.65)
        )
        dynamics, states = simulate_inertial_entry(
            self.obj_points["cutting_edge"][1],
            0.0,
            target_impact_speed_m_s=float(target_speed),
            post_contact_drive_command=float(
                np.clip(0.30 + 0.20 * trajectory.speed_scale, 0.35, 0.55)
            ),
            bucket_width_m=trajectory.bucket_width,
            max_depth_m=trajectory.max_depth,
            nominal_penetration_m=trajectory.travel_length,
            vehicle_x_m=0.0,
            terrain_height_function=local_terrain,
            vehicle_parameters=self.vehicle_parameters,
            axle_center_offset_y=-2.20,
        )
        reached = min(
            trajectory.travel_length,
            max(0.05, float(dynamics[-1, 4])),
        )
        after, raw_volume, removed = scoop_loader_bucket(
            self._initial,
            entry,
            (self.spacing, self.spacing),
            travel_length=reached,
            bucket_width=trajectory.bucket_width,
            max_depth=trajectory.max_depth,
            heading_deg=trajectory.heading_deg,
        )
        loaded = min(raw_volume, self.bucket_capacity_m3)
        if raw_volume > self.bucket_capacity_m3:
            scale = self.bucket_capacity_m3 / raw_volume
            removed *= scale
            after = self._initial - removed
        # The current dynamics table records forces but not slip history.
        # Requested drive stays below the configured traction limit here.
        max_slip = 0.0
        max_pitch = float(max(abs(state.pitch_rad) for state in states))
        max_roll = float(max(abs(state.roll_rad) for state in states))
        energy = float(states[-1].energy_j if states else 0.0)
        reward = (
            loaded
            - 0.10 * max_slip
            - 0.08 * max(0.0, np.rad2deg(max_pitch) - 12.0) ** 2 / 100.0
            - 0.08 * max(0.0, np.rad2deg(max_roll) - 10.0) ** 2 / 100.0
            - 2e-8 * energy
        )
        self.last_dynamics = dynamics
        self.last_vehicle_states = states
        self.last_removed = removed
        self.last_after_scoop = after
        self._episode_done = True
        info = {
            "loaded_volume_m3": float(loaded),
            "raw_swept_volume_m3": float(raw_volume),
            "bucket_capacity_m3": self.bucket_capacity_m3,
            "fill_factor": float(loaded / self.bucket_capacity_m3),
            "requested_penetration_m": trajectory.travel_length,
            "dynamic_penetration_m": float(reached),
            "impact_speed_m_s": float(np.max(dynamics[:, 3])),
            "peak_excavation_force_n": float(np.max(dynamics[:, 9])),
            "max_pitch_deg": float(np.rad2deg(max_pitch)),
            "max_roll_deg": float(np.rad2deg(max_roll)),
            "drive_energy_j": energy,
            "trajectory": asdict(trajectory),
            "entry_xy_m": tuple(float(value) for value in entry),
            "obj_path": str(self.obj_path),
            "vehicle_wheelbase_m": self.vehicle_parameters.wheelbase_m,
        }
        return self._observation(), float(reward), True, False, info
