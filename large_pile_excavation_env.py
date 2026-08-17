"""Continuous excavation environment for Perlin piles above 20 metres."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from animate_loader_3d import LoaderTrajectory
from dynamic_obj_scooping_env import DynamicObjScoopingTrajectoryEnv
from physics_aware_trajectory import MachineLimits, MaterialParameters
from slope_model import random_pile


class LargePileExcavationEnv(DynamicObjScoopingTrajectoryEnv):
    """One episode repeatedly scoops until less than 20% material remains."""

    def __init__(
        self,
        *,
        grid_size: int = 81,
        workspace_size_m: float = 30.0,
        observation_grid: int = 15,
        peak_height_range_m: tuple[float, float] = (20.5, 24.0),
        spatial_scale_range: tuple[float, float] = (1.55, 2.15),
        target_remaining_fraction: float = 0.20,
        max_scoop_multiplier: float = 1.8,
        record_training_data: bool = False,
        **kwargs,
    ) -> None:
        self.peak_height_range_m = peak_height_range_m
        self.spatial_scale_range = spatial_scale_range
        self.target_remaining_fraction = target_remaining_fraction
        self.max_scoop_multiplier = max_scoop_multiplier
        self.record_training_data = record_training_data
        self.episode_records: list[dict[str, Any]] = []
        self.completed_episodes: list[dict[str, Any]] = []
        self.scoop_count = 0
        self.initial_volume_m3 = 0.0
        self.remaining_volume_m3 = 0.0
        self.minimum_scoop_label = 0
        self.episode_soil_seed = 0
        super().__init__(
            grid_size=grid_size,
            workspace_size_m=workspace_size_m,
            observation_grid=observation_grid,
            **kwargs,
        )
        if not 0.0 < target_remaining_fraction < 1.0:
            raise ValueError("target remaining fraction must be between 0 and 1")

    def _sample_domain(self) -> None:
        self.episode_soil_seed = int(self.np_random.integers(0, 2**31 - 1))
        peak_height = float(self.np_random.uniform(*self.peak_height_range_m))
        spatial_scale = float(self.np_random.uniform(*self.spatial_scale_range))
        self._initial = random_pile(
            self.grid_size,
            self.grid_size,
            self.spacing,
            self.spacing,
            seed=self.episode_soil_seed,
            peak_height=peak_height,
            spatial_scale=spatial_scale,
        )
        friction_angle = float(self.np_random.uniform(36.0, 46.0))
        self._material = MaterialParameters(
            bulk_density_kg_m3=float(self.np_random.uniform(1750.0, 2250.0)),
            cohesion_pa=float(self.np_random.uniform(3000.0, 12000.0)),
            internal_friction_deg=friction_angle,
            bucket_friction=float(self.np_random.uniform(0.35, 0.55)),
            velocity_drag=float(self.np_random.uniform(0.8, 1.5)),
        )
        self._limits = MachineLimits(
            max_resistance_n=float(self.np_random.uniform(190_000.0, 285_000.0)),
            ground_clearance_m=0.12,
            max_cutting_edge_lift_m=float(self.np_random.uniform(1.0, 1.7)),
        )

    def action_to_trajectory(self, action: np.ndarray) -> LoaderTrajectory:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError("action must have shape (7,)")
        return LoaderTrajectory(
            heading_deg=self._scale(action[0], -180.0, 180.0),
            lateral_offset=self._scale(action[1], -5.0, 5.0),
            approach_distance=self._scale(action[2], 1.8, 3.2),
            travel_length=self._scale(action[3], 0.75, 1.65),
            bucket_width=self.obj_bucket_width_m,
            max_depth=self._scale(action[4], 0.50, 1.10),
            curl_angle_deg=self._scale(action[6], 44.0, 52.0),
            lift_height=self.obj_max_edge_lift_m,
            speed_scale=self._scale(action[5], 0.65, 1.20),
        )

    def _observation(self) -> np.ndarray:
        indices = np.linspace(
            0, self.grid_size - 1, self.observation_grid
        ).round().astype(int)
        coarse = self._initial[np.ix_(indices, indices)]
        peak_i, peak_j = np.unravel_index(
            np.argmax(self._initial), self._initial.shape
        )
        height_scale = max(self.peak_height_range_m[1] * 1.05, 1.0)
        terrain = np.clip(coarse / height_scale, 0.0, 1.0) * 2.0 - 1.0
        remaining_fraction = (
            self.remaining_volume_m3 / self.initial_volume_m3
            if self.initial_volume_m3 > 0 else 1.0
        )
        context = np.array(
            [
                2.0 * peak_i / (self.grid_size - 1) - 1.0,
                2.0 * peak_j / (self.grid_size - 1) - 1.0,
                np.clip(float(self._initial.max()) / height_scale, 0, 1) * 2 - 1,
                np.clip((self._material.bulk_density_kg_m3 - 1750) / 500, 0, 1)
                * 2 - 1,
                np.clip((self._material.cohesion_pa - 3000) / 9000, 0, 1)
                * 2 - 1,
                np.clip(remaining_fraction, 0, 1) * 2 - 1,
            ],
            dtype=np.float32,
        )
        return np.concatenate((terrain.ravel(), context)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.initial_volume_m3 = float(self._initial.sum() * self.spacing**2)
        self.remaining_volume_m3 = self.initial_volume_m3
        required_removal = (
            (1.0 - self.target_remaining_fraction) * self.initial_volume_m3
        )
        self.minimum_scoop_label = math.ceil(
            required_removal / self.bucket_capacity_m3
        )
        self.max_steps = max(
            self.minimum_scoop_label + 5,
            math.ceil(self.minimum_scoop_label * self.max_scoop_multiplier),
        )
        self.scoop_count = 0
        self.episode_records = []
        self._episode_done = False
        info.update(
            {
                "soil_seed": self.episode_soil_seed,
                "initial_volume_m3": self.initial_volume_m3,
                "minimum_scoop_label": self.minimum_scoop_label,
                "target_remaining_fraction": self.target_remaining_fraction,
            }
        )
        return self._observation(), info

    def step(self, action):
        self._episode_done = False
        observation_before = self._observation().copy()
        _, base_reward, _, _, info = super().step(action)
        self._initial = self.last_after_scoop.copy()
        self.remaining_volume_m3 = float(
            self._initial.sum() * self.spacing**2
        )
        self.scoop_count += 1
        remaining_fraction = self.remaining_volume_m3 / self.initial_volume_m3
        terminated = remaining_fraction < self.target_remaining_fraction
        truncated = self.scoop_count >= self.max_steps and not terminated
        load = float(info["loaded_volume_m3"])
        # A per-scoop cost turns maximizing payload into minimizing the number
        # of scoops. Full buckets receive reward near 1; empty scoops are costly.
        reward = (
            load / self.bucket_capacity_m3
            - 0.08
            + 2.0 * float(terminated)
            - 0.15 * float(load < 0.25)
            + 0.10 * base_reward / self.bucket_capacity_m3
        )
        record = {
            "scoop": self.scoop_count,
            "observation": observation_before,
            "action": np.asarray(action, dtype=np.float32).copy(),
            "reward": float(reward),
            "loaded_volume_m3": load,
            "remaining_volume_m3": self.remaining_volume_m3,
            "remaining_fraction": float(remaining_fraction),
            "entry_xy_m": info["entry_xy_m"],
            "trajectory": info["trajectory"],
            "dynamic_penetration_m": info["dynamic_penetration_m"],
            "peak_excavation_force_n": info["peak_excavation_force_n"],
        }
        if self.record_training_data:
            self.episode_records.append(record)
        info.update(
            {
                "scoop_count": self.scoop_count,
                "initial_volume_m3": self.initial_volume_m3,
                "remaining_volume_m3": self.remaining_volume_m3,
                "remaining_fraction": float(remaining_fraction),
                "minimum_scoop_label": self.minimum_scoop_label,
                "scoop_efficiency": (
                    self.minimum_scoop_label / self.scoop_count
                    if terminated else 0.0
                ),
                "soil_seed": self.episode_soil_seed,
            }
        )
        self._episode_done = terminated or truncated
        if self.record_training_data and self._episode_done:
            self.completed_episodes.append(
                {
                    "soil_seed": self.episode_soil_seed,
                    "initial_volume_m3": self.initial_volume_m3,
                    "remaining_volume_m3": self.remaining_volume_m3,
                    "remaining_fraction": float(remaining_fraction),
                    "scoop_count": self.scoop_count,
                    "minimum_scoop_label": self.minimum_scoop_label,
                    "terminated": terminated,
                    "records": self.episode_records.copy(),
                }
            )
        return self._observation(), float(reward), terminated, truncated, info
