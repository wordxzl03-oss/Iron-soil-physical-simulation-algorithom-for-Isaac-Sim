"""Gymnasium environment for learning a high-yield wheel-loader scoop."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from animate_loader_3d import LoaderTrajectory
from generate_loader_dataset import trajectory_geometry
from physics_aware_trajectory import (
    MachineLimits,
    MaterialParameters,
    apply_planned_cut,
    plan_resistance_aware_dig,
)
from slope_model import random_pile


class ScoopingTrajectoryEnv(gym.Env[np.ndarray, np.ndarray]):
    """One-scoop contextual control task.

    The policy observes a coarse height map and chooses a complete digging
    trajectory.  The action components, all normalized to ``[-1, 1]``, are:

    ``heading, lateral offset, approach distance, penetration length,
    cutting depth, speed, lift height``.

    One episode represents one scoop. This deliberately starts at the
    trajectory-planning level; the existing resistance-aware planner converts
    the action into a collision-limited bucket path.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        grid_size: int = 61,
        workspace_size_m: float = 15.0,
        observation_grid: int = 15,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if grid_size < 21 or observation_grid < 5:
            raise ValueError("grid_size >= 21 and observation_grid >= 5 required")
        self.grid_size = grid_size
        self.workspace_size_m = float(workspace_size_m)
        self.observation_grid = observation_grid
        self.spacing = self.workspace_size_m / (grid_size - 1)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        # Coarse normalized terrain plus peak x/y, peak height, material
        # density, cohesion and force limit.
        observation_size = observation_grid**2 + 6
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(observation_size,), dtype=np.float32
        )
        self._initial = np.zeros((grid_size, grid_size), dtype=np.float64)
        self._material = MaterialParameters()
        self._limits = MachineLimits()
        self._episode_done = False
        self._last_result: dict[str, Any] | None = None
        self.np_random = np.random.default_rng(seed)

    def _sample_domain(self) -> None:
        soil_seed = int(self.np_random.integers(0, 2**31 - 1))
        peak_height = float(self.np_random.uniform(4.0, 6.5))
        spatial_scale = float(self.np_random.uniform(1.35, 2.05))
        self._initial = random_pile(
            self.grid_size,
            self.grid_size,
            self.spacing,
            self.spacing,
            seed=soil_seed,
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

    def _observation(self) -> np.ndarray:
        indices = np.linspace(
            0, self.grid_size - 1, self.observation_grid
        ).round().astype(int)
        coarse = self._initial[np.ix_(indices, indices)]
        peak_i, peak_j = np.unravel_index(np.argmax(self._initial), self._initial.shape)
        peak = max(float(self._initial.max()), 1e-6)
        terrain = np.clip(coarse / 7.0, 0.0, 1.0) * 2.0 - 1.0
        context = np.array(
            [
                2.0 * peak_i / (self.grid_size - 1) - 1.0,
                2.0 * peak_j / (self.grid_size - 1) - 1.0,
                np.clip(peak / 7.0, 0.0, 1.0) * 2.0 - 1.0,
                np.clip((self._material.bulk_density_kg_m3 - 1750.0) / 500.0, 0, 1)
                * 2.0
                - 1.0,
                np.clip((self._material.cohesion_pa - 3000.0) / 9000.0, 0, 1)
                * 2.0
                - 1.0,
                np.clip((self._limits.max_resistance_n - 190_000.0) / 95_000.0, 0, 1)
                * 2.0
                - 1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate((terrain.ravel(), context)).astype(np.float32)

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        return low + 0.5 * (float(np.clip(value, -1.0, 1.0)) + 1.0) * (high - low)

    def action_to_trajectory(self, action: np.ndarray) -> LoaderTrajectory:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError("action must have shape (7,)")
        return LoaderTrajectory(
            heading_deg=self._scale(action[0], -28.0, 28.0),
            lateral_offset=self._scale(action[1], -3.2, 3.2),
            approach_distance=self._scale(action[2], 1.8, 3.2),
            travel_length=self._scale(action[3], 1.7, 3.3),
            bucket_width=2.90,
            max_depth=self._scale(action[4], 0.45, 1.25),
            curl_angle_deg=52.0,
            lift_height=self._scale(action[6], 2.8, 4.5),
            speed_scale=self._scale(action[5], 0.65, 1.25),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self._sample_domain()
        self._episode_done = False
        self._last_result = None
        return self._observation(), {"peak_height_m": float(self._initial.max())}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("episode is done; call reset() before step()")
        trajectory = self.action_to_trajectory(action)
        entry, _start, _end, forward, _toe = trajectory_geometry(
            self._initial,
            (self.spacing, self.spacing),
            trajectory,
            self.workspace_size_m,
        )
        plan = plan_resistance_aware_dig(
            self._initial,
            (self.spacing, self.spacing),
            np.array([entry[1], entry[0]]),
            forward,
            trajectory,
            self._material,
            self._limits,
            count=41,
        )
        _scooped, loaded_volume, _removed = apply_planned_cut(
            self._initial,
            (self.spacing, self.spacing),
            entry,
            trajectory.heading_deg,
            trajectory,
            plan,
        )
        nominal_capacity = (
            trajectory.bucket_width
            * trajectory.travel_length
            * trajectory.max_depth
            * 0.65
        )
        fill_factor = loaded_volume / max(nominal_capacity, 1e-6)
        force_ratio = float(plan.resistance_n.max() / self._limits.max_resistance_n)
        limited_fraction = float(plan.force_limited.mean())
        # Cubic metres dominate. Small regularizers discourage force-saturated,
        # unnecessarily fast trajectories without changing the main objective.
        reward = (
            loaded_volume
            - 0.20 * limited_fraction
            - 0.05 * max(0.0, force_ratio - 0.92) ** 2
            - 0.015 * trajectory.speed_scale**2
        )
        self._episode_done = True
        self._last_result = {
            "loaded_volume_m3": float(loaded_volume),
            "fill_factor": float(fill_factor),
            "max_resistance_n": float(plan.resistance_n.max()),
            "force_limit_n": float(self._limits.max_resistance_n),
            "force_limited_fraction": limited_fraction,
            "trajectory": asdict(trajectory),
            "entry_xy_m": tuple(float(v) for v in entry),
        }
        return self._observation(), float(reward), True, False, self._last_result

