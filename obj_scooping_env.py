"""OBJ-geometry constrained variant of the scooping trajectory task."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from animate_loader_3d import LoaderTrajectory
from render_obj_loader_scooping import (
    MeshPart,
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
    transform_point,
)
from rl_scooping_env import ScoopingTrajectoryEnv


class ObjScoopingTrajectoryEnv(ScoopingTrajectoryEnv):
    """Trajectory task parameterized by the exact ``wheel_buck.obj`` geometry."""

    def __init__(
        self,
        *,
        obj_path: str | Path = "wheel_buck.obj",
        bucket_capacity_m3: float = 3.0,
        **kwargs,
    ) -> None:
        self.obj_path = Path(obj_path)
        raw = load_obj_parts(self.obj_path)
        self.obj_source_part_names = tuple(raw)
        simple_required = {
            "simple_chassis", "simple_cab",
            "wheel_front_left", "wheel_front_right",
            "wheel_rear_left", "wheel_rear_right",
            "loader_boom", "loader_bucket",
        }
        legacy_required = {
            "loader_frame_world.stl", "loader_boom_world.stl",
            "loader_bucket_world.stl",
        }
        if simple_required <= raw.keys():
            fixed_names = [
                "simple_chassis", "simple_cab",
                "wheel_front_left", "wheel_front_right",
                "wheel_rear_left", "wheel_rear_right",
            ]
            vertices: list[np.ndarray] = []
            faces: list[np.ndarray] = []
            offset = 0
            for name in fixed_names:
                part = raw[name]
                vertices.append(part.vertices)
                faces.append(part.faces + offset)
                offset += len(part.vertices)
            frame = MeshPart(
                "simple_loader_fixed",
                np.vstack(vertices),
                np.vstack(faces),
            )
            boom = raw["loader_boom"]
            bucket_part = raw["loader_bucket"]

            def nearest_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
                nearest = np.argmin(squared, axis=1)
                distance = squared[np.arange(len(a)), nearest]
                selected = np.argsort(distance)[:40]
                midpoint = 0.5 * (a[selected] + b[nearest[selected]])
                return np.array(
                    [0.0, np.median(midpoint[:, 1]), np.median(midpoint[:, 2])]
                )

            # The simple loader generator preserves the prepared boom/bucket
            # coordinates from wheel_buck.obj. Its box chassis has only corner
            # vertices, so a nearest-surface estimate would move the root pin
            # forward. These are the preserved linkage centers in that model.
            root = np.array([0.0, -0.8480385, 2.056295])
            pin = nearest_midpoint(boom.vertices, bucket_part.vertices)
            front = bucket_part.vertices[
                bucket_part.vertices[:, 1] >= bucket_part.vertices[:, 1].max() - 0.02
            ]
            edge = np.array(
                [0.0, float(np.median(front[:, 1])), float(np.median(front[:, 2]))]
            )
            self.obj_parts = {
                "loader_frame_world.stl": frame,
                "loader_boom_world.stl": boom,
                "loader_bucket_world.stl": bucket_part,
            }
            self.obj_points = {
                "root_pin": root, "bucket_pin": pin, "cutting_edge": edge
            }
        elif legacy_required <= raw.keys():
            self.obj_parts, self.obj_points = _prepare_model(
                raw, infer_landmarks(raw)
            )
        else:
            raise ValueError(
                "unsupported OBJ grouping; expected simple_wheel_loader or wheel_buck parts"
            )
        bucket = self.obj_parts["loader_bucket_world.stl"].vertices
        self.obj_bucket_width_m = float(np.ptp(bucket[:, 0]))
        self.obj_bucket_length_m = float(np.ptp(bucket[:, 1]))
        self.bucket_capacity_m3 = float(bucket_capacity_m3)
        if self.bucket_capacity_m3 <= 0:
            raise ValueError("bucket capacity must be positive")
        root = self.obj_points["root_pin"]
        edge = self.obj_points["cutting_edge"]
        angles = np.linspace(-18.0, 42.0, 2001)
        edge_heights = np.asarray(
            [transform_point(edge, root, angle)[2] for angle in angles]
        )
        self.low_boom_angle_deg = float(
            angles[np.argmin(np.abs(edge_heights - 0.12))]
        )
        self.high_boom_angle_deg = self.low_boom_angle_deg + 34.0
        self.obj_max_edge_lift_m = float(
            transform_point(edge, root, self.high_boom_angle_deg)[2]
            - transform_point(edge, root, self.low_boom_angle_deg)[2]
        )
        super().__init__(**kwargs)

    def action_to_trajectory(self, action: np.ndarray) -> LoaderTrajectory:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,):
            raise ValueError("action must have shape (7,)")
        return LoaderTrajectory(
            heading_deg=self._scale(action[0], -28.0, 28.0),
            lateral_offset=self._scale(action[1], -3.2, 3.2),
            approach_distance=self._scale(action[2], 1.8, 3.2),
            travel_length=self._scale(action[3], 0.75, 1.65),
            bucket_width=self.obj_bucket_width_m,
            max_depth=self._scale(action[4], 0.50, 1.10),
            curl_angle_deg=self._scale(action[6], 44.0, 52.0),
            lift_height=self.obj_max_edge_lift_m,
            speed_scale=self._scale(action[5], 0.65, 1.20),
        )

    def step(self, action: np.ndarray):
        observation, _reward, terminated, truncated, info = super().step(action)
        raw_volume = float(info["loaded_volume_m3"])
        loaded_volume = min(raw_volume, self.bucket_capacity_m3)
        limited_fraction = float(info["force_limited_fraction"])
        trajectory = info["trajectory"]
        reward = (
            loaded_volume
            - 0.20 * limited_fraction
            - 0.015 * float(trajectory["speed_scale"]) ** 2
        )
        info.update(
            {
                "raw_swept_volume_m3": raw_volume,
                "loaded_volume_m3": loaded_volume,
                "bucket_capacity_m3": self.bucket_capacity_m3,
                "fill_factor": loaded_volume / self.bucket_capacity_m3,
                "obj_path": str(self.obj_path),
                "obj_bucket_width_m": self.obj_bucket_width_m,
            }
        )
        return observation, float(reward), terminated, truncated, info
