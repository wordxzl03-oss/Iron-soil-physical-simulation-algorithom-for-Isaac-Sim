"""Policy-guided candidate search used as a maximum-payload supervisor."""

from __future__ import annotations

import numpy as np

from generate_loader_dataset import trajectory_geometry
from slope_model import scoop_loader_bucket


def select_max_scoop_action(
    env,
    policy_action: np.ndarray,
    candidate_count: int = 24,
) -> tuple[np.ndarray, float]:
    """Return the best fast geometric candidate around a policy proposal."""
    base = np.asarray(policy_action, dtype=np.float32)
    candidates = [base.copy()]
    headings = np.linspace(-1.0, 1.0, candidate_count, endpoint=False)
    # Test the full lateral working range per direction. Other controls retain the policy
    # values, while penetration/depth are biased toward productive values.
    for heading in headings:
        for lateral in (-1.0, -0.5, 0.0, 0.5, 1.0):
            action = base.copy()
            action[0] = heading
            action[1] = lateral
            action[3] = max(action[3], 0.80)
            action[4] = max(action[4], 0.70)
            action[5] = max(action[5], 0.65)
            candidates.append(action)
    best_action = candidates[0]
    best_volume = -1.0
    for action in candidates:
        trajectory = env.action_to_trajectory(action)
        entry, _start, _end, _forward, _toe = trajectory_geometry(
            env._initial,
            (env.spacing, env.spacing),
            trajectory,
            env.workspace_size_m,
        )
        _after, volume, _removed = scoop_loader_bucket(
            env._initial,
            entry,
            (env.spacing, env.spacing),
            travel_length=trajectory.travel_length,
            bucket_width=trajectory.bucket_width,
            max_depth=trajectory.max_depth,
            heading_deg=trajectory.heading_deg,
        )
        heading = np.deg2rad(trajectory.heading_deg)
        forward = np.array([np.sin(heading), np.cos(heading)])
        right = np.array([np.cos(heading), -np.sin(heading)])
        # The axle centre is about 5.225 m behind the cutting edge. During the
        # 4 m acceleration run, both wheel tracks sweep the corridor below.
        backtrack = np.linspace(5.2, 9.3, 28)
        corridor_height = 0.0
        for lateral in (-1.42, 1.42):
            points = (
                np.asarray(entry)[None, :]
                - backtrack[:, None] * forward
                + lateral * right
            )
            ii = np.clip(
                np.rint(points[:, 0] / env.spacing).astype(int),
                0, env.grid_size - 1,
            )
            jj = np.clip(
                np.rint(points[:, 1] / env.spacing).astype(int),
                0, env.grid_size - 1,
            )
            corridor_height = max(
                corridor_height, float(env._initial[ii, jj].max())
            )
        # A high theoretical payload is useless when a wheel reaches the pile
        # before the bucket. Prefer a completely cleared approach corridor;
        # the small tolerance only absorbs height-map interpolation noise.
        score = min(volume, env.bucket_capacity_m3) - 20.0 * max(
            0.0, corridor_height - 0.05
        )
        if score > best_volume:
            best_volume = score
            best_action = action
    return np.asarray(best_action, dtype=np.float32), float(best_volume)
