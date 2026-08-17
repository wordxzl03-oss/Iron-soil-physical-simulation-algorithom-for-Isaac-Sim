"""Deterministic evaluation utilities."""

from __future__ import annotations

import numpy as np

from max_scoop_supervisor import select_max_scoop_action


def evaluate_policy(model, env_factory, seeds, *, supervisor_candidates=0):
    episodes = []
    for seed in seeds:
        env = env_factory(seed)
        observation, info0 = env.reset(seed=seed)
        terminated = truncated = False
        loads = []
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            if supervisor_candidates:
                action, _ = select_max_scoop_action(
                    env, action, supervisor_candidates
                )
            observation, _, terminated, truncated, info = env.step(action)
            loads.append(info["loaded_volume_m3"])
        episodes.append({
            "seed": seed,
            "success": bool(terminated),
            "scoops": info["scoop_count"],
            "minimum_scoop_label": info["minimum_scoop_label"],
            "remaining_fraction": info["remaining_fraction"],
            "mean_load_m3": float(np.mean(loads)),
            "payload_efficiency": float(np.mean(loads) / env.bucket_capacity_m3),
            "peak_height_m": info0["peak_height_m"],
        })
    return episodes
