"""Train the terrain-contact-aware simple wheel loader trajectory policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from dynamic_obj_scooping_env import DynamicObjScoopingTrajectoryEnv


def evaluate(model: SAC, episodes: int, seed: int) -> dict:
    values: list[dict] = []
    for index in range(episodes):
        env = DynamicObjScoopingTrajectoryEnv(seed=seed + index)
        observation, _ = env.reset(seed=seed + index)
        action, _ = model.predict(observation, deterministic=True)
        _, reward, _, _, info = env.step(action)
        values.append({**info, "reward": reward})
    return {
        "episodes": episodes,
        "mean_loaded_volume_m3": float(
            np.mean([item["loaded_volume_m3"] for item in values])
        ),
        "mean_dynamic_penetration_m": float(
            np.mean([item["dynamic_penetration_m"] for item in values])
        ),
        "mean_max_pitch_deg": float(
            np.mean([item["max_pitch_deg"] for item in values])
        ),
        "mean_max_roll_deg": float(
            np.mean([item["max_roll_deg"] for item in values])
        ),
        "details": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("rl_runs/sac_dynamic_simple_loader"),
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_env(DynamicObjScoopingTrajectoryEnv(seed=args.seed), warn=True)
    env = DummyVecEnv(
        [lambda: Monitor(DynamicObjScoopingTrajectoryEnv(seed=args.seed))]
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=max(10_000, args.timesteps),
        learning_starts=min(500, max(50, args.timesteps // 10)),
        batch_size=128,
        gamma=0.0,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=str(args.output_dir / "tensorboard"),
        verbose=0,
    )
    model.learn(args.timesteps, log_interval=100)
    model.save(args.output_dir / "dynamic_simple_loader_sac")
    result = evaluate(model, args.eval_episodes, args.seed + 30_000)
    result.update(
        {
            "timesteps": args.timesteps,
            "device": str(model.device),
            "obj_path": "simple_wheel_loader.obj",
            "wheelbase_m": 2.30,
            "track_width_m": 2.84,
            "wheel_radius_m": 0.72,
        }
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "details"}, indent=2))


if __name__ == "__main__":
    main()
