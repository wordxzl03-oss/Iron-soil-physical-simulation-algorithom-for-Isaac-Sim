"""Train SAC using dimensions and capacity from wheel_buck.obj."""

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

from obj_scooping_env import ObjScoopingTrajectoryEnv


def evaluate(model: SAC, episodes: int, seed: int) -> dict:
    learned: list[float] = []
    random: list[float] = []
    rng = np.random.default_rng(seed)
    for index in range(episodes):
        episode_seed = seed + index
        env = ObjScoopingTrajectoryEnv(seed=episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        action, _ = model.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        learned.append(info["loaded_volume_m3"])
        baseline = ObjScoopingTrajectoryEnv(seed=episode_seed)
        baseline.reset(seed=episode_seed)
        _, _, _, _, random_info = baseline.step(
            rng.uniform(-1, 1, 7).astype(np.float32)
        )
        random.append(random_info["loaded_volume_m3"])
    learned_mean = float(np.mean(learned))
    random_mean = float(np.mean(random))
    return {
        "episodes": episodes,
        "mean_loaded_volume_m3": learned_mean,
        "mean_fill_factor": learned_mean / 3.0,
        "random_mean_loaded_volume_m3": random_mean,
        "improvement_over_random_percent": 100.0 * (learned_mean / random_mean - 1.0),
        "min_loaded_volume_m3": float(np.min(learned)),
        "max_loaded_volume_m3": float(np.max(learned)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--obj", type=Path, default=Path("wheel_buck.obj"))
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("rl_runs/sac_obj_scooping")
    )
    parser.add_argument("--eval-episodes", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_env(ObjScoopingTrajectoryEnv(obj_path=args.obj, seed=args.seed), warn=True)
    env = DummyVecEnv(
        [lambda: Monitor(ObjScoopingTrajectoryEnv(obj_path=args.obj, seed=args.seed))]
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=max(20_000, args.timesteps),
        learning_starts=min(1_000, max(100, args.timesteps // 10)),
        batch_size=256,
        gamma=0.0,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(args.output_dir / "tensorboard"),
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=1,
    )
    model.learn(args.timesteps)
    model.save(args.output_dir / "scooping_sac_obj_final")
    # Evaluation uses the same geometry selected for training.
    learned: list[float] = []
    random: list[float] = []
    rng = np.random.default_rng(args.seed + 20_000)
    for index in range(args.eval_episodes):
        episode_seed = args.seed + 20_000 + index
        evaluation_env = ObjScoopingTrajectoryEnv(
            obj_path=args.obj, seed=episode_seed
        )
        observation, _ = evaluation_env.reset(seed=episode_seed)
        action, _ = model.predict(observation, deterministic=True)
        learned.append(evaluation_env.step(action)[4]["loaded_volume_m3"])
        random_env = ObjScoopingTrajectoryEnv(obj_path=args.obj, seed=episode_seed)
        random_env.reset(seed=episode_seed)
        random.append(
            random_env.step(rng.uniform(-1, 1, 7).astype(np.float32))[4][
                "loaded_volume_m3"
            ]
        )
    learned_mean, random_mean = float(np.mean(learned)), float(np.mean(random))
    result = {
        "episodes": args.eval_episodes,
        "mean_loaded_volume_m3": learned_mean,
        "mean_fill_factor": learned_mean / 3.0,
        "random_mean_loaded_volume_m3": random_mean,
        "improvement_over_random_percent": 100 * (learned_mean / random_mean - 1),
        "min_loaded_volume_m3": float(np.min(learned)),
        "max_loaded_volume_m3": float(np.max(learned)),
    }
    geometry = ObjScoopingTrajectoryEnv(obj_path=args.obj)
    result.update(
        {
            "timesteps": args.timesteps,
            "device": str(model.device),
            "obj_path": str(geometry.obj_path),
            "obj_source_part_names": geometry.obj_source_part_names,
            "obj_bucket_width_m": geometry.obj_bucket_width_m,
            "obj_bucket_length_m": geometry.obj_bucket_length_m,
            "bucket_capacity_m3": geometry.bucket_capacity_m3,
            "low_boom_angle_deg": geometry.low_boom_angle_deg,
            "high_boom_angle_deg": geometry.high_boom_angle_deg,
        }
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
