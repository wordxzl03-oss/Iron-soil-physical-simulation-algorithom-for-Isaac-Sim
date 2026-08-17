"""Train and evaluate a SAC policy for maximum single-scoop material yield."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_scooping_env import ScoopingTrajectoryEnv


def evaluate(model: SAC, episodes: int, seed: int) -> dict:
    env = ScoopingTrajectoryEnv(seed=seed)
    random_env = ScoopingTrajectoryEnv(seed=seed)
    neutral_env = ScoopingTrajectoryEnv(seed=seed)
    action_rng = np.random.default_rng(seed)
    volumes: list[float] = []
    random_volumes: list[float] = []
    neutral_volumes: list[float] = []
    rewards: list[float] = []
    trajectories: list[dict] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        action, _ = model.predict(observation, deterministic=True)
        _, reward, _, _, info = env.step(action)
        volumes.append(info["loaded_volume_m3"])
        rewards.append(reward)
        trajectories.append(info["trajectory"])
        random_env.reset(seed=seed + episode)
        _, _, _, _, random_info = random_env.step(
            action_rng.uniform(-1.0, 1.0, 7).astype(np.float32)
        )
        random_volumes.append(random_info["loaded_volume_m3"])
        neutral_env.reset(seed=seed + episode)
        _, _, _, _, neutral_info = neutral_env.step(
            np.zeros(7, dtype=np.float32)
        )
        neutral_volumes.append(neutral_info["loaded_volume_m3"])
    random_mean = float(np.mean(random_volumes))
    neutral_mean = float(np.mean(neutral_volumes))
    learned_mean = float(np.mean(volumes))
    return {
        "episodes": episodes,
        "mean_loaded_volume_m3": learned_mean,
        "std_loaded_volume_m3": float(np.std(volumes)),
        "min_loaded_volume_m3": float(np.min(volumes)),
        "max_loaded_volume_m3": float(np.max(volumes)),
        "random_mean_loaded_volume_m3": random_mean,
        "neutral_mean_loaded_volume_m3": neutral_mean,
        "improvement_over_random_percent": (
            100.0 * (learned_mean / random_mean - 1.0)
        ),
        "improvement_over_neutral_percent": (
            100.0 * (learned_mean / neutral_mean - 1.0)
        ),
        "mean_reward": float(np.mean(rewards)),
        "trajectories": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--output-dir", type=Path, default=Path("rl_runs/sac_scooping"))
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.timesteps < 1 or args.eval_episodes < 1:
        raise ValueError("timesteps and eval episodes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_env(ScoopingTrajectoryEnv(seed=args.seed), warn=True)
    env = DummyVecEnv(
        [lambda: Monitor(ScoopingTrajectoryEnv(seed=args.seed))]
    )
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device == "auto":
        device = "cpu"
    learning_starts = min(1_000, max(10, args.timesteps // 10))
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=max(10_000, min(200_000, args.timesteps)),
        learning_starts=learning_starts,
        batch_size=64 if args.smoke_test else 256,
        gamma=0.0,  # Each episode is a one-scoop contextual decision.
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(args.output_dir / "tensorboard"),
        seed=args.seed,
        device=device,
        verbose=1,
    )
    checkpoint = CheckpointCallback(
        save_freq=max(1_000, args.timesteps // 5),
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="scooping_sac",
    )
    model.learn(total_timesteps=args.timesteps, callback=checkpoint)
    model.save(args.output_dir / "scooping_sac_final")
    result = evaluate(model, args.eval_episodes, args.seed + 10_000)
    result.update(
        {
            "timesteps": args.timesteps,
            "device": str(model.device),
            "torch_version": torch.__version__,
        }
    )
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "trajectories"}, indent=2))


if __name__ == "__main__":
    main()
