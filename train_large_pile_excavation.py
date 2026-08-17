"""Train SAC on continuous excavation of >20 m Perlin piles."""

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

from large_pile_excavation_env import LargePileExcavationEnv


def save_training_rollouts(env: LargePileExcavationEnv, output: Path) -> dict:
    episodes = env.completed_episodes
    observations, actions, rewards, loads, remaining = [], [], [], [], []
    episode_index = []
    summaries = []
    for index, episode in enumerate(episodes):
        for record in episode["records"]:
            observations.append(record["observation"])
            actions.append(record["action"])
            rewards.append(record["reward"])
            loads.append(record["loaded_volume_m3"])
            remaining.append(record["remaining_fraction"])
            episode_index.append(index)
        summaries.append(
            {key: value for key, value in episode.items() if key != "records"}
        )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "training_transitions.npz",
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        loaded_volume_m3=np.asarray(loads, dtype=np.float32),
        remaining_fraction=np.asarray(remaining, dtype=np.float32),
        episode_index=np.asarray(episode_index, dtype=np.int32),
    )
    summary = {
        "completed_episodes": len(episodes),
        "transitions": len(actions),
        "episodes": summaries,
    }
    (output / "training_rollouts.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("rl_runs/sac_large_pile_excavation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    check_env(LargePileExcavationEnv(seed=args.seed), warn=True)
    raw_env = LargePileExcavationEnv(
        seed=args.seed, record_training_data=True
    )
    vec_env = DummyVecEnv([lambda: Monitor(raw_env)])
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=max(30_000, args.timesteps),
        learning_starts=min(1_000, max(100, args.timesteps // 10)),
        batch_size=256,
        gamma=0.995,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log=str(args.output_dir / "tensorboard"),
        verbose=0,
    )
    model.learn(args.timesteps, log_interval=10)
    model.save(args.output_dir / "large_pile_sac")
    rollout_summary = save_training_rollouts(
        raw_env, args.output_dir / "training_data"
    )
    metadata = {
        "timesteps": args.timesteps,
        "device": str(model.device),
        "seed": args.seed,
        "pile_peak_range_m": [20.5, 24.0],
        "target_remaining_fraction": 0.20,
        "bucket_capacity_m3": 3.0,
        "obj_path": "simple_wheel_loader.obj",
        **{key: value for key, value in rollout_summary.items() if key != "episodes"},
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

