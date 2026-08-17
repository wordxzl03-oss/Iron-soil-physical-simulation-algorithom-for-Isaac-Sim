"""Excavate a batch of >20 m Perlin piles and save every policy action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from large_pile_excavation_env import LargePileExcavationEnv
from max_scoop_supervisor import select_max_scoop_action


def run_pile(
    model: SAC,
    seed: int,
    output: Path,
    candidate_actions: int = 24,
    target_remaining_fraction: float = 0.20,
    peak_height_range_m: tuple[float, float] = (20.5, 24.0),
) -> dict:
    env = LargePileExcavationEnv(
        seed=seed,
        target_remaining_fraction=target_remaining_fraction,
        peak_height_range_m=peak_height_range_m,
    )
    observation, reset_info = env.reset(seed=seed)
    initial = env._initial.copy()
    observations, actions, rewards, loads, fractions = [], [], [], [], []
    records = []
    terminated = truncated = False
    hard_limit = env.minimum_scoop_label * 6
    empty_streak = 0
    while not terminated and env.scoop_count < hard_limit:
        action, _ = model.predict(observation, deterministic=True)
        supervised_volume = None
        if candidate_actions > 0:
            action, supervised_volume = select_max_scoop_action(
                env, action, candidate_actions
            )
        if empty_streak >= 2:
            # Residual steep faces can look productive geometrically while a
            # deep cut stalls the vehicle at contact. Shallow high-drive cuts
            # restore a reachable working face.
            action = np.asarray(action, dtype=np.float32).copy()
            action[3] = 1.0
            action[4] = -0.55
            action[5] = 1.0
        observations.append(observation.copy())
        actions.append(np.asarray(action, dtype=np.float32))
        observation, reward, terminated, truncated, info = env.step(action)
        # TimeLimit protects training from pathological rollouts. Batch
        # excavation must continue until the physical 20% target is reached.
        if truncated:
            env._episode_done = False
        rewards.append(reward)
        loads.append(info["loaded_volume_m3"])
        empty_streak = (
            empty_streak + 1 if info["loaded_volume_m3"] < 0.05 else 0
        )
        fractions.append(info["remaining_fraction"])
        records.append(
            {
                "scoop": info["scoop_count"],
                "action": np.asarray(action, dtype=float).tolist(),
                "reward": float(reward),
                "loaded_volume_m3": info["loaded_volume_m3"],
                "remaining_fraction": info["remaining_fraction"],
                "entry_xy_m": info["entry_xy_m"],
                "dynamic_penetration_m": info["dynamic_penetration_m"],
                "trajectory": info["trajectory"],
                "supervisor_predicted_volume_m3": supervised_volume,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "excavation_data.npz",
        initial_height=initial.astype(np.float32),
        final_height=env._initial.astype(np.float32),
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        loaded_volume_m3=np.asarray(loads, dtype=np.float32),
        remaining_fraction=np.asarray(fractions, dtype=np.float32),
        grid_spacing_m=np.float32(env.spacing),
    )
    summary = {
        "soil_seed": reset_info["soil_seed"],
        "requested_seed": seed,
        "peak_height_m": reset_info["peak_height_m"],
        "initial_volume_m3": reset_info["initial_volume_m3"],
        "final_volume_m3": info["remaining_volume_m3"],
        "remaining_fraction": info["remaining_fraction"],
        "scoop_count": info["scoop_count"],
        "minimum_scoop_label": info["minimum_scoop_label"],
        "target_remaining_fraction": target_remaining_fraction,
        "peak_height_range_m": list(peak_height_range_m),
        "excess_scoops": info["scoop_count"] - info["minimum_scoop_label"],
        "mean_load_m3": float(np.mean(loads)),
        "terminated": bool(terminated),
        "truncated": bool(not terminated),
        "actions": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path,
        default=Path("rl_runs/sac_large_pile_excavation/large_pile_sac.zip"),
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261201)
    parser.add_argument(
        "--candidate-actions", type=int, default=24,
        help="policy-guided maximum-scoop candidates; 0 disables supervision",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("large_pile_validation_100"),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="reuse completed pile_NNNN/summary.json files",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    model = SAC.load(args.model, device="auto")
    summaries = []
    for index in range(args.count):
        pile_dir = args.output_dir / f"pile_{index + 1:04d}"
        summary_path = pile_dir / "summary.json"
        if args.resume and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        else:
            summary = run_pile(
                model, args.seed + index, pile_dir, args.candidate_actions
            )
        summaries.append(summary)
        print(
            f"[{index + 1:04d}/{args.count:04d}] "
            f"peak={summary['peak_height_m']:.2f}m "
            f"scoops={summary['scoop_count']} "
            f"remaining={100 * summary['remaining_fraction']:.1f}%"
        )
    aggregate = {
        "count": args.count,
        "seed": args.seed,
        "model": str(args.model),
        "successes": sum(item["terminated"] for item in summaries),
        "mean_scoop_count": float(np.mean([item["scoop_count"] for item in summaries])),
        "mean_minimum_scoop_label": float(
            np.mean([item["minimum_scoop_label"] for item in summaries])
        ),
        "mean_remaining_fraction": float(
            np.mean([item["remaining_fraction"] for item in summaries])
        ),
        "piles": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in aggregate.items() if k != "piles"}, indent=2))


if __name__ == "__main__":
    main()
