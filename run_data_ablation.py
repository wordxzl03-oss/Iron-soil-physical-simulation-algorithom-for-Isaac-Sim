"""Train equal-configuration SAC policies at several data budgets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2_000, 5_000, 10_000])
    parser.add_argument("--eval-piles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20262000)
    parser.add_argument("--output", type=Path, default=Path("ablation/data_scale"))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete policy/metrics pairs already present in the output directory.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for budget in args.budgets:
        run_dir = args.output / f"steps_{budget:06d}"
        metrics_path = run_dir / "metrics.json"
        policy_path = run_dir / "policy.zip"
        if args.resume and metrics_path.is_file() and policy_path.is_file():
            try:
                row = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                row = None
            if (
                row is not None
                and row.get("timesteps") == budget
                and len(row.get("evaluation", [])) == args.eval_piles
            ):
                rows.append(row)
                print(f"Reusing complete result for {budget} timesteps")
                continue

        from stable_baselines3.common.monitor import Monitor

        from wheel_loader_rl.algorithms import build_sac
        from wheel_loader_rl.config import EnvironmentConfig, SACConfig
        from wheel_loader_rl.envs import LargePileExcavationEnv
        from wheel_loader_rl.evaluation import evaluate_policy

        env_config = EnvironmentConfig()
        raw_env = LargePileExcavationEnv(
            **env_config.make_kwargs(), seed=args.seed, record_training_data=True
        )
        model = build_sac(
            Monitor(raw_env),
            SACConfig(timesteps=budget, seed=args.seed, device=args.device),
        )
        model.learn(budget)
        run_dir.mkdir(exist_ok=True)
        model.save(run_dir / "policy")
        episodes = evaluate_policy(
            model,
            lambda seed: LargePileExcavationEnv(
                **env_config.make_kwargs(), seed=seed
            ),
            range(args.seed + 10_000, args.seed + 10_000 + args.eval_piles),
        )
        completed = len(raw_env.completed_episodes)
        transitions = sum(
            len(x["records"]) for x in raw_env.completed_episodes
        )
        row = {
            "timesteps": budget,
            "training_piles": completed,
            "training_trajectories": transitions,
            "success_rate": float(np.mean([x["success"] for x in episodes])),
            "mean_load_m3": float(np.mean([x["mean_load_m3"] for x in episodes])),
            "payload_efficiency": float(
                np.mean([x["payload_efficiency"] for x in episodes])
            ),
            "mean_remaining_fraction": float(
                np.mean([x["remaining_fraction"] for x in episodes])
            ),
            "mean_scoops": float(np.mean([x["scoops"] for x in episodes])),
            "evaluation": episodes,
        }
        rows.append(row)
        (run_dir / "metrics.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in row.items() if k != "evaluation"}))

    (args.output / "ablation_results.json").write_text(
        json.dumps({"runs": rows}, indent=2), encoding="utf-8"
    )
    x = [r["training_trajectories"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    plots = [
        ("mean_load_m3", "Mean payload [m³]", "#1976d2"),
        ("mean_remaining_fraction", "Remaining fraction", "#ef6c00"),
        ("success_rate", "Success rate", "#2e7d32"),
    ]
    for ax, (key, label, color) in zip(axes, plots):
        ax.plot(x, [r[key] for r in rows], "o-", color=color, linewidth=2)
        ax.set(xlabel="Training scoop trajectories", ylabel=label)
        ax.grid(alpha=0.25)
    fig.suptitle("SAC data-scale ablation (fixed 12-pile test set)")
    fig.savefig(args.output / "data_scale_ablation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
