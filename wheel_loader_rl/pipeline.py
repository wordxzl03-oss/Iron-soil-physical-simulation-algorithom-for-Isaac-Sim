"""Unified command line pipeline: train, generate and evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

from .algorithms import build_sac, load_sac
from .config import EnvironmentConfig, SACConfig, resolve_output
from .envs import LargePileExcavationEnv
from .evaluation import evaluate_policy


def train(args) -> None:
    output = resolve_output(args.output)
    env_config = EnvironmentConfig()
    algorithm_config = SACConfig(
        timesteps=args.timesteps, seed=args.seed, device=args.device
    )
    env = Monitor(LargePileExcavationEnv(
        **env_config.make_kwargs(), seed=args.seed
    ))
    model = build_sac(env, algorithm_config)
    model.learn(args.timesteps, log_interval=10)
    model.save(output / "policy")
    metadata = {
        "environment": env_config.to_dict(),
        "algorithm": {"name": "SAC", **algorithm_config.to_dict()},
        "weights": "policy.zip",
    }
    (output / "config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def evaluate(args) -> None:
    config = EnvironmentConfig()
    model = load_sac(args.model, device=args.device)
    seeds = list(range(args.seed, args.seed + args.piles))
    results = evaluate_policy(
        model,
        lambda seed: LargePileExcavationEnv(**config.make_kwargs(), seed=seed),
        seeds,
        supervisor_candidates=args.supervisor_candidates,
    )
    summary = {
        "piles": len(results),
        "success_rate": sum(x["success"] for x in results) / len(results),
        "mean_scoops": sum(x["scoops"] for x in results) / len(results),
        "mean_payload_efficiency": (
            sum(x["payload_efficiency"] for x in results) / len(results)
        ),
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p_train = commands.add_parser("train")
    p_train.add_argument("--timesteps", type=int, default=20_000)
    p_train.add_argument("--seed", type=int, default=20261101)
    p_train.add_argument("--device", default="auto")
    p_train.add_argument("--output", type=Path, default=Path("runs/reference_sac"))
    p_train.set_defaults(function=train)
    p_eval = commands.add_parser("evaluate")
    p_eval.add_argument("--model", type=Path, required=True)
    p_eval.add_argument("--piles", type=int, default=20)
    p_eval.add_argument("--seed", type=int, default=20270001)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--supervisor-candidates", type=int, default=0)
    p_eval.add_argument(
        "--output", type=Path, default=Path("runs/evaluation.json")
    )
    p_eval.set_defaults(function=evaluate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
