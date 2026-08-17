"""Render one random-pile scoop selected by a trained SAC policy."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from generate_loader_dataset import trajectory_geometry
from physics_aware_trajectory import apply_planned_cut, plan_resistance_aware_dig
from render_rl_four_panel import render_rl_scoop_four_panel
from rl_scooping_env import ScoopingTrajectoryEnv
from slope_model import relax_critical_slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("rl_runs/sac_scooping_baseline/scooping_sac_final.zip"),
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--output", type=Path, default=Path("rl_scoop_demo/trained_scoop.gif")
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=85)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"trained model not found: {args.model}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = ScoopingTrajectoryEnv(seed=args.seed)
    observation, reset_info = env.reset(seed=args.seed)
    model = SAC.load(args.model, device=args.device)
    action, _ = model.predict(observation, deterministic=True)
    trajectory = env.action_to_trajectory(action)
    entry, _start, _end, forward, _toe = trajectory_geometry(
        env._initial,
        (env.spacing, env.spacing),
        trajectory,
        env.workspace_size_m,
    )
    plan = plan_resistance_aware_dig(
        env._initial,
        (env.spacing, env.spacing),
        np.array([entry[1], entry[0]]),
        forward,
        trajectory,
        env._material,
        env._limits,
        count=61,
    )
    scooped, loaded_volume, removed = apply_planned_cut(
        env._initial,
        (env.spacing, env.spacing),
        entry,
        trajectory.heading_deg,
        trajectory,
        plan,
    )
    final, _ = relax_critical_slope(
        scooped,
        (env.spacing, env.spacing),
        env._material.internal_friction_deg,
    )
    render_rl_scoop_four_panel(
        args.output,
        env._initial,
        scooped,
        final,
        removed,
        plan.path,
        plan.pitch_deg,
        plan.cut_fraction,
        trajectory,
        loaded_volume,
        env.spacing,
        env.workspace_size_m,
        plan.resistance_n,
        plan.speed_m_s,
        args.fps,
        args.dpi,
    )
    summary = {
        "model": str(args.model),
        "seed": args.seed,
        "device": str(model.device),
        "animation": str(args.output),
        "peak_height_m": reset_info["peak_height_m"],
        "loaded_volume_m3": float(loaded_volume),
        "entry_xy_m": [float(value) for value in entry],
        "peak_resistance_n": float(plan.resistance_n.max()),
        "force_limit_n": float(env._limits.max_resistance_n),
        "force_limited_steps": int(plan.force_limited.sum()),
        "action": np.asarray(action, dtype=float).tolist(),
        "trajectory": asdict(trajectory),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
