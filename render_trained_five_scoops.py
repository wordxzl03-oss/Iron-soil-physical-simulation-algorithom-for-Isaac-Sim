"""Run a trained SAC policy for five consecutive scoops and save one GIF."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from stable_baselines3 import SAC

from generate_loader_dataset import trajectory_geometry
from physics_aware_trajectory import apply_planned_cut, plan_resistance_aware_dig
from render_rl_four_panel import render_rl_scoop_four_panel
from rl_scooping_env import ScoopingTrajectoryEnv
from slope_model import relax_critical_slope


def _join_gifs(parts: list[Path], output: Path, fps: int) -> None:
    frames: list[Image.Image] = []
    for part in parts:
        with Image.open(part) as animation:
            for index in range(animation.n_frames):
                animation.seek(index)
                frames.append(animation.convert("RGB").copy())
    if not frames:
        raise RuntimeError("no animation frames were rendered")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("rl_runs/sac_scooping_baseline/scooping_sac_final.zip"),
    )
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--scoops", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rl_scoop_demo/trained_five_scoops.gif"),
    )
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=70)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.scoops < 1:
        raise ValueError("--scoops must be positive")
    if not args.model.exists():
        raise FileNotFoundError(f"trained model not found: {args.model}")

    env = ScoopingTrajectoryEnv(seed=args.seed)
    env.reset(seed=args.seed)
    terrain = env._initial.copy()
    original = terrain.copy()
    model = SAC.load(args.model, device=args.device)
    cumulative_volume = 0.0
    summaries: list[dict] = []

    with TemporaryDirectory(prefix="rl_five_scoops_") as temporary:
        temporary_dir = Path(temporary)
        gif_parts: list[Path] = []
        for scoop_index in range(1, args.scoops + 1):
            # The policy sees the terrain left by every preceding scoop.
            env._initial = terrain.copy()
            observation = env._observation()
            action, _ = model.predict(observation, deterministic=True)
            trajectory = env.action_to_trajectory(action)
            entry, _start, _end, forward, _toe = trajectory_geometry(
                terrain,
                (env.spacing, env.spacing),
                trajectory,
                env.workspace_size_m,
            )
            plan = plan_resistance_aware_dig(
                terrain,
                (env.spacing, env.spacing),
                np.array([entry[1], entry[0]]),
                forward,
                trajectory,
                env._material,
                env._limits,
                count=61,
            )
            scooped, loaded_volume, removed = apply_planned_cut(
                terrain,
                (env.spacing, env.spacing),
                entry,
                trajectory.heading_deg,
                trajectory,
                plan,
            )
            stable, _ = relax_critical_slope(
                scooped,
                (env.spacing, env.spacing),
                env._material.internal_friction_deg,
            )
            cumulative_volume += loaded_volume
            part = temporary_dir / f"scoop_{scoop_index:02d}.gif"
            render_rl_scoop_four_panel(
                part,
                terrain,
                scooped,
                stable,
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
                scoop_number=scoop_index,
                total_scoops=args.scoops,
                cumulative_loaded_volume=cumulative_volume,
                change_reference=original,
            )
            gif_parts.append(part)
            summaries.append(
                {
                    "scoop": scoop_index,
                    "loaded_volume_m3": float(loaded_volume),
                    "cumulative_loaded_volume_m3": float(cumulative_volume),
                    "entry_xy_m": [float(value) for value in entry],
                    "peak_resistance_n": float(plan.resistance_n.max()),
                    "force_limited_steps": int(plan.force_limited.sum()),
                    "action": np.asarray(action, dtype=float).tolist(),
                    "trajectory": asdict(trajectory),
                }
            )
            terrain = stable
        _join_gifs(gif_parts, args.output, args.fps)

    summary = {
        "model": str(args.model),
        "seed": args.seed,
        "device": str(model.device),
        "animation": str(args.output),
        "scoops": args.scoops,
        "total_loaded_volume_m3": float(cumulative_volume),
        "mean_loaded_volume_m3": float(cumulative_volume / args.scoops),
        "per_scoop": summaries,
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
