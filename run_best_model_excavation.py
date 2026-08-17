"""Select a large random pile and excavate it below a target fraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from large_pile_excavation_env import LargePileExcavationEnv
from validate_large_pile_batch import run_pile


def inspect_candidate(
    seed: int, peak_height_range_m: tuple[float, float]
) -> dict:
    env = LargePileExcavationEnv(
        seed=seed,
        peak_height_range_m=peak_height_range_m,
    )
    _, info = env.reset(seed=seed)
    return {
        "requested_seed": seed,
        "soil_seed": int(info["soil_seed"]),
        "peak_height_m": float(info["peak_height_m"]),
        "initial_volume_m3": float(info["initial_volume_m3"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026072801)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--candidate-actions", type=int, default=24)
    parser.add_argument("--target", type=float, default=0.10)
    parser.add_argument("--peak-min", type=float, default=23.5)
    parser.add_argument("--peak-max", type=float, default=24.0)
    args = parser.parse_args()
    if not 0.0 < args.target < 1.0:
        raise ValueError("--target must be between zero and one")
    if args.candidates < 1:
        raise ValueError("--candidates must be positive")

    peak_range = (args.peak_min, args.peak_max)
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31 - 1, size=args.candidates)
    candidates = [
        inspect_candidate(int(seed), peak_range) for seed in seeds
    ]
    selected = max(candidates, key=lambda item: item["initial_volume_m3"])
    print(
        "Selected random pile: "
        f"seed={selected['requested_seed']} "
        f"peak={selected['peak_height_m']:.3f} m "
        f"volume={selected['initial_volume_m3']:.3f} m^3"
    )

    model = SAC.load(args.model, device="auto")
    summary = run_pile(
        model,
        int(selected["requested_seed"]),
        args.output,
        args.candidate_actions,
        target_remaining_fraction=args.target,
        peak_height_range_m=peak_range,
    )
    summary["model"] = str(args.model)
    summary["selection_seed"] = args.seed
    summary["selection_criterion"] = "maximum initial volume"
    summary["selection_candidates"] = candidates
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {key: value for key, value in summary.items()
         if key not in {"actions", "selection_candidates"}},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
