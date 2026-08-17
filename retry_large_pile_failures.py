"""Retry failed piles in an existing validation set and rebuild its summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from validate_large_pile_batch import run_pile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("large_pile_validation_100")
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("rl_runs/sac_large_pile_excavation_v2/large_pile_sac.zip"),
    )
    parser.add_argument("--candidate-actions", type=int, default=24)
    args = parser.parse_args()
    aggregate_path = args.dataset / "dataset_summary.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    model = SAC.load(args.model, device="auto")
    summaries = aggregate["piles"]
    failures = [
        index for index, summary in enumerate(summaries)
        if not summary["terminated"]
    ]
    for position, index in enumerate(failures, start=1):
        seed = int(aggregate["seed"]) + index
        pile_dir = args.dataset / f"pile_{index + 1:04d}"
        summary = run_pile(
            model, seed, pile_dir, args.candidate_actions
        )
        summaries[index] = summary
        print(
            f"[retry {position:02d}/{len(failures):02d}] pile={index + 1:04d} "
            f"scoops={summary['scoop_count']} "
            f"remaining={100 * summary['remaining_fraction']:.2f}%"
        )
    aggregate.update(
        {
            "successes": sum(item["terminated"] for item in summaries),
            "mean_scoop_count": float(np.mean(
                [item["scoop_count"] for item in summaries]
            )),
            "mean_minimum_scoop_label": float(np.mean(
                [item["minimum_scoop_label"] for item in summaries]
            )),
            "mean_remaining_fraction": float(np.mean(
                [item["remaining_fraction"] for item in summaries]
            )),
            "piles": summaries,
        }
    )
    aggregate_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {key: value for key, value in aggregate.items() if key != "piles"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
