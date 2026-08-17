"""Versioned trajectory dataset schema for RL, diffusion and world models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "wheel-loader-trajectory-v1"


def save_episode(path: Path, *, initial_height, final_height, records, metadata):
    """Save one episode as dense tensors plus human-readable metadata."""
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": np.asarray(
            [x["observation"] for x in records], dtype=np.float32
        ),
        "actions": np.asarray([x["action"] for x in records], dtype=np.float32),
        "rewards": np.asarray([x["reward"] for x in records], dtype=np.float32),
        "loaded_volume_m3": np.asarray(
            [x["loaded_volume_m3"] for x in records], dtype=np.float32
        ),
        "remaining_fraction": np.asarray(
            [x["remaining_fraction"] for x in records], dtype=np.float32
        ),
        "initial_height": np.asarray(initial_height, dtype=np.float32),
        "final_height": np.asarray(final_height, dtype=np.float32),
    }
    np.savez_compressed(path / "trajectory.npz", **arrays)
    document = {"schema_version": SCHEMA_VERSION, **metadata}
    (path / "metadata.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_episode(path: Path) -> tuple[dict, dict]:
    arrays = dict(np.load(path / "trajectory.npz"))
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema: {metadata['schema_version']}")
    return arrays, metadata
