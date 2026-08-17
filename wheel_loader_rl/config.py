"""Typed configuration shared by training, generation and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentConfig:
    grid_size: int = 81
    workspace_size_m: float = 30.0
    observation_grid: int = 15
    peak_height_min_m: float = 20.5
    peak_height_max_m: float = 24.0
    target_remaining_fraction: float = 0.20
    obj_path: str = "simple_wheel_loader.obj"

    def make_kwargs(self) -> dict:
        return {
            "grid_size": self.grid_size,
            "workspace_size_m": self.workspace_size_m,
            "observation_grid": self.observation_grid,
            "peak_height_range_m": (
                self.peak_height_min_m, self.peak_height_max_m
            ),
            "target_remaining_fraction": self.target_remaining_fraction,
            "obj_path": self.obj_path,
        }

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SACConfig:
    timesteps: int = 20_000
    learning_rate: float = 3e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 256
    gamma: float = 0.995
    tau: float = 0.005
    hidden_sizes: tuple[int, ...] = (256, 256)
    seed: int = 20261101
    device: str = "auto"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        return data


def resolve_output(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
