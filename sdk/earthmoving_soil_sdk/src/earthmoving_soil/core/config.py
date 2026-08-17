"""Versioned configuration facade for the extracted production soil core."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from isaac_bulk_pipeline.bulk_interaction import (
    LargeAvalancheTransitionConfig,
    TrackSoilConfig,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, UNCALIBRATED_LABEL
from isaac_bulk_pipeline.terrain import TerrainGrid


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class MaterialConfig:
    """Uniform-density material profile used by the production core."""

    profile: str
    bulk_density_kg_m3: float
    internal_friction_angle_deg: float
    cohesion_pa: float
    tool_friction_coefficient: float
    start_angle_deg: float
    stop_angle_deg: float
    mobile_friction_coefficient: float
    calibration_status: str = "NOT_YET_PHYSICALLY_CALIBRATED"
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.calibration_status != "NOT_YET_PHYSICALLY_CALIBRATED":
            raise ValueError("alpha accepts only NOT_YET_PHYSICALLY_CALIBRATED profiles")
        if not self.profile or self.bulk_density_kg_m3 <= 0.0:
            raise ValueError("material profile/density is invalid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MaterialConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        values = raw.get("material", raw)
        if not isinstance(values, Mapping):
            raise TypeError("material YAML must contain a mapping")
        return cls(**dict(values))

    def to_scenario(self) -> MaterialScenario:
        return MaterialScenario(
            name=self.profile,
            assumed_bulk_density_kg_m3=self.bulk_density_kg_m3,
            internal_friction_angle_deg=self.internal_friction_angle_deg,
            cohesion_proxy_pa=self.cohesion_pa,
            tool_friction_coefficient=self.tool_friction_coefficient,
            start_angle_deg=self.start_angle_deg,
            stop_angle_deg=self.stop_angle_deg,
            mobile_friction_coefficient=self.mobile_friction_coefficient,
            calibration_status=UNCALIBRATED_LABEL,
        )

    @property
    def config_hash(self) -> str:
        return _stable_hash({
            "profile": self.profile,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
            "internal_friction_angle_deg": self.internal_friction_angle_deg,
            "cohesion_pa": self.cohesion_pa,
            "tool_friction_coefficient": self.tool_friction_coefficient,
            "start_angle_deg": self.start_angle_deg,
            "stop_angle_deg": self.stop_angle_deg,
            "mobile_friction_coefficient": self.mobile_friction_coefficient,
            "calibration_status": self.calibration_status,
        })


@dataclass(frozen=True)
class SoilConfig:
    """Vehicle-independent solver/runtime settings; SI units throughout."""

    grid: TerrainGrid
    material: MaterialConfig
    runtime_backend: str = "HOST_REFERENCE"
    gpu_device: str = "cuda:0"
    solver_backend: str = "OPTIMIZED"
    slope_backend: str = "CPU_OPTIMIZED_COMPACT_TILE_FRONTIER"
    tile_size: int = 64
    physics_dt_s: float = 1.0 / 60.0
    minislope_round_budget_per_step: int = 1
    minislope_tolerance_m: float = 0.002
    numerical_safety_max_iterations: int = 1_000_000
    mass_tolerance_m3: float = 1.0e-8
    terrain_patch_radius_cells: int = 8
    large_avalanche: LargeAvalancheTransitionConfig = field(
        default_factory=LargeAvalancheTransitionConfig
    )
    track_soil: TrackSoilConfig = field(default_factory=TrackSoilConfig)

    def __post_init__(self) -> None:
        if self.runtime_backend not in {"HOST_REFERENCE", "GPU_RUNTIME"}:
            raise ValueError("runtime_backend must be HOST_REFERENCE or GPU_RUNTIME")
        if self.physics_dt_s <= 0.0 or self.tile_size < 8:
            raise ValueError("invalid physics_dt_s/tile_size")

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        grid: TerrainGrid,
        material: MaterialConfig,
    ) -> "SoilConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        values = dict(raw.get("solver", {}))
        values.update(dict(raw.get("runtime", {})))
        avalanche = LargeAvalancheTransitionConfig.from_mapping(
            dict(raw.get("large_avalanche_transition", {}))
        )
        track = TrackSoilConfig(**dict(raw.get("track_soil", {})))
        return cls(grid=grid, material=material, large_avalanche=avalanche,
                   track_soil=track, **values)

    @property
    def solver_config_hash(self) -> str:
        return _stable_hash({
            "solver_backend": self.solver_backend,
            "slope_backend": self.slope_backend,
            "tile_size": self.tile_size,
            "physics_dt_s": self.physics_dt_s,
            "minislope_round_budget_per_step": self.minislope_round_budget_per_step,
            "minislope_tolerance_m": self.minislope_tolerance_m,
            "numerical_safety_max_iterations": self.numerical_safety_max_iterations,
            "mass_tolerance_m3": self.mass_tolerance_m3,
        })

    @property
    def runtime_config_hash(self) -> str:
        return _stable_hash({
            "runtime_backend": self.runtime_backend,
            "gpu_device": self.gpu_device,
            "state_authority": self.state_authority,
        })

    @property
    def state_authority(self) -> str:
        return "DEVICE" if self.runtime_backend == "GPU_RUNTIME" else "HOST"

