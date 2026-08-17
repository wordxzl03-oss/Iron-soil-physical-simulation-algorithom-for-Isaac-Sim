"""Reproducibility metadata wrapping observational production diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .._version import PACKAGE_VERSION, PHYSICS_CORE_VERSION


@dataclass(frozen=True)
class PhysicsDiagnostics:
    package_version: str
    physics_core_version: str
    git_commit: str | None
    material_profile: str
    material_config_hash: str
    solver_config_hash: str
    runtime_backend: str
    state_authority: str
    flow_arrest_closure: str
    material_calibration: str
    production: Any

    @classmethod
    def from_step(cls, config: Any, production: Any) -> "PhysicsDiagnostics":
        return cls(
            package_version=PACKAGE_VERSION,
            physics_core_version=PHYSICS_CORE_VERSION,
            git_commit=os.environ.get("EARTHMOVING_SOIL_GIT_COMMIT"),
            material_profile=config.material.profile,
            material_config_hash=config.material.config_hash,
            solver_config_hash=config.solver_config_hash,
            runtime_backend=config.runtime_backend,
            state_authority=config.state_authority,
            flow_arrest_closure="FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED",
            material_calibration="NOT_YET_PHYSICALLY_CALIBRATED",
            production=production,
        )

