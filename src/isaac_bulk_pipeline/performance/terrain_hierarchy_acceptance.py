"""Strict acceptance schema for the hierarchical terrain backends.

The schema deliberately separates physics equivalence from computational
feasibility.  A backend cannot pass by silently executing a CPU fallback, and
an interrupted or non-settled terrain state cannot be used as final terrain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np


class TerrainAcceptancePath(str, Enum):
    CPU_REFERENCE = "CPU_REFERENCE"
    GPU_OPTIMIZED = "GPU_OPTIMIZED"
    LARGE_AVALANCHE_MOBILE_PATH = "LARGE_AVALANCHE_MOBILE_PATH"


class TerrainAcceptanceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class TerrainPathMetrics:
    """Measured result for one explicitly named execution path."""

    path: TerrainAcceptancePath
    status: TerrainAcceptanceStatus
    case_name: str
    resolution_m: float
    simulated_time_s: float
    wall_time_s: float
    final_terrain_linf_error_m: float
    final_terrain_rmse_m: float
    volume_balance_error_m3: float
    momentum_balance_error_kg_m_s: np.ndarray
    soil_impulse_terrain_ns: np.ndarray
    soil_impulse_error_ns: np.ndarray
    gpu_utilization_percent_mean: float | None = None
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    synchronization_count: int = 0
    arrays_resident_on_device: bool = False
    backend_identity: str = ""
    failure_reason: str | None = None
    parameter_status: str = "LITERATURE_BASED_REDUCED_ORDER_UNCALIBRATED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", TerrainAcceptancePath(self.path))
        object.__setattr__(self, "status", TerrainAcceptanceStatus(self.status))
        if not isinstance(self.case_name, str) or not self.case_name:
            raise ValueError("[TerrainAcceptance] case_name must be non-empty")
        finite_nonnegative = {
            "resolution_m": self.resolution_m,
            "simulated_time_s": self.simulated_time_s,
            "wall_time_s": self.wall_time_s,
            "final_terrain_linf_error_m": self.final_terrain_linf_error_m,
            "final_terrain_rmse_m": self.final_terrain_rmse_m,
        }
        for name, value in finite_nonnegative.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"[TerrainAcceptance] {name} must be finite/non-negative")
        if self.resolution_m <= 0.0:
            raise ValueError("[TerrainAcceptance] resolution_m must be positive")
        if not np.isfinite(self.volume_balance_error_m3):
            raise ValueError("[TerrainAcceptance] volume error must be finite")
        for name in (
            "momentum_balance_error_kg_m_s",
            "soil_impulse_terrain_ns",
            "soil_impulse_error_ns",
        ):
            vector = np.asarray(getattr(self, name), dtype=np.float64)
            if vector.shape != (3,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"[TerrainAcceptance] {name} must be finite shape (3,)")
            copy = np.ascontiguousarray(vector.copy())
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)
        for name in ("h2d_bytes", "d2h_bytes", "synchronization_count"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"[TerrainAcceptance] {name} must be non-negative")
            object.__setattr__(self, name, value)
        utilization = self.gpu_utilization_percent_mean
        if utilization is not None and (
            not np.isfinite(utilization) or not 0.0 <= utilization <= 100.0
        ):
            raise ValueError("[TerrainAcceptance] GPU utilization must be in [0,100]")
        if self.path is TerrainAcceptancePath.GPU_OPTIMIZED:
            if self.status is TerrainAcceptanceStatus.PASS:
                if not self.arrays_resident_on_device:
                    raise ValueError(
                        "[TerrainAcceptance] GPU PASS requires device-resident arrays"
                    )
                if utilization is None:
                    raise ValueError(
                        "[TerrainAcceptance] GPU PASS requires measured utilization"
                    )
                if "GPU" not in self.backend_identity.upper() and "WARP" not in self.backend_identity.upper():
                    raise ValueError(
                        "[TerrainAcceptance] GPU PASS backend identity is not a GPU backend"
                    )
        elif self.arrays_resident_on_device:
            raise ValueError(
                "[TerrainAcceptance] only GPU_OPTIMIZED may claim device-resident arrays"
            )
        if self.status in {
            TerrainAcceptanceStatus.FAIL,
            TerrainAcceptanceStatus.UNAVAILABLE,
        } and not self.failure_reason:
            raise ValueError(
                "[TerrainAcceptance] failed/unavailable result requires failure_reason"
            )

    @property
    def real_time_factor(self) -> float:
        if self.wall_time_s <= 0.0:
            return float("inf") if self.simulated_time_s > 0.0 else 0.0
        return float(self.simulated_time_s / self.wall_time_s)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path.value,
            "status": self.status.value,
            "case_name": self.case_name,
            "resolution_m": self.resolution_m,
            "simulated_time_s": self.simulated_time_s,
            "wall_time_s": self.wall_time_s,
            "rtf": self.real_time_factor,
            "final_terrain_error": {
                "linf_m": self.final_terrain_linf_error_m,
                "rmse_m": self.final_terrain_rmse_m,
            },
            "volume_balance_error_m3": self.volume_balance_error_m3,
            "momentum_balance_error_kg_m_s": self.momentum_balance_error_kg_m_s.tolist(),
            "soil_impulse_terrain_ns": self.soil_impulse_terrain_ns.tolist(),
            "soil_impulse_error_ns": self.soil_impulse_error_ns.tolist(),
            "gpu_utilization_percent_mean": self.gpu_utilization_percent_mean,
            "h2d_bytes": self.h2d_bytes,
            "d2h_bytes": self.d2h_bytes,
            "synchronization_count": self.synchronization_count,
            "arrays_resident_on_device": self.arrays_resident_on_device,
            "backend_identity": self.backend_identity,
            "failure_reason": self.failure_reason,
            "parameter_status": self.parameter_status,
        }


@dataclass(frozen=True)
class TerrainHierarchyAcceptanceReport:
    """One case with all three mandatory execution paths reported separately."""

    case_name: str
    results: tuple[TerrainPathMetrics, ...]
    formal_resolution_m: float = 0.05

    def __post_init__(self) -> None:
        rows = tuple(self.results)
        object.__setattr__(self, "results", rows)
        paths = [row.path for row in rows]
        required = set(TerrainAcceptancePath)
        if set(paths) != required or len(paths) != len(required):
            raise ValueError(
                "[TerrainAcceptance] report requires exactly one CPU_REFERENCE, "
                "GPU_OPTIMIZED and LARGE_AVALANCHE_MOBILE_PATH result"
            )
        if not np.isclose(self.formal_resolution_m, 0.05, atol=0.0, rtol=0.0):
            raise ValueError("[TerrainAcceptance] formal resolution must remain 0.05 m")
        if any(
            not np.isclose(row.resolution_m, self.formal_resolution_m, atol=0.0, rtol=0.0)
            for row in rows
        ):
            raise ValueError("[TerrainAcceptance] a path lowered the formal resolution")
        if any(row.case_name != self.case_name for row in rows):
            raise ValueError("[TerrainAcceptance] result case names do not match")

    @property
    def overall_status(self) -> str:
        return (
            "PASS"
            if all(row.status is TerrainAcceptanceStatus.PASS for row in self.results)
            else "NOT_PASSED"
        )

    def to_dict(self) -> dict[str, Any]:
        by_path = {row.path.value: row.to_dict() for row in self.results}
        return {
            "schema": "terrain-hierarchy-acceptance/v1",
            "case_name": self.case_name,
            "formal_resolution_m": self.formal_resolution_m,
            "overall_status": self.overall_status,
            "results": by_path,
        }

    @classmethod
    def build(
        cls,
        case_name: str,
        results: Iterable[TerrainPathMetrics],
    ) -> "TerrainHierarchyAcceptanceReport":
        return cls(case_name=case_name, results=tuple(results))
