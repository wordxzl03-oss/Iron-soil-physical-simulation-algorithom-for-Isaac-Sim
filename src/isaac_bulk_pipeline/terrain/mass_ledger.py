"""Geometric terrain-volume ledger with explicitly estimated density outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..config import MaterialConfig
from ..interaction import ExcavationResult
from ..solvers import RelaxationResult
from .terrain_grid import TerrainGrid


@dataclass
class MassLedger:
    """Track volume balance; payload and mass are not treated as measured truth."""

    initial_terrain_volume_m3: float
    current_terrain_volume_m3: float
    removed_volume_m3: float = 0.0
    boundary_outflow_m3: float = 0.0
    numerical_error_m3: float = 0.0
    payload_volume_m3: float = 0.0
    spill_volume_m3: float = 0.0
    deposited_volume_m3: float = 0.0
    exported_volume_m3: float = 0.0
    airborne_volume_m3: float = 0.0
    bulk_density_kg_m3: float | None = None
    density_is_estimated: bool = True

    @classmethod
    def initialize(
        cls,
        grid: TerrainGrid,
        initial_heightmap: np.ndarray,
        material: MaterialConfig | None = None,
    ) -> "MassLedger":
        """Construct a zero-loss ledger from authoritative terrain."""

        volume = grid.compute_volume(initial_heightmap)
        settings = material or MaterialConfig()
        return cls(
            initial_terrain_volume_m3=volume,
            current_terrain_volume_m3=volume,
            bulk_density_kg_m3=settings.bulk_density_kg_m3,
            density_is_estimated=settings.density_is_estimated,
        )

    def record_excavation(
        self,
        result: ExcavationResult,
        grid: TerrainGrid,
    ) -> None:
        """Accumulate geometric removal and recompute balance from the new map."""

        self.removed_volume_m3 += float(result.removed_volume_m3)
        self.current_terrain_volume_m3 = grid.compute_volume(
            result.heightmap_excavated
        )
        self._recompute_error()

    def record_relaxation(self, result: RelaxationResult) -> None:
        """Accumulate open-boundary loss and accept solver stable volume."""

        self.boundary_outflow_m3 += float(result.boundary_outflow_m3)
        self.current_terrain_volume_m3 = float(result.volume_after_m3)
        self._recompute_error()

    def reset(self, grid: TerrainGrid, initial_heightmap: np.ndarray) -> None:
        """Restore the initial volume and zero all transfer reservoirs."""

        volume = grid.compute_volume(initial_heightmap)
        self.initial_terrain_volume_m3 = volume
        self.current_terrain_volume_m3 = volume
        self.removed_volume_m3 = 0.0
        self.boundary_outflow_m3 = 0.0
        self.numerical_error_m3 = 0.0
        self.payload_volume_m3 = 0.0
        self.spill_volume_m3 = 0.0
        self.deposited_volume_m3 = 0.0
        self.exported_volume_m3 = 0.0
        self.airborne_volume_m3 = 0.0

    @property
    def removed_mass_estimate_kg(self) -> float | None:
        """Return a density-based estimate, never a measured bucket payload."""

        if self.bulk_density_kg_m3 is None:
            return None
        return float(self.removed_volume_m3 * self.bulk_density_kg_m3)

    def to_mapping(self) -> dict[str, Any]:
        """Serialize ledger values with an explicit mass-estimate disclaimer."""

        values = asdict(self)
        values.update(
            {
                "removed_mass_estimate_kg": self.removed_mass_estimate_kg,
                "mass_value_kind": "density-based estimate",
                "is_measured_bucket_payload": False,
            }
        )
        return values

    def _recompute_error(self) -> None:
        self.numerical_error_m3 = float(
            self.initial_terrain_volume_m3
            - self.current_terrain_volume_m3
            - self.removed_volume_m3
            - self.boundary_outflow_m3
            + self.deposited_volume_m3
        )
        values = np.asarray(
            [
                self.initial_terrain_volume_m3,
                self.current_terrain_volume_m3,
                self.removed_volume_m3,
                self.boundary_outflow_m3,
                self.numerical_error_m3,
            ]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("[MassLedger] non-finite volume balance detected")
