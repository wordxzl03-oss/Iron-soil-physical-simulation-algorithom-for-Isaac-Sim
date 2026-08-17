"""Backend-neutral lifecycle for terrain contact representations.

The authoritative state remains the height map owned by the terrain model.  A
contact backend only derives a representation that an engine may cook.  USD and
PhysX are deliberately absent from this module so lifecycle and commit policy
can be tested with ordinary CPython.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..terrain.terrain_grid import TerrainGrid


CommitReason = Literal["action_end", "low_frequency", "reset"]


@dataclass(frozen=True)
class ContactBackendConfig:
    """Configuration shared by terrain-contact backends.

    ``contact_prim_path`` and ``visual_prim_path`` must differ: this is the
    executable guard that prevents the RTX mesh becoming the physics state.
    """

    contact_prim_path: str = "/World/TerrainContact"
    visual_prim_path: str = "/World/TerrainVisual"
    target_spacing_m: float = 0.10
    commit_policy: Literal["action_end", "low_frequency"] = "action_end"
    low_frequency_hz: float = 2.0
    visible: bool = False
    rigid_body_enabled: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("contact_prim_path", self.contact_prim_path),
            ("visual_prim_path", self.visual_prim_path),
        ):
            if not isinstance(value, str) or not value.startswith("/"):
                raise ValueError(f"[ContactConfig] {name} must be an absolute USD path")
        if self.contact_prim_path == self.visual_prim_path:
            raise ValueError(
                "[ContactConfig] contact and visual prim paths must be different"
            )
        if not np.isfinite(self.target_spacing_m) or self.target_spacing_m <= 0.0:
            raise ValueError("[ContactConfig] target_spacing_m must be positive")
        if self.commit_policy not in {"action_end", "low_frequency"}:
            raise ValueError(
                "[ContactConfig] commit_policy must be action_end or low_frequency"
            )
        if not np.isfinite(self.low_frequency_hz) or self.low_frequency_hz <= 0.0:
            raise ValueError("[ContactConfig] low_frequency_hz must be positive")
        if self.visible:
            raise ValueError("[ContactConfig] contact representation must stay hidden")
        if self.rigid_body_enabled:
            raise ValueError("[ContactConfig] terrain contact must be a static collider")


@dataclass(frozen=True)
class ContactCommitResult:
    """Outcome of publishing one pending contact representation."""

    committed: bool
    generation: int
    reason: str
    vertex_count: int
    triangle_count: int
    recook_required: bool
    rate_limited: bool = False


class TerrainContactBackend(ABC):
    """Interface for a view derived from authoritative resting terrain."""

    @abstractmethod
    def initialize(self, grid: TerrainGrid, heightmap: np.ndarray) -> None:
        """Build and publish an initial contact representation."""

    @abstractmethod
    def update_from_heightmap(
        self,
        heightmap: np.ndarray,
        *,
        action_index: int | None = None,
        sim_time_s: float | None = None,
    ) -> None:
        """Build a pending representation without recooking engine state."""

    @abstractmethod
    def commit(
        self,
        *,
        reason: CommitReason = "action_end",
        sim_time_s: float | None = None,
    ) -> ContactCommitResult:
        """Publish pending data at an action boundary or configured low rate."""

    @abstractmethod
    def reset(self) -> ContactCommitResult:
        """Restore the initialized contact representation."""
