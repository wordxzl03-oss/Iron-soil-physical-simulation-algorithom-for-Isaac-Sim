"""Typed loader for the additive Phase-C contact configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .backend import ContactBackendConfig
from .collision_filter import CollisionGroup, CollisionMatrix


@dataclass(frozen=True)
class WheelTerrainConfig:
    deformation_enabled: bool = False

    def __post_init__(self) -> None:
        if self.deformation_enabled:
            raise ValueError("[PhaseCConfig] wheel terrain deformation is not implemented")


@dataclass(frozen=True)
class PhaseCContactConfig:
    backend_type: str
    backend: ContactBackendConfig
    collision_matrix: CollisionMatrix
    group_prim_root: str
    wheel_terrain: WheelTerrainConfig
    slope_test_degrees: tuple[float, ...]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"[PhaseCConfig] {name} must be a mapping")
    return value


def load_phase_c_contact_config(path: str | Path) -> PhaseCContactConfig:
    source = Path(path)
    root = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "root")
    contact = _mapping(root.get("contact", {}), "contact")
    backend_type = str(contact.get("backend", "triangle_mesh"))
    if backend_type != "triangle_mesh":
        raise ValueError(
            "[PhaseCConfig] only triangle_mesh is enabled until HeightField capability passes"
        )
    commit = _mapping(contact.get("commit", {}), "contact.commit")
    backend = ContactBackendConfig(
        contact_prim_path=str(contact.get("prim_path", "/World/TerrainContact")),
        visual_prim_path=str(contact.get("visual_prim_path", "/World/TerrainVisual")),
        target_spacing_m=float(contact.get("target_spacing_m", 0.10)),
        commit_policy=str(commit.get("policy", "action_end")),  # type: ignore[arg-type]
        low_frequency_hz=float(commit.get("low_frequency_hz", 2.0)),
        visible=bool(contact.get("visible", False)),
        rigid_body_enabled=bool(contact.get("rigid_body_enabled", False)),
    )
    filtering = _mapping(root.get("collision_filtering", {}), "collision_filtering")
    group_root = str(filtering.get("group_prim_root", "/World/CollisionGroups"))
    if not group_root.startswith("/"):
        raise ValueError("[PhaseCConfig] group_prim_root must be absolute")
    matrix = CollisionMatrix()
    requested_pair = filtering.get(
        "disabled_pairs", [["TerrainSupport", "BucketInteraction"]]
    )
    if requested_pair != [["TerrainSupport", "BucketInteraction"]]:
        raise ValueError(
            "[PhaseCConfig] Phase C requires exactly TerrainSupport/BucketInteraction filtering"
        )
    wheel_values = _mapping(root.get("wheel_terrain", {}), "wheel_terrain")
    wheel = WheelTerrainConfig(
        deformation_enabled=bool(wheel_values.get("deformation_enabled", False))
    )
    slope_values = _mapping(root.get("slope_tests", {}), "slope_tests")
    degrees = tuple(float(value) for value in slope_values.get("degrees", (0, 10, 20)))
    if not degrees or any(value < 0.0 or value >= 45.0 for value in degrees):
        raise ValueError("[PhaseCConfig] slope test degrees must be in [0,45)")
    return PhaseCContactConfig(
        backend_type=backend_type,
        backend=backend,
        collision_matrix=matrix,
        group_prim_root=group_root,
        wheel_terrain=wheel,
        slope_test_degrees=degrees,
    )
