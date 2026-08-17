"""Explicit collision matrix and validation for custom soil interaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Iterable, Mapping


class CollisionGroup(str, Enum):
    TERRAIN_SUPPORT = "TerrainSupport"
    WHEEL_CONTACT = "WheelContact"
    CHASSIS_CONTACT = "ChassisContact"
    BUCKET_INTERACTION = "BucketInteraction"
    ENVIRONMENT_STATIC = "EnvironmentStatic"


ALL_COLLISION_GROUPS = tuple(CollisionGroup)


@dataclass(frozen=True)
class CollisionDecision:
    first: CollisionGroup
    second: CollisionGroup
    enabled: bool


@dataclass(frozen=True)
class CollisionValidation:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def require_valid(self) -> None:
        if not self.valid:
            raise ValueError("[CollisionFilter] " + "; ".join(self.errors))


class CollisionMatrix:
    """Symmetric matrix with bucket/terrain as the sole soil-filtered pair."""

    def __init__(self, decisions: Iterable[CollisionDecision] | None = None) -> None:
        self._values: dict[tuple[CollisionGroup, CollisionGroup], bool] = {}
        for first in ALL_COLLISION_GROUPS:
            for second in ALL_COLLISION_GROUPS:
                self._values[(first, second)] = True
        self._set_symmetric(
            CollisionGroup.TERRAIN_SUPPORT,
            CollisionGroup.BUCKET_INTERACTION,
            False,
        )
        if decisions is not None:
            for decision in decisions:
                self._set_symmetric(decision.first, decision.second, decision.enabled)
        validate_collision_matrix(self).require_valid()

    def _set_symmetric(
        self,
        first: CollisionGroup,
        second: CollisionGroup,
        enabled: bool,
    ) -> None:
        self._values[(first, second)] = bool(enabled)
        self._values[(second, first)] = bool(enabled)

    def allows(self, first: CollisionGroup | str, second: CollisionGroup | str) -> bool:
        return self._values[(CollisionGroup(first), CollisionGroup(second))]

    def filtered_pairs(self) -> tuple[tuple[CollisionGroup, CollisionGroup], ...]:
        result: list[tuple[CollisionGroup, CollisionGroup]] = []
        for index, first in enumerate(ALL_COLLISION_GROUPS):
            for second in ALL_COLLISION_GROUPS[index + 1 :]:
                if not self.allows(first, second):
                    result.append((first, second))
        return tuple(result)

    def as_dict(self) -> dict[str, dict[str, bool]]:
        return {
            first.value: {
                second.value: self.allows(first, second)
                for second in ALL_COLLISION_GROUPS
            }
            for first in ALL_COLLISION_GROUPS
        }


def validate_collision_matrix(matrix: CollisionMatrix) -> CollisionValidation:
    errors: list[str] = []
    warnings: list[str] = []
    for first in ALL_COLLISION_GROUPS:
        for second in ALL_COLLISION_GROUPS:
            if matrix._values.get((first, second)) != matrix._values.get((second, first)):
                errors.append(f"matrix is asymmetric for {first.value}/{second.value}")
    required = {
        (CollisionGroup.WHEEL_CONTACT, CollisionGroup.TERRAIN_SUPPORT): True,
        (CollisionGroup.CHASSIS_CONTACT, CollisionGroup.TERRAIN_SUPPORT): True,
        (CollisionGroup.BUCKET_INTERACTION, CollisionGroup.TERRAIN_SUPPORT): False,
        (CollisionGroup.ENVIRONMENT_STATIC, CollisionGroup.WHEEL_CONTACT): True,
        (CollisionGroup.ENVIRONMENT_STATIC, CollisionGroup.CHASSIS_CONTACT): True,
        (CollisionGroup.ENVIRONMENT_STATIC, CollisionGroup.BUCKET_INTERACTION): True,
    }
    for (first, second), expected in required.items():
        actual = matrix.allows(first, second)
        if actual != expected:
            errors.append(
                f"{first.value}<->{second.value} expected={expected}, actual={actual}"
            )
    unexpected = [
        (a.value, b.value)
        for a, b in matrix.filtered_pairs()
        if {a, b}
        != {CollisionGroup.TERRAIN_SUPPORT, CollisionGroup.BUCKET_INTERACTION}
    ]
    if unexpected:
        warnings.append(f"additional filtered pairs are configured: {unexpected}")
    return CollisionValidation(not errors, tuple(errors), tuple(warnings))


def validate_collision_memberships(
    memberships: Mapping[CollisionGroup | str, Iterable[str]],
    *,
    contact_prim_path: str,
    visual_prim_path: str,
    require_nonempty: bool = True,
) -> CollisionValidation:
    """Validate collider-collection membership before USD authoring."""

    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[CollisionGroup, tuple[str, ...]] = {}
    for group in ALL_COLLISION_GROUPS:
        values = tuple(str(path) for path in memberships.get(group, memberships.get(group.value, ())))
        normalized[group] = values
        if require_nonempty and not values:
            errors.append(f"{group.value} has no collider members")
        for value in values:
            if not value.startswith("/") or str(PurePosixPath(value)) != value:
                errors.append(f"{group.value} has invalid absolute prim path: {value!r}")

    owners: dict[str, list[CollisionGroup]] = {}
    for group, paths in normalized.items():
        for path in paths:
            owners.setdefault(path, []).append(group)
    for path, groups in owners.items():
        if len(groups) > 1:
            errors.append(
                f"collider {path} belongs to multiple groups: "
                + ", ".join(group.value for group in groups)
            )

    terrain_members = normalized[CollisionGroup.TERRAIN_SUPPORT]
    if terrain_members != (contact_prim_path,):
        errors.append(
            "TerrainSupport must contain exactly the contact prim; "
            f"expected={(contact_prim_path,)}, actual={terrain_members}"
        )
    if any(
        path == visual_prim_path or path.startswith(visual_prim_path + "/")
        for paths in normalized.values()
        for path in paths
    ):
        errors.append("visual terrain prim must not belong to any collision group")
    return CollisionValidation(not errors, tuple(errors), tuple(warnings))
