"""Pure-Python validation for the Phase-A WheelLoader USD inventory.

The Isaac probe intentionally lives outside the production simulation path.  This
module contains no USD or Isaac imports so its output contract can be regression
tested with the normal project Python environment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


INVENTORY_SCHEMA_VERSION = "phase_a_usd_inventory.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"[USDInventory] {message}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_authored_value(record: Any, location: str) -> None:
    _require(isinstance(record, Mapping), f"{location} must be a mapping")
    _require(set(record) == {"authored", "value"}, f"{location} has invalid keys")
    _require(isinstance(record["authored"], bool), f"{location}.authored must be bool")


def _validate_sorted_unique_paths(items: Any, location: str) -> None:
    _require(isinstance(items, list), f"{location} must be a list")
    paths: list[str] = []
    for index, item in enumerate(items):
        _require(isinstance(item, Mapping), f"{location}[{index}] must be a mapping")
        path = item.get("path")
        _require(
            isinstance(path, str) and path.startswith("/"),
            f"{location}[{index}].path must be an absolute USD path",
        )
        paths.append(path)
    _require(paths == sorted(paths), f"{location} must be sorted by path")
    _require(len(paths) == len(set(paths)), f"{location} contains duplicate paths")


def validate_inventory(document: Mapping[str, Any]) -> None:
    """Validate one deterministic inventory document or raise ``ValueError``."""

    _require(isinstance(document, Mapping), "document must be a mapping")
    required = {
        "schema_version",
        "source_asset",
        "runtime",
        "stage",
        "summary",
        "prims",
        "articulation_roots",
        "rigid_bodies",
        "joints",
        "colliders",
        "collision_groups",
        "filtered_pairs",
        "physics_materials",
        "physx_attributes",
    }
    missing = sorted(required - set(document))
    _require(not missing, f"missing top-level keys={missing}")
    _require(
        document["schema_version"] == INVENTORY_SCHEMA_VERSION,
        f"schema_version must be {INVENTORY_SCHEMA_VERSION!r}",
    )

    source = document["source_asset"]
    _require(isinstance(source, Mapping), "source_asset must be a mapping")
    _require(
        isinstance(source.get("path"), str) and bool(source["path"]),
        "source_asset.path must be non-empty",
    )
    digest = source.get("sha256")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "source_asset.sha256 must be a lowercase SHA-256 digest",
    )
    _require(
        isinstance(source.get("size_bytes"), int) and source["size_bytes"] > 0,
        "source_asset.size_bytes must be a positive integer",
    )

    stage = document["stage"]
    _require(isinstance(stage, Mapping), "stage must be a mapping")
    for key in ("meters_per_unit", "kilograms_per_unit", "time_codes_per_second"):
        value = stage.get(key)
        _require(
            _is_number(value) and math.isfinite(float(value)) and float(value) > 0.0,
            f"stage.{key} must be finite and positive",
        )
    _require(stage.get("up_axis") in {"X", "Y", "Z"}, "stage.up_axis is invalid")

    collection_names = (
        "prims",
        "articulation_roots",
        "rigid_bodies",
        "joints",
        "colliders",
        "collision_groups",
        "filtered_pairs",
        "physics_materials",
        "physx_attributes",
    )
    for name in collection_names:
        _validate_sorted_unique_paths(document[name], name)

    summary = document["summary"]
    _require(isinstance(summary, Mapping), "summary must be a mapping")
    count_mapping = {
        "prim_count": "prims",
        "articulation_root_count": "articulation_roots",
        "rigid_body_count": "rigid_bodies",
        "joint_count": "joints",
        "collider_count": "colliders",
        "collision_group_count": "collision_groups",
        "filtered_pair_owner_count": "filtered_pairs",
        "physics_material_count": "physics_materials",
        "physx_attribute_owner_count": "physx_attributes",
    }
    for count_key, collection_name in count_mapping.items():
        _require(
            summary.get(count_key) == len(document[collection_name]),
            f"summary.{count_key} does not match {collection_name}",
        )

    for joint_index, joint in enumerate(document["joints"]):
        location = f"joints[{joint_index}]"
        drives = joint.get("drives")
        _require(isinstance(drives, list), f"{location}.drives must be a list")
        instances: list[str] = []
        for drive_index, drive in enumerate(drives):
            drive_location = f"{location}.drives[{drive_index}]"
            _require(isinstance(drive, Mapping), f"{drive_location} must be a mapping")
            instance = drive.get("instance")
            _require(isinstance(instance, str) and instance, f"{drive_location}.instance")
            instances.append(instance)
            for field in (
                "type",
                "stiffness",
                "damping",
                "max_force",
                "target_position",
                "target_velocity",
            ):
                _validate_authored_value(drive.get(field), f"{drive_location}.{field}")
                value = drive[field]["value"]
                if field != "type" and value is not None:
                    _require(
                        _is_number(value) and math.isfinite(float(value)),
                        f"{drive_location}.{field}.value must be finite numeric or null",
                    )
        _require(instances == sorted(set(instances)), f"{location}.drives is not stable")

    for collider_index, collider in enumerate(document["colliders"]):
        location = f"colliders[{collider_index}]"
        _validate_authored_value(collider.get("collision_enabled"), f"{location}.collision_enabled")
        _require(
            isinstance(collider.get("applied_schemas"), list),
            f"{location}.applied_schemas must be a list",
        )

    for item in document["physx_attributes"]:
        attributes = item.get("attributes")
        _require(isinstance(attributes, Mapping) and attributes, "empty PhysX attribute owner")
        _require(
            list(attributes) == sorted(attributes),
            f"PhysX attributes are not sorted at {item['path']}",
        )
        for name, record in attributes.items():
            _require(name.startswith("physx"), f"invalid PhysX attribute name={name!r}")
            _validate_authored_value(record, f"physx_attributes[{item['path']}].{name}")
