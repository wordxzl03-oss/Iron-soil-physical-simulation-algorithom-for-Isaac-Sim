"""Read a WheelLoader USD with Isaac Sim 4.5 and emit a stable JSON inventory.

This is a read-only Phase-A capability probe.  It never opens the asset for edit,
never saves the USD layer, and does not create a World or start physics.

Example::

    /home/eric/isaacsim/python.sh isaac_loader/probe_usd_inventory.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--usd",
        type=Path,
        default=Path(__file__).with_name("wheel_loader.usd"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "phase_a_usd_inventory.json",
    )
    args, kit_args = parser.parse_known_args()
    sys.argv = [sys.argv[0], *kit_args]
    return args


ARGS = _parse_args()

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from isaac_bulk_pipeline.audit import (  # noqa: E402
    INVENTORY_SCHEMA_VERSION,
    validate_inventory,
)

try:  # noqa: E402 - SimulationApp must start before importing pxr schemas.
    from isaacsim import SimulationApp
except ImportError:  # Isaac Sim 4.2 compatibility namespace.
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": True,
        "multi_gpu": False,
        "width": 64,
        "height": 64,
    }
)

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite USD value={value}")
        return value
    if hasattr(value, "pathString"):
        return str(value.pathString)
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        imaginary = value.GetImaginary()
        return [
            _json_value(float(value.GetReal())),
            *[_json_value(float(component)) for component in imaginary],
        ]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    try:
        return [_json_value(item) for item in value]
    except TypeError:
        text = str(value)
        if text:
            return text
        raise TypeError(f"unsupported USD value type={type(value).__name__}")


def _attribute_record(prim: Usd.Prim, name: str) -> dict[str, Any]:
    attribute = prim.GetAttribute(name)
    if not attribute:
        return {"authored": False, "value": None}
    authored = bool(attribute.HasAuthoredValueOpinion())
    return {
        # Schema fallbacks can contain sentinel values such as (-inf, -inf,
        # -inf) for an unspecified COM.  The audit contract records authored
        # asset state only; an unauthored field is represented unambiguously by
        # null instead of leaking schema implementation sentinels into JSON.
        "authored": authored,
        "value": _json_value(attribute.Get()) if authored else None,
    }


def _relationship_targets(prim: Usd.Prim, name: str) -> list[str]:
    relationship = prim.GetRelationship(name)
    if not relationship:
        return []
    return sorted(str(target) for target in relationship.GetTargets())


def _all_relationships(prim: Usd.Prim) -> dict[str, list[str]]:
    return {
        relationship.GetName(): sorted(
            str(target) for target in relationship.GetTargets()
        )
        for relationship in sorted(prim.GetRelationships(), key=lambda item: item.GetName())
        if relationship.GetTargets()
    }


def _physx_attributes(prim: Usd.Prim) -> dict[str, dict[str, Any]]:
    return {
        attribute.GetName(): _attribute_record(prim, attribute.GetName())
        for attribute in sorted(prim.GetAttributes(), key=lambda item: item.GetName())
        if attribute.GetName().startswith("physx")
    }


def _applied_schemas(prim: Usd.Prim) -> list[str]:
    return sorted(str(schema) for schema in prim.GetAppliedSchemas())


def _nearest_rigid_body_path(prim: Usd.Prim) -> str | None:
    candidate = prim
    while candidate and candidate.GetPath() != Usd.Prim().GetPath():
        if candidate.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(candidate.GetPath())
        candidate = candidate.GetParent()
    return None


def _xform_attributes(prim: Usd.Prim) -> dict[str, dict[str, Any]]:
    return {
        attribute.GetName(): _attribute_record(prim, attribute.GetName())
        for attribute in sorted(prim.GetAttributes(), key=lambda item: item.GetName())
        if attribute.GetName().startswith("xformOp")
    }


def _geometry_summary(prim: Usd.Prim) -> dict[str, Any]:
    type_name = prim.GetTypeName()
    if type_name == "Cube":
        return {"size": _attribute_record(prim, "size")}
    if type_name == "Cylinder":
        return {
            "axis": _attribute_record(prim, "axis"),
            "height": _attribute_record(prim, "height"),
            "radius": _attribute_record(prim, "radius"),
        }
    if type_name == "Mesh":
        points = prim.GetAttribute("points").Get() or []
        counts = prim.GetAttribute("faceVertexCounts").Get() or []
        return {
            "point_count": len(points),
            "face_count": len(counts),
            "approximation": _attribute_record(prim, "physics:approximation"),
        }
    return {}


def _drive_instances(prim: Usd.Prim) -> list[str]:
    prefix = "PhysicsDriveAPI:"
    return sorted(
        schema[len(prefix) :]
        for schema in _applied_schemas(prim)
        if schema.startswith(prefix)
    )


def _drive_inventory(prim: Usd.Prim, instance: str) -> dict[str, Any]:
    prefix = f"drive:{instance}:physics:"
    return {
        "instance": instance,
        "type": _attribute_record(prim, f"{prefix}type"),
        "stiffness": _attribute_record(prim, f"{prefix}stiffness"),
        "damping": _attribute_record(prim, f"{prefix}damping"),
        "max_force": _attribute_record(prim, f"{prefix}maxForce"),
        "target_position": _attribute_record(prim, f"{prefix}targetPosition"),
        "target_velocity": _attribute_record(prim, f"{prefix}targetVelocity"),
    }


def _joint_inventory(prim: Usd.Prim) -> dict[str, Any]:
    return {
        "path": str(prim.GetPath()),
        "name": prim.GetName(),
        "type_name": prim.GetTypeName(),
        "applied_schemas": _applied_schemas(prim),
        "body0": _relationship_targets(prim, "physics:body0"),
        "body1": _relationship_targets(prim, "physics:body1"),
        "axis": _attribute_record(prim, "physics:axis"),
        "lower_limit": _attribute_record(prim, "physics:lowerLimit"),
        "upper_limit": _attribute_record(prim, "physics:upperLimit"),
        "local_pos0": _attribute_record(prim, "physics:localPos0"),
        "local_pos1": _attribute_record(prim, "physics:localPos1"),
        "local_rot0": _attribute_record(prim, "physics:localRot0"),
        "local_rot1": _attribute_record(prim, "physics:localRot1"),
        "drives": [_drive_inventory(prim, name) for name in _drive_instances(prim)],
        "physx_attributes": _physx_attributes(prim),
    }


def _rigid_body_inventory(prim: Usd.Prim) -> dict[str, Any]:
    return {
        "path": str(prim.GetPath()),
        "type_name": prim.GetTypeName(),
        "applied_schemas": _applied_schemas(prim),
        "mass": _attribute_record(prim, "physics:mass"),
        "density": _attribute_record(prim, "physics:density"),
        "center_of_mass": _attribute_record(prim, "physics:centerOfMass"),
        "diagonal_inertia": _attribute_record(prim, "physics:diagonalInertia"),
        "principal_axes": _attribute_record(prim, "physics:principalAxes"),
        "rigid_body_enabled": _attribute_record(prim, "physics:rigidBodyEnabled"),
        "kinematic_enabled": _attribute_record(prim, "physics:kinematicEnabled"),
        "physx_attributes": _physx_attributes(prim),
    }


def _collider_inventory(prim: Usd.Prim) -> dict[str, Any]:
    material_relationships = {
        name: targets
        for name, targets in _all_relationships(prim).items()
        if "material" in name.lower()
    }
    return {
        "path": str(prim.GetPath()),
        "type_name": prim.GetTypeName(),
        "rigid_body_path": _nearest_rigid_body_path(prim),
        "applied_schemas": _applied_schemas(prim),
        "collision_enabled": _attribute_record(prim, "physics:collisionEnabled"),
        "geometry": _geometry_summary(prim),
        "xform_attributes": _xform_attributes(prim),
        "material_relationships": material_relationships,
        "physx_attributes": _physx_attributes(prim),
    }


def _collision_group_inventory(prim: Usd.Prim) -> dict[str, Any]:
    relationships = _all_relationships(prim)
    return {
        "path": str(prim.GetPath()),
        "type_name": prim.GetTypeName(),
        "applied_schemas": _applied_schemas(prim),
        "invert_filtered_groups": _attribute_record(
            prim, "physics:invertFilteredGroups"
        ),
        "merge_group_name": _attribute_record(prim, "physics:mergeGroupName"),
        "relationships": relationships,
        "physx_attributes": _physx_attributes(prim),
    }


def _physics_material_inventory(prim: Usd.Prim) -> dict[str, Any]:
    return {
        "path": str(prim.GetPath()),
        "type_name": prim.GetTypeName(),
        "applied_schemas": _applied_schemas(prim),
        "static_friction": _attribute_record(prim, "physics:staticFriction"),
        "dynamic_friction": _attribute_record(prim, "physics:dynamicFriction"),
        "restitution": _attribute_record(prim, "physics:restitution"),
        "density": _attribute_record(prim, "physics:density"),
        "physx_attributes": _physx_attributes(prim),
    }


def _runtime_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {"application": "Isaac Sim", "headless": True}
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        result["app_version"] = str(app.get_app_version())
        result["build_version"] = str(app.get_build_version())
    except Exception as exc:  # Version metadata must not block the asset probe.
        result["version_query_error"] = type(exc).__name__
    return result


def _portable_asset_identifier(asset_path: Path) -> str:
    """Return a deterministic logical identifier without embedding host paths."""

    try:
        return asset_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        # External audit targets retain only their filename.  The SHA-256 below
        # remains the authoritative identity; CLI stdout reports actual paths.
        return asset_path.name


def build_inventory(stage: Usd.Stage, asset_path: Path) -> dict[str, Any]:
    prims = sorted(stage.Traverse(), key=lambda item: str(item.GetPath()))
    prim_inventory = [
        {
            "path": str(prim.GetPath()),
            "parent_path": str(prim.GetParent().GetPath()),
            "type_name": prim.GetTypeName(),
            "active": bool(prim.IsActive()),
            "instance": bool(prim.IsInstance()),
            "applied_schemas": _applied_schemas(prim),
            "relationships": _all_relationships(prim),
            "xform_attributes": _xform_attributes(prim),
        }
        for prim in prims
    ]
    articulation_roots = [
        {
            "path": str(prim.GetPath()),
            "type_name": prim.GetTypeName(),
            "applied_schemas": _applied_schemas(prim),
            "physx_attributes": _physx_attributes(prim),
        }
        for prim in prims
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    rigid_bodies = [
        _rigid_body_inventory(prim)
        for prim in prims
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    joints = [
        _joint_inventory(prim)
        for prim in prims
        if prim.IsA(UsdPhysics.Joint)
    ]
    colliders = [
        _collider_inventory(prim)
        for prim in prims
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    collision_groups = [
        _collision_group_inventory(prim)
        for prim in prims
        if prim.IsA(UsdPhysics.CollisionGroup)
    ]
    filtered_pairs = [
        {
            "path": str(prim.GetPath()),
            "targets": _relationship_targets(prim, "physics:filteredPairs"),
        }
        for prim in prims
        if _relationship_targets(prim, "physics:filteredPairs")
    ]
    physics_materials = [
        _physics_material_inventory(prim)
        for prim in prims
        if prim.HasAPI(UsdPhysics.MaterialAPI)
    ]
    physx_attribute_owners = [
        {"path": str(prim.GetPath()), "attributes": _physx_attributes(prim)}
        for prim in prims
        if _physx_attributes(prim)
    ]
    source_bytes = asset_path.read_bytes()
    asset_identifier = _portable_asset_identifier(asset_path)
    document = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_asset": {
            "path": asset_identifier,
            "size_bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "runtime": _runtime_inventory(),
        "stage": {
            "root_layer_identifier": asset_identifier,
            "default_prim": (
                str(stage.GetDefaultPrim().GetPath())
                if stage.GetDefaultPrim().IsValid()
                else None
            ),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "kilograms_per_unit": float(UsdPhysics.GetStageKilogramsPerUnit(stage)),
            "time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        },
        "summary": {
            "prim_count": len(prim_inventory),
            "articulation_root_count": len(articulation_roots),
            "rigid_body_count": len(rigid_bodies),
            "joint_count": len(joints),
            "collider_count": len(colliders),
            "collision_group_count": len(collision_groups),
            "filtered_pair_owner_count": len(filtered_pairs),
            "physics_material_count": len(physics_materials),
            "physx_attribute_owner_count": len(physx_attribute_owners),
        },
        "prims": prim_inventory,
        "articulation_roots": articulation_roots,
        "rigid_bodies": rigid_bodies,
        "joints": joints,
        "colliders": colliders,
        "collision_groups": collision_groups,
        "filtered_pairs": filtered_pairs,
        "physics_materials": physics_materials,
        "physx_attributes": physx_attribute_owners,
    }
    validate_inventory(document)
    return document


def main() -> None:
    asset_path = ARGS.usd.expanduser().resolve()
    output_path = ARGS.output.expanduser().resolve()
    if not asset_path.is_file():
        raise FileNotFoundError(f"WheelLoader USD not found: {asset_path}")
    stage = Usd.Stage.Open(str(asset_path), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"failed to open USD: {asset_path}")
    inventory = build_inventory(stage, asset_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PHASE_A_USD_INVENTORY_OK",
                "output": str(output_path),
                "summary": inventory["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


try:
    main()
except BaseException:
    traceback.print_exc()
    try:
        simulation_app.close()
    except SystemExit:
        pass
    raise
else:
    simulation_app.close()
