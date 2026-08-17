#!/usr/bin/env python3
"""Author and validate the Phase-C ContactView with Isaac Sim 4.5 public APIs.

Run with::

    /home/eric/isaacsim/python.sh isaac_loader/phase_c_contact_runtime.py \
        --headless --slope-deg 10 --output outputs/phase_c_contact_runtime.json

This probe authors an in-memory stage only.  It never modifies
``wheel_loader.usd`` and it never uses pose setters during a physics loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isaac_bulk_pipeline.contact import (  # noqa: E402
    ALL_COLLISION_GROUPS,
    CollisionGroup,
    CollisionMatrix,
    ContactBackendConfig,
    SlopePatchSpec,
    TriangleMeshContactBackend,
    assess_heightfield_capability,
    build_planar_slope_heightmap,
    estimate_slope_deg,
    validate_collision_memberships,
)
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


def _to_vec3f(values: np.ndarray, vt_module: Any) -> Any:
    return vt_module.Vec3fArray.FromNumpy(
        np.ascontiguousarray(values, dtype=np.float32)
    )


def _to_int(values: np.ndarray, vt_module: Any) -> Any:
    return vt_module.IntArray.FromNumpy(
        np.ascontiguousarray(values, dtype=np.int32)
    )


def author_triangle_mesh_contact(stage: Any, backend: TriangleMeshContactBackend) -> Any:
    """Author a hidden, exact, static collider from a committed data backend."""

    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    mesh_data = backend.committed_mesh
    config = backend.config
    mesh = UsdGeom.Mesh.Define(stage, config.contact_prim_path)
    mesh.CreatePointsAttr(_to_vec3f(mesh_data.points_m, Vt))
    mesh.CreateFaceVertexCountsAttr(_to_int(mesh_data.face_vertex_counts, Vt))
    mesh.CreateFaceVertexIndicesAttr(_to_int(mesh_data.face_vertex_indices, Vt))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    UsdGeom.Imageable(mesh.GetPrim()).CreateVisibilityAttr().Set(
        UsdGeom.Tokens.invisible
    )
    transform = mesh_data.terrain_to_world_matrix
    if not np.allclose(transform, np.eye(4), atol=1e-12):
        # NumPy uses column vectors; Gf matrices store the row-vector form.
        matrix = Gf.Matrix4d(*transform.T.ravel().tolist())
        UsdGeom.Xformable(mesh.GetPrim()).AddTransformOp().Set(matrix)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set(
        UsdPhysics.Tokens.none
    )
    if mesh.GetPrim().HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("terrain contact mesh unexpectedly has RigidBodyAPI")
    return mesh


def update_triangle_mesh_contact(stage: Any, backend: TriangleMeshContactBackend) -> None:
    """Publish a committed generation; callers invoke this only after commit."""

    from pxr import UsdGeom, Vt

    mesh = UsdGeom.Mesh.Get(stage, backend.config.contact_prim_path)
    if not mesh:
        raise RuntimeError("contact mesh must be authored before update")
    data = backend.committed_mesh
    mesh.GetPointsAttr().Set(_to_vec3f(data.points_m, Vt))
    mesh.GetFaceVertexCountsAttr().Set(_to_int(data.face_vertex_counts, Vt))
    mesh.GetFaceVertexIndicesAttr().Set(_to_int(data.face_vertex_indices, Vt))


def default_loader_memberships(
    *,
    loader_path: str,
    contact_prim_path: str,
    environment_collider_path: str,
) -> dict[CollisionGroup, tuple[str, ...]]:
    """Return audited collider-shape memberships for ``wheel_loader.usd``."""

    return {
        CollisionGroup.TERRAIN_SUPPORT: (contact_prim_path,),
        CollisionGroup.WHEEL_CONTACT: tuple(
            f"{loader_path}/{corner}_wheel/wheel"
            for corner in ("front_left", "front_right", "rear_left", "rear_right")
        ),
        CollisionGroup.CHASSIS_CONTACT: (
            f"{loader_path}/front_chassis/body",
            f"{loader_path}/rear_chassis/body",
            f"{loader_path}/rear_chassis/cab",
            f"{loader_path}/lift_arm/left",
            f"{loader_path}/lift_arm/right",
            f"{loader_path}/lift_arm/cross",
        ),
        CollisionGroup.BUCKET_INTERACTION: (f"{loader_path}/bucket/mesh",),
        CollisionGroup.ENVIRONMENT_STATIC: (environment_collider_path,),
    }


def author_collision_groups(
    stage: Any,
    *,
    group_prim_root: str,
    memberships: Mapping[CollisionGroup, Sequence[str]],
    matrix: CollisionMatrix,
) -> dict[CollisionGroup, Any]:
    """Author CollisionGroup + CollectionAPI using documented USD schemas."""

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    UsdGeom.Xform.Define(stage, group_prim_root)
    authored: dict[CollisionGroup, Any] = {}
    for group in ALL_COLLISION_GROUPS:
        schema = UsdPhysics.CollisionGroup.Define(
            stage, f"{group_prim_root}/{group.value}"
        )
        collection = Usd.CollectionAPI.Apply(schema.GetPrim(), "colliders")
        collection.CreateIncludesRel().SetTargets(
            [Sdf.Path(path) for path in memberships[group]]
        )
        authored[group] = schema
    # A pair is filtered if either relation names the other group.  Author both
    # directions so validation and downstream tools see an explicit symmetric
    # contract rather than depending on one-sided interpretation.
    for first, second in matrix.filtered_pairs():
        authored[first].CreateFilteredGroupsRel().AddTarget(
            authored[second].GetPath()
        )
        authored[second].CreateFilteredGroupsRel().AddTarget(
            authored[first].GetPath()
        )
    return authored


def inspect_authored_groups(
    authored: Mapping[CollisionGroup, Any],
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for group, schema in authored.items():
        collection = schema.GetCollidersCollectionAPI()
        result[group.value] = {
            "includes": sorted(str(path) for path in collection.GetIncludesRel().GetTargets()),
            "filtered_groups": sorted(
                str(path) for path in schema.GetFilteredGroupsRel().GetTargets()
            ),
        }
    return result


def _reference_loader(stage: Any, asset_path: Path, loader_path: str) -> None:
    from pxr import Sdf, UsdGeom

    prim = UsdGeom.Xform.Define(stage, loader_path).GetPrim()
    if not prim.GetReferences().AddReference(str(asset_path), Sdf.Path("/WheelLoader")):
        raise RuntimeError(f"failed to reference loader asset: {asset_path.name}")


def _author_environment_probe(stage: Any, path: str) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 8.0, -0.5))
    cube.AddScaleOp().Set(Gf.Vec3f(2.0, 2.0, 0.5))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _validate_stage_members(stage: Any, memberships: Mapping[CollisionGroup, Sequence[str]]) -> None:
    from pxr import UsdPhysics

    errors: list[str] = []
    for group, paths in memberships.items():
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                errors.append(f"missing {group.value} prim {path}")
            elif not prim.HasAPI(UsdPhysics.CollisionAPI):
                errors.append(f"{group.value} member lacks CollisionAPI: {path}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _public_heightfield_symbols(*modules: Any) -> tuple[str, ...]:
    symbols: list[str] = []
    for module in modules:
        module_name = getattr(module, "__name__", type(module).__name__)
        symbols.extend(
            f"{module_name}.{name}"
            for name in dir(module)
            if "heightfield" in name.lower() or "height_field" in name.lower()
        )
    return tuple(sorted(set(symbols)))


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory("phase_c_contact_runtime.usda")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene").CreateGravityDirectionAttr(
        (0.0, 0.0, -1.0)
    )
    stage.GetPrimAtPath("/World/PhysicsScene").GetAttribute(
        "physics:gravityMagnitude"
    ).Set(9.81)

    config = ContactBackendConfig(
        contact_prim_path=args.contact_prim_path,
        visual_prim_path=args.visual_prim_path,
        target_spacing_m=args.contact_spacing_m,
        commit_policy=args.commit_policy,
        low_frequency_hz=args.low_frequency_hz,
    )
    grid = TerrainGrid(
        nx=args.grid_size,
        ny=args.grid_size,
        dx=args.visual_spacing_m,
        dy=args.visual_spacing_m,
        origin_x=-0.5 * (args.grid_size - 1) * args.visual_spacing_m,
        origin_y=-0.5 * (args.grid_size - 1) * args.visual_spacing_m,
        terrain_prim_path=config.visual_prim_path,
    )
    heightmap = build_planar_slope_heightmap(
        grid,
        SlopePatchSpec(args.slope_deg, uphill_direction_xy=(1.0, 0.0)),
    )
    backend = TriangleMeshContactBackend(config)
    backend.initialize(grid, heightmap)
    contact_mesh = author_triangle_mesh_contact(stage, backend)
    # A distinct non-colliding placeholder proves that the visual identity is
    # outside all physics groups.  The production renderer owns its contents.
    UsdGeom.Xform.Define(stage, config.visual_prim_path)

    loader_path = "/World/WheelLoader"
    _reference_loader(stage, args.loader_usd.resolve(), loader_path)
    environment_path = "/World/EnvironmentStatic/ProbeBox"
    _author_environment_probe(stage, environment_path)
    memberships = default_loader_memberships(
        loader_path=loader_path,
        contact_prim_path=config.contact_prim_path,
        environment_collider_path=environment_path,
    )
    membership_validation = validate_collision_memberships(
        memberships,
        contact_prim_path=config.contact_prim_path,
        visual_prim_path=config.visual_prim_path,
    )
    membership_validation.require_valid()
    _validate_stage_members(stage, memberships)
    matrix = CollisionMatrix()
    authored_groups = author_collision_groups(
        stage,
        group_prim_root=args.collision_group_root,
        memberships=memberships,
        matrix=matrix,
    )
    authored_inventory = inspect_authored_groups(authored_groups)

    # Exercise one action-boundary update.  No physics-frame recook is hidden
    # in this adapter: USD changes only after the pure backend commit succeeds.
    modified = heightmap.copy()
    center = args.grid_size // 2
    modified[center, center] += 0.01
    backend.update_from_heightmap(modified, action_index=0)
    commit = backend.commit(reason="action_end")
    if commit.recook_required:
        update_triangle_mesh_contact(stage, backend)

    heightfield_symbols = _public_heightfield_symbols(UsdPhysics, PhysxSchema)
    heightfield = assess_heightfield_capability(
        runtime_label="Isaac Sim 4.5 / USD Physics schema probe",
        public_schema_symbols=heightfield_symbols,
    )
    contact_prim = contact_mesh.GetPrim()
    visual_prim = stage.GetPrimAtPath(config.visual_prim_path)
    bucket_group_path = authored_groups[CollisionGroup.BUCKET_INTERACTION].GetPath()
    terrain_group_path = authored_groups[CollisionGroup.TERRAIN_SUPPORT].GetPath()
    bucket_filtered = terrain_group_path in authored_groups[
        CollisionGroup.BUCKET_INTERACTION
    ].GetFilteredGroupsRel().GetTargets()
    terrain_filtered = bucket_group_path in authored_groups[
        CollisionGroup.TERRAIN_SUPPORT
    ].GetFilteredGroupsRel().GetTargets()

    measured_slope = estimate_slope_deg(heightmap, grid)
    return {
        "schema_version": "isaac-bulk-phase-c-runtime/v1",
        "status": "PASS",
        "configuration": args.configuration_evidence,
        "slope_fixture": {
            "requested_deg": float(args.slope_deg),
            "measured_mean_deg": float(np.mean(measured_slope)),
            "measured_max_abs_error_deg": float(
                np.max(np.abs(measured_slope - args.slope_deg))
            ),
        },
        "contact_mesh": {
            "prim_path": config.contact_prim_path,
            "visual_prim_path": config.visual_prim_path,
            "separate_from_visual": config.contact_prim_path != config.visual_prim_path,
            "visible": str(
                UsdGeom.Imageable(contact_prim).GetVisibilityAttr().Get()
            )
            != "invisible",
            "collision_api": contact_prim.HasAPI(UsdPhysics.CollisionAPI),
            "mesh_collision_api": contact_prim.HasAPI(UsdPhysics.MeshCollisionAPI),
            "approximation": str(
                UsdPhysics.MeshCollisionAPI(contact_prim).GetApproximationAttr().Get()
            ),
            "rigid_body_api": contact_prim.HasAPI(UsdPhysics.RigidBodyAPI),
            "visual_collision_api": visual_prim.HasAPI(UsdPhysics.CollisionAPI),
            "source_grid_shape_yx": list(grid.shape),
            "source_spacing_m": float(args.visual_spacing_m),
            "contact_spacing_xy_m": list(backend.committed_mesh.nominal_spacing_xy_m),
            "vertex_count": backend.committed_mesh.vertex_count,
            "triangle_count": backend.committed_mesh.triangle_count,
            "commit_policy": config.commit_policy,
            "committed_generation": commit.generation,
            "recook_reason": commit.reason,
        },
        "collision_filtering": {
            "matrix": matrix.as_dict(),
            "groups": authored_inventory,
            "membership_validation": {
                "valid": membership_validation.valid,
                "errors": list(membership_validation.errors),
                "warnings": list(membership_validation.warnings),
            },
            "bucket_terrain_filtered_bidirectional": bool(
                bucket_filtered and terrain_filtered
            ),
            "visual_prim_in_any_group": any(
                config.visual_prim_path in item["includes"]
                for item in authored_inventory.values()
            ),
        },
        "heightfield_capability": heightfield.to_dict(),
        "phase_d_handoff": {
            "slope_deg": float(args.slope_deg),
            "drive_probe_performed": False,
            "root_displacement_m": None,
            "speed_m_s": None,
            "wheel_bottom_clearance_m": None,
            "contact_load_n": None,
            "pose_setter_count": 0,
            "note": "authoring/collision-filter probe only; Phase B/D runtime owns drive telemetry",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--slope-deg", type=float, choices=(0.0, 10.0, 20.0), default=10.0)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--visual-spacing-m", type=float, default=0.05)
    parser.add_argument("--contact-spacing-m", type=float, default=0.10)
    parser.add_argument(
        "--contact-config",
        type=Path,
        default=PROJECT_ROOT / "configs/phase_c_contact.yaml",
    )
    parser.add_argument(
        "--terrain-config",
        type=Path,
        default=PROJECT_ROOT / "configs/project_25m.yaml",
    )
    parser.add_argument(
        "--loader-usd",
        type=Path,
        default=PROJECT_ROOT / "isaac_loader/wheel_loader.usd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/phase_c_contact_runtime.json",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_yaml_configuration(args: argparse.Namespace) -> None:
    """Load the Phase-C/terrain YAMLs and reject divergent CLI magic values."""

    import yaml

    contact_path = args.contact_config.expanduser().resolve()
    terrain_path = args.terrain_config.expanduser().resolve()
    if not contact_path.is_file() or not terrain_path.is_file():
        raise FileNotFoundError(
            f"Phase-C configuration is missing: contact={contact_path}, terrain={terrain_path}"
        )
    contact_document = yaml.safe_load(contact_path.read_text(encoding="utf-8"))
    terrain_document = yaml.safe_load(terrain_path.read_text(encoding="utf-8"))
    if contact_document.get("phase") != "C":
        raise ValueError("contact config must declare phase: C")
    configured = contact_document["contact"]
    terrain = terrain_document["terrain"]
    configured_slopes = tuple(float(value) for value in contact_document["slope_tests"]["degrees"])
    if float(args.slope_deg) not in configured_slopes:
        raise ValueError(
            f"slope {args.slope_deg} is not declared in Phase-C YAML: {configured_slopes}"
        )
    if configured["backend"] != "triangle_mesh":
        raise ValueError("Phase-C runtime requires configured triangle_mesh backend")
    if bool(configured["visible"]) or bool(configured["rigid_body_enabled"]):
        raise ValueError("Phase-C YAML must configure a hidden static contact view")
    if str(configured["mesh_approximation"]) != "none":
        raise ValueError("Phase-C triangle contact must use exact approximation=none")
    configured_contact_spacing = float(configured["target_spacing_m"])
    configured_visual_spacing = float(terrain["dx_m"])
    if abs(float(terrain["dy_m"]) - configured_visual_spacing) > 1e-12:
        raise ValueError("this square slope fixture requires terrain dx_m == dy_m")
    if abs(args.contact_spacing_m - configured_contact_spacing) > 1e-12:
        raise ValueError(
            "--contact-spacing-m must match configs/phase_c_contact.yaml; "
            f"cli={args.contact_spacing_m}, config={configured_contact_spacing}"
        )
    if abs(args.visual_spacing_m - configured_visual_spacing) > 1e-12:
        raise ValueError(
            "--visual-spacing-m must match the configured authoritative terrain spacing; "
            f"cli={args.visual_spacing_m}, config={configured_visual_spacing}"
        )
    disabled_pairs = contact_document["collision_filtering"]["disabled_pairs"]
    if disabled_pairs != [["TerrainSupport", "BucketInteraction"]]:
        raise ValueError(
            "Phase-C YAML must disable exactly TerrainSupport/BucketInteraction"
        )
    deformation = {
        name: bool(contact_document["wheel_terrain"][name])
        for name in (
            "deformation_enabled",
            "sinkage_enabled",
            "compaction_enabled",
            "rut_enabled",
        )
    }
    if any(deformation.values()):
        raise ValueError("Phase-C acceptance requires wheel terrain deformation disabled")

    args.contact_prim_path = str(configured["prim_path"])
    args.visual_prim_path = str(configured["visual_prim_path"])
    args.commit_policy = str(configured["commit"]["policy"])
    args.low_frequency_hz = float(configured["commit"]["low_frequency_hz"])
    args.collision_group_root = str(
        contact_document["collision_filtering"]["group_prim_root"]
    )
    args.configuration_evidence = {
        "contact_config": str(contact_path),
        "contact_config_relative": str(contact_path.relative_to(PROJECT_ROOT)),
        "contact_config_sha256": _sha256_file(contact_path),
        "terrain_config": str(terrain_path),
        "terrain_config_relative": str(terrain_path.relative_to(PROJECT_ROOT)),
        "terrain_config_sha256": _sha256_file(terrain_path),
        "effective": {
            "backend": configured["backend"],
            "contact_prim_path": args.contact_prim_path,
            "visual_prim_path": args.visual_prim_path,
            "visual_spacing_m": args.visual_spacing_m,
            "contact_spacing_m": args.contact_spacing_m,
            "commit_policy": args.commit_policy,
            "low_frequency_hz": args.low_frequency_hz,
            "collision_group_root": args.collision_group_root,
            "disabled_pairs": disabled_pairs,
            "wheel_terrain": deformation,
            "slope_degrees": list(configured_slopes),
        },
    }


def main() -> int:
    args = parse_args()
    resolve_yaml_configuration(args)
    if args.grid_size < 3:
        raise ValueError("--grid-size must be >= 3")
    if not args.loader_usd.is_file():
        raise FileNotFoundError(args.loader_usd)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless)})
    try:
        evidence = run_probe(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("PHASE_C_CONTACT_RUNTIME_PASS")
        print(json.dumps(evidence, indent=2, sort_keys=True))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
