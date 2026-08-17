#!/usr/bin/env python3
"""Audit the delivered 390F USD and emit its soil-interaction descriptor.

Run this with Isaac Sim's Python interpreter because it needs ``pxr``.  The
resulting JSON is pure data and is loaded by the normal Python runtime without
USD or Isaac dependencies.  Semantic vertex indices are deliberate CAD
annotations; no convex collision approximation is used as bucket interior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_bulk_pipeline.tools import (  # noqa: E402
    BucketGeometryDescriptor,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolDescriptorLoader,
)


BUCKET_PRIM = (
    "/World/_90F_LME_ISAAC_DETAIL_DELIVERY/"
    "tn__390F_LME_ISAAC_DETAIL_DELIVERY_/tn__aaa0ab9/"
    "tn__390F_DETAIL_DELIVERY_Bucket1_zf0"
)
MESH_PRIM = f"{BUCKET_PRIM}/Mesh"
JOINT_PRIM = (
    "/World/_90F_LME_ISAAC_DETAIL_DELIVERY/"
    "tn__390F_LME_ISAAC_DETAIL_DELIVERY_/tn__aaa0ab9/Joints/bucket_joint"
)
EXPECTED_ASSET_SHA256 = (
    "db2bb54fd81b43c9ce57f8ab9ad7d7b07dbe81b6dbf4dbd9dbb321eabe747b7c"
)

# Audited against SHA-256 db2bb54f... The pairs define virtual centre markers
# where the production CAD has a central tooth gap.
SEMANTIC_VERTICES = {
    "CuttingEdgeLeft": [24],
    "CuttingEdgeCenter": [12, 13],
    "CuttingEdgeRight": [2],
    "TopEdgeLeft": [392],
    "TopEdgeRight": [390],
    "RearBottomLeft": [435],
    "RearBottomRight": [434],
}

# Ordered concave interior side profile: rear-bottom -> curved shell -> top,
# with the cutting-edge centre appended by the extractor to close the profile.
INTERIOR_PROFILE_VERTICES = [
    434, 430, 427, 422, 419, 421, 416, 414, 407,
    410, 412, 405, 401, 400, 396, 393, 390,
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _column_matrix(cache: object, prim: object) -> np.ndarray:
    return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64).T


def _points(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((values, np.ones(len(values))))
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :3] / transformed[:, 3, None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        type=Path,
        default=Path("/home/eric/桌面/bulldozer_sim/bulldozer_main.usd"),
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        default=ROOT / "configs/excavator_390f_bucket_descriptor.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "outputs/real_390f_bucket_audit.json",
    )
    args = parser.parse_args()

    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError("run with Isaac Sim python.sh (pxr is required)") from exc

    asset = args.asset.expanduser().resolve()
    asset_hash = _sha256(asset)
    if asset_hash != EXPECTED_ASSET_SHA256:
        raise RuntimeError(
            "the 390F USD hash changed; semantic vertex indices must be re-audited "
            f"before extraction (expected={EXPECTED_ASSET_SHA256}, actual={asset_hash})"
        )
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {asset}")
    bucket = stage.GetPrimAtPath(BUCKET_PRIM)
    mesh_prim = stage.GetPrimAtPath(MESH_PRIM)
    joint_prim = stage.GetPrimAtPath(JOINT_PRIM)
    if not all(prim and prim.IsValid() for prim in (bucket, mesh_prim, joint_prim)):
        raise RuntimeError("audited bucket/mesh/joint prim path is missing")

    mesh = UsdGeom.Mesh(mesh_prim)
    local_points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    face_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    cache = UsdGeom.XformCache()
    world_from_mesh = _column_matrix(cache, mesh_prim)
    world_from_bucket = _column_matrix(cache, bucket)
    world_points = _points(world_from_mesh, local_points)

    def marker(name: str) -> np.ndarray:
        return np.mean(world_points[SEMANTIC_VERTICES[name]], axis=0)

    cutting_left = marker("CuttingEdgeLeft")
    cutting_center = marker("CuttingEdgeCenter")
    cutting_right = marker("CuttingEdgeRight")
    top_left = marker("TopEdgeLeft")
    top_right = marker("TopEdgeRight")
    rear_left = marker("RearBottomLeft")
    rear_right = marker("RearBottomRight")

    tool_x = cutting_right - cutting_left
    tool_x /= np.linalg.norm(tool_x)
    tool_y = cutting_center - 0.5 * (rear_left + rear_right)
    tool_y -= np.dot(tool_y, tool_x) * tool_x
    tool_y /= np.linalg.norm(tool_y)
    tool_z = np.cross(tool_x, tool_y)
    tool_z /= np.linalg.norm(tool_z)
    if np.dot(0.5 * (top_left + top_right) - cutting_center, tool_z) < 0.0:
        raise RuntimeError("semantic left/right ordering produces downward Tool +Z")
    world_from_tool = np.eye(4)
    world_from_tool[:3, :3] = np.column_stack((tool_x, tool_y, tool_z))
    world_from_tool[:3, 3] = cutting_center
    tool_from_world = np.linalg.inv(world_from_tool)

    def tool_points(values: np.ndarray) -> np.ndarray:
        return _points(tool_from_world, values)

    cutting = tool_points(np.stack((cutting_left, cutting_center, cutting_right)))
    top = tool_points(np.stack((top_left, top_right)))
    rear = tool_points(np.stack((rear_left, rear_right)))
    profile_world = world_points[INTERIOR_PROFILE_VERTICES]
    profile = tool_points(profile_world)
    profile[:, 0] = 0.0
    profile = np.vstack((profile, np.zeros((1, 3))))
    rear_center = np.mean(rear, axis=0)
    rear_center[0] = 0.0
    bottom = np.stack((rear_center, np.zeros(3)))
    interior_width = float(np.linalg.norm(top_right - top_left))
    geometry = BucketGeometryDescriptor.from_extruded_profile(
        cutting_edge_local=cutting,
        bottom_profile_local=bottom,
        interior_profile_local=profile,
        top_edge_local=top,
        rated_capacity_m3=None,
        interior_width_m=interior_width,
        geometry_source=GeometrySource.USD_MESH_MARKERS,
        geometry_quality=GeometryQuality.REDUCED_ORDER,
        metadata={
            "asset_sha256": asset_hash,
            "mesh_prim": MESH_PRIM,
            "semantic_vertex_indices": SEMANTIC_VERTICES,
            "interior_profile_vertex_indices": INTERIOR_PROFILE_VERTICES,
            "geometry_inference": "CAD marker constrained concave extruded profile",
            "convex_hull_used_as_soil_geometry": False,
            "capacity_classification": "LITERATURE_INDEPENDENT_CAD_REDUCED_ORDER",
        },
    )
    link_from_tool = np.linalg.inv(world_from_bucket) @ world_from_tool
    descriptor = ToolDescriptor(
        tool_type="tracked_excavator_bucket_390f",
        tool_frame_prim=f"{BUCKET_PRIM}/SoilInteraction/ToolOrigin",
        cutting_edge_local=cutting,
        bottom_profile_local=bottom,
        left_boundary_local=geometry.left_side_wall_local,
        right_boundary_local=geometry.right_side_wall_local,
        interior_profile_local=profile,
        nominal_width_m=geometry.cutting_edge_length_m,
        nominal_capacity_m3=None,
        proxy_level="L1",
        actual_proxy_type="ExtrudedProfileBucket_L1",
        tool_to_link_matrix=link_from_tool,
        metadata={
            "vehicle_architecture": "TRACKED_HYDRAULIC_EXCAVATOR",
            "descriptor_source": "REAL_USD_MESH_PLUS_SEMANTIC_VERTEX_MARKERS",
            "asset_path_at_extraction": str(asset),
            "asset_sha256": asset_hash,
            "bucket_terrain_physx_collision_policy": (
                "DISABLE_BUCKET_VS_RIGID_BULK_TERRAIN_WHEN_CUSTOM_SOIL_FORCE_ACTIVE"
            ),
        },
        bucket_geometry=geometry,
    )
    ToolDescriptorLoader.save(descriptor, args.descriptor)

    joint = UsdPhysics.RevoluteJoint(joint_prim)
    local_bucket_transform = np.asarray(
        UsdGeom.Xformable(bucket).GetLocalTransformation(), dtype=np.float64
    ).T
    bucket_mass_kg = UsdPhysics.MassAPI(bucket).GetMassAttr().Get()
    audit = {
        "schema_version": "real-390f-bucket-audit-v1",
        "asset": str(asset),
        "asset_sha256": asset_hash,
        "stage_meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "stage_up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "bucket_rigid_body_prim": BUCKET_PRIM,
        "visual_mesh_prim": MESH_PRIM,
        "mesh_count_below_bucket": sum(
            1 for prim in Usd.PrimRange(bucket) if prim.IsA(UsdGeom.Mesh)
        ),
        "vertex_count": int(len(local_points)),
        "triangle_count": int(np.sum(face_counts - 2)),
        "bucket_local_transform_column_matrix": local_bucket_transform.tolist(),
        "bucket_local_to_world_column_matrix": world_from_bucket.tolist(),
        "bucket_mass_kg": None if bucket_mass_kg is None else float(bucket_mass_kg),
        "bucket_joint_prim": JOINT_PRIM,
        "joint_parent_body": [str(path) for path in joint.GetBody0Rel().GetTargets()],
        "joint_child_body": [str(path) for path in joint.GetBody1Rel().GetTargets()],
        "joint_axis": str(joint.GetAxisAttr().Get()),
        "joint_lower_limit_deg": float(joint.GetLowerLimitAttr().Get()),
        "joint_upper_limit_deg": float(joint.GetUpperLimitAttr().Get()),
        "collision_approximation": str(mesh_prim.GetAttribute("physics:approximation").Get()),
        "collision_classification": "CURRENT_COLLISION_ONLY_NOT_SOIL_GEOMETRY",
        "visual_mesh": "REAL_USD_MESH",
        "physx_collider": "CONVEX_HULL_CURRENT_RUNTIME_COLLIDER",
        "soil_interaction_geometry": "USD_MESH_MARKERS",
        "convex_hull_used_as_soil_geometry": False,
        "cutting_edge_width_m": geometry.cutting_edge_length_m,
        "interior_width_m": geometry.interior_width_m,
        "mouth_area_m2": geometry.mouth_area_m2,
        "geometric_capacity_m3": geometry.geometric_capacity_m3,
        "descriptor": str(args.descriptor.resolve()),
        "bucket_terrain_physx_collision_policy": (
            "tracks/chassis<->ground PhysX; bucket<->bulk custom earthmoving; "
            "disable overlapping rigid terrain response to prevent double counting"
        ),
        "future_rigid_collision_recommendation": (
            "validated convex decomposition for coarse rigid obstacles; SDF only after "
            "runtime cost/scale validation; neither is soil-interaction geometry"
        ),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
