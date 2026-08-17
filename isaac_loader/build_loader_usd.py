"""Build a simulation-ready articulated wheel loader USD.

Run with Isaac Sim's Python, for example:
    <ISAAC_SIM>/python.bat build_loader_usd.py --output wheel_loader.usd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None


COLORS = {
    "yellow": Gf.Vec3f(0.95, 0.55, 0.04),
    "dark": Gf.Vec3f(0.08, 0.09, 0.10),
    "glass": Gf.Vec3f(0.10, 0.25, 0.32),
    "steel": Gf.Vec3f(0.28, 0.30, 0.32),
}


def _xform(stage: Usd.Stage, path: str, position) -> UsdGeom.Xform:
    item = UsdGeom.Xform.Define(stage, path)
    item.AddTranslateOp().Set(Gf.Vec3d(*position))
    return item


def _rigid(prim: Usd.Prim, mass: float) -> None:
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)


def _box(
    stage: Usd.Stage,
    path: str,
    size,
    offset=(0, 0, 0),
    color="yellow",
    collision=True,
) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*offset))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([COLORS[color]])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _wheel(
    stage: Usd.Stage,
    path: str,
    position,
    radius=0.78,
    width=0.48,
) -> UsdGeom.Xform:
    link = _xform(stage, path, position)
    _rigid(link.GetPrim(), 240.0)
    wheel = UsdGeom.Cylinder.Define(stage, f"{path}/wheel")
    wheel.CreateAxisAttr("Y")
    wheel.CreateRadiusAttr(radius)
    wheel.CreateHeightAttr(width)
    wheel.CreateDisplayColorAttr([COLORS["dark"]])
    UsdPhysics.CollisionAPI.Apply(wheel.GetPrim())
    return link


def _revolute_joint(
    stage: Usd.Stage,
    path: str,
    body0: str,
    body1: str,
    local0,
    local1,
    axis: str,
    limits,
    stiffness: float,
    damping: float,
    max_force: float,
) -> UsdPhysics.RevoluteJoint:
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*local1))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(float(limits[0]))
    joint.CreateUpperLimitAttr(float(limits[1]))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(max_force)
    drive.CreateTargetPositionAttr(0.0)
    return joint


def _bucket_mesh(stage: Usd.Stage, path: str) -> UsdGeom.Mesh:
    """Open-top 3.2 m wide general-purpose bucket, local X points forward."""
    # Rear lower/upper, cutting lower/upper corners, duplicated left/right.
    profile = [
        (-1.10, 0.00),
        (-1.05, 1.25),
        (0.95, 0.90),
        (1.30, 0.00),
    ]
    points = []
    for y in (-1.60, 1.60):
        points.extend((x, y, z) for x, z in profile)
    # Side plates plus floor, rear and upper lip. Front/top remain open.
    faces = [
        (0, 1, 2, 3),
        (7, 6, 5, 4),
        (0, 4, 5, 1),
        (0, 3, 7, 4),
        (1, 5, 6, 2),
        (3, 2, 6, 7),
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr(
        [index for face in faces for index in face]
    )
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorAttr([COLORS["yellow"]])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateApproximationAttr("convexHull")
    return mesh


def build(output: Path) -> None:
    stage = Usd.Stage.CreateNew(str(output.resolve()))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    stage.SetTimeCodesPerSecond(60)

    loader = UsdGeom.Xform.Define(stage, "/WheelLoader")
    loader.GetPrim().SetMetadata("kind", "assembly")

    rear = _xform(stage, "/WheelLoader/rear_chassis", (-1.0, 0, 1.55))
    _rigid(rear.GetPrim(), 5200.0)
    UsdPhysics.ArticulationRootAPI.Apply(rear.GetPrim())
    if PhysxSchema is not None:
        PhysxSchema.PhysxArticulationAPI.Apply(
            rear.GetPrim()
        ).CreateEnabledSelfCollisionsAttr(False)
    _box(stage, f"{rear.GetPath()}/body", (2.9, 2.35, 0.75))
    _box(
        stage, f"{rear.GetPath()}/cab",
        (1.35, 1.75, 1.75), (-0.45, 0, 1.15), "glass"
    )

    front = _xform(stage, "/WheelLoader/front_chassis", (1.35, 0, 1.45))
    _rigid(front.GetPrim(), 3100.0)
    _box(stage, f"{front.GetPath()}/body", (2.25, 2.20, 0.68))

    _revolute_joint(
        stage, "/WheelLoader/joints/articulation_joint",
        str(rear.GetPath()), str(front.GetPath()),
        (1.17, 0, -0.10), (-1.18, 0, 0),
        "Z", (-38, 38), 4.0e5, 8.0e4, 2.0e6,
    )

    wheel_specs = [
        ("rear_left", (-1.65, 1.38, 0.78), rear, (-0.65, 1.38, -0.77)),
        ("rear_right", (-1.65, -1.38, 0.78), rear, (-0.65, -1.38, -0.77)),
        ("front_left", (1.65, 1.38, 0.78), front, (0.30, 1.38, -0.67)),
        ("front_right", (1.65, -1.38, 0.78), front, (0.30, -1.38, -0.67)),
    ]
    for name, position, parent, parent_anchor in wheel_specs:
        wheel = _wheel(stage, f"/WheelLoader/{name}_wheel", position)
        _revolute_joint(
            stage, f"/WheelLoader/joints/{name}_wheel_joint",
            str(parent.GetPath()), str(wheel.GetPath()),
            parent_anchor, (0, 0, 0), "Y", (-1.0e6, 1.0e6),
            0.0, 8.0e3, 2.5e4,
        )

    arm = _xform(stage, "/WheelLoader/lift_arm", (2.65, 0, 2.10))
    _rigid(arm.GetPrim(), 850.0)
    _box(stage, f"{arm.GetPath()}/left", (2.35, 0.20, 0.28), (0, 0.78, 0))
    _box(stage, f"{arm.GetPath()}/right", (2.35, 0.20, 0.28), (0, -0.78, 0))
    _box(stage, f"{arm.GetPath()}/cross", (0.25, 1.75, 0.30), (0.85, 0, 0))
    _revolute_joint(
        stage, "/WheelLoader/joints/lift_joint",
        str(front.GetPath()), str(arm.GetPath()),
        (0.25, 0, 0.55), (-1.05, 0, -0.10),
        "Y", (-12, 52), 7.0e5, 1.0e5, 3.0e6,
    )

    bucket = _xform(stage, "/WheelLoader/bucket", (4.15, 0, 1.00))
    _rigid(bucket.GetPrim(), 980.0)
    _bucket_mesh(stage, f"{bucket.GetPath()}/mesh")
    _revolute_joint(
        stage, "/WheelLoader/joints/bucket_joint",
        str(arm.GetPath()), str(bucket.GetPath()),
        (1.05, 0, -0.10), (-0.45, 0, 0.30),
        "Y", (-45, 65), 6.0e5, 8.0e4, 2.0e6,
    )

    # Contact parameters for tyres and bucket are intentionally conservative.
    if PhysxSchema is not None:
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                physx_collision.CreateContactOffsetAttr(0.03)
                physx_collision.CreateRestOffsetAttr(0.0)

    stage.SetDefaultPrim(loader.GetPrim())
    stage.GetRootLayer().Save()
    print(f"created: {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("wheel_loader.usd"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.output)


if __name__ == "__main__":
    main()
