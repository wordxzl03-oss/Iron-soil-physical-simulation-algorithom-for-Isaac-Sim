"""Bucket-local 3D coarse-granular canonical benchmark for Isaac Sim 4.5.

Purpose
-------
This is an isolated falsification test for the production 2.5D bucket-filling
failure. It intentionally leaves Resting/Mobile V2 untouched. The benchmark
uses the real 390F USD bucket and PhysX solid PBD particles to answer one
question: does restoring XYZ particle motion allow material to climb the bucket
floor and remain in the cavity under the same audited articulation target
sequence?

The particle parameters are engineering test parameters, not calibrated iron-ore
properties. No result from this script may be labelled as site-calibrated soil
physics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml


PARSER = argparse.ArgumentParser()
PARSER.add_argument("--config", default="configs/390f_v2_interactive.yaml")
PARSER.add_argument("--prototype-config", default="configs/bucket_local_3d_physx.yaml")
PARSER.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
PARSER.add_argument(
    "--trajectory-json",
    type=Path,
    default=None,
    help=(
        "Optional exact joint-position trace. JSON must contain records with "
        "t_s and joint_position_rad. If omitted, the benchmark replays the "
        "audited penetrate->coordinated_cut->curl_filling->breakout target sequence."
    ),
)
PARSER.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/390f_v2/bucket_local_3d/canonical_audit.json"),
)
ARGS, _UNKNOWN = PARSER.parse_known_args()

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {"headless": ARGS.headless, "multi_gpu": False, "width": 1280, "height": 720}
)

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _matrix_world_m(prim, UsdGeom) -> np.ndarray:
    return np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=np.float64
    ).T


def _resolve_tool_origin(stage, bucket_link: str):
    direct = stage.GetPrimAtPath(f"{bucket_link}/SoilInteraction/ToolOrigin")
    if direct.IsValid():
        return direct
    bucket_prefix = bucket_link.rstrip("/") + "/"
    for prim in stage.Traverse():
        if (
            prim.GetName() == "ToolOrigin"
            and prim.GetPath().pathString.startswith(bucket_prefix)
        ):
            return prim
    raise RuntimeError(
        f"[BucketLocal3D] cannot resolve SoilInteraction/ToolOrigin under {bucket_link}"
    )


def _physics_scene_prim(stage, UsdPhysics):
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError("[BucketLocal3D] no UsdPhysics.Scene found after World creation")
    return scenes[0]


def _author_particle_system(
    *,
    stage,
    scene_prim,
    points_world_m: np.ndarray,
    prototype_cfg: dict,
    Gf,
    Sdf,
    UsdGeom,
    UsdShade,
    PhysxSchema,
):
    root_path = Sdf.Path("/World/BucketLocal3D")
    system_path = root_path.AppendChild("ParticleSystem")
    points_path = root_path.AppendChild("Particles")
    material_path = root_path.AppendChild("GranularMaterial")

    spacing = float(prototype_cfg["particle_spacing_m"])
    radius = float(prototype_cfg["particle_radius_m"])
    rest_offset = float(prototype_cfg.get("solid_rest_offset_m", radius))
    contact_offset = float(
        prototype_cfg.get("particle_contact_offset_m", max(radius, 0.5 * spacing))
    )
    if not (0.0 < rest_offset <= contact_offset <= spacing):
        raise ValueError(
            "require 0 < solid_rest_offset_m <= particle_contact_offset_m <= particle_spacing_m"
        )

    system = PhysxSchema.PhysxParticleSystem.Define(stage, system_path)
    system.CreateParticleSystemEnabledAttr().Set(True)
    system.CreateRestOffsetAttr().Set(float(rest_offset))
    system.CreateSolidRestOffsetAttr().Set(float(rest_offset))
    system.CreateContactOffsetAttr().Set(float(contact_offset))
    system.CreateParticleContactOffsetAttr().Set(float(contact_offset))
    system.CreateSolverPositionIterationCountAttr().Set(
        int(prototype_cfg.get("solver_position_iterations", 8))
    )
    if hasattr(system, "CreateGlobalSelfCollisionEnabledAttr"):
        system.CreateGlobalSelfCollisionEnabledAttr().Set(True)
    if hasattr(system, "CreateNonParticleCollisionEnabledAttr"):
        system.CreateNonParticleCollisionEnabledAttr().Set(True)
    if hasattr(system, "CreateMaxVelocityAttr"):
        system.CreateMaxVelocityAttr().Set(
            float(prototype_cfg.get("maximum_particle_velocity_m_s", 20.0))
        )
    system.CreateSimulationOwnerRel().SetTargets([scene_prim.GetPath()])

    material = UsdShade.Material.Define(stage, material_path)
    pbd = PhysxSchema.PhysxPBDMaterialAPI.Apply(material.GetPrim())
    pbd.CreateFrictionAttr().Set(float(prototype_cfg["friction"]))
    pbd.CreateParticleFrictionScaleAttr().Set(
        float(prototype_cfg.get("particle_friction_scale", 1.0))
    )
    pbd.CreateDensityAttr().Set(float(prototype_cfg["bulk_density_kg_m3"]))
    pbd.CreateDampingAttr().Set(float(prototype_cfg.get("damping", 0.0)))
    if hasattr(pbd, "CreateAdhesionAttr"):
        pbd.CreateAdhesionAttr().Set(float(prototype_cfg.get("adhesion", 0.0)))
    if hasattr(pbd, "CreateParticleAdhesionScaleAttr"):
        pbd.CreateParticleAdhesionScaleAttr().Set(
            float(prototype_cfg.get("particle_adhesion_scale", 1.0))
        )

    points = UsdGeom.Points.Define(stage, points_path)
    positions = [Gf.Vec3f(*map(float, row)) for row in points_world_m]
    points.CreatePointsAttr().Set(positions)
    points.CreateVelocitiesAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)] * len(positions))
    points.CreateWidthsAttr().Set([2.0 * radius] * len(positions))
    points.CreateDisplayColorAttr().Set([Gf.Vec3f(0.38, 0.24, 0.12)])

    particle_api = PhysxSchema.PhysxParticleAPI.Apply(points.GetPrim())
    particle_api.CreateParticleEnabledAttr().Set(True)
    particle_api.CreateSelfCollisionAttr().Set(True)
    particle_api.CreateParticleGroupAttr().Set(0)
    particle_api.CreateParticleSystemRel().SetTargets([system.GetPath()])

    particle_set_api = PhysxSchema.PhysxParticleSetAPI.Apply(points.GetPrim())
    particle_set_api.CreateFluidAttr().Set(False)

    UsdShade.MaterialBindingAPI.Apply(points.GetPrim()).Bind(
        material, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
    return points, particle_set_api, system


def _particle_positions_world(points, particle_set_api) -> np.ndarray:
    simulation_attr = particle_set_api.GetSimulationPointsAttr()
    values = simulation_attr.Get() if simulation_attr and simulation_attr.IsValid() else None
    if values is None or len(values) == 0:
        values = points.GetPointsAttr().Get()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise RuntimeError(
            f"[BucketLocal3D] unexpected particle position array shape {array.shape}"
        )
    return array


def _load_exact_trace(path: Path, dof_count: int) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) and "records" in data else data
    times = []
    positions = []
    for record in records:
        if "t_s" not in record or "joint_position_rad" not in record:
            continue
        q = np.asarray(record["joint_position_rad"], dtype=np.float64).reshape(-1)
        if len(q) != dof_count:
            continue
        times.append(float(record["t_s"]))
        positions.append(q)
    if len(times) < 2:
        raise ValueError("--trajectory-json must contain at least two compatible records")
    times = np.asarray(times, dtype=np.float64)
    positions = np.vstack(positions)
    order = np.argsort(times)
    times = times[order]
    positions = positions[order]
    times = times - times[0]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("--trajectory-json timestamps must be strictly increasing")
    return times, positions


def _interpolate_trace(times: np.ndarray, positions: np.ndarray, t_s: float) -> np.ndarray:
    if t_s <= times[0]:
        return positions[0].copy()
    if t_s >= times[-1]:
        return positions[-1].copy()
    hi = int(np.searchsorted(times, t_s, side="right"))
    lo = hi - 1
    alpha = (t_s - times[lo]) / (times[hi] - times[lo])
    return (1.0 - alpha) * positions[lo] + alpha * positions[hi]


def _target_sequence(config, prototype_cfg: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    phases_cfg = prototype_cfg["target_sequence"]
    names = [str(item["target"]) for item in phases_cfg]
    durations = [float(item["duration_s"]) for item in phases_cfg]
    if any(value <= 0.0 for value in durations):
        raise ValueError("target_sequence durations must be positive")

    positions = [np.asarray(config.phase_targets_rad[names[0]], dtype=np.float64)]
    times = [0.0]
    current_time = 0.0
    for name, duration in zip(names[1:], durations[1:]):
        current_time += duration
        positions.append(np.asarray(config.phase_targets_rad[name], dtype=np.float64))
        times.append(current_time)
    settle = durations[0]
    times = np.asarray(times, dtype=np.float64) + settle
    times = np.insert(times, 0, 0.0)
    positions = np.vstack([positions[0], *positions])
    return times, positions, names


def main() -> None:
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.types import ArticulationAction

    from isaac_bulk_pipeline.experimental.bucket_local_3d import (
        BucketCavityGeometry,
        ParticleVolumeAccounting,
        inverse_transform_points,
        sample_rectangular_lattice,
        transform_points,
    )
    from isaac_bulk_pipeline.runtime import Interactive390FConfig

    config = Interactive390FConfig.load(ROOT / ARGS.config)
    prototype_cfg = yaml.safe_load((ROOT / ARGS.prototype_config).read_text(encoding="utf-8"))
    descriptor_path = ROOT / "configs" / "excavator_390f_bucket_descriptor.json"
    cavity = BucketCavityGeometry.from_descriptor(descriptor_path)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=config.physics_dt_s,
        rendering_dt=config.physics_dt_s,
    )
    add_reference_to_stage(str(config.vehicle_asset), "/World/Excavator")
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    ground = UsdGeom.Cube.Define(stage, "/World/Local3DSupportGround")
    ground.CreateSizeAttr(2.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.10))
    ground.AddScaleOp().Set(Gf.Vec3f(30.0, 30.0, 0.10))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    if config.mobile_base_enabled:
        anchor_prim = stage.GetPrimAtPath(config.world_anchor_joint)
        if not anchor_prim.IsValid():
            raise RuntimeError(
                f"[BucketLocal3D] missing configured world anchor {config.world_anchor_joint}"
            )
        anchor = UsdPhysics.Joint(anchor_prim)
        anchor.GetJointEnabledAttr().Set(False)
        anchor_prim.SetActive(False)
        stage.RemovePrim(config.world_anchor_joint)

    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    track_min_z = min(
        float(
            bounds.ComputeWorldBound(stage.GetPrimAtPath(path))
            .ComputeAlignedRange()
            .GetMin()[2]
        )
        for path in (config.left_track_body, config.right_track_body)
    )
    if config.auto_align_track_bottom_to_ground:
        lift = float(config.track_ground_clearance_m - track_min_z)
        UsdGeom.Xformable(stage.GetPrimAtPath("/World/Excavator")).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, lift)
        )

    excavator = world.scene.add(
        SingleArticulation(config.articulation_root, name="bucket_local_3d_390f")
    )
    reset_pose = np.asarray(config.phase_targets_rad["penetrate"], dtype=np.float64)
    excavator.set_joints_default_state(positions=reset_pose)
    world.reset()
    controller = excavator.get_articulation_controller()

    tool_prim = _resolve_tool_origin(stage, config.bucket_link)
    tool_world = _matrix_world_m(tool_prim, UsdGeom)

    spacing = float(prototype_cfg["particle_spacing_m"])
    seed_local = sample_rectangular_lattice(
        prototype_cfg["seed_box_tool_local_min_m"],
        prototype_cfg["seed_box_tool_local_max_m"],
        spacing,
    )
    seed_world = transform_points(tool_world, seed_local)
    minimum_allowed_z = float(
        prototype_cfg.get("minimum_seed_world_z_m", prototype_cfg["particle_radius_m"])
    )
    seed_lift_m = max(0.0, minimum_allowed_z - float(np.min(seed_world[:, 2])))
    seed_world[:, 2] += seed_lift_m

    scene_prim = _physics_scene_prim(stage, UsdPhysics)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    if hasattr(physx_scene, "CreateEnableGPUDynamicsAttr"):
        physx_scene.CreateEnableGPUDynamicsAttr().Set(True)

    points, particle_set_api, _system = _author_particle_system(
        stage=stage,
        scene_prim=scene_prim,
        points_world_m=seed_world,
        prototype_cfg=prototype_cfg,
        Gf=Gf,
        Sdf=Sdf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        PhysxSchema=PhysxSchema,
    )

    world.reset()
    controller = excavator.get_articulation_controller()

    if ARGS.trajectory_json is not None:
        trace_path = (
            ROOT / ARGS.trajectory_json
            if not ARGS.trajectory_json.is_absolute()
            else ARGS.trajectory_json
        )
        trace_times, trace_positions = _load_exact_trace(
            trace_path, len(excavator.dof_names)
        )
        trajectory_source = "EXACT_RECORDED_TRACE"
        phase_names = []
    else:
        trace_times, trace_positions, phase_names = _target_sequence(config, prototype_cfg)
        trajectory_source = "AUDITED_CONFIG_TARGET_SEQUENCE"

    represented_volume = float(
        prototype_cfg.get("represented_bulk_volume_per_particle_m3", spacing ** 3)
    )
    accounting = ParticleVolumeAccounting(represented_volume)
    initial_particle_count = int(len(seed_world))
    initial_represented_volume_m3 = accounting.volume_from_count(initial_particle_count)

    sample_period = float(prototype_cfg.get("audit_sample_period_s", 0.05))
    final_hold = float(prototype_cfg.get("final_hold_s", 1.0))
    end_time = float(trace_times[-1] + final_hold)
    next_sample = 0.0
    start_sim = float(world.current_time)
    wall_start = perf_counter()

    records = []
    peak_inside_count = 0
    peak_inside_volume_m3 = 0.0
    peak_inside_mean_local_z_m = None
    peak_deep_cavity_count = 0
    max_particle_count_error = 0

    while float(world.current_time) - start_sim < end_time - 0.5 * config.physics_dt_s:
        elapsed = float(world.current_time) - start_sim
        target = _interpolate_trace(trace_times, trace_positions, elapsed)
        controller.apply_action(ArticulationAction(joint_positions=target))
        world.step(render=not ARGS.headless)

        elapsed = float(world.current_time) - start_sim
        if elapsed + 1.0e-9 < next_sample:
            continue
        next_sample += sample_period

        positions_world = _particle_positions_world(points, particle_set_api)
        max_particle_count_error = max(
            max_particle_count_error, abs(len(positions_world) - initial_particle_count)
        )
        tool_world_now = _matrix_world_m(tool_prim, UsdGeom)
        positions_local = inverse_transform_points(tool_world_now, positions_world)
        inside = cavity.contains_local(
            positions_local,
            wall_margin_m=float(prototype_cfg.get("cavity_wall_margin_m", 0.0)),
        )
        inside_count = int(np.count_nonzero(inside))
        inside_volume = accounting.volume_from_count(inside_count)
        if inside_count:
            mean_local_z = float(np.mean(positions_local[inside, 2]))
            deep_count = int(
                np.count_nonzero(
                    inside
                    & (
                        positions_local[:, 1]
                        <= float(prototype_cfg.get("deep_cavity_y_threshold_m", -1.0))
                    )
                )
            )
        else:
            mean_local_z = None
            deep_count = 0

        peak_inside_count = max(peak_inside_count, inside_count)
        peak_inside_volume_m3 = max(peak_inside_volume_m3, inside_volume)
        peak_deep_cavity_count = max(peak_deep_cavity_count, deep_count)
        if mean_local_z is not None:
            peak_inside_mean_local_z_m = (
                mean_local_z
                if peak_inside_mean_local_z_m is None
                else max(peak_inside_mean_local_z_m, mean_local_z)
            )

        records.append(
            {
                "t_s": elapsed,
                "target_joint_position_rad": target.tolist(),
                "measured_joint_position_rad": np.asarray(
                    excavator.get_joint_positions(), dtype=np.float64
                ).reshape(-1).tolist(),
                "particle_count": int(len(positions_world)),
                "cavity_particle_count": inside_count,
                "cavity_represented_bulk_volume_m3": inside_volume,
                "cavity_mean_local_z_m": mean_local_z,
                "deep_cavity_particle_count": deep_count,
            }
        )

    wall_seconds = perf_counter() - wall_start
    simulated_seconds = float(world.current_time) - start_sim
    final_world = _particle_positions_world(points, particle_set_api)
    final_local = inverse_transform_points(_matrix_world_m(tool_prim, UsdGeom), final_world)
    final_inside = cavity.contains_local(
        final_local,
        wall_margin_m=float(prototype_cfg.get("cavity_wall_margin_m", 0.0)),
    )
    final_inside_count = int(np.count_nonzero(final_inside))
    final_retained_volume_m3 = accounting.volume_from_count(final_inside_count)

    baseline_payload = float(prototype_cfg["comparison_2p5d_payload_m3"])
    capture_improvement_ratio = (
        final_retained_volume_m3 / baseline_payload if baseline_payload > 0.0 else None
    )
    minimum_meaningful = float(prototype_cfg["minimum_meaningful_retained_volume_m3"])
    dimensionality_hypothesis_supported = (
        final_retained_volume_m3 >= minimum_meaningful
        and final_retained_volume_m3 > baseline_payload
        and max_particle_count_error == 0
    )

    report = {
        "schema": "bucket-local-3d-canonical/v0.1",
        "status": "UNVALIDATED_RUNTIME_PROTOTYPE",
        "physics_provenance": {
            "global_mobile_v2_modified": False,
            "production_coupling_enabled": False,
            "local_solver": "ISAAC_SIM_4_5_PHYSX_PBD_SOLID_PARTICLES",
            "material_status": "ENGINEERING_TEST_PARAMETERS_NOT_IRON_ORE_CALIBRATED",
            "trajectory_source": trajectory_source,
            "phase_names": phase_names,
        },
        "particle_system": {
            "initial_particle_count": initial_particle_count,
            "represented_bulk_volume_per_particle_m3": represented_volume,
            "initial_represented_bulk_volume_m3": initial_represented_volume_m3,
            "seed_world_vertical_shift_m": seed_lift_m,
            "particle_count_conservation_error": int(max_particle_count_error),
        },
        "bucket_fill": {
            "peak_cavity_particle_count": int(peak_inside_count),
            "peak_cavity_represented_bulk_volume_m3": float(peak_inside_volume_m3),
            "final_retained_particle_count": final_inside_count,
            "final_retained_represented_bulk_volume_m3": final_retained_volume_m3,
            "peak_inside_mean_local_z_m": peak_inside_mean_local_z_m,
            "peak_deep_cavity_particle_count": int(peak_deep_cavity_count),
        },
        "comparison": {
            "production_2p5d_payload_m3": baseline_payload,
            "final_retained_to_2p5d_ratio": capture_improvement_ratio,
            "minimum_meaningful_retained_volume_m3": minimum_meaningful,
            "dimensionality_hypothesis_supported": dimensionality_hypothesis_supported,
        },
        "performance": {
            "simulated_seconds": simulated_seconds,
            "wall_seconds": wall_seconds,
            "rtf": simulated_seconds / wall_seconds if wall_seconds > 0.0 else None,
        },
        "acceptance": {
            "LOCAL_3D_PARTICLE_COUNT_CONSERVATION": max_particle_count_error == 0,
            "BUCKET_CAVITY_CAPTURE": peak_inside_volume_m3 > baseline_payload,
            "BUCKET_RETENTION": final_retained_volume_m3 >= minimum_meaningful,
            "DEEP_CAVITY_TRANSPORT": peak_deep_cavity_count > 0,
            "DIMENSIONALITY_HYPOTHESIS": dimensionality_hypothesis_supported,
        },
        "records": records,
    }

    output = ROOT / ARGS.output if not ARGS.output.is_absolute() else ARGS.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    print(f"[BucketLocal3D] report: {output}")


try:
    main()
finally:
    simulation_app.close()
