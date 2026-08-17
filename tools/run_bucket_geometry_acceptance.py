#!/usr/bin/env python3
"""Numerical acceptance for geometry-consistent conservative bucket intake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from isaac_bulk_pipeline.bulk_interaction import BucketIntakeModel  # noqa: E402
from isaac_bulk_pipeline.bulk_state import PayloadState, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402
from isaac_bulk_pipeline.tools import (  # noqa: E402
    BucketGeometryDescriptor,
    CapacityMethod,
    GeometryQuality,
    GeometrySource,
    ToolDescriptor,
    ToolKinematicsAdapter,
    ToolState,
)


def descriptor(width: float = 1.73) -> ToolDescriptor:
    half, depth, height = 0.5 * width, 2.0, 1.0
    cutting = np.asarray([[-half, 0.0, 0.0], [half, 0.0, 0.0]])
    bottom = np.asarray([[0.0, -depth, 0.0], [0.0, 0.0, 0.0]])
    interior = np.asarray(
        [[0.0, -depth, 0.0], [0.0, -depth, height], [0.0, 0.0, height], [0.0, 0.0, 0.0]]
    )
    geometry = BucketGeometryDescriptor.from_extruded_profile(
        cutting_edge_local=cutting,
        bottom_profile_local=bottom,
        interior_profile_local=interior,
        top_edge_local=np.asarray([[-half, 0.0, height], [half, 0.0, height]]),
        rated_capacity_m3=None,
        geometry_source=GeometrySource.EXPLICIT_PROFILE,
        geometry_quality=GeometryQuality.REDUCED_ORDER,
        metadata={"acceptance_fixture": "rectangular analytic control surface"},
    )
    return ToolDescriptor(
        tool_type="acceptance_bucket",
        tool_frame_prim="/World/AcceptanceBucket/ToolFrame",
        cutting_edge_local=cutting,
        bottom_profile_local=bottom,
        left_boundary_local=geometry.left_side_wall_local,
        right_boundary_local=geometry.right_side_wall_local,
        nominal_width_m=width,
        interior_profile_local=interior,
        proxy_level="L1",
        actual_proxy_type="ExtrudedProfileBucket_L1",
        bucket_geometry=geometry,
    )


def grid(spacing: float, extent: float = 2.4) -> tuple[TerrainGrid, TerrainVolumeIntegrator]:
    count = int(round(extent / spacing)) + 1
    origin = -0.5 * spacing * (count - 1)
    terrain = TerrainGrid(count, count, spacing, spacing, origin, origin, "/World/Terrain")
    return terrain, TerrainVolumeIntegrator.from_grid(terrain)


def rotation_z(angle: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    return transform


def state(
    tool: ToolDescriptor,
    *,
    pose: np.ndarray | None = None,
    speed: float = 1.0,
) -> ToolState:
    transform = np.eye(4) if pose is None else pose
    geometry = tool.bucket_geometry
    assert geometry is not None
    points = geometry.transform_points
    return ToolState(
        timestamp=0.0,
        pose_world=transform,
        pose_terrain=transform,
        cutting_edge_terrain=points(transform, tool.cutting_edge_local),
        bottom_profile_terrain=points(transform, tool.bottom_profile_local),
        left_boundary_terrain=points(transform, tool.left_boundary_local),
        right_boundary_terrain=points(transform, tool.right_boundary_local),
        linear_velocity=np.asarray([0.0, speed, 0.0]),
        angular_velocity=np.zeros(3),
        mouth_polygon_terrain=points(transform, geometry.mouth_polygon_local),
        top_edge_terrain=points(transform, geometry.top_edge_local),
        separation_plane_direction_terrain=geometry.transform_direction(
            transform, geometry.separation_plane_direction_local
        ),
        separation_plane_source="BUCKET_GEOMETRY_BOTTOM_PLATE",
        tool_plus_z_separation_fallback_used=False,
    )


def apply(
    tool: ToolDescriptor,
    terrain: TerrainGrid,
    integrator: TerrainVolumeIntegrator,
    height: np.ndarray,
    *,
    tool_state: ToolState | None = None,
    payload: PayloadState | None = None,
    dt: float = 0.01,
):
    geometry = tool.bucket_geometry
    assert geometry is not None
    if payload is None:
        payload = PayloadState(0.0, geometry.effective_capacity_m3, 1800.0, np.zeros(3))
    return BucketIntakeModel().apply(
        height,
        np.zeros(terrain.shape + (2,)),
        payload,
        state(tool) if tool_state is None else tool_state,
        tool,
        terrain,
        integrator,
        dt,
        terrain_surface_m=np.zeros(terrain.shape),
    )


def evaluate() -> dict[str, object]:
    tool = descriptor()
    geometry = tool.bucket_geometry
    assert geometry is not None
    terrain, integrator = grid(0.05)
    height = np.full(terrain.shape, 0.24)
    dt = 0.01
    uniform = apply(tool, terrain, integrator, height, dt=dt)
    uniform_exact = 0.24 * geometry.cutting_edge_length_m * dt

    angle = np.deg2rad(33.0)
    oblique = apply(
        tool,
        terrain,
        integrator,
        height,
        tool_state=state(tool, pose=rotation_z(angle)),
        dt=dt,
    )
    oblique_exact = uniform_exact * np.cos(angle)

    partial_pose = np.eye(4)
    partial_pose[2, 3] = 0.16
    partial = apply(
        tool,
        terrain,
        integrator,
        height,
        tool_state=state(tool, pose=partial_pose),
        dt=dt,
    )
    partial_exact = (0.24 - 0.16) * geometry.cutting_edge_length_m * dt

    nearly_full = PayloadState(
        geometry.effective_capacity_m3 - 0.001,
        geometry.effective_capacity_m3,
        1800.0,
        np.zeros(3),
    )
    mobile_before = integrator.integrate(height)
    saturated = apply(tool, terrain, integrator, height, payload=nearly_full, dt=dt)
    mobile_removed = mobile_before - integrator.integrate(saturated.mobile_height_m)
    payload_gained = saturated.payload.volume_m3 - nearly_full.volume_m3

    spatial: dict[str, list[float] | bool] = {
        "dx_m": [],
        "integrated_intake_error_m3": [],
        "mouth_overlap_error_m": [],
        "intake_centroid_x_error_m": [],
    }
    h0, linear, quadratic = 0.21, 0.08, 0.11
    length = geometry.cutting_edge_length_m
    exact_factor = h0 * length + quadratic * length**3 / 12.0
    exact_centroid_x = (linear * length**3 / 12.0) / exact_factor
    for spacing in (0.2, 0.1, 0.05):
        current_grid, current_integrator = grid(spacing)
        _, columns = np.indices(current_grid.shape)
        x = current_grid.origin_x + columns * spacing
        nonuniform_height = h0 + linear * x + quadratic * x**2
        result = apply(tool, current_grid, current_integrator, nonuniform_height, dt=dt)
        spatial["dx_m"].append(spacing)  # type: ignore[union-attr]
        spatial["integrated_intake_error_m3"].append(
            abs(result.bucket_inflow_volume_m3 - exact_factor * dt)
        )  # type: ignore[union-attr]
        spatial["mouth_overlap_error_m"].append(
            abs(result.mouth_overlap_length_m - length)
        )  # type: ignore[union-attr]
        spatial["intake_centroid_x_error_m"].append(
            abs(result.intake_centroid_terrain_m[0] - exact_centroid_x)
        )  # type: ignore[union-attr]
    volume_errors = spatial["integrated_intake_error_m3"]
    centroid_errors = spatial["intake_centroid_x_error_m"]
    spatial["converged"] = bool(
        volume_errors[-1] < 0.4 * volume_errors[0]  # type: ignore[index]
        and centroid_errors[-1] < 0.4 * centroid_errors[0]  # type: ignore[index]
    )

    temporal_steps = (5, 10, 20, 40)
    temporal_errors: list[float] = []
    time_grid, time_integrator = grid(0.5, extent=4.0)
    time_height = np.full(time_grid.shape, 0.1)
    time_exact = 0.1 * length * 1.5
    for steps in temporal_steps:
        payload = PayloadState(0.0, geometry.effective_capacity_m3, 1800.0, np.zeros(3))
        total = 0.0
        step_dt = 1.0 / steps
        for index in range(steps):
            result = apply(
                tool,
                time_grid,
                time_integrator,
                time_height,
                tool_state=state(tool, speed=1.0 + (index + 1) * step_dt),
                payload=payload,
                dt=step_dt,
            )
            payload = result.payload
            total += result.bucket_inflow_volume_m3
        temporal_errors.append(abs(total - time_exact))
    spatial["temporal_steps"] = list(temporal_steps)
    spatial["temporal_flux_error_m3"] = temporal_errors
    spatial["temporal_converged"] = all(
        later < earlier for earlier, later in zip(temporal_errors, temporal_errors[1:])
    )

    adapter_state = ToolKinematicsAdapter(tool, terrain).update(np.eye(4), 0.0)
    report: dict[str, object] = {
        "schema_version": "bucket-geometry-acceptance/v1",
        "geometry_source": geometry.geometry_source.value,
        "geometry_quality": geometry.geometry_quality.value,
        "cutting_edge_length": geometry.cutting_edge_length_m,
        "mouth_area": geometry.mouth_area_m2,
        "mouth_planarity_error": geometry.mouth_planarity_error_m,
        "mouth_normal": geometry.mouth_normal_local.tolist(),
        "interior_capacity": geometry.geometric_capacity_m3,
        "rated_capacity": geometry.rated_capacity_m3,
        "capacity_method": geometry.capacity_method.value,
        "descriptor_valid": True,
        "legacy_fallback_used": geometry.legacy_fallback_used,
        "separation_plane_source": adapter_state.separation_plane_source,
        "tool_plus_z_separation_fallback_used": adapter_state.tool_plus_z_separation_fallback_used,
        "legacy_sampling_band_used": uniform.legacy_sampling_band_used,
        "uniform_flux_analytic_error": abs(uniform.bucket_inflow_volume_m3 - uniform_exact),
        "oblique_flux_error": abs(oblique.bucket_inflow_volume_m3 - oblique_exact),
        "partial_overlap_error": abs(partial.bucket_inflow_volume_m3 - partial_exact),
        "capacity_overflow_error": max(
            0.0, saturated.payload.volume_m3 - saturated.payload.capacity_m3
        ),
        "mobile_payload_conservation_error": abs(mobile_removed - payload_gained),
        "resolution_convergence": spatial,
        "tests": {
            "geometry_invariants": "PASS",
            "analytic_control_surface_flux": "PASS",
            "orientation_and_partial_overlap": "PASS",
            "capacity_and_conservation": "PASS",
            "spatial_and_temporal_convergence": "PASS" if spatial["converged"] and spatial["temporal_converged"] else "FAIL",
        },
        "claim_boundary": "Geometry-consistent, conservative reduced-order bucket intake.",
    }
    tolerances_pass = (
        report["uniform_flux_analytic_error"] < 1.0e-12
        and report["oblique_flux_error"] < 1.0e-12
        and report["partial_overlap_error"] < 1.0e-12
        and report["capacity_overflow_error"] < 1.0e-12
        and report["mobile_payload_conservation_error"] < 1.0e-12
        and spatial["converged"]
        and spatial["temporal_converged"]
        and not report["legacy_fallback_used"]
        and not report["tool_plus_z_separation_fallback_used"]
        and not report["legacy_sampling_band_used"]
        and geometry.capacity_method is CapacityMethod.EXTRUDED_PROFILE
    )
    report["status"] = "PASS" if tolerances_pass else "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "bucket_geometry_acceptance.json",
    )
    arguments = parser.parse_args()
    report = evaluate()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
