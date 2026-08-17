#!/usr/bin/env python3
"""Generate reproducible pure-Python Failure Zone acceptance evidence."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from time import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_bulk_pipeline.bulk_interaction import FailureZoneModel, ToolTerrainIntersection  # noqa: E402
from isaac_bulk_pipeline.bulk_state import (  # noqa: E402
    BulkStateManager,
    ConservativeTransfer,
    MaterialScenario,
    PayloadState,
    Reservoir,
    TerrainState,
    TerrainVolumeIntegrator,
)
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


OUTPUT = ROOT / "outputs/failure_zone_acceptance.json"


def material(phi: float = 32.0, cohesion: float = 500.0, delta_deg: float = 19.29) -> MaterialScenario:
    return MaterialScenario(
        name="failure_zone_acceptance_UNCALIBRATED",
        assumed_bulk_density_kg_m3=1800.0,
        internal_friction_angle_deg=phi,
        cohesion_proxy_pa=cohesion,
        tool_friction_coefficient=float(np.tan(np.deg2rad(delta_deg))),
        start_angle_deg=50.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )


def solve(
    *,
    spacing: float = 0.1,
    depth: float = 0.25,
    phi: float = 32.0,
    cohesion: float = 500.0,
    delta_deg: float = 19.29,
    rake_deg: float = 90.0,
    slope_deg: float = 0.0,
) -> tuple[TerrainGrid, TerrainVolumeIntegrator, np.ndarray, MaterialScenario, ToolTerrainIntersection, object]:
    count = int(round(6.0 / spacing)) + 1
    grid = TerrainGrid(count, count, spacing, spacing, -3.0, -3.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    rows, columns = np.indices(grid.shape)
    x = grid.origin_x + columns * grid.dx
    y = grid.origin_y + rows * grid.dy
    grade = np.tan(np.deg2rad(slope_deg))
    height = 2.0 + grade * x
    mask = (np.abs(x) <= 0.51 * spacing) & (np.abs(y) <= 1.001)
    penetration = np.where(mask, depth, 0.0)
    surface = np.where(mask, height - depth, np.inf)
    normal = np.asarray([-grade, 0.0, 1.0])
    normal /= np.linalg.norm(normal)
    rake = np.deg2rad(rake_deg)
    intersection = ToolTerrainIntersection(
        affected_mask=mask,
        penetration_depth_m=penetration,
        cutting_surface_m=surface,
        bucket_velocity_terrain_m_s=np.asarray([1.0, 0.0, 0.0]),
        cutting_edge_velocity_terrain_m_s=np.asarray([1.0, 0.0, 0.0]),
        local_terrain_normal=normal,
        local_slope_rad=abs(np.deg2rad(slope_deg)),
        candidate_intersection_volume_m3=integrator.integrate(penetration),
        affected_bbox_grid=(0, 0, grid.ny, grid.nx),
        cutting_edge_points_terrain_m=np.asarray([[0.0, -1.0, 2.0 - depth], [0.0, 1.0, 2.0 - depth]]),
        separation_plane_direction_terrain=np.asarray([np.cos(rake), 0.0, np.sin(rake)]),
    )
    scenario = material(phi, cohesion, delta_deg)
    failure = FailureZoneModel().compute(intersection, height, scenario, grid, integrator)
    return grid, integrator, height, scenario, intersection, failure


def historical_regression_fixture() -> tuple[ToolTerrainIntersection, object]:
    """Re-evaluate the exact Phase-F fixture that exposed the old 70° bug."""

    grid = TerrainGrid(41, 41, 0.1, 0.1, -2.0, -2.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    height = np.ones(grid.shape)
    rows, columns = np.indices(grid.shape)
    x = grid.origin_x + columns * grid.dx
    y = grid.origin_y + rows * grid.dy
    mask = (np.abs(x) <= 0.9) & (np.abs(y) <= 0.08)
    penetration = np.where(mask, 0.25, 0.0)
    intersection = ToolTerrainIntersection(
        affected_mask=mask,
        penetration_depth_m=penetration,
        cutting_surface_m=np.where(mask, 0.75, np.inf),
        bucket_velocity_terrain_m_s=np.asarray([0.0, 1.0, 0.0]),
        cutting_edge_velocity_terrain_m_s=np.asarray([0.0, 1.0, 0.0]),
        local_terrain_normal=np.asarray([0.0, 0.0, 1.0]),
        local_slope_rad=0.0,
        candidate_intersection_volume_m3=integrator.integrate(penetration),
        affected_bbox_grid=(19, 11, 22, 30),
        cutting_edge_points_terrain_m=np.asarray([[-1.0, 0.0, 0.75], [1.0, 0.0, 0.75]]),
        separation_plane_direction_terrain=np.asarray([0.0, 0.0, 1.0]),
    )
    failure = FailureZoneModel().compute(intersection, height, material(), grid, integrator)
    return intersection, failure


def ledger_error(
    height: np.ndarray,
    scenario: MaterialScenario,
    failure: object,
    integrator: TerrainVolumeIntegrator,
) -> float:
    payload = PayloadState(0.0, 2.0, scenario.assumed_bulk_density_kg_m3, np.zeros(3))
    before = TerrainState(
        H_resting_m=height,
        mobile_height_m=np.zeros_like(height),
        mobile_momentum_m2_s=np.zeros(height.shape + (2,)),
        payload=payload,
        airborne_parcels=(),
        material=scenario,
        outflow_volume_m3=0.0,
        timestamp_s=0.0,
        action_index=0,
    )
    manager = BulkStateManager(before, integrator)
    next_state = TerrainState(
        H_resting_m=height - failure.active_thickness_m,
        mobile_height_m=failure.active_thickness_m,
        mobile_momentum_m2_s=np.zeros(height.shape + (2,)),
        payload=payload,
        airborne_parcels=(),
        material=scenario,
        outflow_volume_m3=0.0,
        timestamp_s=0.01,
        action_index=0,
    )
    manager.commit_transfers(
        next_state,
        [ConservativeTransfer(Reservoir.RESTING, Reservoir.MOBILE, failure.active_volume_m3, "failure_zone_acceptance")],
    )
    return manager.ledger_snapshot().balance.absolute_volume_error_m3


def main() -> int:
    _, integrator, height, scenario, intersection, baseline = solve()
    historical_intersection, historical_failure = historical_regression_fixture()
    sensitivity_specs = {
        "internal_friction_angle_deg": (20.0, 25.0, 30.0, 35.0, 40.0, 45.0),
        "cohesion_pa": (0.0, 500.0, 2000.0, 5000.0),
        "soil_tool_friction_angle_deg": (8.0, 18.0, 28.0),
        "penetration_depth_m": (0.10, 0.25, 0.45),
        "rake_angle_deg": (60.0, 75.0, 90.0, 105.0),
        "terrain_slope_deg": (-10.0, 0.0, 10.0),
    }
    argument_names = {
        "internal_friction_angle_deg": "phi",
        "cohesion_pa": "cohesion",
        "soil_tool_friction_angle_deg": "delta_deg",
        "penetration_depth_m": "depth",
        "rake_angle_deg": "rake_deg",
        "terrain_slope_deg": "slope_deg",
    }
    sensitivity = {}
    all_strips = []
    for name, values in sensitivity_specs.items():
        samples = []
        for value in values:
            failure = solve(**{argument_names[name]: value})[-1]
            all_strips.extend(failure.strip_geometries)
            samples.append(
                {
                    "value": value,
                    "mean_beta_deg": failure.estimated_failure_angle_deg,
                    "analytical_wedge_volume_m3": failure.analytical_wedge_volume_m3,
                    "estimated_static_resistance_n": failure.estimated_total_resistance_n,
                    "boundary_hit_rate": failure.failure_angle_boundary_hit_rate,
                }
            )
        sensitivity[name] = samples

    convergence = []
    for spacing in (0.20, 0.10, 0.05):
        failure = solve(spacing=spacing, rake_deg=75.0, slope_deg=7.0)[-1]
        convergence.append(
            {
                "grid_spacing_m": spacing,
                "analytical_wedge_volume_m3": failure.analytical_wedge_volume_m3,
                "rasterized_activated_volume_m3": failure.active_volume_m3,
                "absolute_error_m3": abs(failure.active_volume_m3 - failure.analytical_wedge_volume_m3),
                "relative_error": failure.relative_volume_error,
            }
        )

    beta = np.asarray([strip.failure_angle_deg for strip in all_strips])
    hit_rate = float(np.mean([strip.failure_angle_boundary_hit for strip in all_strips]))
    statuses = Counter(strip.solver_status for strip in all_strips)
    volume_tolerance = 1.0e-8
    boundary_tolerance = 0.20
    ledger = ledger_error(height, scenario, baseline, integrator)
    passed = bool(
        hit_rate <= boundary_tolerance
        and baseline.relative_volume_error <= volume_tolerance
        and max(item["relative_error"] for item in convergence) <= volume_tolerance
        and ledger <= 1.0e-9
        and np.ptp(beta) > 1.0
    )
    report = {
        "schema_version": "isaac-bulk-failure-zone-acceptance/v1",
        "generated_unix_s": time(),
        "status": "PASS" if passed else "FAIL",
        "theory_trace": "docs/FAILURE_ZONE_THEORY_TRACE.md",
        "solver_status": dict(statuses),
        "beta_distribution_deg": {
            "count": int(len(beta)),
            "minimum": float(np.min(beta)),
            "median": float(np.median(beta)),
            "mean": float(np.mean(beta)),
            "maximum": float(np.max(beta)),
        },
        "failure_angle_boundary_hit_rate": hit_rate,
        "baseline_volume": {
            "candidate_intersection_m3": intersection.candidate_intersection_volume_m3,
            "analytical_wedge_m3": baseline.analytical_wedge_volume_m3,
            "rasterized_requested_m3": baseline.rasterized_requested_volume_m3,
            "rasterized_activated_m3": baseline.active_volume_m3,
            "relative_volume_error": baseline.relative_volume_error,
        },
        "historical_regression_fixture_after_refactor": {
            "candidate_intersection_m3": historical_intersection.candidate_intersection_volume_m3,
            "analytical_wedge_m3": historical_failure.analytical_wedge_volume_m3,
            "rasterized_requested_m3": historical_failure.rasterized_requested_volume_m3,
            "rasterized_activated_m3": historical_failure.active_volume_m3,
            "relative_volume_error": historical_failure.relative_volume_error,
            "mean_beta_deg": historical_failure.estimated_failure_angle_deg,
            "failure_angle_boundary_hit_rate": historical_failure.failure_angle_boundary_hit_rate,
        },
        "material_ledger_absolute_volume_error_m3": ledger,
        "resolution_convergence": convergence,
        "sensitivity_summary": sensitivity,
        "acceptance_limits": {
            "failure_angle_boundary_hit_rate_max": boundary_tolerance,
            "relative_volume_error_max": volume_tolerance,
            "material_ledger_absolute_volume_error_m3_max": 1.0e-9,
        },
        "claim_boundary": "Pure-Python reduced-order geometry; material parameters are uncalibrated and constant-density.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(OUTPUT)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
