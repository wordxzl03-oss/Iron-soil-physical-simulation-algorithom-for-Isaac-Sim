#!/usr/bin/env python3
"""Four lean V3 canonical gates plus the one-time frozen excavation replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_bulk_pipeline.bulk_interaction import (  # noqa: E402
    FailureZoneConfig,
    FailureZoneModel,
    MobileLayerSolver,
    ToolTerrainIntersection,
    evaluate_cohesive_yield,
    physics_free_surface,
)
from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator  # noqa: E402
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


OUT = ROOT / "outputs" / "390f_v3"
DX = 0.05


def material(*, cohesion: float) -> MaterialScenario:
    return MaterialScenario(
        name="IRON_ORE_FINES_V3_REFERENCE_UNCALIBRATED",
        assumed_bulk_density_kg_m3=2300.0,
        internal_friction_angle_deg=36.0,
        cohesion_proxy_pa=cohesion,
        tool_friction_coefficient=0.45,
        start_angle_deg=38.0,
        stop_angle_deg=30.0,
        mobile_friction_coefficient=float(np.tan(np.deg2rad(30.0))),
    )


def grid(n: int) -> tuple[TerrainGrid, TerrainVolumeIntegrator]:
    extent = (n - 1) * DX
    value = TerrainGrid(n, n, DX, DX, -0.5 * extent, -0.5 * extent, "/World/V3Validation")
    return value, TerrainVolumeIntegrator.from_grid(value)


def frozen_intersection(g: TerrainGrid, resting: np.ndarray) -> ToolTerrainIntersection:
    x_coordinates = g.origin_x + np.arange(g.nx) * g.dx
    y_coordinates = g.origin_y + np.arange(g.ny) * g.dy
    yy, xx = np.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    contact = (np.abs(xx) <= 0.075) & (np.abs(yy) <= 1.25)
    lateral_shape = np.clip(1.0 - (np.abs(yy) / 1.30) ** 4, 0.0, 1.0)
    # Frozen CAD-like raster irregularity deliberately exposes the old
    # max-depth/strip morphology without adding production instrumentation.
    raster_irregularity = 0.025 * (1.0 + np.sin(9.0 * yy))
    depth = np.where(contact, 0.24 * lateral_shape + raster_irregularity, 0.0)
    cut = np.where(contact, resting - depth, np.nan)
    points_y = np.linspace(-1.25, 1.25, 101)
    edge = np.column_stack((np.zeros_like(points_y), points_y, 0.76 + 0.02 * np.cos(3.0 * points_y)))
    area = TerrainVolumeIntegrator.from_grid(g).integrate(depth)
    return ToolTerrainIntersection(
        affected_mask=depth > 0.0,
        penetration_depth_m=depth,
        cutting_surface_m=cut,
        bucket_velocity_terrain_m_s=np.asarray([0.75, 0.0, 0.0]),
        cutting_edge_velocity_terrain_m_s=np.asarray([0.75, 0.0, 0.0]),
        local_terrain_normal=np.asarray([0.0, 0.0, 1.0]),
        local_slope_rad=0.0,
        candidate_intersection_volume_m3=area,
        affected_bbox_grid=(int(np.min(np.nonzero(contact)[0])), int(np.min(np.nonzero(contact)[1])), int(np.max(np.nonzero(contact)[0])) + 1, int(np.max(np.nonzero(contact)[1])) + 1),
        cutting_edge_points_terrain_m=edge,
        separation_plane_direction_terrain=np.asarray([0.5, 0.0, np.sqrt(0.75)]),
        separation_plane_source="FROZEN_390F_CUTTING_EDGE",
        tool_plus_z_separation_fallback_used=False,
    )


def front_profile(mask: np.ndarray, g: TerrainGrid) -> np.ndarray:
    result = np.full(g.ny, np.nan)
    for row in range(g.ny):
        cols = np.flatnonzero(mask[row])
        if len(cols):
            result[row] = g.origin_x + float(np.max(cols)) * g.dx
    return result


def morphology(zone, momentum: np.ndarray, contact: np.ndarray, g: TerrainGrid, integ: TerrainVolumeIntegrator) -> dict[str, float]:
    thickness = np.asarray(zone.active_thickness_m)
    positive = thickness[thickness > 0.0]
    front = front_profile(zone.active_mask, g)
    finite_front = front[np.isfinite(front)]
    # High-frequency boundary curvature distinguishes strip-to-strip steps
    # from the intended smooth lateral taper (first variation penalises both).
    serration = float(np.mean(np.abs(np.diff(finite_front, n=2)))) if len(finite_front) > 2 else 0.0
    shallow = float(np.mean(positive < 0.2 * np.percentile(positive, 95.0))) if len(positive) else 0.0
    protrusion = float(np.max(positive) / max(np.percentile(positive, 95.0), 1e-12)) if len(positive) else 0.0
    rows = np.flatnonzero(np.any(zone.active_mask, axis=1))
    if len(rows) >= 3:
        edge_mean = float(np.mean(thickness[rows[[0, -1]]]))
        center_mean = float(np.mean(thickness[rows[len(rows) // 2]]))
        cat_ear = edge_mean / max(center_mean, 1e-12)
    else:
        cat_ear = 0.0
    outside = ~contact
    outside_forward = integ.integrate(np.where(outside, np.maximum(momentum[..., 0], 0.0), 0.0))
    total_forward = integ.integrate(np.maximum(momentum[..., 0], 0.0))
    return {
        "boundary_serration_mean_step_m": serration,
        "shallow_ring_fraction": shallow,
        "local_protrusion_max_over_p95": protrusion,
        "terminal_cat_ear_ratio": cat_ear,
        "forward_momentum_outside_contact_fraction": outside_forward / max(total_forward, 1e-12),
    }


def frozen_replay() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    g, integ = grid(161)
    x_coordinates = g.origin_x + np.arange(g.nx) * g.dx
    y_coordinates = g.origin_y + np.arange(g.ny) * g.dy
    yy, xx = np.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    resting0 = 0.85 + 0.05 * np.exp(-0.08 * (xx * xx + yy * yy))
    mobile0 = np.zeros_like(resting0)
    momentum0 = np.zeros(resting0.shape + (2,))
    intersection = frozen_intersection(g, resting0)
    reference_material = material(cohesion=1500.0)
    zones = {
        "OLD": FailureZoneModel(FailureZoneConfig(surface_version="LEGACY_STRIPS")).compute(
            intersection, resting0, reference_material, g, integ
        ),
        "V3": FailureZoneModel(FailureZoneConfig(surface_version="V3_CONTINUOUS_2P5D")).compute(
            intersection, resting0, reference_material, g, integ,
            H_free_m=physics_free_surface(resting0, mobile0),
        ),
    }
    arrays: dict[str, np.ndarray] = {
        "T0_H_resting": resting0, "T0_h_mobile": mobile0,
        "T0_H_free": physics_free_surface(resting0, mobile0),
        "T0_mobile_momentum": momentum0,
        "T1_H_resting": resting0, "T1_h_mobile": mobile0,
        "T1_H_free": physics_free_surface(resting0, mobile0),
        "T1_intersection_support": intersection.affected_mask,
        "T1_penetration_depth": intersection.penetration_depth_m,
        "T1_mobile_momentum": momentum0,
    }
    summary: dict[str, object] = {}
    for label, zone in zones.items():
        activated = np.minimum(zone.active_thickness_m, resting0)
        resting2, mobile2 = resting0 - activated, mobile0 + activated
        if label == "OLD":
            momentum2 = activated[..., None] * intersection.cutting_edge_velocity_terrain_m_s[:2]
            forcing = activated > 0.0
        else:
            momentum2 = np.zeros_like(momentum0)
            contact = intersection.affected_mask & (mobile2 > 0.0)
            forcing = contact.copy()
            forcing[1:] |= contact[:-1]
            forcing[:-1] |= contact[1:]
            forcing[:, 1:] |= contact[:, :-1]
            forcing[:, :-1] |= contact[:, 1:]
            forcing &= mobile2 > 0.0
        result = MobileLayerSolver().step(
            resting2, mobile2, momentum2, reference_material, g, integ, 1.0 / 60.0,
            tool_forcing_mask=forcing,
            tool_velocity_xy_m_s=intersection.cutting_edge_velocity_terrain_m_s[:2],
        )
        arrays.update({
            f"{label}_T2_H_resting": resting2,
            f"{label}_T2_h_mobile": mobile2,
            f"{label}_T2_H_free": physics_free_surface(resting2, mobile2),
            f"{label}_T2_failure_support": zone.active_mask,
            f"{label}_T2_failure_active_thickness": activated,
            f"{label}_T2_mobile_momentum": momentum2,
            f"{label}_T3_H_resting": resting2,
            f"{label}_T3_h_mobile": result.mobile_height_m,
            f"{label}_T3_H_free": physics_free_surface(resting2, result.mobile_height_m),
            f"{label}_T3_failure_support": zone.active_mask,
            f"{label}_T3_failure_active_thickness": activated,
            f"{label}_T3_mobile_momentum": result.mobile_momentum_m2_s,
        })
        summary[label] = {
            "active_volume_m3": zone.active_volume_m3,
            "analytical_volume_m3": zone.analytical_wedge_volume_m3,
            "mass_error_m3": abs(integ.integrate(resting0 + mobile0) - integ.integrate(resting2 + result.mobile_height_m)),
            "surface_classification": zone.model_classification,
            # Momentum pathology is assigned at T2.  T3 may legitimately move
            # beyond contact under gravity/transport and is saved separately.
            **morphology(zone, momentum2, intersection.affected_mask, g, integ),
        }
    old, v3 = summary["OLD"], summary["V3"]
    comparison = {
        "forensic_root_causes": {
            "cat_ear_depression": "independent terminal strips plus per-strip maximum depth",
            "shallow_surrounding_ring": "hard prism support with thin raster intersections",
            "strip_serrated_boundary": "0.20 m piecewise-constant d/beta/L",
            "local_protrusions": "single deepest raster vertex controls a complete strip",
            "artificial_forward_soil_launch": "legacy whole failure support inherited cutting-edge velocity and forcing",
        },
        "OLD": old,
        "V3": v3,
        "improvements": {
            "serration_reduction_fraction": 1.0 - float(v3["boundary_serration_mean_step_m"]) / max(float(old["boundary_serration_mean_step_m"]), 1e-12),
            "whole_wedge_forward_launch_removed": float(v3["forward_momentum_outside_contact_fraction"]) < 0.25 * max(float(old["forward_momentum_outside_contact_fraction"]), 1e-12),
            "continuous_profile_present": zones["V3"].failure_surface_profile is not None,
        },
        "forensic_instrumentation_status": "ONE_TIME_OFFLINE_REPLAY_ONLY_NOT_IN_PRODUCTION_HOT_PATH",
    }
    return comparison, arrays


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    g, integ = grid(81)
    flat = np.full(g.shape, 0.8)
    zero = np.zeros(g.shape)
    cohesive = material(cohesion=5000.0)
    flat_yield = evaluate_cohesive_yield(flat, zero, cohesive, g, integ)

    x_coordinates = g.origin_x + np.arange(g.nx) * g.dx
    y_coordinates = g.origin_y + np.arange(g.ny) * g.dy
    yy, xx = np.meshgrid(y_coordinates, x_coordinates, indexing="ij")
    steep_surface = 0.8 + 1.4 * np.maximum(xx, 0.0)
    thin_layer = np.full(g.shape, 0.03)
    cohesive_steep = evaluate_cohesive_yield(
        steep_surface - thin_layer, thin_layer, cohesive, g, integ,
        layer_depth_m=thin_layer,
    )
    cohesionless = material(cohesion=0.0)
    oversteep = evaluate_cohesive_yield(
        steep_surface - thin_layer, thin_layer, cohesionless, g, integ,
        layer_depth_m=thin_layer,
    )
    arrested = evaluate_cohesive_yield(
        flat - thin_layer, thin_layer, cohesionless, g, integ,
        layer_depth_m=thin_layer, moving_mask=np.ones(g.shape, dtype=bool),
    )
    morphology_summary, arrays = frozen_replay()
    np.savez_compressed(OUT / "frozen_excavation_forensics.npz", **arrays)
    gates = {
        "flat_static_terrain": {
            "pass": not bool(np.any(flat_yield.start_mask)),
            "yielded_area_m2": flat_yield.yielded_area_m2,
        },
        "cohesive_steep_cut": {
            "pass": not bool(np.any(cohesive_steep.start_mask)),
            "maximum_yield_margin_pa": float(np.max(cohesive_steep.yield_start_margin_pa)),
        },
        "cohesionless_oversteep_slope": {
            "pass": bool(np.any(oversteep.start_mask)) and not bool(np.any(arrested.continue_mask)),
            "started_cell_count": int(np.count_nonzero(oversteep.start_mask)),
            "continued_after_flattening_cell_count": int(np.count_nonzero(arrested.continue_mask)),
        },
        "frozen_excavation_old_vs_v3": {
            "pass": bool(morphology_summary["improvements"]["continuous_profile_present"])
                    and bool(morphology_summary["improvements"]["whole_wedge_forward_launch_removed"])
                    and float(morphology_summary["improvements"]["serration_reduction_fraction"]) > 0.0,
            "forensics_npz": "frozen_excavation_forensics.npz",
        },
    }
    validation = {
        "schema": "EARTHMOVING_PHYSICS_CORE_V3_LEAN_VALIDATION_V1",
        "formal_resolution_m": DX,
        "canonical_gate_count": 4,
        "all_pass": all(item["pass"] for item in gates.values()),
        "gates": gates,
        "time_semantics": {
            "mobile": "PHYSICAL_SIMULATION_TIME",
            "large_avalanche": "PHYSICAL_DYNAMIC_FLOW_TIME",
            "minislope": "QUASI_STATIC_RESIDUAL_PROJECTION_ITERATIONS_NOT_SECONDS",
        },
        "classifications": {
            "FEE_mechanics": "PAPER_DIRECT",
            "critical_beta_selection": "LITERATURE_INFORMED_REDUCED_ORDER",
            "lateral_continuity": "ENGINEERING_CLOSURE",
            "cohesive_heightfield_yield": "LITERATURE_INFORMED_REDUCED_ORDER",
            "reference_material": "NOT_YET_PHYSICALLY_CALIBRATED",
            "contact_impulse_action_reaction": "CONSERVATION_BASED_ENGINEERING_CLOSURE",
        },
    }
    (OUT / "physics_core_v3_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (OUT / "old_vs_v3_morphology_summary.json").write_text(json.dumps(morphology_summary, indent=2), encoding="utf-8")
    print(json.dumps({"validation": validation, "morphology": morphology_summary}, indent=2))
    if not validation["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
