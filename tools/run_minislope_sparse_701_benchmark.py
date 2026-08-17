#!/usr/bin/env python3
"""Benchmark 701x701 reference, bbox baseline and compact tile MiniSlope."""

from __future__ import annotations

import argparse
import json
import platform
import resource
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from isaac_bulk_pipeline.config import SolverConfig
from isaac_bulk_pipeline.solvers import (
    EventDrivenMinimumSlopeAdapter,
    MinimumSlopeAdapter,
    SparseTileFrontierMinimumSlopeAdapter,
)
from isaac_bulk_pipeline.terrain import TerrainGrid


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "390f_v2" / "minislope_sparse_701_benchmark.json"


def _bbox(mask: np.ndarray) -> list[int]:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return [0, 0, 0, 0]
    return [int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1]


def _bbox_cells(bounds: list[int]) -> int:
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _slopes(height: np.ndarray, spacing: float) -> np.ndarray:
    diagonal = float(np.sqrt(2.0) * spacing)
    return np.concatenate(
        (
            (np.abs(height[1:, :] - height[:-1, :]) / spacing).ravel(),
            (np.abs(height[:, 1:] - height[:, :-1]) / spacing).ravel(),
            (np.abs(height[1:, 1:] - height[:-1, :-1]) / diagonal).ravel(),
            (np.abs(height[1:, :-1] - height[:-1, 1:]) / diagonal).ravel(),
        )
    )


def _case_toe(grid: TerrainGrid) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    critical_drop = np.tan(np.deg2rad(30.0)) * grid.dx
    baseline = np.repeat(
        (1.0 + np.arange(grid.nx) * 0.95 * critical_drop)[None, :],
        grid.ny,
        axis=0,
    )
    disturbed = baseline.copy()
    disturbed[346:355, 0] -= 0.10
    return baseline, disturbed, {
        "name": "NEAR_CRITICAL_TOE_COLLAPSE",
        "slope_fraction_of_critical": 0.95,
        "toe_seed_bbox_yx": [346, 0, 355, 1],
        "toe_cut_depth_m": 0.10,
    }


def _case_whole(grid: TerrainGrid) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    baseline = np.ones(grid.shape, dtype=np.float64)
    rows, cols = np.indices(grid.shape)
    disturbed = baseline + 0.030 * ((rows + cols) & 1)
    return baseline, disturbed, {
        "name": "WHOLE_PILE_STRESS",
        "disturbance": "domain-wide connected checkerboard over critical edge drop",
        "checkerboard_amplitude_m": 0.030,
    }


def _run_solver(
    name: str,
    solver,
    disturbed: np.ndarray,
    baseline: np.ndarray,
    grid: TerrainGrid,
) -> tuple[dict[str, Any], Any]:
    if hasattr(solver, "set_reference_height"):
        solver.set_reference_height(baseline)
    started = perf_counter()
    result = solver.solve(disturbed)
    wall_time = perf_counter() - started
    changed = np.abs(result.heightmap_stable - disturbed) > 1.0e-12
    changed_bbox = _bbox(changed)
    diagnostics = dict(result.diagnostics)
    if name == "FULL_DOMAIN_REFERENCE":
        active_cells = int(grid.nx * grid.ny)
        active_tiles = None
        bounding_box_cells = active_cells
    else:
        active_cells = int(diagnostics["reached_cell_count"])
        active_tiles = int(
            diagnostics.get(
                "ever_active_tile_count", diagnostics.get("active_tile_count", 0)
            )
        )
        bounding_box_cells = int(
            diagnostics.get(
                "bounding_box_cell_count",
                _bbox_cells(diagnostics["propagation_bbox_yx"]),
            )
        )
    record = {
        "backend": name,
        "implementation": solver.__class__.__name__,
        "iterations": int(result.iteration_count),
        "wall_time_s": wall_time,
        "active_cells": active_cells,
        "active_tiles": active_tiles,
        "bounding_box_cells": bounding_box_cells,
        "changed_bbox_yx": changed_bbox,
        "changed_bbox_cells": _bbox_cells(changed_bbox),
        "volume_before_m3": result.volume_before_m3,
        "volume_after_m3": result.volume_after_m3,
        "volume_balance_error_m3": result.volume_before_m3 - result.volume_after_m3,
        "converged": bool(result.converged),
        "classification": diagnostics.get("avalanche_classification"),
        "propagation_bbox_yx": diagnostics.get("propagation_bbox_yx"),
        "peak_active_tiles": diagnostics.get("peak_active_tile_count"),
        "retired_tile_total": diagnostics.get("retired_tile_total"),
        "compact_edge_evaluation_count": diagnostics.get(
            "compact_edge_evaluation_count"
        ),
        "full_domain_edge_evaluation_equivalent": diagnostics.get(
            "full_domain_edge_evaluation_equivalent"
        ),
        "true_sparse_tile_frontier": diagnostics.get(
            "true_sparse_tile_frontier", False
        ),
    }
    return record, result


def _comparison(
    reference, candidate, disturbed: np.ndarray, grid: TerrainGrid
) -> dict[str, Any]:
    difference = candidate.heightmap_stable - reference.heightmap_stable
    reference_delta = reference.heightmap_stable - disturbed
    reference_slopes = _slopes(reference.heightmap_stable, grid.dx)
    candidate_slopes = _slopes(candidate.heightmap_stable, grid.dx)
    quantiles = [0.0, 0.5, 0.9, 0.99, 1.0]
    reference_bbox = _bbox(
        np.abs(reference.heightmap_stable - disturbed) > 1.0e-12
    )
    candidate_bbox = _bbox(
        np.abs(candidate.heightmap_stable - disturbed) > 1.0e-12
    )
    comparison = {
        "height_linf_m": float(np.max(np.abs(difference))),
        "height_normalized_l1": float(
            np.sum(np.abs(difference))
            / max(np.sum(np.abs(reference_delta)), 1.0e-12)
        ),
        "volume_after_error_m3": float(
            candidate.volume_after_m3 - reference.volume_after_m3
        ),
        "iteration_difference": int(
            candidate.iteration_count - reference.iteration_count
        ),
        "slope_quantiles": quantiles,
        "reference_slope_values": np.quantile(reference_slopes, quantiles).tolist(),
        "candidate_slope_values": np.quantile(candidate_slopes, quantiles).tolist(),
        "slope_quantile_linf": float(
            np.max(
                np.abs(
                    np.quantile(candidate_slopes, quantiles)
                    - np.quantile(reference_slopes, quantiles)
                )
            )
        ),
        "reference_changed_bbox_yx": reference_bbox,
        "candidate_changed_bbox_yx": candidate_bbox,
        "changed_bbox_equal": candidate_bbox == reference_bbox,
    }
    comparison["acceptance"] = (
        comparison["height_linf_m"] <= 2.0e-8
        and abs(comparison["volume_after_error_m3"]) <= 1.0e-8
        and comparison["slope_quantile_linf"] <= 2.0e-8
        and comparison["changed_bbox_equal"]
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tile-size", type=int, default=32)
    args = parser.parse_args()
    grid = TerrainGrid(701, 701, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    config = SolverConfig(
        critical_angle_deg=30.0,
        max_iterations=5_000,
        large_avalanche_iteration_threshold=1_000,
        numerical_safety_max_iterations=20_000,
        tolerance=1.0e-8,
        boundary_condition="closed",
        conservation_tolerance_m3=1.0e-7,
    )
    cases = []
    for build_case in (_case_toe, _case_whole):
        baseline, disturbed, case_metadata = build_case(grid)
        records = []
        results = {}
        solvers = (
            ("FULL_DOMAIN_REFERENCE", MinimumSlopeAdapter(grid, config)),
            (
                "REACHABLE_BBOX_CPU_BASELINE",
                EventDrivenMinimumSlopeAdapter(
                    grid, config, tile_size=args.tile_size
                ),
            ),
            (
                "COMPACT_TILE_FRONTIER_CPU",
                SparseTileFrontierMinimumSlopeAdapter(
                    grid, config, tile_size=args.tile_size
                ),
            ),
        )
        for name, solver in solvers:
            record, result = _run_solver(
                name, solver, disturbed, baseline, grid
            )
            records.append(record)
            results[name] = result
        reference = results["FULL_DOMAIN_REFERENCE"]
        comparisons = {
            name: _comparison(reference, result, disturbed, grid)
            for name, result in results.items()
            if name != "FULL_DOMAIN_REFERENCE"
        }
        cases.append(
            {
                **case_metadata,
                "seed_cell_count": int(np.count_nonzero(disturbed != baseline)),
                "backends": records,
                "reference_comparisons": comparisons,
            }
        )
    all_passed = all(
        comparison["acceptance"]
        for case in cases
        for comparison in case["reference_comparisons"].values()
    )
    report = {
        "schema": "MINISLOPE_SPARSE_701_BENCHMARK_V1",
        "grid_shape_yx": list(grid.shape),
        "grid_spacing_m": grid.dx,
        "tile_size": args.tile_size,
        "classification_threshold": config.large_avalanche_iteration_threshold,
        "numerical_safety_limit": config.numerical_safety_max_iterations,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        },
        "cases": cases,
        "acceptance": "PASS" if all_passed else "FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["acceptance"],
                "output": str(args.output),
            }
        )
    )
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
