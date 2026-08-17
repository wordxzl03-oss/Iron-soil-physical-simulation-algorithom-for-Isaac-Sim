"""Resolution/tool-size scalability benchmark using production pure modules."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..bulk_interaction import MobileLayerConfig, MobileLayerSolver
from ..bulk_state import MaterialScenario, TerrainVolumeIntegrator
from ..planning import AttackCandidateGenerator, PlannerTerrainView, TerrainCostConfig
from ..terrain import TerrainGrid


@dataclass(frozen=True)
class BenchmarkCase:
    resolution: int
    tool_size: str
    tool_width_m: float


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    module_samples_ms: dict[str, tuple[float, ...]]
    active_bbox_grid: tuple[int, int, int, int]
    mobile_balance_error_m3: float
    candidate_count: int

    def statistics(self) -> dict[str, dict[str, float]]:
        return {
            name: {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "max_ms": float(np.max(values)), "sample_count": len(values)}
            for name, values in self.module_samples_ms.items()
        }


class OfflineScalabilityBenchmark:
    TOOL_WIDTHS = {"small": 1.6, "medium": 2.7, "large": 4.0}

    def __init__(self, repeats: int = 3) -> None:
        if repeats < 1:
            raise ValueError("[Benchmark] repeats must be positive")
        self.repeats = int(repeats)

    def run_case(self, resolution: int, tool_size: str) -> BenchmarkResult:
        if resolution not in {128, 256, 512, 701} or tool_size not in self.TOOL_WIDTHS:
            raise ValueError("[Benchmark] unsupported resolution/tool size")
        width = self.TOOL_WIDTHS[tool_size]; span = 35.0; spacing = span / (resolution - 1)
        grid = TerrainGrid(resolution, resolution, spacing, spacing, -span / 2, -span / 2, "/World/Terrain")
        coords = np.linspace(-span / 2, span / 2, resolution); xx, yy = np.meshgrid(coords, coords)
        H = 3.6 * np.exp(-(xx**2 / 55.0 + yy**2 / 38.0)) * (1.0 + 0.07 * np.sin(0.7 * xx) * np.cos(0.5 * yy))
        mobile = np.zeros_like(H); momentum = np.zeros(H.shape + (2,))
        radius = max(width * 0.35, spacing); active = xx**2 + yy**2 < radius**2
        mobile[active] = 0.04; momentum[active, 0] = mobile[active] * 0.7
        material = MaterialScenario("benchmark_uncalibrated", 2100.0, 34.0, 1000.0, 0.45, 36.0, 30.0, 0.55)
        integrator = TerrainVolumeIntegrator.from_grid(grid)
        solver = MobileLayerSolver(MobileLayerConfig(active_buffer_cells=4))
        samples = {"planner_view": [], "attack_candidates": [], "mobile_layer": []}
        bbox = (0, 0, 0, 0); balance = 0.0; count = 0
        for _ in range(self.repeats):
            start = perf_counter(); view = PlannerTerrainView.derive(grid, H, TerrainCostConfig(maximum_slope_deg=55.0)); samples["planner_view"].append((perf_counter() - start) * 1000)
            start = perf_counter(); candidates = AttackCandidateGenerator(spacing_m=max(width * 0.5, spacing), pre_dig_distance_m=4.0).generate(view, H); samples["attack_candidates"].append((perf_counter() - start) * 1000); count = len(candidates)
            start = perf_counter(); result = solver.step(H, mobile, momentum, material, grid, integrator, 0.01); samples["mobile_layer"].append((perf_counter() - start) * 1000)
            bbox = result.active_bbox_grid; balance = result.volume_before_m3 - result.volume_after_m3 - result.outflow_volume_m3
        return BenchmarkResult(BenchmarkCase(resolution, tool_size, width), {name: tuple(values) for name, values in samples.items()}, bbox, float(balance), count)

    def run_matrix(self) -> tuple[BenchmarkResult, ...]:
        return tuple(self.run_case(resolution, size) for resolution in (128, 256, 512, 701) for size in ("small", "medium", "large"))
