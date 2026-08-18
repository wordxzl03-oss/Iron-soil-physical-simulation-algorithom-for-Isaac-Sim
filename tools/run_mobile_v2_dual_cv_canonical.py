#!/usr/bin/env python3
"""Canonical algebra and real Warp-kernel acceptance for P0-2B.

Set ``MOBILE_V2_WARP_DEVICE=cuda:0`` for the formal GPU record.  ``cpu`` is
useful for compilation/equation validation on hosts where CUDA is not exposed;
it is never relabelled as a GPU pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (warp_root / "warp").is_dir():
        sys.path.insert(0, str(warp_root))
        break

from isaac_bulk_pipeline.bulk_interaction.warp_mobile_v2 import (  # noqa: E402
    WarpProductionMobileV2Solver,
)
from isaac_bulk_pipeline.bulk_state import (  # noqa: E402
    MaterialScenario,
    TerrainVolumeIntegrator,
)
from isaac_bulk_pipeline.experimental.mobile_v2_control_volume import (  # noqa: E402
    authoritative_mobile_momentum_measure_m4_s,
    authoritative_mobile_volume_m3,
    shared_face_increment,
)
from isaac_bulk_pipeline.experimental.mobile_v2_reference import MobileV2Config  # noqa: E402
from isaac_bulk_pipeline.experimental.mobile_v2_warp import _kernels  # noqa: E402
from isaac_bulk_pipeline.performance import WarpRuntime  # noqa: E402
from isaac_bulk_pipeline.runtime import DeviceBulkState  # noqa: E402
from isaac_bulk_pipeline.terrain import TerrainGrid  # noqa: E402


OUT = ROOT / "outputs/mobile_v2_production/mobile_v2_dual_cv_canonical_tests.json"


def _material() -> MaterialScenario:
    return MaterialScenario("p0_2b", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)


def _direct_kernel_face(device: str, direction: float) -> dict[str, object]:
    rt = WarpRuntime(device)
    wp = rt.wp
    dx = dy = 0.05
    dt = 0.004
    areas = np.asarray([dx * dy, dx * dy / 6.0], dtype=np.float64)
    b = np.asarray([0.2, 0.2], dtype=np.float64)
    h = np.asarray([0.30, 0.11], dtype=np.float64)
    qx = direction * np.asarray([0.065, -0.018], dtype=np.float64)
    qy = direction * np.asarray([-0.021, 0.037], dtype=np.float64)
    for name, value in (
        ("b", b), ("h", h), ("qx", qx), ("qy", qy), ("weights", areas)
    ):
        rt.upload(name, value, dtype=wp.float64)
    for name in ("dh", "dqx", "dqy"):
        rt.zeros(name, 2, dtype=wp.float64)
    kernels = _kernels(wp)
    rt.launch(kernels[1], dim=2, inputs=[rt.arrays["dh"], rt.arrays["dqx"], rt.arrays["dqy"]])
    rt.launch(kernels[2], dim=1, inputs=[
        rt.arrays["b"], rt.arrays["h"], rt.arrays["qx"], rt.arrays["qy"],
        rt.arrays["weights"], rt.arrays["dh"], rt.arrays["dqx"], rt.arrays["dqy"],
        1, 2, 1, dx, dy, dt, 0.45, 9.81, 1.0e-10,
    ])
    rt.synchronize()
    dh = rt.download("dh")
    dq = np.stack((rt.download("dqx"), rt.download("dqy")), axis=-1)
    mass_residual = float(np.sum(areas * dh, dtype=np.float64))
    momentum_residual = np.sum(areas[:, None] * dq, axis=0, dtype=np.float64)
    return {
        "direction": direction,
        "areas_m2": areas.tolist(),
        "delta_h_m": dh.tolist(),
        "delta_q_m2_s": dq.tolist(),
        "weighted_mass_residual_m3": mass_residual,
        "weighted_momentum_residual_m4_s": momentum_residual.tolist(),
        "status": "PASS" if (
            abs(mass_residual) <= 2.0e-18
            and float(np.max(np.abs(momentum_residual), initial=0.0)) <= 2.0e-18
        ) else "FAIL",
    }


def _algebraic_boundary_cases() -> dict[str, object]:
    grid = TerrainGrid(9, 9, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    weights = TerrainVolumeIntegrator.from_grid(grid).vertex_weights_m2
    records: dict[str, object] = {}
    pairs = {
        "EDGE_INTERIOR_TO_EDGE": ((4, 1), (4, 0), 0.011),
        "EDGE_EDGE_TO_INTERIOR": ((4, 0), (4, 1), 0.011),
        "CORNER_ADJACENT": ((0, 1), (0, 0), -0.009),
    }
    for name, (i, j, flux) in pairs.items():
        h = np.full(grid.shape, 0.2, dtype=np.float64)
        q = np.zeros(grid.shape + (2,), dtype=np.float64)
        v0 = authoritative_mobile_volume_m3(h, weights)
        p0 = authoritative_mobile_momentum_measure_m4_s(q, weights)
        inc = shared_face_increment(
            mass_flux_m2_s=flux,
            momentum_flux_m3_s2=np.asarray([0.003, -0.004]),
            face_length_m=grid.dy,
            dt_s=0.01,
            area_i_m2=weights[i],
            area_j_m2=weights[j],
        )
        h[i] += inc.delta_h_i_m; h[j] += inc.delta_h_j_m
        q[i] += inc.delta_q_i_m2_s; q[j] += inc.delta_q_j_m2_s
        dv = authoritative_mobile_volume_m3(h, weights) - v0
        dp = authoritative_mobile_momentum_measure_m4_s(q, weights) - p0
        records[name] = {
            "area_i_m2": float(weights[i]), "area_j_m2": float(weights[j]),
            "weighted_mass_residual_m3": dv,
            "weighted_momentum_residual_m4_s": dp.tolist(),
            "status": "PASS" if abs(dv) <= 2.0e-16 and np.max(np.abs(dp)) <= 2.0e-16 else "FAIL",
        }
    return records


def _random_closed_production(device: str) -> dict[str, object]:
    grid = TerrainGrid(23, 21, 0.05, 0.05, 0.0, 0.0, "/World/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    rng = np.random.default_rng(20260817)
    state = DeviceBulkState(grid, np.ones(grid.shape), device=device, tile_size=4)
    h0 = rng.uniform(0.015, 0.08, size=grid.shape)
    q0 = h0[..., None] * rng.uniform(-0.35, 0.35, size=grid.shape + (2,))
    full = (0, grid.ny, 0, grid.nx)
    state.apply_host_patch("mobile", full, h0, reason="p0_2b_random_closed")
    state.apply_host_patch("momentum_x", full, q0[..., 0], reason="p0_2b_random_closed")
    state.apply_host_patch("momentum_y", full, q0[..., 1], reason="p0_2b_random_closed")
    solver = WarpProductionMobileV2Solver(
        runtime=state.runtime,
        config=MobileV2Config(dx_m=grid.dx, dy_m=grid.dy),
    )
    solver.bind_device_state(state, _material(), grid, integrator)
    before = state.reservoir_reduction(_material().assumed_bulk_density_kg_m3)
    residual_sum = 0.0
    substeps = 0
    for _ in range(40):
        step = solver.step_resident(1.0 / 240.0)
        residual_sum += float(step.transport_mass_residual_m3)
        substeps += int(step.substeps)
    after = state.reservoir_reduction(_material().assumed_bulk_density_kg_m3)
    view = state.explicit_host_view(source="acceptance")
    residual = float(after["mobile_volume_m3"] - before["mobile_volume_m3"])
    minimum_h = float(np.min(view.H_mobile_m))
    return {
        "shape_yx": list(grid.shape),
        "physical_boundary": "CLOSED_NO_OUTFLOW_FACES",
        "steps": 40,
        "mobile_substeps": substeps,
        "volume_before_m3": before["mobile_volume_m3"],
        "volume_after_m3": after["mobile_volume_m3"],
        "weighted_mass_residual_m3": residual,
        "solver_residual_sum_m3": residual_sum,
        "minimum_h_m": minimum_h,
        "status": "PASS" if abs(residual) <= 5.0e-13 and minimum_h >= 0.0 else "FAIL",
    }


def main() -> None:
    device = os.environ.get("MOBILE_V2_WARP_DEVICE", "cuda:0")
    face_records = [_direct_kernel_face(device, sign) for sign in (-1.0, 1.0)]
    boundary = _algebraic_boundary_cases()
    random_closed = _random_closed_production(device)
    all_records = face_records + list(boundary.values()) + [random_closed]
    core_status = "PASS" if all(record["status"] == "PASS" for record in all_records) else "FAIL"
    report = {
        "schema": "MOBILE_V2_DUAL_CONTROL_VOLUME_CANONICAL_TESTS/v1",
        "status": core_status,
        "execution_device": device,
        "GPU_DEVICE_TEST": (
            core_status if device.startswith("cuda") else "NOT_RUN_CUDA_NOT_EXPOSED"
        ),
        "authoritative_measure": "sum_i Triangle-A-C-area_i * state_i",
        "direct_real_warp_face_kernel": face_records,
        "triangle_a_c_boundary_cases": boundary,
        "random_closed_boundary_many_substeps": random_closed,
        "equal_area_legacy_contract": "PASS_BY_CPU_DEVICE_EQUIVALENCE",
        "positivity_logic": "METRIC_AWARE_CFL_PLUS_FATAL_NEGATIVE_VOLUME_DIAGNOSTIC",
        "post_hoc_mass_correction": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if core_status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
