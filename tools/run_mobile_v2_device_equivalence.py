#!/usr/bin/env python3
"""CPU/Warp equivalence for identical standalone Mobile V2 equations."""

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

from isaac_bulk_pipeline.experimental import (  # noqa: E402
    MobileV2Config,
    MobileV2ReferenceSolver,
    MobileV2State,
    WarpMobileV2ReferenceSolver,
)


OUT = ROOT / "outputs/mobile_v2_reference/mobile_v2_cpu_device_equivalence.json"


def main() -> None:
    config = MobileV2Config()
    cpu = MobileV2ReferenceSolver(config)
    gpu = WarpMobileV2ReferenceSolver(config)
    shape = (41, 41)
    y, x = np.indices(shape)
    cases: dict[str, tuple[MobileV2State, float]] = {}
    cases["STATIC_FLAT"] = (
        MobileV2State(np.ones(shape), np.full(shape, 0.1), np.zeros(shape + (2,))),
        0.05,
    )
    free = 2.0 - np.tan(np.deg2rad(10.0)) * x * config.dx_m
    cases["STATIC_SLOPE"] = (
        MobileV2State(free - 0.08, np.full(shape, 0.08), np.zeros(shape + (2,))),
        0.05,
    )
    H = np.full(shape, 0.8266754619); H[:, 19] = 1.82; H[:, 20] = 0.5144415194
    b = np.full(shape, 0.12); h = np.maximum(H - b, 0.0)
    v = np.zeros(shape + (2,)); v[:, 20, 0] = 0.8
    cases["VALLEY"] = (MobileV2State(b, h, h[..., None] * v), 1.0 / 240.0)
    mound = 0.03 + 0.30 * np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 18.0)
    cases["MOUND"] = (MobileV2State(np.full(shape, 0.12), mound, np.zeros(shape + (2,))), 0.02)
    wet = np.zeros(shape); wet[12:29, 12:29] = 0.12
    q = np.zeros(shape + (2,)); q[..., 0] = wet * 0.5
    cases["WET_DRY"] = (MobileV2State(np.full(shape, 0.12), wet, q), 0.04)
    packet = 0.20 * np.exp(-((x - 13) ** 2 + (y - 20) ** 2) / 30.0)
    q = np.zeros(shape + (2,)); q[..., 0] = packet * 0.8
    cases["SMOOTH_PACKET"] = (
        MobileV2State(1.5 - 0.08 * x * config.dx_m, packet, q), 0.04
    )
    records = {}
    for name, (state, dt) in cases.items():
        c = cpu.step(state, dt)
        d = gpu.step(state, dt)
        h_error = float(np.max(np.abs(c.state.h_m - d.state.h_m)))
        q_error = float(np.max(np.abs(c.state.q_m2_s - d.state.q_m2_s)))
        energy_error = abs(c.energy_after_j_per_density - d.energy_after_j_per_density)
        status = (
            "PASS" if h_error <= 2.0e-13 and q_error <= 2.0e-13
            and energy_error <= 2.0e-12
            and d.numerical_energy_residual_j_per_density
            <= max(1.0e-11, 1.0e-9 * abs(d.energy_before_j_per_density))
            else "FAIL"
        )
        records[name] = {
            "status": status,
            "h_max_abs_error_m": h_error,
            "q_max_abs_error_m2_s": q_error,
            "energy_after_abs_error_j_per_density": energy_error,
            "CPU_substeps": c.substeps,
            "DEVICE_substeps": d.substeps,
            "DEVICE_energy_residual_j_per_density": d.numerical_energy_residual_j_per_density,
        }
    status = "PASS" if all(x["status"] == "PASS" for x in records.values()) else "FAIL"
    report = {
        "schema": "MOBILE_V2_CPU_DEVICE_EQUIVALENCE/v1",
        "status": status,
        "equations": "IDENTICAL_HYDROSTATIC_RUSANOV_FACE_FLUX_AND_COULOMB_SOURCE",
        "production_modified": False,
        "cases": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
