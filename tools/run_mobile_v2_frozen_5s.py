#!/usr/bin/env python3
"""One combined 0.5/2/5 s DEVICE validation of standalone Mobile V2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (warp_root / "warp").is_dir():
        sys.path.insert(0, str(warp_root))
        break

from isaac_bulk_pipeline.experimental import (  # noqa: E402
    MobileV2Config, MobileV2ReferenceSolver, MobileV2State,
    WarpMobileV2ReferenceSolver,
)


BOUNDARY = ROOT / "outputs/mobile_large_avalanche_causal_audit/strongest_operator_boundary.npz"
OUT = ROOT / "outputs/mobile_v2_reference/mobile_v2_frozen_5s.json"


def _jumps(field: np.ndarray) -> np.ndarray:
    return np.concatenate((np.abs(np.diff(field, axis=0)).ravel(), np.abs(np.diff(field, axis=1)).ravel()))


def _envelope(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = [field]
    left = np.empty_like(field); left[:, 0] = field[:, 0]; left[:, 1:] = field[:, :-1]
    right = np.empty_like(field); right[:, -1] = field[:, -1]; right[:, :-1] = field[:, 1:]
    down = np.empty_like(field); down[0] = field[0]; down[1:] = field[:-1]
    up = np.empty_like(field); up[-1] = field[-1]; up[:-1] = field[1:]
    values += [left, right, down, up]
    stack = np.stack(values)
    return np.max(stack, axis=0), np.min(stack, axis=0)


def _metrics(
    reference: MobileV2ReferenceSolver,
    state: MobileV2State,
    initial_mass: float,
    envelope_max: np.ndarray,
    envelope_min: np.ndarray,
    step: object | None,
) -> dict[str, object]:
    H = state.H_free_m
    jumps = _jumps(H)
    h = state.h_m
    speed = np.divide(
        np.linalg.norm(state.q_m2_s, axis=-1), h,
        out=np.zeros_like(h), where=h > reference.config.dry_tolerance_m,
    )
    maxima = H > envelope_max + 5.0e-14
    minima = H < envelope_min - 5.0e-14
    k, g, internal, total = reference.energy_terms(state)
    return {
        "mass_m3": reference.mass(state),
        "mass_error_m3": reference.mass(state) - initial_mass,
        "minimum_h_m": float(np.min(h, initial=0.0)),
        "H_free_jump_max_m": float(np.max(jumps, initial=0.0)),
        "H_free_jump_p99_m": float(np.percentile(jumps, 99.0)),
        "mobile_volume_m3": reference.mass(state),
        "moving_mobile_volume_m3": float(np.sum(
            h[speed > 0.01], dtype=np.float64
        ) * reference.config.dx_m * reference.config.dy_m),
        "mobile_speed_max_m_s": float(np.max(speed, initial=0.0)),
        "mobile_speed_p99_m_s": float(np.percentile(speed[h > reference.config.dry_tolerance_m], 99.0)),
        "energy_j_per_density": {
            "kinetic": k, "gravitational": g, "internal_pressure": internal,
            "total": total,
            "chunk_numerical_residual": (
                0.0 if step is None else step.numerical_energy_residual_j_per_density
            ),
        },
        "new_extrema": {
            "count": int(np.count_nonzero(maxima | minima)),
            "max_count": int(np.count_nonzero(maxima)),
            "min_count": int(np.count_nonzero(minima)),
            "above_envelope_volume_m3": float(np.sum(
                np.maximum(H - envelope_max, 0.0), dtype=np.float64
            ) * reference.config.dx_m * reference.config.dy_m),
            "below_envelope_volume_m3": float(np.sum(
                np.maximum(envelope_min - H, 0.0), dtype=np.float64
            ) * reference.config.dx_m * reference.config.dy_m),
        },
    }


def main() -> None:
    with np.load(BOUNDARY, allow_pickle=False) as data:
        old_resting_label = np.asarray(data["PRE_resting"], dtype=np.float64)
        h = np.asarray(data["PRE_mobile"], dtype=np.float64)
        q = np.stack((data["PRE_momentum_x"], data["PRE_momentum_y"]), axis=-1)
    H_free = old_resting_label + h
    # Explicit compatibility boundary. Algebraically this snapshot adapter is
    # H_free-h; semantically it remains uncalibrated because the old checkpoint
    # did not carry an independent physical support/interface history.
    b_eff = H_free - h
    state = MobileV2State(b_eff, h, q)
    config = MobileV2Config()
    reference = MobileV2ReferenceSolver(config)
    device = WarpMobileV2ReferenceSolver(config)
    initial_mass = reference.mass(state)
    envelope_max, envelope_min = _envelope(H_free)
    initial = _metrics(reference, state, initial_mass, envelope_max, envelope_min, None)
    samples = {}
    energy_residuals = []
    elapsed = 0.0
    wall = perf_counter()
    for horizon in (0.5, 2.0, 5.0):
        result = device.step(state, horizon - elapsed)
        state = result.state
        elapsed = horizon
        energy_residuals.append(result.numerical_energy_residual_j_per_density)
        key = str(horizon).replace(".", "p") + "s"
        samples[key] = _metrics(
            reference, state, initial_mass, envelope_max, envelope_min, result
        )
        print(
            f"MOBILE_V2_FROZEN HORIZON={horizon:.1f}s "
            f"JMAX={samples[key]['H_free_jump_max_m']:.9g} "
            f"ENERGY={samples[key]['energy_j_per_density']['total']:.9g}",
            flush=True,
        )
    mass_max = max(abs(float(x["mass_error_m3"])) for x in samples.values())
    positivity = min(float(x["minimum_h_m"]) for x in samples.values()) >= -1.0e-13
    energy_tolerance = max(1.0e-10, 1.0e-8 * abs(float(initial["energy_j_per_density"]["total"])))
    energy_ok = all(value <= energy_tolerance for value in energy_residuals)
    status = "PASS" if mass_max <= 2.0e-9 and positivity and energy_ok else "FAIL"
    report = {
        "schema": "MOBILE_V2_REAL_FROZEN_5S/v1",
        "status": status,
        "source_boundary": str(BOUNDARY.relative_to(ROOT)),
        "source_boundary_physical_time_s": 3.75,
        "one_combined_continuous_run": True,
        "isaac_or_gui_launched": False,
        "support_adapter": {
            "mapping": "b_eff := frozen H_free - frozen h_mobile",
            "classification": "UNCALIBRATED_SNAPSHOT_COMPATIBILITY_ADAPTER",
            "claim_H_resting_is_physical_support": False,
            "limitation": "checkpoint lacks independent b_eff history across prior entrainment/deposition/relabel events",
        },
        "initial": initial,
        "horizons": samples,
        "mass_error_max_abs_m3": mass_max,
        "positivity": "PASS" if positivity else "FAIL",
        "energy_consistency": "PASS" if energy_ok else "FAIL",
        "energy_residuals_j_per_density": energy_residuals,
        "energy_tolerance_j_per_density": energy_tolerance,
        "wall_time_s": perf_counter() - wall,
        "production_modified": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": status, "mass_error_max_abs_m3": mass_max, "positivity": positivity, "energy_consistency": energy_ok, "output": str(OUT)}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
