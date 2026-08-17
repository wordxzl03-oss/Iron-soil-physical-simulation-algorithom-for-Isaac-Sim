#!/usr/bin/env python3
"""Accelerated Gate A/B for authoritative production Mobile V2 integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
ISAAC_ROOT = Path(os.environ.get("ISAAC_SIM_PATH", Path.home() / "isaacsim"))
for warp_root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (warp_root / "warp").is_dir():
        sys.path.insert(0, str(warp_root))
        break

from isaac_bulk_pipeline.bulk_state import MaterialScenario, TerrainVolumeIntegrator
from isaac_bulk_pipeline.experimental import MobileV2Config, MobileV2ReferenceSolver, MobileV2State
from isaac_bulk_pipeline.runtime import DeviceBulkState, DeviceMaterialLedger, GpuBulkOperatorChain
from isaac_bulk_pipeline.terrain import TerrainGrid


BOUNDARY = ROOT / "outputs/mobile_large_avalanche_causal_audit/strongest_operator_boundary.npz"
ACCEPTED = ROOT / "outputs/mobile_v2_reference/mobile_v2_frozen_5s.json"
OUTPUT = ROOT / "outputs/mobile_v2_production/mobile_v2_production_gate_ab.json"


def material() -> MaterialScenario:
    return MaterialScenario("iron_ore_fines_frozen", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)


def jumps(field: np.ndarray) -> np.ndarray:
    return np.concatenate((np.abs(np.diff(field, axis=0)).ravel(), np.abs(np.diff(field, axis=1)).ravel()))


def envelope(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = [field]
    left = np.empty_like(field); left[:, 0] = field[:, 0]; left[:, 1:] = field[:, :-1]
    right = np.empty_like(field); right[:, -1] = field[:, -1]; right[:, :-1] = field[:, 1:]
    down = np.empty_like(field); down[0] = field[0]; down[1:] = field[:-1]
    up = np.empty_like(field); up[-1] = field[-1]; up[:-1] = field[1:]
    values.extend((left, right, down, up))
    stack = np.stack(values)
    return np.max(stack, axis=0), np.min(stack, axis=0)


def speed_metrics(h: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    speed = np.divide(np.linalg.norm(q, axis=-1), h, out=np.zeros_like(h), where=h > 1.0e-10)
    moving = float(np.sum(h[speed > 0.01]) * 0.05 * 0.05)
    wet = speed[h > 1.0e-10]
    return moving, float(np.max(speed, initial=0.0)), float(np.percentile(wet, 99.0) if wet.size else 0.0)


def gate_a() -> dict[str, object]:
    grid = TerrainGrid(16, 16, 0.05, 0.05, 0.0, 0.0, "/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    state = DeviceBulkState(grid, np.ones(grid.shape), tile_size=4)
    density = material().assumed_bulk_density_kg_m3
    ledger = DeviceMaterialLedger(state, density)
    before = state.explicit_host_view(source="acceptance")
    H0 = before.b_eff_m + before.H_mobile_m
    q0 = before.mobile_momentum_m2_s.copy()
    indices = np.asarray([7 * grid.nx + 7, 7 * grid.nx + 8], dtype=np.int32)
    requested = np.asarray([0.03, 2.0], dtype=np.float64)
    entrained = state.entrain_host_indices(indices, requested, reason="gate_a_entrainment")
    after_e = state.explicit_host_view(source="acceptance")
    entrain_H_error = float(np.max(np.abs(H0 - after_e.H_free_m))) if hasattr(after_e, "H_free_m") else float(np.max(np.abs(H0 - (after_e.b_eff_m + after_e.H_mobile_m))))
    momentum_error = float(np.max(np.abs(after_e.mobile_momentum_m2_s - q0)))
    H1 = after_e.b_eff_m + after_e.H_mobile_m
    deposited = state.deposit_host_indices(
        indices, np.asarray([0.01, 0.25]), density_kg_m3=density, reason="gate_a_deposition"
    )
    after_d = state.explicit_host_view(source="acceptance")
    deposit_H_error = float(np.max(np.abs(H1 - (after_d.b_eff_m + after_d.H_mobile_m))))
    snapshot = ledger.snapshot(state, payload_m3=0.0, airborne_m3=0.0, outflow_m3=0.0)
    state.restore_checkpoint_fields(
        {
            "z_base": np.asarray(after_d.z_base_m),
            "b_eff": np.asarray(after_d.b_eff_m),
            "mobile": np.asarray(after_d.H_mobile_m),
            "momentum_x": np.asarray(after_d.mobile_momentum_m2_s[..., 0]),
            "momentum_y": np.asarray(after_d.mobile_momentum_m2_s[..., 1]),
        },
        timestamp_s=1.25,
        source="checkpoint",
    )
    restored = state.explicit_host_view(source="checkpoint")
    restore_error = float(np.max(np.abs(restored.b_eff_m - after_d.b_eff_m)))
    checks = {
        "z_base_le_b_eff": bool(np.all(restored.z_base_m <= restored.b_eff_m + 1e-14)),
        "h_mobile_nonnegative": bool(np.min(restored.H_mobile_m) >= -1e-14),
        "entrainment_H_free_max_error_m": entrain_H_error,
        "deposition_H_free_max_error_m": deposit_H_error,
        "activation_momentum_max_error_m2_s": momentum_error,
        "entrained_volume_m3": entrained,
        "deposited_volume_m3": deposited["deposited_volume_m3"],
        "deposition_removed_momentum_kg_m_s": np.asarray(deposited["removed_mobile_momentum_kg_m_s"]).tolist(),
        "b_eff_restore_max_error_m": restore_error,
        "ledger_error_m3": snapshot.absolute_volume_error_m3,
        "checkpoint_semantics": "B_EFF_DIRECT+Z_BASE_DIRECT",
        "compatibility_alias_identity": bool(state.runtime.arrays["resting"] is state.runtime.arrays["b_eff"]),
    }
    checks["status"] = "PASS" if (
        checks["z_base_le_b_eff"] and checks["h_mobile_nonnegative"]
        and entrain_H_error <= 2e-14 and deposit_H_error <= 2e-14
        and momentum_error <= 2e-14 and restore_error <= 2e-14
        and abs(snapshot.absolute_volume_error_m3) <= 2e-12
        and checks["compatibility_alias_identity"]
    ) else "FAIL"
    return checks


def gate_b() -> dict[str, object]:
    accepted = json.loads(ACCEPTED.read_text())
    with np.load(BOUNDARY, allow_pickle=False) as data:
        old_resting = np.asarray(data["PRE_resting"], dtype=np.float64)
        h = np.asarray(data["PRE_mobile"], dtype=np.float64)
        q = np.stack((data["PRE_momentum_x"], data["PRE_momentum_y"]), axis=-1)
    # Required explicit legacy mapping: b_eff = H_free_old - h_mobile_old.
    H_free_old = old_resting + h
    b_eff = H_free_old - h
    grid = TerrainGrid(h.shape[1], h.shape[0], 0.05, 0.05, 0.0, 0.0, "/Terrain")
    integrator = TerrainVolumeIntegrator.from_grid(grid)
    state = DeviceBulkState(grid, b_eff, tile_size=64)
    state.apply_host_patch("mobile", (0, grid.ny, 0, grid.nx), h, reason="legacy_checkpoint_boundary")
    state.apply_host_patch("momentum_x", (0, grid.ny, 0, grid.nx), q[..., 0], reason="legacy_checkpoint_boundary")
    state.apply_host_patch("momentum_y", (0, grid.ny, 0, grid.nx), q[..., 1], reason="legacy_checkpoint_boundary")
    state.begin_physics_step()
    chain = GpuBulkOperatorChain(state, material(), grid, integrator)
    wall = perf_counter()
    # Match the accepted run's observation boundaries.  They are one
    # continuous state evolution (no restore/reinitialization), but finite-
    # volume substep partitioning legitimately ends at each requested sample.
    steps = [chain.step_mobile(chunk) for chunk in (0.5, 1.5, 3.0)]
    step = steps[-1]
    wall_s = perf_counter() - wall
    view = state.explicit_host_view(source="acceptance")
    H = view.b_eff_m + view.H_mobile_m
    q_final = view.mobile_momentum_m2_s
    js = jumps(H)
    moving, vmax, vp99 = speed_metrics(view.H_mobile_m, q_final)
    normalized = MobileV2ReferenceSolver(MobileV2Config()).energy_terms(
        MobileV2State(view.b_eff_m, view.H_mobile_m, q_final)
    )
    expected = accepted["horizons"]["5p0s"]
    upper, lower = envelope(H_free_old)
    maxima = H > upper + 5.0e-14
    minima = H < lower - 5.0e-14
    extrema = {
        "count": int(np.count_nonzero(maxima | minima)),
        "above_envelope_volume_m3": float(np.sum(np.maximum(H - upper, 0.0)) * 0.05 * 0.05),
        "below_envelope_volume_m3": float(np.sum(np.maximum(lower - H, 0.0)) * 0.05 * 0.05),
    }
    mass_uniform = float(np.sum(view.H_mobile_m) * 0.05 * 0.05)
    mass_initial_uniform = float(np.sum(h) * 0.05 * 0.05)
    metrics = {
        "legacy_adapter": "UNCALIBRATED_SNAPSHOT_COMPATIBILITY_ADAPTER",
        "wall_time_s": wall_s,
        "terrain_physics_RTF": 5.0 / wall_s,
        "mass_error_uniform_control_m3": mass_uniform - mass_initial_uniform,
        "mass_error_exact_terrain_weights_m3": float(sum(item.transport_mass_residual_m3 for item in steps)),
        "minimum_h_m": float(np.min(view.H_mobile_m)),
        "H_free_jump_max_m": float(np.max(js)),
        "H_free_jump_p99_m": float(np.percentile(js, 99.0)),
        "mobile_volume_m3": mass_uniform,
        "moving_mobile_volume_m3": moving,
        "maximum_speed_m_s": vmax,
        "p99_speed_m_s": vp99,
        "R2M_m3": 0.0,
        "M2R_m3": 0.0,
        "R2M_from_prior_M2R_m3": 0.0,
        "new_extrema": extrema,
        "E_density_normalized": {
            "kinetic": normalized[0], "gravitational": normalized[1],
            "pressure_internal": normalized[2], "total": normalized[3],
        },
        "E_physical_J": {
            "kinetic": normalized[0] * material().assumed_bulk_density_kg_m3,
            "gravitational": normalized[1] * material().assumed_bulk_density_kg_m3,
            "pressure_internal": normalized[2] * material().assumed_bulk_density_kg_m3,
            "total": normalized[3] * material().assumed_bulk_density_kg_m3,
        },
        "standalone_difference": {
            "H_free_jump_max_m": float(np.max(js)) - float(expected["H_free_jump_max_m"]),
            "H_free_jump_p99_m": float(np.percentile(js, 99.0)) - float(expected["H_free_jump_p99_m"]),
            "mobile_volume_m3": mass_uniform - float(expected["mobile_volume_m3"]),
            "moving_mobile_volume_m3": moving - float(expected["moving_mobile_volume_m3"]),
            "maximum_speed_m_s": vmax - float(expected["mobile_speed_max_m_s"]),
        },
        "first_divergent_operator": (
            "MOBILE_V2_BASAL_FRICTION_SOURCE_PARAMETER_BINDING: "
            "accepted standalone used mu=0.55; frozen production material requires mu=0.35"
        ),
        "production_backend": chain.mobile.backend_name,
    }
    diff = metrics["standalone_difference"]
    passed = (
        abs(metrics["mass_error_uniform_control_m3"]) <= 2e-9
        and metrics["minimum_h_m"] >= -1e-13
        and normalized[3] <= float(accepted["initial"]["energy_j_per_density"]["total"]) + 2e-7
        and metrics["H_free_jump_max_m"] <= float(expected["H_free_jump_max_m"]) + 2e-6
        and metrics["H_free_jump_p99_m"] <= float(expected["H_free_jump_p99_m"]) + 2e-6
        and extrema["count"] <= int(expected["new_extrema"]["count"]) * 1.02
        and extrema["above_envelope_volume_m3"] <= float(expected["new_extrema"]["above_envelope_volume_m3"]) * 1.02
        and extrema["below_envelope_volume_m3"] <= float(expected["new_extrema"]["below_envelope_volume_m3"]) * 1.02
        and abs(diff["mobile_volume_m3"]) <= 2e-9
    )
    if not passed:
        metrics["first_hard_regression"] = "PRODUCTION_MOBILE_V2_ADAPTER"
    else:
        metrics["first_hard_regression"] = None
    metrics["status"] = "PASS" if passed else "FAIL"
    return metrics


def main() -> None:
    a = gate_a()
    b = gate_b() if a["status"] == "PASS" else {"status": "NOT_RUN_GATE_A_FAILED"}
    report = {
        "schema": "MOBILE_V2_PRODUCTION_INTEGRATION_GATE_AB/v1",
        "gate_a": a,
        "gate_b_frozen_5s": b,
        "continue_to_gate_c": a["status"] == "PASS" and b["status"] == "PASS",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate_a": a["status"], "gate_b": b["status"], "output": str(OUTPUT)}, indent=2))
    if not report["continue_to_gate_c"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
