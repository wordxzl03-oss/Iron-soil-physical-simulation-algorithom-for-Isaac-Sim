#!/usr/bin/env python3
"""Fast no-Isaac contract suite for the standalone Mobile V2 CPU reference."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from isaac_bulk_pipeline.experimental import (  # noqa: E402
    MobileV2Config,
    MobileV2ReferenceSolver,
    MobileV2State,
)


OUT = ROOT / "outputs/mobile_v2_reference/mobile_v2_cpu_contracts.json"
CONFIG = MobileV2Config()
SOLVER = MobileV2ReferenceSolver(CONFIG)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _state(b: np.ndarray, h: np.ndarray, velocity: np.ndarray | None = None) -> MobileV2State:
    v = np.zeros(h.shape + (2,)) if velocity is None else np.broadcast_to(velocity, h.shape + (2,)).copy()
    return MobileV2State(b, h, h[..., None] * v)


def _contract(name: str, before: MobileV2State, dt: float = 0.05) -> tuple[dict[str, Any], Any]:
    step = SOLVER.step(before, dt)
    energy_tolerance = max(1.0e-11, 1.0e-9 * max(abs(step.energy_before_j_per_density), 1.0))
    checks = {
        "mass": abs(step.mass_residual_m3) <= 2.0e-13,
        "momentum": float(np.max(np.abs(step.momentum_accounting_residual_m4_s))) <= 2.0e-12,
        "positivity": step.minimum_h_m >= -1.0e-13,
        "finite_CFL": np.isfinite(step.maximum_cfl) and step.maximum_cfl <= CONFIG.cfl + 1.0e-12,
        "energy": step.numerical_energy_residual_j_per_density <= energy_tolerance,
        "finite": np.all(np.isfinite(step.state.h_m)) and np.all(np.isfinite(step.state.q_m2_s)),
    }
    report = {
        "name": name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mass_residual_m3": step.mass_residual_m3,
        "momentum_accounting_residual_m4_s": step.momentum_accounting_residual_m4_s.tolist(),
        "minimum_h_m": step.minimum_h_m,
        "maximum_CFL": step.maximum_cfl,
        "substeps": step.substeps,
        "energy": {
            "before_j_per_density": step.energy_before_j_per_density,
            "after_j_per_density": step.energy_after_j_per_density,
            "kinetic_before_j_per_density": step.kinetic_before_j_per_density,
            "kinetic_after_j_per_density": step.kinetic_after_j_per_density,
            "gravitational_before_j_per_density": step.gravitational_before_j_per_density,
            "gravitational_after_j_per_density": step.gravitational_after_j_per_density,
            "internal_before_j_per_density": step.internal_before_j_per_density,
            "internal_after_j_per_density": step.internal_after_j_per_density,
            "friction_dissipation_j_per_density": step.friction_dissipation_j_per_density,
            "external_work_j_per_density": step.external_work_j_per_density,
            "numerical_residual_j_per_density": step.numerical_energy_residual_j_per_density,
            "tolerance_j_per_density": energy_tolerance,
        },
    }
    return report, step


def main() -> None:
    shape = (41, 41)
    y, x = np.indices(shape)
    cases: dict[str, dict[str, Any]] = {}

    flat = _state(np.ones(shape), np.full(shape, 0.1))
    cases["STATIC_FLAT"], flat_step = _contract("STATIC_FLAT", flat)
    cases["STATIC_FLAT"]["no_spurious_motion"] = bool(np.max(np.abs(flat_step.state.q_m2_s)) == 0.0)
    cases["STATIC_FLAT"]["status"] = "PASS" if cases["STATIC_FLAT"]["status"] == "PASS" and cases["STATIC_FLAT"]["no_spurious_motion"] else "FAIL"

    free = 2.0 - np.tan(np.deg2rad(10.0)) * x * CONFIG.dx_m
    stable = _state(free - 0.08, np.full(shape, 0.08))
    cases["STATIC_SLOPE"], stable_step = _contract("STATIC_SLOPE", stable)
    cases["STATIC_SLOPE"]["no_spurious_motion"] = bool(np.max(np.abs(stable_step.state.q_m2_s)) == 0.0)
    cases["STATIC_SLOPE"]["status"] = "PASS" if cases["STATIC_SLOPE"]["status"] == "PASS" and cases["STATIC_SLOPE"]["no_spurious_motion"] else "FAIL"

    # Exact historical free-surface triplet. b_eff is independently specified
    # as the physical support adapter; it is not inferred from a relabelled
    # production H_resting field.
    H = np.full(shape, 0.8266754619)
    H[:, 19] = 1.82
    H[:, 20] = 0.5144415194
    H[:, 21] = 0.8266754619
    b = np.full(shape, 0.12)
    h = np.maximum(H - b, 0.0)
    velocity = np.zeros(shape + (2,))
    velocity[:, 20, 0] = 0.8
    valley = MobileV2State(b, h, h[..., None] * velocity)
    cases["VALLEY"], valley_step = _contract("VALLEY", valley, 1.0 / 240.0)
    # Shared forward face response: reconstruct only that face. Its contribution
    # to the low cell must oppose positive/uphill q. Rear-face physics is kept
    # separate instead of leaking through a centered stencil.
    dh = np.zeros_like(h)
    dq = np.zeros_like(velocity)
    SOLVER._face_updates(b[:, 20:22], h[:, 20:22], valley.q_m2_s[:, 20:22], 1.0 / 240.0, dh[:, 20:22], dq[:, 20:22], axis=1)
    forward_impulse = float(np.mean(dq[:, 20, 0]))
    cases["VALLEY"].update({
        "historical_triplet_H_free_m": [1.82, 0.5144415194, 0.8266754619],
        "low_to_forward_face_delta_q_m2_s": forward_impulse,
        "no_artificial_uphill_gravity": forward_impulse <= 0.0,
        "face_source_special_case": False,
    })
    cases["VALLEY"]["status"] = "PASS" if cases["VALLEY"]["status"] == "PASS" and forward_impulse <= 0.0 else "FAIL"

    mound_h = 0.03 + 0.30 * np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 18.0)
    cases["MOUND"], _ = _contract("MOUND", _state(np.full(shape, 0.12), mound_h), 0.02)

    packet_h = 0.20 * np.exp(-((x - 13) ** 2 + (y - 20) ** 2) / 30.0)
    packet_v = np.zeros(shape + (2,)); packet_v[..., 0] = 0.8
    cases["SMOOTH_PACKET"], _ = _contract("SMOOTH_PACKET", _state(1.5 - 0.08 * x * CONFIG.dx_m, packet_h, packet_v), 0.04)

    opposing_h = 0.12 * (np.exp(-((x - 12) ** 2 + (y - 20) ** 2) / 20.0) + np.exp(-((x - 28) ** 2 + (y - 20) ** 2) / 20.0))
    opposing_v = np.zeros(shape + (2,)); opposing_v[..., 0] = np.where(x < 20, 0.8, -0.8)
    cases["OPPOSING_PACKETS"], _ = _contract("OPPOSING_PACKETS", _state(np.full(shape, 0.12), opposing_h, opposing_v), 0.04)

    wet_h = np.zeros(shape); wet_h[12:29, 12:29] = 0.12
    wet_v = np.zeros(shape + (2,)); wet_v[..., 0] = 0.5
    cases["WET_DRY"], _ = _contract("WET_DRY", _state(np.full(shape, 0.12), wet_h, wet_v), 0.04)

    relabel_before = _state(np.full(shape, 0.3), np.full(shape, 0.1), np.asarray([0.2, -0.1]))
    labels = 0.04 * np.sin(x) * np.cos(y)
    relabel_after = SOLVER.bookkeeping_relabel(relabel_before, labels)
    invariant = all(np.array_equal(a, b_) for a, b_ in (
        (relabel_before.b_eff_m, relabel_after.b_eff_m),
        (relabel_before.h_m, relabel_after.h_m),
        (relabel_before.q_m2_s, relabel_after.q_m2_s),
        (relabel_before.H_free_m, relabel_after.H_free_m),
    ))
    cases["BOOKKEEPING_RELABEL"] = {
        "status": "PASS" if invariant else "FAIL",
        "b_eff_h_q_H_free_bitwise_invariant": invariant,
        "classification": "PURE_BOOKKEEPING_RELABEL",
    }

    event_state = _state(np.full(shape, 0.5), np.full(shape, 0.1), np.asarray([0.3, 0.0]))
    event_depth = np.zeros(shape); event_depth[18:23, 18:23] = 0.02
    entrained, entrainment_energy = SOLVER.physical_entrainment(event_state, event_depth, np.asarray([0.0, 0.0]))
    entrainment_ok = (
        np.max(np.abs(entrained.H_free_m - event_state.H_free_m)) <= 2.0e-16
        and abs(SOLVER.mass(entrained) - SOLVER.mass(event_state) - np.sum(event_depth) * CONFIG.dx_m * CONFIG.dy_m) <= 2.0e-15
        and np.isfinite(entrainment_energy)
    )
    cases["ENTRAINMENT"] = {
        "status": "PASS" if entrainment_ok else "FAIL",
        "classification": "PHYSICAL_ENTRAINMENT",
        "H_free_instantaneously_conserved": bool(np.max(np.abs(entrained.H_free_m - event_state.H_free_m)) <= 2.0e-16),
        "energy_transfer_j_per_density": entrainment_energy,
    }

    deposited, deposition_energy = SOLVER.physical_deposition(event_state, event_depth)
    deposition_ok = (
        np.max(np.abs(deposited.H_free_m - event_state.H_free_m)) <= 2.0e-16
        and abs(SOLVER.mass(event_state) - SOLVER.mass(deposited) - np.sum(event_depth) * CONFIG.dx_m * CONFIG.dy_m) <= 2.0e-15
        and deposition_energy <= 1.0e-14
    )
    cases["DEPOSITION"] = {
        "status": "PASS" if deposition_ok else "FAIL",
        "classification": "PHYSICAL_DEPOSITION",
        "H_free_instantaneously_conserved": bool(np.max(np.abs(deposited.H_free_m - event_state.H_free_m)) <= 2.0e-16),
        "energy_transfer_j_per_density": deposition_energy,
    }

    status = "PASS" if all(v["status"] == "PASS" for v in cases.values()) else "FAIL"
    report = {
        "schema": "MOBILE_V2_REFERENCE_CPU_CONTRACTS/v1",
        "status": status,
        "continuous_model": {
            "mass": "dh/dt+div(h*u)=E-D",
            "momentum": "d(h*u)/dt+div(h*u_tensor_u+P(h)I)=-g*h*grad(b_eff)+friction+external+exchange_momentum",
            "pressure": "P(h)=0.5*K*g*h^2",
            "energy": "0.5*h*|u|^2+g*h*b_eff+0.5*K*g*h^2",
            "K_classification": "ENGINEERING_CLOSURE_UNCALIBRATED",
        },
        "B_EFF_STATE_DEFINED": "YES",
        "production_imported_or_modified": False,
        "isaac_launched": False,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(json.dumps({"status": status, "cases": {k: v["status"] for k, v in cases.items()}, "output": str(OUT)}, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
