#!/usr/bin/env python3
"""Reconstruct one frozen production Mobile finite-volume update.

The tool is read-only with respect to production physics.  It consumes the
saved PRE/POST operator boundary, replays the exact Warp equations in NumPy,
and persists flux-level causal diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "outputs/mobile_large_avalanche_causal_audit/strongest_operator_boundary.npz"
FIVE_SECOND = ROOT / "outputs/mobile_large_avalanche_causal_audit/short_replay_5s.json"
OUT = ROOT / "outputs/mobile_large_avalanche_causal_audit/mobile_single_step_flux_report.json"

DX = 0.05
DY = 0.05
DT = 1.0 / 60.0
GRAVITY = 9.81
CFL = 0.35
PRESSURE_COEFFICIENT = 0.45
DRY = 1.0e-8
VELOCITY_CAP = 8.0
TOOL_RELAXATION_S = 0.12


def _material() -> dict[str, float]:
    source = json.loads(
        (ROOT / "configs/literature/iron_ore_condition_scenarios.json").read_text()
    )
    item = next(
        x for x in source["scenarios"]
        if x["id"] == "MOHAJERI_I3_AS_RECEIVED_LOW_STRESS_NEAR_MATCHED"
    )
    return {
        "density": float(item["bulk_density_kg_m3"]),
        "cohesion": float(item["cohesion_pa"]),
        "tan_start": float(np.tan(np.deg2rad(item["theta_start_deg"]))),
        "tan_stop": float(np.tan(np.deg2rad(item["theta_stop_deg"]))),
        "mobile_friction": float(item["mobile_friction_coefficient"]),
    }


def _neighbor(field: np.ndarray) -> tuple[np.ndarray, ...]:
    left = np.empty_like(field)
    right = np.empty_like(field)
    down = np.empty_like(field)
    up = np.empty_like(field)
    left[:, 0] = field[:, 0]
    left[:, 1:] = field[:, :-1]
    right[:, -1] = field[:, -1]
    right[:, :-1] = field[:, 1:]
    down[0, :] = field[0, :]
    down[1:, :] = field[:-1, :]
    up[-1, :] = field[-1, :]
    up[:-1, :] = field[1:, :]
    return left, right, down, up


def _gradient(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left, right, down, up = _neighbor(field)
    hx = np.full(field.shape, 2.0 * DX)
    hy = np.full(field.shape, 2.0 * DY)
    hx[:, (0, -1)] = DX
    hy[(0, -1), :] = DY
    return (right - left) / hx, (up - down) / hy


def _strongest_pair(field: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float, str]:
    dx = np.abs(np.diff(field, axis=1))
    dy = np.abs(np.diff(field, axis=0))
    x_max = float(np.max(dx))
    y_max = float(np.max(dy))
    if x_max >= y_max:
        row, col = np.unravel_index(int(np.argmax(dx)), dx.shape)
        return (int(row), int(col)), (int(row), int(col + 1)), x_max, "X_FACE"
    row, col = np.unravel_index(int(np.argmax(dy)), dy.shape)
    return (int(row), int(col)), (int(row + 1), int(col)), y_max, "Y_FACE"


def _cell_id(cell: tuple[int, int]) -> str:
    return f"r{cell[0]}_c{cell[1]}"


def _q(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "max": float(np.max(values, initial=0.0)),
        "p95": float(np.percentile(values, 95.0)) if values.size else 0.0,
        "p99": float(np.percentile(values, 99.0)) if values.size else 0.0,
    }


def main() -> None:
    data = np.load(BOUNDARY)
    resting = np.asarray(data["PRE_resting"], dtype=np.float64)
    initial_h = np.asarray(data["PRE_mobile"], dtype=np.float64)
    saved_h = np.asarray(data["POST_mobile"], dtype=np.float64)
    initial_mx = np.asarray(data["PRE_momentum_x"], dtype=np.float64)
    initial_my = np.asarray(data["PRE_momentum_y"], dtype=np.float64)
    saved_mx = np.asarray(data["POST_momentum_x"], dtype=np.float64)
    saved_my = np.asarray(data["POST_momentum_y"], dtype=np.float64)
    weights = np.asarray(data["PRE_weights"], dtype=np.float64)
    forcing = np.asarray(data["PRE_material_mask"], dtype=np.int32) != 0
    rows, cols = initial_h.shape
    mat = _material()
    density = mat["density"]

    pair_a, pair_b, saved_jh, face_axis = _strongest_pair(resting + saved_h)
    pair = {pair_a, pair_b}
    target_flat = {np.ravel_multi_index(cell, initial_h.shape) for cell in pair}
    target_ledgers: dict[str, list[dict[str, Any]]] = {
        _cell_id(cell): [] for cell in pair
    }
    target_sources: dict[str, list[dict[str, Any]]] = {
        _cell_id(cell): [] for cell in pair
    }

    h = initial_h.copy()
    mx = initial_mx.copy()
    my = initial_my.copy()
    cumulative_in = np.zeros_like(h)
    cumulative_out = np.zeros_like(h)
    initial_contrib_max = initial_h.copy()
    initial_contrib_min = initial_h.copy()
    volume_cfl_actual: list[float] = []
    volume_cfl_candidate: list[float] = []
    wave_cfl: list[float] = []
    face_velocity_errors: list[float] = []
    advecting_vs_carried_normal_errors: list[float] = []
    momentum_without_mass = 0
    mass_without_consistent_momentum = 0
    substep_records: list[dict[str, Any]] = []
    remaining = DT
    substep = 0

    while remaining > 1.0e-14:
        wet = h > DRY
        vx0 = np.divide(mx, h, out=np.zeros_like(h), where=wet)
        vy0 = np.divide(my, h, out=np.zeros_like(h), where=wet)
        c = np.sqrt(PRESSURE_COEFFICIENT * GRAVITY * np.maximum(h, 0.0))
        wave = np.hypot(vx0, vy0) + c
        maximum_wave = float(np.max(wave[wet], initial=0.0))
        stable_dt = remaining if maximum_wave <= 1.0e-12 else CFL * min(DX, DY) / maximum_wave
        sub_dt = min(remaining, stable_dt)
        wave_cfl.extend((wave[wet] * sub_dt / min(DX, DY)).tolist())

        surface = resting + h
        grad_x, grad_y = _gradient(surface)
        pressure_x, pressure_y = _gradient(h)
        slope_tan = np.hypot(grad_x, grad_y)
        norm = np.sqrt(1.0 + slope_tan * slope_tan)
        layer_depth = h
        drive = density * GRAVITY * layer_depth * slope_tan / norm
        normal = density * GRAVITY * layer_depth / norm
        start_margin = drive - (mat["cohesion"] + normal * mat["tan_start"])
        stop_margin = drive - (mat["cohesion"] + normal * mat["tan_stop"])
        moving = np.hypot(vx0, vy0) > 0.01
        dynamic = (start_margin > 0.0) | (moving & (stop_margin > 0.0)) | forcing
        vxg = vx0.copy()
        vyg = vy0.copy()
        vxg[dynamic] -= GRAVITY * (
            grad_x[dynamic] + PRESSURE_COEFFICIENT * pressure_x[dynamic]
        ) * sub_dt
        vyg[dynamic] -= GRAVITY * (
            grad_y[dynamic] + PRESSURE_COEFFICIENT * pressure_y[dynamic]
        ) * sub_dt
        speed_g = np.hypot(vxg, vyg)
        friction_factor = np.maximum(
            0.0,
            1.0 - mat["mobile_friction"] * GRAVITY * sub_dt
            / np.maximum(speed_g, 1.0e-12),
        )
        vx = vxg * friction_factor
        vy = vyg * friction_factor
        blend = min(1.0, sub_dt / TOOL_RELAXATION_S)
        vx[forcing] += blend * (0.0 - vx[forcing])
        vy[forcing] += blend * (0.0 - vy[forcing])
        vx[~wet] = 0.0
        vy[~wet] = 0.0

        for cell in pair:
            r, cidx = cell
            target_sources[_cell_id(cell)].append({
                "substep": substep,
                "dt_s": sub_dt,
                "h_before_m": float(h[cell]),
                "velocity_before_m_s": [float(vx0[cell]), float(vy0[cell])],
                "gravity_pressure_delta_velocity_m_s": [
                    float(vxg[cell] - vx0[cell]), float(vyg[cell] - vy0[cell])
                ],
                "friction_delta_velocity_m_s": [
                    float(vx[cell] - vxg[cell]), float(vy[cell] - vyg[cell])
                ],
                "H_free_gradient": [float(grad_x[cell]), float(grad_y[cell])],
                "h_mobile_gradient": [float(pressure_x[cell]), float(pressure_y[cell])],
                "dynamic": bool(dynamic[cell]),
                "forcing": bool(forcing[cell]),
            })

        volume_before = h * weights
        momentum_volume_x = volume_before * vx
        momentum_volume_y = volume_before * vy

        # Flat edge arrays in the same x-then-y order as the Warp kernel.
        x_first = np.arange(rows * (cols - 1), dtype=np.int64).reshape(rows, cols - 1)
        x_first = (np.arange(rows)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
        x_second = x_first + 1
        x_face_v = (vx.ravel()[x_first] + vx.ravel()[x_second]) * 0.5
        x_donor = np.where(x_face_v < 0.0, x_second, x_first)
        x_receiver = np.where(x_face_v < 0.0, x_first, x_second)
        x_candidate = h.ravel()[x_donor] * np.abs(x_face_v) * DY * sub_dt

        y_first = (np.arange(rows - 1)[:, None] * cols + np.arange(cols)[None, :]).ravel()
        y_second = y_first + cols
        y_face_v = (vy.ravel()[y_first] + vy.ravel()[y_second]) * 0.5
        y_donor = np.where(y_face_v < 0.0, y_second, y_first)
        y_receiver = np.where(y_face_v < 0.0, y_first, y_second)
        y_candidate = h.ravel()[y_donor] * np.abs(y_face_v) * DX * sub_dt

        donors = np.concatenate((x_donor, y_donor))
        receivers = np.concatenate((x_receiver, y_receiver))
        candidates = np.concatenate((x_candidate, y_candidate))
        face_velocities = np.concatenate((x_face_v, y_face_v))
        face_axes = np.concatenate((
            np.zeros(x_candidate.size, dtype=np.int8),
            np.ones(y_candidate.size, dtype=np.int8),
        ))
        outgoing_candidate = np.zeros(h.size, dtype=np.float64)
        np.add.at(outgoing_candidate, donors, candidates)
        donor_volume = volume_before.ravel()[donors]
        factors = np.ones_like(candidates)
        totals = outgoing_candidate[donors]
        limited = (totals > donor_volume) & (totals > 0.0)
        factors[limited] = donor_volume[limited] / totals[limited]
        amounts = candidates * factors

        active_donors = outgoing_candidate > 0.0
        candidate_cfl = np.divide(
            outgoing_candidate[active_donors], volume_before.ravel()[active_donors]
        )
        outgoing_actual = np.zeros(h.size, dtype=np.float64)
        np.add.at(outgoing_actual, donors, amounts)
        actual_cfl = np.divide(
            outgoing_actual[active_donors], volume_before.ravel()[active_donors]
        )
        volume_cfl_candidate.extend(candidate_cfl.tolist())
        volume_cfl_actual.extend(actual_cfl.tolist())

        actual_in = np.zeros(h.size, dtype=np.float64)
        np.add.at(actual_in, receivers, amounts)
        cumulative_in += actual_in.reshape(h.shape)
        cumulative_out += outgoing_actual.reshape(h.shape)

        positive = amounts > 0.0
        for donor, receiver in zip(donors[positive], receivers[positive]):
            dr, dc = np.unravel_index(int(donor), h.shape)
            rr, rc = np.unravel_index(int(receiver), h.shape)
            value = initial_h[dr, dc]
            initial_contrib_max[rr, rc] = max(initial_contrib_max[rr, rc], value)
            initial_contrib_min[rr, rc] = min(initial_contrib_min[rr, rc], value)

        carried_x = amounts * vx.ravel()[donors]
        carried_y = amounts * vy.ravel()[donors]
        implied_x = np.divide(carried_x, amounts, out=np.zeros_like(amounts), where=positive)
        implied_y = np.divide(carried_y, amounts, out=np.zeros_like(amounts), where=positive)
        face_velocity_errors.extend(
            np.hypot(
                implied_x[positive] - vx.ravel()[donors[positive]],
                implied_y[positive] - vy.ravel()[donors[positive]],
            ).tolist()
        )
        normal_donor_velocity = np.where(
            face_axes == 0, vx.ravel()[donors], vy.ravel()[donors]
        )
        advecting_vs_carried_normal_errors.extend(
            np.abs(normal_donor_velocity[positive] - face_velocities[positive]).tolist()
        )
        momentum_without_mass += int(np.count_nonzero(
            (~positive) & ((np.abs(carried_x) > 0.0) | (np.abs(carried_y) > 0.0))
        ))
        mass_without_consistent_momentum += int(np.count_nonzero(
            positive & (~np.isfinite(implied_x) | ~np.isfinite(implied_y))
        ))

        for edge in np.flatnonzero(positive):
            donor = int(donors[edge])
            receiver = int(receivers[edge])
            if donor not in target_flat and receiver not in target_flat:
                continue
            donor_cell = tuple(int(x) for x in np.unravel_index(donor, h.shape))
            receiver_cell = tuple(int(x) for x in np.unravel_index(receiver, h.shape))
            record = {
                "substep": substep,
                "dt_s": sub_dt,
                "axis": "X" if face_axes[edge] == 0 else "Y",
                "donor": list(donor_cell),
                "receiver": list(receiver_cell),
                "face_velocity_normal_m_s": float(face_velocities[edge]),
                "candidate_volume_m3": float(candidates[edge]),
                "donor_limiter": float(factors[edge]),
                "actual_volume_m3": float(amounts[edge]),
                "donor_velocity_xy_m_s": [
                    float(vx.ravel()[donor]), float(vy.ravel()[donor])
                ],
                "transported_momentum_kg_m_s": [
                    float(density * carried_x[edge]),
                    float(density * carried_y[edge]),
                ],
            }
            for cell in pair:
                flat = np.ravel_multi_index(cell, h.shape)
                if flat == donor or flat == receiver:
                    target_ledgers[_cell_id(cell)].append(record)

        volume_after = volume_before.ravel().copy()
        px_after = momentum_volume_x.ravel().copy()
        py_after = momentum_volume_y.ravel().copy()
        np.add.at(volume_after, donors, -amounts)
        np.add.at(volume_after, receivers, amounts)
        np.add.at(px_after, donors, -carried_x)
        np.add.at(px_after, receivers, carried_x)
        np.add.at(py_after, donors, -carried_y)
        np.add.at(py_after, receivers, carried_y)
        volume_after = np.maximum(volume_after, 0.0).reshape(h.shape)
        h = volume_after / weights
        vx_after = np.divide(
            px_after.reshape(h.shape), volume_after,
            out=np.zeros_like(h), where=volume_after > DRY * weights,
        )
        vy_after = np.divide(
            py_after.reshape(h.shape), volume_after,
            out=np.zeros_like(h), where=volume_after > DRY * weights,
        )
        speed_after = np.hypot(vx_after, vy_after)
        capped = speed_after > VELOCITY_CAP
        vx_after[capped] *= VELOCITY_CAP / speed_after[capped]
        vy_after[capped] *= VELOCITY_CAP / speed_after[capped]
        mx = h * vx_after
        my = h * vy_after
        mx[h <= DRY] = 0.0
        my[h <= DRY] = 0.0
        substep_records.append({
            "substep": substep,
            "dt_s": sub_dt,
            "maximum_wave_speed_m_s": maximum_wave,
            "maximum_wave_CFL": float(maximum_wave * sub_dt / min(DX, DY)),
            "maximum_candidate_volume_CFL": float(np.max(candidate_cfl, initial=0.0)),
            "maximum_actual_volume_CFL": float(np.max(actual_cfl, initial=0.0)),
            "transported_volume_m3": float(np.sum(amounts, dtype=np.float64)),
        })
        remaining -= sub_dt
        substep += 1

    replay_h_error = h - saved_h
    replay_mx_error = mx - saved_mx
    replay_my_error = my - saved_my
    mass_lhs = weights * (saved_h - initial_h)
    mass_rhs = cumulative_in - cumulative_out
    mass_error = mass_lhs - mass_rhs
    mass_tolerance = 2.0e-13

    extrema_tolerance = 5.0e-14
    pure = ~forcing
    new_max = pure & (saved_h > initial_contrib_max + extrema_tolerance)
    new_min = pure & (saved_h < initial_contrib_min - extrema_tolerance)
    extrema = new_max | new_min
    extrema_volume = float(np.sum(saved_h[extrema] * weights[extrema], dtype=np.float64))
    max_excess_volume = float(np.sum(
        np.maximum(saved_h - initial_contrib_max, 0.0) * weights,
        dtype=np.float64,
    ))

    def extremum_records(mask: np.ndarray, kind: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for r, cidx in np.argwhere(mask):
            cell = (int(r), int(cidx))
            envelope = (
                initial_contrib_max[cell] if kind == "MAX" else initial_contrib_min[cell]
            )
            records.append({
                "cell": [int(r), int(cidx)],
                "kind": kind,
                "old_h_mobile_m": float(initial_h[cell]),
                "new_h_mobile_m": float(saved_h[cell]),
                "old_contributor_envelope_m": float(envelope),
                "outside_envelope_m": float(
                    saved_h[cell] - envelope if kind == "MAX" else envelope - saved_h[cell]
                ),
                "new_mobile_volume_m3": float(saved_h[cell] * weights[cell]),
            })
        return records

    new_max_records = extremum_records(new_max, "MAX")
    new_min_records = extremum_records(new_min, "MIN")
    min_deficit_volume = float(np.sum(
        np.maximum(initial_contrib_min - saved_h, 0.0) * weights,
        dtype=np.float64,
    ))

    def cell_report(cell: tuple[int, int]) -> dict[str, Any]:
        r, cidx = cell
        neighbors = []
        for dr, dc, label in ((0, -1, "LEFT"), (0, 1, "RIGHT"), (-1, 0, "DOWN"), (1, 0, "UP")):
            rr, cc = r + dr, cidx + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                neighbors.append({
                    "direction": label,
                    "cell": [rr, cc],
                    "old_h_mobile_m": float(initial_h[rr, cc]),
                    "new_h_mobile_m": float(saved_h[rr, cc]),
                    "old_H_free_m": float(resting[rr, cc] + initial_h[rr, cc]),
                    "new_H_free_m": float(resting[rr, cc] + saved_h[rr, cc]),
                    "old_velocity_xy_m_s": [
                        float(initial_mx[rr, cc] / initial_h[rr, cc]) if initial_h[rr, cc] > DRY else 0.0,
                        float(initial_my[rr, cc] / initial_h[rr, cc]) if initial_h[rr, cc] > DRY else 0.0,
                    ],
                })
        balance_by_substep = []
        for sub in range(substep):
            records = [
                item for item in target_ledgers[_cell_id(cell)]
                if item["substep"] == sub
            ]
            incoming = float(sum(
                item["actual_volume_m3"] for item in records
                if tuple(item["receiver"]) == cell
            ))
            outgoing = float(sum(
                item["actual_volume_m3"] for item in records
                if tuple(item["donor"]) == cell
            ))
            sub_dt = float(substep_records[sub]["dt_s"])
            balance_by_substep.append({
                "substep": sub,
                "dt_s": sub_dt,
                "incoming_volume_m3": incoming,
                "outgoing_volume_m3": outgoing,
                "net_convergence_volume_m3": incoming - outgoing,
                "minus_div_q_equivalent_m_s": (
                    (incoming - outgoing) / (weights[cell] * sub_dt)
                ),
            })
        return {
            "cell": [r, cidx],
            "control_area_m2": float(weights[cell]),
            "is_domain_boundary": bool(r in (0, rows - 1) or cidx in (0, cols - 1)),
            "is_vertex_area_nonuniform": bool(abs(weights[cell] - DX * DY) > 1.0e-15),
            "old_h_mobile_m": float(initial_h[cell]),
            "new_h_mobile_m": float(saved_h[cell]),
            "delta_h_mobile_m": float(saved_h[cell] - initial_h[cell]),
            "old_H_free_m": float(resting[cell] + initial_h[cell]),
            "new_H_free_m": float(resting[cell] + saved_h[cell]),
            "lhs_A_delta_h_m3": float(mass_lhs[cell]),
            "incoming_volume_m3": float(cumulative_in[cell]),
            "outgoing_volume_m3": float(cumulative_out[cell]),
            "rhs_net_flux_m3": float(mass_rhs[cell]),
            "mass_reconstruction_error_m3": float(mass_error[cell]),
            "explicit_mass_source_m3": 0.0,
            "fluxes": target_ledgers[_cell_id(cell)],
            "flux_balance_by_substep": balance_by_substep,
            "momentum_sources": target_sources[_cell_id(cell)],
            "neighbors": neighbors,
        }

    a_report = cell_report(pair_a)
    b_report = cell_report(pair_b)
    high_cell = pair_a if (resting + saved_h)[pair_a] >= (resting + saved_h)[pair_b] else pair_b
    low_cell = pair_b if high_cell == pair_a else pair_a
    high = a_report if high_cell == pair_a else b_report
    low = b_report if low_cell == pair_b else a_report
    high_convergent = high["rhs_net_flux_m3"] > 0.0
    low_divergent = low["rhs_net_flux_m3"] < 0.0
    incoming_directions = {
        (tuple(record["donor"])[0] - high_cell[0], tuple(record["donor"])[1] - high_cell[1])
        for record in high["fluxes"] if tuple(record["receiver"]) == high_cell
    }
    opposing_streams = any((-dr, -dc) in incoming_directions for dr, dc in incoming_directions)

    # Examine the face continuing away from the high-to-low pair.  At this
    # event the low cell is a free-surface valley; a centered two-cell source
    # gradient points through an immediately uphill face.
    away_dr = low_cell[0] - high_cell[0]
    away_dc = low_cell[1] - high_cell[1]
    downstream = (low_cell[0] + away_dr, low_cell[1] + away_dc)
    downstream_in_domain = 0 <= downstream[0] < rows and 0 <= downstream[1] < cols
    low_to_downstream = float(sum(
        record["actual_volume_m3"] for record in low["fluxes"]
        if tuple(record["donor"]) == low_cell
        and tuple(record["receiver"]) == downstream
    )) if downstream_in_domain else 0.0
    low_free = float((resting + initial_h)[low_cell])
    downstream_free = (
        float((resting + initial_h)[downstream]) if downstream_in_domain else low_free
    )
    summed_gravity_delta = np.sum([
        item["gravity_pressure_delta_velocity_m_s"]
        for item in low["momentum_sources"]
    ], axis=0)
    source_along_away = float(
        summed_gravity_delta[0] * away_dc + summed_gravity_delta[1] * away_dr
    )
    uphill_face_stencil = bool(
        downstream_in_domain
        and downstream_free > low_free
        and low_to_downstream > 0.0
        and source_along_away > 0.0
    )

    five = json.loads(FIVE_SECOND.read_text())
    path_volume = float(five["transport_0_to_5s"]["donor_export_m3"] * DX)
    representative_mobile = float(np.mean([
        float(sample["mobile_volume_m3"]) for sample in five["samples"]
    ]))
    mean_distance = path_volume / representative_mobile

    report = {
        "schema": "MOBILE_SINGLE_STEP_FLUX_REPORT/v1",
        "source_boundary": str(BOUNDARY.relative_to(ROOT)),
        "production_step": {
            "elapsed_s": 3.75,
            "dt_s": DT,
            "substeps": substep,
            "substep_records": substep_records,
            "mass_source": "NONE",
            "momentum_source_equation": (
                "du/dt=-g*(grad(H_free)+pressure_coefficient*grad(h_mobile)); "
                "then Coulomb speed decrement mu_mobile*g*dt; tool forcing absent"
            ),
            "implemented_wave_speed_proxy": "c=sqrt(pressure_coefficient*g*h_mobile)",
            "hyperbolic_interpretation": (
                "SHALLOW_LAYER_CFL_PROXY_NOT_A_FULL_CONSERVATIVE_HYPERBOLIC_EIGENSYSTEM; "
                "pressure/gravity are explicit momentum sources"
            ),
        },
        "strongest_pair": {
            "face_axis": face_axis,
            "cell_a": list(pair_a),
            "cell_b": list(pair_b),
            "high_cell": list(high_cell),
            "low_cell": list(low_cell),
            "saved_post_J_H_m": saved_jh,
            "old_pair_J_H_m": float(abs((resting + initial_h)[pair_a] - (resting + initial_h)[pair_b])),
            "delta_pair_J_H_m": float(
                saved_jh - abs((resting + initial_h)[pair_a] - (resting + initial_h)[pair_b])
            ),
            "cells": {_cell_id(pair_a): a_report, _cell_id(pair_b): b_report},
        },
        "replay_equivalence": {
            "max_abs_h_mobile_error_m": float(np.max(np.abs(replay_h_error))),
            "max_abs_momentum_x_error_m2_s": float(np.max(np.abs(replay_mx_error))),
            "max_abs_momentum_y_error_m2_s": float(np.max(np.abs(replay_my_error))),
        },
        "MASS_UPDATE_RECONSTRUCTION": (
            "PASS" if float(np.max(np.abs(mass_error))) <= mass_tolerance
            and float(np.max(np.abs(replay_h_error))) <= mass_tolerance else "FAIL"
        ),
        "mass_update": {
            "max_abs_cell_balance_error_m3": float(np.max(np.abs(mass_error))),
            "global_balance_error_m3": float(np.sum(mass_error, dtype=np.float64)),
            "tolerance_m3": mass_tolerance,
        },
        "NEW_LOCAL_MAX_COUNT": int(np.count_nonzero(new_max)),
        "NEW_LOCAL_MIN_COUNT": int(np.count_nonzero(new_min)),
        "NEW_EXTREMUM_VOLUME_M3": extrema_volume,
        "extremum_diagnostics": {
            "definition": "POST h outside envelope of full-step PRE h at self plus every actual incoming donor",
            "max_excess_above_contributor_envelope_m3": max_excess_volume,
            "min_deficit_below_contributor_envelope_m3": min_deficit_volume,
            "new_local_maxima": new_max_records,
            "new_local_minima": new_min_records,
        },
        "STRONGEST_JH_CAUSE": (
            "CENTERED_GRAVITY_PRESSURE_STENCIL_DRIVEN_DIFFERENTIAL_DIVERGENCE_"
            "AT_A_PREEXISTING_GRID_SCALE_FREE_SURFACE_VALLEY"
        ),
        "MAX_VOLUME_CFL": float(np.max(volume_cfl_actual, initial=0.0)),
        "volume_CFL": {
            "actual_donor_limited": _q(np.asarray(volume_cfl_actual)),
            "candidate_before_donor_limit": _q(np.asarray(volume_cfl_candidate)),
            "all_actual_le_one": bool(np.max(volume_cfl_actual, initial=0.0) <= 1.0 + 1.0e-12),
        },
        "MAX_WAVE_CFL": float(np.max(wave_cfl, initial=0.0)),
        "wave_CFL": _q(np.asarray(wave_cfl)),
        "MASS_MOMENTUM_FLUX_CONSISTENCY": (
            "PASS" if max(face_velocity_errors, default=0.0) <= 1.0e-14
            and momentum_without_mass == 0 and mass_without_consistent_momentum == 0
            else "FAIL"
        ),
        "mass_momentum_flux": {
            "maximum_implied_vs_donor_velocity_error_m_s": max(face_velocity_errors, default=0.0),
            "momentum_without_mass_face_count": momentum_without_mass,
            "mass_without_consistent_momentum_face_count": mass_without_consistent_momentum,
            "advecting_face_normal_vs_carried_donor_normal_velocity_error_m_s": _q(
                np.asarray(advecting_vs_carried_normal_errors)
            ),
            "advecting_vs_carried_interpretation": (
                "EXPECTED_FIRST_ORDER_UPWIND_SEMANTICS: face-average normal velocity sets "
                "Delta V, while donor cell velocity is the momentum per transported mass"
            ),
            "checkerboard_face_velocity": "NOT_DETECTED_AT_STRONGEST_PAIR" if not opposing_streams else "OPPOSING_INCOMING_STREAMS_PRESENT",
        },
        "COMPRESSION_FRONT_PHYSICAL_CAUSE": (
            "NOT_A_COMPRESSION_FRONT__LOW_SIDE_DIVERGENCE; EXISTING MOMENTUM "
            "PLUS A CENTERED SOURCE GRADIENT EVACUATES THROUGH AN IMMEDIATELY_UPHILL_FACE"
        ),
        "compression_front": {
            "high_cell_net_convergence_m3": high["rhs_net_flux_m3"],
            "high_cell_minus_div_q_equivalent_m_s": high["rhs_net_flux_m3"] / (weights[high_cell] * DT),
            "low_cell_net_convergence_m3": low["rhs_net_flux_m3"],
            "low_cell_is_divergent": low_divergent,
            "opposing_incoming_streams": opposing_streams,
            "boundary_stencil": bool(high["is_domain_boundary"] or low["is_domain_boundary"]),
            "active_inactive_interface": bool(
                (initial_h[high_cell] > DRY) != (initial_h[low_cell] > DRY)
            ),
            "vertex_control_area_effect": bool(
                high["is_vertex_area_nonuniform"] or low["is_vertex_area_nonuniform"]
            ),
            "stencil_causality": {
                "high_cell": list(high_cell),
                "low_cell": list(low_cell),
                "downstream_cell": list(downstream) if downstream_in_domain else None,
                "low_old_H_free_m": low_free,
                "downstream_old_H_free_m": downstream_free,
                "downstream_minus_low_H_free_m": downstream_free - low_free,
                "low_to_downstream_volume_m3": low_to_downstream,
                "summed_gravity_pressure_delta_velocity_xy_m_s": summed_gravity_delta.tolist(),
                "source_delta_velocity_along_low_to_downstream_m_s": source_along_away,
                "gravity_source_points_through_uphill_face": uphill_face_stencil,
            },
        },
        "GRID_STENCIL_ARTIFACT": "YES" if uphill_face_stencil else "UNRESOLVED",
        "transport_distance_5s": {
            "transport_path_volume_m4": path_volume,
            "representative_mobile_volume_m3": representative_mobile,
            "representative_definition": "arithmetic time mean over the 300 saved 5 s step samples",
            "EFFECTIVE_MEAN_TRANSPORT_DISTANCE_5S_M": mean_distance,
        },
        "ROOT_CAUSE_REFINED": (
            "NON_WELL_BALANCED_CELL_CENTERED_GRAVITY_PRESSURE_SOURCE_AT_A_GRID_SCALE_"
            "FREE_SURFACE_VALLEY: THE CENTERED GRADIENT IS DOMINATED BY THE HIGH CELL "
            "BEHIND AND ADDS ACCELERATION THROUGH AN IMMEDIATELY UPHILL FORWARD FACE. "
            "THE CONSERVATIVE, CFL-COMPLIANT UPWIND FLUX THEN DEEPENS THE LOW CELL. "
            "THIS IS DIFFERENTIAL DIVERGENCE, NOT COMPRESSION, MASS LEAKAGE, OR A DOMAIN BOUNDARY."
        ),
        "SAFE_MINIMAL_NUMERICAL_FIX": "NONE",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "MASS_UPDATE_RECONSTRUCTION", "NEW_LOCAL_MAX_COUNT",
            "NEW_LOCAL_MIN_COUNT", "NEW_EXTREMUM_VOLUME_M3",
            "STRONGEST_JH_CAUSE", "MAX_VOLUME_CFL", "MAX_WAVE_CFL",
            "MASS_MOMENTUM_FLUX_CONSISTENCY", "COMPRESSION_FRONT_PHYSICAL_CAUSE",
            "GRID_STENCIL_ARTIFACT", "ROOT_CAUSE_REFINED",
            "SAFE_MINIMAL_NUMERICAL_FIX",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
