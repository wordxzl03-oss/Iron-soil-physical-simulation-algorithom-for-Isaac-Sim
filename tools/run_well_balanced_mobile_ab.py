#!/usr/bin/env python3
"""Experimental face-consistent Mobile source A/B on one frozen boundary.

This file is deliberately outside the production runtime.  It neither changes
the default CPU/Warp operators nor imports itself from the package.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
ISAAC_ROOT = Path("/home/eric/isaacsim")
for _root in sorted(ISAAC_ROOT.glob("extscache/omni.warp.core-*")):
    if (_root / "warp").is_dir():
        sys.path.insert(0, str(_root))
        break

from isaac_bulk_pipeline.bulk_interaction.warp_mobile_layer import _kernels as production_kernels  # noqa: E402
from isaac_bulk_pipeline.performance import WarpRuntime  # noqa: E402

from tools.run_mobile_single_step_flux_audit import (  # noqa: E402
    BOUNDARY,
    CFL,
    DRY,
    DT,
    DX,
    DY,
    GRAVITY,
    PRESSURE_COEFFICIENT,
    TOOL_RELAXATION_S,
    VELOCITY_CAP,
    _gradient,
    _material,
    _q,
    _strongest_pair,
)


OUT = ROOT / "outputs/mobile_large_avalanche_causal_audit/well_balanced_mobile_ab.json"
FIELDS_OUT = ROOT / "outputs/mobile_large_avalanche_causal_audit/well_balanced_mobile_ab_fields.npz"
_EXPERIMENTAL_KERNELS: dict[int, Any] = {}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _directional_gradient(
    field: np.ndarray, velocity: np.ndarray, spacing: float, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a flow-face gradient and selected neighbor offset.

    Moving material samples the face it is moving through. Material at rest
    sees only immediately downhill faces; a local valley therefore receives no
    acceleration merely because the opposite cell is high.
    """

    minus = np.roll(field, 1, axis=axis)
    plus = np.roll(field, -1, axis=axis)
    if axis == 1:
        minus[:, 0] = field[:, 0]
        plus[:, -1] = field[:, -1]
    else:
        minus[0, :] = field[0, :]
        plus[-1, :] = field[-1, :]
    backward = (field - minus) / spacing
    forward = (plus - field) / spacing
    result = np.zeros_like(field)
    choice = np.zeros(field.shape, dtype=np.int8)
    positive = velocity > 1.0e-12
    negative = velocity < -1.0e-12
    result[positive] = forward[positive]
    choice[positive] = 1
    result[negative] = backward[negative]
    choice[negative] = -1
    quiet = ~(positive | negative)
    downhill_plus = quiet & (forward < 0.0)
    downhill_minus = quiet & (backward > 0.0)
    only_plus = downhill_plus & ~downhill_minus
    only_minus = downhill_minus & ~downhill_plus
    both = downhill_plus & downhill_minus
    result[only_plus] = forward[only_plus]
    choice[only_plus] = 1
    result[only_minus] = backward[only_minus]
    choice[only_minus] = -1
    choose_plus = both & (np.abs(forward) >= np.abs(backward))
    choose_minus = both & ~choose_plus
    result[choose_plus] = forward[choose_plus]
    choice[choose_plus] = 1
    result[choose_minus] = backward[choose_minus]
    choice[choose_minus] = -1
    return result, choice


def _gradient_on_selected_face(
    field: np.ndarray,
    choice: np.ndarray,
    spacing: float,
    axis: int,
) -> np.ndarray:
    """Evaluate another field on the face selected from ``H_free``."""
    minus = np.roll(field, 1, axis=axis)
    plus = np.roll(field, -1, axis=axis)
    if axis == 1:
        minus[:, 0] = field[:, 0]
        plus[:, -1] = field[:, -1]
    else:
        minus[0, :] = field[0, :]
        plus[-1, :] = field[-1, :]
    backward = (field - minus) / spacing
    forward = (plus - field) / spacing
    return np.where(choice > 0, forward, np.where(choice < 0, backward, 0.0))


def _experimental_source_kernel(wp: Any) -> Any:
    cached = _EXPERIMENTAL_KERNELS.get(id(wp))
    if cached is not None:
        return cached

    @wp.kernel
    def source(
        resting: wp.array(dtype=wp.float64),
        height: wp.array(dtype=wp.float64),
        momentum_x: wp.array(dtype=wp.float64),
        momentum_y: wp.array(dtype=wp.float64),
        forcing: wp.array(dtype=wp.int32),
        weights: wp.array(dtype=wp.float64),
        velocity_x: wp.array(dtype=wp.float64),
        velocity_y: wp.array(dtype=wp.float64),
        rows: int,
        cols: int,
        dx: wp.float64,
        dy: wp.float64,
        dt: wp.float64,
        gravity: wp.float64,
        pressure_coefficient: wp.float64,
        friction_coefficient: wp.float64,
        cohesion_pa: wp.float64,
        tan_start: wp.float64,
        tan_stop: wp.float64,
        dry: wp.float64,
        density: wp.float64,
        source_impulse: wp.array(dtype=wp.float64),
    ):
        index = wp.tid()
        row = index // cols
        col = index - row * cols
        h = height[index]
        if h <= dry:
            velocity_x[index] = wp.float64(0.0)
            velocity_y[index] = wp.float64(0.0)
            return
        vx0 = momentum_x[index] / h
        vy0 = momentum_y[index] / h
        left = index
        right = index
        down = index
        up = index
        if col > 0:
            left = index - 1
        if col + 1 < cols:
            right = index + 1
        if row > 0:
            down = index - cols
        if row + 1 < rows:
            up = index + cols
        s = resting[index] + h
        sl = resting[left] + height[left]
        sr = resting[right] + height[right]
        sd = resting[down] + height[down]
        su = resting[up] + height[up]

        # Central H_free slope is retained only for the accepted cohesive
        # start/stop classification, not for the experimental force direction.
        hx = wp.float64(2.0) * dx
        hy = wp.float64(2.0) * dy
        if col == 0 or col + 1 == cols:
            hx = dx
        if row == 0 or row + 1 == rows:
            hy = dy
        cgx = (sr - sl) / hx
        cgy = (su - sd) / hy
        magnitude = wp.sqrt(cgx * cgx + cgy * cgy)
        normalization = wp.sqrt(wp.float64(1.0) + magnitude * magnitude)
        drive = density * gravity * h * magnitude / normalization
        normal = density * gravity * h / normalization
        start_margin = drive - (cohesion_pa + normal * tan_start)
        stop_margin = drive - (cohesion_pa + normal * tan_stop)
        moving = wp.sqrt(vx0 * vx0 + vy0 * vy0) > wp.float64(0.01)
        dynamic = start_margin > wp.float64(0.0)
        if moving and stop_margin > wp.float64(0.0):
            dynamic = True
        if forcing[index] != 0:
            dynamic = True

        # X face selected by motion. At rest, choose an immediately downhill
        # face only. Pressure uses the same selected face.
        gx = wp.float64(0.0)
        ghx = wp.float64(0.0)
        if vx0 > wp.float64(1.0e-12):
            gx = (sr - s) / dx
            ghx = (height[right] - h) / dx
        elif vx0 < wp.float64(-1.0e-12):
            gx = (s - sl) / dx
            ghx = (h - height[left]) / dx
        else:
            forward = (sr - s) / dx
            backward = (s - sl) / dx
            if forward < wp.float64(0.0) and backward <= wp.float64(0.0):
                gx = forward
                ghx = (height[right] - h) / dx
            elif backward > wp.float64(0.0) and forward >= wp.float64(0.0):
                gx = backward
                ghx = (h - height[left]) / dx
            elif forward < wp.float64(0.0) and backward > wp.float64(0.0):
                if wp.abs(forward) >= wp.abs(backward):
                    gx = forward
                    ghx = (height[right] - h) / dx
                else:
                    gx = backward
                    ghx = (h - height[left]) / dx
        gy = wp.float64(0.0)
        ghy = wp.float64(0.0)
        if vy0 > wp.float64(1.0e-12):
            gy = (su - s) / dy
            ghy = (height[up] - h) / dy
        elif vy0 < wp.float64(-1.0e-12):
            gy = (s - sd) / dy
            ghy = (h - height[down]) / dy
        else:
            forward = (su - s) / dy
            backward = (s - sd) / dy
            if forward < wp.float64(0.0) and backward <= wp.float64(0.0):
                gy = forward
                ghy = (height[up] - h) / dy
            elif backward > wp.float64(0.0) and forward >= wp.float64(0.0):
                gy = backward
                ghy = (h - height[down]) / dy
            elif forward < wp.float64(0.0) and backward > wp.float64(0.0):
                if wp.abs(forward) >= wp.abs(backward):
                    gy = forward
                    ghy = (height[up] - h) / dy
                else:
                    gy = backward
                    ghy = (h - height[down]) / dy
        vxg = vx0
        vyg = vy0
        if dynamic:
            vxg = vx0 - gravity * (gx + pressure_coefficient * ghx) * dt
            vyg = vy0 - gravity * (gy + pressure_coefficient * ghy) * dt
        speed = wp.sqrt(vxg * vxg + vyg * vyg)
        factor = wp.float64(0.0)
        if speed > wp.float64(1.0e-12):
            factor = wp.max(
                wp.float64(0.0),
                wp.float64(1.0) - friction_coefficient * gravity * dt / speed,
            )
        vx = vxg * factor
        vy = vyg * factor
        mass = density * h * weights[index]
        wp.atomic_add(source_impulse, 0, mass * (vx - vx0))
        wp.atomic_add(source_impulse, 1, mass * (vy - vy0))
        velocity_x[index] = vx
        velocity_y[index] = vy

    _EXPERIMENTAL_KERNELS[id(wp)] = source
    return source


def _step_cpu(
    resting: np.ndarray,
    initial_h: np.ndarray,
    initial_momentum: np.ndarray,
    weights: np.ndarray,
    material: dict[str, float],
    dt: float,
    *,
    forcing: np.ndarray | None = None,
    audit_pair: set[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    h = np.asarray(initial_h, dtype=np.float64).copy()
    momentum = np.asarray(initial_momentum, dtype=np.float64).copy()
    forcing_mask = (
        np.zeros(h.shape, dtype=bool) if forcing is None else np.asarray(forcing, dtype=bool)
    )
    rows, cols = h.shape
    remaining = float(dt)
    ledgers: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    incoming_sources: dict[int, set[int]] = {}
    source_impulse = np.zeros(2)
    source_impulse_cells = np.zeros(h.shape + (2,))
    cumulative_in = np.zeros(h.shape)
    cumulative_out = np.zeros(h.shape)
    momentum_before = material["density"] * np.sum(
        momentum * weights[..., None], axis=(0, 1), dtype=np.float64
    )
    max_volume_cfl = 0.0
    max_wave_cfl = 0.0
    velocity_cap_events = 0
    velocity_cap_impulse = np.zeros(2)
    dry_cleanup_events = 0
    dry_cleanup_impulse = np.zeros(2)
    substeps = 0
    while remaining > 1.0e-14:
        wet = h > DRY
        velocity0 = np.divide(
            momentum, h[..., None], out=np.zeros_like(momentum),
            where=wet[..., None],
        )
        c = np.sqrt(PRESSURE_COEFFICIENT * GRAVITY * np.maximum(h, 0.0))
        wave = np.linalg.norm(velocity0, axis=-1) + c
        maximum_wave = float(np.max(wave[wet], initial=0.0))
        sub_dt = min(
            remaining,
            remaining if maximum_wave <= 1.0e-12 else CFL * min(DX, DY) / maximum_wave,
        )
        max_wave_cfl = max(max_wave_cfl, maximum_wave * sub_dt / min(DX, DY))
        surface = resting + h
        cgx, cgy = _gradient(surface)
        slope = np.hypot(cgx, cgy)
        norm = np.sqrt(1.0 + slope * slope)
        drive = material["density"] * GRAVITY * h * slope / norm
        normal = material["density"] * GRAVITY * h / norm
        start_margin = drive - (material["cohesion"] + normal * material["tan_start"])
        stop_margin = drive - (material["cohesion"] + normal * material["tan_stop"])
        moving = np.linalg.norm(velocity0, axis=-1) > 0.01
        dynamic = (start_margin > 0.0) | (moving & (stop_margin > 0.0)) | forcing_mask
        gx, choice_x = _directional_gradient(surface, velocity0[..., 0], DX, 1)
        gy, choice_y = _directional_gradient(surface, velocity0[..., 1], DY, 0)
        ghx = _gradient_on_selected_face(h, choice_x, DX, 1)
        ghy = _gradient_on_selected_face(h, choice_y, DY, 0)
        velocity_g = velocity0.copy()
        velocity_g[..., 0][dynamic] -= GRAVITY * (
            gx[dynamic] + PRESSURE_COEFFICIENT * ghx[dynamic]
        ) * sub_dt
        velocity_g[..., 1][dynamic] -= GRAVITY * (
            gy[dynamic] + PRESSURE_COEFFICIENT * ghy[dynamic]
        ) * sub_dt
        speed = np.linalg.norm(velocity_g, axis=-1)
        factor = np.maximum(
            0.0,
            1.0 - material["mobile_friction"] * GRAVITY * sub_dt
            / np.maximum(speed, 1.0e-12),
        )
        velocity = velocity_g * factor[..., None]
        velocity[~wet] = 0.0
        cell_mass = material["density"] * h * weights
        source_impulse_step = cell_mass[..., None] * (velocity - velocity0)
        source_impulse_cells += source_impulse_step
        source_impulse += np.sum(source_impulse_step, axis=(0, 1), dtype=np.float64)
        if audit_pair:
            for cell in sorted(audit_pair):
                r, cidx = cell
                source_records.append({
                    "substep": substeps,
                    "cell": [r, cidx],
                    "dt_s": sub_dt,
                    "velocity_before_m_s": velocity0[cell].tolist(),
                    "velocity_after_source_m_s": velocity[cell].tolist(),
                    "gravity_pressure_delta_velocity_m_s": (
                        velocity_g[cell] - velocity0[cell]
                    ).tolist(),
                    "gravity_pressure_x_face_impulse_kg_m_s": [
                        float(cell_mass[cell] * (velocity_g[cell][0] - velocity0[cell][0])),
                        0.0,
                    ],
                    "gravity_pressure_y_face_impulse_kg_m_s": [
                        0.0,
                        float(cell_mass[cell] * (velocity_g[cell][1] - velocity0[cell][1])),
                    ],
                    "basal_friction_cell_impulse_kg_m_s": (
                        cell_mass[cell] * (velocity[cell] - velocity_g[cell])
                    ).tolist(),
                    "source_impulse_kg_m_s": (
                        cell_mass[cell] * (velocity[cell] - velocity0[cell])
                    ).tolist(),
                    "selected_x_face_offset": int(choice_x[cell]),
                    "selected_y_face_offset": int(choice_y[cell]),
                    "selected_face_H_free_gradients": [float(gx[cell]), float(gy[cell])],
                    "selected_face_h_gradients": [float(ghx[cell]), float(ghy[cell])],
                    "dynamic": bool(dynamic[cell]),
                })
        volume = h * weights
        momentum_volume = volume[..., None] * velocity
        x_first = (np.arange(rows)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
        x_second = x_first + 1
        x_face = 0.5 * (
            velocity[..., 0].ravel()[x_first] + velocity[..., 0].ravel()[x_second]
        )
        x_donor = np.where(x_face < 0.0, x_second, x_first)
        x_receiver = np.where(x_face < 0.0, x_first, x_second)
        x_candidate = h.ravel()[x_donor] * np.abs(x_face) * DY * sub_dt
        y_first = (np.arange(rows - 1)[:, None] * cols + np.arange(cols)[None, :]).ravel()
        y_second = y_first + cols
        y_face = 0.5 * (
            velocity[..., 1].ravel()[y_first] + velocity[..., 1].ravel()[y_second]
        )
        y_donor = np.where(y_face < 0.0, y_second, y_first)
        y_receiver = np.where(y_face < 0.0, y_first, y_second)
        y_candidate = h.ravel()[y_donor] * np.abs(y_face) * DX * sub_dt
        donors = np.concatenate((x_donor, y_donor))
        receivers = np.concatenate((x_receiver, y_receiver))
        candidate = np.concatenate((x_candidate, y_candidate))
        face_velocity = np.concatenate((x_face, y_face))
        axes = np.concatenate((np.zeros(x_face.size, dtype=np.int8), np.ones(y_face.size, dtype=np.int8)))
        outgoing = np.zeros(h.size)
        np.add.at(outgoing, donors, candidate)
        donor_volume = volume.ravel()[donors]
        factors = np.ones_like(candidate)
        limited = (outgoing[donors] > donor_volume) & (outgoing[donors] > 0.0)
        factors[limited] = donor_volume[limited] / outgoing[donors][limited]
        amounts = candidate * factors
        actual_out = np.zeros(h.size)
        np.add.at(actual_out, donors, amounts)
        actual_in = np.zeros(h.size)
        np.add.at(actual_in, receivers, amounts)
        cumulative_out += actual_out.reshape(h.shape)
        cumulative_in += actual_in.reshape(h.shape)
        active = volume.ravel() > 0.0
        ratios = np.divide(
            actual_out[active], volume.ravel()[active],
            out=np.zeros(np.count_nonzero(active)), where=volume.ravel()[active] > 0.0,
        )
        max_volume_cfl = max(max_volume_cfl, float(np.max(ratios, initial=0.0)))
        carried = amounts[:, None] * velocity.reshape(-1, 2)[donors]
        positive = amounts > 0.0
        for donor, receiver in zip(donors[positive], receivers[positive]):
            incoming_sources.setdefault(int(receiver), set()).add(int(donor))
        if audit_pair:
            pair_flat = {np.ravel_multi_index(cell, h.shape) for cell in audit_pair}
            for edge in np.flatnonzero(positive):
                donor = int(donors[edge])
                receiver = int(receivers[edge])
                if donor not in pair_flat and receiver not in pair_flat:
                    continue
                ledgers.append({
                    "substep": substeps,
                    "axis": "X" if axes[edge] == 0 else "Y",
                    "donor": list(np.unravel_index(donor, h.shape)),
                    "receiver": list(np.unravel_index(receiver, h.shape)),
                    "face_mass_velocity_m_s": float(face_velocity[edge]),
                    "volume_m3": float(amounts[edge]),
                    "momentum_kg_m_s": (
                        material["density"] * carried[edge]
                    ).tolist(),
                    "donor_limiter": float(factors[edge]),
                })
        flat_volume = volume.ravel().copy()
        flat_momentum = momentum_volume.reshape(-1, 2).copy()
        np.add.at(flat_volume, donors, -amounts)
        np.add.at(flat_volume, receivers, amounts)
        np.add.at(flat_momentum, donors, -carried)
        np.add.at(flat_momentum, receivers, carried)
        flat_volume = np.maximum(flat_volume, 0.0)
        h = (flat_volume / weights.ravel()).reshape(h.shape)
        velocity_after = np.divide(
            flat_momentum,
            flat_volume[:, None],
            out=np.zeros_like(flat_momentum),
            where=flat_volume[:, None] > DRY * weights.ravel()[:, None],
        ).reshape(h.shape + (2,))
        speed_after = np.linalg.norm(velocity_after, axis=-1)
        cap = speed_after > VELOCITY_CAP
        velocity_before_cap = velocity_after.copy()
        velocity_after[cap] *= (VELOCITY_CAP / speed_after[cap])[:, None]
        velocity_cap_events += int(np.count_nonzero(cap))
        velocity_cap_impulse += material["density"] * np.sum(
            h[..., None] * weights[..., None] * (velocity_after - velocity_before_cap),
            axis=(0, 1),
            dtype=np.float64,
        )
        momentum = h[..., None] * velocity_after
        dry_cleanup = h <= DRY
        dry_cleanup_events += int(np.count_nonzero(dry_cleanup & (np.linalg.norm(momentum, axis=-1) > 0.0)))
        dry_cleanup_impulse -= material["density"] * np.sum(
            momentum[dry_cleanup] * weights[dry_cleanup, None], axis=0, dtype=np.float64
        )
        momentum[dry_cleanup] = 0.0
        remaining -= sub_dt
        substeps += 1
    momentum_after = material["density"] * np.sum(
        momentum * weights[..., None], axis=(0, 1), dtype=np.float64
    )
    return {
        "h": h,
        "momentum": momentum,
        "source_impulse": source_impulse,
        "momentum_balance_residual": momentum_after - momentum_before - source_impulse,
        "momentum_balance_residual_after_existing_cap": (
            momentum_after - momentum_before - source_impulse - velocity_cap_impulse
        ),
        "momentum_balance_residual_after_numerical_cleanup": (
            momentum_after - momentum_before - source_impulse
            - velocity_cap_impulse - dry_cleanup_impulse
        ),
        "velocity_cap_events": velocity_cap_events,
        "velocity_cap_impulse_kg_m_s": velocity_cap_impulse,
        "dry_cleanup_events": dry_cleanup_events,
        "dry_cleanup_impulse_kg_m_s": dry_cleanup_impulse,
        "substeps": substeps,
        "max_volume_cfl": max_volume_cfl,
        "max_wave_cfl": max_wave_cfl,
        "ledgers": ledgers,
        "source_records": source_records,
        "incoming_sources": incoming_sources,
        "cumulative_in_m3": cumulative_in,
        "cumulative_out_m3": cumulative_out,
        "source_impulse_cells_kg_m_s": source_impulse_cells,
    }


def _step_device(
    resting: np.ndarray,
    initial_h: np.ndarray,
    initial_momentum: np.ndarray,
    weights: np.ndarray,
    material: dict[str, float],
    dt: float,
) -> dict[str, np.ndarray | int]:
    runtime = WarpRuntime("cuda:0")
    wp = runtime.wp
    shape = initial_h.shape
    size = initial_h.size
    x_edges = shape[0] * (shape[1] - 1)
    edge_count = x_edges + (shape[0] - 1) * shape[1]
    for name, value, dtype in (
        ("resting", resting, wp.float64),
        ("mobile", initial_h, wp.float64),
        ("momentum_x", initial_momentum[..., 0], wp.float64),
        ("momentum_y", initial_momentum[..., 1], wp.float64),
        ("weights", weights, wp.float64),
        ("forcing", np.zeros(shape, dtype=np.int32), wp.int32),
        ("latch", np.zeros(shape, dtype=np.int32), wp.int32),
    ):
        runtime.upload(name, np.asarray(value).ravel(), dtype=dtype)
    for name in ("velocity_x", "velocity_y", "volume", "momentum_volume_x", "momentum_volume_y", "outgoing"):
        runtime.empty(name, size, dtype=wp.float64)
    runtime.empty("donors", edge_count, dtype=wp.int32)
    runtime.empty("receivers", edge_count, dtype=wp.int32)
    runtime.empty("amounts", edge_count, dtype=wp.float64)
    runtime.zeros("export", size, dtype=wp.float64)
    runtime.zeros("flux_export", size, dtype=wp.float64)
    source_impulse = wp.zeros(2, dtype=wp.float64, device=runtime.device)
    crossings = wp.zeros(3, dtype=wp.float64, device=runtime.device)
    kernels = production_kernels(wp)
    source = _experimental_source_kernel(wp)
    remaining = float(dt)
    substeps = 0
    while remaining > 1.0e-14:
        maximum_wave = wp.zeros(1, dtype=wp.float64, device=runtime.device)
        runtime.launch(kernels[0], dim=size, inputs=[
            runtime.arrays["mobile"], runtime.arrays["momentum_x"],
            runtime.arrays["momentum_y"], PRESSURE_COEFFICIENT * GRAVITY,
            DRY, maximum_wave,
        ])
        runtime.synchronize()
        wave = float(np.asarray(maximum_wave.numpy())[0])
        sub_dt = min(
            remaining,
            remaining if wave <= 1.0e-12 else CFL * min(DX, DY) / wave,
        )
        runtime.launch(source, dim=size, inputs=[
            runtime.arrays["resting"], runtime.arrays["mobile"],
            runtime.arrays["momentum_x"], runtime.arrays["momentum_y"],
            runtime.arrays["forcing"], runtime.arrays["weights"],
            runtime.arrays["velocity_x"], runtime.arrays["velocity_y"],
            shape[0], shape[1], DX, DY, sub_dt, GRAVITY, PRESSURE_COEFFICIENT,
            material["mobile_friction"], material["cohesion"],
            material["tan_start"], material["tan_stop"], DRY,
            material["density"], source_impulse,
        ])
        runtime.launch(kernels[2], dim=size, inputs=[
            runtime.arrays["mobile"], runtime.arrays["weights"],
            runtime.arrays["velocity_x"], runtime.arrays["velocity_y"],
            runtime.arrays["volume"], runtime.arrays["momentum_volume_x"],
            runtime.arrays["momentum_volume_y"], runtime.arrays["outgoing"],
        ])
        runtime.launch(kernels[3], dim=edge_count, inputs=[
            runtime.arrays["mobile"], runtime.arrays["velocity_x"],
            runtime.arrays["velocity_y"], runtime.arrays["donors"],
            runtime.arrays["receivers"], runtime.arrays["amounts"],
            runtime.arrays["outgoing"], shape[0], shape[1], DY, DX,
            sub_dt, x_edges,
        ])
        runtime.launch(kernels[4], dim=edge_count, inputs=[
            runtime.arrays["donors"], runtime.arrays["receivers"],
            runtime.arrays["amounts"], runtime.arrays["outgoing"],
            runtime.arrays["volume"], runtime.arrays["velocity_x"],
            runtime.arrays["velocity_y"], runtime.arrays["momentum_volume_x"],
            runtime.arrays["momentum_volume_y"], runtime.arrays["latch"],
            runtime.arrays["export"], runtime.arrays["flux_export"], crossings,
        ])
        runtime.launch(kernels[5], dim=size, inputs=[
            runtime.arrays["weights"], runtime.arrays["volume"],
            runtime.arrays["momentum_volume_x"], runtime.arrays["momentum_volume_y"],
            runtime.arrays["mobile"], runtime.arrays["momentum_x"],
            runtime.arrays["momentum_y"], DRY, VELOCITY_CAP,
        ])
        remaining -= sub_dt
        substeps += 1
    runtime.synchronize()
    return {
        "h": np.asarray(runtime.arrays["mobile"].numpy()).reshape(shape),
        "momentum": np.stack((
            np.asarray(runtime.arrays["momentum_x"].numpy()).reshape(shape),
            np.asarray(runtime.arrays["momentum_y"].numpy()).reshape(shape),
        ), axis=-1),
        "source_impulse": np.asarray(source_impulse.numpy()),
        "substeps": substeps,
    }


def _weights(shape: tuple[int, int]) -> np.ndarray:
    value = np.full(shape, DX * DY)
    value[(0, -1), :] *= 0.5
    value[:, (0, -1)] *= 0.5
    return value


def _jump(field: np.ndarray) -> float:
    return float(max(np.max(np.abs(np.diff(field, axis=0))), np.max(np.abs(np.diff(field, axis=1)))))


def _extrema(
    old: np.ndarray, new: np.ndarray, weights: np.ndarray,
    sources: dict[int, set[int]],
    cumulative_in: np.ndarray,
    cumulative_out: np.ndarray,
    source_impulse_cells: np.ndarray,
) -> dict[str, Any]:
    maximum = old.ravel().copy()
    minimum = old.ravel().copy()
    for receiver, donors in sources.items():
        for donor in donors:
            maximum[receiver] = max(maximum[receiver], old.ravel()[donor])
            minimum[receiver] = min(minimum[receiver], old.ravel()[donor])
    maximum = maximum.reshape(old.shape)
    minimum = minimum.reshape(old.shape)
    max_mask = new > maximum + 5.0e-14
    min_mask = new < minimum - 5.0e-14
    outside = np.where(max_mask, new - maximum, np.where(min_mask, minimum - new, 0.0))
    positive_outside = outside[outside > 0.0]
    large_threshold = float(np.quantile(positive_outside, 0.99)) if positive_outside.size else 0.0
    large_mask = (max_mask | min_mask) & (outside >= large_threshold)
    large_records: list[dict[str, Any]] = []
    for row, col in np.argwhere(large_mask):
        cell = (int(row), int(col))
        net = float(cumulative_in[cell] - cumulative_out[cell])
        delta = float((new[cell] - old[cell]) * weights[cell])
        kind = "MAX" if max_mask[cell] else "MIN"
        expected = (kind == "MAX" and net > 0.0) or (kind == "MIN" and net < 0.0)
        large_records.append({
            "cell": [int(row), int(col)],
            "kind": kind,
            "outside_contributor_envelope_m": float(outside[cell]),
            "cell_volume_change_m3": delta,
            "cumulative_incoming_face_mass_m3": float(cumulative_in[cell]),
            "cumulative_outgoing_face_mass_m3": float(cumulative_out[cell]),
            "net_face_convergence_m3": net,
            "cumulative_face_selected_source_impulse_kg_m_s": source_impulse_cells[cell].tolist(),
            "classification": (
                "FACE_CONSISTENT_CONVERGENCE_OR_DIVERGENCE_SUPPORTED"
                if expected and abs(delta - net) <= 2.0e-13
                else "NOT_EXPLAINED_BY_RECORDED_FACE_BALANCE"
            ),
        })
    return {
        "new_local_max_count": int(np.count_nonzero(max_mask)),
        "new_local_min_count": int(np.count_nonzero(min_mask)),
        "extremum_volume_m3": float(np.sum(new[max_mask | min_mask] * weights[max_mask | min_mask])),
        "volume_above_envelope_m3": float(np.sum(np.maximum(new - maximum, 0.0) * weights)),
        "volume_below_envelope_m3": float(np.sum(np.maximum(minimum - new, 0.0) * weights)),
        "large_extremum_definition": (
            "outside contributor-envelope distance >= the 99th percentile among all "
            "nonzero new extrema; reporting threshold only, not a physics limiter"
        ),
        "large_extremum_threshold_m": large_threshold,
        "large_extrema": large_records,
    }


def _contract_cases(material: dict[str, float]) -> dict[str, dict[str, Any]]:
    shape = (21, 21)
    weights = _weights(shape)
    x = np.arange(shape[1]) * DX
    zero_p = np.zeros(shape + (2,))
    cases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cases["STATIC_FLAT"] = (np.ones(shape), np.full(shape, 0.10), zero_p)
    free = 2.0 - np.tan(np.deg2rad(10.0)) * x[None, :]
    free = np.repeat(free, shape[0], axis=0)
    cases["STATIC_STABLE_SLOPE"] = (free - 0.05, np.full(shape, 0.05), zero_p)
    free_stop = 2.0 - np.tan(np.deg2rad(25.0)) * x[None, :]
    free_stop = np.repeat(free_stop, shape[0], axis=0)
    cases["STATIC_AFTER_YSTOP"] = (free_stop - 0.05, np.full(shape, 0.05), zero_p)
    h_wet = np.zeros(shape)
    h_wet[8:13, 8:13] = 0.10
    p_wet = np.zeros(shape + (2,))
    p_wet[..., 0] = h_wet * 0.5
    cases["WET_DRY"] = (np.ones(shape), h_wet, p_wet)
    results: dict[str, dict[str, Any]] = {}
    for name, (resting, h, momentum) in cases.items():
        cpu = _step_cpu(resting, h, momentum, weights, material, DT)
        device = _step_device(resting, h, momentum, weights, material, DT)
        volume_error = float(np.sum((cpu["h"] - h) * weights))
        static = name != "WET_DRY"
        results[name] = {
            "CPU_static_motion_max_m2_s": float(np.max(np.linalg.norm(cpu["momentum"], axis=-1))),
            "CPU_min_h_m": float(np.min(cpu["h"])),
            "CPU_volume_error_m3": volume_error,
            "DEVICE_min_h_m": float(np.min(device["h"])),
            "CPU_DEVICE_h_max_abs_error_m": float(np.max(np.abs(cpu["h"] - device["h"]))),
            "CPU_DEVICE_momentum_max_abs_error_m2_s": float(np.max(np.abs(cpu["momentum"] - device["momentum"]))),
            "status": "PASS" if (
                (not static or float(np.max(np.abs(cpu["momentum"]))) <= 1.0e-13)
                and float(np.min(cpu["h"])) >= -1.0e-14
                and abs(volume_error) <= 1.0e-13
                and float(np.max(np.abs(cpu["h"] - device["h"]))) <= 2.0e-13
                and float(np.max(np.abs(cpu["momentum"] - device["momentum"]))) <= 2.0e-13
            ) else "FAIL",
        }
    # Same H_free, two discontinuous R/M labelings, zero momentum.
    free = np.full(shape, 1.5)
    h_a = np.full(shape, 0.10)
    h_b = np.where(np.indices(shape)[1] < shape[1] // 2, 0.05, 0.15)
    relabel_entries = []
    for h in (h_a, h_b):
        cpu = _step_cpu(free - h, h, zero_p, weights, material, DT)
        device = _step_device(free - h, h, zero_p, weights, material, DT)
        relabel_entries.append({
            "CPU_max_momentum_m2_s": float(np.max(np.abs(cpu["momentum"]))),
            "CPU_H_free_max_abs_change_m": float(np.max(np.abs((free - h) + cpu["h"] - free))),
            "CPU_DEVICE_h_max_abs_error_m": float(np.max(np.abs(cpu["h"] - device["h"]))),
            "CPU_DEVICE_momentum_max_abs_error_m2_s": float(np.max(np.abs(cpu["momentum"] - device["momentum"]))),
        })
    results["R_M_RELABEL_INVARIANCE"] = {
        "cases": relabel_entries,
        "status": "PASS" if all(
            item["CPU_max_momentum_m2_s"] <= 1.0e-13
            and item["CPU_H_free_max_abs_change_m"] <= 1.0e-13
            and item["CPU_DEVICE_h_max_abs_error_m"] <= 2.0e-13
            and item["CPU_DEVICE_momentum_max_abs_error_m2_s"] <= 2.0e-13
            for item in relabel_entries
        ) else "FAIL",
    }
    return results


def main() -> None:
    data = np.load(BOUNDARY)
    resting = np.asarray(data["PRE_resting"])
    old_h = np.asarray(data["PRE_mobile"])
    old_momentum = np.stack((data["PRE_momentum_x"], data["PRE_momentum_y"]), axis=-1)
    current_h = np.asarray(data["POST_mobile"])
    current_momentum = np.stack((data["POST_momentum_x"], data["POST_momentum_y"]), axis=-1)
    weights = np.asarray(data["PRE_weights"])
    material = _material()
    pair_a, pair_b, _, _ = _strongest_pair(resting + current_h)
    pair = {pair_a, pair_b}
    cpu = _step_cpu(resting, old_h, old_momentum, weights, material, DT, audit_pair=pair)
    device = _step_device(resting, old_h, old_momentum, weights, material, DT)
    old_jump = _jump(resting + old_h)
    current_jump = _jump(resting + current_h)
    experimental_jump = _jump(resting + cpu["h"])
    current_detail = json.loads(
        (ROOT / "outputs/mobile_large_avalanche_causal_audit/mobile_single_step_flux_report.json").read_text()
    )
    current_extrema = {
        "new_local_max_count": current_detail["NEW_LOCAL_MAX_COUNT"],
        "new_local_min_count": current_detail["NEW_LOCAL_MIN_COUNT"],
        "extremum_volume_m3": current_detail["NEW_EXTREMUM_VOLUME_M3"],
        "volume_above_envelope_m3": current_detail["extremum_diagnostics"]["max_excess_above_contributor_envelope_m3"],
        "volume_below_envelope_m3": current_detail["extremum_diagnostics"]["min_deficit_below_contributor_envelope_m3"],
    }
    experimental_extrema = _extrema(
        old_h,
        cpu["h"],
        weights,
        cpu["incoming_sources"],
        cpu["cumulative_in_m3"],
        cpu["cumulative_out_m3"],
        cpu["source_impulse_cells_kg_m_s"],
    )
    current_records = (
        current_detail["extremum_diagnostics"]["new_local_maxima"]
        + current_detail["extremum_diagnostics"]["new_local_minima"]
    )
    current_outside = np.asarray(
        [record["outside_envelope_m"] for record in current_records], dtype=np.float64
    )
    current_large_threshold = float(np.quantile(current_outside, 0.99))
    current_extrema["large_extremum_definition"] = (
        "outside contributor-envelope distance >= the 99th percentile among all "
        "nonzero new extrema; reporting threshold only, not a physics limiter"
    )
    current_extrema["large_extremum_threshold_m"] = current_large_threshold
    current_extrema["large_extrema"] = [
        {
            **record,
            "classification": (
                "MASS_FLUX_RECONSTRUCTED_BUT_SOURCE_NOT_FACE_ATTRIBUTABLE_"
                "BECAUSE_CURRENT_SOURCE_IS_CELL_CENTERED"
            ),
        }
        for record in current_records
        if record["outside_envelope_m"] >= current_large_threshold
    ]
    contracts = _contract_cases(material)
    low = (394, 136)
    downstream = (395, 136)
    experimental_low_sources = [x for x in cpu["source_records"] if tuple(x["cell"]) == low]
    experimental_source_y = float(sum(
        x["gravity_pressure_delta_velocity_m_s"][1] for x in experimental_low_sources
    ))
    current_source_y = float(
        current_detail["compression_front"]["stencil_causality"]
        ["source_delta_velocity_along_low_to_downstream_m_s"]
    )
    experimental_low_downstream = [
        x for x in cpu["ledgers"]
        if tuple(x["donor"]) == low and tuple(x["receiver"]) == downstream
    ]
    initial_volume = float(np.sum(old_h * weights))
    final_volume = float(np.sum(cpu["h"] * weights))
    cpu_device_h = float(np.max(np.abs(cpu["h"] - device["h"])))
    cpu_device_p = float(np.max(np.abs(cpu["momentum"] - device["momentum"])))
    report = {
        "schema": "WELL_BALANCED_MOBILE_AB_REPORT/v1",
        "experimental_only": True,
        "production_defaults_modified": False,
        "frozen_operator_boundary": {
            "path": str(BOUNDARY.relative_to(ROOT)),
            "physical_time_s": 3.75,
            "single_step_dt_s": DT,
        },
        "CONTINUOUS_MODEL_DEFINED": "YES",
        "EFFECTIVE_BASAL_SURFACE_REQUIRED": "YES",
        "equation_classification": {
            "mass_balance": "PAPER_DIRECT_DEPTH_AVERAGED_CONSERVATION_ARCHITECTURE",
            "depth_averaged_momentum": "LITERATURE_INFORMED_REDUCED_ORDER",
            "earth_pressure_coefficient_0p45": "ENGINEERING_CLOSURE_UNCALIBRATED",
            "cohesive_start_stop_gate": "LITERATURE_INFORMED_REDUCED_ORDER_UNCALIBRATED",
            "coulomb_mobile_friction": "LITERATURE_INFORMED_REDUCED_ORDER_UNCALIBRATED",
            "flow_direction_face_source": "EXPERIMENTAL_ENGINEERING_CLOSURE",
        },
        "continuous_model": {
            "mass": "partial_t h + div(h u) = 0",
            "momentum": "partial_t(h u)+div(h u tensor u)=h*g_face(H_free,h,K)-tau_b/rho",
            "H_free": "authoritative physical published surface H_resting+h_mobile",
            "H_resting": "reservoir label field; explicitly not accepted as physical bed",
            "effective_basal_surface": (
                "required for a constitutively complete hydrostatic/well-balanced model, "
                "but not currently represented independently; experiment uses face-local H_free/h states"
            ),
        },
        "STATIC_FLAT": contracts["STATIC_FLAT"]["status"],
        "STATIC_STABLE_SLOPE": contracts["STATIC_STABLE_SLOPE"]["status"],
        "STATIC_AFTER_YSTOP": contracts["STATIC_AFTER_YSTOP"]["status"],
        "R_M_RELABEL_INVARIANCE": contracts["R_M_RELABEL_INVARIANCE"]["status"],
        "WET_DRY_POSITIVITY": contracts["WET_DRY"]["status"],
        "KNOWN_UPHILL_FACE_SOURCE": {
            "CURRENT": {
                "classification": "REINFORCES_UPHILL_MOTION",
                "cumulative_low_cell_delta_v_y_m_s": current_source_y,
            },
            "EXPERIMENTAL": {
                "classification": (
                    "DECELERATES_UPHILL_MOTION" if experimental_source_y < 0.0 else "FAILS_TO_DECELERATE"
                ),
                "cumulative_low_cell_delta_v_y_m_s": experimental_source_y,
                "low_to_downstream_volume_m3": float(sum(x["volume_m3"] for x in experimental_low_downstream)),
            },
        },
        "DELTA_JH_CURRENT_M": current_jump - old_jump,
        "DELTA_JH_EXPERIMENTAL_M": experimental_jump - old_jump,
        "J_H": {"old_m": old_jump, "current_m": current_jump, "experimental_m": experimental_jump},
        "NEW_EXTREMA_CURRENT": current_extrema,
        "NEW_EXTREMA_EXPERIMENTAL": experimental_extrema,
        "MASS_CONSERVATION": "PASS" if abs(final_volume - initial_volume) <= 2.0e-13 else "FAIL",
        "conservation": {
            "initial_volume_m3": initial_volume,
            "final_volume_m3": final_volume,
            "mass_residual_m3": final_volume - initial_volume,
            "momentum_balance_residual_kg_m_s": cpu["momentum_balance_residual"].tolist(),
            "integrated_source_impulse_kg_m_s": cpu["source_impulse"].tolist(),
            "existing_velocity_cap_events": cpu["velocity_cap_events"],
            "existing_velocity_cap_impulse_kg_m_s": cpu["velocity_cap_impulse_kg_m_s"].tolist(),
            "momentum_balance_residual_after_existing_cap_kg_m_s": (
                cpu["momentum_balance_residual_after_existing_cap"].tolist()
            ),
            "dry_threshold_cleanup_events": cpu["dry_cleanup_events"],
            "dry_threshold_cleanup_impulse_kg_m_s": cpu["dry_cleanup_impulse_kg_m_s"].tolist(),
            "momentum_balance_residual_after_numerical_cleanup_kg_m_s": (
                cpu["momentum_balance_residual_after_numerical_cleanup"].tolist()
            ),
            "positivity_min_h_m": float(np.min(cpu["h"])),
            "max_volume_CFL": cpu["max_volume_cfl"],
            "max_wave_CFL": cpu["max_wave_cfl"],
        },
        "CPU_DEVICE_EQUIVALENCE": (
            "PASS" if cpu_device_h <= 2.0e-13 and cpu_device_p <= 2.0e-13
            and all(value["status"] == "PASS" for value in contracts.values())
            else "FAIL"
        ),
        "cpu_device": {
            "h_max_abs_error_m": cpu_device_h,
            "momentum_max_abs_error_m2_s": cpu_device_p,
            "CPU_substeps": cpu["substeps"],
            "DEVICE_substeps": int(device["substeps"]),
            "contracts": contracts,
        },
        "strongest_pair": {
            "cells": [list(pair_a), list(pair_b)],
            "old": {
                "H_free_m": [float((resting + old_h)[pair_a]), float((resting + old_h)[pair_b])],
                "h_mobile_m": [float(old_h[pair_a]), float(old_h[pair_b])],
            },
            "current": {
                "H_free_m": [float((resting + current_h)[pair_a]), float((resting + current_h)[pair_b])],
                "h_mobile_m": [float(current_h[pair_a]), float(current_h[pair_b])],
                "face_fluxes": current_detail["strongest_pair"]["cells"],
            },
            "experimental": {
                "H_free_m": [float((resting + cpu["h"])[pair_a]), float((resting + cpu["h"])[pair_b])],
                "h_mobile_m": [float(cpu["h"][pair_a]), float(cpu["h"][pair_b])],
                "face_mass_and_momentum_fluxes": cpu["ledgers"],
                "source_impulses": cpu["source_records"],
            },
        },
        "PHYSICAL_INTERPRETATION": (
            "The experimental source is face-consistent with the direction of Mobile motion: "
            "the known immediately uphill face produces deceleration rather than the current "
            "opposite-cell-dominated reinforcement. This is an engineering numerical closure, "
            "not a claim that shallow-water hydrostatic reconstruction is a granular constitutive law."
        ),
        "REMAINING_MODEL_AMBIGUITY": (
            "The authoritative model has no independent effective basal/support surface. "
            "H_resting cannot fill that role because conservative R/M relabeling makes it discontinuous. "
            "Therefore the experiment can test face balance and equilibrium contracts but cannot "
            "establish a unique granular hydrostatic reconstruction or calibrated earth-pressure law."
        ),
        "PROMOTION_STATUS": "EXPERIMENTAL_ONLY_NOT_PRODUCTION",
    }
    np.savez_compressed(
        FIELDS_OUT,
        resting=resting,
        pre_mobile=old_h,
        current_mobile=current_h,
        experimental_cpu_mobile=cpu["h"],
        experimental_device_mobile=device["h"],
        pre_momentum=old_momentum,
        current_momentum=current_momentum,
        experimental_cpu_momentum=cpu["momentum"],
        experimental_device_momentum=device["momentum"],
    )
    OUT.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "CONTINUOUS_MODEL_DEFINED", "EFFECTIVE_BASAL_SURFACE_REQUIRED",
            "STATIC_FLAT", "STATIC_STABLE_SLOPE", "R_M_RELABEL_INVARIANCE",
            "WET_DRY_POSITIVITY", "KNOWN_UPHILL_FACE_SOURCE",
            "DELTA_JH_CURRENT_M", "DELTA_JH_EXPERIMENTAL_M",
            "NEW_EXTREMA_CURRENT", "NEW_EXTREMA_EXPERIMENTAL",
            "MASS_CONSERVATION", "CPU_DEVICE_EQUIVALENCE",
            "PHYSICAL_INTERPRETATION", "REMAINING_MODEL_AMBIGUITY",
        )
    }, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
