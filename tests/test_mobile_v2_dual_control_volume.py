"""Canonical P0-2B tests for the Mobile V2 discrete-measure contract."""

from __future__ import annotations

import numpy as np
import pytest

from isaac_bulk_pipeline.bulk_state import TerrainVolumeIntegrator
from isaac_bulk_pipeline.experimental.mobile_v2_control_volume import (
    authoritative_mobile_momentum_measure_m4_s,
    authoritative_mobile_volume_m3,
    shared_face_increment,
)
from isaac_bulk_pipeline.terrain import TerrainGrid


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_unequal_area_shared_face_conserves_weighted_mass(direction: float):
    area_i, area_j = 0.0025, 0.000625
    h = np.asarray([0.40, 0.30], dtype=np.float64)
    before = float(area_i * h[0] + area_j * h[1])
    update = shared_face_increment(
        mass_flux_m2_s=direction * 0.017,
        momentum_flux_m3_s2=np.asarray([0.003, -0.002]),
        face_length_m=0.05,
        dt_s=0.01,
        area_i_m2=area_i,
        area_j_m2=area_j,
    )
    h += np.asarray([update.delta_h_i_m, update.delta_h_j_m])
    after = float(area_i * h[0] + area_j * h[1])
    assert after == pytest.approx(before, abs=4.0 * np.finfo(np.float64).eps)
    assert np.min(h) >= 0.0


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_unequal_area_shared_face_conserves_both_weighted_momenta(direction: float):
    areas = np.asarray([0.0025, 0.000625], dtype=np.float64)
    q = np.asarray([[0.12, -0.04], [-0.03, 0.07]], dtype=np.float64)
    before = np.sum(q * areas[:, None], axis=0, dtype=np.float64)
    update = shared_face_increment(
        mass_flux_m2_s=direction * 0.01,
        momentum_flux_m3_s2=direction * np.asarray([0.031, -0.047]),
        face_length_m=0.05,
        dt_s=0.0125,
        area_i_m2=areas[0],
        area_j_m2=areas[1],
    )
    q[0] += update.delta_q_i_m2_s
    q[1] += update.delta_q_j_m2_s
    after = np.sum(q * areas[:, None], axis=0, dtype=np.float64)
    np.testing.assert_allclose(after, before, atol=4.0 * np.finfo(np.float64).eps)


def _triangle_ac_weights(shape=(7, 7), spacing=0.05) -> np.ndarray:
    grid = TerrainGrid(*shape, spacing, spacing, 0.0, 0.0, "/World/Terrain")
    return TerrainVolumeIntegrator.from_grid(grid).vertex_weights_m2


@pytest.mark.parametrize("reverse", [False, True])
def test_real_triangle_ac_interior_edge_face_conserves_weighted_mass(reverse: bool):
    weights = _triangle_ac_weights()
    i, j = ((3, 1), (3, 0)) if not reverse else ((3, 0), (3, 1))
    h = np.full(weights.shape, 0.2, dtype=np.float64)
    before = authoritative_mobile_volume_m3(h, weights)
    update = shared_face_increment(
        mass_flux_m2_s=0.012,
        momentum_flux_m3_s2=np.asarray([0.002, 0.001]),
        face_length_m=0.05,
        dt_s=0.01,
        area_i_m2=weights[i],
        area_j_m2=weights[j],
    )
    h[i] += update.delta_h_i_m
    h[j] += update.delta_h_j_m
    assert authoritative_mobile_volume_m3(h, weights) == pytest.approx(
        before, abs=2.0e-17
    )


def test_real_triangle_ac_corner_adjacent_face_conserves_weighted_mass():
    weights = _triangle_ac_weights()
    h = np.full(weights.shape, 0.2, dtype=np.float64)
    before = authoritative_mobile_volume_m3(h, weights)
    update = shared_face_increment(
        mass_flux_m2_s=-0.008,
        momentum_flux_m3_s2=np.asarray([-0.001, 0.004]),
        face_length_m=0.05,
        dt_s=0.01,
        area_i_m2=weights[0, 1],
        area_j_m2=weights[0, 0],
    )
    h[0, 1] += update.delta_h_i_m
    h[0, 0] += update.delta_h_j_m
    assert authoritative_mobile_volume_m3(h, weights) == pytest.approx(
        before, abs=2.0e-17
    )


def test_random_closed_domain_many_shared_transfers_conserve_weighted_state():
    rng = np.random.default_rng(20260817)
    weights = _triangle_ac_weights(shape=(13, 17))
    h = rng.uniform(0.15, 0.40, size=weights.shape)
    q = rng.uniform(-0.03, 0.03, size=weights.shape + (2,))
    before_volume = authoritative_mobile_volume_m3(h, weights)
    before_momentum = authoritative_mobile_momentum_measure_m4_s(q, weights)
    rows, cols = weights.shape
    for _ in range(5000):
        row = int(rng.integers(rows))
        col = int(rng.integers(cols))
        if rng.random() < 0.5 and col + 1 < cols:
            other = (row, col + 1)
        elif row + 1 < rows:
            other = (row + 1, col)
        elif col > 0:
            other = (row, col - 1)
        else:
            other = (row - 1, col)
        here = (row, col)
        update = shared_face_increment(
            mass_flux_m2_s=float(rng.uniform(-2.0e-4, 2.0e-4)),
            momentum_flux_m3_s2=rng.uniform(-1.0e-4, 1.0e-4, size=2),
            face_length_m=0.05,
            dt_s=1.0e-4,
            area_i_m2=weights[here],
            area_j_m2=weights[other],
        )
        h[here] += update.delta_h_i_m
        h[other] += update.delta_h_j_m
        q[here] += update.delta_q_i_m2_s
        q[other] += update.delta_q_j_m2_s
    assert np.min(h) >= 0.0
    assert authoritative_mobile_volume_m3(h, weights) == pytest.approx(
        before_volume, abs=2.0e-15
    )
    np.testing.assert_allclose(
        authoritative_mobile_momentum_measure_m4_s(q, weights),
        before_momentum,
        atol=2.0e-15,
    )


def test_equal_area_face_is_identical_to_legacy_uniform_grid_increment():
    dx, dy, dt = 0.05, 0.04, 0.007
    area = dx * dy
    mass_flux = -0.023
    momentum_flux = np.asarray([0.017, -0.011])
    update = shared_face_increment(
        mass_flux_m2_s=mass_flux,
        momentum_flux_m3_s2=momentum_flux,
        face_length_m=dy,
        dt_s=dt,
        area_i_m2=area,
        area_j_m2=area,
    )
    roundoff = 2.0 * np.finfo(np.float64).eps
    assert update.delta_h_i_m == pytest.approx(-dt / dx * mass_flux, abs=roundoff)
    assert update.delta_h_j_m == pytest.approx(dt / dx * mass_flux, abs=roundoff)
    np.testing.assert_allclose(
        update.delta_q_i_m2_s, -dt / dx * momentum_flux, rtol=0.0, atol=roundoff
    )
    np.testing.assert_allclose(
        update.delta_q_j_m2_s, dt / dx * momentum_flux, rtol=0.0, atol=roundoff
    )
