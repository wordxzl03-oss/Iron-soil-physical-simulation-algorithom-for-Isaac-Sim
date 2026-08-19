from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.experimental.bucket_local_3d import (
    BucketCavityGeometry,
    ParticleVolumeAccounting,
    inverse_transform_points,
    sample_rectangular_lattice,
    transform_points,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_390f_descriptor_contains_known_profile_interior_point():
    cavity = BucketCavityGeometry.from_descriptor(
        ROOT / "configs" / "excavator_390f_bucket_descriptor.json"
    )
    point = np.array([[0.0, -1.75, 0.55]], dtype=np.float64)
    assert cavity.contains_local(point)[0]


def test_real_390f_descriptor_rejects_point_ahead_of_mouth():
    cavity = BucketCavityGeometry.from_descriptor(
        ROOT / "configs" / "excavator_390f_bucket_descriptor.json"
    )
    point = np.array([[0.0, 0.50, 0.25]], dtype=np.float64)
    assert not cavity.contains_local(point)[0]


def test_real_390f_descriptor_rejects_point_outside_sidewall():
    cavity = BucketCavityGeometry.from_descriptor(
        ROOT / "configs" / "excavator_390f_bucket_descriptor.json"
    )
    point = np.array([[2.0, -1.75, 0.55]], dtype=np.float64)
    assert not cavity.contains_local(point)[0]


def test_transform_round_trip_is_machine_precision():
    angle = np.deg2rad(31.0)
    c, s = np.cos(angle), np.sin(angle)
    matrix = np.array(
        [
            [c, -s, 0.0, 1.2],
            [s, c, 0.0, -0.4],
            [0.0, 0.0, 1.0, 2.1],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [-0.3, 0.8, 1.1]],
        dtype=np.float64,
    )
    world = transform_points(matrix, points)
    recovered = inverse_transform_points(matrix, world)
    np.testing.assert_allclose(recovered, points, atol=1.0e-12, rtol=0.0)


def test_rectangular_lattice_count_and_bounds():
    points = sample_rectangular_lattice(
        minimum_m=(-1.0, 0.0, 0.0),
        maximum_m=(1.0, 1.0, 0.5),
        spacing_m=0.1,
    )
    assert points.shape == (20 * 10 * 5, 3)
    assert np.all(points[:, 0] > -1.0)
    assert np.all(points[:, 0] < 1.0)
    assert np.all(points[:, 1] > 0.0)
    assert np.all(points[:, 1] < 1.0)


def test_particle_volume_accounting_is_exact_count_based():
    accounting = ParticleVolumeAccounting(0.001)
    mask = np.array([True, False, True, True])
    assert accounting.volume_from_mask(mask) == 0.003
    assert accounting.volume_from_count(7) == 0.007
