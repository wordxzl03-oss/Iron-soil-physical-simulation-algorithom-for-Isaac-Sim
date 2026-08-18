import numpy as np

from isaac_bulk_pipeline.runtime.curl_scoop_trajectory import (
    RealisticCurlScoopTrajectory,
)


def trajectory():
    return RealisticCurlScoopTrajectory(
        np.deg2rad([0.0, -10.0, 16.0, -60.0]),
        np.deg2rad([0.0, -25.0, 24.0, 42.0]),
    )


def test_five_physical_stages_end_at_existing_breakout():
    item = trajectory()
    assert [stage.name[0] for stage in item.stages] == list("ABCDE")
    assert item.duration_s < 12.0
    sample = None
    for _ in range(int(np.ceil(item.duration_s * 60.0)) + 1):
        sample = item.sample(1.0 / 60.0)
    assert sample.complete
    np.testing.assert_allclose(sample.target_rad, np.deg2rad([0, -25, 24, 42]))


def test_schedule_is_continuous_and_has_coordinated_retract_curl_boom():
    item = trajectory()
    samples = [item.sample(1.0 / 600.0).target_rad for _ in range(int(item.duration_s * 600))]
    values = np.asarray(samples)
    velocity = np.diff(values, axis=0) * 600.0
    assert np.max(np.abs(velocity), axis=0)[1] <= 0.45 + 1e-5
    assert np.max(np.abs(velocity), axis=0)[2] <= 0.55 + 1e-5
    assert np.max(np.abs(velocity), axis=0)[3] <= 0.70 + 1e-5
    coordinated = (velocity[:, 1] < 0.0) & (velocity[:, 2] > 0.0) & (velocity[:, 3] > 0.0)
    assert np.count_nonzero(coordinated) > len(velocity) // 2


def test_reset_starts_exactly_from_measured_entry_without_target_jump():
    item = trajectory()
    entry = np.deg2rad([1.4, -5.5, 16.0, -60.0])
    item.reset(entry)
    first = item.sample(1.0e-9)
    np.testing.assert_allclose(first.target_rad, entry, atol=1.0e-15)
