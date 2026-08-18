import numpy as np

from isaac_bulk_pipeline.runtime.curl_scoop_trajectory import (
    RealisticCurlScoopTrajectory,
)


def trajectory():
    return RealisticCurlScoopTrajectory(
        np.deg2rad([0.0, -10.0, 16.0, -60.0]),
        np.deg2rad([0.0, -11.0, 20.0, -30.0]),
        np.deg2rad([0.0, -12.0, 24.0, 20.0]),
        np.deg2rad([0.0, -25.0, 24.0, 42.0]),
        target_cut_depth_m=0.20,
        minimum_engaged_fraction=0.70,
    )


def test_cut_is_feedback_driven_and_does_not_self_breakout():
    item = trajectory()
    entry = np.deg2rad([0.0, -10.0, 16.0, -60.0])
    item.reset(entry)
    sample = None
    for i in range(180):
        sample = item.sample_cut(
            1.0 / 60.0,
            mean_depth_m=0.20,
            engaged_fraction=0.90,
            cutting_distance_m=0.5 * i / 179.0,
            payload_volume_m3=0.005,
        )
    assert sample is not None
    assert not sample.complete
    assert sample.stage in {
        "B_ENGAGED_INITIAL_CUT",
        "C_COORDINATED_DRAG_AND_CURL",
    }
    assert sample.target_rad[2] > entry[2]
    assert sample.target_rad[3] > entry[3]
    # CUT never jumps to the authored breakout boom target on an internal clock.
    assert sample.target_rad[1] > np.deg2rad(-25.0) + 1.0e-6


def test_breakout_closes_bucket_before_substantial_lift():
    item = trajectory()
    entry = np.deg2rad([0.0, -12.0, 24.0, 20.0])
    item.begin_breakout(entry)
    first = item.sample_breakout(1.0 / 60.0)
    assert first.stage == "D_CLOSE_CAPTURE_BEFORE_LIFT"
    np.testing.assert_allclose(first.target_rad[1], entry[1], atol=1.0e-15)

    sample = first
    for _ in range(1200):
        sample = item.sample_breakout(1.0 / 60.0)
        if sample.complete:
            break
    assert sample.complete
    np.testing.assert_allclose(
        sample.target_rad, np.deg2rad([0.0, -25.0, 24.0, 42.0]), atol=np.deg2rad(0.5)
    )


def test_reset_starts_from_measured_entry_without_target_jump_when_depth_is_on_target():
    item = trajectory()
    entry = np.deg2rad([1.4, -5.5, 16.0, -60.0])
    item.reset(entry)
    first = item.sample_cut(
        1.0e-9,
        mean_depth_m=0.20,
        engaged_fraction=0.0,
        cutting_distance_m=0.0,
        payload_volume_m3=0.0,
    )
    np.testing.assert_allclose(first.target_rad, entry, atol=1.0e-15)
