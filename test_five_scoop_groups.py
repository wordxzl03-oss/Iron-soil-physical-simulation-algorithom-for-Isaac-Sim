from pathlib import Path

import numpy as np

from generate_five_scoop_groups import (
    loader_mechanism_points,
    sample_repeated_trajectories,
    simulate_group,
)
from render_obj_loader_scooping import (
    _prepare_model,
    infer_landmarks,
    load_obj_parts,
)


def test_group_has_five_sequential_capacity_limited_scoops():
    group = simulate_group(1, 12345, resolution=0.5)
    assert len(group["passes"]) == 5
    assert group["frames"].shape[0] == 11
    assert np.all(group["initial"] >= 0.0)
    assert np.all(group["final"] >= 0.0)
    assert not np.array_equal(group["initial"], group["final"])
    for scoop in group["passes"]:
        assert 0.0 < scoop["loaded_volume_m3"] <= 3.0 + 1e-6
        assert 0.0 < scoop["fill_factor"] <= 1.0 + 1e-6
        assert scoop["impact_speed_m_s"] > 0.0
        assert scoop["peak_resistance_n"] > 0.0


def test_visual_link_lengths_are_constant_during_curl_and_lift():
    raw = load_obj_parts(Path("wheel_buck.obj"))
    _, points = _prepare_model(raw, infer_landmarks(raw))
    lengths = []
    for boom, curl in ((-5.0, 0.0), (10.0, 24.0), (28.0, 47.0)):
        root, pin, edge = loader_mechanism_points(
            points["root_pin"], points["bucket_pin"], points["cutting_edge"],
            -5.0, boom, curl,
        )
        lengths.append((
            np.linalg.norm(pin - root),
            np.linalg.norm(edge - pin),
        ))
    np.testing.assert_allclose(
        lengths, np.tile(lengths[0], (len(lengths), 1)), atol=1e-10
    )


def test_hundred_scoop_trajectory_cycles_cover_five_lanes():
    trajectories = sample_repeated_trajectories(100, 2.7, seed=77)
    assert len(trajectories) == 100
    assert [item["pass"] for item in trajectories] == list(range(1, 101))
    for cycle_start in range(0, 100, 5):
        offsets = np.sort([
            item["lateral_offset_m"]
            for item in trajectories[cycle_start:cycle_start + 5]
        ])
        assert np.min(np.diff(offsets)) > 1.0
