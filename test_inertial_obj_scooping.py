from pathlib import Path

import numpy as np

from build_simple_wheel_loader_obj import build
from render_obj_loader_scooping import load_obj_parts
from simulate_inertial_obj_scooping import simulate_inertial_entry


def test_simple_loader_obj_has_vehicle_and_working_equipment(tmp_path: Path):
    output = tmp_path / "simple_loader.obj"
    build(Path("wheel_buck.obj"), output)
    parts = load_obj_parts(output)
    assert {
        "simple_chassis",
        "simple_cab",
        "wheel_front_left",
        "wheel_front_right",
        "wheel_rear_left",
        "wheel_rear_right",
        "loader_boom",
        "loader_bucket",
    } == set(parts)
    chassis = parts["simple_chassis"].vertices
    assert np.ptp(chassis[:, 0]) == 2.6
    assert np.ptp(chassis[:, 1]) == 3.85
    for name in (part for part in parts if part.startswith("wheel_")):
        wheel = parts[name].vertices
        assert 1.40 < np.ptp(wheel[:, 1]) < 1.45
        assert np.min(wheel[:, 2]) >= 0.0


def test_inertial_entry_accelerates_then_slows_under_excavation():
    data, states = simulate_inertial_entry(
        cutting_edge_offset_y=3.0,
        entry_y=0.0,
    )
    contact = np.flatnonzero(data[:, 4] > 0.0)
    assert len(contact) > 10
    contact_index = int(contact[0])
    assert data[contact_index, 3] > 1.7
    assert np.max(data[contact_index:, 9]) > 150_000.0
    assert data[-1, 3] < 0.2
    assert 1.0 < data[-1, 4] < 2.5
    assert np.all(np.diff(data[:, 1]) >= -1e-9)
    assert len(states) == len(data)
