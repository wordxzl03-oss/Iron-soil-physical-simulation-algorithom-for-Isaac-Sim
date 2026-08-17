from pathlib import Path
from types import SimpleNamespace

import numpy as np

from isaac_bulk_pipeline.bulk_state import MaterialScenario
from isaac_bulk_pipeline.runtime import EarthmovingPhysicsCore, SoilForceMode
from isaac_bulk_pipeline.terrain import TerrainGrid
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


ROOT = Path(__file__).resolve().parents[1]


def _core(shape=(129, 129)):
    grid = TerrainGrid(shape[1], shape[0], 0.05, 0.05, 0.0, 0.0, "/Terrain")
    descriptor = ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(ROOT / "configs/excavator_390f_real_bucket.yaml")
    )
    material = MaterialScenario("test", 1370.0, 29.8, 800.0, 0.5, 38.0, 30.0, 0.35)
    return EarthmovingPhysicsCore(
        grid=grid,
        descriptor=descriptor,
        material=material,
        initial_heightmap_m=np.ones(shape),
        tile_size=32,
    )


def test_track_soil_core_commit_is_conservative_and_spatial():
    core = _core()
    before = core.manager.ledger_snapshot()
    left = np.zeros(core.grid.shape, dtype=bool)
    right = np.zeros(core.grid.shape, dtype=bool)
    left[50:70, 25:55] = True
    right[50:70, 75:105] = True
    result = core.apply_track_soil(
        left_footprint_mask=left,
        right_footprint_mask=right,
        left_track_velocity_xy_m_s=np.array([1.0, 0.0]),
        right_track_velocity_xy_m_s=np.array([1.0, 0.0]),
        base_velocity_xy_m_s=np.zeros(2),
        dt_s=1.0 / 60.0,
    )
    after = core.manager.ledger_snapshot()
    assert result.resting_to_mobile_volume_m3 > 0.0
    assert result.active_bbox_grid[1] < 25
    assert result.active_bbox_grid[3] > 105
    assert after.transfer_count == before.transfer_count + 1
    assert after.balance.absolute_volume_error_m3 <= 1.0e-10
    assert np.all(core.state.H_resting_m[left | right] < 1.0)
    assert core.integrator.integrate(core.state.mobile_height_m) > 0.0


def test_track_soil_reset_all_restores_ruts_and_model_reference():
    from isaac_bulk_pipeline.runtime import ResetLevel

    core = _core((65, 65))
    mask = np.zeros(core.grid.shape, dtype=bool)
    mask[20:40, 20:45] = True
    core.apply_track_soil(
        left_footprint_mask=mask,
        right_footprint_mask=np.zeros_like(mask),
        left_track_velocity_xy_m_s=np.array([0.8, 0.0]),
        right_track_velocity_xy_m_s=np.zeros(2),
        base_velocity_xy_m_s=np.zeros(2),
        dt_s=1.0 / 60.0,
    )
    core.reset(ResetLevel.ALL)
    np.testing.assert_array_equal(core.state.H_resting_m, np.ones(core.grid.shape))
    np.testing.assert_array_equal(core.state.mobile_height_m, np.zeros(core.grid.shape))
    assert core.manager.ledger_snapshot().balance.absolute_volume_error_m3 == 0.0


def test_zero_volume_dump_attempt_does_not_latch_release_phase():
    core = _core((33, 33))
    descriptor = core.descriptor
    pose = np.eye(4)

    def transform(points):
        points = np.asarray(points, dtype=np.float64)
        return points.copy()

    tool = ToolState(
        timestamp=0.0,
        pose_world=pose,
        pose_terrain=pose,
        cutting_edge_terrain=transform(descriptor.cutting_edge_local),
        bottom_profile_terrain=transform(descriptor.bottom_profile_local),
        left_boundary_terrain=transform(descriptor.left_boundary_local),
        right_boundary_terrain=transform(descriptor.right_boundary_local),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )

    class RetryDump:
        def __init__(self):
            self.release_calls = 0
            self.advance_kwargs = []

        def release(self, manager, tool_state, descriptor, *, target):
            self.release_calls += 1
            released = 0.0 if self.release_calls == 1 else 0.02
            payload = SimpleNamespace(volume_m3=0.05 - released)
            state = SimpleNamespace(payload=payload)
            return SimpleNamespace(
                released_volume_m3=released,
                state=state,
            )

        def advance_airborne(self, *args, **kwargs):
            self.advance_kwargs.append(dict(kwargs))
            return None

    dump = RetryDump()
    core.dump_operator = dump
    core.initialize_tool(tool)
    kwargs = dict(
        phase="dump_spill",
        cycle=1,
        dt_s=1.0 / 60.0,
        soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
    )
    first = core.step(tool, **kwargs)
    second = core.step(tool, **kwargs)
    third = core.step(tool, **kwargs)

    assert first.dump_release.released_volume_m3 == 0.0
    assert second.dump_release.released_volume_m3 == 0.02
    assert third.dump_release is None
    assert dump.release_calls == 2

    # A phase boundary is metadata only.  It must never smuggle the old
    # convergence-to-equilibrium callback into the interactive physics step.
    assert not hasattr(core, "_relax_resting")
    ending = core.step(
        tool,
        phase="deposition",
        cycle=1,
        dt_s=1.0 / 60.0,
        soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        phase_ending=True,
    )
    assert ending.terrain_settled
    assert all("resting_relaxation" not in item for item in dump.advance_kwargs)


def test_dump_event_reference_is_pre_release_authoritative_surface():
    core = _core((33, 33))
    mask = np.zeros(core.grid.shape, dtype=bool)
    mask[12:18, 12:18] = True
    core.apply_track_soil(
        left_footprint_mask=mask,
        right_footprint_mask=np.zeros_like(mask),
        left_track_velocity_xy_m_s=np.array([0.8, 0.0]),
        right_track_velocity_xy_m_s=np.zeros(2),
        base_velocity_xy_m_s=np.zeros(2),
        dt_s=1.0 / 60.0,
    )
    expected = np.array(core.state.H_resting_m, copy=True)
    descriptor = core.descriptor
    pose = np.eye(4)
    tool = ToolState(
        timestamp=0.0,
        pose_world=pose,
        pose_terrain=pose,
        cutting_edge_terrain=np.array(descriptor.cutting_edge_local, copy=True),
        bottom_profile_terrain=np.array(descriptor.bottom_profile_local, copy=True),
        left_boundary_terrain=np.array(descriptor.left_boundary_local, copy=True),
        right_boundary_terrain=np.array(descriptor.right_boundary_local, copy=True),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    core.initialize_tool(tool)
    core.step(
        tool,
        phase="dump_spill",
        cycle=1,
        dt_s=1.0 / 60.0,
        soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
    )
    np.testing.assert_array_equal(core.slope_solver._reference_height, expected)
