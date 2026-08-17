from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import earthmoving_soil
from earthmoving_soil import (
    MaterialConfig, PhysicsDiagnostics, RLFeedbackAccumulator, SoilConfig,
    SoilPhysics, ToolGeometry, ToolState, TrackGeometry, TrackState,
)
from earthmoving_soil.gpu import probe_warp
from earthmoving_soil.isaac import IsaacSoilAdapter
from earthmoving_soil.terrain import TerrainGrid


def make_material():
    return MaterialConfig(
        profile="TEST_UNCALIBRATED", bulk_density_kg_m3=1370.0,
        internal_friction_angle_deg=29.8, cohesion_pa=800.0,
        tool_friction_coefficient=0.6847142903416485,
        start_angle_deg=38.0, stop_angle_deg=30.0,
        mobile_friction_coefficient=0.35,
    )


def make_soil(with_tracks=False):
    grid = TerrainGrid(41, 41, 0.05, 0.05, -1.0, -1.0, "/World/Terrain")
    config = SoilConfig(grid=grid, material=make_material(), physics_dt_s=1/60)
    soil = SoilPhysics(config, np.full(grid.shape, 0.25))
    soil.register_tool("bucket", ToolGeometry.parameterized_bucket(
        width_m=0.4, mouth_depth_m=0.25, rear_height_m=0.2, capacity_m3=0.02,
    ))
    pose = np.eye(4); pose[:3, 3] = [0.0, 0.0, 1.0]
    soil.set_tool_state("bucket", ToolState(pose, np.zeros(3), np.zeros(3), 0.0))
    if with_tracks:
        for track_id, y in (("left_track", 0.3), ("right_track", -0.3)):
            soil.register_track(track_id, TrackGeometry(0.5, 0.2))
            track_pose = np.eye(4); track_pose[:3, 3] = [0.0, y, 0.25]
            soil.set_track_state(track_id, TrackState(
                track_pose, np.zeros(3), np.zeros(3), 0.1, 0.0
            ))
    return soil


def test_package_import_and_public_surface():
    assert earthmoving_soil.__version__ == "0.3.0-alpha.1"
    assert "SoilPhysics" in earthmoving_soil.__all__


def test_config_loading():
    root = Path(__file__).parents[1]
    material = MaterialConfig.from_yaml(root / "configs/materials/iron_ore_reference.yaml")
    assert material.calibration_status == "NOT_YET_PHYSICALLY_CALIBRATED"
    assert len(material.config_hash) == 64
    grid = TerrainGrid(41, 41, .05, .05, -1, -1, "/World/Terrain")
    solver = SoilConfig.from_yaml(
        root / "configs/solver/default.yaml", grid=grid, material=material
    )
    assert solver.physics_dt_s == 1/60
    assert solver.track_soil.parameter_status.endswith("NOT_SITE_CALIBRATED")


def test_registration_deterministic_step_wrench_payload_and_ledger():
    first = make_soil(with_tracks=True)
    second = make_soil(with_tracks=True)
    a = first.step(1/60)
    b = second.step(1/60)
    assert set(a.tool_wrench) == {"bucket"}
    assert a.tool_wrench["bucket"].force_world.shape == (3,)
    assert set(a.track_wrench) == {"left_track", "right_track"}
    assert a.track_wrench["left_track"] is None  # explicitly unavailable, not fake zero
    assert a.payload_volume_m3 >= 0.0
    assert a.payload_mass_kg == a.payload_volume_m3 * 1370.0
    assert abs(a.mass_balance_error_m3) <= 1e-8
    assert np.allclose(a.rl_feedback.local_heightmap_patch_m,
                       b.rl_feedback.local_heightmap_patch_m)
    assert a.payload_volume_m3 == b.payload_volume_m3


def test_rl_feedback_aggregation():
    soil = make_soil()
    result = soil.step(1/60)
    accumulator = RLFeedbackAccumulator()
    accumulator.add(result.rl_feedback, 1/60,
                    tool_linear_velocity_world=np.zeros(3),
                    tool_angular_velocity_world=np.zeros(3))
    feedback = accumulator.emit(result.rl_feedback)
    assert feedback.bucket_impulse_since_last_rl_step_ns.shape == (3,)
    assert feedback.bucket_force_peak_since_last_rl_step_n >= 0.0
    assert feedback.soil_work_delta_j == 0.0


def test_diagnostics_version_metadata():
    result = make_soil().step(1/60)
    diagnostics = result.diagnostics
    assert isinstance(diagnostics, PhysicsDiagnostics)
    assert diagnostics.physics_core_version == "EARTHMOVING_PHYSICS_CORE_V3"
    assert diagnostics.flow_arrest_closure == "FLOW_ARREST_CLOSURE_NOT_YET_DEMONSTRATED"


def test_no_vehicle_controller_or_simulationapp_ownership():
    source = inspect.getsource(IsaacSoilAdapter)
    for forbidden in ("SimulationApp(", "world.step(", "set_joint", "set_world_pose("):
        assert forbidden not in source
    assert "consume_dirty_surface_tiles" in source
    assert "update_tile_samples" in source


def test_cpu_gpu_capability_detection_is_explicit():
    status = probe_warp()
    assert hasattr(status, "available")
