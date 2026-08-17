from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isaac_bulk_pipeline.dataset import (
    EpisodeDatasetReader,
    EpisodeDatasetWriter,
    EpisodeRecord,
    OfflineScalabilityBenchmark,
    SoilForceSensitivityStudy,
)


def record() -> EpisodeRecord:
    count = 3; shape = (8, 9)
    arrays = {
        "terrain/H_before_m": np.ones(shape),
        "terrain/H_after_interaction_m": np.ones(shape) * 0.9,
        "terrain/H_after_deposition_m": np.ones(shape) * 0.95,
        "terrain/H_stable_m": np.ones(shape) * 0.95,
        "vehicle/timestamp_s": np.array([0.0, 0.1, 0.2]),
        "vehicle/pose_world_m_quat": np.zeros((count, 7)),
        "vehicle/joint_position_rad": np.zeros((count, 6)),
        "vehicle/joint_velocity_rad_s": np.zeros((count, 6)),
        "vehicle/joint_torque_nm": np.zeros((count, 6)),
        "tool/pose_world_m_quat": np.zeros((count, 7)),
        "tool/velocity_world_m_s": np.zeros((count, 6)),
        "force/soil_force_world_n": np.zeros((count, 3)),
        "force/application_point_world_m": np.zeros((count, 3)),
        "planner/planned_path_xy_yaw_articulation": np.zeros((4, 4)),
    }
    sections = {
        "material": {"mobilized_volume_m3": 0.2, "payload_volume_m3": 0.1, "estimated_payload_mass_kg": 210.0, "spill_volume_m3": 0.0, "deposited_volume_m3": 0.1, "airborne_volume_m3": 0.0, "outflow_volume_m3": 0.0},
        "performance": {"time_s": 0.2, "distance_m": 0.1, "total_mechanical_energy_j": 12.0, "work_j": 12.0, "max_slope_rad": 0.1, "max_roll_rad": 0.01, "max_pitch_rad": 0.02},
        "planner": {"goal": {"kind": "target_volume", "value": 1.0}, "candidate_attacks": [], "selected_candidate": None, "cost_breakdown": {}},
        "mass_ledger": {"absolute_error_m3": 0.0, "relative_error": 0.0},
        "metadata": {"grid": {"semantic": "vertex_field", "axis_order": "yx", "topology": "triangle_a_c"}, "units": "SI", "tool_geometry": "fixture", "vehicle_model": "fixture", "material_scenario": "UNCALIBRATED_SCENARIO_PARAMETER", "solver_config": {}, "random_seed": 7, "isaac_version": "4.5.0", "git_commit": "UNAVAILABLE"},
    }
    return EpisodeRecord("episode-1", "action-1", arrays, sections)


class TestPhaseKDatasetBenchmark(unittest.TestCase):
    def test_atomic_roundtrip_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = EpisodeDatasetWriter().write(record(), directory)
            self.assertTrue(path.is_file())
            restored = EpisodeDatasetReader().read(directory)
            np.testing.assert_array_equal(restored.arrays["terrain/H_stable_m"], record().arrays["terrain/H_stable_m"])
            manifest = json.loads(path.read_text())
            self.assertEqual(len(manifest["arrays_sha256"]), 64)

    def test_rejects_missing_schema_data(self) -> None:
        valid = record()
        arrays = dict(valid.arrays); arrays.pop("force/soil_force_world_n")
        with self.assertRaises(ValueError):
            EpisodeRecord("e", "a", arrays, valid.sections)

    def test_rejects_unsynchronized_samples(self) -> None:
        valid = record(); arrays = dict(valid.arrays)
        arrays["tool/velocity_world_m_s"] = np.zeros((2, 6))
        with self.assertRaises(ValueError):
            EpisodeRecord("e", "a", arrays, valid.sections)

    def test_resolution_and_tool_matrix_is_configured(self) -> None:
        benchmark = OfflineScalabilityBenchmark(repeats=1)
        self.assertEqual(set(benchmark.TOOL_WIDTHS), {"small", "medium", "large"})
        result = benchmark.run_case(128, "small")
        self.assertEqual(result.case.resolution, 128)
        self.assertLess(abs(result.mobile_balance_error_m3), 1e-9)
        self.assertEqual(set(result.statistics()), {"planner_view", "attack_candidates", "mobile_layer"})
        self.assertGreater(result.candidate_count, 0)

    def test_soil_force_sensitivity_trends_are_monotonic(self) -> None:
        study = SoilForceSensitivityStudy()
        for samples in study.run().values():
            self.assertTrue(study.monotonic(samples))


if __name__ == "__main__":
    unittest.main()
