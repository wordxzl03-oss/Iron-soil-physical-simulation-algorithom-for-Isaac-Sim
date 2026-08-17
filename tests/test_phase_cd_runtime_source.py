import ast
from pathlib import Path
import unittest

from isaac_loader.phase_cd_slope_drive_runtime import _stats


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PhaseCDRuntimeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            REPOSITORY_ROOT
            / "isaac_loader"
            / "phase_cd_slope_drive_runtime.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(cls.path))

    def test_runtime_low_level_step_always_receives_measured_velocities(self) -> None:
        low_level_steps = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "low_level"
            and node.func.attr == "step"
        ]
        self.assertEqual(len(low_level_steps), 1)
        keyword_names = {item.arg for item in low_level_steps[0].keywords}
        self.assertIn("measured_joint_velocities", keyword_names)

    def test_runtime_has_no_pose_setter(self) -> None:
        forbidden = {
            "set_world_pose",
            "set_world_poses",
            "set_local_pose",
            "set_default_state",
        }
        called_attributes = {
            node.func.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertEqual(called_attributes & forbidden, set())
        self.assertIn(
            '"normal_loop_pose_setter_zero": True',
            self.source,
        )
        self.assertIn(
            '"initialization_pose_setter_zero_usd_transform_used": True',
            self.source,
        )

    def test_runtime_persists_and_accepts_power_cap_semantics(self) -> None:
        required_fragments = (
            '"MEASURED_OR_TARGET_WORST_CASE"',
            '"TARGET_JOINT_VELOCITY_FALLBACK"',
            '"power_cap_basis_code"',
            '"power_cap_input_full_joint_velocity_rad_s"',
            '"wheel_installed_effort_limit_nm"',
            '"wheel_installed_effort_post_step_measured_speed_power_bound_w"',
            '"wheel_power_min_guard_rad_s"',
            '"minimum_speed_guard_rad_s"',
            '"pipeline_python_source_tree_sha256"',
            '"phase_c_runtime_source_sha256"',
            '"target_speed_power_cap_fallback_frames_zero"',
            '"installed_effort_post_step_measured_speed_power_bound_within_160kw_limit"',
            '"measured_speed_runtime_power_cap_verified_for_all"',
            '"configured_minimum_speed_power_guard_used_every_frame"',
            '"post_step_bound_within_configured_limit"',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)
        self.assertIn(
            "measured solver tau*omega and does not claim measured input power",
            self.source,
        )

    def test_public_contact_loads_and_points_are_retained(self) -> None:
        called_attributes = {
            node.func.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }
        self.assertIn("get_contact_force_matrix", called_attributes)
        self.assertIn("get_contact_force_data", called_attributes)
        required_fragments = (
            '"public_contact_load_available"',
            '"public_contact_points_available"',
            '"wheel_contact_point_world_m"',
            '"wheel_body_centers"',
            '"wheel_body_centers_stored_separately": True',
            '"no_synthetic_or_estimated_loads": True',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_full_step_npz_contains_runtime_and_power_evidence(self) -> None:
        savez_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "savez_compressed"
        ]
        self.assertEqual(len(savez_calls), 1)
        saved_fields = {item.arg for item in savez_calls[0].keywords}
        required_fields = {
            "timestamps_s",
            "root_position_world_m",
            "joint_velocity_rad_s",
            "applied_effort_command_nm",
            "measured_solver_joint_effort_nm",
            "wheel_contact_point_world_m",
            "power_cap_basis_code",
            "wheel_power_min_guard_rad_s",
            "power_cap_input_full_joint_velocity_rad_s",
            "wheel_installed_effort_limit_nm",
            "wheel_post_step_measured_omega_rad_s",
            "wheel_installed_effort_post_step_measured_speed_power_bound_w",
        }
        self.assertTrue(required_fields.issubset(saved_fields))
        self.assertIn('"all_steps_persisted": all_steps_persisted', self.source)
        self.assertIn('"all_required_fields_present": persisted_fields_present', self.source)

    def test_runtime_references_all_authoritative_configs_and_hashes(self) -> None:
        required_fragments = (
            '"phase_b_vehicle.yaml"',
            '"phase_c_contact.yaml"',
            '"phase_d_telemetry.yaml"',
            '"runtime_script_sha256"',
            '"pipeline_python_source_tree_sha256"',
            '"phase_c_runtime_source_sha256"',
            '"loader_usd_sha256"',
            '"vehicle_config_sha256"',
            '"contact_config_sha256"',
            '"telemetry_config_sha256"',
            '"scenario_fingerprint_sha256"',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.source)

    def test_clearance_statistics_expose_minimum(self) -> None:
        statistics = _stats([0.25, -0.03, 0.10])
        self.assertEqual(statistics["count"], 3)
        self.assertEqual(statistics["min"], -0.03)
        self.assertEqual(_stats([])["min"], None)


if __name__ == "__main__":
    unittest.main()
