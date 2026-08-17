import ast
from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PhaseBNoRuntimeTeleportTests(unittest.TestCase):
    def test_phase_b_vehicle_core_has_no_pose_setter(self):
        forbidden = {"set_world_pose", "set_world_poses", "set_local_pose", "set_default_state"}
        vehicle_root = REPOSITORY_ROOT / "src/isaac_bulk_pipeline/vehicle"
        for path in sorted(vehicle_root.glob("*.py")):
            with self.subTest(path=path.name):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                calls = {
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                }
                self.assertTrue(forbidden.isdisjoint(calls), f"forbidden calls in {path}: {forbidden & calls}")

    def test_manual_runtime_has_sparse_actions_real_effort_install_and_no_pose_setter(self):
        path = REPOSITORY_ROOT / "isaac_loader/phase_b_manual_demo.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("set_world_pose", calls)
        self.assertNotIn("set_world_poses", calls)
        self.assertGreaterEqual(calls.count("set_max_efforts"), 2)
        self.assertGreaterEqual(calls.count("apply_action"), 2)
        source = path.read_text(encoding="utf-8")
        self.assertIn("joint_indices=", source)
        self.assertIn('"isaac_interactive_run"', source)
        self.assertIn('"episode_0002"', source)
        self.assertIn('"H_initial.npy"', source)
        self.assertIn("DynamicMeshAdapter", source)
        self.assertIn("TerrainVolumeIntegrator", source)
        self.assertIn("SurfaceTopology.TRIANGLE_A_C", source)
        self.assertIn("collision_enabled=False", source)
        self.assertIn('"visual_no_collision"', source)
        self.assertIn('"ground_and_h0_share_visual_material"', source)
        self.assertEqual(source.count("loader.get_articulation_controller()"), 1)
        self.assertIn(
            "measured_joint_velocities=power_cap_input_velocities", source
        )
        self.assertIn(
            '"measured_or_target_worst_case_cap_used_every_frame"', source
        )
        self.assertIn(
            '"post_step_measured_speed_power_bound_within_limit"', source
        )
        self.assertIn('"wheel_power_limit_evidence"', source)
        self.assertNotIn('"applied_wheel_effort_limit_nm"', source)
        self.assertNotIn('"wheel_power_cap_activated"', source)
        self.assertIn("wheel_power_limit_w", (
            REPOSITORY_ROOT / "configs/phase_b_vehicle.yaml"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
