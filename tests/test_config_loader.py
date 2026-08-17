import tempfile
import unittest
from pathlib import Path

from isaac_bulk_pipeline.config import load_config


class ConfigLoaderTests(unittest.TestCase):
    def test_repository_resolution_configs_load_without_source_changes(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        expected_sizes = {
            "terrain_128.yaml": 128,
            "terrain_256.yaml": 256,
            "terrain_512.yaml": 512,
        }
        for filename, size in expected_sizes.items():
            with self.subTest(filename=filename):
                config = load_config(repository_root / "configs" / filename)
                self.assertEqual(config.terrain.to_grid().shape, (size, size))

        large = load_config(repository_root / "configs" / "project_25m.yaml")
        self.assertEqual(large.terrain.to_grid().shape, (701, 701))
        self.assertEqual(large.terrain.source_axis_order, "xy")
        self.assertEqual(large.terrain.dx_m, 0.05)
        self.assertEqual(large.solver.max_iterations, 15000)
        self.assertEqual(large.solver.tolerance, 2.0e-3)
        self.assertFalse(large.solver.sequence_enabled)
        self.assertEqual(large.solver.sequence_max_frames, 32)
        self.assertEqual(large.solver.sequence_dtype, "float32")
        self.assertEqual(large.tool.actual_proxy_type, "FlatBottomQuadProxy_L0")
        self.assertEqual(large.tool.parameters["nominal_width_m"], 3.2)
        modular_tool = load_config(repository_root / "configs" / "project.yaml")
        self.assertEqual(modular_tool.tool.actual_proxy_type, "ExtrudedProfileBucket_L1")
        large_l1 = load_config(repository_root / "configs" / "project_25m_l1.yaml")
        self.assertEqual(large_l1.terrain.to_grid().shape, (701, 701))
        self.assertEqual(large_l1.tool.actual_proxy_type, "ExtrudedProfileBucket_L1")

    def test_loads_units_paths_and_preserves_future_sections(self) -> None:
        text = """
project:
  name: phase1_test
  seed: 7
  output_dir: outputs
terrain:
  heightmap_path: data/H.npy
  nx: 4
  ny: 3
  dx_m: 0.5
  dy_m: 0.25
  origin_x_m: -1.0
  origin_y_m: 2.0
  terrain_prim_path: /World/Terrain/DynamicSurface
  update_rate_hz: 12
  source_axis_order: yx
mesh:
  collision_enabled: false
  update_normals: true
  normal_update_rate_hz: 4
  subdivision_scheme: none
robot:
  robot_root_prim: /World/Robot
  articulation_root_prim: /World/Robot
  tool_link_prim: /World/Robot/bucket
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "project.yaml"
            source.write_text(text, encoding="utf-8")
            config = load_config(source)
            self.assertEqual(config.terrain.to_grid().shape, (3, 4))
            self.assertEqual(config.mesh.mesh_update_rate_hz, 12.0)
            self.assertEqual(config.mesh.normal_update_rate_hz, 4.0)
            self.assertIsNotNone(config.robot)
            self.assertEqual(config.robot.tool_link_prim, "/World/Robot/bucket")
            self.assertEqual(config.project.output_dir, Path(directory) / "outputs")
            self.assertEqual(config.terrain.heightmap_path, Path(directory) / "data/H.npy")

    def test_dynamic_collision_is_rejected(self) -> None:
        source_text = """
project: {name: bad}
terrain:
  heightmap_path: H.npy
  nx: 2
  ny: 2
  dx_m: 1
  dy_m: 1
  origin_x_m: 0
  origin_y_m: 0
  terrain_prim_path: /World/Terrain
mesh: {collision_enabled: true}
"""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.yaml"
            source.write_text(source_text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collision"):
                load_config(source)


if __name__ == "__main__":
    unittest.main()
