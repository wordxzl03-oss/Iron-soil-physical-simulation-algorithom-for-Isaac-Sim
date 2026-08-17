import ast
import argparse
from pathlib import Path
import unittest

from isaac_loader.phase_c_contact_runtime import resolve_yaml_configuration


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PhaseCRuntimeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = (
            Path(__file__).resolve().parents[1]
            / "isaac_loader/phase_c_contact_runtime.py"
        )
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_runtime_has_no_pose_setter_or_private_physx_hook(self) -> None:
        forbidden = {
            "set_world_pose",
            "set_world_poses",
            "set_local_pose",
            "acquire_physx_interface",
        }
        called = {
            node.func.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertTrue(forbidden.isdisjoint(called), forbidden.intersection(called))

    def test_collision_groups_use_public_collection_api(self) -> None:
        self.assertIn('Usd.CollectionAPI.Apply(schema.GetPrim(), "colliders")', self.source)
        self.assertIn("CreateFilteredGroupsRel", self.source)
        self.assertIn("UsdPhysics.MeshCollisionAPI.Apply", self.source)

    def test_runtime_resolves_phase_c_and_terrain_yaml_as_authoritative(self) -> None:
        args = argparse.Namespace(
            slope_deg=10.0,
            contact_spacing_m=0.10,
            visual_spacing_m=0.05,
            contact_config=REPOSITORY_ROOT / "configs/phase_c_contact.yaml",
            terrain_config=REPOSITORY_ROOT / "configs/project_25m.yaml",
        )
        resolve_yaml_configuration(args)
        self.assertEqual(args.contact_prim_path, "/World/TerrainContact")
        self.assertEqual(args.visual_prim_path, "/World/TerrainVisual")
        self.assertEqual(args.commit_policy, "action_end")
        self.assertEqual(
            args.configuration_evidence["effective"]["wheel_terrain"],
            {
                "deformation_enabled": False,
                "sinkage_enabled": False,
                "compaction_enabled": False,
                "rut_enabled": False,
            },
        )
        self.assertEqual(len(args.configuration_evidence["contact_config_sha256"]), 64)

        bad = argparse.Namespace(
            slope_deg=10.0,
            contact_spacing_m=0.15,
            visual_spacing_m=0.05,
            contact_config=REPOSITORY_ROOT / "configs/phase_c_contact.yaml",
            terrain_config=REPOSITORY_ROOT / "configs/project_25m.yaml",
        )
        with self.assertRaisesRegex(ValueError, "contact-spacing-m"):
            resolve_yaml_configuration(bad)


if __name__ == "__main__":
    unittest.main()
