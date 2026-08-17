import unittest
from pathlib import Path

from isaac_bulk_pipeline.contact import (
    CollisionGroup,
    CollisionMatrix,
    load_phase_c_contact_config,
    validate_collision_memberships,
)


class PhaseCCollisionFilterTests(unittest.TestCase):
    def test_required_collision_matrix(self) -> None:
        matrix = CollisionMatrix()
        self.assertTrue(
            matrix.allows(CollisionGroup.WHEEL_CONTACT, CollisionGroup.TERRAIN_SUPPORT)
        )
        self.assertTrue(
            matrix.allows(CollisionGroup.CHASSIS_CONTACT, CollisionGroup.TERRAIN_SUPPORT)
        )
        self.assertFalse(
            matrix.allows(
                CollisionGroup.BUCKET_INTERACTION,
                CollisionGroup.TERRAIN_SUPPORT,
            )
        )
        self.assertTrue(
            matrix.allows(
                CollisionGroup.ENVIRONMENT_STATIC,
                CollisionGroup.BUCKET_INTERACTION,
            )
        )
        self.assertEqual(
            matrix.filtered_pairs(),
            ((CollisionGroup.TERRAIN_SUPPORT, CollisionGroup.BUCKET_INTERACTION),),
        )

    def test_membership_rejects_visual_mesh_and_duplicate_collider(self) -> None:
        memberships = {
            CollisionGroup.TERRAIN_SUPPORT: ["/World/TerrainContact"],
            CollisionGroup.WHEEL_CONTACT: ["/World/Vehicle/Wheel"],
            CollisionGroup.CHASSIS_CONTACT: ["/World/Vehicle/Body"],
            CollisionGroup.BUCKET_INTERACTION: ["/World/Vehicle/Bucket"],
            CollisionGroup.ENVIRONMENT_STATIC: [
                "/World/TerrainVisual",
                "/World/Vehicle/Body",
            ],
        }
        result = validate_collision_memberships(
            memberships,
            contact_prim_path="/World/TerrainContact",
            visual_prim_path="/World/TerrainVisual",
        )
        self.assertFalse(result.valid)
        self.assertTrue(any("multiple groups" in error for error in result.errors))
        self.assertTrue(any("visual terrain" in error for error in result.errors))

    def test_phase_c_yaml_loads_strict_safe_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_phase_c_contact_config(root / "configs/phase_c_contact.yaml")
        self.assertEqual(config.backend_type, "triangle_mesh")
        self.assertEqual(config.backend.target_spacing_m, 0.10)
        self.assertEqual(config.backend.commit_policy, "action_end")
        self.assertFalse(config.backend.visible)
        self.assertFalse(config.backend.rigid_body_enabled)
        self.assertFalse(config.wheel_terrain.deformation_enabled)
        self.assertEqual(config.slope_test_degrees, (0.0, 10.0, 20.0))


if __name__ == "__main__":
    unittest.main()
