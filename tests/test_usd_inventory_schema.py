import copy
import unittest

from isaac_bulk_pipeline.audit import (
    INVENTORY_SCHEMA_VERSION,
    validate_inventory,
)


def _attribute(value=None, authored=False):
    return {"authored": authored, "value": value}


def _fixture():
    collections = {
        "prims": [{"path": "/Loader"}, {"path": "/Loader/joint"}],
        "articulation_roots": [{"path": "/Loader"}],
        "rigid_bodies": [{"path": "/Loader"}],
        "joints": [
            {
                "path": "/Loader/joint",
                "drives": [
                    {
                        "instance": "angular",
                        "type": _attribute("force", True),
                        "stiffness": _attribute(100.0, True),
                        "damping": _attribute(10.0, True),
                        "max_force": _attribute(200.0, True),
                        "target_position": _attribute(0.0, True),
                        "target_velocity": _attribute(0.0, False),
                    }
                ],
            }
        ],
        "colliders": [
            {
                "path": "/Loader/collider",
                "applied_schemas": ["PhysicsCollisionAPI"],
                "collision_enabled": _attribute(True, False),
            }
        ],
        "collision_groups": [],
        "filtered_pairs": [],
        "physics_materials": [],
        "physx_attributes": [],
    }
    summary = {
        "prim_count": 2,
        "articulation_root_count": 1,
        "rigid_body_count": 1,
        "joint_count": 1,
        "collider_count": 1,
        "collision_group_count": 0,
        "filtered_pair_owner_count": 0,
        "physics_material_count": 0,
        "physx_attribute_owner_count": 0,
    }
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_asset": {
            "path": "isaac_loader/wheel_loader.usd",
            "size_bytes": 123,
            "sha256": "a" * 64,
        },
        "runtime": {"application": "Isaac Sim", "headless": True},
        "stage": {
            "meters_per_unit": 1.0,
            "kilograms_per_unit": 1.0,
            "time_codes_per_second": 60.0,
            "up_axis": "Z",
        },
        "summary": summary,
        **collections,
    }


class UsdInventorySchemaTests(unittest.TestCase):
    def test_valid_fixture(self):
        validate_inventory(_fixture())

    def test_rejects_collection_count_mismatch(self):
        document = _fixture()
        document["summary"]["joint_count"] = 2
        with self.assertRaisesRegex(ValueError, "joint_count"):
            validate_inventory(document)

    def test_rejects_unsorted_or_duplicate_paths(self):
        document = _fixture()
        document["prims"] = [
            {"path": "/Loader/z"},
            {"path": "/Loader/a"},
        ]
        with self.assertRaisesRegex(ValueError, "sorted"):
            validate_inventory(document)

        document = _fixture()
        document["prims"] = [{"path": "/Loader"}, {"path": "/Loader"}]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_inventory(document)

    def test_rejects_non_finite_drive_value(self):
        document = copy.deepcopy(_fixture())
        document["joints"][0]["drives"][0]["max_force"]["value"] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_inventory(document)

    def test_rejects_bad_digest(self):
        document = _fixture()
        document["source_asset"]["sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_inventory(document)


if __name__ == "__main__":
    unittest.main()
