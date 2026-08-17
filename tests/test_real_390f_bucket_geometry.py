import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.config import load_config
from isaac_bulk_pipeline.tools import ToolDescriptorLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class Real390FBucketGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project = load_config(
            REPOSITORY_ROOT / "configs" / "project_25m_390f.yaml"
        )
        assert self.project.tool is not None
        self.descriptor = ToolDescriptorLoader.load(self.project.tool)
        self.geometry = self.descriptor.bucket_geometry
        assert self.geometry is not None
        self.audit = json.loads(
            (REPOSITORY_ROOT / "outputs" / "real_390f_bucket_audit.json").read_text(
                encoding="utf-8"
            )
        )

    def test_real_cad_semantic_geometry_is_authoritative(self) -> None:
        self.assertEqual(self.descriptor.tool_type, "tracked_excavator_bucket_390f")
        self.assertEqual(self.descriptor.geometry_source, "USD_MESH_MARKERS")
        self.assertEqual(self.descriptor.proxy_level, "L1")
        self.assertFalse(self.geometry.legacy_fallback_used)
        self.assertFalse(
            self.geometry.metadata["convex_hull_used_as_soil_geometry"]
        )
        self.assertEqual(
            self.geometry.metadata["asset_sha256"], self.audit["asset_sha256"]
        )

    def test_cutting_mouth_bottom_and_concave_interior_are_valid(self) -> None:
        self.assertAlmostEqual(self.geometry.cutting_edge_length_m, 2.7420991211)
        self.assertAlmostEqual(self.geometry.interior_width_m, 2.56)
        self.assertGreater(self.geometry.mouth_area_m2, 0.0)
        self.assertGreater(self.geometry.geometric_capacity_m3, 0.0)
        self.assertAlmostEqual(
            self.geometry.closed_mesh_volume_m3,
            self.geometry.geometric_capacity_m3,
            places=10,
        )
        self.assertEqual(self.geometry.mouth_polygon_local.shape, (4, 3))
        self.assertGreater(len(self.geometry.interior_profile_local), 10)
        self.assertTrue(np.all(np.isfinite(self.geometry.bottom_plate_normal_local)))
        self.assertAlmostEqual(
            np.linalg.norm(self.geometry.bottom_plate_normal_local), 1.0
        )

    def test_runtime_wiring_uses_tracked_excavator_paths_and_file_descriptor(self) -> None:
        assert self.project.robot is not None and self.project.tool is not None
        self.assertIn("390F", self.project.robot.articulation_root_prim)
        self.assertIn("Bucket1", self.project.robot.tool_link_prim)
        self.assertEqual(self.project.tool.descriptor_source, "file")
        self.assertEqual(self.project.terrain.dx_m, 0.05)
        self.assertEqual(self.project.terrain.dy_m, 0.05)
        self.assertEqual(
            self.project.extra_sections["asset_integration"]["vehicle_architecture"],
            "TRACKED_HYDRAULIC_EXCAVATOR",
        )
        self.assertEqual(
            self.project.extra_sections["asset_integration"][
                "bucket_rigid_bulk_collision"
            ],
            "DISABLED_TO_PREVENT_DOUBLE_COUNTING",
        )

    def test_audit_fixture_rejects_convex_hull_as_soil_geometry(self) -> None:
        self.assertEqual(self.audit["collision_approximation"], "convexHull")
        self.assertEqual(
            self.audit["collision_classification"],
            "CURRENT_COLLISION_ONLY_NOT_SOIL_GEOMETRY",
        )
        self.assertFalse(self.audit["convex_hull_used_as_soil_geometry"])
        self.assertEqual(self.audit["bucket_mass_kg"], 6000.0)
        self.assertEqual(self.audit["mesh_count_below_bucket"], 1)
        self.assertEqual(self.audit["vertex_count"], 9566)
        self.assertEqual(self.audit["triangle_count"], 10086)

        asset = Path(self.audit["asset"])
        if asset.is_file():
            checksum = hashlib.sha256()
            with asset.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(block)
            digest = checksum.hexdigest()
            self.assertEqual(digest, self.audit["asset_sha256"])


if __name__ == "__main__":
    unittest.main()
