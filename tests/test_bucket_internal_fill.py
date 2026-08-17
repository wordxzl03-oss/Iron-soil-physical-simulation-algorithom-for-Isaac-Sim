from pathlib import Path
import unittest

import numpy as np

from isaac_bulk_pipeline.bulk_state import (
    BucketFillPhase,
    BucketInternalFillModel,
    PayloadState,
)
from isaac_bulk_pipeline.tools import ToolDescriptorLoader


ROOT = Path(__file__).resolve().parents[1]


class BucketInternalFillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.descriptor = ToolDescriptorLoader.load(
            ToolDescriptorLoader.load_config(
                ROOT / "configs" / "excavator_390f_real_bucket.yaml"
            )
        )
        cls.geometry = cls.descriptor.bucket_geometry
        assert cls.geometry is not None
        cls.model = BucketInternalFillModel(volume_tolerance_m3=1.0e-10)

    def payload(self, volume: float = 0.0) -> PayloadState:
        return PayloadState(
            volume,
            self.geometry.effective_capacity_m3,
            1370.0,
            np.zeros(3),
        )

    def test_bisection_matches_requested_volume_without_lip_constraint(self) -> None:
        target = 0.37 * self.geometry.effective_capacity_m3
        result = self.model.solve_payload(
            self.payload(),
            self.descriptor,
            np.asarray([0.0, 0.35, 0.94]),
            volume_m3=target,
            phase=BucketFillPhase.RELAXING,
        )
        fill = result.internal_fill
        assert fill is not None
        profile = fill.occupied_polygon_bucket_frame_m[:, 1:3]
        area = abs(self.model._polygon_properties(profile)[0])
        self.assertAlmostEqual(
            area * self.geometry.interior_width_m, target, places=9
        )
        self.assertAlmostEqual(fill.mass_kg, target * 1370.0, places=8)
        self.assertAlmostEqual(fill.geometric_fill_ratio, 0.37, places=10)
        self.assertEqual(fill.phase, BucketFillPhase.RELAXING)
        lip_center = np.mean(self.geometry.lip_local, axis=0)
        lip_offset = float(np.dot(fill.free_surface.normal_bucket_frame, lip_center))
        self.assertGreater(abs(fill.free_surface.offset_m - lip_offset), 1.0e-3)

    def test_com_second_moment_and_inertia_are_finite_positive(self) -> None:
        result = self.model.solve_payload(
            self.payload(),
            self.descriptor,
            np.asarray([0.0, 0.0, 1.0]),
            volume_m3=0.6 * self.geometry.effective_capacity_m3,
        )
        fill = result.internal_fill
        assert fill is not None
        self.assertTrue(np.all(np.isfinite(fill.center_of_mass_bucket_frame_m)))
        self.assertTrue(np.allclose(fill.second_moment_volume_m5, fill.second_moment_volume_m5.T))
        self.assertTrue(np.allclose(fill.inertia_tensor_kg_m2, fill.inertia_tensor_kg_m2.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(fill.inertia_tensor_kg_m2) >= -1.0e-8))
        self.assertFalse(fill.inertia_tensor_kg_m2.flags.writeable)

    def test_secondary_plate_moves_continuously_from_bottom_to_mouth(self) -> None:
        empty = self.model.solve_payload(
            self.payload(), self.descriptor, np.asarray([0.0, 0.0, 1.0]), volume_m3=0.0
        ).internal_fill
        half = self.model.solve_payload(
            self.payload(),
            self.descriptor,
            np.asarray([0.0, 0.0, 1.0]),
            volume_m3=0.5 * self.geometry.effective_capacity_m3,
        ).internal_fill
        full = self.model.solve_payload(
            self.payload(),
            self.descriptor,
            np.asarray([0.0, 0.0, 1.0]),
            volume_m3=self.geometry.effective_capacity_m3,
        ).internal_fill
        assert empty is not None and half is not None and full is not None
        primary = np.array(self.geometry.separation_plane_direction_local, copy=True)
        primary /= np.linalg.norm(primary)
        lip = np.mean(self.geometry.cutting_edge_local, axis=0)
        mouth = np.mean(self.geometry.top_edge_local, axis=0) - lip
        mouth /= np.linalg.norm(mouth)
        np.testing.assert_allclose(
            empty.secondary_separation_direction_bucket_frame, primary, atol=1.0e-12
        )
        np.testing.assert_allclose(
            full.secondary_separation_direction_bucket_frame, mouth, atol=1.0e-12
        )
        self.assertGreater(
            np.dot(half.secondary_separation_direction_bucket_frame, primary), 0.0
        )
        self.assertGreater(
            np.dot(half.secondary_separation_direction_bucket_frame, mouth), 0.0
        )


if __name__ == "__main__":
    unittest.main()
