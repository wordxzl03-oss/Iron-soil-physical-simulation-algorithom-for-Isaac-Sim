import unittest

import numpy as np

from isaac_bulk_pipeline.tools import validate_marker_positions


class MarkerValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.markers = {
            "ToolOrigin": [2.0, 3.0, 0.5],
            "CuttingEdgeLeft": [2.0, 4.6, 0.5],
            "CuttingEdgeCenter": [2.0, 3.0, 0.5],
            "CuttingEdgeRight": [2.0, 1.4, 0.5],
            "BottomRearLeft": [-0.4, 4.6, 0.5],
            "BottomRearRight": [-0.4, 1.4, 0.5],
            "SideTopLeft": [-0.4, 4.6, 1.75],
            "SideTopRight": [-0.4, 1.4, 1.75],
        }

    def test_builds_right_handed_unified_tool_frame(self) -> None:
        result = validate_marker_positions(self.markers)
        transform = result.reference_from_tool
        self.assertAlmostEqual(result.nominal_width_m, 3.2)
        np.testing.assert_allclose(transform[:3, 3], [2.0, 3.0, 0.5])
        # left-to-right is world -Y; rear-to-mouth is world +X.
        np.testing.assert_allclose(transform[:3, 0], [0.0, -1.0, 0.0])
        np.testing.assert_allclose(transform[:3, 1], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(transform[:3, 2], [0.0, 0.0, 1.0])
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)

    def test_missing_or_mislabelled_markers_fail_with_context(self) -> None:
        missing = dict(self.markers)
        missing.pop("SideTopRight")
        with self.assertRaisesRegex(ValueError, "SideTopRight"):
            validate_marker_positions(missing)
        inverted = dict(self.markers)
        inverted["SideTopLeft"] = [-0.4, 4.6, -1.0]
        inverted["SideTopRight"] = [-0.4, 1.4, -1.0]
        with self.assertRaisesRegex(ValueError, "SideTop"):
            validate_marker_positions(inverted)


if __name__ == "__main__":
    unittest.main()
