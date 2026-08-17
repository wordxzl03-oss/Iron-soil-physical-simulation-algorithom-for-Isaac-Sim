import unittest

import numpy as np

from isaac_bulk_pipeline.contact import (
    PhysXHeightFieldContactBackend,
    SlopePatchSpec,
    assess_heightfield_capability,
    build_planar_slope_heightmap,
    estimate_slope_deg,
)
from isaac_bulk_pipeline.terrain import TerrainGrid


class PhaseCHeightFieldSlopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TerrainGrid(
            nx=31,
            ny=21,
            dx=0.1,
            dy=0.1,
            origin_x=-1.5,
            origin_y=-1.0,
            terrain_prim_path="/World/TerrainVisual",
        )

    def test_missing_public_schema_selects_triangle_mesh(self) -> None:
        status = assess_heightfield_capability(
            runtime_label="Isaac Sim 4.5",
            public_schema_symbols=[],
        )
        self.assertFalse(status.usable)
        self.assertEqual(status.state, "unavailable_public_api")
        self.assertEqual(status.selected_backend, "TriangleMeshContactBackend")
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            PhysXHeightFieldContactBackend(status)

    def test_symbol_alone_does_not_claim_stable_heightfield(self) -> None:
        status = assess_heightfield_capability(
            runtime_label="synthetic",
            public_schema_symbols=["PhysxHeightFieldSchema"],
        )
        self.assertEqual(status.state, "authoring_only")
        self.assertFalse(status.usable)

    def test_planar_slope_fixtures_are_exact_for_acceptance_angles(self) -> None:
        for expected in (0.0, 10.0, 20.0):
            with self.subTest(expected=expected):
                height = build_planar_slope_heightmap(
                    self.grid,
                    SlopePatchSpec(expected, uphill_direction_xy=(1.0, 0.4)),
                )
                actual = estimate_slope_deg(height, self.grid)
                self.assertEqual(height.shape, self.grid.shape)
                self.assertGreaterEqual(float(height.min()), 0.0)
                np.testing.assert_allclose(actual, expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
