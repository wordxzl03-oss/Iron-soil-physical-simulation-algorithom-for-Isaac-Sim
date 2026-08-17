import unittest

import numpy as np

from render_scooping_avalanche import (
    _CUTTING_EDGE_FROM_PIVOT,
    _bucket_polygon,
    _pivot_for_edge,
)


class LoaderBucketKinematicsTests(unittest.TestCase):
    def test_edge_position_defines_matching_rear_pivot(self):
        for pitch in (0.0, 20.0, 50.0):
            pivot = _pivot_for_edge(-5.5, 1.2, pitch)
            edge = _bucket_polygon(*pivot, pitch)[0]
            np.testing.assert_allclose(edge, (-5.5, 1.2), atol=1e-12)

    def test_curl_keeps_pivot_fixed_and_edge_on_circle(self):
        pivot = (-7.2, 0.95)
        radius = float(np.linalg.norm(_CUTTING_EDGE_FROM_PIVOT))
        edges = np.array(
            [_bucket_polygon(*pivot, pitch)[0] for pitch in np.linspace(0, 50, 21)]
        )
        distances = np.linalg.norm(edges - np.asarray(pivot), axis=1)
        np.testing.assert_allclose(distances, radius, atol=1e-12)
        self.assertGreater(edges[-1, 1], edges[0, 1])


if __name__ == "__main__":
    unittest.main()
