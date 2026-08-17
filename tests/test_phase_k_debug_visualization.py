import unittest
import numpy as np

from isaac_bulk_pipeline.visualization.debug_layers import (
    DebugVisualizationConfig,
    DebugVisualizationFrame,
)


class TestDebugVisualization(unittest.TestCase):
    def test_debug_frame_is_non_authoritative_immutable_data(self):
        empty = np.empty((0, 3))
        frame = DebugVisualizationFrame(empty, empty, np.zeros((2, 3)), np.zeros(3), np.zeros(3), np.ones(3), empty, np.zeros((3, 3)), np.zeros((2, 3)))
        self.assertFalse(frame.bucket_mouth_world_m.flags.writeable)
        self.assertFalse(DebugVisualizationConfig().show_failure_zone)


if __name__ == "__main__":
    unittest.main()
