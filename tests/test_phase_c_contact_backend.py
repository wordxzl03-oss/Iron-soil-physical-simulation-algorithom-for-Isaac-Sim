import unittest

import numpy as np

from isaac_bulk_pipeline.contact import ContactBackendConfig, TriangleMeshContactBackend
from isaac_bulk_pipeline.terrain import TerrainGrid


class PhaseCContactBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TerrainGrid(
            nx=7,
            ny=7,
            dx=0.05,
            dy=0.05,
            origin_x=0.0,
            origin_y=0.0,
            terrain_prim_path="/World/TerrainVisual",
        )
        self.initial = np.zeros(self.grid.shape)

    def test_update_is_staged_until_action_end_commit(self) -> None:
        backend = TriangleMeshContactBackend()
        backend.initialize(self.grid, self.initial)
        initial_z = backend.committed_mesh.points_m[:, 2].copy()
        changed = np.full(self.grid.shape, 0.4)
        backend.update_from_heightmap(changed, action_index=0)
        self.assertTrue(backend.pending)
        np.testing.assert_allclose(backend.committed_mesh.points_m[:, 2], initial_z)
        result = backend.commit(reason="action_end")
        self.assertTrue(result.committed)
        self.assertTrue(result.recook_required)
        self.assertEqual(result.generation, 1)
        np.testing.assert_allclose(backend.committed_mesh.points_m[:, 2], 0.4)

    def test_no_pending_update_does_not_request_recook(self) -> None:
        backend = TriangleMeshContactBackend()
        backend.initialize(self.grid, self.initial)
        result = backend.commit(reason="action_end")
        self.assertFalse(result.committed)
        self.assertFalse(result.recook_required)

    def test_low_frequency_policy_rate_limits_recooking(self) -> None:
        config = ContactBackendConfig(commit_policy="low_frequency", low_frequency_hz=2.0)
        backend = TriangleMeshContactBackend(config)
        backend.initialize(self.grid, self.initial)
        backend.update_from_heightmap(np.full(self.grid.shape, 0.1))
        first = backend.commit(reason="low_frequency", sim_time_s=1.0)
        self.assertTrue(first.committed)
        backend.update_from_heightmap(np.full(self.grid.shape, 0.2))
        limited = backend.commit(reason="low_frequency", sim_time_s=1.2)
        self.assertTrue(limited.rate_limited)
        self.assertTrue(backend.pending)
        accepted = backend.commit(reason="low_frequency", sim_time_s=1.5)
        self.assertTrue(accepted.committed)
        np.testing.assert_allclose(backend.committed_mesh.points_m[:, 2], 0.2)

    def test_action_end_policy_rejects_physics_frame_recook_mode(self) -> None:
        backend = TriangleMeshContactBackend()
        backend.initialize(self.grid, self.initial)
        backend.update_from_heightmap(np.full(self.grid.shape, 0.1))
        with self.assertRaisesRegex(ValueError, "disabled by policy"):
            backend.commit(reason="low_frequency", sim_time_s=1.0)

    def test_reset_restores_initial_contact_surface(self) -> None:
        backend = TriangleMeshContactBackend()
        backend.initialize(self.grid, self.initial)
        backend.update_from_heightmap(np.full(self.grid.shape, 0.5))
        backend.commit(reason="action_end")
        result = backend.reset()
        self.assertTrue(result.recook_required)
        np.testing.assert_allclose(backend.committed_mesh.points_m[:, 2], 0.0)

    def test_visual_and_contact_paths_cannot_alias(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be different"):
            ContactBackendConfig(
                contact_prim_path="/World/Terrain",
                visual_prim_path="/World/Terrain",
            )


if __name__ == "__main__":
    unittest.main()
