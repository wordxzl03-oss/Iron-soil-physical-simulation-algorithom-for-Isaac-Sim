import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.config import (
    ExcavationConfig,
    MaterialConfig,
    SolverConfig,
    SweepConfig,
)
from isaac_bulk_pipeline.interaction import ContinuousSweepBuilder, ExcavationOperator
from isaac_bulk_pipeline.runtime import ActionRecorder, SimulationController
from isaac_bulk_pipeline.solvers import MinimumSlopeAdapter
from isaac_bulk_pipeline.terrain import MassLedger, TerrainGrid, TerrainStateManager
from isaac_bulk_pipeline.tools import ToolDescriptorLoader, ToolState


def _descriptor():
    root = Path(__file__).resolve().parents[1]
    return ToolDescriptorLoader.load(
        ToolDescriptorLoader.load_config(root / "configs" / "bucket_medium.yaml")
    )


def _transform(pose, points):
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (pose @ homogeneous.T).T[:, :3]


def _state(descriptor, pose, timestamp):
    return ToolState(
        timestamp=timestamp,
        pose_world=pose,
        pose_terrain=pose,
        cutting_edge_terrain=_transform(pose, descriptor.cutting_edge_local),
        bottom_profile_terrain=_transform(pose, descriptor.bottom_profile_local),
        left_boundary_terrain=_transform(pose, descriptor.left_boundary_local),
        right_boundary_terrain=_transform(pose, descriptor.right_boundary_local),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )


class _MeshRecorder:
    def __init__(self):
        self.updates = []
        self.reset_height = None

    def update(self, heightmap, affected_bbox=None):
        self.updates.append((np.array(heightmap, copy=True), affected_bbox))
        return {"affected_bbox": affected_bbox}

    def reset(self, heightmap):
        self.reset_height = np.array(heightmap, copy=True)
        return {"reset": True}


class MultiActionStateTests(unittest.TestCase):
    @staticmethod
    def _grid_and_height():
        grid = TerrainGrid(
            nx=101,
            ny=101,
            dx=0.1,
            dy=0.1,
            origin_x=-5.0,
            origin_y=-5.0,
            terrain_prim_path="/World/Terrain",
        )
        y = grid.origin_y + np.arange(grid.ny) * grid.dy
        x = grid.origin_x + np.arange(grid.nx) * grid.dx
        xx, yy = np.meshgrid(x, y, indexing="xy")
        height = (
            2.4
            * np.exp(-((xx + 0.4) / 3.0) ** 4 - ((yy - 0.5) / 2.6) ** 4)
            + 0.35 * np.exp(-((xx - 1.4) / 0.8) ** 2 - ((yy + 0.3) / 1.0) ** 2)
        )
        height[[0, -1], :] = 0.0
        height[:, [0, -1]] = 0.0
        return grid, height

    def _run_episode(self, output_root: Path):
        grid, initial = self._grid_and_height()
        descriptor = _descriptor()
        state_manager = TerrainStateManager(grid, initial)
        ledger = MassLedger.initialize(
            grid,
            initial,
            MaterialConfig(
                bulk_density_kg_m3=2350.0,
                density_is_estimated=True,
            ),
        )
        solver = MinimumSlopeAdapter(
            grid,
            SolverConfig(
                solve_trigger="action_end",
                critical_angle_deg=34.0,
                max_iterations=2000,
                tolerance=1e-5,
                sequence_enabled=False,
                sequence_stride=100,
                boundary_condition="closed",
                conservation_tolerance_m3=1e-8,
            ),
        )
        recorder = ActionRecorder(output_root, episode_index=1)
        episode_dir = recorder.initialize(
            initial,
            {
                "seed": 1234,
                "grid_shape_yx": list(grid.shape),
                "grid_spacing_m": [grid.dx, grid.dy],
                "tool_width_m": descriptor.nominal_width_m,
            },
        )
        mesh = _MeshRecorder()
        controller = SimulationController(
            grid=grid,
            descriptor=descriptor,
            sweep_builder=ContinuousSweepBuilder(SweepConfig()),
            excavation_operator=ExcavationOperator(ExcavationConfig()),
            solver=solver,
            state_manager=state_manager,
            mass_ledger=ledger,
            mesh_adapter=mesh,
            recorder=recorder,
        )
        summaries = []
        previous_stable = initial.copy()
        lateral_offsets = [-2.0, -1.4, -0.8, -0.2, 0.4, 1.0, 1.6, -1.1, 0.1, 1.2]
        for action_index, x_offset in enumerate(lateral_offsets):
            self.assertEqual(controller.begin_action(), action_index)
            np.testing.assert_allclose(
                state_manager.state.H_before_action,
                previous_stable,
                atol=0.0,
            )
            start = np.eye(4)
            end = np.eye(4)
            start[:3, 3] = [x_offset - 0.3, 1.0, 0.62]
            end[:3, 3] = [x_offset + 0.35, 1.75, 0.48]
            cached = controller.process_tool_state(
                _state(descriptor, start, action_index * 2.0)
            )
            self.assertTrue(cached.cached_only)
            step = controller.process_tool_state(
                _state(descriptor, end, action_index * 2.0 + 1.0)
            )
            self.assertIsNotNone(step.excavation)
            summary = controller.end_action()
            self.assertEqual(summary.action_index, action_index)
            self.assertTrue(summary.relaxation.converged)
            self.assertGreaterEqual(summary.removed_volume_m3, 0.0)
            self.assertTrue(np.all(np.isfinite(summary.state.H_current)))
            self.assertTrue(np.all(summary.state.H_current >= 0.0))
            previous_stable = np.array(summary.state.H_stable, copy=True)
            summaries.append(summary)

        self.assertEqual(state_manager.state.action_index, 10)
        self.assertGreater(ledger.removed_volume_m3, 0.0)
        self.assertLess(abs(ledger.numerical_error_m3), 1e-8)
        self.assertEqual(len(mesh.updates), 20)
        return controller, summaries, episode_dir, initial, mesh

    def test_ten_actions_are_continuous_recorded_and_resettable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            controller, summaries, episode_dir, initial, mesh = self._run_episode(
                Path(directory)
            )
            required_common = {
                "metadata.json",
                "H_initial.npy",
                "action_log.json",
                "volume_log.csv",
            }
            self.assertTrue(required_common.issubset({p.name for p in episode_dir.iterdir()}))
            for index in range(10):
                for prefix in (
                    "H_before_action",
                    "H_excavated",
                    "H_stable",
                    "tool_trajectory",
                ):
                    self.assertTrue((episode_dir / f"{prefix}_{index:03d}.npy").is_file())
                self.assertFalse(
                    (episode_dir / f"H_relax_sequence_{index:03d}.npy").exists()
                )
            action_log = json.loads(
                (episode_dir / "action_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(action_log["actions"]), 10)
            self.assertTrue(
                all(
                    item["mass_value_kind"] == "density-based estimate"
                    for item in action_log["actions"]
                )
            )
            self.assertFalse(
                np.array_equal(summaries[1].state.H_before_action, initial)
            )

            controller.reset()
            np.testing.assert_array_equal(
                controller.state_manager.state.H_current,
                initial,
            )
            self.assertEqual(controller.state_manager.state.action_index, 0)
            self.assertEqual(controller.mass_ledger.removed_volume_m3, 0.0)
            self.assertEqual(controller.mass_ledger.boundary_outflow_m3, 0.0)
            self.assertEqual(controller.mass_ledger.numerical_error_m3, 0.0)
            self.assertIsNone(controller.previous_tool_state)
            np.testing.assert_array_equal(mesh.reset_height, initial)
            reset_log = json.loads(
                (episode_dir / "action_log.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reset_log["reset_count"], 1)

    def test_reproducible_episode_has_identical_stable_maps_and_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first, first_summaries, _, _, _ = self._run_episode(Path(first_directory))
            second, second_summaries, _, _, _ = self._run_episode(Path(second_directory))
            np.testing.assert_array_equal(
                first.state_manager.state.H_current,
                second.state_manager.state.H_current,
            )
            np.testing.assert_allclose(
                [item.removed_volume_m3 for item in first_summaries],
                [item.removed_volume_m3 for item in second_summaries],
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(
                first.mass_ledger.to_mapping(),
                second.mass_ledger.to_mapping(),
            )


if __name__ == "__main__":
    unittest.main()
