import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.runtime import ActionRecorder
from isaac_bulk_pipeline.solvers import RelaxationResult
from isaac_bulk_pipeline.terrain import MassLedger, TerrainState


class ActionRecorderFrameTests(unittest.TestCase):
    def test_six_actions_write_full_frame_schema_and_float32_h0_h6(self) -> None:
        initial = np.full((3, 4), 2.0, dtype=np.float64)
        ledger = MassLedger(
            initial_terrain_volume_m3=24.0,
            current_terrain_volume_m3=24.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = ActionRecorder(Path(directory), episode_index=1)
            episode = recorder.initialize(
                initial,
                {"actual_proxy_type": "FlatBottomQuadProxy_L0"},
            )
            previous = initial.copy()
            for action_index in range(6):
                current = previous - 0.1
                removed = float(np.sum(previous - current))
                ledger.removed_volume_m3 += removed
                ledger.current_terrain_volume_m3 -= removed
                state = TerrainState(
                    H_initial=initial.copy(),
                    H_current=current.copy(),
                    H_before_action=previous.copy(),
                    H_excavated=current.copy(),
                    H_stable=current.copy(),
                    action_index=action_index + 1,
                    total_removed_volume_m3=ledger.removed_volume_m3,
                )
                relaxation = RelaxationResult(
                    heightmap_stable=current,
                    heightmap_sequence=(
                        previous.astype(np.float32),
                        current.astype(np.float32),
                    ),
                    iteration_count=1,
                    volume_before_m3=float(previous.sum()),
                    volume_after_m3=float(current.sum()),
                    boundary_outflow_m3=0.0,
                    converged=True,
                    diagnostics={
                        "volume_balance_error_m3": 0.0,
                        "sequence_enabled": False,
                        "sequence_stride": 250,
                        "sequence_max_frames": 32,
                        "sequence_dtype": "float32",
                        "sequence_memory_limit_mb": 8.0,
                    },
                )
                recorder.record_frame(
                    timestamp=float(action_index),
                    simulation_frame=action_index + 1,
                    action_index=action_index,
                    action_phase="penetration",
                    joint_names=("lift_joint", "bucket_joint"),
                    joint_positions=np.array([0.1, 0.2]),
                    joint_velocities=np.array([0.01, 0.02]),
                    tool_link_pose_world=np.eye(4),
                    tool_pose_terrain=np.eye(4),
                    cutting_enabled=True,
                    affected_cell_count=1 if action_index % 2 == 0 else 0,
                    removed_volume_m3=0.1 if action_index % 2 == 0 else 0.0,
                )
                recorder.record_action(
                    action_index=action_index,
                    state=state,
                    relaxation=relaxation,
                    tool_trajectory=(np.eye(4),),
                    removed_volume_m3=removed,
                    ledger=ledger,
                )
                self.assertFalse(
                    (episode / f"H_relax_sequence_{action_index:03d}.npy").exists()
                )
                previous = current

            output = recorder.finalize_episode(expected_action_count=6)
            with np.load(output, allow_pickle=False) as archive:
                heightmaps = archive["heightmaps_m"]
                self.assertEqual(heightmaps.shape, (7, 3, 4))
                self.assertEqual(heightmaps.dtype, np.float32)
            frame_log = json.loads(
                (episode / "joint_state_log.json").read_text(encoding="utf-8")
            )
            events = json.loads(
                (episode / "effective_excavation_events.json").read_text(
                    encoding="utf-8"
                )
            )
            metadata = json.loads(
                (episode / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frame_log["frame_count"], 6)
            self.assertEqual(events["event_count"], 3)
            self.assertEqual(metadata["h0_h6_shape"], [7, 3, 4])
            self.assertEqual(metadata["h0_h6_dtype"], "float32")
            required = {
                "timestamp",
                "simulation_frame",
                "action_index",
                "action_phase",
                "joint_names",
                "joint_positions",
                "joint_velocities",
                "tool_link_pose_world",
                "tool_pose_terrain",
                "cutting_enabled",
                "affected_cell_count",
                "removed_volume_m3",
            }
            self.assertTrue(required.issubset(frame_log["frames"][0]))


if __name__ == "__main__":
    unittest.main()
