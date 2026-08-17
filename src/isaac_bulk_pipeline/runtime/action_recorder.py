"""Write deterministic per-action height maps, trajectories and ledgers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..solvers import RelaxationResult
from ..terrain import MassLedger, TerrainState


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class ActionRecorder:
    """Persist one episode without owning or mutating authoritative state."""

    def __init__(
        self,
        output_root: str | Path,
        *,
        episode_index: int | None = None,
    ) -> None:
        self._output_root = Path(output_root).expanduser().resolve()
        self._requested_episode_index = episode_index
        self._episode_dir: Path | None = None
        self._action_records: list[dict[str, Any]] = []
        self._frame_records: list[dict[str, Any]] = []
        self._effective_excavation_events: list[dict[str, Any]] = []
        self._metadata_document: dict[str, Any] | None = None
        self._reset_count = 0

    @property
    def episode_dir(self) -> Path:
        if self._episode_dir is None:
            raise RuntimeError("[ActionRecorder] initialize must be called first")
        return self._episode_dir

    def initialize(
        self,
        initial_heightmap: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> Path:
        """Create a new episode directory and save immutable run metadata."""

        if self._episode_dir is not None:
            raise RuntimeError("[ActionRecorder] initialize may only be called once")
        self._output_root.mkdir(parents=True, exist_ok=True)
        if self._requested_episode_index is None:
            index = 1
            while (self._output_root / f"episode_{index:04d}").exists():
                index += 1
        else:
            index = int(self._requested_episode_index)
            if index < 0:
                raise ValueError("[ActionRecorder] episode_index must be non-negative")
        episode_dir = self._output_root / f"episode_{index:04d}"
        if episode_dir.exists():
            raise FileExistsError(
                f"[ActionRecorder] episode directory already exists: {episode_dir}"
            )
        episode_dir.mkdir(parents=False)
        initial = np.asarray(initial_heightmap, dtype=np.float64)
        if initial.ndim != 2 or not np.all(np.isfinite(initial)):
            raise ValueError(
                "[ActionRecorder] initial_heightmap must be finite 2-D; "
                f"shape={initial.shape}"
            )
        np.save(episode_dir / "H_initial.npy", np.ascontiguousarray(initial))
        document = {
            "schema_version": "isaac-bulk-episode-v2",
            "heightmap_axis_order": "yx",
            "length_unit": "metre",
            "heightmap_is_authoritative": True,
            "mesh_is_derived_visualization": True,
            "model_fidelity": (
                "reduced-order height-field/minimum-slope simulation; "
                "not validated iron-ore physical truth"
            ),
            **_jsonable(metadata),
        }
        self._episode_dir = episode_dir
        self._metadata_document = document
        self._write_metadata()
        self._write_action_log()
        self._write_frame_logs()
        return episode_dir

    def record_frame(
        self,
        *,
        timestamp: float,
        simulation_frame: int,
        action_index: int,
        action_phase: str,
        joint_names: Sequence[str],
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        tool_link_pose_world: np.ndarray,
        tool_pose_terrain: np.ndarray,
        cutting_enabled: bool,
        affected_cell_count: int,
        removed_volume_m3: float,
    ) -> dict[str, Any]:
        """Record every physics frame, including non-cutting motion frames."""

        names = tuple(str(name) for name in joint_names)
        positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
        velocities = np.asarray(joint_velocities, dtype=np.float64).reshape(-1)
        world_pose = np.asarray(tool_link_pose_world, dtype=np.float64)
        terrain_pose = np.asarray(tool_pose_terrain, dtype=np.float64)
        if positions.shape != (len(names),) or velocities.shape != (len(names),):
            raise ValueError(
                "[ActionRecorder] joint frame shape mismatch; "
                f"names={len(names)}, positions={positions.shape}, "
                f"velocities={velocities.shape}"
            )
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("[ActionRecorder] joint frame contains NaN or Inf")
        for label, pose in (
            ("tool_link_pose_world", world_pose),
            ("tool_pose_terrain", terrain_pose),
        ):
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError(
                    f"[ActionRecorder] {label} must be finite (4,4); shape={pose.shape}"
                )
        if simulation_frame < 0 or action_index < 0 or affected_cell_count < 0:
            raise ValueError(
                "[ActionRecorder] frame/action/cell indices must be non-negative"
            )
        if not action_phase:
            raise ValueError("[ActionRecorder] action_phase must not be empty")
        if not np.isfinite(timestamp) or not np.isfinite(removed_volume_m3):
            raise ValueError("[ActionRecorder] frame time/volume must be finite")
        if removed_volume_m3 < 0.0:
            raise ValueError("[ActionRecorder] removed_volume_m3 must be non-negative")
        record = {
            "timestamp": float(timestamp),
            "simulation_frame": int(simulation_frame),
            "action_index": int(action_index),
            "action_phase": str(action_phase),
            "joint_names": list(names),
            "joint_positions": positions.tolist(),
            "joint_velocities": velocities.tolist(),
            "tool_link_pose_world": world_pose.tolist(),
            "tool_pose_terrain": terrain_pose.tolist(),
            "cutting_enabled": bool(cutting_enabled),
            "affected_cell_count": int(affected_cell_count),
            "removed_volume_m3": float(removed_volume_m3),
        }
        self._frame_records.append(record)
        if removed_volume_m3 > 0.0:
            self._effective_excavation_events.append(dict(record))
        return dict(record)

    def record_action(
        self,
        *,
        action_index: int,
        state: TerrainState,
        relaxation: RelaxationResult,
        tool_trajectory: Sequence[np.ndarray],
        removed_volume_m3: float,
        ledger: MassLedger,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save all required state arrays and append JSON/CSV diagnostics."""

        if state.H_before_action is None or state.H_excavated is None or state.H_stable is None:
            raise ValueError(
                "[ActionRecorder] action state is incomplete; "
                f"action_index={action_index}"
            )
        suffix = f"{action_index:03d}"
        np.save(self.episode_dir / f"H_before_action_{suffix}.npy", state.H_before_action)
        np.save(self.episode_dir / f"H_excavated_{suffix}.npy", state.H_excavated)
        np.save(self.episode_dir / f"H_stable_{suffix}.npy", state.H_stable)
        sequence_path = self._write_bounded_relaxation_sequence(
            suffix, relaxation
        )
        trajectory = np.asarray(tool_trajectory, dtype=np.float64)
        if trajectory.size == 0:
            trajectory = np.empty((0, 4, 4), dtype=np.float64)
        if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
            raise ValueError(
                "[ActionRecorder] tool trajectory must have shape (N,4,4); "
                f"received={trajectory.shape}, action_index={action_index}"
            )
        np.save(self.episode_dir / f"tool_trajectory_{suffix}.npy", trajectory)

        record = {
            "action_index": int(action_index),
            "removed_volume_m3": float(removed_volume_m3),
            "relaxation_iteration_count": relaxation.iteration_count,
            "relaxation_converged": relaxation.converged,
            "boundary_outflow_m3": relaxation.boundary_outflow_m3,
            "volume_before_relaxation_m3": relaxation.volume_before_m3,
            "volume_after_relaxation_m3": relaxation.volume_after_m3,
            "relaxation_balance_error_m3": relaxation.diagnostics[
                "volume_balance_error_m3"
            ],
            "trajectory_pose_count": int(len(trajectory)),
            "relaxation_sequence_file": None
            if sequence_path is None
            else sequence_path.name,
            "relaxation_sequence_frame_count": len(
                relaxation.heightmap_sequence
            ),
            "terrain_volume_m3": ledger.current_terrain_volume_m3,
            "cumulative_removed_volume_m3": ledger.removed_volume_m3,
            "cumulative_boundary_outflow_m3": ledger.boundary_outflow_m3,
            "cumulative_numerical_error_m3": ledger.numerical_error_m3,
            "removed_mass_estimate_kg": ledger.removed_mass_estimate_kg,
            "mass_value_kind": "density-based estimate",
            "diagnostics": _jsonable(diagnostics or {}),
        }
        self._action_records.append(record)
        self._write_action_log()
        self._write_volume_log()
        self._write_frame_logs()
        return dict(record)

    def finalize_episode(
        self,
        *,
        expected_action_count: int | None = None,
    ) -> Path:
        """Write H0-HN and final frame counts after all actions commit."""

        action_count = len(self._action_records)
        if expected_action_count is not None and action_count != expected_action_count:
            raise ValueError(
                "[ActionRecorder] unexpected action count; "
                f"expected={expected_action_count}, actual={action_count}"
            )
        heightmaps = [
            np.asarray(
                np.load(self.episode_dir / "H_initial.npy", allow_pickle=False),
                dtype=np.float32,
            )
        ]
        for record in self._action_records:
            suffix = f"{int(record['action_index']):03d}"
            heightmaps.append(
                np.asarray(
                    np.load(
                        self.episode_dir / f"H_stable_{suffix}.npy",
                        allow_pickle=False,
                    ),
                    dtype=np.float32,
                )
            )
        shapes = {item.shape for item in heightmaps}
        if len(shapes) != 1 or not all(np.all(np.isfinite(item)) for item in heightmaps):
            raise ValueError(
                f"[ActionRecorder] H0-HN shape/value mismatch; shapes={sorted(shapes)}"
            )
        sequence = np.ascontiguousarray(np.stack(heightmaps, axis=0), dtype=np.float32)
        output = self.episode_dir / "H0_H6.npz"
        np.savez_compressed(output, heightmaps_m=sequence)
        assert self._metadata_document is not None
        self._metadata_document.update(
            {
                "completed_action_count": action_count,
                "full_frame_log_count": len(self._frame_records),
                "effective_excavation_event_count": len(
                    self._effective_excavation_events
                ),
                "h0_h6_file": output.name,
                "h0_h6_shape": list(sequence.shape),
                "h0_h6_dtype": str(sequence.dtype),
            }
        )
        self._write_metadata()
        self._write_frame_logs()
        return output

    def reset(self) -> None:
        """Record a runtime reset without deleting already captured evidence."""

        self._reset_count += 1
        if self._episode_dir is not None:
            self._write_action_log()

    @property
    def frame_count(self) -> int:
        return len(self._frame_records)

    @property
    def effective_excavation_event_count(self) -> int:
        return len(self._effective_excavation_events)

    def _write_bounded_relaxation_sequence(
        self,
        suffix: str,
        relaxation: RelaxationResult,
    ) -> Path | None:
        diagnostics = relaxation.diagnostics
        if not bool(diagnostics.get("sequence_enabled", False)):
            return None
        dtype = np.dtype(str(diagnostics["sequence_dtype"]))
        frames = relaxation.heightmap_sequence
        max_frames = int(diagnostics["sequence_max_frames"])
        memory_limit_mb = float(diagnostics["sequence_memory_limit_mb"])
        if len(frames) > max_frames:
            raise RuntimeError(
                "[ActionRecorder] solver returned more sequence frames than allowed; "
                f"frames={len(frames)}, max_frames={max_frames}"
            )
        required_bytes = len(frames) * frames[0].size * dtype.itemsize
        if required_bytes > memory_limit_mb * 1024 * 1024:
            raise RuntimeError(
                "[ActionRecorder] relaxation sequence exceeds configured memory; "
                f"required_mb={required_bytes / 1024**2:.3f}, "
                f"limit_mb={memory_limit_mb:.3f}"
            )
        path = self.episode_dir / f"H_relax_sequence_{suffix}.npy"
        mapped = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(len(frames), *frames[0].shape),
        )
        for index, frame in enumerate(frames):
            mapped[index] = np.asarray(frame, dtype=dtype)
        mapped.flush()
        del mapped
        return path

    def _write_metadata(self) -> None:
        if self._metadata_document is None:
            raise RuntimeError("[ActionRecorder] metadata is not initialized")
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(self._metadata_document, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_frame_logs(self) -> None:
        document = {
            "schema_version": "isaac-bulk-joint-frame-log-v1",
            "frame_count": len(self._frame_records),
            "joint_position_units": (
                "articulation native: radians for revolute joints, metres for prismatic joints"
            ),
            "joint_velocity_units": (
                "articulation native per second: rad/s for revolute, m/s for prismatic"
            ),
            "pose_convention": "4x4 column-vector affine transforms; translations in metres",
            "frames": self._frame_records,
        }
        (self.episode_dir / "joint_state_log.json").write_text(
            json.dumps(document, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.episode_dir / "effective_excavation_events.json").write_text(
            json.dumps(
                {
                    "schema_version": "isaac-bulk-effective-events-v1",
                    "event_definition": "physics frames with removed_volume_m3 > 0",
                    "event_count": len(self._effective_excavation_events),
                    "events": self._effective_excavation_events,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_action_log(self) -> None:
        (self.episode_dir / "action_log.json").write_text(
            json.dumps(
                {
                    "reset_count": self._reset_count,
                    "actions": self._action_records,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_volume_log(self) -> None:
        fields = [
            "action_index",
            "removed_volume_m3",
            "boundary_outflow_m3",
            "terrain_volume_m3",
            "cumulative_removed_volume_m3",
            "cumulative_boundary_outflow_m3",
            "cumulative_numerical_error_m3",
            "removed_mass_estimate_kg",
            "mass_value_kind",
        ]
        with (self.episode_dir / "volume_log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in self._action_records:
                writer.writerow({field: record.get(field) for field in fields})
