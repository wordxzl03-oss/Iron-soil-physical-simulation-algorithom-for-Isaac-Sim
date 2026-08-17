"""Atomic NPZ + JSON episode schema for future world-model ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EPISODE_SCHEMA_VERSION = "isaac-bulk-episode/v1"
REQUIRED_ARRAYS = {
    "terrain/H_before_m",
    "terrain/H_after_interaction_m",
    "terrain/H_after_deposition_m",
    "terrain/H_stable_m",
    "vehicle/timestamp_s",
    "vehicle/pose_world_m_quat",
    "vehicle/joint_position_rad",
    "vehicle/joint_velocity_rad_s",
    "vehicle/joint_torque_nm",
    "tool/pose_world_m_quat",
    "tool/velocity_world_m_s",
    "force/soil_force_world_n",
    "force/application_point_world_m",
    "planner/planned_path_xy_yaw_articulation",
}
REQUIRED_SECTIONS = {"material", "performance", "planner", "metadata", "mass_ledger"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"[Dataset] value is not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    action_id: str
    arrays: Mapping[str, np.ndarray]
    sections: Mapping[str, Mapping[str, Any]]
    schema_version: str = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EPISODE_SCHEMA_VERSION:
            raise ValueError("[Dataset] unsupported episode schema")
        if not self.episode_id.strip() or not self.action_id.strip():
            raise ValueError("[Dataset] episode/action IDs must be non-empty")
        missing_arrays = REQUIRED_ARRAYS - set(self.arrays)
        missing_sections = REQUIRED_SECTIONS - set(self.sections)
        if missing_arrays or missing_sections:
            raise ValueError(f"[Dataset] incomplete record arrays={sorted(missing_arrays)} sections={sorted(missing_sections)}")
        if any("true_mass" in key.lower() for key in (*self.arrays.keys(), *self.sections.keys())):
            raise ValueError("[Dataset] true_mass is forbidden; mass is an assumed-density estimate")
        arrays: dict[str, np.ndarray] = {}
        for name, value in self.arrays.items():
            array = np.asarray(value)
            if array.dtype == object or not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
                raise ValueError(f"[Dataset] array {name!r} must be finite numeric data")
            copy = np.ascontiguousarray(array.copy()); copy.setflags(write=False); arrays[str(name)] = copy
        terrain_shapes = {arrays[name].shape for name in REQUIRED_ARRAYS if name.startswith("terrain/")}
        if len(terrain_shapes) != 1:
            raise ValueError("[Dataset] terrain snapshots must have identical shape")
        timestamps = arrays["vehicle/timestamp_s"]
        if timestamps.ndim != 1 or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError("[Dataset] vehicle timestamps must be 1-D and strictly increasing")
        count = timestamps.size
        for name in ("vehicle/pose_world_m_quat", "vehicle/joint_position_rad", "vehicle/joint_velocity_rad_s", "vehicle/joint_torque_nm", "tool/pose_world_m_quat", "tool/velocity_world_m_s", "force/soil_force_world_n", "force/application_point_world_m"):
            if arrays[name].shape[0] != count:
                raise ValueError(f"[Dataset] synchronized array {name!r} length mismatch")
        sections = {str(name): _jsonable(value) for name, value in self.sections.items()}
        metadata = sections["metadata"]
        for key in ("grid", "units", "tool_geometry", "vehicle_model", "material_scenario", "solver_config", "random_seed", "isaac_version", "git_commit"):
            if key not in metadata:
                raise ValueError(f"[Dataset] metadata missing {key!r}")
        object.__setattr__(self, "arrays", arrays); object.__setattr__(self, "sections", sections)


class EpisodeDatasetWriter:
    def write(self, record: EpisodeRecord, output_directory: Path | str) -> Path:
        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays_path = directory / "arrays.npz"
        temp_arrays = directory / ".arrays.npz.tmp"
        with temp_arrays.open("wb") as stream:
            np.savez_compressed(stream, **record.arrays)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp_arrays, arrays_path)
        manifest = {
            "schema_version": record.schema_version,
            "episode_id": record.episode_id,
            "action_id": record.action_id,
            "arrays_file": arrays_path.name,
            "arrays_sha256": _sha256(arrays_path),
            "array_schema": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "units": name.rsplit("_", 1)[-1] if "_" in name else "declared_in_metadata"} for name, value in record.arrays.items()},
            "sections": record.sections,
        }
        manifest_path = directory / "manifest.json"
        temp_manifest = directory / ".manifest.json.tmp"
        temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        return manifest_path


class EpisodeDatasetReader:
    def read(self, directory: Path | str) -> EpisodeRecord:
        root = Path(directory)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        arrays_path = root / manifest["arrays_file"]
        if _sha256(arrays_path) != manifest["arrays_sha256"]:
            raise ValueError("[Dataset] arrays checksum mismatch")
        with np.load(arrays_path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        return EpisodeRecord(manifest["episode_id"], manifest["action_id"], arrays, manifest["sections"], manifest["schema_version"])
