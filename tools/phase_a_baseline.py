#!/usr/bin/env python3
"""Generate and validate the frozen Phase A six-scoop baseline.

This tool deliberately has no Isaac/Kit dependency.  It uses NumPy, which is
already a project dependency, only to inspect the recorded ``.npy/.npz``
heightmaps.  Every path stored in the manifest is relative to the repository
root so the checkout and its baseline can be moved as a unit.

Examples::

    .venv/bin/python tools/phase_a_baseline.py generate
    .venv/bin/python tools/phase_a_baseline.py validate
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "isaac-bulk-phase-a-baseline-v1"
DEFAULT_EPISODE_DIR = Path("outputs/modular_25m_six_scoop/episode_0005")
DEFAULT_MANIFEST_PATH = Path("outputs/phase_a_baseline_manifest.json")

EXPECTED_EPISODE_FILES = tuple(
    sorted(
        (
            "H0_H6.npz",
            "H_initial.npy",
            "action_log.json",
            "effective_excavation_events.json",
            "joint_state_log.json",
            "metadata.json",
            "run_summary.json",
            "volume_log.csv",
            *(f"H_before_action_{index:03d}.npy" for index in range(6)),
            *(f"H_excavated_{index:03d}.npy" for index in range(6)),
            *(f"H_stable_{index:03d}.npy" for index in range(6)),
            *(f"tool_trajectory_{index:03d}.npy" for index in range(6)),
        )
    )
)

# These are the inputs and executable sources needed to explain the reference
# episode.  Historical RL/OBJ experiments are intentionally not part of this
# baseline because they are outside the modular six-scoop call path.
TRACKED_FILE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "heightmap_input": (
        "continuous_heightmap_25m_closed_dataset/sequence_000_H0_initial_m.csv",
    ),
    "configuration": (
        "configs/project_25m.yaml",
        "configs/minislope.yaml",
    ),
    "usd_asset": ("isaac_loader/wheel_loader.usd",),
    "entrypoints": (
        "isaac_loader/interactive_dig_demo.py",
        "run_isaac_demo.sh",
        "run_isaac_headless_acceptance.sh",
    ),
    "pipeline_sources": (
        "slope_model.py",
        "src/isaac_bulk_pipeline/__init__.py",
        "src/isaac_bulk_pipeline/config/__init__.py",
        "src/isaac_bulk_pipeline/config/loader.py",
        "src/isaac_bulk_pipeline/interaction/__init__.py",
        "src/isaac_bulk_pipeline/interaction/continuous_sweep.py",
        "src/isaac_bulk_pipeline/interaction/excavation_operator.py",
        "src/isaac_bulk_pipeline/robot/__init__.py",
        "src/isaac_bulk_pipeline/robot/robot_adapter.py",
        "src/isaac_bulk_pipeline/runtime/__init__.py",
        "src/isaac_bulk_pipeline/runtime/action_recorder.py",
        "src/isaac_bulk_pipeline/runtime/simulation_controller.py",
        "src/isaac_bulk_pipeline/solvers/__init__.py",
        "src/isaac_bulk_pipeline/solvers/base_solver.py",
        "src/isaac_bulk_pipeline/solvers/minimum_slope_adapter.py",
        "src/isaac_bulk_pipeline/terrain/__init__.py",
        "src/isaac_bulk_pipeline/terrain/heightmap_io.py",
        "src/isaac_bulk_pipeline/terrain/mass_ledger.py",
        "src/isaac_bulk_pipeline/terrain/terrain_grid.py",
        "src/isaac_bulk_pipeline/terrain/terrain_state.py",
        "src/isaac_bulk_pipeline/tools/__init__.py",
        "src/isaac_bulk_pipeline/tools/marker_validator.py",
        "src/isaac_bulk_pipeline/tools/tool_descriptor.py",
        "src/isaac_bulk_pipeline/tools/tool_descriptor_loader.py",
        "src/isaac_bulk_pipeline/tools/tool_kinematics_adapter.py",
        "src/isaac_bulk_pipeline/visualization/__init__.py",
        "src/isaac_bulk_pipeline/visualization/dynamic_mesh_adapter.py",
    ),
}

EXPECTED_REMOVED_VOLUMES_M3 = (
    20.44041262969698,
    15.720377833446427,
    15.129730178094626,
    10.374785433541199,
    10.341418803300193,
    14.840314708164897,
)
EXPECTED_EVENTS_PER_ACTION = (87, 126, 125, 104, 104, 115)
EXPECTED_CONTRACT: Mapping[str, Any] = {
    "episode_file_count": 32,
    "heightmaps_key": "heightmaps_m",
    "heightmaps_shape": [7, 701, 701],
    "heightmaps_dtype": "float32",
    "heightmaps_all_finite": True,
    "heightmaps_all_nonnegative": True,
    "action_count": 6,
    "action_indices": [0, 1, 2, 3, 4, 5],
    "full_frame_count": 1440,
    "frames_per_action": [240, 240, 240, 240, 240, 240],
    "effective_event_count": 661,
    "events_per_action": list(EXPECTED_EVENTS_PER_ACTION),
    "action_removed_volumes_m3": list(EXPECTED_REMOVED_VOLUMES_M3),
    "initial_to_first_action_continuity": True,
    "between_action_continuity": [True, True, True, True, True],
    "h0_h6_matches_recorded_states": True,
}

# Independent anchors prevent an accidental ``generate`` command from silently
# accepting a different run as the Phase A reference.  The complete manifest
# carries hashes for every other episode and pipeline file.
FROZEN_KEY_HASHES: Mapping[str, str] = {
    "outputs/modular_25m_six_scoop/episode_0005/H0_H6.npz": (
        "dbd94a1198a537c24b34ce474624a06c782fcc57b49e743c4bfe08e1e2433c89"
    ),
    "outputs/modular_25m_six_scoop/episode_0005/action_log.json": (
        "b946ae38beb39b0aa2a406d96223a17b1f2d1be99d3fe6decc20ed13bbf5d46e"
    ),
    "outputs/modular_25m_six_scoop/episode_0005/volume_log.csv": (
        "c8461abfdd32d8912c6051465bac4b0e7f3f20e52d1343fce7c07be4c30ec5a2"
    ),
    "configs/project_25m.yaml": (
        "41936d66f211db6258b08ed88f2a66485a11a427f5ca85a89808a9fd335a4a7e"
    ),
    "configs/minislope.yaml": (
        "4a29d946eda51d3c607771c09389e0c5b44a80e942f263c2bd782ca2be9a4df2"
    ),
    "isaac_loader/wheel_loader.usd": (
        "5f1a3b10102d1113ca865ac2b60443eca300b994b3b81330bbf73b82f663ef0b"
    ),
    "continuous_heightmap_25m_closed_dataset/sequence_000_H0_initial_m.csv": (
        "e16a9c3c757cc809cc023e4642ac6ddfb2cc4e085ee4e430edc464d79c04b458"
    ),
    "isaac_loader/interactive_dig_demo.py": (
        "a4825a2047df100455276208a36f504a4752381759b2d7139826c687393ed881"
    ),
    "src/isaac_bulk_pipeline/runtime/simulation_controller.py": (
        "e00031f0dc333005fd102d4ce8af98edade8de31433ada56352a23f05d73f0f1"
    ),
    "slope_model.py": (
        "9d67a9a612591882587161060124b7c8bd942ac33c4adb79085542130dc7d75a"
    ),
}


class BaselineValidationError(RuntimeError):
    """Raised when generation is attempted from a non-reference baseline."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(errors)
        super().__init__("Phase A baseline validation failed:\n- " + "\n- ".join(errors))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_path(repo_root: Path, path: Path | str) -> str:
    root = repo_root.resolve()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repository root: {path}") from exc


def _resolve_repo_path(repo_root: Path, relative_path: str) -> Path:
    if Path(relative_path).is_absolute():
        raise ValueError(f"manifest path must be repository-relative: {relative_path}")
    normalized = _repo_relative_path(repo_root, relative_path)
    if normalized != Path(relative_path).as_posix():
        raise ValueError(f"manifest path is not normalized: {relative_path}")
    return repo_root.resolve() / normalized


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _float_lists_match(
    actual: Sequence[float], expected: Sequence[float], tolerance: float = 1.0e-12
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(abs(float(a) - float(e)) <= tolerance for a, e in zip(actual, expected))


def _compare_contract(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"contract is missing {key!r}")
            continue
        actual_value = actual[key]
        if key == "action_removed_volumes_m3":
            if not _float_lists_match(actual_value, expected_value):
                errors.append(
                    f"contract {key} mismatch: expected {expected_value}, got {actual_value}"
                )
        elif actual_value != expected_value:
            errors.append(
                f"contract {key} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    unexpected = sorted(set(actual) - set(expected))
    if unexpected:
        errors.append(f"contract has unexpected fields: {unexpected}")
    return errors


def inspect_episode(episode_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Inspect semantic invariants in an episode without trusting its metadata."""

    errors: list[str] = []
    actual_names = sorted(path.name for path in episode_dir.iterdir() if path.is_file())
    expected_names = list(EXPECTED_EPISODE_FILES)
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        errors.append(f"episode file set mismatch; missing={missing}, extra={extra}")

    heightmaps_key = ""
    heightmaps = np.empty((0,), dtype=np.float32)
    try:
        with np.load(episode_dir / "H0_H6.npz", allow_pickle=False) as archive:
            archive_keys = list(archive.files)
            if archive_keys != ["heightmaps_m"]:
                errors.append(
                    "H0_H6.npz keys mismatch: expected ['heightmaps_m'], "
                    f"got {archive_keys}"
                )
            if "heightmaps_m" in archive:
                heightmaps_key = "heightmaps_m"
                heightmaps = np.array(archive[heightmaps_key], copy=True)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot load H0_H6.npz: {exc}")

    action_count = -1
    action_indices: list[int] = []
    action_removed: list[float] = []
    try:
        action_log = _read_json(episode_dir / "action_log.json")
        actions = action_log["actions"]
        action_count = len(actions)
        action_indices = [int(item["action_index"]) for item in actions]
        action_removed = [float(item["removed_volume_m3"]) for item in actions]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cannot inspect action_log.json: {exc}")

    full_frame_count = -1
    frames_per_action: list[int] = []
    try:
        frame_log = _read_json(episode_dir / "joint_state_log.json")
        frames = frame_log["frames"]
        full_frame_count = len(frames)
        if int(frame_log["frame_count"]) != full_frame_count:
            errors.append(
                "joint_state_log frame_count does not equal the number of frames"
            )
        frame_counter = Counter(int(item["action_index"]) for item in frames)
        frames_per_action = [frame_counter[index] for index in range(6)]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cannot inspect joint_state_log.json: {exc}")

    effective_event_count = -1
    events_per_action: list[int] = []
    try:
        event_log = _read_json(episode_dir / "effective_excavation_events.json")
        events = event_log["events"]
        effective_event_count = len(events)
        if int(event_log["event_count"]) != effective_event_count:
            errors.append(
                "effective event_count does not equal the number of event records"
            )
        event_counter = Counter(int(item["action_index"]) for item in events)
        events_per_action = [event_counter[index] for index in range(6)]
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cannot inspect effective_excavation_events.json: {exc}")

    volume_log_removed: list[float] = []
    try:
        with (episode_dir / "volume_log.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if [int(row["action_index"]) for row in rows] != list(range(6)):
            errors.append("volume_log.csv action indices are not 0..5")
        volume_log_removed = [float(row["removed_volume_m3"]) for row in rows]
    except (OSError, ValueError, TypeError, KeyError, csv.Error) as exc:
        errors.append(f"cannot inspect volume_log.csv: {exc}")

    summary_removed: list[float] = []
    try:
        summary = _read_json(episode_dir / "run_summary.json")
        summary_removed = [
            float(value) for value in summary["action_removed_volumes_m3"]
        ]
        if int(summary["full_frame_log_count"]) != full_frame_count:
            errors.append("run_summary full_frame_log_count disagrees with frame log")
        if int(summary["effective_excavation_event_count"]) != effective_event_count:
            errors.append("run_summary event count disagrees with event log")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"cannot inspect run_summary.json: {exc}")

    if action_removed and not _float_lists_match(action_removed, volume_log_removed):
        errors.append("per-action removed volumes disagree between action and CSV logs")
    if action_removed and not _float_lists_match(action_removed, summary_removed):
        errors.append("per-action removed volumes disagree between action log and summary")

    initial_to_first = False
    between_actions: list[bool] = []
    archive_matches = False
    try:
        initial = _load_array(episode_dir / "H_initial.npy")
        first_before = _load_array(episode_dir / "H_before_action_000.npy")
        initial_to_first = bool(np.array_equal(initial, first_before))
        if not initial_to_first:
            errors.append("H_before_action_000 is not identical to H_initial")

        recorded_states_match = (
            heightmaps.ndim == 3
            and heightmaps.shape[0] == 7
            and np.array_equal(heightmaps[0], initial.astype(np.float32))
        )
        previous_stable: np.ndarray | None = None
        for index in range(6):
            before = _load_array(episode_dir / f"H_before_action_{index:03d}.npy")
            excavated = _load_array(episode_dir / f"H_excavated_{index:03d}.npy")
            stable = _load_array(episode_dir / f"H_stable_{index:03d}.npy")
            for label, array in (
                ("before", before),
                ("excavated", excavated),
                ("stable", stable),
            ):
                if not bool(np.all(np.isfinite(array))):
                    errors.append(f"action {index} {label} heightmap has non-finite values")
                if not bool(np.all(array >= 0.0)):
                    errors.append(f"action {index} {label} heightmap has negative values")
            if previous_stable is not None:
                continuous = bool(np.array_equal(before, previous_stable))
                between_actions.append(continuous)
                if not continuous:
                    errors.append(
                        f"action continuity failed: before[{index}] != stable[{index - 1}]"
                    )
            if heightmaps.ndim == 3 and heightmaps.shape[0] == 7:
                recorded_states_match = recorded_states_match and bool(
                    np.array_equal(heightmaps[index + 1], stable.astype(np.float32))
                )
            previous_stable = stable
        archive_matches = bool(recorded_states_match)
        if not archive_matches:
            errors.append("H0_H6 archive does not match H_initial/H_stable action files")
    except (OSError, ValueError) as exc:
        errors.append(f"cannot inspect action heightmap continuity: {exc}")

    contract = {
        "episode_file_count": len(actual_names),
        "heightmaps_key": heightmaps_key,
        "heightmaps_shape": list(heightmaps.shape),
        "heightmaps_dtype": str(heightmaps.dtype),
        "heightmaps_all_finite": bool(np.all(np.isfinite(heightmaps))),
        "heightmaps_all_nonnegative": bool(np.all(heightmaps >= 0.0)),
        "action_count": action_count,
        "action_indices": action_indices,
        "full_frame_count": full_frame_count,
        "frames_per_action": frames_per_action,
        "effective_event_count": effective_event_count,
        "events_per_action": events_per_action,
        "action_removed_volumes_m3": action_removed,
        "initial_to_first_action_continuity": initial_to_first,
        "between_action_continuity": between_actions,
        "h0_h6_matches_recorded_states": archive_matches,
    }
    errors.extend(_compare_contract(contract, EXPECTED_CONTRACT))
    return contract, errors


def _file_record(repo_root: Path, relative_path: str) -> dict[str, Any]:
    path = _resolve_repo_path(repo_root, relative_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _episode_relative_files(repo_root: Path, episode_dir: Path) -> list[str]:
    episode_relative = _repo_relative_path(repo_root, episode_dir)
    return [f"{episode_relative}/{name}" for name in EXPECTED_EPISODE_FILES]


def build_manifest(
    repo_root: Path,
    episode_dir: Path | str = DEFAULT_EPISODE_DIR,
) -> dict[str, Any]:
    """Build a deterministic manifest, refusing anything but the frozen run."""

    root = repo_root.resolve()
    episode_relative = _repo_relative_path(root, episode_dir)
    episode_path = _resolve_repo_path(root, episode_relative)
    if not episode_path.is_dir():
        raise BaselineValidationError([f"episode directory does not exist: {episode_path}"])

    contract, errors = inspect_episode(episode_path)

    try:
        episode_records = [
            _file_record(root, path)
            for path in _episode_relative_files(root, episode_path)
        ]
        tracked_records = {
            group: [_file_record(root, path) for path in paths]
            for group, paths in TRACKED_FILE_GROUPS.items()
        }
    except (FileNotFoundError, ValueError) as exc:
        raise BaselineValidationError([str(exc)]) from exc

    all_records = {
        record["path"]: record
        for record in (
            *episode_records,
            *(record for records in tracked_records.values() for record in records),
        )
    }
    key_hashes: dict[str, str] = {}
    for path, expected_hash in FROZEN_KEY_HASHES.items():
        record = all_records.get(path)
        if record is None:
            errors.append(f"frozen key path is not tracked: {path}")
            continue
        actual_hash = str(record["sha256"])
        key_hashes[path] = actual_hash
        if actual_hash != expected_hash:
            errors.append(
                f"frozen key hash mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    if errors:
        raise BaselineValidationError(errors)

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": "phase_a_modular_25m_six_scoop_episode_0005",
        "repository_root": ".",
        "episode_dir": episode_relative,
        "contract": contract,
        "key_hashes": dict(sorted(key_hashes.items())),
        "episode_files": episode_records,
        "tracked_files": tracked_records,
    }


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def _iter_manifest_records(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    episode_files = manifest.get("episode_files", [])
    if isinstance(episode_files, list):
        yield from episode_files
    tracked_files = manifest.get("tracked_files", {})
    if isinstance(tracked_files, Mapping):
        for group in TRACKED_FILE_GROUPS:
            records = tracked_files.get(group, [])
            if isinstance(records, list):
                yield from records


def validate_manifest_data(
    repo_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate hashes and semantic contracts recorded in *manifest*."""

    root = repo_root.resolve()
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {manifest.get('schema_version')!r}"
        )
    if manifest.get("repository_root") != ".":
        errors.append("repository_root must be '.'")

    episode_relative = manifest.get("episode_dir")
    episode_path: Path | None = None
    if not isinstance(episode_relative, str):
        errors.append("episode_dir must be a repository-relative string")
    else:
        try:
            episode_path = _resolve_repo_path(root, episode_relative)
        except ValueError as exc:
            errors.append(str(exc))

    actual_contract: dict[str, Any] = {}
    if episode_path is not None and episode_path.is_dir():
        actual_contract, content_errors = inspect_episode(episode_path)
        errors.extend(content_errors)
    elif episode_path is not None:
        errors.append(f"episode directory does not exist: {episode_path}")

    manifest_contract = manifest.get("contract")
    if not isinstance(manifest_contract, Mapping):
        errors.append("manifest contract must be an object")
    else:
        errors.extend(_compare_contract(manifest_contract, EXPECTED_CONTRACT))
        if actual_contract and manifest_contract != actual_contract:
            errors.append("manifest contract does not match current episode content")

    expected_episode_paths = (
        set(_episode_relative_files(root, episode_path)) if episode_path else set()
    )
    manifest_episode_records = manifest.get("episode_files", [])
    if not isinstance(manifest_episode_records, list):
        errors.append("episode_files must be a list")
        manifest_episode_records = []
    recorded_episode_paths = {
        item.get("path") for item in manifest_episode_records if isinstance(item, Mapping)
    }
    if expected_episode_paths and recorded_episode_paths != expected_episode_paths:
        errors.append("manifest does not contain exactly the 32 expected episode files")

    tracked_files = manifest.get("tracked_files")
    if not isinstance(tracked_files, Mapping):
        errors.append("tracked_files must be an object")
        tracked_files = {}
    if set(tracked_files) != set(TRACKED_FILE_GROUPS):
        errors.append("tracked file groups do not match the frozen baseline schema")
    for group, expected_paths in TRACKED_FILE_GROUPS.items():
        records = tracked_files.get(group, [])
        recorded_paths = [
            item.get("path") for item in records if isinstance(item, Mapping)
        ] if isinstance(records, list) else []
        if recorded_paths != list(expected_paths):
            errors.append(f"tracked paths mismatch in group {group!r}")

    actual_hashes: dict[str, str] = {}
    seen_paths: set[str] = set()
    for record in _iter_manifest_records(manifest):
        if not isinstance(record, Mapping):
            errors.append("manifest file record must be an object")
            continue
        relative_path = record.get("path")
        if not isinstance(relative_path, str):
            errors.append("manifest file record has a non-string path")
            continue
        if relative_path in seen_paths:
            errors.append(f"duplicate manifest file record: {relative_path}")
            continue
        seen_paths.add(relative_path)
        try:
            path = _resolve_repo_path(root, relative_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.is_file():
            errors.append(f"tracked file is missing: {relative_path}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        actual_hashes[relative_path] = actual_hash
        if record.get("size_bytes") != actual_size:
            errors.append(
                f"size mismatch for {relative_path}: expected "
                f"{record.get('size_bytes')}, got {actual_size}"
            )
        if record.get("sha256") != actual_hash:
            errors.append(
                f"SHA-256 mismatch for {relative_path}: expected "
                f"{record.get('sha256')}, got {actual_hash}"
            )

    manifest_key_hashes = manifest.get("key_hashes")
    if not isinstance(manifest_key_hashes, Mapping):
        errors.append("key_hashes must be an object")
        manifest_key_hashes = {}
    if dict(manifest_key_hashes) != dict(FROZEN_KEY_HASHES):
        errors.append("manifest key_hashes differ from the frozen independent anchors")
    for path, expected_hash in FROZEN_KEY_HASHES.items():
        actual_hash = actual_hashes.get(path)
        if actual_hash is None:
            try:
                actual_hash = sha256_file(_resolve_repo_path(root, path))
            except (FileNotFoundError, ValueError):
                errors.append(f"cannot verify frozen key hash: {path}")
                continue
        if actual_hash != expected_hash:
            errors.append(
                f"frozen key hash mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    return {
        "ok": not errors,
        "schema_version": SCHEMA_VERSION,
        "baseline_id": manifest.get("baseline_id"),
        "episode_dir": episode_relative,
        "verified_episode_file_count": len(recorded_episode_paths),
        "verified_tracked_file_count": sum(
            len(paths) for paths in TRACKED_FILE_GROUPS.values()
        ),
        "contract": actual_contract,
        "errors": errors,
    }


def validate_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "baseline_id": None,
            "episode_dir": None,
            "verified_episode_file_count": 0,
            "verified_tracked_file_count": 0,
            "contract": {},
            "errors": [f"cannot read manifest {manifest_path}: {exc}"],
        }
    if not isinstance(manifest, Mapping):
        return {
            "ok": False,
            "schema_version": SCHEMA_VERSION,
            "baseline_id": None,
            "episode_dir": None,
            "verified_episode_file_count": 0,
            "verified_tracked_file_count": 0,
            "contract": {},
            "errors": ["manifest root must be a JSON object"],
        }
    return validate_manifest_data(repo_root, manifest)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("generate", "validate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--repo-root",
            type=Path,
            default=Path(__file__).resolve().parents[1],
            help="repository root (default: inferred from this script)",
        )
        command_parser.add_argument(
            "--manifest",
            type=Path,
            default=DEFAULT_MANIFEST_PATH,
            help=f"manifest path (default: {DEFAULT_MANIFEST_PATH})",
        )
        if command == "generate":
            command_parser.add_argument(
                "--episode",
                type=Path,
                default=DEFAULT_EPISODE_DIR,
                help=f"reference episode (default: {DEFAULT_EPISODE_DIR})",
            )
    return parser


def _absolute_cli_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root.resolve() / path


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest_path = _absolute_cli_path(repo_root, args.manifest)
    if args.command == "generate":
        try:
            manifest = build_manifest(repo_root, args.episode)
            write_manifest(manifest, manifest_path)
        except (BaselineValidationError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(
            json.dumps(
                {
                    "ok": True,
                    "manifest": _repo_relative_path(repo_root, manifest_path),
                    "episode_file_count": len(manifest["episode_files"]),
                    "tracked_file_count": sum(
                        len(records) for records in manifest["tracked_files"].values()
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    report = validate_manifest(repo_root, manifest_path)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
