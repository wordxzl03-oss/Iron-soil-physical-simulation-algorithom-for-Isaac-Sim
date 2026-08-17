"""Explicit DEVICE checkpoint boundaries for production GPU terrain replay.

This module is deliberately outside the normal physics step.  Full-field
downloads/uploads are permitted only when an acceptance runner names a
checkpoint boundary; they are never hidden inside ``EarthmovingPhysicsCore``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .gpu_runtime_metadata import GpuRuntimeMetadata
from .v2_physics_core import EarthmovingPhysicsCore


FLOAT_FIELDS = (
    "z_base",
    "b_eff",
    "mobile",
    "momentum_x",
    "momentum_y",
    "track_rut",
    "avalanche_first_activation_time",
    "avalanche_last_activation_time",
    "avalanche_owned_surface",
    "avalanche_owned_export_baseline",
    "mobile_export_cumulative",
    "mobile_flux_export_cumulative",
    "avalanche_r2m_cumulative",
    "avalanche_m2r_cumulative",
)
INT_FIELDS = (
    "active_mask",
    "material_mask",
    "frontier_reached",
    "avalanche_latch",
    "avalanche_previous_component",
    "avalanche_activation_count",
    "deposition_mask",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_device_checkpoint(
    core: EarthmovingPhysicsCore,
    path: str | Path,
    *,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist the complete replay-relevant production GPU state."""

    if core.runtime_backend != "GPU_RUNTIME":
        raise RuntimeError("DEVICE_CHECKPOINT_REQUIRES_GPU_RUNTIME")
    state = core.device_state
    chain = core.gpu_chain
    metadata = core.gpu_metadata
    if state is None or chain is None or metadata is None:
        raise RuntimeError("DEVICE_CHECKPOINT_CORE_NOT_INITIALIZED")
    arrays: dict[str, np.ndarray] = {}
    for name in (*FLOAT_FIELDS, *INT_FIELDS):
        arrays[name] = np.asarray(state.runtime.download(name)).reshape(
            state.shape
        ).copy()
    terrain_hash = sha256_array(arrays["b_eff"])
    record = {
        "schema": "390F_PRODUCTION_DEVICE_CHECKPOINT/v2",
        "terrain_state_semantics": {
            "authoritative": ["z_base", "b_eff", "mobile", "momentum_x", "momentum_y"],
            "H_free": "b_eff + mobile",
            "h_resting": "b_eff - z_base",
        },
        "runtime_backend": core.runtime_backend,
        "state_authority": state.authority.value,
        "shape_yx": list(state.shape),
        "resolution_xy_m": [core.grid.dx, core.grid.dy],
        "device_timestamp_s": state.timestamp_device_s,
        "metadata": metadata.checkpoint_record(),
        "large_avalanche": {
            "persistence_s": chain.large_avalanche._persistence_s,
            "has_previous": chain.large_avalanche._has_previous,
        },
        "minislope": {
            "active_tiles": core._gpu_frontier_active_tiles.tolist(),
            "iterations": core._gpu_frontier_iterations,
            "pending": core._static_relaxation_pending,
        },
        "dirty_tiles": sorted(state._dirty_tile_ids),
        "terrain_hash_sha256": terrain_hash,
        "provenance": dict(provenance),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **arrays,
        checkpoint_record_json=np.asarray(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
        ),
    )
    result = dict(record)
    result["checkpoint_path"] = str(target)
    result["checkpoint_hash_sha256"] = sha256_file(target)
    return result


def restore_device_checkpoint(
    core: EarthmovingPhysicsCore,
    path: str | Path,
) -> dict[str, Any]:
    """Restore a checkpoint into the same production GPU Core implementation."""

    if core.runtime_backend != "GPU_RUNTIME":
        raise RuntimeError("DEVICE_CHECKPOINT_REQUIRES_GPU_RUNTIME")
    state = core.device_state
    chain = core.gpu_chain
    if state is None or chain is None:
        raise RuntimeError("DEVICE_CHECKPOINT_CORE_NOT_INITIALIZED")
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        record = json.loads(str(archive["checkpoint_record_json"]))
        if tuple(record["shape_yx"]) != state.shape:
            raise RuntimeError("DEVICE_CHECKPOINT_SHAPE_MISMATCH")
        if list(record["resolution_xy_m"]) != [core.grid.dx, core.grid.dy]:
            raise RuntimeError("DEVICE_CHECKPOINT_RESOLUTION_MISMATCH")
        fields = {
            name: np.asarray(archive[name]).copy()
            for name in (*FLOAT_FIELDS, *INT_FIELDS)
            if name in archive
        }
        legacy_migration = "NONE_V2_NATIVE"
        if "b_eff" not in fields:
            if "resting" not in archive:
                raise RuntimeError("DEVICE_CHECKPOINT_MISSING_STATIC_BED")
            fields["b_eff"] = np.asarray(archive["resting"]).copy()
            fields["z_base"] = np.zeros_like(fields["b_eff"])
            legacy_migration = (
                "UNCALIBRATED_SNAPSHOT_COMPATIBILITY_ADAPTER:"
                "B_EFF=H_FREE_OLD-H_MOBILE_OLD;Z_BASE=0"
            )
        elif "z_base" not in fields:
            fields["z_base"] = np.zeros_like(fields["b_eff"])
            legacy_migration = (
                "UNCALIBRATED_SNAPSHOT_COMPATIBILITY_ADAPTER:"
                "B_EFF_DIRECT;IMPLICIT_Z_BASE=0"
            )
        # Checkpoints written before flow/arrest closure do not contain the
        # per-tranche free-surface ownership reference.  Reconstruct it from
        # the authoritative checkpoint fields: a latched tranche owns the
        # current surface until actual net departure is observed.
        if "avalanche_owned_surface" not in fields:
            fields["avalanche_owned_surface"] = (
                fields["b_eff"] + fields["mobile"]
            )
        for name in (
            "avalanche_owned_export_baseline",
            "mobile_export_cumulative",
        ):
            if name not in fields:
                fields[name] = np.zeros_like(fields["b_eff"])
    state.restore_checkpoint_fields(
        fields,
        timestamp_s=float(record["device_timestamp_s"]),
        source="checkpoint",
    )
    core.gpu_metadata = GpuRuntimeMetadata.from_checkpoint_record(
        state, dict(record["metadata"])
    )
    chain.large_avalanche._persistence_s = float(
        record["large_avalanche"]["persistence_s"]
    )
    chain.large_avalanche._has_previous = bool(
        record["large_avalanche"]["has_previous"]
    )
    core._gpu_frontier_active_tiles = np.asarray(
        record["minislope"]["active_tiles"], dtype=np.int32
    )
    core._gpu_frontier_iterations = int(record["minislope"]["iterations"])
    core._static_relaxation_pending = bool(record["minislope"]["pending"])
    core._static_relaxation_active_tiles = int(
        core._gpu_frontier_active_tiles.size
    )
    state._dirty_tile_ids.clear()
    state.mark_dirty_tiles(record.get("dirty_tiles", []))
    restored_hash = sha256_array(fields["b_eff"])
    if restored_hash != record["terrain_hash_sha256"]:
        raise RuntimeError("DEVICE_CHECKPOINT_TERRAIN_HASH_MISMATCH")
    result = dict(record)
    result["legacy_checkpoint_migration"] = legacy_migration
    result["checkpoint_path"] = str(source)
    result["checkpoint_hash_sha256"] = sha256_file(source)
    return result
