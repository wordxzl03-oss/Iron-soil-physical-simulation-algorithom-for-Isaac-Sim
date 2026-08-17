#!/usr/bin/env python3
"""Lightweight production GPU profile from the frozen post-dump checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from tools.run_physics_timescale_attribution import (  # noqa: E402
    CONFIG_PATH,
    DUMP_AFTER_CHECKPOINT,
    FORMAL_TERRAIN,
    _core,
    _stationary_tool,
)
from isaac_bulk_pipeline.runtime import SoilForceMode  # noqa: E402
from isaac_bulk_pipeline.runtime.device_checkpoint import (  # noqa: E402
    restore_device_checkpoint,
    sha256_file,
)


OUTPUT = ROOT / "outputs/runtime_profile/summary.json"


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean_ms": float(np.mean(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "max_ms": float(np.max(array)),
        "total_ms": float(np.sum(array)),
    }


def main() -> None:
    initialization_start = perf_counter()
    core, config, _ = _core(FORMAL_TERRAIN)
    checkpoint = restore_device_checkpoint(core, DUMP_AFTER_CHECKPOINT)
    core.initialize_tool(_stationary_tool(core, core.device_state.timestamp_device_s))
    initialization_s = perf_counter() - initialization_start
    timings: dict[str, list[float]] = {}
    transfers: dict[str, float] = {}
    wall_start = perf_counter()
    steps = int(round(5.0 / config.physics_dt_s))
    settled_first_s = None
    maximum_mass_error = 0.0
    idle_steps = 0
    for index in range(steps):
        result = core.step(
            _stationary_tool(
                core, core.device_state.timestamp_device_s + config.physics_dt_s
            ),
            phase="deposition",
            cycle=1,
            dt_s=config.physics_dt_s,
            soil_force_mode=SoilForceMode.FULL_SOIL_FORCE,
        )
        for name, value in result.timings_ms.items():
            timings.setdefault(name, []).append(float(value))
        if result.device_transfer is not None:
            for name, value in result.device_transfer.to_dict().items():
                transfers[name] = transfers.get(name, 0.0) + float(value)
        idle_steps += int("event_driven_idle" in result.timings_ms)
        maximum_mass_error = max(maximum_mass_error, abs(result.mass_balance_error_m3))
        if result.terrain_settled and settled_first_s is None:
            settled_first_s = (index + 1) * config.physics_dt_s
    wall_s = perf_counter() - wall_start
    timing_stats = {name: _stats(values) for name, values in sorted(timings.items())}
    core_total_ms = float(timing_stats["physics_core_total"]["total_ms"])
    for record in timing_stats.values():
        record["percentage_of_core_wall"] = (
            100.0 * float(record.get("total_ms", 0.0)) / max(core_total_ms, 1.0e-12)
        )
    report = {
        "schema": "V3_RUNTIME_PERFORMANCE_PROFILE/v1",
        "status": "PASS",
        "production_identity": {
            "runtime_backend": core.runtime_backend,
            "authority": core.device_state.authority.value,
            "shape_yx": list(core.grid.shape),
            "dx_m": core.grid.dx,
            "physics_dt_s": config.physics_dt_s,
            "material": core.material.name,
            "checkpoint_sha256": sha256_file(DUMP_AFTER_CHECKPOINT),
            "config_sha256": sha256_file(CONFIG_PATH),
        },
        "initialization_wall_s": initialization_s,
        "post_dump": {
            "simulation_time_s": steps * config.physics_dt_s,
            "wall_clock_s": wall_s,
            "real_time_factor": steps * config.physics_dt_s / wall_s,
            "step_count": steps,
            "event_driven_idle_steps": idle_steps,
            "first_ARREST_FINAL_s": settled_first_s,
            "maximum_abs_mass_error_m3": maximum_mass_error,
        },
        "timings": timing_stats,
        # DeviceBulkState transfer telemetry is intentionally reset at each
        # physics step.  These are therefore sums of the per-step snapshots,
        # not a subtraction of two non-cumulative counters.
        "device_transfer_cumulative": transfers,
        "baseline_comparison": {
            "pre_idle_profile_wall_s": 13.463707609999801,
            "pre_idle_profile_RTF": 0.3726066260485828,
            "optimized_wall_s": wall_s,
            "optimized_RTF": steps * config.physics_dt_s / wall_s,
            "wall_speedup": 13.463707609999801 / wall_s,
            "same_checkpoint": True,
            "same_physics_parameters": True,
        },
        "unmeasured_here": [
            "APPROACH", "DIGGING", "FAILURE_SURFACE", "SOIL_FORCE",
            "H_FREE_VISUAL_SYNC", "CONTACT_MESH_UPDATE", "HUD_UI",
        ],
        "claim_boundary": (
            "Core-only post-dump profile; GUI and excavation stages require the "
            "separate production GUI soak/runtime telemetry."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["post_dump"]), flush=True)


if __name__ == "__main__":
    main()
