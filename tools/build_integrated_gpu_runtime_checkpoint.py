#!/usr/bin/env python3
"""Build an honest integrated GPU runtime checkpoint from production evidence.

This report deliberately separates computational feasibility from one-cycle
physics completion.  An in-flight sustained avalanche can pass the former but
cannot pass the latter until terrain settling and READY_NEXT_CYCLE occur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/390f_v2"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _statistics(values) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0}
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "p99": float(np.percentile(data, 99.0)),
        "maximum": float(np.max(data)),
    }


def _gpu_samples(count: int = 10, interval_s: float = 0.25) -> dict[str, object]:
    records = []
    for index in range(count):
        try:
            text = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=3.0,
            ).stdout.splitlines()[0]
            values = [float(value.strip()) for value in text.split(",")]
            records.append(values)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            break
        if index + 1 < count:
            time.sleep(interval_s)
    if not records:
        return {"status": "UNAVAILABLE", "sample_count": 0}
    data = np.asarray(records, dtype=np.float64)
    return {
        "status": "SAMPLED_DURING_INTEGRATED_RUN",
        "sample_count": int(len(records)),
        "interval_s": interval_s,
        "gpu_utilization_percent": _statistics(data[:, 0]),
        "memory_used_mib": _statistics(data[:, 1]),
        "power_w": _statistics(data[:, 2]),
        "temperature_c": _statistics(data[:, 3]),
    }


def _latest_gpu_run() -> Path:
    candidates = []
    for directory in (OUTPUT / "interactive_runs").glob("run_*"):
        telemetry = directory / "runtime_telemetry.json"
        if not telemetry.is_file():
            continue
        try:
            records = _load(telemetry)
        except (OSError, json.JSONDecodeError):
            continue
        if records and "gpu_large_avalanche" in records[-1].get("core_timings_ms", {}):
            candidates.append(directory)
    if not candidates:
        raise RuntimeError("NO_INTEGRATED_GPU_RUNTIME_TELEMETRY")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--skip-gpu-sampling", action="store_true")
    args = parser.parse_args()
    run_dir = (args.run_dir or _latest_gpu_run()).resolve()
    telemetry = _load(run_dir / "runtime_telemetry.json")
    if not telemetry:
        raise RuntimeError("INTEGRATED_GPU_RUNTIME_TELEMETRY_EMPTY")
    production = _load(OUTPUT / "production_gpu_core_acceptance.json")
    avalanche = _load(OUTPUT / "gpu_large_avalanche_acceptance.json")
    warp = _load(OUTPUT / "warp_backend_benchmarks.json")
    last = telemetry[-1]
    frame_ms = [record["wall_ms"]["total_frame"] for record in telemetry]
    core_ms = [record["wall_ms"]["bulk_core"] for record in telemetry]
    rtf = [1000.0 / 60.0 / max(value, 1.0e-12) for value in frame_ms]
    mass_errors = [
        abs(float(record.get("mass_error_m3", 0.0)))
        for record in telemetry
        if "mass_error_m3" in record
    ]
    transitions_path = run_dir / "state_transitions.json"
    transitions = _load(transitions_path) if transitions_path.is_file() else []
    manifest_path = run_dir / "runtime_manifest.json"
    manifest = _load(manifest_path) if manifest_path.is_file() else {}
    completed = (
        manifest.get("task_completion_status") == "PASS"
        and int(manifest.get("completed_cycle_count", 0)) >= 1
    )
    failed = (run_dir / "runtime_failure.json").is_file()
    full_transfer_pass = (
        production["normal_physics_transfer"]["full_field_h2d_count"] == 0
        and production["normal_physics_transfer"]["full_field_d2h_count"] == 0
    )
    feasibility_pass = (
        production.get("status") == "PASS"
        and full_transfer_pass
        and float(np.mean(rtf)) >= 0.25
    )
    avalanche_runtime = last.get("large_avalanche")
    if avalanche_runtime is None:
        avalanche_runtime = {
            "classification": (
                "LARGE_SUSTAINED_AVALANCHE_IN_PROGRESS_INFERRED_FROM_MOBILE_RESERVOIR"
                if last.get("state") == "DEPOSITION" and last.get("mobile_m3", 0.0) > 1.0e-5
                else "NOT_RECORDED_BY_THIS_PRE_TELEMETRY_SCHEMA_RUN"
            ),
            "terrain_settled": False,
            "note": "classification is explicit inference; connected diagnostics require the next run schema",
        }
    whole_avalanche = next(case for case in avalanche["cases"] if case["case"] == "whole_701")
    whole_mobile = next(case for case in warp["mobile_layer"] if case["case"] == "whole_pile")
    whole_track = next(case for case in warp["track_soil"] if case["case"] == "whole_pile")
    gpu_sampling = (
        {"status": "SKIPPED_BY_REQUEST", "sample_count": 0}
        if args.skip_gpu_sampling
        else _gpu_samples()
    )
    report = {
        "schema": "390F_INTEGRATED_GPU_RUNTIME_CHECKPOINT/v1",
        "status": "PASS" if feasibility_pass else "FAIL",
        "run_dir": str(run_dir),
        "run_state": "FAILED" if failed else ("COMPLETE" if completed else "IN_PROGRESS"),
        "simulation_time_s": float(last["timestamp_s"]),
        "current_phase": last["state"],
        "CPU_REFERENCE": {
            "large_avalanche_whole_701_wall_time_ms": whole_avalanche["cpu_wall_time_ms"],
            "mobile_whole_701_wall_time_s": whole_mobile["CPU_REFERENCE"]["wall_time_s"],
        },
        "GPU_OPTIMIZED": {
            "production_core_status": production["status"],
            "large_avalanche_whole_701_wall_time_ms": whole_avalanche["gpu_wall_time_ms"],
            "mobile_whole_701_wall_time_s": whole_mobile["GPU_OPTIMIZED"]["wall_time_s"],
            "integrated_core_step_ms": _statistics(core_ms),
            "integrated_total_frame_ms": _statistics(frame_ms),
            "integrated_rtf": _statistics(rtf),
            "gpu_utilization": gpu_sampling,
        },
        "LARGE_AVALANCHE_MOBILE_PATH": {
            "subsystem_reference_acceptance": avalanche["status"],
            "runtime": avalanche_runtime,
            "current_mobile_volume_m3": float(last["mobile_m3"]),
            "current_airborne_volume_m3": float(last["airborne_m3"]),
        },
        "numerical_equivalence": {
            "final_terrain_error_linf_m": whole_avalanche["errors"]["resting_linf_m"],
            "mobile_error_linf_m": whole_avalanche["errors"]["mobile_linf_m"],
            "momentum_error_linf_m2_s": whole_avalanche["errors"]["momentum_linf_m2_s"],
            "large_avalanche_volume_error_m3": whole_avalanche["errors"]["transferred_volume_m3"],
            "large_avalanche_impulse_relative_linf": whole_avalanche["errors"]["initiation_impulse_relative_linf"],
            "track_soil_impulse_proxy_momentum_linf_m2_s": whole_track["equivalence"]["momentum_max_abs_m2_s"],
            "track_soil_volume_conservation_error_m3": whole_track["equivalence"]["resting_mobile_conservation_error_m3"],
            "integrated_mass_error_max_abs_m3": max(mass_errors, default=None),
        },
        "normal_physics_transfer": production["normal_physics_transfer"],
        "phase_transitions": transitions,
        "computational_feasibility": {
            "minimum_required_rtf": 0.25,
            "status": "PASS" if feasibility_pass else "FAIL",
        },
        "one_cycle": {
            "status": "PASS" if completed and not failed else ("FAIL" if failed else "IN_PROGRESS"),
            "terrain_settling_required": True,
            "note": "An in-flight terrain state is never accepted as the final physics result.",
        },
    }
    path = OUTPUT / "integrated_gpu_runtime_checkpoint.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "one_cycle": report["one_cycle"]["status"],
        "mean_rtf": report["GPU_OPTIMIZED"]["integrated_rtf"].get("mean"),
        "output": str(path),
    }))


if __name__ == "__main__":
    main()
