#!/usr/bin/env python3
"""Reproducible pure acceptance and honest Isaac-runtime gate for Phases F-K."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase_f_to_k_acceptance.json"

PHASES = {
    "F": {"tests": ["tests/test_phase_f_bulk_interaction.py"], "sources": ["src/isaac_bulk_pipeline/bulk_interaction"], "config": "configs/phase_f_transport.yaml", "runtime_gate": "full bucket trajectory -> terrain material interaction in Isaac"},
    "G": {"tests": ["tests/test_phase_g_spill_dump.py"], "sources": ["src/isaac_bulk_pipeline/bulk_exchange"], "config": "configs/phase_g_spill_dump.yaml", "runtime_gate": "visual bucket dump -> airborne -> terrain redeposition in Isaac"},
    "H": {"tests": ["tests/test_phase_h_soil_force.py", "tests/test_phase_i_cycle_coordinator.py"], "sources": ["src/isaac_bulk_pipeline/soil_force"], "config": "configs/phase_h_soil_force.yaml", "runtime_gate": "measured vehicle response/energy change with applied soil force"},
    "I": {"tests": ["tests/test_phase_i_operation.py", "tests/test_phase_i_cycle_coordinator.py"], "sources": ["src/isaac_bulk_pipeline/operation"], "config": "configs/phase_i_operation.yaml", "runtime_gate": "continuous drive-dig-reverse-dump-return Isaac episode without pose writes"},
    "J": {"tests": ["tests/test_phase_j_planning.py"], "sources": ["src/isaac_bulk_pipeline/planning"], "config": "configs/phase_j_planner.yaml", "runtime_gate": "Isaac state-lattice tracking and action-level replanning to target termination"},
    "K": {"tests": ["tests/test_phase_k_dataset_benchmark.py", "tests/test_phase_k_debug_visualization.py"], "sources": ["src/isaac_bulk_pipeline/dataset", "src/isaac_bulk_pipeline/visualization/debug_layers.py"], "config": "configs/phase_k_dataset_benchmark.yaml", "runtime_gate": "complete synchronized episode captured from the integrated Isaac runtime"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(path.rglob("*.py")):
            digest.update(child.relative_to(path).as_posix().encode()); digest.update(b"\0"); digest.update(child.read_bytes()); digest.update(b"\0")
    else:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    reports = {}
    pure_pass = True
    for phase, spec in PHASES.items():
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *spec["tests"]],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        passed = completed.returncode == 0
        pure_pass &= passed
        reports[phase] = {
            "pure_acceptance": "PASS" if passed else "FAIL",
            "pytest_command": [sys.executable, "-m", "pytest", "-q", *spec["tests"]],
            "pytest_output": completed.stdout.strip().splitlines()[-2:],
            "source_sha256": {path: sha256(ROOT / path) for path in spec["sources"]},
            "config": spec["config"],
            "config_sha256": sha256(ROOT / spec["config"]),
            "isaac_runtime_acceptance": "NOT_EXECUTED_IN_PURE_ACCEPTANCE",
            "runtime_gate_not_yet_proven": spec["runtime_gate"],
        }
    benchmark_path = ROOT / "outputs/phase_k_scalability_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.is_file() else None
    if benchmark is None or benchmark.get("status") != "PASS" or benchmark.get("case_count") != 12:
        pure_pass = False
    report = {
        "schema_version": "isaac-bulk-phase-f-k-acceptance/v1",
        "generated_unix_s": time(),
        "overall_status": "BLOCKED" if pure_pass else "FAIL",
        "pure_cpu_status": "PASS" if pure_pass else "FAIL",
        "isaac_runtime_status": "NOT_EXECUTED_REQUIRES_FRESH_CLOSED_LOOP_EVIDENCE",
        "claim_boundary": "Pure numerical mechanisms and static public-API adapters pass. No new Phase D/H/I/J/K Isaac execution is claimed.",
        "phases": reports,
        "phase_k_scalability_report": "outputs/phase_k_scalability_benchmark.json",
        "phase_k_scalability_sha256": None if benchmark is None else sha256(benchmark_path),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall_status": report["overall_status"], "pure_cpu_status": report["pure_cpu_status"], "output": str(OUTPUT)}, sort_keys=True))
    return 0 if pure_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
