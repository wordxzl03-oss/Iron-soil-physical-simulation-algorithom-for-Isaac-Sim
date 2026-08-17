#!/usr/bin/env python3
"""Run the complete 4x3 pure-CPU Phase-K scalability matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from time import time

from isaac_bulk_pipeline.dataset import OfflineScalabilityBenchmark, SoilForceSensitivityStudy


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phase_k_scalability_benchmark.json")
    args = parser.parse_args()
    results = OfflineScalabilityBenchmark(args.repeats).run_matrix()
    cases = []
    for result in results:
        cases.append(
            {
                "resolution": result.case.resolution,
                "tool_size": result.case.tool_size,
                "tool_width_m": result.case.tool_width_m,
                "active_bbox_grid": list(result.active_bbox_grid),
                "mobile_balance_error_m3": result.mobile_balance_error_m3,
                "candidate_count": result.candidate_count,
                "statistics": result.statistics(),
                "checks": {
                    "finite_volume_conservative": abs(result.mobile_balance_error_m3) <= 1e-9,
                    "attack_candidates_nonempty": result.candidate_count > 0,
                },
            }
        )
    passed = all(all(case["checks"].values()) for case in cases)
    sensitivity_study = SoilForceSensitivityStudy()
    sensitivity = sensitivity_study.run()
    sensitivity_report = {
        name: {
            "samples": [
                {"value": item.value, "resultant_force_n": item.resultant_force_n}
                for item in samples
            ],
            "strictly_increasing": sensitivity_study.monotonic(samples),
        }
        for name, samples in sensitivity.items()
    }
    passed = passed and all(item["strictly_increasing"] for item in sensitivity_report.values())
    report = {
        "schema_version": "isaac-bulk-phase-k-scalability/v1",
        "status": "PASS" if passed else "FAIL",
        "scope": "PURE_CPU_ONLY",
        "generated_unix_s": time(),
        "host": {"python": platform.python_version(), "platform": platform.platform()},
        "repeats": args.repeats,
        "case_count": len(cases),
        "cases": cases,
        "uncalibrated_parameter_sensitivity": sensitivity_report,
        "not_measured": ["isaac_vehicle_physics", "physx_soil_feedback", "visual_mesh_usd_update", "collision_proxy_recook", "continuous_loading_cycle", "path_tracking_in_isaac"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "case_count": len(cases), "output": str(args.output)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
