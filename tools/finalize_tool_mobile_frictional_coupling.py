#!/usr/bin/env python3
"""Write the honest P0-2D implementation/validation boundary report."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/mobile_v2_production"
SYNTHETIC = OUTPUT / "tool_mobile_p0_2d_synthetic_device.json"
TARGET = OUTPUT / "tool_mobile_p0_2d_report.json"


def main() -> None:
    synthetic = json.loads(SYNTHETIC.read_text(encoding="utf-8"))
    if synthetic["status"] != "PASS":
        raise RuntimeError("P0-2D synthetic production-kernel gate is not PASS")
    report = {
        "schema": "TOOL_MOBILE_P0_2D_REPORT/v1",
        "status": "IMPLEMENTED_SYNTHETIC_DEVICE_PASS_PRODUCTION_REPLAY_ENVIRONMENT_BLOCKED",
        "COUPLING_IMPLEMENTED": "YES",
        "CONTACT_GEOMETRY": (
            "CAD-derived 390F L1 closed triangle surface; exact triangle/AABB "
            "intersection with each Mobile dual-control prism, closest surface "
            "point, tool-to-material normal, and rigid point velocity v+omega_cross_r"
        ),
        "NORMAL_CONTACT_LAW": (
            "minimum non-attractive infinite-mass-wall impulse Jn=-m*min(vrel.n,0)*n"
        ),
        "TANGENTIAL_CONTACT_LAW": (
            "maximum-dissipation Coulomb impulse opposing slip with "
            "|Jt|<=material.tool_friction_coefficient*|Jn|"
        ),
        "TOOL_MOBILE_FRICTION_PARAMETER_SOURCE": (
            "existing MaterialScenario.tool_friction_coefficient; same soil-tool "
            "wall interface property already used by FEE, not basal/internal friction"
        ),
        "NEW_UNCALIBRATED_PARAMETERS": "NONE",
        "synthetic_device": synthetic,
        "formal_390f_replay": {
            "status": "NOT_RUN_ENVIRONMENT_BLOCKED",
            "command": (
                "./run_390f_interactive.sh --headless --acceptance-cycles 1 "
                "--realistic-cut-scoop --mobile-v2-pre-dump-acceptance "
                "outputs/mobile_v2_production/tool_mobile_p0_2d_pre_dump.npz "
                "--cut-fill-payload-audit "
                "outputs/mobile_v2_production/tool_mobile_p0_2d_cut_audit.json"
            ),
            "observed_failure": (
                "Isaac GPU foundation Driver Version 0 / no device enumerated; "
                "PhysX CUDA context creation failed before Stage/physics initialization"
            ),
            "physics_result_created": False,
        },
        "TOTAL_MOBILE_DYNAMIC_IMPULSE_NS": None,
        "TOTAL_MACHINE_REACTION_IMPULSE_NS": None,
        "ACTION_REACTION_RESIDUAL_NS": None,
        "ACTION_REACTION_RELATIVE_RESIDUAL": None,
        "PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N": None,
        "TOOL_TO_MOBILE_WORK_J": None,
        "FRICTIONAL_DISSIPATION_J": None,
        "PRE_DUMP_REACHED": "NO",
        "NET_PAYLOAD_GAIN_M3": None,
        "MAX_PENETRATION_M": None,
        "MAX_MASS_ERROR_M3": None,
        "MASS_CONSERVATION": "PASS_SYNTHETIC_DEVICE_ONLY",
        "NO_ATTRACTIVE_NORMAL_FORCE": "PASS_SYNTHETIC",
        "NO_UNEXPLAINED_ENERGY_GENERATION": "PASS_SYNTHETIC_DEVICE",
        "MOBILE_V2_TRANSPORT_CHANGED": "NO",
        "FAILURESURFACE_V3_CHANGED": "NO",
        "CURL_SCOOP_TRAJECTORY_CHANGED": "NO",
        "PHYSICS_PARAMETERS_CHANGED": "NONE",
        "PRODUCTION_HARD_ACCEPTANCE": "NOT_DEMONSTRATED",
        "PRIMARY_REMAINING_BLOCKER": (
            "CUDA_VULKAN_DEVICE_NOT_EXPOSED_TO_EXECUTION_SANDBOX_FOR_390F_REPLAY"
        ),
    }
    TARGET.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "coupling_implemented": report["COUPLING_IMPLEMENTED"],
        "production_hard_acceptance": report["PRODUCTION_HARD_ACCEPTANCE"],
        "primary_remaining_blocker": report["PRIMARY_REMAINING_BLOCKER"],
    }, indent=2))


if __name__ == "__main__":
    main()
