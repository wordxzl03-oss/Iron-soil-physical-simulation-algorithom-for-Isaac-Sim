#!/usr/bin/env python3
"""Materialize the P0-2C causal audit from frozen production evidence.

This script is intentionally read-only with respect to production physics.  It
fails if the causal source signatures have changed, and it never interprets a
trivial all-zero action/reaction ledger as validated coupling.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac_bulk_pipeline.soil_force import (
    ToolMobileFrameContractLedger,
    ToolMobileSubstepContract,
    external_acceleration_impulse_ns,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "mobile_v2_production"
FROZEN_AUDIT = OUTPUT / "cut_fill_realistic_trajectory_candidate2_audit.json"
FROZEN_REPORT = OUTPUT / "cut_fill_realistic_trajectory_report.json"


def dump(name: str, value: object) -> None:
    target = OUTPUT / name
    target.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def assert_source_causality() -> None:
    failure = (ROOT / "src/isaac_bulk_pipeline/runtime/gpu_failure_bridge.py").read_text()
    chain = (ROOT / "src/isaac_bulk_pipeline/runtime/gpu_bulk_operator_chain.py").read_text()
    mobile = (ROOT / "src/isaac_bulk_pipeline/bulk_interaction/warp_mobile_v2.py").read_text()
    required = (
        (failure, 'apply_host_indices("material_mask", forcing, 1'),
        (failure, "activation_tool_impulse_on_mobile_terrain_ns=np.zeros(3"),
        (chain, "result = self.mobile.step_resident(dt_s)"),
        (mobile, 'self.runtime.arrays["v2_external_x"].zero_()'),
        (mobile, 'self.runtime.arrays["v2_external_y"].zero_()'),
        (mobile, "tool_impulse_on_mobile_terrain_ns=np.zeros(3)"),
        (mobile, "tool_work_j=0.0"),
    )
    missing = [needle for source, needle in required if needle not in source]
    if missing:
        raise RuntimeError(f"P0-2C source causality changed; missing {missing}")


def first_expected(rows: list[dict[str, object]]) -> tuple[int, dict[str, object]]:
    # The frozen replay did not retain the compact material_mask.  Mouth-prism
    # Mobile plus positive relative-normal speed is the earliest retained,
    # conservative geometric proxy.  It is explicitly not represented as a
    # Mobile CFL substep measurement.
    for index, row in enumerate(rows):
        volume = float(row["proximity"]["mouth_prism_m3"])
        relative = float(row["mean_positive_relative_normal_speed_m_s"])
        if row["phase"] == "CUT_AND_FILL" and volume > 1.0e-12 and relative > 0.0:
            return index, row
    raise RuntimeError("frozen replay contains no expected tool-Mobile event")


def synthetic_contract() -> dict[str, object]:
    depth = np.asarray([[0.1, 0.2], [0.3, 0.4]])
    area = np.asarray([[0.25, 0.5], [0.75, 1.0]])
    acceleration = np.asarray(
        [[[2.0, -1.0], [1.0, 3.0]], [[-2.0, 0.5], [4.0, -3.0]]]
    )
    density = 1370.0
    dt = 0.0125
    expected = density * np.sum(
        area[..., None] * depth[..., None] * acceleration * dt,
        axis=(0, 1),
    )
    measured = external_acceleration_impulse_ns(
        depth, area, acceleration, bulk_density_kg_m3=density, dt_s=dt
    )
    np.testing.assert_allclose(measured, expected, rtol=0.0, atol=1.0e-14)

    single = ToolMobileSubstepContract(
        accepted_tool_to_mobile_impulse_xy_ns=np.asarray([12.5, -3.25]),
        measured_mobile_source_delta_xy_ns=np.asarray([12.5, -3.25]),
        contact_active=True,
        dt_sub_s=1.0 / 240.0,
    )
    impulses = (
        np.asarray([1.0, 2.0]),
        np.asarray([-0.25, 3.0]),
        np.asarray([4.5, -1.0]),
    )
    frame = ToolMobileFrameContractLedger()
    for index, impulse in enumerate(impulses):
        frame.append(
            ToolMobileSubstepContract(
                accepted_tool_to_mobile_impulse_xy_ns=impulse,
                measured_mobile_source_delta_xy_ns=impulse,
                contact_active=True,
                dt_sub_s=(index + 1) / 1000.0,
            )
        )
    zero = ToolMobileSubstepContract(
        accepted_tool_to_mobile_impulse_xy_ns=np.zeros(2),
        measured_mobile_source_delta_xy_ns=np.zeros(2),
        contact_active=False,
        dt_sub_s=1.0 / 60.0,
    )
    return {
        "classification": "SYNTHETIC_INTERFACE_CONTRACT_ONLY_NOT_PRODUCTION_COUPLING",
        "external_source_momentum_test": {
            "status": "PASS",
            "expected_impulse_xy_ns": expected.tolist(),
            "measured_impulse_xy_ns": measured.tolist(),
            "absolute_residual_ns": float(np.linalg.norm(measured - expected)),
        },
        "action_reaction_unit_test": {
            "status": "PASS",
            "mobile_impulse_xy_ns": single.measured_mobile_source_delta_xy_ns.tolist(),
            "machine_reaction_xy_ns": single.machine_reaction_impulse_xy_ns.tolist(),
            "residual_xy_ns": single.action_reaction_residual_xy_ns.tolist(),
        },
        "multi_substep_accumulation": {
            "status": "PASS",
            "substep_count": len(impulses),
            "mobile_impulse_xy_ns": frame.measured_mobile_source_delta_xy_ns.tolist(),
            "machine_reaction_xy_ns": frame.machine_reaction_impulse_xy_ns.tolist(),
            "residual_xy_ns": frame.action_reaction_residual_xy_ns.tolist(),
            "dt_double_scaling": "NO",
        },
        "zero_contact": {
            "status": "PASS",
            "mobile_impulse_xy_ns": zero.measured_mobile_source_delta_xy_ns.tolist(),
            "machine_reaction_xy_ns": zero.machine_reaction_impulse_xy_ns.tolist(),
        },
    }


def main() -> None:
    assert_source_causality()
    audit = json.loads(FROZEN_AUDIT.read_text(encoding="utf-8"))
    trajectory = json.loads(FROZEN_REPORT.read_text(encoding="utf-8"))
    rows = audit["time_series"]
    first_index, first = first_expected(rows)
    active_rows = [
        row
        for row in rows
        if float(row["proximity"]["mouth_prism_m3"]) > 1.0e-12
        and float(row["mean_positive_relative_normal_speed_m_s"]) > 0.0
    ]

    timeseries = {
        "schema": "TOOL_MOBILE_MOMENTUM_TIMESERIES/v1",
        "status": "FROZEN_PRODUCTION_FRAME_EVIDENCE_SUBSTEP_TELEMETRY_UNAVAILABLE",
        "source": str(FROZEN_AUDIT.relative_to(ROOT)),
        "production_replay_identity": {
            "vehicle": "390F",
            "runtime_backend": "GPU_RUNTIME",
            "state_authority": "DEVICE",
            "grid_shape_yx": [701, 701],
            "dx_m": 0.05,
            "trajectory": "ACCEPTED_COORDINATED_CURL_SCOOP",
        },
        "granularity": "FRAME_LEVEL_ACCEPTANCE_TELEMETRY",
        "substep_limitation": (
            "The accepted frozen replay predates P0-2C and contains no Mobile-CFL "
            "substep/contact-mask archive. No substep values were fabricated. The "
            "production coupling request stage is absent, so there is no physical "
            "coupling substep to replay without a new reviewed contact law."
        ),
        "records": [
            {
                "simulation_time_s": row["simulation_time_s"],
                "physics_step": None,
                "mobile_substep_index": None,
                "dt_sub_s": None,
                "phase": row["phase"],
                "tool_state": {
                    "cutting_edge_position_terrain_m": row["cutting_edge_center_terrain_m"],
                    "cutting_edge_velocity_terrain_m_s": row["cutting_edge_velocity_terrain_m_s"],
                    "bucket_pose": None,
                    "bucket_pose_proxy_mouth_pose_terrain": row["mouth_pose_terrain"],
                    "bucket_linear_velocity_m_s": row["bucket_linear_velocity_m_s"],
                    "bucket_angular_velocity_rad_s": row["bucket_angular_velocity_rad_s"],
                },
                "contact_state": {
                    "tool_mobile_contact_active_proxy": bool(
                        float(row["proximity"]["mouth_prism_m3"]) > 1.0e-12
                        and float(row["mean_positive_relative_normal_speed_m_s"]) > 0.0
                    ),
                    "contact_cell_count": None,
                    "contact_weighted_area_m2": row["mouth_intersection_area_m2"],
                    "mobile_volume_in_contact_proxy_m3": row["proximity"]["mouth_prism_m3"],
                    "mean_contact_mobile_velocity_xy_m_s": row["local_mobile"]["mouth_prism_volume_weighted_mobile_velocity_m_s"],
                    "relative_normal_velocity_m_s": row["mean_positive_relative_normal_speed_m_s"],
                    "relative_tangential_velocity_m_s": None,
                },
                "requested_coupling": {
                    "status": "ABSENT",
                    "requested_tool_to_mobile_impulse_xy_ns": None,
                },
                "accepted_coupling": {
                    "status": "PRESENT_ZERO_INPUT",
                    "accepted_tool_to_mobile_impulse_xy_ns": [0.0, 0.0],
                },
                "mobile_source": {
                    "v2_external_xy_m_s2": [0.0, 0.0],
                    "measured_delta_mobile_momentum_due_to_external_source_xy_ns": [0.0, 0.0],
                },
                "machine": {
                    "requested_reaction_impulse_xy_ns": [0.0, 0.0],
                    "applied_reaction_impulse_xy_ns": [0.0, 0.0],
                    "applied_dynamic_reaction_force_xy_n": row["dynamic_momentum_force_n"][:2],
                },
                "other_force_channels": {
                    "quasi_static_soil_force_terrain_n": row["quasi_static_force_n"],
                    "payload_force_n": None,
                    "payload_force_status": "NOT_RETAINED_IN_FROZEN_AUDIT",
                    "total_machine_soil_force_terrain_n": row["applied_soil_force_n"],
                },
            }
            for row in rows
        ],
    }

    first_record = {
        "schema": "TOOL_MOBILE_FIRST_EXPECTED_COUPLING/v1",
        "status": "EXPECTED_EVENT_FOUND_CAUSAL_CHAIN_BREAKS_AT_REQUEST_STAGE",
        "definition": (
            "first frozen CUT_AND_FILL frame with mouth-prism Mobile volume > "
            "1e-12 m3 and positive relative-normal speed"
        ),
        "evidence_granularity": "FRAME_LEVEL_PROXY_NOT_MOBILE_CFL_SUBSTEP",
        "record_index": first_index,
        "simulation_time_s": first["simulation_time_s"],
        "phase": first["phase"],
        "contact_mobile_volume_proxy_m3": first["proximity"]["mouth_prism_m3"],
        "contact_weighted_area_proxy_m2": first["mouth_intersection_area_m2"],
        "relative_normal_velocity_m_s": first["mean_positive_relative_normal_speed_m_s"],
        "mean_contact_mobile_velocity_xy_m_s": first["local_mobile"]["mouth_prism_volume_weighted_mobile_velocity_m_s"],
        "bucket_linear_velocity_m_s": first["bucket_linear_velocity_m_s"],
        "cutting_edge_velocity_terrain_m_s": first["cutting_edge_velocity_terrain_m_s"],
        "requested_impulse_xy_ns": None,
        "requested_impulse_status": "ABSENT_COUPLING_LAW",
        "accepted_impulse_xy_ns": [0.0, 0.0],
        "mobile_source_momentum_change_xy_ns": [0.0, 0.0],
        "machine_reaction_impulse_xy_ns": [0.0, 0.0],
        "first_break": "REQUESTED_MOBILE_MOMENTUM_TRANSFER_ABSENT",
    }

    synthetic = synthetic_contract()
    ledger = {
        "schema": "TOOL_MOBILE_ACTION_REACTION_LEDGER/v1",
        "production": {
            "status": "NOT_VALIDATED_TRIVIAL_ZERO_BECAUSE_COUPLING_ABSENT",
            "accepted_tool_to_mobile_impulse_xyz_ns": [0.0, 0.0, 0.0],
            "measured_mobile_tool_momentum_change_xyz_ns": [0.0, 0.0, 0.0],
            "machine_reaction_impulse_xyz_ns": [0.0, 0.0, 0.0],
            "action_reaction_residual_xyz_ns": [0.0, 0.0, 0.0],
            "action_reaction_relative_residual": None,
            "tool_to_mobile_work_j": 0.0,
            "warning": "zero plus zero is not evidence of two-way coupling",
        },
        "synthetic_contract": synthetic,
    }

    report = {
        "schema": "TOOL_MOBILE_MOMENTUM_CAUSAL_REPORT/v1",
        "status": "CAUSAL_AUDIT_COMPLETE_PRODUCTION_FIX_INTENTIONALLY_NOT_APPLIED",
        "FIRST_EXPECTED_COUPLING_EVENT_FOUND": "YES",
        "FIRST_EXPECTED_COUPLING_TIME_S": first["simulation_time_s"],
        "TOOL_MOBILE_CONTACT_CONFIRMED": "YES",
        "TOOL_MOBILE_CONTACT_CONFIRMATION_BASIS": "FRAME_LEVEL_GEOMETRIC_AND_MOBILE_PROXY",
        "CONTACT_MOBILE_VOLUME_M3": first["proximity"]["mouth_prism_m3"],
        "REQUESTED_DYNAMIC_IMPULSE_NONZERO": "NO",
        "REQUESTED_DYNAMIC_IMPULSE_STATUS": "REQUEST_STAGE_ABSENT",
        "MOBILE_SOURCE_INPUT_NONZERO": "NO",
        "MOBILE_TOOL_MOMENTUM_CHANGE_NONZERO": "NO",
        "MACHINE_REACTION_IMPULSE_NONZERO": "NO",
        "ROOT_CAUSE_CLASS": "COUPLING_NOT_IMPLEMENTED",
        "ROOT_CAUSE": (
            "GPU FailureZone writes a valid compact tool-contact material_mask, "
            "but production Mobile V2 receives neither tool state nor a reviewed "
            "contact impulse; it clears external_x/y to zero, never consumes the "
            "mask, and therefore returns zero tool impulse/work to the otherwise "
            "connected MobileMomentumBudget and Isaac reaction path."
        ),
        "RESPONSIBLE_OPERATOR": "WarpProductionMobileV2Solver",
        "RESPONSIBLE_FILE_FUNCTION": "src/isaac_bulk_pipeline/bulk_interaction/warp_mobile_v2.py:WarpProductionMobileV2Solver.step_resident",
        "FIX_APPLIED": "NO",
        "FIX_DESCRIPTION": "NONE; a contact constitutive law requires a dedicated model-design review",
        "PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N_BEFORE": 0.0,
        "PEAK_DYNAMIC_MOBILE_MOMENTUM_FORCE_N_AFTER": None,
        "TOOL_MOBILE_CONTACT_ACTIVE_STEP_COUNT": len(active_rows),
        "TOOL_MOBILE_CONTACT_ACTIVE_SUBSTEP_COUNT": None,
        "PEAK_CONTACT_MOBILE_VOLUME_M3": max(float(row["proximity"]["mouth_prism_m3"]) for row in rows),
        "PEAK_REQUESTED_TOOL_MOBILE_FORCE_N": None,
        "PEAK_REQUESTED_TOOL_MOBILE_IMPULSE_NS": None,
        "PEAK_ACCEPTED_TOOL_MOBILE_FORCE_N": 0.0,
        "TOTAL_ACCEPTED_TOOL_MOBILE_IMPULSE_NS": 0.0,
        "TOTAL_MOBILE_TOOL_MOMENTUM_CHANGE_NS": 0.0,
        "TOTAL_MACHINE_REACTION_IMPULSE_NS": 0.0,
        "ACTION_REACTION_RESIDUAL_NS": 0.0,
        "ACTION_REACTION_RELATIVE_RESIDUAL": None,
        "TOOL_TO_MOBILE_WORK_J": 0.0,
        "PRE_DUMP_REACHED": trajectory["PRE_DUMP_REACHED"],
        "NET_PAYLOAD_GAIN_M3": audit["NET_PAYLOAD_GAIN_M3"],
        "GROSS_CAPTURE_RATIO": audit["GROSS_CAPTURE_RATIO"],
        "MAX_PENETRATION_M": audit["MAX_PENETRATION_M"],
        "PEAK_QUASISTATIC_SOIL_FORCE_N": audit["SOIL_FORCE"]["peak_quasi_static_n"],
        "MAX_MASS_ERROR_M3": trajectory["MAX_MASS_ERROR_M3"],
        "CUT_MAX_MASS_ERROR_M3": audit["MAX_MASS_ERROR_M3"],
        "P0_2B_FORMAL_CUDA_STATUS": "NOT_RUN_CUDA_DEVICE_NOT_EXPOSED",
        "PRODUCTION_REPLAY_THIS_TASK": "NOT_RUN_CUDA_DEVICE_NOT_EXPOSED",
        "PHYSICS_PARAMETERS_CHANGED": "NONE",
        "MOBILE_V2_TRANSPORT_CHANGED": "NO",
        "FAILURESURFACE_V3_CHANGED": "NO",
        "CURL_SCOOP_TRAJECTORY_CHANGED": "NO",
        "TRACKSOIL_CHANGED": "NO",
        "VISIBLE_NEEDLE_FOREST": trajectory["VISIBLE_NEEDLE_FOREST"],
        "VISIBLE_TRIANGULAR_FINS": trajectory["VISIBLE_TRIANGULAR_FINS"],
        "GRID_SCALE_SPIKE_PROLIFERATION": trajectory["GRID_SCALE_SPIKE_PROLIFERATION"],
        "PRIMARY_REMAINING_BLOCKER": "NO_PHYSICALLY_REVIEWED_TOOL_MOBILE_CONTACT_IMPULSE_LAW_IN_PRODUCTION_MOBILE_V2",
        "synthetic_contract_tests": synthetic,
    }

    dump("tool_mobile_momentum_timeseries.json", timeseries)
    dump("tool_mobile_first_expected_coupling.json", first_record)
    dump("tool_mobile_action_reaction_ledger.json", ledger)
    dump("tool_mobile_momentum_causal_report.json", report)
    print(json.dumps({
        "status": report["status"],
        "root_cause": report["ROOT_CAUSE_CLASS"],
        "first_expected_time_s": report["FIRST_EXPECTED_COUPLING_TIME_S"],
        "active_frame_count": report["TOOL_MOBILE_CONTACT_ACTIVE_STEP_COUNT"],
        "fix_applied": report["FIX_APPLIED"],
    }, indent=2))


if __name__ == "__main__":
    main()
