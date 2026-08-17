from __future__ import annotations

import numpy as np

from isaac_bulk_pipeline.runtime.cut_fill_payload_audit import CutFillPayloadCausalAudit


def test_report_counts_failure_zone_and_avalanche_r2m_and_finds_first_link() -> None:
    audit = CutFillPayloadCausalAudit(
        state=None,
        grid=None,
        descriptor=None,
        phase_target_rad=np.zeros(4),
        dt_s=1.0 / 30.0,
    )
    audit.entry_payload_m3 = 0.0
    audit.r2m_start_m3 = 0.0
    audit.m2r_start_m3 = 0.0
    audit.failure_zone_r2m_gross_m3 = 4.6
    proximity = {
        "radius_0p25_m3": 0.01,
        "radius_0p5_m3": 0.02,
        "radius_1p0_m3": 0.5,
        "radius_2p0_m3": 1.0,
        "front_m3": 0.8,
        "mouth_prism_m3": 0.002,
        "bucket_interior_m3": 0.1,
        "left_bypass_m3": 0.2,
        "right_bypass_m3": 0.2,
        "behind_m3": 0.3,
    }
    audit.records = [{
        "payload_after_m3": 0.014,
        "gross_intake_m3": 0.014,
        "candidate_mouth_flux_m3": 0.014,
        "payload_spill_m3": 0.0,
        "r2m_cumulative_m3": 0.18,
        "m2r_cumulative_m3": 0.0,
        "proximity": proximity,
        "mouth_intersection_area_m2": 1.0,
        "mean_positive_relative_normal_speed_m_s": 0.01,
        "actuator_effort_saturated": [False] * 4,
        "actuator_shared_power_scale": 1.0,
        "tool_progress_fraction": 1.0,
        "maximum_penetration_m": 1.0,
        "applied_soil_force_n": [1.0, 0.0, 0.0],
        "quasi_static_force_n": [1.0, 0.0, 0.0],
        "dynamic_momentum_force_n": [0.0, 0.0, 0.0],
        "soil_power_on_terrain_w": 0.0,
    }]

    report = audit.report(failure_code="CUT_AND_FILL_TIMEOUT")

    assert np.isclose(report["R2M_GROSS_M3"], 4.78)
    assert np.isclose(report["FAILURE_ZONE_R2M_GROSS_M3"], 4.6)
    assert np.isclose(report["LARGE_AVALANCHE_R2M_GROSS_M3"], 0.18)
    assert report["MOUTH_CAPTURE_RATIO"] == 1.0
    assert report["PRIMARY_BOTTLENECK"] == "MOBILE_NOT_REACHING_MOUTH"
    assert report["INTAKE_CLASSIFICATION"] == "INTAKE_FLUX_TOO_SMALL"
