from types import SimpleNamespace

from isaac_bulk_pipeline.runtime.tracksoil_conservation_audit import (
    TrackSoilConservationAudit,
)


def snapshot(resting, mobile, *, initial=10.0, payload=0.0):
    total = resting + mobile + payload
    return SimpleNamespace(
        material_ledger=SimpleNamespace(
            resting_m3=resting,
            mobile_m3=mobile,
            payload_m3=payload,
            airborne_m3=0.0,
            outflow_m3=0.0,
            initial_total_m3=initial,
            absolute_volume_error_m3=total - initial,
        )
    )


def quiet_core():
    return SimpleNamespace(
        interaction=None,
        dump_advance=SimpleNamespace(
            mobile=SimpleNamespace(
                volume_before_m3=1.0,
                volume_after_m3=1.0,
                transport_mass_residual_m3=0.0,
            ),
            deposition=SimpleNamespace(deposited_volume_m3=0.0),
            airborne=SimpleNamespace(landed_volume_m3=0.0),
        ),
        dump_release=None,
        avalanche_transition=SimpleNamespace(
            cumulative_resting_to_mobile_m3=0.0,
            cumulative_mobile_to_resting_m3=0.0,
        ),
    )


def test_tracksoil_boundary_records_equal_and_opposite_transfer():
    audit = TrackSoilConservationAudit()
    audit.begin_step(
        simulation_time_s=12.0,
        phase="REVERSE_TRAVEL",
        physics_step=42,
        snapshot=snapshot(9.0, 1.0),
    )
    audit.after_tracksoil(
        snapshot=snapshot(8.8, 1.2),
        result=SimpleNamespace(resting_to_mobile_volume_m3=0.2),
        affected_cell_count=8,
        footprint_area_m2=0.02,
        requested_r2m_m3=0.2,
    )
    audit.operator_boundary("AFTER_MOBILE_TRANSPORT", snapshot(8.8, 1.2))
    audit.finish_step(snapshot=snapshot(8.8, 1.2), core_result=quiet_core())

    report = audit.report()
    control = report["records"][0]["tracksoil_control_volume"]
    assert abs(control["resting_volume_removed_m3"] - 0.2) < 1e-12
    assert abs(control["mobile_volume_added_m3"] - 0.2) < 1e-12
    assert abs(control["tracksoil_local_residual_m3"]) < 1e-12
    assert report["first_bad_step_found"] is False


def test_first_bad_boundary_is_operator_that_introduces_unmatched_delta():
    audit = TrackSoilConservationAudit()
    audit.begin_step(
        simulation_time_s=12.0,
        phase="REVERSE_TRAVEL",
        physics_step=42,
        snapshot=snapshot(9.0, 1.0),
    )
    audit.after_tracksoil(
        snapshot=snapshot(8.8, 1.2),
        result=SimpleNamespace(resting_to_mobile_volume_m3=0.2),
        affected_cell_count=8,
        footprint_area_m2=0.02,
        requested_r2m_m3=0.2,
    )
    audit.operator_boundary("AFTER_MOBILE_TRANSPORT", snapshot(8.8, 1.1999))
    audit.finish_step(snapshot=snapshot(8.8, 1.1999), core_result=quiet_core())

    report = audit.report()
    assert report["first_bad_step_found"] is True
    assert report["first_bad_boundary"]["label"] == "AFTER_MOBILE_TRANSPORT"
    assert abs(
        report["first_bad_boundary"]["delta_from_previous"]["mobile"] + 1e-4
    ) < 1e-12
