from isaac_bulk_pipeline.runtime.avalanche_persistence import (
    PHYSICAL_LONG_RANGE_PROPAGATION,
    PHYSICS_NOT_SETTLED,
    REMOBILIZATION_LIMIT_CYCLE,
    PersistenceSample,
    classify_persistence,
)


def sample(t, *, new, repeat, unique, repeated, mobile, active, settled,
           r2m, m2r, spatial, retired=0.0):
    return PersistenceSample(
        simulation_time_s=t,
        newly_activated_volume_m3=new,
        reactivated_volume_m3=repeat,
        cumulative_r2m_m3=r2m,
        cumulative_m2r_m3=m2r,
        net_terrain_volume_change_m3=0.0,
        net_spatial_transfer_m3=spatial,
        current_mobile_volume_m3=mobile,
        maximum_mobile_speed_m_s=0.1 if mobile else 0.0,
        newly_activated_area_m2=0.1 if new else 0.0,
        reactivated_area_m2=0.2 if repeated else 0.0,
        retired_candidate_area_m2=retired,
        active_tile_count=active,
        connected_unstable_area_m2=0.0 if settled else unique,
        unique_activated_area_m2=unique,
        unique_cells_ever_activated=max(1, int(unique / 0.0025)),
        cells_activated_more_than_once=repeated,
        mass_error_m3=1.0e-11,
        terrain_settled=settled,
    )


def test_classifies_advancing_then_retired_event_as_physical():
    rows = [
        sample(1, new=.2, repeat=0, unique=.2, repeated=0, mobile=.2,
               active=2, settled=False, r2m=.2, m2r=0, spatial=.2),
        sample(2, new=.2, repeat=.01, unique=.4, repeated=1, mobile=.1,
               active=2, settled=False, r2m=.41, m2r=.2, spatial=.35),
        sample(3, new=0, repeat=0, unique=.4, repeated=1, mobile=0,
               active=0, settled=True, r2m=.41, m2r=.41, spatial=.4,
               retired=.4),
    ]
    decision = classify_persistence(rows)
    assert decision.status == "PASS"
    assert decision.classification == PHYSICAL_LONG_RANGE_PROPAGATION


def test_event_retirement_does_not_require_material_to_deposit_on_source_cells():
    rows = [
        sample(1, new=.2, repeat=0, unique=.2, repeated=0, mobile=.2,
               active=2, settled=False, r2m=.2, m2r=0, spatial=.2),
        # Locally counted source-cell retirement can remain small because the
        # material transported elsewhere. The complete event is nevertheless
        # permanently retired once Mobile/connected/frontier state is settled.
        sample(2, new=0, repeat=0, unique=.2, repeated=0, mobile=0,
               active=0, settled=True, r2m=.2, m2r=.2, spatial=.3,
               retired=.01),
    ]
    decision = classify_persistence(rows)
    assert decision.status == "PASS"
    assert decision.metrics["retired_area_fraction"] < 0.95


def test_classifies_reactivation_flux_on_plateau_as_limit_cycle():
    rows = []
    for index in range(12):
        rows.append(sample(
            index + 1,
            new=.2 if index < 2 else 0.0,
            repeat=0.0 if index < 2 else .3,
            unique=.4,
            repeated=80,
            mobile=.2,
            active=3,
            settled=False,
            r2m=.4 + max(0, index - 1) * .3,
            m2r=max(0, index - 1) * .3,
            spatial=.4,
        ))
    decision = classify_persistence(rows)
    assert decision.status == "FAIL"
    assert decision.classification == REMOBILIZATION_LIMIT_CYCLE
    assert decision.metrics["tail_unique_area_growth_fraction"] == 0.0


def test_active_but_inconclusive_horizon_is_not_passed():
    decision = classify_persistence([
        sample(1, new=.1, repeat=0, unique=.1, repeated=0, mobile=.1,
               active=1, settled=False, r2m=.1, m2r=0, spatial=.1)
    ])
    assert decision.status == "FAIL"
    assert decision.classification == PHYSICS_NOT_SETTLED
